"""Live-cycle Stage 2/3 overlay runner.

This is the single integration point that wires `adaptive_threshold`,
`sign_mismatch_monitor`, and the unified safety overlay into the live
trader loop. Call `run_live_overlays(plan, state, env)` once per cycle
*after* the plan is built and after the existing Stage 0 overlays
(D1 max-hold, R3 CVaR cut, stale-price guard) have run.

Behaviour
---------

For every pair in `plan["pair_plans"]`:

1. Append the current target_weight and regime_score to a rolling
   history kept on `state` (288-bar window for sign instability,
   8 640-bar window — 30 days at 5-min — for adaptive bands).
2. Run `sign_instability(window=288, threshold=0.30)`. If the sign
   flipped on more than 30 % of the eligible transitions in the last
   24 hours, force the pair flat.
3. Run `last_threshold(window=288, k_sigma=3.0, floor_sigma=0.005)`.
   If the live `regime_score` falls inside the adaptive ±3 σ band but
   the plan asks for a non-flat trade, force the pair flat — the gate
   is in noise zone and the legacy `±0.02` cutoff would have given a
   coin-flip signal.

Toggle behaviour with the env var `PAIRWISE_LIVE_OVERLAYS`
(default `1`). When `0`, the runner is a no-op and the legacy gate
behaviour is preserved exactly. This keeps the wiring reversible
without a rollback commit.

Integrates with the existing decision_journal / Telegram pipeline by
appending an `override_reason` entry per forced-flat pair so the
post-fix monitor and `daily_diagnosis.py` can attribute the
force-flat correctly.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np


_DEFAULT_INSTABILITY_THRESHOLD = 0.30
_DEFAULT_INSTABILITY_WINDOW = 288  # 24 h at 5 min
_DEFAULT_REGIME_WINDOW = 288       # 24 h tail
_DEFAULT_REGIME_K_SIGMA = 3.0
_DEFAULT_REGIME_FLOOR_SIGMA = 0.005
_REGIME_HISTORY_CAP = 8640         # 30 d at 5 min


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_truthy(env: Mapping[str, str], key: str, default: str = "1") -> bool:
    return str(env.get(key, default)).strip().lower() not in {"0", "false", "no", "off"}


def _trim(history: list, cap: int) -> None:
    if len(history) > cap:
        del history[0 : len(history) - cap]


def run_live_overlays(
    plan: dict[str, Any],
    state: dict[str, Any],
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply Stage 2/3 overlays. Returns {pair: reason} for forced flats."""
    env = env if env is not None else os.environ
    if not _env_truthy(env, "PAIRWISE_LIVE_OVERLAYS"):
        return {}

    pair_plans = plan.get("pair_plans") or {}
    target_weights = plan.setdefault("target_weights", {})

    recent_tw: dict[str, list[float]] = state.setdefault("recent_target_weights", {})
    recent_rs: dict[str, list[float]] = state.setdefault("recent_regime_scores", {})

    overlay_decisions: dict[str, str] = {}

    # Lazy imports so callers without numpy in their critical path
    # do not pay the cost when overlays are toggled off.
    from adaptive_threshold import (
        adaptive_band_signal,
        last_threshold,
    )
    from sign_mismatch_monitor import sign_instability

    instability_threshold = float(
        env.get("PAIRWISE_SIGN_INSTABILITY_THRESHOLD", _DEFAULT_INSTABILITY_THRESHOLD)
    )
    instability_window = int(
        env.get("PAIRWISE_SIGN_INSTABILITY_WINDOW", _DEFAULT_INSTABILITY_WINDOW)
    )
    regime_window = int(env.get("PAIRWISE_ADAPTIVE_REGIME_WINDOW", _DEFAULT_REGIME_WINDOW))
    regime_k_sigma = float(env.get("PAIRWISE_ADAPTIVE_REGIME_K_SIGMA", _DEFAULT_REGIME_K_SIGMA))
    regime_floor_sigma = float(
        env.get("PAIRWISE_ADAPTIVE_REGIME_FLOOR_SIGMA", _DEFAULT_REGIME_FLOOR_SIGMA)
    )

    for pair, pp in pair_plans.items():
        try:
            current_tw = float(pp.get("target_weight") or 0.0)
        except (TypeError, ValueError):
            current_tw = 0.0
        weights = recent_tw.setdefault(pair, [])
        weights.append(current_tw)
        _trim(weights, max(instability_window, _DEFAULT_INSTABILITY_WINDOW))

        rs_raw = pp.get("regime_score")
        scores = recent_rs.setdefault(pair, [])
        if rs_raw is not None:
            try:
                scores.append(float(rs_raw))
            except (TypeError, ValueError):
                pass
        _trim(scores, _REGIME_HISTORY_CAP)

        force_flat_reason: str | None = None
        force_flat_metadata: dict[str, Any] = {}

        # 1. Sign instability — fires only with at least window_bars of history.
        if len(weights) >= instability_window:
            verdict = sign_instability(
                np.asarray(weights[-instability_window:], dtype=np.float64),
                window_bars=instability_window,
                instability_threshold=instability_threshold,
            )
            if verdict.force_flat:
                force_flat_reason = "sign_instability"
                force_flat_metadata = {
                    "flip_rate": verdict.value,
                    "threshold": verdict.threshold,
                    "n_bars": verdict.n_bars,
                }

        # 2. Adaptive regime threshold — only if instability didn't already fire.
        if force_flat_reason is None and len(scores) >= regime_window:
            band = last_threshold(
                np.asarray(scores[-regime_window:], dtype=np.float64),
                window=regime_window,
                k_sigma=regime_k_sigma,
                floor_sigma=regime_floor_sigma,
            )
            current_rs = float(scores[-1]) if scores else 0.0
            band_signal = adaptive_band_signal(current_rs, band)
            tw_sign = 0
            if current_tw > 1e-6:
                tw_sign = 1
            elif current_tw < -1e-6:
                tw_sign = -1
            if tw_sign != 0 and band_signal == 0:
                force_flat_reason = "adaptive_threshold_noise"
                force_flat_metadata = {
                    "regime_score": current_rs,
                    "band_upper": band.upper,
                    "band_lower": band.lower,
                    "band_sigma": band.sigma,
                    "k_sigma": regime_k_sigma,
                }

        if force_flat_reason:
            pp["target_weight"] = 0.0
            pp["overlay_force_flat"] = force_flat_reason
            pp["overlay_force_flat_metadata"] = force_flat_metadata
            target_weights[pair] = 0.0
            overlay_decisions[pair] = force_flat_reason
            journal = state.setdefault("decision_journal", [])
            journal.append(
                {
                    "at": _iso_now(),
                    "pair": pair,
                    "override_reason": force_flat_reason,
                    "target_weight_forced": 0.0,
                    **force_flat_metadata,
                }
            )

    return overlay_decisions


__all__ = ["run_live_overlays"]
