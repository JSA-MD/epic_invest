"""Single source of truth for the live↔backtest regime-gate parity hooks.

Stage 0/1 lockdown introduced two env-driven knobs:

    PAIRWISE_REGIME_THRESHOLD_SCALE  (default 1.0)
        Multiplies the candidate's static `regime_threshold` so the gate
        can be loosened or tightened uniformly without recertifying.

    PAIRWISE_REGIME_GATE_DISABLED    (default 0)
        Bypasses the long_ok/short_ok check entirely. Smoke-test only —
        recommended only with PAIRWISE_GROSS_CAP=0.01 lockdown.

Every place that checks `regime_score`/`breadth_score` against
`regime_threshold` (live trader, conformal gate, plain backtest kernel,
shadow trader, stress-test) must read the same hook so apples_to_apples
parity holds. This helper is that single point of read.

Two convenience entry points:

    apply_gate(regime_score, breadth_score, requested_weight, *,
               regime_threshold, breadth_threshold, env=None)
        Returns the post-gate `requested_weight` (0 when blocked,
        unchanged otherwise) — drop-in replacement for the inline
        long_ok/short_ok branches scattered across the codebase.

    gate_overrides(env=None) -> (scale: float, disabled: bool)
        Lower-level: just the two knobs. Use when you need to apply
        them inside a numpy/numba inner loop where calling apply_gate
        per-bar would be wasteful.
"""

from __future__ import annotations

import os
from typing import Mapping


__all__ = ["gate_overrides", "apply_gate"]


def gate_overrides(env: Mapping[str, str] | None = None) -> tuple[float, bool]:
    """Return `(threshold_scale, gate_disabled)` honouring fail-safe defaults.

    Garbage env values fall back to the legacy `(1.0, False)` so a typo
    in `pairwise_live_launchd_env.sh` cannot accidentally disable the
    gate without an operator noticing.
    """
    e = env if env is not None else os.environ
    raw_scale = e.get("PAIRWISE_REGIME_THRESHOLD_SCALE", "1.0")
    try:
        scale = float(raw_scale)
        if scale < 0.0:
            scale = 1.0
    except (TypeError, ValueError):
        scale = 1.0
    raw_disabled = str(e.get("PAIRWISE_REGIME_GATE_DISABLED", "0")).strip().lower()
    disabled = raw_disabled in {"1", "true", "yes", "on"}
    return scale, disabled


def apply_gate(
    regime_score: float,
    breadth_score: float,
    requested_weight: float,
    *,
    regime_threshold: float,
    breadth_threshold: float,
    extra_threshold_mult: float = 1.0,
    env: Mapping[str, str] | None = None,
) -> tuple[float, bool, bool]:
    """Apply the live-equivalent gate to a single bar.

    Returns `(post_gate_weight, long_ok, short_ok)` — `post_gate_weight`
    is `0.0` when the gate blocks, otherwise unchanged from the input
    `requested_weight`. The `long_ok`/`short_ok` flags are exposed so
    callers can record them in the decision log without re-deriving.
    """
    scale, disabled = gate_overrides(env)
    effective_threshold = float(regime_threshold) * float(extra_threshold_mult) * float(scale)
    if disabled:
        long_ok = True
        short_ok = True
    else:
        long_ok = regime_score >= effective_threshold and breadth_score >= float(breadth_threshold)
        short_ok = regime_score <= -effective_threshold and breadth_score <= (1.0 - float(breadth_threshold))
    if requested_weight > 0.0 and not long_ok:
        return 0.0, long_ok, short_ok
    if requested_weight < 0.0 and not short_ok:
        return 0.0, long_ok, short_ok
    return float(requested_weight), long_ok, short_ok
