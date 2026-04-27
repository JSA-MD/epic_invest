"""Sign-mismatch / sign-instability monitor (Stage 2 — Chan).

Two complementary measurements that bolt on top of the live decision log
(`logs/pairwise_regime_decisions.jsonl`).

1. **Sign instability** — fraction of consecutive bars where the live
   `target_weight` sign flipped during the trailing window. This catches
   the "regime gate noise zone" pattern that drove 33% of bars to
   sign-flip on `regime_score = ±0.02` in the live drift report.

2. **Live↔backtest sign mismatch** — when the caller can supply a
   matching backtest sign series for the same window, this compares the
   two sign vectors bar by bar and reports the mismatch fraction. Above
   the configured threshold (default 0.30 → 70% match), the live system
   should force-flat all positions until the underlying issue is fixed.

Both checks return a structured `MonitorVerdict` with a boolean
`force_flat` field that the live loop can OR into its existing safety
overlays (`max_hold`, `cvar_cut`, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonitorVerdict:
    metric: str  # "sign_instability" or "sign_mismatch"
    value: float  # observed metric in [0, 1]
    threshold: float
    n_bars: int
    force_flat: bool
    reason: str


def _signs(arr: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.int8)
    out[arr > eps] = 1
    out[arr < -eps] = -1
    return out


def sign_instability(
    target_weights: np.ndarray,
    *,
    window_bars: int,
    instability_threshold: float = 0.30,
    eps: float = 1e-6,
) -> MonitorVerdict:
    """Fraction of bars whose sign flipped from the previous non-flat bar.

    A bar with weight near zero (|w| < eps) is treated as a continuation
    of the prior non-flat sign — flat-to-flat is *not* a flip. This
    avoids penalising the gate when it correctly stands aside.
    """
    arr = np.asarray(target_weights, dtype=np.float64)
    n = arr.size
    if window_bars < 2:
        raise ValueError("window_bars must be >= 2")
    if n < window_bars:
        return MonitorVerdict(
            metric="sign_instability",
            value=float("nan"),
            threshold=instability_threshold,
            n_bars=n,
            force_flat=False,
            reason=f"insufficient history ({n} < {window_bars})",
        )
    tail = arr[-window_bars:]
    signs = _signs(tail, eps=eps)
    # Forward-fill flats with the last non-flat sign so flat→flat doesn't count.
    last_non_flat = 0
    filled = np.zeros_like(signs)
    for i, s in enumerate(signs):
        if s != 0:
            last_non_flat = int(s)
        filled[i] = last_non_flat
    flips = int(np.sum((np.diff(filled) != 0) & (filled[:-1] != 0)))
    flip_rate = flips / max(window_bars - 1, 1)
    force_flat = flip_rate >= instability_threshold
    reason = (
        f"sign flipped on {flips}/{window_bars - 1} eligible transitions "
        f"({flip_rate:.1%}) — {'≥' if force_flat else '<'} threshold {instability_threshold:.1%}"
    )
    return MonitorVerdict(
        metric="sign_instability",
        value=float(flip_rate),
        threshold=instability_threshold,
        n_bars=int(window_bars),
        force_flat=force_flat,
        reason=reason,
    )


def sign_mismatch(
    live_weights: np.ndarray,
    backtest_weights: np.ndarray,
    *,
    window_bars: int,
    mismatch_threshold: float = 0.30,
    eps: float = 1e-6,
) -> MonitorVerdict:
    """Fraction of bars where live and backtest sign disagree."""
    live_arr = np.asarray(live_weights, dtype=np.float64)
    bt_arr = np.asarray(backtest_weights, dtype=np.float64)
    if live_arr.size != bt_arr.size:
        raise ValueError(
            f"live ({live_arr.size}) and backtest ({bt_arr.size}) weights must align"
        )
    n = live_arr.size
    if window_bars < 1:
        raise ValueError("window_bars must be >= 1")
    if n < window_bars:
        return MonitorVerdict(
            metric="sign_mismatch",
            value=float("nan"),
            threshold=mismatch_threshold,
            n_bars=n,
            force_flat=False,
            reason=f"insufficient history ({n} < {window_bars})",
        )
    live_tail = live_arr[-window_bars:]
    bt_tail = bt_arr[-window_bars:]
    live_signs = _signs(live_tail, eps=eps)
    bt_signs = _signs(bt_tail, eps=eps)
    # Bars where both are flat are unanimous-flat → not a mismatch.
    eligible = (live_signs != 0) | (bt_signs != 0)
    n_eligible = int(np.sum(eligible))
    if n_eligible == 0:
        return MonitorVerdict(
            metric="sign_mismatch",
            value=0.0,
            threshold=mismatch_threshold,
            n_bars=int(window_bars),
            force_flat=False,
            reason="no eligible (non-flat) bars in window",
        )
    mismatches = int(np.sum((live_signs != bt_signs) & eligible))
    rate = mismatches / n_eligible
    force_flat = rate >= mismatch_threshold
    reason = (
        f"live≠backtest on {mismatches}/{n_eligible} eligible bars "
        f"({rate:.1%}) — {'≥' if force_flat else '<'} threshold {mismatch_threshold:.1%}"
    )
    return MonitorVerdict(
        metric="sign_mismatch",
        value=float(rate),
        threshold=mismatch_threshold,
        n_bars=int(window_bars),
        force_flat=force_flat,
        reason=reason,
    )


__all__ = ["MonitorVerdict", "sign_instability", "sign_mismatch"]
