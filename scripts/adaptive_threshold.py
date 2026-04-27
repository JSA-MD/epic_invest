"""Adaptive regime / signal thresholds (Stage 2 — Simons / Chan).

The legacy candidate JSON ships hard-coded thresholds like
`regime_threshold = 0.02` and `breadth_threshold = 0.65`. The Stage 0/1
post-mortem showed those constants live so close to the noise distribution
that the gating decision flips on a single 5-min bar (`regime_score` of
+0.019 vs threshold −0.02). Hard cutoffs near noise scale produce the
"33% sign-flip per bar" pattern documented in `docs/live_drift_root_cause_20260426.md`.

This module provides drop-in helpers that replace point thresholds with
data-driven `mean ± k·std` bands computed on a rolling window. By default
`k=3.0` (≈3σ), which is several times wider than the legacy point cutoff
and forces the gate to wait for genuine regime breaks instead of noise
fluctuations. Both functions are vectorised; `last_threshold` is the
right call inside the live loop where only the trailing band is needed.

API
---
- `rolling_std_threshold(scores, window, k_sigma)` — full vector of
  thresholds aligned to `scores`. Mostly useful for back-tests / plots.
- `last_threshold(scores, window, k_sigma)` — `(mean, threshold_pos,
  threshold_neg)` for the most recent bar. Used live.
- `adaptive_band_pass(value, mean, threshold)` — returns `+1`, `-1`, or
  `0` depending on whether `value` exceeds the upper, lower, or neither
  band. Drop into the gate that previously compared against
  `regime_threshold`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveBand:
    mean: float
    sigma: float
    upper: float  # mean + k_sigma × sigma
    lower: float  # mean − k_sigma × sigma
    n_observations: int
    k_sigma: float


def rolling_std_threshold(
    scores: np.ndarray,
    window: int,
    *,
    k_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (upper, lower) threshold series aligned to `scores`.

    Bars before `window` get NaN — callers should guard the gate with
    `np.isfinite` before applying it.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    if k_sigma <= 0.0:
        raise ValueError(f"k_sigma must be > 0; got {k_sigma}")
    n = arr.size
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < window:
        return upper, lower
    # Cumulative-sum trick for rolling mean and std (population std with
    # Bessel correction approximation). For small windows a python loop is
    # acceptable; numpy vectorised version below.
    cumsum = np.concatenate(([0.0], np.cumsum(arr)))
    cumsq = np.concatenate(([0.0], np.cumsum(arr ** 2)))
    for i in range(window - 1, n):
        s = cumsum[i + 1] - cumsum[i + 1 - window]
        ss = cumsq[i + 1] - cumsq[i + 1 - window]
        mean = s / window
        var = max(ss / window - mean * mean, 0.0)
        sigma = float(np.sqrt(var))
        upper[i] = mean + k_sigma * sigma
        lower[i] = mean - k_sigma * sigma
    return upper, lower


def last_threshold(
    scores: np.ndarray,
    window: int,
    *,
    k_sigma: float = 3.0,
    floor_sigma: float = 0.0,
) -> AdaptiveBand:
    """Return the band for the most recent bar.

    `floor_sigma` enforces a minimum width on the threshold (for example,
    set it to 0.005 to avoid a degenerate near-zero band when scores are
    flatlined).
    """
    arr = np.asarray(scores, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window}")
    if n < window:
        return AdaptiveBand(
            mean=float("nan"),
            sigma=float("nan"),
            upper=float("nan"),
            lower=float("nan"),
            n_observations=n,
            k_sigma=k_sigma,
        )
    tail = arr[-window:]
    mean = float(np.mean(tail))
    sigma = float(np.std(tail, ddof=1))
    if sigma < floor_sigma:
        sigma = float(floor_sigma)
    return AdaptiveBand(
        mean=mean,
        sigma=sigma,
        upper=mean + k_sigma * sigma,
        lower=mean - k_sigma * sigma,
        n_observations=window,
        k_sigma=k_sigma,
    )


def adaptive_band_signal(
    value: float,
    band: AdaptiveBand,
) -> int:
    """Return +1 if `value > band.upper`, −1 if `value < band.lower`, else 0.

    Returns `0` for any non-finite band (insufficient history) — callers
    should treat that as a "stand aside" signal.
    """
    if not (np.isfinite(band.upper) and np.isfinite(band.lower)):
        return 0
    if value > band.upper:
        return 1
    if value < band.lower:
        return -1
    return 0


__all__ = [
    "AdaptiveBand",
    "rolling_std_threshold",
    "last_threshold",
    "adaptive_band_signal",
]
