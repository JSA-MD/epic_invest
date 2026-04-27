"""Post-blend overlay gates — backtest-vectorised equivalents of the live overlays.

These two functions mirror the Stage 2/3 gates applied in
``live_overlay_runner.run_live_overlays`` so the backtest replay kernel
produces the same gating decisions as live.

Live reference
--------------
* sign instability  — ``live_overlay_runner.py:127-139`` via
  ``sign_mismatch_monitor.sign_instability``
* adaptive threshold — ``live_overlay_runner.py:141-164`` via
  ``adaptive_threshold.last_threshold`` / ``adaptive_band_signal``

Environment knobs (consistent defaults with live)
--------------------------------------------------
* ``PAIRWISE_SIGN_INSTABILITY_THRESHOLD``  default ``0.30``
* ``PAIRWISE_ADAPTIVE_K_SIGMA``            default ``3.0``
* ``PAIRWISE_OVERLAY_WINDOW``              default ``288``
"""

from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------------------
# Defaults — must match the hard-coded live values in live_overlay_runner.py
# (_DEFAULT_INSTABILITY_THRESHOLD = 0.30, _DEFAULT_INSTABILITY_WINDOW = 288,
#  _DEFAULT_REGIME_K_SIGMA = 3.0).
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLD = 0.30
_DEFAULT_K_SIGMA = 3.0
_DEFAULT_WINDOW = 288


def _read_env_defaults() -> tuple[int, float, float]:
    """Return (window, threshold, k_sigma) from env with live-matching defaults."""
    window = int(os.environ.get("PAIRWISE_OVERLAY_WINDOW", _DEFAULT_WINDOW))
    threshold = float(
        os.environ.get("PAIRWISE_SIGN_INSTABILITY_THRESHOLD", _DEFAULT_THRESHOLD)
    )
    k_sigma = float(os.environ.get("PAIRWISE_ADAPTIVE_K_SIGMA", _DEFAULT_K_SIGMA))
    return window, threshold, k_sigma


def apply_sign_instability(
    weights_arr: np.ndarray,
    window: int = 288,
    threshold: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """For each bar, compute the sign-flip rate over the trailing ``window`` bars.

    Mirrors ``sign_mismatch_monitor.sign_instability`` exactly:

    * Flat bars (|w| < 1e-6) are forward-filled with the last non-flat sign so
      flat→flat transitions do not count as flips.
    * flip_rate = flips / (window - 1)
    * If flip_rate >= threshold the bar is gated (weight forced to 0).

    Bars before ``window`` bars of history are accumulated are never gated
    (not enough history to compute a reliable estimate).

    Parameters
    ----------
    weights_arr:
        1-D array of target weights, length N.
    window:
        Rolling look-back in bars.
    threshold:
        Flip-rate at or above which the gate fires (default 0.30 = 30 %).

    Returns
    -------
    gated_weights:
        Copy of ``weights_arr`` with gated bars zeroed.
    gate_mask:
        Boolean array; ``True`` where the gate fired.
    """
    arr = np.asarray(weights_arr, dtype=np.float64)
    n = arr.size
    gated = arr.copy()
    mask = np.zeros(n, dtype=bool)

    eps = 1e-6

    # Pre-compute sign series with forward-fill of flats (matches live logic).
    signs = np.zeros(n, dtype=np.int8)
    for i in range(n):
        if arr[i] > eps:
            signs[i] = 1
        elif arr[i] < -eps:
            signs[i] = -1
        else:
            signs[i] = 0

    # Forward-fill zeros with the last non-flat sign.
    filled = np.zeros(n, dtype=np.int8)
    last_non_flat = 0
    for i in range(n):
        if signs[i] != 0:
            last_non_flat = int(signs[i])
        filled[i] = last_non_flat

    # Compute flip counts over a rolling window of size `window`.
    # A flip is a pair (i-1, i) where filled[i] != filled[i-1] and filled[i-1] != 0.
    # We precompute a flip indicator on the diff series then use a prefix sum.
    if n < 2:
        return gated, mask

    flip_indicator = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if filled[i - 1] != 0 and filled[i] != filled[i - 1]:
            flip_indicator[i] = 1

    prefix = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        prefix[i + 1] = prefix[i] + flip_indicator[i]

    denom = max(window - 1, 1)
    # Gate starts once we have a full window (bar index window-1 onward).
    for i in range(window - 1, n):
        # Flip indicator indices i-(window-2) .. i cover transitions within the window.
        # The window of weights is arr[i-window+1 .. i] (inclusive).
        # Transitions within that window are at positions i-window+2 .. i in flip_indicator.
        lo = i - window + 2  # first transition index inside the window
        hi = i + 1           # exclusive
        if lo < 1:
            lo = 1
        flips = int(prefix[hi] - prefix[lo])
        flip_rate = flips / denom
        if flip_rate >= threshold:
            mask[i] = True
            gated[i] = 0.0

    return gated, mask


def apply_adaptive_threshold(
    weights_arr: np.ndarray,
    window: int = 288,
    k_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Force weight=0 when the weight falls inside the rolling ±k_sigma band.

    Band is computed from ``weights_arr`` itself.  This is the fallback used
    when no separate ``regime_scores_arr`` is available (e.g. single-asset
    ``btc_convex_blend`` path where regime_score is not in scope).

    For the live-equivalent path use ``apply_adaptive_threshold_regime`` which
    computes the band from the regime-score series while gating the weight —
    matching ``live_overlay_runner`` exactly.

    * Compute rolling mean and std (ddof=1) over the trailing ``window`` bars.
    * A bar is inside the noise band when ``band_lower <= w <= band_upper``.
    * Gate fires (weight zeroed) when the weight is non-flat AND inside the band.

    Bars without a full ``window`` of history are never gated.

    Parameters
    ----------
    weights_arr:
        1-D array of target weights, length N.
    window:
        Rolling look-back in bars.
    k_sigma:
        Band half-width in standard deviations (default 3.0, matching live).

    Returns
    -------
    gated_weights:
        Copy of ``weights_arr`` with gated bars zeroed.
    gate_mask:
        Boolean array; ``True`` where the gate fired.
    """
    arr = np.asarray(weights_arr, dtype=np.float64)
    n = arr.size
    gated = arr.copy()
    mask = np.zeros(n, dtype=bool)

    eps = 1e-6

    if n < window:
        return gated, mask

    # Prefix sums for rolling mean and variance.
    cumsum = np.concatenate(([0.0], np.cumsum(arr)))
    cumsq = np.concatenate(([0.0], np.cumsum(arr ** 2)))

    for i in range(window - 1, n):
        # Window is arr[i-window+1 .. i].
        s = cumsum[i + 1] - cumsum[i + 1 - window]
        ss = cumsq[i + 1] - cumsq[i + 1 - window]
        mean = s / window
        # Use population variance then correct to sample std (ddof=1), matching
        # adaptive_threshold.last_threshold which calls np.std(tail, ddof=1).
        var = max(ss / window - mean * mean, 0.0)
        # Convert population std to sample std: multiply by sqrt(window/(window-1))
        sigma = float(np.sqrt(var * window / max(window - 1, 1)))

        upper = mean + k_sigma * sigma
        lower = mean - k_sigma * sigma

        w = arr[i]
        # Non-flat weight AND inside band → gate fires (noise zone).
        if abs(w) > eps and lower <= w <= upper:
            mask[i] = True
            gated[i] = 0.0

    return gated, mask


def apply_adaptive_threshold_regime(
    regime_scores_arr: np.ndarray,
    weights_arr: np.ndarray,
    window: int = 288,
    k_sigma: float = 3.0,
    floor_sigma: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Force weight=0 when the regime_score falls inside its rolling ±k_sigma band.

    This is the **live-equivalent** form of the adaptive gate.  It mirrors
    ``live_overlay_runner`` exactly:

    * Band is computed from ``regime_scores_arr`` (``recent_regime_scores``
      in live), not from the weights.
    * Gate fires when ``regime_scores_arr[i]`` is inside the band AND
      ``weights_arr[i]`` is non-flat — i.e. the regime signal is in the
      noise zone but the kernel wants to trade.
    * ``floor_sigma`` matches live's ``_DEFAULT_REGIME_FLOOR_SIGMA = 0.005``.

    Bars without a full ``window`` of regime-score history are never gated.

    Parameters
    ----------
    regime_scores_arr:
        1-D array of regime scores aligned bar-for-bar with ``weights_arr``,
        length N.  Corresponds to ``context["regime"]`` in the backtest.
    weights_arr:
        1-D array of target weights (or signal proxy), length N.
    window:
        Rolling look-back in bars (default 288 = 24 h at 5 min).
    k_sigma:
        Band half-width in standard deviations (default 3.0).
    floor_sigma:
        Minimum sigma to avoid degenerate near-zero bands (default 0.005).

    Returns
    -------
    gated_weights:
        Copy of ``weights_arr`` with gated bars zeroed.
    gate_mask:
        Boolean array; ``True`` where the gate fired.
    """
    rs = np.asarray(regime_scores_arr, dtype=np.float64)
    wt = np.asarray(weights_arr, dtype=np.float64)
    if rs.size != wt.size:
        raise ValueError(
            f"regime_scores_arr length ({rs.size}) must match weights_arr length ({wt.size})"
        )
    n = rs.size
    gated = wt.copy()
    mask = np.zeros(n, dtype=bool)

    eps = 1e-6

    if n < window:
        return gated, mask

    # Prefix sums over regime scores for rolling mean and variance.
    rs_cumsum = np.concatenate(([0.0], np.cumsum(rs)))
    rs_cumsq = np.concatenate(([0.0], np.cumsum(rs ** 2)))

    for i in range(window - 1, n):
        s = rs_cumsum[i + 1] - rs_cumsum[i + 1 - window]
        ss = rs_cumsq[i + 1] - rs_cumsq[i + 1 - window]
        mean = s / window
        var = max(ss / window - mean * mean, 0.0)
        # Sample std (ddof=1), matching live's np.std(tail, ddof=1).
        sigma = float(np.sqrt(var * window / max(window - 1, 1)))
        if sigma < floor_sigma:
            sigma = floor_sigma

        upper = mean + k_sigma * sigma
        lower = mean - k_sigma * sigma

        current_rs = rs[i]
        w = wt[i]
        # Gate fires: regime score is inside noise band AND weight is non-flat.
        if abs(w) > eps and lower <= current_rs <= upper:
            mask[i] = True
            gated[i] = 0.0

    return gated, mask


def apply_post_blend_overlays(
    weights_arr: np.ndarray,
    *,
    window: int | None = None,
    sign_instability_threshold: float | None = None,
    adaptive_k_sigma: float | None = None,
    regime_scores_arr: np.ndarray | None = None,
    adaptive_floor_sigma: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply both post-blend gates in sequence (sign instability then adaptive band).

    Reads env knobs when parameters are not supplied explicitly.

    When ``regime_scores_arr`` is provided the adaptive gate uses
    ``apply_adaptive_threshold_regime`` — the live-equivalent form that
    computes the noise band from the *regime-score* series (matching
    ``live_overlay_runner.run_live_overlays``) while gating ``weights_arr``.

    When ``regime_scores_arr`` is ``None`` (e.g. the single-asset
    ``btc_convex_blend`` path where regime_score is not available), the
    legacy ``apply_adaptive_threshold`` is used instead, which computes the
    band from ``weights_arr`` itself.

    Returns
    -------
    gated_weights:
        Weights after both gates applied.
    sign_mask:
        Boolean mask from ``apply_sign_instability``.
    adaptive_mask:
        Boolean mask from the adaptive-threshold gate (regime-based or
        weight-based depending on whether ``regime_scores_arr`` was supplied).
    """
    env_window, env_threshold, env_k_sigma = _read_env_defaults()
    if window is None:
        window = env_window
    if sign_instability_threshold is None:
        sign_instability_threshold = env_threshold
    if adaptive_k_sigma is None:
        adaptive_k_sigma = env_k_sigma

    after_sign, sign_mask = apply_sign_instability(
        weights_arr, window=window, threshold=sign_instability_threshold
    )
    if regime_scores_arr is not None:
        # Live-equivalent: band computed from regime_score history; gate on weight.
        after_adaptive, adaptive_mask = apply_adaptive_threshold_regime(
            regime_scores_arr,
            after_sign,
            window=window,
            k_sigma=adaptive_k_sigma,
            floor_sigma=adaptive_floor_sigma,
        )
    else:
        # Fallback (no regime_score available): band computed from weight history.
        after_adaptive, adaptive_mask = apply_adaptive_threshold(
            after_sign, window=window, k_sigma=adaptive_k_sigma
        )
    return after_adaptive, sign_mask, adaptive_mask


__all__ = [
    "apply_sign_instability",
    "apply_adaptive_threshold",
    "apply_adaptive_threshold_regime",
    "apply_post_blend_overlays",
]
