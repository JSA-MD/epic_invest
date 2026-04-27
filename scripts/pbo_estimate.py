"""Probability of Backtest Overfitting (PBO) via CSCV.

Reference: David H. Bailey, Jonathan M. Borwein, Marcos López de Prado &
Qiji Jim Zhu, *The Probability of Backtest Overfitting* (Journal of
Computational Finance, 2014).

Combinatorially Symmetric Cross-Validation (CSCV) takes an `(T, N)` matrix
of `N` candidate-strategy returns over `T` periods and asks: when we pick
the best strategy on one half of the time index, how often does it land in
the *bottom* half on the held-out other half?

Procedure:

1. Cut the time axis into `S` (even) equal-length subgroups.
2. For every way of choosing `S/2` subgroups as in-sample (IS), the
   complementary `S/2` form the out-of-sample (OOS) block. There are
   `C(S, S/2)` such symmetric splits.
3. On each split:
   a. Compute the IS performance metric (Sharpe by default) for every
      strategy and pick the IS-best `n*`.
   b. Compute the OOS performance metric for every strategy and rank `n*`
      against the others (1 = best, N = worst).
   c. Convert the OOS rank to a percentile `ω = rank / (N + 1)` and the
      logit transform `λ = ln(ω / (1 - ω))`.
4. PBO = `P(λ < 0)` ≈ fraction of splits where the IS-best strategy lands
   below the OOS median.

A PBO close to 0.5 means selecting the IS-best strategy gives no edge OOS;
0 means the selection generalises perfectly. A common decision rule is
`pbo < 0.5` to deem the selection process informative.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_splits: int
    n_strategies: int
    n_periods: int
    s_subgroups: int
    median_oos_rank_pct: float  # median ω across splits (0..1)
    mean_logit: float
    fraction_oos_top_quartile: float  # diagnostic — how often IS-best stays top-25%


def _sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return float("nan")
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 0.0:
        return 0.0
    return mu / sd * sqrt(returns.size)  # period-scaled (no annualisation needed for ranking)


def _matrix_metric(returns_matrix: np.ndarray, mode: str) -> np.ndarray:
    """Per-strategy performance metric across the time axis.

    `returns_matrix` is shape (T, N). Returns shape (N,).
    """
    if mode == "sharpe":
        mu = np.mean(returns_matrix, axis=0)
        sd = np.std(returns_matrix, axis=0, ddof=1)
        out = np.where(sd > 0, mu / sd * sqrt(returns_matrix.shape[0]), 0.0)
        return out
    if mode == "mean":
        return np.mean(returns_matrix, axis=0)
    if mode == "total":
        return np.sum(returns_matrix, axis=0)
    raise ValueError(f"unknown metric mode {mode!r}")


def estimate_pbo(
    returns_matrix: np.ndarray,
    *,
    s_subgroups: int = 16,
    metric: str = "sharpe",
) -> PBOResult:
    """Estimate PBO for an (T, N) matrix of strategy returns.

    Parameters
    ----------
    returns_matrix : 2-D array of shape (T, N) — N candidate strategies'
        per-period returns over T common periods.
    s_subgroups : even integer; the time axis is split into `s_subgroups`
        equal blocks. With S=16 there are C(16,8)=12,870 symmetric splits.
    metric : "sharpe", "mean", or "total" — how to score strategies on
        each in-sample / out-of-sample block.
    """
    arr = np.asarray(returns_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"returns_matrix must be 2-D; got shape {arr.shape}")
    t, n = arr.shape
    if n < 2:
        raise ValueError(f"need at least 2 strategies; got {n}")
    if s_subgroups < 2 or s_subgroups % 2 != 0:
        raise ValueError(f"s_subgroups must be even and >= 2; got {s_subgroups}")
    if t < s_subgroups * 2:
        raise ValueError(
            f"need T >= 2 * s_subgroups; got T={t}, s={s_subgroups}"
        )

    # Equal-length contiguous subgroup index ranges.
    bin_size = t // s_subgroups
    subgroups: list[np.ndarray] = []
    for i in range(s_subgroups):
        lo = i * bin_size
        hi = (i + 1) * bin_size if i < s_subgroups - 1 else t
        subgroups.append(np.arange(lo, hi))

    half = s_subgroups // 2
    logits: list[float] = []
    rank_pcts: list[float] = []
    top_quartile_hits = 0
    n_splits = 0
    for is_choice in combinations(range(s_subgroups), half):
        n_splits += 1
        is_idx = np.concatenate([subgroups[i] for i in is_choice])
        oos_idx = np.concatenate(
            [subgroups[i] for i in range(s_subgroups) if i not in is_choice]
        )
        is_perf = _matrix_metric(arr[is_idx], metric)
        # IS-best strategy (highest score is winner).
        n_star = int(np.argmax(is_perf))
        oos_perf = _matrix_metric(arr[oos_idx], metric)
        # OOS rank: 1 = best, N = worst.
        # Tie-handling: average rank.
        order = np.argsort(-oos_perf, kind="stable")  # descending
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        rank_n_star = float(ranks[n_star])
        omega = rank_n_star / (n + 1.0)
        # Clip for finite logit
        omega_c = min(max(omega, 1e-9), 1.0 - 1e-9)
        logits.append(float(np.log(omega_c / (1.0 - omega_c))))
        rank_pcts.append(omega)
        if rank_n_star <= n / 4.0:
            top_quartile_hits += 1

    logits_arr = np.array(logits, dtype=np.float64)
    rank_pcts_arr = np.array(rank_pcts, dtype=np.float64)
    # rank=1 → best, so omega = rank/(N+1) is small ⇒ logit is negative for
    # *good* OOS ranks. PBO is the probability the IS-best lands in the OOS
    # bottom half ⇒ rank > N/2 ⇒ omega > 0.5 ⇒ logit > 0. Hence:
    pbo = float(np.mean(logits_arr > 0.0))
    return PBOResult(
        pbo=pbo,
        n_splits=n_splits,
        n_strategies=n,
        n_periods=t,
        s_subgroups=s_subgroups,
        median_oos_rank_pct=float(np.median(rank_pcts_arr)),
        mean_logit=float(np.mean(logits_arr)),
        fraction_oos_top_quartile=float(top_quartile_hits / n_splits),
    )


__all__ = ["PBOResult", "estimate_pbo"]
