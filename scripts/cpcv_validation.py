"""Combinatorial Purged Cross-Validation (CPCV).

Reference: Marcos López de Prado, *Advances in Financial Machine Learning*,
Chapter 12 (especially section 12.4 — Combinatorial Purged CV).

CPCV addresses three weaknesses of plain k-fold CV in financial time series:

1. **Purging** — when label (target) horizon spans multiple bars, training
   samples whose label window overlaps any test sample must be removed,
   otherwise the model gets to "see" the future of test samples through
   leaked label information.

2. **Embargo** — even after purging, a buffer is needed *after* every test
   block because nearby training samples can still be statistically dependent
   on test outcomes via serial correlation.

3. **Multiple back-test paths** — instead of one back-test, CPCV partitions
   the sample into N groups, picks k as test, and enumerates every
   `C(N, k)` combination. Each sample lands in `binom(N-1, k-1)` test sets,
   so we get φ = `binom(N, k) × k / N` independent OOS paths. This estimator
   is far less prone to lucky-split overfitting than a single hold-out.

The module exports two pieces:

- `cpcv_splits()` — generator over `(train_idx, test_idx)` arrays.
- `cpcv_paths()` — assembles per-sample OOS predictions into multiple
  back-test paths so callers can study the *distribution* of strategy
  performance instead of relying on a single number.

Both are pure numpy and therefore safe for the live container (no torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Callable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int
    k_test_groups: int = 2
    embargo_pct: float = 0.01
    label_horizon: int = 0  # in bars; 0 disables purging

    def validate(self) -> None:
        if self.n_groups < 2:
            raise ValueError(f"n_groups must be >=2, got {self.n_groups}")
        if not (1 <= self.k_test_groups < self.n_groups):
            raise ValueError(
                f"k_test_groups must satisfy 1 <= k < n_groups; got "
                f"k={self.k_test_groups} n={self.n_groups}"
            )
        if not (0.0 <= self.embargo_pct < 1.0):
            raise ValueError(f"embargo_pct must be in [0, 1); got {self.embargo_pct}")
        if self.label_horizon < 0:
            raise ValueError(f"label_horizon must be >=0; got {self.label_horizon}")

    @property
    def n_combinations(self) -> int:
        return comb(self.n_groups, self.k_test_groups)

    @property
    def n_paths(self) -> int:
        # Number of OOS paths reconstructed by `cpcv_paths`.
        return comb(self.n_groups - 1, self.k_test_groups - 1)


def _group_assignments(n_samples: int, n_groups: int) -> np.ndarray:
    """Return an int array g[i] ∈ [0, n_groups) marking each sample's group."""
    if n_samples < n_groups:
        raise ValueError(
            f"n_samples={n_samples} must be >= n_groups={n_groups}"
        )
    # Contiguous chunks preserve temporal ordering; this is critical for
    # purging/embargo semantics in time-series back-tests.
    sizes = np.full(n_groups, n_samples // n_groups, dtype=np.int64)
    sizes[: n_samples % n_groups] += 1
    return np.repeat(np.arange(n_groups), sizes)


def _purge_and_embargo(
    train_mask: np.ndarray,
    test_idx: np.ndarray,
    *,
    label_horizon: int,
    embargo_size: int,
) -> np.ndarray:
    """Drop training samples whose label window overlaps test set or its embargo."""
    if test_idx.size == 0:
        return train_mask
    # Each test sample t blocks (t - label_horizon, t + embargo_size] in train.
    n = train_mask.size
    # Iterate over contiguous test runs to keep this O(n + #runs) rather than O(n × |test|).
    sorted_test = np.sort(test_idx)
    # Find boundaries of contiguous runs.
    breaks = np.where(np.diff(sorted_test) > 1)[0]
    starts = np.concatenate(([sorted_test[0]], sorted_test[breaks + 1]))
    ends = np.concatenate((sorted_test[breaks], [sorted_test[-1]]))
    out = train_mask.copy()
    for s, e in zip(starts, ends):
        purge_lo = max(0, int(s) - int(label_horizon))
        embargo_hi = min(n, int(e) + 1 + int(embargo_size))
        out[purge_lo:embargo_hi] = False
    return out


def cpcv_splits(
    n_samples: int,
    config: CPCVConfig,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) for every combinatorial group selection."""
    config.validate()
    groups = _group_assignments(n_samples, config.n_groups)
    embargo_size = int(round(config.embargo_pct * n_samples))
    for chosen in combinations(range(config.n_groups), config.k_test_groups):
        test_mask = np.isin(groups, chosen)
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_mask = _purge_and_embargo(
            train_mask,
            test_idx,
            label_horizon=config.label_horizon,
            embargo_size=embargo_size,
        )
        train_idx = np.where(train_mask)[0]
        yield train_idx, test_idx


def cpcv_paths(
    n_samples: int,
    config: CPCVConfig,
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Run CPCV and return predictions for every reconstructed OOS path.

    `fit_predict(train_idx, test_idx)` must return predictions aligned with
    `test_idx`. The returned array has shape `(n_paths, n_samples)`.

    Each sample lands in `n_paths = C(N-1, k-1)` distinct test sets across
    all combinations. We deal one prediction per sample to each path slot in
    a deterministic order (per-sample slot counter), giving each path a
    contiguous OOS prediction trace whose values were never produced by a
    model that saw that sample at training time.
    """
    config.validate()
    n_paths = config.n_paths
    if n_paths < 1:
        raise ValueError("CPCV produces no paths with this configuration")
    out = np.full((n_paths, n_samples), np.nan, dtype=np.float64)
    slot = np.zeros(n_samples, dtype=np.int64)
    for train_idx, test_idx in cpcv_splits(n_samples, config):
        preds = np.asarray(fit_predict(train_idx, test_idx), dtype=np.float64)
        if preds.shape[0] != test_idx.shape[0]:
            raise ValueError(
                f"fit_predict returned {preds.shape[0]} predictions for "
                f"{test_idx.shape[0]} test samples"
            )
        for i, idx in enumerate(test_idx):
            s = int(slot[idx])
            if s >= n_paths:
                raise RuntimeError(
                    f"CPCV slot overflow at sample {idx}: each sample should "
                    f"land in exactly {n_paths} test folds; got {s + 1}"
                )
            out[s, idx] = preds[i]
            slot[idx] = s + 1
    return out


def cpcv_oos_sharpe(
    returns: np.ndarray,
    signal_paths: np.ndarray,
    *,
    annualization: float = 252.0,
) -> dict:
    """Aggregate per-path OOS Sharpe statistics.

    `signal_paths` is `(n_paths, n_samples)` from `cpcv_paths`; `returns`
    is `(n_samples,)`. Returns mean / median / 5%-quantile of per-path
    Sharpe and the fraction of paths with positive Sharpe.
    """
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-D; got shape {returns.shape}")
    if signal_paths.shape[1] != returns.shape[0]:
        raise ValueError(
            f"signal_paths cols ({signal_paths.shape[1]}) != "
            f"returns len ({returns.shape[0]})"
        )
    sharpes = []
    for path in signal_paths:
        mask = np.isfinite(path) & np.isfinite(returns)
        pnl = path[mask] * returns[mask]
        if pnl.size < 8:  # too few OOS bars to trust
            continue
        std = float(np.std(pnl, ddof=1))
        if std <= 0.0:
            continue
        s = float(np.mean(pnl) / std * np.sqrt(annualization))
        sharpes.append(s)
    sharpes = np.array(sharpes, dtype=np.float64)
    if sharpes.size == 0:
        return {
            "n_paths_evaluated": 0,
            "mean_sharpe": float("nan"),
            "median_sharpe": float("nan"),
            "p05_sharpe": float("nan"),
            "positive_fraction": float("nan"),
        }
    return {
        "n_paths_evaluated": int(sharpes.size),
        "mean_sharpe": float(np.mean(sharpes)),
        "median_sharpe": float(np.median(sharpes)),
        "p05_sharpe": float(np.quantile(sharpes, 0.05)),
        "positive_fraction": float(np.mean(sharpes > 0.0)),
    }


__all__ = [
    "CPCVConfig",
    "cpcv_splits",
    "cpcv_paths",
    "cpcv_oos_sharpe",
]
