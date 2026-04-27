"""Deflated Sharpe Ratio (DSR).

Reference: David H. Bailey & Marcos López de Prado, *The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality*
(Journal of Portfolio Management, 2014).

DSR is the probability that the *true* Sharpe ratio is greater than zero,
after correcting for:

1. **Multiple-testing selection** — when N candidate strategies are tried
   and the best is reported, the realised Sharpe is biased upward by the
   maximum-of-N order statistic. The Expected-Maximum-Sharpe-under-Null
   (`expected_max_sharpe_under_null`) quantifies this bias.

2. **Non-normal returns** — the variance of the Sharpe estimator under the
   null depends on skewness and excess kurtosis of the strategy's bar
   returns. Ignoring this gives over-optimistic significance.

3. **Sample size** — short back-tests have wider Sharpe sampling
   distributions even under normal returns.

The headline output is `dsr` ∈ [0, 1]: the probability that the true
Sharpe is > 0 given everything observed. A common decision rule is
`dsr >= 0.95` to consider a strategy statistically distinguishable from
no-skill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


_EULER_GAMMA = 0.5772156649015329  # γ — Euler-Mascheroni constant


# --- Normal-distribution helpers (scipy-free) ----------------------------
def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard-normal inverse CDF via Beasley-Springer-Moro approximation.

    Accurate to ~1e-9 across (0, 1). Returns +/- math.inf at the endpoints.
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        raise ValueError(f"p must be in (0,1); got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _sample_skew(arr: np.ndarray) -> float:
    n = arr.size
    if n < 3:
        return float("nan")
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= 0.0:
        return 0.0
    m3 = float(np.mean((arr - mean) ** 3))
    g1 = m3 / (sd ** 3)
    # Bias correction (Fisher-Pearson):
    return g1 * (n * (n - 1)) ** 0.5 / (n - 2)


def _sample_excess_kurtosis(arr: np.ndarray) -> float:
    n = arr.size
    if n < 4:
        return float("nan")
    mean = float(np.mean(arr))
    var = float(np.var(arr, ddof=1))
    if var <= 0.0:
        return 0.0
    m4 = float(np.mean((arr - mean) ** 4))
    g2 = m4 / (var ** 2) - 3.0
    # Sample-bias correction (Joanes-Gill 1998 G2 estimator):
    correction = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0)
    return float(correction)


@dataclass(frozen=True)
class DSRResult:
    sharpe: float
    expected_max_sharpe: float
    dsr: float  # P(true Sharpe > 0)
    skew: float
    kurtosis_excess: float
    n_observations: int
    n_trials: int


def expected_max_sharpe_under_null(n_trials: int, sharpe_std: float = 1.0) -> float:
    """E[max_{i<=N} SR_i] when each SR_i is drawn from a zero-mean normal.

    Approximation from Bailey & López de Prado (2014, eq. 6):
        E[max] ≈ sqrt(V[SR]) × (
            (1 - γ) × Z(1 - 1/N)
            + γ × Z(1 - 1/(N*e))
        )

    `sharpe_std` is the standard deviation of the per-trial Sharpe estimator
    under the null hypothesis (typically 1 for annualised returns).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1; got {n_trials}")
    if n_trials == 1:
        # No multiple-testing correction needed.
        return 0.0
    n = float(n_trials)
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return float(sharpe_std * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def _annualise(per_bar_sharpe: float, periods_per_year: float) -> float:
    return per_bar_sharpe * math.sqrt(periods_per_year)


def deflated_sharpe(
    returns: np.ndarray,
    *,
    n_trials: int,
    periods_per_year: float = 252.0,
    threshold: float = 0.0,
) -> DSRResult:
    """Compute the deflated Sharpe ratio for a return series.

    Parameters
    ----------
    returns : 1-D array of strategy returns at the bar frequency.
    n_trials : number of independent strategy variants tried before this
        one was selected (e.g. multitree v1..v8 → n_trials=8). Setting
        n_trials=1 disables the multiple-testing correction; usually wrong.
    periods_per_year : annualisation factor (252 for daily, 24×365 for
        crypto hourly, etc.).
    threshold : Sharpe value the *true* SR must beat — typically 0.0
        ("does this strategy beat cash?"). Pass `expected_max_sharpe_under_null`
        if you want to test against the no-skill maximum directly.
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 4:
        raise ValueError(f"Need at least 4 observations for DSR; got {n}")
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= 0.0:
        return DSRResult(
            sharpe=float("nan"),
            expected_max_sharpe=float("nan"),
            dsr=float("nan"),
            skew=float("nan"),
            kurtosis_excess=float("nan"),
            n_observations=n,
            n_trials=n_trials,
        )
    sharpe_per_bar = mu / sd
    sharpe_ann = _annualise(sharpe_per_bar, periods_per_year)
    skew = _sample_skew(arr)
    kurt_excess = _sample_excess_kurtosis(arr)

    # Sharpe-ratio sampling variance correction for skew/kurtosis & sample size
    # (Mertens 2002; cited in Bailey-Lopez de Prado 2014 eq. 9):
    #   V[SR] = (1 - γ_3 × SR + ((γ_4)/4) × SR^2) / (n - 1)
    # γ_3 = skew, γ_4 = excess kurtosis. Use *per-bar* Sharpe here.
    var_sr = (
        1.0
        - skew * sharpe_per_bar
        + (kurt_excess / 4.0) * (sharpe_per_bar ** 2)
    ) / max(n - 1, 1)
    if var_sr <= 0.0:
        # Pathological returns (nearly deterministic); treat as no-information.
        return DSRResult(
            sharpe=sharpe_ann,
            expected_max_sharpe=float("nan"),
            dsr=float("nan"),
            skew=skew,
            kurtosis_excess=kurt_excess,
            n_observations=n,
            n_trials=n_trials,
        )
    sr_std_per_bar = math.sqrt(var_sr)

    # Expected-max-Sharpe under the null hypothesis given n_trials.
    # Bailey-LdP express this in the same units as `sharpe_per_bar`,
    # using sr_std_per_bar as the per-trial dispersion under the null.
    sr0_per_bar = expected_max_sharpe_under_null(n_trials, sharpe_std=sr_std_per_bar)

    # DSR: probability that SR_per_bar - threshold > sr0_per_bar (one-sided).
    # In annualised terms the cdf is identical (multiplicative scaling cancels).
    z = (sharpe_per_bar - threshold - sr0_per_bar) / sr_std_per_bar
    dsr = _norm_cdf(z)

    return DSRResult(
        sharpe=sharpe_ann,
        expected_max_sharpe=_annualise(sr0_per_bar, periods_per_year),
        dsr=dsr,
        skew=skew,
        kurtosis_excess=kurt_excess,
        n_observations=n,
        n_trials=n_trials,
    )


__all__ = [
    "DSRResult",
    "deflated_sharpe",
    "expected_max_sharpe_under_null",
]
