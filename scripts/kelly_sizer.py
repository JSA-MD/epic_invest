"""Kelly-fraction position sizer (Stage 3 — Thorp).

Edward Thorp's Kelly criterion in continuous form is

    f* = μ / σ²

where μ is the per-bar expected return and σ² is the per-bar variance
of the return distribution. Plugging in the realised sample mean and
variance is the *full Kelly* and is universally too aggressive for live
trading: the realised μ has its own sampling error and you can be
unwittingly betting on an overestimate.

The community standard fix is **fractional Kelly** at 0.25× — bet a
quarter of the full-Kelly weight. *Beat the Market* (Thorp 1967) and
later writing by López de Prado both endorse this. Fractional Kelly
keeps growth rate close to optimal while drastically reducing draw-down
variance, and (importantly) makes the strategy robust to overestimating
μ by ~2× — exactly the bias produced by Stage 1's PBO concerns.

The sizer here goes one step further: instead of using the raw sample
mean we use the *lower confidence bound* on μ. With the standard error
`SE = σ / √n`, a one-sided 95% lower bound is `μ_low = μ − 1.645 × SE`.
If `μ_low ≤ 0` the Kelly fraction is forced to zero — there is not
enough evidence that the edge is positive.

This module is pure-numpy and intended to be called once per cycle
from the live loop with the recent return window for each pair. It
returns a per-pair Kelly weight in `[-1, 1]` that the caller composes
with the regime / blend / barbell signals (always min-clipped, never
multiplicatively boosted).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


DEFAULT_KELLY_MULTIPLIER = 0.25
DEFAULT_CONFIDENCE_Z = 1.645  # one-sided 95%


@dataclass(frozen=True)
class KellyEstimate:
    pair: str
    n_samples: int
    mu: float          # per-bar mean return
    sigma: float       # per-bar std dev
    mu_lower: float    # one-sided lower bound on μ
    full_kelly: float  # μ_lower / σ²
    fractional_kelly: float  # multiplier × full_kelly, clipped to [-1, 1]
    direction_hint: int  # +1, −1, 0 from baseline signal sign
    reason: str


def _kelly_clip(value: float, max_abs: float = 1.0) -> float:
    if value > max_abs:
        return float(max_abs)
    if value < -max_abs:
        return float(-max_abs)
    return float(value)


def kelly_fraction(
    returns: np.ndarray,
    *,
    pair: str = "",
    direction_hint: int = 0,
    kelly_multiplier: float = DEFAULT_KELLY_MULTIPLIER,
    confidence_z: float = DEFAULT_CONFIDENCE_Z,
    min_samples: int = 30,
    max_abs_weight: float = 1.0,
) -> KellyEstimate:
    """Compute the recommended fractional-Kelly position for one pair.

    `direction_hint` is the sign the upstream signal proposes (+1 long,
    −1 short, 0 flat). When the Kelly bet has a sign that disagrees with
    the hint, we abstain — Kelly should never push you against your
    own model's directional view.
    """
    arr = np.asarray(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < min_samples:
        return KellyEstimate(
            pair=pair,
            n_samples=n,
            mu=float("nan"),
            sigma=float("nan"),
            mu_lower=float("nan"),
            full_kelly=0.0,
            fractional_kelly=0.0,
            direction_hint=int(direction_hint),
            reason=f"insufficient history ({n} < {min_samples})",
        )
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1))
    # Guard against floating-point dust that would make Kelly = mu/0 → inf.
    if sigma <= 1e-12:
        return KellyEstimate(
            pair=pair,
            n_samples=n,
            mu=mu,
            sigma=sigma,
            mu_lower=float("nan"),
            full_kelly=0.0,
            fractional_kelly=0.0,
            direction_hint=int(direction_hint),
            reason="zero variance — Kelly undefined",
        )
    se = sigma / math.sqrt(n)
    mu_lower = mu - confidence_z * se
    full = mu_lower / (sigma * sigma)
    if mu_lower <= 0.0:
        # not enough evidence the edge is positive
        return KellyEstimate(
            pair=pair,
            n_samples=n,
            mu=mu,
            sigma=sigma,
            mu_lower=mu_lower,
            full_kelly=float(full),
            fractional_kelly=0.0,
            direction_hint=int(direction_hint),
            reason=f"μ_lower={mu_lower:.4g} <= 0 — abstain (no positive edge)",
        )
    raw = kelly_multiplier * full
    # Honour direction hint: if Kelly says long but upstream says short,
    # flip to flat. Symmetrically for short.
    sign = 1 if raw > 0 else (-1 if raw < 0 else 0)
    if direction_hint != 0 and sign != 0 and sign != direction_hint:
        return KellyEstimate(
            pair=pair,
            n_samples=n,
            mu=mu,
            sigma=sigma,
            mu_lower=mu_lower,
            full_kelly=float(full),
            fractional_kelly=0.0,
            direction_hint=int(direction_hint),
            reason=(
                f"Kelly sign {sign} opposes direction_hint {direction_hint} — abstain"
            ),
        )
    fractional = _kelly_clip(raw * direction_hint if direction_hint != 0 else raw, max_abs_weight)
    return KellyEstimate(
        pair=pair,
        n_samples=n,
        mu=mu,
        sigma=sigma,
        mu_lower=mu_lower,
        full_kelly=float(full),
        fractional_kelly=float(fractional),
        direction_hint=int(direction_hint),
        reason=f"OK — fractional Kelly={fractional:.4f}",
    )


__all__ = [
    "KellyEstimate",
    "DEFAULT_KELLY_MULTIPLIER",
    "DEFAULT_CONFIDENCE_Z",
    "kelly_fraction",
]
