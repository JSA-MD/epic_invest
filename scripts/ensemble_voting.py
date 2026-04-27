"""Multi-candidate ensemble voting (Stage 2 — Simons / Renaissance pattern).

Single-candidate live trading is brittle: a single regime gate flip or
data anomaly toggles the position from full short to flat. The legacy
log shows BTC `requested_weight` flipping between −0.07 and 0 within
two consecutive 5-min bars on the same `regime_score = 0.019` simply
because of a `±0.02` cutoff. Renaissance-style robustness comes from
*ensembling many weak signals* and demanding a quorum before acting.

This module gives the live loop a way to combine N candidate weights
into one robust target weight using a quorum rule:

- Majority sign vote: count how many candidates ask for long, short, or
  flat. The combined sign is the strict majority side; if no side has a
  majority the combined signal is flat.
- Per-side magnitude: median of the absolute weights from candidates
  that voted for the winning side. Median is preferred over mean because
  it ignores outliers (one runaway candidate cannot single-handedly
  dictate sizing).

The output is intentionally conservative: requiring a strict majority
(e.g. ≥3 of 5) cuts whip-saw rate dramatically. The trade-off is fewer
trades — exactly what López de Prado / Simons recommend after PBO
analysis flags overfit risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnsembleVote:
    combined_weight: float
    sign: int  # +1, -1, or 0
    n_long: int
    n_short: int
    n_flat: int
    quorum_required: int
    candidates_total: int


def majority_vote(
    candidate_weights: list[float],
    *,
    weight_eps: float = 1e-6,
    quorum: int | None = None,
) -> EnsembleVote:
    """Combine candidate weights into a single robust target.

    Parameters
    ----------
    candidate_weights : per-candidate target_weight values (signed).
    weight_eps : magnitudes below this count as a "flat" vote.
    quorum : number of votes required on a single side to act. Defaults
        to `ceil((n + 1) / 2)` — i.e. strict majority.
    """
    arr = np.asarray(candidate_weights, dtype=np.float64)
    n = arr.size
    if n == 0:
        return EnsembleVote(0.0, 0, 0, 0, 0, 0, 0)
    if quorum is None:
        quorum = (n // 2) + 1
    if quorum < 1:
        raise ValueError(f"quorum must be >= 1; got {quorum}")
    long_mask = arr > weight_eps
    short_mask = arr < -weight_eps
    n_long = int(np.sum(long_mask))
    n_short = int(np.sum(short_mask))
    n_flat = int(n - n_long - n_short)
    sign = 0
    combined = 0.0
    if n_long >= quorum and n_long > n_short:
        sign = 1
        combined = float(np.median(np.abs(arr[long_mask])))
    elif n_short >= quorum and n_short > n_long:
        sign = -1
        combined = -float(np.median(np.abs(arr[short_mask])))
    return EnsembleVote(
        combined_weight=combined,
        sign=sign,
        n_long=n_long,
        n_short=n_short,
        n_flat=n_flat,
        quorum_required=quorum,
        candidates_total=n,
    )


def ensemble_pair_weights(
    per_candidate_pair_weights: dict[str, dict[str, float]],
    *,
    pairs: list[str],
    weight_eps: float = 1e-6,
    quorum: int | None = None,
) -> dict[str, EnsembleVote]:
    """Apply majority_vote per pair across all candidates.

    `per_candidate_pair_weights` maps candidate_id → pair → weight, e.g.
    `{"v2": {"BTCUSDT": -0.05, "BNBUSDT": 0.0}, "v3": {...}}`.
    Missing pairs default to 0 (flat).
    """
    out: dict[str, EnsembleVote] = {}
    for pair in pairs:
        weights = [
            float(c.get(pair, 0.0))
            for c in per_candidate_pair_weights.values()
        ]
        out[pair] = majority_vote(weights, weight_eps=weight_eps, quorum=quorum)
    return out


__all__ = ["EnsembleVote", "majority_vote", "ensemble_pair_weights"]
