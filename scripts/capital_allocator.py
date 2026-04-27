"""Barbell capital allocator (Stage 3 — Taleb).

Taleb's *Antifragile* (Ch. 19) prescribes a "barbell" portfolio: most of
the capital sits in something that cannot blow up (cash, T-bills, hard
collateral) and only a small slice is exposed to convex bets that *can*
blow up. The crypto-trader equivalent is:

    total_equity_usd = cash_reserve + trading_account
    cash_reserve   = total_equity_usd × (1 − trading_fraction)
    trading_account = total_equity_usd × trading_fraction

`trading_fraction` defaults to **0.20** (20%). With Stage 0 lockdown's
`gross_cap = 0.01`, the worst-case daily exposure is
`0.20 × 0.01 = 0.002 = 0.2%` of total equity per pair, which is the
"survives a Black Swan" regime Taleb argues you must protect.

The allocator publishes a per-pair max-notional that the live loop can
clip against. It does not move money for you — moving funds between
sub-accounts is a manual, journal-logged operation by design (skin in
the game; Stage 3.4 gates that behind a 2-step approval).

Decision rule
-------------

`per_pair_max_notional_usd = cash_reserve == 0  →  trading_account / N
                            cash_reserve != 0  →  trading_account × gross_cap`

Callers should treat the result as an *additional* upper bound on the
sizing computed by Kelly / regime / etc. — never a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_TRADING_FRACTION = 0.20
DEFAULT_PAIR_FLOOR_USD = 25.0  # below this, trading is uneconomic at maker fees


@dataclass(frozen=True)
class BarbellAllocation:
    total_equity_usd: float
    trading_fraction: float
    cash_reserve_usd: float
    trading_account_usd: float
    per_pair_max_notional_usd: dict[str, float]
    n_pairs: int
    notes: list[str]


def allocate_barbell(
    total_equity_usd: float,
    *,
    pairs: list[str],
    trading_fraction: float = DEFAULT_TRADING_FRACTION,
    gross_cap_per_pair: float = 1.0,
    pair_floor_usd: float = DEFAULT_PAIR_FLOOR_USD,
) -> BarbellAllocation:
    """Split equity into cash reserve + trading account; cap per-pair notional.

    Parameters
    ----------
    total_equity_usd : combined cash across all sub-accounts.
    pairs : tradable symbols.
    trading_fraction : fraction of equity allowed to live in the trading
        account at any time. Default 0.20.
    gross_cap_per_pair : multiplier applied to the trading account when
        deriving per-pair notional. Should be ≤ 1.0 typically; with the
        Stage 0 lockdown gross_cap = 0.01, you'd pass 0.01 here.
    pair_floor_usd : per-pair minimum notional below which we just zero
        out (no point sending dust orders).
    """
    if not (0.0 < trading_fraction <= 1.0):
        raise ValueError(
            f"trading_fraction must be in (0,1]; got {trading_fraction}"
        )
    if gross_cap_per_pair < 0.0:
        raise ValueError(f"gross_cap_per_pair must be >= 0; got {gross_cap_per_pair}")
    if total_equity_usd < 0.0:
        raise ValueError(f"total_equity_usd must be >= 0; got {total_equity_usd}")

    notes: list[str] = []
    cash_reserve = total_equity_usd * (1.0 - trading_fraction)
    trading_account = total_equity_usd * trading_fraction
    n = len(pairs)
    raw_per_pair = trading_account * gross_cap_per_pair
    per_pair_max: dict[str, float] = {}
    for pair in pairs:
        cap = raw_per_pair
        if cap < pair_floor_usd:
            cap = 0.0
            notes.append(
                f"{pair}: per-pair notional {raw_per_pair:.2f} USD < floor {pair_floor_usd:.2f} → set to 0"
            )
        per_pair_max[pair] = float(cap)

    if trading_account < pair_floor_usd * max(n, 1):
        notes.append(
            f"trading_account {trading_account:.2f} USD < floor × pairs "
            f"({pair_floor_usd * max(n, 1):.2f}) — barbell may starve all positions"
        )
    if trading_fraction > 0.5:
        notes.append(
            f"trading_fraction {trading_fraction:.2f} > 0.5 — this is no longer a "
            f"barbell; consider lowering to <=0.20 per Taleb"
        )

    return BarbellAllocation(
        total_equity_usd=float(total_equity_usd),
        trading_fraction=float(trading_fraction),
        cash_reserve_usd=float(cash_reserve),
        trading_account_usd=float(trading_account),
        per_pair_max_notional_usd=per_pair_max,
        n_pairs=n,
        notes=notes,
    )


def clip_target_weight_to_barbell(
    desired_weight: float,
    pair: str,
    allocation: BarbellAllocation,
    *,
    base_notional_usd: float,
) -> tuple[float, list[str]]:
    """Clip `desired_weight` so its absolute notional respects the barbell cap.

    `base_notional_usd` is the notional that `desired_weight = ±1.0`
    would generate (typically the live trader's account equity *before*
    barbell). Returns `(clipped_weight, notes)`.
    """
    notes: list[str] = []
    cap = allocation.per_pair_max_notional_usd.get(pair, 0.0)
    if cap <= 0.0:
        return 0.0, [f"{pair}: barbell cap is 0 — forcing flat"]
    desired_notional = abs(float(desired_weight)) * float(base_notional_usd)
    if desired_notional <= cap:
        return float(desired_weight), notes
    # Clip — preserve sign, scale magnitude.
    sign = 1.0 if desired_weight >= 0.0 else -1.0
    scaled = sign * (cap / max(base_notional_usd, 1e-9))
    notes.append(
        f"{pair}: barbell clip {desired_weight:+.4f} → {scaled:+.4f} "
        f"(notional {desired_notional:.2f} > cap {cap:.2f})"
    )
    return float(scaled), notes


__all__ = [
    "BarbellAllocation",
    "DEFAULT_TRADING_FRACTION",
    "DEFAULT_PAIR_FLOOR_USD",
    "allocate_barbell",
    "clip_target_weight_to_barbell",
]
