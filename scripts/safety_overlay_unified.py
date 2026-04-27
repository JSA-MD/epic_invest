"""Unified safety overlay framework (Stage 3 — Taleb).

Background — patch debt
-----------------------

The live trader carries a long list of independent overlay patches
accreted over months:

    D1  — max-hold auto-flatten
    D2  — gross-cap ceiling
    D3  — cooldown override
    D4  — operation_watchdog stuck-price detector
    D5  — drift-fix env defaults pin
    R2  — 6-pair scaffold
    R3  — CVaR-99 cut
    R4  — trade-frequency override
    "12 sabotage guards" — accumulated edge-case fixes

Each landed in `pairwise_regime_live.py` as bespoke control flow.
That is the *naive interventionism* pattern Taleb warns about: every
new patch removes one symptom while increasing the total surface area
of hidden interactions.

Goal of this module
-------------------

Provide a single evaluator that consumes a structured input and returns
one decision (`UnifiedDecision`). It does not replace the inline
overlays today — instead it gives Stage 3.3 a contract for migrating
each patch one at a time. New overlays added going forward should land
here and *only* here.

Contract
--------

`evaluate_overlays(context)` runs each overlay in priority order and
short-circuits the moment any of them forces flat. The first overlay to
trigger owns the decision; later overlays still record their state in
the journal but do not change the outcome. This makes audit logs
unambiguous: there is always exactly one binding overlay per decision.

Priority (highest → lowest, immutable across calls):

    1. PROMOTION_FREEZE      — Stage 0 master kill switch
    2. CVAR_CUT              — tail-loss circuit breaker (R3)
    3. MAX_HOLD              — hold-period auto-flatten (D1)
    4. SIGN_INSTABILITY      — Stage 2.4 noise-zone detector
    5. STALE_PRICE           — D-pre stale price guard
    6. PARITY_VIOLATION      — Stage 2.3 apples-to-apples failure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OverlayResult(Enum):
    PASS = "pass"
    FLAT = "flat"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class OverlayDecision:
    name: str
    priority: int
    result: OverlayResult
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnifiedDecision:
    force_flat: bool
    binding_overlay: str | None
    overlays: list[OverlayDecision]


def _make(name: str, priority: int, result: OverlayResult, reason: str, **metadata: Any) -> OverlayDecision:
    return OverlayDecision(
        name=name,
        priority=priority,
        result=result,
        reason=reason,
        metadata=dict(metadata),
    )


def evaluate_overlays(context: dict[str, Any]) -> UnifiedDecision:
    """Run every safety overlay in priority order; short-circuit on first FLAT.

    `context` is a flat dict containing per-overlay inputs:

    - `promotion_freeze`: bool (Stage 0)
    - `cvar_cut_active`: bool, optional `cvar_cut_until_ts` for journal
    - `max_hold_exceeded`: bool, optional `max_hold_age_seconds`
    - `sign_instability_verdict`: MonitorVerdict (Stage 2.4)
    - `stale_price_pairs`: list[str]
    - `parity_holds`: bool (Stage 2.3)
    """
    overlays: list[OverlayDecision] = []
    binding: OverlayDecision | None = None

    # 1. Promotion freeze
    if context.get("promotion_freeze"):
        d = _make(
            "PROMOTION_FREEZE", 1, OverlayResult.FLAT,
            "Stage 0 master kill switch",
        )
        overlays.append(d)
        binding = d

    # 2. CVaR-99 cut (R3)
    cvar_active = bool(context.get("cvar_cut_active"))
    if cvar_active:
        d = _make(
            "CVAR_CUT", 2, OverlayResult.FLAT,
            "30-day realised return below CVaR-99 threshold",
            cvar_cut_until_ts=context.get("cvar_cut_until_ts"),
        )
        overlays.append(d)
        if binding is None:
            binding = d
    else:
        overlays.append(_make("CVAR_CUT", 2, OverlayResult.PASS, "no tail-loss circuit"))

    # 3. Max-hold auto-flatten (D1)
    if context.get("max_hold_exceeded"):
        d = _make(
            "MAX_HOLD", 3, OverlayResult.FLAT,
            "position held beyond PAIRWISE_MAX_HOLD_BARS",
            age_seconds=context.get("max_hold_age_seconds"),
        )
        overlays.append(d)
        if binding is None:
            binding = d
    else:
        overlays.append(_make("MAX_HOLD", 3, OverlayResult.PASS, "within hold limit"))

    # 4. Sign-instability monitor (Stage 2.4)
    sv = context.get("sign_instability_verdict")
    if sv is not None and getattr(sv, "force_flat", False):
        d = _make(
            "SIGN_INSTABILITY", 4, OverlayResult.FLAT,
            getattr(sv, "reason", "sign-flip rate exceeds threshold"),
            value=getattr(sv, "value", None),
            threshold=getattr(sv, "threshold", None),
        )
        overlays.append(d)
        if binding is None:
            binding = d
    else:
        overlays.append(_make("SIGN_INSTABILITY", 4, OverlayResult.PASS, "stable signs"))

    # 5. Stale-price guard
    stale_pairs = list(context.get("stale_price_pairs") or [])
    if stale_pairs:
        d = _make(
            "STALE_PRICE", 5, OverlayResult.FLAT,
            f"price feed stalled on {','.join(stale_pairs)}",
            pairs=stale_pairs,
        )
        overlays.append(d)
        if binding is None:
            binding = d
    else:
        overlays.append(_make("STALE_PRICE", 5, OverlayResult.PASS, "fresh price"))

    # 6. Parity violation
    if "parity_holds" in context and not bool(context.get("parity_holds")):
        d = _make(
            "PARITY_VIOLATION", 6, OverlayResult.FLAT,
            "live↔backtest runtime_env parity check failed",
            issues=context.get("parity_issues", []),
        )
        overlays.append(d)
        if binding is None:
            binding = d
    else:
        overlays.append(_make("PARITY_VIOLATION", 6, OverlayResult.PASS, "envs match"))

    return UnifiedDecision(
        force_flat=binding is not None,
        binding_overlay=binding.name if binding else None,
        overlays=overlays,
    )


__all__ = [
    "OverlayResult",
    "OverlayDecision",
    "UnifiedDecision",
    "evaluate_overlays",
]
