"""Stage 1 promotion gate guard.

Bridges the runtime live trader (`pairwise_regime_live.run_live_once`)
with the offline candidate recertification pipeline
(`scripts/recertify_candidates.py`) and the Stage 3.4 two-step approval
workflow (`scripts/promotion_approval.py`).

Unlock paths
------------

There are two ways for the gate to return `unlocked=True`:

1. **Auto-unlock (freeze off):**
   `PAIRWISE_PROMOTION_FREEZE` is unset or `0` AND a fresh, green
   recertification report exists. Used during normal Stage 1+ operation
   once a candidate has earned the right to run.

2. **Approved-unlock (freeze on):**
   `PAIRWISE_PROMOTION_FREEZE=1` (the default Stage 0 lockdown), a
   fresh green recertification report exists, AND a valid two-step
   Telegram approval token is provided via
   `PAIRWISE_PROMOTION_APPROVAL_TOKEN`. The approval must be confirmed
   by an *operator other than the requester* and must not have expired.

Either way, recertification is mandatory: a missing, stale, or red
report blocks both paths. This means the offline CPCV/PBO/DSR pipeline
is never bypassable, while still letting freeze + approval grant the
live system the *opportunity* to act after due diligence — exactly what
Codex stop-time review flagged was previously impossible.

Decision precedence: recertification → freeze → approval. The first
failing gate determines the rejection reason so audit logs are
unambiguous.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping


DEFAULT_REPORT_PATH = Path("models/candidate_recertification_report.json")
DEFAULT_MAX_AGE_DAYS = 7.0


@dataclass(frozen=True)
class GateStatus:
    unlocked: bool
    reason: str
    report_age_days: float | None = None
    report_decision: dict | None = None


def _env_truthy(env: Mapping[str, str], key: str, default: str = "0") -> bool:
    return str(env.get(key, default)).strip().lower() in {"1", "true", "yes", "on"}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _report_age_days(report: dict) -> float | None:
    raw = report.get("generated_at") or report.get("timestamp")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (_now_utc() - ts).total_seconds() / 86400.0


def _check_recertification(
    report_path: Path | str,
    max_age_days: float,
) -> tuple[bool, str, float | None, dict | None]:
    """Return (ok, reason, age_days, decision)."""
    path = Path(report_path)
    if not path.exists():
        return False, f"recertification report missing at {path}", None, None
    try:
        report = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        return (
            False,
            f"recertification report unreadable: {type(exc).__name__}: {exc}",
            None,
            None,
        )
    age = _report_age_days(report)
    decision = report.get("decision") or None
    if age is None:
        return False, "recertification report is undated", None, decision
    # Reject future-dated reports (clock skew / forged timestamp).
    # Allow a small forward tolerance (60s) to absorb routine NTP drift
    # but anything beyond that is treated as evidence of tampering.
    FUTURE_TOLERANCE_DAYS = 60.0 / 86400.0  # 60 seconds in days
    if age < -FUTURE_TOLERANCE_DAYS:
        return (
            False,
            (
                f"recertification report timestamp is {-age * 86400:.0f}s in the "
                f"future — refusing (clock skew or tampering)"
            ),
            age,
            decision,
        )
    if age > max_age_days:
        return (
            False,
            f"recertification report is {age:.1f}d old (max {max_age_days:.1f})",
            age,
            decision,
        )
    if not decision:
        return False, "recertification report has no decision block", age, None
    if not bool(decision.get("ready_for_promotion", False)):
        failed = [k for k, v in decision.items() if k.startswith("pass_") and not v]
        return (
            False,
            "recertification not green: "
            + (",".join(failed) if failed else "decision missing pass flags"),
            age,
            decision,
        )
    return True, "recertification green", age, decision


def _check_approval_token(token: str) -> tuple[bool, str]:
    """Use the Stage 3.4 approval workflow to validate a token."""
    if not token:
        return False, "no approval token supplied"
    try:
        from promotion_approval import is_approval_active
    except Exception as exc:  # noqa: BLE001
        return False, f"promotion_approval module unavailable: {exc}"
    decision = is_approval_active(token)
    if not decision.active:
        return False, f"approval token {token[:8]!r}: {decision.reason}"
    return True, f"approval token {token[:8]!r} active"


def _discover_active_approval() -> tuple[bool, str]:
    """Scan the approval directory for any unexpired, confirmed approval.

    The dynamic token cannot reasonably travel through the static
    `pairwise_live_launchd_env.sh` shipped by `pairwise_live_service.sh`
    — that file is rewritten on every `start` and would leak whatever
    short-lived token is current. Instead we let the operator drop a
    confirmed approval onto disk and have the live process pick it up
    automatically. Whichever approval is still active wins; expired or
    unconfirmed ones are ignored.
    """
    try:
        from promotion_approval import APPROVAL_DIR, is_approval_active
    except Exception as exc:  # noqa: BLE001
        return False, f"promotion_approval module unavailable: {exc}"
    if not APPROVAL_DIR.exists():
        return False, "no approval directory present"
    for path in sorted(APPROVAL_DIR.glob("*.approved.json")):
        token = path.name[:-len(".approved.json")]
        decision = is_approval_active(token)
        if decision.active:
            return True, f"discovered active approval token {token[:8]!r}"
    return False, "no active approvals on disk"


def evaluate_gate(
    env: Mapping[str, str] | None = None,
    *,
    report_path: Path | str | None = None,
    max_age_days: float | None = None,
) -> GateStatus:
    """Evaluate the live promotion gate.

    Returns `unlocked=True` only when:
    - the recertification report is fresh and green, AND
    - either freeze is off, or freeze is on with a valid 2-step
      approval token in `PAIRWISE_PROMOTION_APPROVAL_TOKEN` or an
      active approval discovered on disk.

    `report_path` and `max_age_days` default to the module-level
    constants resolved *at call time* so monkey-patching `DEFAULT_*`
    or rolling out new defaults via env vars takes effect immediately
    without a process restart.
    """
    env = env if env is not None else os.environ
    if report_path is None:
        report_path = DEFAULT_REPORT_PATH
    if max_age_days is None:
        max_age_days = DEFAULT_MAX_AGE_DAYS
    # Stage 1 — recertification is mandatory on every path.
    recert_ok, recert_reason, age, decision = _check_recertification(report_path, max_age_days)
    if not recert_ok:
        return GateStatus(
            unlocked=False,
            reason=recert_reason,
            report_age_days=age,
            report_decision=decision,
        )

    freeze = _env_truthy(env, "PAIRWISE_PROMOTION_FREEZE", "0")
    if not freeze:
        return GateStatus(
            unlocked=True,
            reason="recertification green; freeze=0 — auto-unlock",
            report_age_days=age,
            report_decision=decision,
        )

    # Stage 0 master kill switch is engaged → require Stage 3.4 approval.
    # Look in env first (explicit token preferred), then fall back to the
    # filesystem so production launchd restarts can also unlock without
    # rewriting the static env file every cycle.
    token = str(env.get("PAIRWISE_PROMOTION_APPROVAL_TOKEN", "")).strip()
    if token:
        appr_ok, appr_reason = _check_approval_token(token)
    else:
        appr_ok, appr_reason = _discover_active_approval()
    if not appr_ok:
        return GateStatus(
            unlocked=False,
            reason=(
                "PAIRWISE_PROMOTION_FREEZE=1 — "
                f"approved-unlock requires 2-step approval ({appr_reason})"
            ),
            report_age_days=age,
            report_decision=decision,
        )
    return GateStatus(
        unlocked=True,
        reason=f"recertification green + freeze=1 + {appr_reason}",
        report_age_days=age,
        report_decision=decision,
    )


def is_promotion_unlocked(
    env: Mapping[str, str] | None = None,
    *,
    report_path: Path | str | None = None,
    max_age_days: float | None = None,
) -> bool:
    return evaluate_gate(env, report_path=report_path, max_age_days=max_age_days).unlocked


__all__ = [
    "GateStatus",
    "evaluate_gate",
    "is_promotion_unlocked",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_MAX_AGE_DAYS",
]
