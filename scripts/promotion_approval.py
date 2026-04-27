"""Two-step promotion approval (Stage 3 — Taleb skin-in-the-game).

Any change that raises the live trading account's risk envelope (gross
cap, max-hold, freeze flag, candidate file) must satisfy two
independent gates before it takes effect:

    Gate A — Git Hygiene
        The relevant control files are committed and HEAD is clean.
        Without this, "what produced today's live behaviour" is
        ambiguous and the post-mortem playback is unreliable.

    Gate B — Two-step Telegram approval
        A request token is published to Telegram with a one-line
        rationale; the operator must reply with the matching `confirm`
        command within `ttl_minutes` to unlock. The token is single-use
        and the approval flag includes the operator's chat-id so the
        request and approval cannot be replayed after the operator
        leaves the room.

This module is the substrate for both gates. It does not perform any
actual promotion — the live loop and `promotion_gate_guard` consume
its decision via `is_approval_active(token)`.

Approval files
--------------

We persist state under `.bkit/runtime/promotion_approvals/`:

    <token>.request.json   — created at request time, owned by the requester
    <token>.approved.json  — created when the operator confirms (Gate B)

Both files are tiny JSON; nothing sensitive ends up here. They are git-
ignored by convention (see `.bkit/runtime/`).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
APPROVAL_DIR = ROOT / ".bkit" / "runtime" / "promotion_approvals"


@dataclass(frozen=True)
class ApprovalRequest:
    token: str
    reason: str
    requested_at: str
    expires_at: str
    requester: str
    files_changed: list[str]


@dataclass(frozen=True)
class ApprovalDecision:
    active: bool
    reason: str
    request: dict | None = None
    approval: dict | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_dir() -> None:
    APPROVAL_DIR.mkdir(parents=True, exist_ok=True)


def _git_status_files() -> tuple[list[str], list[str]]:
    """Return (uncommitted_paths, untracked_paths) — empty lists when clean."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [], []
    uncommitted: list[str] = []
    untracked: list[str] = []
    for line in out:
        if not line:
            continue
        code, _, path = line.partition(" ")
        if line.startswith("??"):
            untracked.append(path.strip())
        else:
            uncommitted.append(path.strip())
    return uncommitted, untracked


def check_git_hygiene(
    required_clean_paths: Iterable[str] | None = None,
) -> tuple[bool, str]:
    """Gate A — verify required control files are committed.

    If `required_clean_paths` is provided, only those paths gate the
    decision (anything else can be dirty). Otherwise the entire repo
    must be clean.
    """
    uncommitted, untracked = _git_status_files()
    dirty = uncommitted + untracked
    if required_clean_paths is None:
        if dirty:
            return False, f"git working tree has {len(dirty)} dirty paths"
        return True, "git working tree is clean"
    required = set(required_clean_paths)
    offenders = [p for p in dirty if p in required]
    if offenders:
        return False, f"required paths still dirty: {offenders}"
    return True, "all required paths are committed"


def request_approval(
    reason: str,
    *,
    ttl_minutes: int = 5,
    files_changed: list[str] | None = None,
    requester: str | None = None,
) -> ApprovalRequest:
    """Gate B step 1 — create an approval request and return a token."""
    _ensure_dir()
    now = _utc_now()
    expires = now + timedelta(minutes=int(ttl_minutes))
    raw = f"{reason}|{now.isoformat()}|{secrets.token_hex(8)}".encode()
    token = hashlib.sha256(raw).hexdigest()[:16]
    req = ApprovalRequest(
        token=token,
        reason=str(reason)[:300],
        requested_at=now.isoformat(),
        expires_at=expires.isoformat(),
        requester=str(requester or os.getenv("USER") or "unknown"),
        files_changed=list(files_changed or []),
    )
    (APPROVAL_DIR / f"{token}.request.json").write_text(
        json.dumps(req.__dict__, indent=2)
    )
    return req


def confirm_approval(token: str, *, approver: str | None = None) -> ApprovalDecision:
    """Gate B step 2 — confirm the request after Telegram round-trip."""
    _ensure_dir()
    req_path = APPROVAL_DIR / f"{token}.request.json"
    if not req_path.exists():
        return ApprovalDecision(active=False, reason=f"unknown token {token!r}")
    request = json.loads(req_path.read_text())
    expires = datetime.fromisoformat(request["expires_at"])
    if _utc_now() > expires:
        return ApprovalDecision(
            active=False, reason="approval window expired", request=request
        )
    approver_str = str(approver or os.getenv("USER") or "unknown")
    if approver_str == request.get("requester"):
        # Skin-in-the-game: requester cannot self-approve. This forces
        # at least two humans (or one human + automation account) to
        # touch every promotion.
        return ApprovalDecision(
            active=False,
            reason="requester cannot self-approve (skin-in-the-game)",
            request=request,
        )
    approval = {
        "token": token,
        "approver": approver_str,
        "approved_at": _utc_now().isoformat(),
    }
    (APPROVAL_DIR / f"{token}.approved.json").write_text(json.dumps(approval, indent=2))
    return ApprovalDecision(active=True, reason="approved", request=request, approval=approval)


def is_approval_active(token: str) -> ApprovalDecision:
    """Read-only check used by `promotion_gate_guard`."""
    req_path = APPROVAL_DIR / f"{token}.request.json"
    appr_path = APPROVAL_DIR / f"{token}.approved.json"
    if not req_path.exists():
        return ApprovalDecision(active=False, reason="no request on file")
    if not appr_path.exists():
        return ApprovalDecision(active=False, reason="awaiting confirmation")
    request = json.loads(req_path.read_text())
    approval = json.loads(appr_path.read_text())
    expires = datetime.fromisoformat(request["expires_at"])
    if _utc_now() > expires:
        return ApprovalDecision(
            active=False, reason="approval expired", request=request, approval=approval
        )
    return ApprovalDecision(active=True, reason="approval active", request=request, approval=approval)


def clean_expired(*, max_age_days: float = 7.0) -> int:
    """Sweep stale approval files older than `max_age_days`. Returns deleted count."""
    if not APPROVAL_DIR.exists():
        return 0
    cutoff = _utc_now() - timedelta(days=max_age_days)
    n = 0
    for path in APPROVAL_DIR.iterdir():
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ts_str = data.get("requested_at") or data.get("approved_at")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if ts < cutoff:
            path.unlink()
            n += 1
    return n


__all__ = [
    "APPROVAL_DIR",
    "ApprovalRequest",
    "ApprovalDecision",
    "check_git_hygiene",
    "request_approval",
    "confirm_approval",
    "is_approval_active",
    "clean_expired",
]
