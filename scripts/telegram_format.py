#!/usr/bin/env python3
"""Shared Telegram notification formatting library for Epic Invest agents.

All user-facing strings are in Korean. Import this module; do not duplicate
formatting logic in individual agent scripts.
"""

from __future__ import annotations

import enum
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------

KST = timezone(timedelta(hours=9))

# ---------------------------------------------------------------------------
# Alert levels
# ---------------------------------------------------------------------------


class AlertLevel(enum.Enum):
    CRITICAL = "critical"   # 🚨 immediate, inline buttons, 1-min cooldown
    HIGH = "high"           # ⚠️ immediate, 15-min debounce per (key, level)
    INFO = "info"           # 🔵 daily digest only (DO NOT send standalone)
    SUCCESS = "success"     # ✅ once per state transition
    DIGEST = "digest"       # 📊 hourly/daily summary cards


EMOJI: dict[AlertLevel, str] = {
    AlertLevel.CRITICAL: "🚨",
    AlertLevel.HIGH: "⚠️",
    AlertLevel.INFO: "🔵",
    AlertLevel.SUCCESS: "✅",
    AlertLevel.DIGEST: "📊",
}

# ---------------------------------------------------------------------------
# i18n strings
# ---------------------------------------------------------------------------

LEVEL_NAMES: dict[AlertLevel, str] = {
    AlertLevel.CRITICAL: "긴급",
    AlertLevel.HIGH: "주의",
    AlertLevel.INFO: "정보",
    AlertLevel.SUCCESS: "정상",
    AlertLevel.DIGEST: "요약",
}

SAFETY_LABELS: dict[str, str] = {
    "gross_cap": "Gross cap (D2)",
    "max_hold": "Max-hold 24h (D1)",
    "cvar_cut": "CVaR-99 cut (R3)",
    "reconciliation": "포지션 일치 (D3)",
    "price_feed": "가격 피드 신선 (D4)",
}

BUTTON_LABELS: dict[str, str] = {
    "stop": "🛑 정지",
    "snooze_1h": "⏸ 1h 스누즈",
    "snooze_6h": "⏸ 6h 스누즈",
    "details": "📊 상세",
    "chart": "📈 차트",
    "resume": "🔓 재개",
}

# Inline i18n table (Korean only)
_I18N: dict[str, str] = {
    **{f"level.{k.value}": v for k, v in LEVEL_NAMES.items()},
    **{f"safety.{k}": v for k, v in SAFETY_LABELS.items()},
    **{f"button.{k}": v for k, v in BUTTON_LABELS.items()},
    "bot.name": "pairwise demo",
    "time.label": "⏰",
    "bot.label": "🤖",
}

# Markdown special characters that must be escaped in Markdown parse_mode
_MD_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def _md_escape(text: str) -> str:
    """Escape Markdown special characters in user-supplied content."""
    return _MD_SPECIAL.sub(r"\\\1", text)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def now_kst() -> datetime:
    """Return current time as timezone-aware datetime in KST."""
    return datetime.now(tz=KST)


def format_kst(dt: datetime | str) -> str:
    """Format ISO string or datetime as 'MM/DD HH:MM KST'."""
    if isinstance(dt, str):
        # Accept ISO 8601 with or without timezone
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_kst = dt.astimezone(KST)
    return dt_kst.strftime("%m/%d %H:%M") + " KST"


# ---------------------------------------------------------------------------
# i18n lookup
# ---------------------------------------------------------------------------


def localize(message_id: str, **kwargs: str) -> str:
    """Return the Korean string for *message_id*, with optional format args."""
    template = _I18N.get(message_id, message_id)
    return template.format(**kwargs) if kwargs else template


# ---------------------------------------------------------------------------
# Debounce helpers
# ---------------------------------------------------------------------------

_DEBOUNCE_SECONDS: dict[AlertLevel, int] = {
    AlertLevel.CRITICAL: 60,
    AlertLevel.HIGH: 900,
    AlertLevel.INFO: 86400,
    AlertLevel.SUCCESS: 3600,
    AlertLevel.DIGEST: 3600,
}


def debounce_seconds(level: AlertLevel) -> int:
    """Return debounce window in seconds for the given level."""
    return _DEBOUNCE_SECONDS[level]


def in_quiet_hours(now: datetime | None = None) -> bool:
    """Return True if current KST time is in quiet hours (22:00–08:00)."""
    if now is None:
        now = now_kst()
    t = now.astimezone(KST)
    hour = t.hour
    return hour >= 22 or hour < 8


def should_send(
    level: AlertLevel,
    key: str,
    *,
    snooze_until: Optional[datetime] = None,
) -> bool:
    """Apply per-level debounce + quiet-hours + snooze.

    State is persisted to /tmp/epic-invest-tg-debounce-<key>.json.
    Returns True if the alert should be sent now.
    """
    state_path = Path(f"/tmp/epic-invest-tg-debounce-{key}.json")
    now_ts = time.time()

    # Snooze check
    if snooze_until is not None:
        snooze_ts = snooze_until.timestamp()
        if now_ts < snooze_ts:
            return False

    # Quiet hours suppress non-critical alerts
    if level not in (AlertLevel.CRITICAL, AlertLevel.HIGH):
        if in_quiet_hours():
            return False

    # INFO is digest-only; never send standalone
    if level == AlertLevel.INFO:
        return False

    # Read existing debounce state
    last_sent: float = 0.0
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text())
            last_sent = float(data.get("last_sent", 0))
        except (json.JSONDecodeError, ValueError, OSError):
            last_sent = 0.0

    window = debounce_seconds(level)
    if now_ts - last_sent < window:
        return False

    # Persist new timestamp
    try:
        state_path.write_text(json.dumps({"last_sent": now_ts}))
    except OSError:
        pass

    return True


# ---------------------------------------------------------------------------
# Core formatter
# ---------------------------------------------------------------------------

_BOT_NAME = "pairwise demo"
_DISABLE_NOTIFICATION_LEVELS = {AlertLevel.INFO, AlertLevel.DIGEST, AlertLevel.SUCCESS}


def format_alert(
    level: AlertLevel,
    title: str,
    body: str,
    context: dict[str, str] | None = None,
    buttons: Sequence[dict[str, str]] | None = None,
    timestamp_kst: datetime | None = None,
    pair: str | None = None,
) -> dict:
    """Build a Telegram sendMessage payload dict.

    Returns:
        {
          "text": str,
          "parse_mode": "Markdown",
          "reply_markup": {...} | None,
          "disable_notification": bool,
        }
    """
    if timestamp_kst is None:
        timestamp_kst = now_kst()

    ts_str = format_kst(timestamp_kst)
    emoji = EMOJI[level]

    # Header line: emoji + bold title
    # Escape user content; title is already expected to be a trusted label,
    # but we escape anyway for safety.
    header = f"{emoji} *{_md_escape(title)}*"

    # Meta line: time + bot tag (+ pair if provided)
    meta_parts = [f"⏰ {ts_str}", f"🤖 {_BOT_NAME}"]
    if pair:
        meta_parts.append(f"📌 {_md_escape(pair)}")
    meta_line = "  |  ".join(meta_parts)

    # Body (escape special chars)
    body_escaped = _md_escape(body)

    # Context block
    context_block = ""
    if context:
        lines = [f"  · {_md_escape(k)}: {_md_escape(v)}" for k, v in context.items()]
        context_block = "\n".join(lines)

    # Assemble text
    parts = [header, meta_line, "", body_escaped]
    if context_block:
        parts += ["", context_block]
    text = "\n".join(parts)

    # Inline keyboard
    reply_markup = None
    if buttons:
        # All buttons on a single row; callers may pass multiple rows by
        # using a list-of-lists pattern in future — for now one row.
        row = [{"text": btn["label"], "callback_data": btn["callback_data"]} for btn in buttons]
        reply_markup = {"inline_keyboard": [row]}

    disable_notification = level in _DISABLE_NOTIFICATION_LEVELS

    return {
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": reply_markup,
        "disable_notification": disable_notification,
    }
