"""Unit tests for scripts/telegram_format.py."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from telegram_format import (
    AlertLevel,
    KST,
    debounce_seconds,
    format_alert,
    format_kst,
    in_quiet_hours,
    now_kst,
    should_send,
)


# ---------------------------------------------------------------------------
# 1. CRITICAL message has 🚨 prefix
# ---------------------------------------------------------------------------

def test_critical_emoji_prefix():
    result = format_alert(AlertLevel.CRITICAL, "테스트 제목", "본문 내용")
    assert result["text"].startswith("🚨 ")


# ---------------------------------------------------------------------------
# 2. KST timestamp formatted as MM/DD HH:MM KST
# ---------------------------------------------------------------------------

def test_format_kst_output():
    # 2025-01-15T14:30:00 UTC  ->  2025-01-15T23:30:00 KST -> "01/15 23:30 KST"
    dt_utc = datetime(2025, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
    result = format_kst(dt_utc)
    assert result == "01/15 23:30 KST"


def test_format_kst_from_iso_string():
    # ISO string without timezone treated as UTC per fromisoformat -> aware
    iso = "2025-06-01T03:00:00+00:00"
    result = format_kst(iso)
    assert result == "06/01 12:00 KST"


# ---------------------------------------------------------------------------
# 3. context dict renders as ordered list
# ---------------------------------------------------------------------------

def test_context_block_rendering():
    ctx = {"🔒 pair": "BTC/USDT", "delta": "0.05"}
    result = format_alert(AlertLevel.HIGH, "제목", "본문", context=ctx)
    text = result["text"]
    # Both keys should appear preceded by "  · "
    assert "  · 🔒 pair: BTC/USDT" in text
    assert "  · delta: 0\\.05" in text  # dot is md-escaped


# ---------------------------------------------------------------------------
# 4. buttons produce inline_keyboard payload
# ---------------------------------------------------------------------------

def test_buttons_produce_inline_keyboard():
    btns = [
        {"label": "🛑 정지", "callback_data": "stop"},
        {"label": "⏸ 1h 스누즈", "callback_data": "snooze_1h"},
    ]
    result = format_alert(AlertLevel.CRITICAL, "제목", "본문", buttons=btns)
    markup = result["reply_markup"]
    assert markup is not None
    keyboard = markup["inline_keyboard"]
    assert len(keyboard) == 1          # single row
    assert len(keyboard[0]) == 2
    assert keyboard[0][0]["text"] == "🛑 정지"
    assert keyboard[0][0]["callback_data"] == "stop"
    assert keyboard[0][1]["callback_data"] == "snooze_1h"


def test_no_buttons_returns_none_markup():
    result = format_alert(AlertLevel.HIGH, "제목", "본문")
    assert result["reply_markup"] is None


# ---------------------------------------------------------------------------
# 5. should_send: True for first call, False on second within debounce
# ---------------------------------------------------------------------------

def test_should_send_debounce(tmp_path, monkeypatch):
    key = "test-debounce-pytest"
    state_file = Path(f"/tmp/epic-invest-tg-debounce-{key}.json")
    # Clean up before test
    state_file.unlink(missing_ok=True)

    # First call should return True (no prior state)
    first = should_send(AlertLevel.CRITICAL, key)
    assert first is True

    # Second call within the 60-second window should return False
    second = should_send(AlertLevel.CRITICAL, key)
    assert second is False

    # Cleanup
    state_file.unlink(missing_ok=True)


def test_should_send_after_window_expires(monkeypatch):
    key = "test-debounce-expired-pytest"
    state_file = Path(f"/tmp/epic-invest-tg-debounce-{key}.json")
    state_file.unlink(missing_ok=True)

    # Write an old timestamp (2 hours ago)
    old_ts = time.time() - 7200
    state_file.write_text(json.dumps({"last_sent": old_ts}))

    result = should_send(AlertLevel.CRITICAL, key)
    assert result is True

    state_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 6. in_quiet_hours: True at 23:00 KST, False at 09:00 KST
# ---------------------------------------------------------------------------

def test_in_quiet_hours_at_23():
    dt = datetime(2025, 1, 15, 23, 0, 0, tzinfo=KST)
    assert in_quiet_hours(dt) is True


def test_in_quiet_hours_at_09():
    dt = datetime(2025, 1, 15, 9, 0, 0, tzinfo=KST)
    assert in_quiet_hours(dt) is False


def test_in_quiet_hours_at_midnight():
    dt = datetime(2025, 1, 15, 0, 30, 0, tzinfo=KST)
    assert in_quiet_hours(dt) is True


def test_in_quiet_hours_boundary_08():
    # 08:00 is NOT quiet (strict < 8 check)
    dt = datetime(2025, 1, 15, 8, 0, 0, tzinfo=KST)
    assert in_quiet_hours(dt) is False
