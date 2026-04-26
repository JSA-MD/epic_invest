"""Unit tests for new telegram_bot slash commands: /help, /status, /snooze, /stop."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import telegram_bot


def _make_state() -> dict:
    return {"pending": {}, "runtime": telegram_bot.default_bot_runtime()}


class HelpCommandTest(unittest.TestCase):
    """Tests for /help command."""

    def test_help_text_contains_key_commands(self) -> None:
        text = telegram_bot.build_help_text()
        self.assertIn("/status", text)
        self.assertIn("/pnl", text)
        self.assertIn("/positions", text)
        self.assertIn("/safety", text)
        self.assertIn("/snooze", text)
        self.assertIn("/stop", text)
        self.assertIn("/start_pairwise", text)

    def test_handle_command_help_returns_help_text(self) -> None:
        state = _make_state()
        response = telegram_bot.handle_command(state, 123, "/help")
        self.assertIn("에픽 인베스트 텔레그램 명령어", response)

    def test_handle_command_start_returns_help_text(self) -> None:
        state = _make_state()
        response = telegram_bot.handle_command(state, 123, "/start")
        self.assertIn("에픽 인베스트 텔레그램 명령어", response)


class StatusCommandTest(unittest.TestCase):
    """Tests for /status with mock state file."""

    def _make_pairwise_state(self) -> dict:
        return {
            "runtime_health": {
                "pid": 42,
                "last_success_at": "2026-04-25T10:00:00+00:00",
            },
            "latest_runtime_snapshot": {
                "generated_at": "2026-04-25T10:00:01+00:00",
                "equity": 5000.0,
                "positions": [],
                "protections": [],
            },
            "latest_decision_snapshot": {},
            "latest_live_sync": {},
            "updated_at": "2026-04-25T10:00:01+00:00",
        }

    def test_status_renders_equity_from_snapshot(self) -> None:
        pairwise_state = self._make_pairwise_state()
        with (
            patch.object(
                telegram_bot,
                "load_runtime_profile",
                return_value={"active_trader": "pairwise", "mode": "demo"},
            ),
            patch.object(telegram_bot, "load_trader_state", return_value=pairwise_state),
            patch.object(telegram_bot, "trader_process_rows", return_value=[]),
            patch.object(telegram_bot, "resolve_default_leverage", return_value=2.0),
        ):
            snapshot = telegram_bot.get_runtime_snapshot()
        text = telegram_bot.format_status(snapshot)
        self.assertIn("5,000.00", text)
        self.assertIn("에픽 인베스트 상태", text)

    def test_handle_read_command_status_delegates_to_snapshot(self) -> None:
        dummy_snapshot: dict = {
            "trader_key": "pairwise",
            "mode": "demo",
            "leverage": 2.0,
            "processes": [],
            "state": {},
            "trader_running": True,
            "trader_pid": None,
            "trader_runtime": {},
            "bot_runtime": {},
            "snapshot_age_seconds": None,
            "snapshot_captured_at": None,
            "snapshot_ready": False,
            "decision": None,
            "equity": None,
            "positions": {},
            "protections": [],
            "exchange_error": None,
            "plan": None,
        }
        with patch.object(telegram_bot, "get_runtime_snapshot", return_value=dummy_snapshot):
            text = telegram_bot.handle_read_command("status")
        self.assertIn("에픽 인베스트 상태", text)


class SnoozeCommandTest(unittest.TestCase):
    """Tests for /snooze: writes file and snooze_active() reads it correctly."""

    def test_snooze_writes_json_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            snooze_path = Path(tf.name)

        original = telegram_bot.SNOOZE_STATE_PATH
        telegram_bot.SNOOZE_STATE_PATH = snooze_path
        try:
            until_dt = telegram_bot.write_snooze(1.0)
            data = json.loads(snooze_path.read_text())
            self.assertIn("snooze_until", data)
            self.assertAlmostEqual(data["hours"], 1.0)
            # snooze_until should be ~1h from now
            from datetime import datetime, timezone
            parsed = datetime.fromisoformat(data["snooze_until"])
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            diff = (parsed - telegram_bot.utc_now()).total_seconds()
            self.assertGreater(diff, 3500)
            self.assertLess(diff, 3700)
        finally:
            telegram_bot.SNOOZE_STATE_PATH = original
            snooze_path.unlink(missing_ok=True)

    def test_snooze_active_suppresses_high_level(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            snooze_path = Path(tf.name)

        original = telegram_bot.SNOOZE_STATE_PATH
        telegram_bot.SNOOZE_STATE_PATH = snooze_path
        try:
            telegram_bot.write_snooze(6.0)
            from telegram_format import AlertLevel
            self.assertTrue(telegram_bot.snooze_active(AlertLevel.HIGH))
            self.assertFalse(telegram_bot.snooze_active(AlertLevel.CRITICAL))
        finally:
            telegram_bot.SNOOZE_STATE_PATH = original
            snooze_path.unlink(missing_ok=True)

    def test_snooze_expired_returns_false(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            snooze_path = Path(tf.name)

        original = telegram_bot.SNOOZE_STATE_PATH
        telegram_bot.SNOOZE_STATE_PATH = snooze_path
        try:
            past = telegram_bot.utc_now() - timedelta(hours=1)
            snooze_path.write_text(json.dumps({"snooze_until": past.isoformat()}))
            from telegram_format import AlertLevel
            self.assertFalse(telegram_bot.snooze_active(AlertLevel.HIGH))
        finally:
            telegram_bot.SNOOZE_STATE_PATH = original
            snooze_path.unlink(missing_ok=True)

    def test_handle_command_snooze_writes_file_and_returns_success(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            snooze_path = Path(tf.name)

        original = telegram_bot.SNOOZE_STATE_PATH
        telegram_bot.SNOOZE_STATE_PATH = snooze_path
        try:
            state = _make_state()
            response = telegram_bot.handle_command(state, 123, "/snooze 6h")
            self.assertIn("6h", response)
            data = json.loads(snooze_path.read_text())
            self.assertAlmostEqual(data["hours"], 6.0)
        finally:
            telegram_bot.SNOOZE_STATE_PATH = original
            snooze_path.unlink(missing_ok=True)


class StopConfirmFlowTest(unittest.TestCase):
    """Tests for /stop 2-step confirm flow."""

    def test_stop_command_does_not_call_execute_stop(self) -> None:
        """The /stop text command must NOT immediately stop the service."""
        with patch.object(telegram_bot, "execute_stop_pairwise") as mock_stop:
            state = _make_state()
            # /stop returns the sentinel; the real send is done by process_update
            response = telegram_bot.handle_command(state, 123, "/stop")
            self.assertEqual(response, "__stop_confirm_prompt__")
            mock_stop.assert_not_called()

    def test_process_update_stop_sends_inline_keyboard(self) -> None:
        """process_update for /stop should call send_stop_confirm_prompt, not send_message."""
        update = {"message": {"chat": {"id": 99999}, "text": "/stop"}}
        state = _make_state()
        with (
            patch.object(telegram_bot, "TELEGRAM_ALLOWED_CHAT_IDS", {99999}),
            patch.object(telegram_bot, "send_stop_confirm_prompt") as mock_prompt,
            patch.object(telegram_bot, "execute_stop_pairwise") as mock_stop,
            patch.object(telegram_bot, "save_bot_state"),
        ):
            telegram_bot.process_update(state, update)
            mock_prompt.assert_called_once_with(99999)
            mock_stop.assert_not_called()

    def test_callback_stop_confirm_calls_execute_stop(self) -> None:
        """callback_query with stop_confirm should call execute_stop_pairwise."""
        callback_query = {
            "id": "cq123",
            "from": {"id": 99999},
            "message": {"chat": {"id": 99999}},
            "data": "stop_confirm",
        }
        state = _make_state()
        with (
            patch.object(telegram_bot, "TELEGRAM_ALLOWED_CHAT_IDS", {99999}),
            patch.object(telegram_bot, "execute_stop_pairwise", return_value="정지 완료") as mock_stop,
            patch.object(telegram_bot, "answer_callback_query"),
            patch.object(telegram_bot, "send_message"),
        ):
            telegram_bot.process_callback_query(state, callback_query)
            mock_stop.assert_called_once()

    def test_callback_stop_cancel_does_not_call_execute_stop(self) -> None:
        """callback_query with stop_cancel must NOT call execute_stop_pairwise."""
        callback_query = {
            "id": "cq456",
            "from": {"id": 99999},
            "message": {"chat": {"id": 99999}},
            "data": "stop_cancel",
        }
        state = _make_state()
        with (
            patch.object(telegram_bot, "TELEGRAM_ALLOWED_CHAT_IDS", {99999}),
            patch.object(telegram_bot, "execute_stop_pairwise") as mock_stop,
            patch.object(telegram_bot, "answer_callback_query"),
            patch.object(telegram_bot, "send_message"),
        ):
            telegram_bot.process_callback_query(state, callback_query)
            mock_stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
