import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import telegram_bot


class TelegramBotProcessTests(unittest.TestCase):
    def test_is_pid_running_treats_permission_error_as_alive(self) -> None:
        with patch.object(telegram_bot.os, "kill", side_effect=PermissionError("denied")):
            self.assertTrue(telegram_bot.is_pid_running(12345))

    def test_is_trader_running_falls_back_to_pid_file(self) -> None:
        def fake_is_pid_running(pid: int | None) -> bool:
            return pid == 222

        with (
            patch.object(telegram_bot, "trader_runtime_pid", return_value=111),
            patch.object(telegram_bot, "read_pid", return_value=222),
            patch.object(telegram_bot, "is_pid_running", side_effect=fake_is_pid_running),
            patch.object(telegram_bot, "trader_process_rows", return_value=[]),
            patch.object(telegram_bot, "live_state_fresh", return_value=False),
        ):
            self.assertTrue(telegram_bot.is_trader_running())


if __name__ == "__main__":
    unittest.main()
