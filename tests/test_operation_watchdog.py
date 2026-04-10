import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import operation_watchdog as watchdog


class OperationWatchdogTests(unittest.TestCase):
    def test_is_pid_running_treats_permission_error_as_alive(self) -> None:
        with patch.object(watchdog.os, "kill", side_effect=PermissionError("denied")):
            self.assertTrue(watchdog.is_pid_running(12345))

    def test_resolve_live_pid_prefers_running_fallback(self) -> None:
        with patch.object(watchdog, "is_pid_running", side_effect=lambda pid: pid == 222):
            self.assertEqual(watchdog.resolve_live_pid(111, 222), 222)

    def test_evaluate_trader_uses_live_fallback_pid(self) -> None:
        state = {
            "updated_at": "2026-04-10T11:58:03+00:00",
            "runtime_health": {
                "pid": 111,
                "last_success_at": "2026-04-10T11:58:03+00:00",
                "last_loop_started_at": "2026-04-10T11:57:03+00:00",
                "last_loop_completed_at": "2026-04-10T11:58:03+00:00",
                "consecutive_errors": 0,
            },
        }
        with (
            patch.object(watchdog, "read_json", return_value=state),
            patch.object(watchdog, "read_pid", return_value=222),
            patch.object(watchdog, "resolve_live_pid", return_value=222),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()
        self.assertEqual(report["pid"], 222)
        self.assertTrue(report["pid_verified"])
        self.assertEqual(report["reasons"], [])

    def test_evaluate_bot_uses_live_fallback_pid(self) -> None:
        state = {
            "runtime": {
                "pid": 111,
                "last_started_at": "2026-04-10T10:23:23+00:00",
                "last_poll_started_at": "2026-04-10T11:58:55+00:00",
                "last_poll_ok_at": "2026-04-10T11:58:56+00:00",
                "last_reply_at": "2026-04-10T10:24:20+00:00",
                "consecutive_poll_errors": 0,
            }
        }
        with (
            patch.object(watchdog, "read_json", return_value=state),
            patch.object(watchdog, "read_pid", return_value=333),
            patch.object(watchdog, "resolve_live_pid", return_value=333),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=3.0),
        ):
            report = watchdog.evaluate_bot()
        self.assertEqual(report["pid"], 333)
        self.assertTrue(report["pid_verified"])
        self.assertEqual(report["reasons"], [])


if __name__ == "__main__":
    unittest.main()
