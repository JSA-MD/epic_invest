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
        self.assertEqual(report["active_profile"], "core")

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

    def test_evaluate_trader_pairwise_threshold_allows_next_cycle(self) -> None:
        state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {
                "pid": 10723,
                "last_success_at": "2026-04-10T13:22:32+00:00",
                "last_loop_started_at": "2026-04-10T13:22:20+00:00",
                "last_loop_completed_at": "2026-04-10T13:22:32+00:00",
                "consecutive_errors": 0,
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "mode": "demo",
            "force_execute": True,
            "stale_threshold_seconds": 390,
            "protect_threshold_seconds": 480,
        }
        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", return_value=state),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=200.0),
        ):
            report = watchdog.evaluate_trader()
        self.assertEqual(report["active_profile"], "pairwise")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["reasons"], [])

    def test_resolve_telegram_chat_ids_dedupes_duplicates(self) -> None:
        with (
            patch.dict(
                watchdog.os.environ,
                {
                    "TELEGRAM_ALLOWED_CHAT_IDS": "8214325134,8214325134",
                    "TELEGRAM_CHAT_ID": "8214325134",
                },
                clear=False,
            ),
        ):
            self.assertEqual(watchdog.resolve_telegram_chat_ids(), ["8214325134"])

    def test_build_alert_fingerprint_ignores_generated_at(self) -> None:
        report_a = {
            "generated_at": "2026-04-10T13:25:49+00:00",
            "trader": {"active_profile": "pairwise", "status": "critical", "reasons": ["state_stale"]},
            "bot": {"status": "ok", "reasons": []},
            "recovery_actions": [{"type": "restart_trader"}],
        }
        report_b = {
            "generated_at": "2026-04-10T13:26:49+00:00",
            "trader": {"active_profile": "pairwise", "status": "critical", "reasons": ["state_stale"]},
            "bot": {"status": "ok", "reasons": []},
            "recovery_actions": [{"type": "restart_trader"}],
        }
        self.assertEqual(
            watchdog.build_alert_fingerprint(report_a),
            watchdog.build_alert_fingerprint(report_b),
        )

    def test_degrade_pairwise_force_execute_writes_runtime_profile(self) -> None:
        with patch.object(watchdog, "write_json") as write_json:
            result = watchdog.degrade_pairwise_force_execute({"mode": "demo"}, "watchdog_critical_recovery")
        self.assertTrue(result["ok"])
        written_payload = write_json.call_args.args[1]
        self.assertEqual(written_payload["active_trader"], "pairwise")
        self.assertFalse(written_payload["force_execute"])
        self.assertEqual(written_payload["degraded_reason"], "watchdog_critical_recovery")

    def test_maybe_recover_demotes_pairwise_force_execute_before_restart(self) -> None:
        report = {
            "trader": {
                "status": "critical",
                "stale_seconds": 500.0,
                "consecutive_errors": 0,
            },
            "bot": {"status": "ok"},
            "recovery_actions": [],
        }
        profile = {
            "key": "pairwise",
            "mode": "demo",
            "force_execute": True,
            "protect_threshold_seconds": 480,
        }
        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "protect_positions", return_value={"ok": True}) as protect_positions,
            patch.object(watchdog, "degrade_pairwise_force_execute", return_value={"ok": True}) as degrade,
            patch.object(watchdog, "restart_active_trader", return_value={"ok": True}) as restart,
        ):
            result = watchdog.maybe_recover(report)
        self.assertEqual(result["recovery_actions"][0]["type"], "protect_positions")
        self.assertEqual(result["recovery_actions"][1]["type"], "degrade_pairwise_force_execute")
        self.assertEqual(result["recovery_actions"][2]["type"], "restart_trader")
        protect_positions.assert_called_once()
        degrade.assert_called_once()
        restarted_profile = restart.call_args.args[0]
        self.assertFalse(restarted_profile["force_execute"])


if __name__ == "__main__":
    unittest.main()
