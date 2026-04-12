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
    def test_maybe_send_alert_is_suppressed_by_default(self) -> None:
        report = {
            "generated_at": "2026-04-12T12:00:00+00:00",
            "trader": {
                "active_profile": "pairwise",
                "status": "critical",
                "stale_seconds": 400.0,
                "consecutive_errors": 0,
                "reasons": ["shadow_signal_stale"],
            },
            "bot": {
                "status": "ok",
                "stale_seconds": 5.0,
                "consecutive_poll_errors": 0,
                "reasons": [],
            },
            "recovery_actions": [],
        }
        with patch.object(watchdog, "send_telegram_notification") as send_notification:
            watchdog.maybe_send_alert(report)
        send_notification.assert_not_called()

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
            "shadow_paper": {
                "last_signal_timestamp": "2026-04-10T13:22:32+00:00",
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
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

    def test_active_pairwise_profile_ignores_stale_runtime_force_without_env_opt_in(self) -> None:
        runtime_profile = {
            "active_trader": "pairwise",
            "mode": "demo",
            "force_execute": True,
        }
        with (
            patch.object(watchdog, "read_runtime_profile", return_value=runtime_profile),
            patch.dict(watchdog.os.environ, {"PAIRWISE_FORCE_EXECUTE": "0"}, clear=False),
        ):
            profile = watchdog.active_trader_profile()
        self.assertTrue(profile["runtime_force_execute_requested"])
        self.assertFalse(profile["force_execute"])

    def test_evaluate_trader_blocks_stale_shadow_signal(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {
                "pid": 10723,
                "last_success_at": "2026-04-10T13:22:32+00:00",
                "consecutive_errors": 0,
            },
        }
        shadow_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {
                "last_success_at": "2026-04-10T13:22:32+00:00",
            },
            "shadow_paper": {
                "last_signal_timestamp": "2026-04-10T13:00:00+00:00",
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "mode": "demo",
            "force_execute": False,
            "stale_threshold_seconds": 390,
            "protect_threshold_seconds": 480,
        }

        def fake_read_json(path: Path, default):
            if path == watchdog.PAIRWISE_SHADOW_STATE_PATH:
                return shadow_state
            return live_state

        def fake_age_seconds(value):
            if value == "2026-04-10T13:00:00+00:00":
                return 500.0
            return 5.0

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", side_effect=fake_age_seconds),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "critical")
        self.assertIn("shadow_signal_stale", report["reasons"])

    def test_evaluate_trader_blocks_stale_shadow_state(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {
                "pid": 10723,
                "last_success_at": "2026-04-10T13:22:32+00:00",
                "consecutive_errors": 0,
            },
        }
        shadow_state = {
            "updated_at": "2026-04-10T13:00:00+00:00",
            "runtime_health": {
                "last_success_at": "2026-04-10T13:00:00+00:00",
            },
            "shadow_paper": {
                "last_signal_timestamp": "2026-04-10T13:22:32+00:00",
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "mode": "demo",
            "force_execute": False,
            "stale_threshold_seconds": 390,
            "protect_threshold_seconds": 480,
        }

        def fake_read_json(path: Path, default):
            if path == watchdog.PAIRWISE_SHADOW_STATE_PATH:
                return shadow_state
            return live_state

        def fake_age_seconds(value):
            if value == "2026-04-10T13:00:00+00:00":
                return 500.0
            return 5.0

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", side_effect=fake_age_seconds),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "critical")
        self.assertIn("shadow_state_stale", report["reasons"])

    def test_restart_active_trader_pairwise_restarts_shadow_before_live(self) -> None:
        profile = {
            "key": "pairwise",
            "mode": "demo",
            "force_execute": False,
        }
        with patch.object(watchdog, "run_command", return_value={"returncode": 0}) as run_command:
            result = watchdog.restart_active_trader(profile)
        self.assertTrue(result["ok"])
        calls = run_command.call_args_list
        self.assertEqual(calls[0].args[0], [str(watchdog.PAIRWISE_SERVICE_SCRIPT), "stop"])
        self.assertEqual(calls[1].args[0], [str(watchdog.PAIRWISE_SHADOW_UNLOAD_SCRIPT)])
        self.assertEqual(calls[2].args[0], [str(watchdog.PAIRWISE_SHADOW_LOAD_SCRIPT)])
        self.assertEqual(calls[3].args[0], [str(watchdog.PAIRWISE_SERVICE_SCRIPT), "start"])
        self.assertEqual(calls[3].kwargs["env_updates"]["PAIRWISE_FORCE_EXECUTE"], "0")

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
