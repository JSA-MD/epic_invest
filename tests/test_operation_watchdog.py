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
            "data_quality": {"status": "warning"},
            "recovery_actions": [],
        }
        with patch.object(watchdog, "send_telegram_notification") as send_notification:
            watchdog.maybe_send_alert(report)
        send_notification.assert_not_called()

    def test_build_report_includes_data_quality(self) -> None:
        with (
            patch.object(watchdog, "evaluate_trader", return_value={"status": "ok"}),
            patch.object(watchdog, "evaluate_bot", return_value={"status": "ok"}),
            patch.object(watchdog, "safe_build_data_quality_snapshot", return_value={"status": "warning"}),
            patch.object(watchdog, "safe_build_decision_quality_snapshot", return_value={"status": "warning"}),
        ):
            report = watchdog.build_report()
        self.assertEqual(report["data_quality"]["status"], "warning")
        self.assertEqual(report["decision_quality"]["status"], "warning")

    def test_is_pid_running_treats_permission_error_as_alive(self) -> None:
        with patch.object(watchdog.os, "kill", side_effect=PermissionError("denied")):
            self.assertTrue(watchdog.is_pid_running(12345))

    def test_resolve_live_pid_prefers_running_fallback(self) -> None:
        with patch.object(watchdog, "is_pid_running", side_effect=lambda pid: pid == 222):
            self.assertEqual(watchdog.resolve_live_pid(111, 222), 222)

    def test_launchd_service_pid_parses_launchctl_output(self) -> None:
        with patch.object(
            watchdog,
            "launchctl_print",
            return_value={"returncode": 0, "stdout": "com.epicinvest.pairwise-trader\n    pid = 4242\n", "stderr": ""},
        ):
            self.assertEqual(watchdog.launchd_service_pid(watchdog.PAIRWISE_LABEL), 4242)

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
            patch.object(watchdog, "launchd_service_pid", return_value=None),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()
        self.assertEqual(report["pid"], 222)
        self.assertTrue(report["pid_verified"])
        self.assertEqual(report["reasons"], [])
        self.assertEqual(report["active_profile"], "core")

    def test_evaluate_trader_uses_pairwise_launchd_fallback_pid(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T11:58:03+00:00",
            "runtime_health": {
                "pid": 111,
                "last_success_at": "2026-04-10T11:58:03+00:00",
                "last_loop_started_at": "2026-04-10T11:57:03+00:00",
                "last_loop_completed_at": "2026-04-10T11:58:03+00:00",
                "consecutive_errors": 0,
            },
            "promotion_gate": {"shadow_required": False},
        }
        shadow_state = {}
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "launchd_label": watchdog.PAIRWISE_LABEL,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
            "mode": "demo",
            "force_execute": False,
            "stale_threshold_seconds": 390,
            "protect_threshold_seconds": 480,
        }

        def fake_read_json(path: Path, default):
            if path == watchdog.PAIRWISE_SHADOW_STATE_PATH:
                return shadow_state
            return live_state

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=None),
            patch.object(watchdog, "launchd_service_pid", return_value=4242),
            patch.object(watchdog, "resolve_live_pid", return_value=4242),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["pid"], 4242)
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
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
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
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
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
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
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
                return 1500.0
            return 5.0

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", side_effect=fake_age_seconds),
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "critical")
        self.assertIn("shadow_signal_stale", report["reasons"])

    def test_evaluate_trader_allows_pairwise_signal_within_execution_stale_cap(self) -> None:
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
                "last_signal_timestamp": "2026-04-10T13:14:30+00:00",
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
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
            if value == "2026-04-10T13:14:30+00:00":
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

        self.assertEqual(report["status"], "ok")
        self.assertNotIn("shadow_signal_stale", report["reasons"])

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
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
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

    def test_evaluate_trader_warns_on_stale_decision_log(self) -> None:
        state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {
                "pid": 10723,
                "last_success_at": "2026-04-10T13:22:32+00:00",
                "consecutive_errors": 0,
            },
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
            "promotion_gate": {"ready_for_live": True, "ready_for_merge": True},
        }
        shadow_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"last_success_at": "2026-04-10T13:22:32+00:00"},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
            "mode": "demo",
            "force_execute": False,
            "stale_threshold_seconds": 390,
            "protect_threshold_seconds": 480,
        }

        def fake_read_json(path: Path, default):
            if path == watchdog.PAIRWISE_SHADOW_STATE_PATH:
                return shadow_state
            return state

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
            patch.object(watchdog, "file_age_seconds", return_value=900.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "warning")
        self.assertIn("decision_log_stale", report["reasons"])

    def test_evaluate_trader_skips_shadow_requirements_when_shadow_is_disabled(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"pid": 10723, "last_success_at": "2026-04-10T13:22:32+00:00", "consecutive_errors": 0},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
            "promotion_gate": {
                "ready_for_shadow_live": True,
                "ready_for_live": True,
                "ready_for_merge": True,
                "shadow_required": False,
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
            "mode": "live",
            "force_execute": False,
            "stale_threshold_seconds": 390,
            "protect_threshold_seconds": 480,
        }

        def fake_read_json(path: Path, default):
            if path == watchdog.PAIRWISE_SHADOW_STATE_PATH:
                return {}
            return live_state

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "ok")
        self.assertNotIn("shadow_state_missing", report["reasons"])
        self.assertNotIn("shadow_signal_missing", report["reasons"])

    def test_evaluate_trader_blocks_live_shadow_signal_divergence(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"pid": 10723, "last_success_at": "2026-04-10T13:22:32+00:00", "consecutive_errors": 0},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
            "promotion_gate": {"ready_for_live": True, "ready_for_merge": True},
        }
        shadow_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"last_success_at": "2026-04-10T13:22:32+00:00"},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "rationale": {"signal_timestamp": "2026-04-10T13:10:00+00:00"},
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
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
            if value == "2026-04-10T13:10:00+00:00":
                return 200.0
            return 5.0

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", side_effect=fake_age_seconds),
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "critical")
        self.assertIn("live_shadow_signal_divergence", report["reasons"])

    def test_evaluate_trader_blocks_live_shadow_weight_divergence(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"pid": 10723, "last_success_at": "2026-04-10T13:22:32+00:00", "consecutive_errors": 0},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "pair_plans": {"BNBUSDT": {"requested_weight": -1.5}},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
            "promotion_gate": {"ready_for_shadow_live": True},
        }
        shadow_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"last_success_at": "2026-04-10T13:22:32+00:00"},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
                "pair_plans": {"BNBUSDT": {"requested_weight": 0.0}},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
            "mode": "demo",
            "force_execute": False,
            "stale_threshold_seconds": 390,
            "protect_threshold_seconds": 480,
        }

        def fake_read_json(path: Path, default):
            if path == watchdog.PAIRWISE_SHADOW_STATE_PATH:
                return shadow_state
            return live_state

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "critical")
        self.assertIn("live_shadow_target_divergence", report["reasons"])

    def test_evaluate_trader_ignores_target_divergence_when_requested_weights_match(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"pid": 10723, "last_success_at": "2026-04-10T13:22:32+00:00", "consecutive_errors": 0},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "pair_plans": {"BNBUSDT": {"requested_weight": 0.0}},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
            "promotion_gate": {"ready_for_shadow_live": True},
        }
        shadow_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"last_success_at": "2026-04-10T13:22:32+00:00"},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
                "pair_plans": {"BNBUSDT": {"requested_weight": 0.0}},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
        }
        profile = {
            "key": "pairwise",
            "state_path": watchdog.PAIRWISE_STATE_PATH,
            "shadow_state_path": watchdog.PAIRWISE_SHADOW_STATE_PATH,
            "pid_path": watchdog.PAIRWISE_PID_PATH,
            "log_path": watchdog.PAIRWISE_LOG_PATH,
            "decision_log_path": watchdog.PAIRWISE_DECISION_LOG_PATH,
            "mode": "demo",
            "force_execute": False,
            "stale_threshold_seconds": 390,
            "protect_threshold_seconds": 480,
        }

        def fake_read_json(path: Path, default):
            if path == watchdog.PAIRWISE_SHADOW_STATE_PATH:
                return shadow_state
            return live_state

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "ok")
        self.assertNotIn("live_shadow_target_divergence", report["reasons"])

    def test_evaluate_trader_blocks_missing_shutdown_protection(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"pid": 10723, "last_success_at": "2026-04-10T13:22:32+00:00", "consecutive_errors": 0},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
            "latest_runtime_snapshot": {
                "plan": {"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5}},
                "positions": [{"pair": "BNBUSDT", "qty": -11.5}],
                "extra": {
                    "execution": {
                        "enabled": True,
                        "shutdown_protection": {
                            "status": "placed",
                            "positions": [{"pair": "BNBUSDT", "qty": -11.5}],
                            "protections": [{"orders": [{"status": "placed"}]}],
                        },
                    }
                },
            },
            "promotion_gate": {"ready_for_shadow_live": True},
        }
        shadow_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"last_success_at": "2026-04-10T13:22:32+00:00"},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
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

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "critical")
        self.assertIn("shutdown_protection_missing", report["reasons"])

    def test_evaluate_trader_blocks_when_execution_is_blocked_despite_open_gate(self) -> None:
        live_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"pid": 10723, "last_success_at": "2026-04-10T13:22:32+00:00", "consecutive_errors": 0},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
            },
            "latest_runtime_snapshot": {
                "plan": {"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5}},
                "extra": {"execution": {"enabled": False, "blocked": True}},
            },
            "promotion_gate": {"ready_for_live": True, "ready_for_merge": True},
        }
        shadow_state = {
            "updated_at": "2026-04-10T13:22:32+00:00",
            "runtime_health": {"last_success_at": "2026-04-10T13:22:32+00:00"},
            "latest_decision_snapshot": {
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "rationale": {"signal_timestamp": "2026-04-10T13:22:32+00:00"},
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

        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", side_effect=fake_read_json),
            patch.object(watchdog, "read_pid", return_value=10723),
            patch.object(watchdog, "resolve_live_pid", return_value=10723),
            patch.object(watchdog, "is_pid_running", return_value=True),
            patch.object(watchdog, "age_seconds", return_value=5.0),
            patch.object(watchdog, "file_age_seconds", return_value=5.0),
        ):
            report = watchdog.evaluate_trader()

        self.assertEqual(report["status"], "critical")
        self.assertIn("execution_blocked_with_open_gate", report["reasons"])

    def test_restart_active_trader_pairwise_restarts_live_only(self) -> None:
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
        self.assertEqual(calls[1].args[0], [str(watchdog.PAIRWISE_SERVICE_SCRIPT), "start"])
        self.assertEqual(calls[1].kwargs["env_updates"]["PAIRWISE_FORCE_EXECUTE"], "0")

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

    def test_build_decision_quality_highlights_stale_bnb_lob(self) -> None:
        state = {
            "latest_runtime_snapshot": {
                "plan": {
                    "pair_plans": {
                        "BNBUSDT": {
                            "equity_corr_source_mode": "market_context",
                        }
                    }
                },
                "extra": {
                    "execution": {
                        "shutdown_protection": {
                            "positions": [
                                {"pair": "BNBUSDT", "qty": -11.4},
                            ]
                        }
                    }
                },
            },
            "latest_decision_snapshot": {
                "target_weights": {
                    "BNBUSDT": -1.5,
                    "BTCUSDT": 0.0,
                }
            },
        }
        data_quality = {
            "market_context": {"status": "warning"},
            "lob": {
                "per_pair": {
                    "BNBUSDT": {
                        "features": {"freshness": "stale"},
                        "snapshots": {"freshness": "stale"},
                        "agg_trades": {"freshness": "stale"},
                    }
                }
            },
            "derivatives": {
                "per_pair": {
                    "BNBUSDT": {
                        "open_interest": {"freshness": "stale"},
                        "basis_perpetual": {"freshness": "stale"},
                    }
                }
            },
            "ohlcv": {
                "per_pair": {
                    "BNBUSDT": {
                        "1d": {"freshness": "missing"},
                    }
                }
            },
        }
        profile = {"state_path": watchdog.PAIRWISE_STATE_PATH}
        with (
            patch.object(watchdog, "active_trader_profile", return_value=profile),
            patch.object(watchdog, "read_json", return_value=state),
        ):
            snapshot = watchdog.build_decision_quality_snapshot({"status": "ok"}, data_quality)
        self.assertEqual(snapshot["status"], "warning")
        self.assertTrue(any(item.get("pair") == "BNBUSDT" for item in snapshot["upgrade_risks"]))
        self.assertTrue(any("BNBUSDT" in insight for insight in snapshot["insights"]))

    def test_write_report_persists_data_quality_and_decision_quality(self) -> None:
        report = {
            "generated_at": "2026-04-13T05:00:00+00:00",
            "trader": {"status": "ok", "stale_seconds": 1.0, "consecutive_errors": 0},
            "bot": {"status": "ok", "stale_seconds": 1.0, "consecutive_poll_errors": 0},
            "data_quality": {"status": "critical"},
            "decision_quality": {"status": "warning"},
            "recovery_actions": [],
        }
        with (
            patch.object(watchdog, "write_json") as write_json,
            patch.object(watchdog, "append_jsonl") as append_jsonl,
        ):
            watchdog.write_report(report)
        written_paths = [call.args[0] for call in write_json.call_args_list]
        self.assertIn(watchdog.WATCHDOG_REPORT_PATH, written_paths)
        self.assertIn(watchdog.DATA_QUALITY_REPORT_PATH, written_paths)
        self.assertIn(watchdog.DECISION_QUALITY_REPORT_PATH, written_paths)
        history_payload = append_jsonl.call_args.args[1]
        self.assertEqual(history_payload["data_quality"]["status"], "critical")
        self.assertEqual(history_payload["decision_quality"]["status"], "warning")


if __name__ == "__main__":
    unittest.main()
