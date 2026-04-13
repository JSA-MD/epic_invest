import sys
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import telegram_bot


class TelegramBotProcessTests(unittest.TestCase):
    def make_update(self, text: str, chat_id: int = 123456) -> dict:
        return {"message": {"chat": {"id": chat_id}, "text": text}}

    def make_state(self) -> dict:
        return {"pending": {}, "runtime": telegram_bot.default_bot_runtime()}

    def immediate_submit(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        fn(*args, **kwargs)
        return object()

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

    def test_active_trader_context_uses_pairwise_profile(self) -> None:
        with patch.object(
            telegram_bot,
            "load_runtime_profile",
            return_value={"active_trader": "pairwise", "mode": "demo"},
        ):
            context = telegram_bot.active_trader_context()
        self.assertEqual(context["key"], "pairwise")
        self.assertEqual(context["state_path"], telegram_bot.PAIRWISE_STATE_PATH)
        self.assertEqual(context["script_path"], telegram_bot.PAIRWISE_TRADER_SCRIPT)

    def test_get_runtime_snapshot_reads_pairwise_positions_and_protections(self) -> None:
        pairwise_state = {
            "runtime_health": {
                "pid": 123,
                "last_success_at": "2026-04-13T02:30:21+00:00",
            },
            "latest_decision_snapshot": {
                "strategy_class": "pairwise_regime_live",
                "session_type": "pairwise",
                "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                "pair_plans": {
                    "BNBUSDT": {
                        "route_state_name": "equity_aligned:bear_narrow",
                        "signal_value": -100.0,
                    }
                },
            },
            "latest_runtime_snapshot": {
                "generated_at": "2026-04-13T02:30:22+00:00",
                "equity": 4500.0,
                "positions": [
                    {
                        "pair": "BNBUSDT",
                        "qty": -11.29,
                        "side": "SHORT",
                        "entry_price": 592.67,
                        "mark_price": 598.13,
                        "margin_mode": "isolated",
                    }
                ],
                "plan": {
                    "session_type": "pairwise",
                    "signal_timestamp": "2026-04-13T02:25:00+00:00",
                    "gross_leverage": 1.5,
                    "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
                },
                "extra": {
                    "execution": {
                        "shutdown_protection": {
                            "protections": [
                                {
                                    "pair": "BNBUSDT",
                                    "status": "retained",
                                    "stop_price": 601.12,
                                    "take_price": 585.0,
                                }
                            ]
                        }
                    }
                },
            },
        }
        with (
            patch.object(telegram_bot, "load_bot_state", return_value={"runtime": telegram_bot.default_bot_runtime()}),
            patch.object(telegram_bot, "load_runtime_profile", return_value={"active_trader": "pairwise", "mode": "demo"}),
            patch.object(telegram_bot, "load_trader_state", return_value=pairwise_state),
            patch.object(telegram_bot, "trader_process_rows", return_value=[]),
            patch.object(telegram_bot, "read_pid", return_value=123),
            patch.object(telegram_bot, "is_pid_running", return_value=True),
        ):
            snapshot = telegram_bot.get_runtime_snapshot()
        self.assertEqual(snapshot["trader_key"], "pairwise")
        self.assertTrue(snapshot["snapshot_ready"])
        self.assertIn("BNBUSDT", snapshot["positions"])
        self.assertEqual(snapshot["protections"][0]["pair"], "BNBUSDT")

    def test_handle_command_routes_all_supported_read_commands(self) -> None:
        state = self.make_state()
        commands = [
            "start",
            "help",
            "ping",
            "status",
            "plan",
            "positions",
            "why",
            "rationale",
            "reason",
            "exitplan",
            "protection",
            "killswitch",
            "logs",
            "recent",
        ]
        with patch.object(telegram_bot, "handle_read_command", side_effect=lambda cmd: f"response:{cmd}"):
            for command in commands:
                with self.subTest(command=command):
                    response = telegram_bot.handle_command(state, 11, f"/{command}")
                    self.assertEqual(response, f"response:{command}")

    def test_handle_read_command_positions_returns_pairwise_position(self) -> None:
        snapshot = {
            "snapshot_ready": True,
            "snapshot_captured_at": "2026-04-13T02:30:22+00:00",
            "positions": {
                "BNBUSDT": {
                    "pair": "BNBUSDT",
                    "qty": -11.29,
                    "side": "SHORT",
                    "entry_price": 592.67,
                    "mark_price": 598.13,
                    "margin_mode": "isolated",
                    "percentage": -5.11,
                }
            },
        }
        with patch.object(telegram_bot, "get_runtime_snapshot", return_value=snapshot):
            message = telegram_bot.handle_read_command("positions")
        self.assertIn("BNBUSDT", message)
        self.assertIn("손익률 -5.11%", message)
        self.assertNotIn("포지션 캐시를 준비 중입니다", message)

    def test_get_runtime_snapshot_prefers_live_sync_position_fields(self) -> None:
        pairwise_state = {
            "runtime_health": {"pid": 123, "last_success_at": "2026-04-13T02:30:21+00:00"},
            "latest_runtime_snapshot": {
                "generated_at": "2026-04-13T02:30:22+00:00",
                "positions": [
                    {
                        "pair": "BNBUSDT",
                        "qty": -11.29,
                        "side": "SHORT",
                        "entry_price": 592.67,
                        "mark_price": 598.13,
                        "margin_mode": "isolated",
                    }
                ],
            },
            "latest_live_sync": {
                "positions": {
                    "BNBUSDT": {
                        "pair": "BNBUSDT",
                        "qty": -11.29,
                        "side": "SHORT",
                        "entry_price": 592.67,
                        "mark_price": 598.13,
                        "margin_mode": "isolated",
                        "percentage": -5.11,
                    }
                },
                "protection_orders": [
                    {
                        "symbol": "BNB/USDT:USDT",
                        "status": "open",
                        "type": "market",
                        "side": "buy",
                        "id": "1",
                        "clientOrderId": "x",
                        "stopPrice": 601.12,
                    }
                ],
            },
        }
        with (
            patch.object(telegram_bot, "load_bot_state", return_value={"runtime": telegram_bot.default_bot_runtime()}),
            patch.object(telegram_bot, "load_runtime_profile", return_value={"active_trader": "pairwise", "mode": "demo"}),
            patch.object(telegram_bot, "load_trader_state", return_value=pairwise_state),
            patch.object(telegram_bot, "trader_process_rows", return_value=[]),
            patch.object(telegram_bot, "read_pid", return_value=123),
            patch.object(telegram_bot, "is_pid_running", return_value=True),
        ):
            snapshot = telegram_bot.get_runtime_snapshot()
        self.assertEqual(snapshot["positions"]["BNBUSDT"]["percentage"], -5.11)
        self.assertEqual(len(snapshot["protections"]), 1)
        self.assertEqual(snapshot["protections"][0]["stop_price"], 601.12)

    def test_control_commands_queue_confirmation_for_all_supported_actions(self) -> None:
        state = self.make_state()
        controls = [
            "starttrader",
            "stoptrader",
            "restarttrader",
            "sync",
            "protect",
            "closeall",
            "flatten",
        ]
        with (
            patch.object(telegram_bot, "save_bot_state"),
            patch.object(telegram_bot.secrets, "token_hex", return_value="abc123"),
            patch.object(telegram_bot, "audit"),
        ):
            for command in controls:
                state["pending"].clear()
                with self.subTest(command=command):
                    response = telegram_bot.handle_command(state, 77, f"/{command}")
                    self.assertIn("/confirm abc123", response)
                    self.assertIn("77", state["pending"])
                    self.assertEqual(state["pending"]["77"]["action"], command)

    def test_confirm_executes_pending_action(self) -> None:
        state = self.make_state()
        state["pending"]["77"] = {
            "action": "sync",
            "summary": "상태 동기화",
            "token": "abc123",
            "expires_at": (telegram_bot.utc_now() + timedelta(seconds=60)).isoformat(),
        }
        with (
            patch.object(telegram_bot, "save_bot_state"),
            patch.object(telegram_bot, "audit"),
            patch.object(telegram_bot, "execute_control_action", return_value="done") as mock_execute,
        ):
            response = telegram_bot.handle_command(state, 77, "/confirm abc123")
        self.assertEqual(response, "done")
        mock_execute.assert_called_once_with("sync")
        self.assertNotIn("77", state["pending"])

    def test_cancel_clears_pending_action(self) -> None:
        state = self.make_state()
        state["pending"]["77"] = {"action": "sync"}
        with patch.object(telegram_bot, "save_bot_state"):
            response = telegram_bot.handle_command(state, 77, "/cancel")
        self.assertEqual(response, "대기 중인 제어 명령을 취소했습니다.")
        self.assertNotIn("77", state["pending"])

    def test_execute_control_action_uses_pairwise_script_for_pairwise_runtime(self) -> None:
        with (
            patch.object(
                telegram_bot,
                "active_trader_context",
                return_value={"key": "pairwise", "script_path": telegram_bot.PAIRWISE_TRADER_SCRIPT},
            ),
            patch.object(
                telegram_bot,
                "run_local_command",
                return_value={"returncode": 0, "stdout": "ok", "stderr": "", "command": []},
            ) as mock_run,
        ):
            telegram_bot.execute_control_action("sync")
        mock_run.assert_called_once_with(
            [str(telegram_bot.PYTHON_BIN), str(telegram_bot.PAIRWISE_TRADER_SCRIPT), "sync-state"]
        )

    def test_execute_control_action_uses_core_script_for_core_runtime(self) -> None:
        with (
            patch.object(
                telegram_bot,
                "active_trader_context",
                return_value={"key": "core", "script_path": telegram_bot.TRADER_SCRIPT},
            ),
            patch.object(
                telegram_bot,
                "run_local_command",
                return_value={"returncode": 0, "stdout": "ok", "stderr": "", "command": []},
            ) as mock_run,
        ):
            telegram_bot.execute_control_action("protect")
        mock_run.assert_called_once_with(
            [str(telegram_bot.PYTHON_BIN), str(telegram_bot.TRADER_SCRIPT), "shutdown-protect", "--execute"]
        )

    def test_process_update_sends_all_async_read_commands(self) -> None:
        for command in sorted(telegram_bot.ASYNC_READ_COMMANDS):
            state = self.make_state()
            notices: list[tuple[int, str]] = []
            with self.subTest(command=command):
                with (
                    patch.object(telegram_bot, "TELEGRAM_ALLOWED_CHAT_IDS", set()),
                    patch.object(telegram_bot, "audit"),
                    patch.object(telegram_bot, "handle_command", return_value=f"reply:{command}"),
                    patch.object(telegram_bot, "send_message", side_effect=lambda chat_id, text: notices.append((chat_id, text))),
                    patch.object(telegram_bot, "send_chat_action"),
                    patch.object(telegram_bot, "send_response_and_audit") as mock_send_response,
                    patch.object(telegram_bot.COMMAND_EXECUTOR, "submit", side_effect=self.immediate_submit),
                ):
                    telegram_bot.process_update(state, self.make_update(f"/{command}"))
                self.assertGreaterEqual(len(notices), 1)
                self.assertIn(f"/{command}", notices[0][1])
                mock_send_response.assert_called_once()
                args = mock_send_response.call_args[0]
                self.assertEqual(args[1], 123456)
                self.assertEqual(args[2], f"/{command}")
                self.assertEqual(args[3], f"reply:{command}")
                self.assertEqual(args[4], command)

    def test_process_update_sends_all_sync_commands(self) -> None:
        cases = [
            ("/help", "help"),
            ("/start", "start"),
            ("/ping", "ping"),
            ("/starttrader", "starttrader"),
            ("/stoptrader", "stoptrader"),
            ("/restarttrader", "restarttrader"),
            ("/sync", "sync"),
            ("/protect", "protect"),
            ("/closeall", "closeall"),
            ("/flatten", "flatten"),
            ("/cancel", "cancel"),
            ("/confirm abc123", "confirm"),
        ]
        for text, command in cases:
            state = self.make_state()
            with self.subTest(command=command):
                with (
                    patch.object(telegram_bot, "TELEGRAM_ALLOWED_CHAT_IDS", set()),
                    patch.object(telegram_bot, "audit"),
                    patch.object(telegram_bot, "handle_command", return_value=f"reply:{command}"),
                    patch.object(telegram_bot, "send_response_and_audit") as mock_send_response,
                ):
                    telegram_bot.process_update(state, self.make_update(text))
                mock_send_response.assert_called_once()
                args = mock_send_response.call_args[0]
                self.assertEqual(args[1], 123456)
                self.assertEqual(args[2], text)
                self.assertEqual(args[3], f"reply:{command}")
                self.assertEqual(args[4], command)

    def test_process_update_supports_bot_username_suffix(self) -> None:
        state = self.make_state()
        notices: list[str] = []
        with (
            patch.object(telegram_bot, "TELEGRAM_ALLOWED_CHAT_IDS", set()),
            patch.object(telegram_bot, "audit"),
            patch.object(telegram_bot, "handle_command", return_value="reply:status"),
            patch.object(telegram_bot, "send_message", side_effect=lambda _chat_id, text: notices.append(text)),
            patch.object(telegram_bot, "send_chat_action"),
            patch.object(telegram_bot, "send_response_and_audit") as mock_send_response,
            patch.object(telegram_bot.COMMAND_EXECUTOR, "submit", side_effect=self.immediate_submit),
        ):
            telegram_bot.process_update(state, self.make_update("/status@EpicInvestBot"))
        self.assertTrue(notices)
        self.assertIn("/status", notices[0])
        mock_send_response.assert_called_once()
        self.assertEqual(mock_send_response.call_args[0][4], "status")

    def test_process_update_rejects_unauthorized_chat(self) -> None:
        state = self.make_state()
        with (
            patch.object(telegram_bot, "audit"),
            patch.object(telegram_bot, "send_message") as mock_send_message,
            patch.object(telegram_bot, "handle_command") as mock_handle_command,
            patch.object(telegram_bot, "TELEGRAM_ALLOWED_CHAT_IDS", {999999}),
        ):
            telegram_bot.process_update(state, self.make_update("/status", chat_id=111))
        mock_send_message.assert_called_once_with(111, "허용되지 않은 chat_id 입니다.")
        mock_handle_command.assert_not_called()

    def test_handle_async_read_command_sends_error_message_when_command_raises(self) -> None:
        state = self.make_state()
        with (
            patch.object(telegram_bot, "audit"),
            patch.object(telegram_bot, "send_chat_action"),
            patch.object(telegram_bot, "handle_command", side_effect=RuntimeError("boom")),
            patch.object(telegram_bot, "record_bot_error"),
            patch.object(telegram_bot, "send_response_and_audit") as mock_send_response,
        ):
            telegram_bot.handle_async_read_command(state, 123456, "/status", "status")
        self.assertIn("오류가 발생했습니다", mock_send_response.call_args[0][3])


if __name__ == "__main__":
    unittest.main()
