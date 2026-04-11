import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import core_strategy_registry
import rotation_target_050_live


class FakeProtectionExchange:
    def __init__(self) -> None:
        self.cancelled: list[tuple[str, str]] = []
        self.created: list[dict[str, object]] = []

    def market(self, _symbol: str) -> dict[str, object]:
        return {
            "precision": {"amount": 0.001},
            "limits": {"amount": {"min": 0.001}},
        }

    def amount_to_precision(self, _symbol: str, amount: float) -> str:
        return f"{abs(float(amount)):.3f}"

    def price_to_precision(self, _symbol: str, price: float) -> str:
        return f"{float(price):.2f}"

    def cancel_order(self, order_id: str, symbol: str, params: dict[str, object] | None = None) -> dict[str, object]:
        del params
        self.cancelled.append((order_id, symbol))
        return {"id": order_id, "symbol": symbol}

    def create_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: float | None,
        params: dict[str, object],
    ) -> dict[str, object]:
        del price
        order_id = f"new-{len(self.created) + 1}"
        row = {
            "id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "params": params,
            "clientOrderId": params["clientOrderId"],
        }
        self.created.append(row)
        return row


def btc_long_position(mark_price: float = 10_000.0) -> dict[str, object]:
    return {
        "BTCUSDT": {
            "pair": "BTCUSDT",
            "symbol": "BTC/USDT:USDT",
            "qty": 0.1,
            "side": "LONG",
            "entry_price": 9_950.0,
            "mark_price": mark_price,
            "margin_mode": "isolated",
        }
    }


def managed_order(order_id: str, tag: str, stop_price: float) -> dict[str, object]:
    return {
        "id": order_id,
        "symbol": "BTC/USDT:USDT",
        "type": "market",
        "side": "sell",
        "amount": 0.1,
        "stopPrice": stop_price,
        "clientOrderId": f"epiP{tag}BTC123456",
    }


class CoreLiveSmokeTests(unittest.TestCase):
    def test_core_champion_artifact_loads(self) -> None:
        artifact = core_strategy_registry.load_core_artifact(ROOT_DIR / "models" / "core_champion.json")
        self.assertIn(artifact.family, {core_strategy_registry.LONG_ONLY_FAMILY, core_strategy_registry.LONG_SHORT_FAMILY})
        self.assertTrue(artifact.key)
        self.assertTrue(artifact.source)
        self.assertIsNotNone(artifact.params)
        for key in (
            "promotion_gate",
            "validation_profile",
            "cpcv_pbo",
            "regime_breakdown_summary",
            "corr_state_summary",
            "candidate_selection_pbo",
        ):
            self.assertIn(key, artifact.metadata)

    def test_live_state_example_matches_loader_defaults(self) -> None:
        with open(ROOT_DIR / "models" / "rotation_target_050_live_state.example.json", "r") as f:
            example = json.load(f)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing_state.json"
            generated = rotation_target_050_live.load_state(path)
        self.assertEqual(example["notification_state"].keys(), generated["notification_state"].keys())
        self.assertEqual(example["runtime_health"].keys(), generated["runtime_health"].keys())
        self.assertEqual(example["latest_runtime_snapshot"].keys(), generated["latest_runtime_snapshot"].keys())
        self.assertEqual(example["decision_journal"].keys(), generated["decision_journal"].keys())

    def test_shutdown_protection_retains_matching_orders(self) -> None:
        exchange = FakeProtectionExchange()
        state: dict[str, object] = {}
        existing_orders = [
            managed_order("sl-1", "SL", 9_950.0),
            managed_order("tp-1", "TP", 10_125.0),
        ]

        with (
            patch.object(rotation_target_050_live, "fetch_open_position_map", return_value=btc_long_position()),
            patch.object(rotation_target_050_live, "fetch_strategy_protection_orders", return_value=existing_orders),
        ):
            report = rotation_target_050_live.install_shutdown_protection(exchange, state, execute=True)

        self.assertEqual(exchange.cancelled, [])
        self.assertEqual(exchange.created, [])
        self.assertEqual(report["status"], "retained")
        self.assertEqual(report["retained_count"], 2)
        self.assertEqual(report["placed_count"], 0)
        self.assertEqual(report["protections"][0]["status"], "retained")

    def test_shutdown_protection_replaces_when_order_is_materially_different(self) -> None:
        exchange = FakeProtectionExchange()
        state: dict[str, object] = {}
        existing_orders = [
            managed_order("sl-1", "SL", 9_800.0),
            managed_order("tp-1", "TP", 10_300.0),
        ]

        with (
            patch.object(rotation_target_050_live, "fetch_open_position_map", return_value=btc_long_position()),
            patch.object(rotation_target_050_live, "fetch_strategy_protection_orders", return_value=existing_orders),
        ):
            report = rotation_target_050_live.install_shutdown_protection(exchange, state, execute=True)

        self.assertEqual(exchange.cancelled, [("sl-1", "BTC/USDT:USDT"), ("tp-1", "BTC/USDT:USDT")])
        self.assertEqual(len(exchange.created), 2)
        self.assertEqual(report["status"], "placed")
        self.assertEqual(report["retained_count"], 0)
        self.assertEqual(report["placed_count"], 2)
        self.assertEqual(report["protections"][0]["status"], "placed")

    def test_routine_notifications_are_sent_at_most_once_per_hour(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = rotation_target_050_live.load_state(Path(tmpdir) / "state.json")

        sent: list[str] = []
        base = datetime(2026, 4, 11, 0, 0, tzinfo=timezone.utc)
        with (
            patch.object(
                rotation_target_050_live,
                "send_telegram_notification",
                side_effect=lambda text: sent.append(text) or True,
            ),
            patch.object(
                rotation_target_050_live,
                "utc_now",
                side_effect=[
                    base,
                    base + timedelta(minutes=10),
                    base + timedelta(minutes=61),
                ],
            ),
        ):
            rotation_target_050_live.dispatch_notifications(
                state,
                ["일일 시작 브리핑\n- 적용일: 2026-04-11"],
            )
            rotation_target_050_live.dispatch_notifications(
                state,
                ["오버레이 진입\n- 적용일: 2026-04-11"],
            )
            rotation_target_050_live.dispatch_notifications(state, [])

        self.assertEqual(len(sent), 2)
        self.assertIn("운영 정기 요약", sent[0])
        self.assertIn("일일 시작 브리핑", sent[0])
        self.assertIn("오버레이 진입", sent[1])
        self.assertEqual(state["notification_state"]["pending_routine_notifications"], [])

    def test_critical_notifications_bypass_hourly_digest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = rotation_target_050_live.load_state(Path(tmpdir) / "state.json")

        sent: list[str] = []
        base = datetime(2026, 4, 11, 0, 0, tzinfo=timezone.utc)
        with (
            patch.object(
                rotation_target_050_live,
                "send_telegram_notification",
                side_effect=lambda text: sent.append(text) or True,
            ),
            patch.object(
                rotation_target_050_live,
                "utc_now",
                side_effect=[
                    base,
                    base + timedelta(minutes=5),
                    base + timedelta(minutes=61),
                ],
            ),
        ):
            rotation_target_050_live.dispatch_notifications(
                state,
                ["일일 시작 브리핑\n- 적용일: 2026-04-11"],
            )
            rotation_target_050_live.dispatch_notifications(
                state,
                [
                    "코어 킬 스위치 발동\n- 손실률: -5.00%",
                    "세션 상태 변경\n- 적용일: 2026-04-11",
                ],
            )
            rotation_target_050_live.dispatch_notifications(state, [])

        self.assertEqual(len(sent), 3)
        self.assertIn("운영 정기 요약", sent[0])
        self.assertEqual(sent[1], "코어 킬 스위치 발동\n- 손실률: -5.00%")
        self.assertIn("세션 상태 변경", sent[2])


if __name__ == "__main__":
    unittest.main()
