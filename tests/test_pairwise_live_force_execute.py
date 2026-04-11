import sys
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_regime_live as pairwise_live


REPAIR_SUMMARY_NAME = "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
VALIDATED_SUMMARY_NAME = "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"


def make_args(*, execute: bool, force_execute: bool) -> Namespace:
    return Namespace(
        command="run-once",
        summary_path=Path("models/mock_summary.json"),
        model_path=Path("models/mock_model.dill"),
        state_path=Path("models/mock_live_state.json"),
        decision_log_path=Path("logs/mock_pairwise_live.jsonl"),
        equity=100000.0,
        refresh_live_data=False,
        execute=execute,
        force_execute=force_execute,
        force_note="manual_primary_switch",
        mode="demo",
        shadow_state_path=Path("models/mock_shadow_state.json"),
    )


class PairwiseLiveForceExecuteTests(unittest.TestCase):
    def test_operational_defaults_use_validated_pairwise_summary(self) -> None:
        self.assertEqual(pairwise_live.DEFAULT_SUMMARY_PATH, ROOT_DIR / "models" / VALIDATED_SUMMARY_NAME)

        shadow_live_source = (SCRIPTS_DIR / "pairwise_regime_mixture_shadow_live.py").read_text()
        self.assertIn(VALIDATED_SUMMARY_NAME, shadow_live_source)
        self.assertNotIn(f'DEFAULT_SUMMARY_PATH = gp.MODELS_DIR / "{REPAIR_SUMMARY_NAME}"', shadow_live_source)

        launchd_source = (SCRIPTS_DIR / "pairwise_shadow_launchd_entry.sh").read_text()
        self.assertIn(f"SUMMARY_PATH=\"${{PAIRWISE_SHADOW_SUMMARY_PATH:-$ROOT_DIR/models/{VALIDATED_SUMMARY_NAME}}}\"", launchd_source)
        self.assertNotIn(REPAIR_SUMMARY_NAME, launchd_source)

    def test_compute_requested_weight_de_risks_when_equity_corr_is_inverse(self) -> None:
        params = MagicMock(
            signal_span=2,
            regime_threshold=0.01,
            breadth_threshold=0.5,
            target_vol_ann=10.0,
            gross_cap=1.5,
        )
        baseline = pairwise_live.compute_requested_weight(
            raw_signal=pairwise_live.np.asarray([150.0, 150.0]),
            params=params,
            regime_score=0.05,
            breadth_score=0.8,
            bar_vol_ann=0.2,
            equity_corr_gross_scale=1.0,
            equity_corr_regime_mult=1.0,
        )
        reduced = pairwise_live.compute_requested_weight(
            raw_signal=pairwise_live.np.asarray([150.0, 150.0]),
            params=params,
            regime_score=0.05,
            breadth_score=0.8,
            bar_vol_ann=0.2,
            equity_corr_gross_scale=0.8,
            equity_corr_regime_mult=1.1,
        )
        self.assertAlmostEqual(baseline, 1.5)
        self.assertAlmostEqual(reduced, 1.2)

    def test_live_execute_blocks_when_gate_fails_without_force(self) -> None:
        args = make_args(execute=True, force_execute=False)
        bridge = MagicMock()

        with (
            patch.object(pairwise_live, "load_state", side_effect=[{}, {}]),
            patch.object(pairwise_live, "build_pairwise_plan", return_value={"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5}}),
            patch.object(pairwise_live, "default_promotion_eval_args", return_value=Namespace()),
            patch.object(pairwise_live, "build_shadow_evaluation", return_value={"promotion_ready": False}),
            patch.object(pairwise_live, "record_runtime_success"),
            patch.object(pairwise_live, "append_jsonl"),
            patch.object(pairwise_live, "save_state"),
            patch.object(pairwise_live, "load_execution_bridge", return_value=bridge),
        ):
            result = pairwise_live.run_live_once(args)

        self.assertEqual(result, 2)
        bridge.get_exchange.assert_not_called()

    def test_live_execute_bypasses_failed_gate_when_forced(self) -> None:
        args = make_args(execute=True, force_execute=True)
        bridge = MagicMock()
        bridge.get_exchange.return_value = object()
        bridge.fetch_equity.return_value = 12345.0
        bridge.reconcile_target_positions.return_value = [{"pair": "BNBUSDT", "action": "SELL"}]
        bridge.install_shutdown_protection.return_value = {"installed": True}

        with (
            patch.object(pairwise_live, "load_state", side_effect=[{}, {}]),
            patch.object(pairwise_live, "build_pairwise_plan", return_value={"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5}}),
            patch.object(pairwise_live, "default_promotion_eval_args", return_value=Namespace()),
            patch.object(pairwise_live, "build_shadow_evaluation", return_value={"promotion_ready": False}),
            patch.object(pairwise_live, "record_runtime_success"),
            patch.object(pairwise_live, "append_jsonl"),
            patch.object(pairwise_live, "save_state"),
            patch.object(pairwise_live, "load_execution_bridge", return_value=bridge),
        ):
            result = pairwise_live.run_live_once(args)

        self.assertEqual(result, 0)
        bridge.get_exchange.assert_called_once_with("demo")
        bridge.fetch_equity.assert_called_once()
        bridge.reconcile_target_positions.assert_called_once()
        bridge.install_shutdown_protection.assert_called_once()

    def test_live_execute_blocks_forced_order_when_shadow_feed_is_stale(self) -> None:
        args = make_args(execute=True, force_execute=True)
        bridge = MagicMock()

        with (
            patch.object(pairwise_live, "load_state", side_effect=[{}, {}]),
            patch.object(pairwise_live, "build_pairwise_plan", return_value={"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5}}),
            patch.object(pairwise_live, "default_promotion_eval_args", return_value=Namespace()),
            patch.object(
                pairwise_live,
                "build_shadow_evaluation",
                return_value={
                    "promotion_ready": False,
                    "shadow_feed_stale": True,
                    "shadow_signal_stale": False,
                    "reasons": ["shadow feed stale 30.0m > cap 20.0m"],
                },
            ),
            patch.object(pairwise_live, "record_runtime_success"),
            patch.object(pairwise_live, "append_jsonl"),
            patch.object(pairwise_live, "save_state"),
            patch.object(pairwise_live, "load_execution_bridge", return_value=bridge),
        ):
            result = pairwise_live.run_live_once(args)

        self.assertEqual(result, 2)
        bridge.get_exchange.assert_not_called()

    def test_load_live_frame_requests_recent_klines_with_datetimes(self) -> None:
        base_index = pd.date_range("2026-04-10 10:00", periods=25, freq="5min", tz="UTC")
        base = pd.DataFrame(
            {
                "BTCUSDT_close": [float(i) for i in range(25)],
                "BNBUSDT_close": [float(i + 100) for i in range(25)],
            },
            index=base_index,
        )
        recent_index = pd.date_range("2026-04-10 12:00", periods=2, freq="5min", tz="UTC")
        recent_btc = pd.DataFrame({"close": [2.1, 2.2]}, index=recent_index)
        recent_bnb = pd.DataFrame({"close": [4.1, 4.2]}, index=recent_index)

        with (
            patch.object(pairwise_live.gp, "load_all_pairs", return_value=base),
            patch.object(
                pairwise_live.gp,
                "fetch_klines",
                side_effect=[recent_btc, recent_bnb],
            ) as fetch_klines,
            patch.object(
                pairwise_live,
                "utc_now",
                return_value=datetime(2026, 4, 10, 12, 7, tzinfo=timezone.utc),
            ),
        ):
            df = pairwise_live.load_live_frame(("BTCUSDT", "BNBUSDT"), refresh_live_data=True, recent_days=1)

        first_call = fetch_klines.call_args_list[0]
        self.assertIsInstance(first_call.args[2], datetime)
        self.assertIsInstance(first_call.args[3], datetime)
        self.assertEqual(df.index.max().isoformat(), "2026-04-10T12:05:00+00:00")


if __name__ == "__main__":
    unittest.main()
