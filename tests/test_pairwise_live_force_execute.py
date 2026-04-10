import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_regime_live as pairwise_live


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


if __name__ == "__main__":
    unittest.main()
