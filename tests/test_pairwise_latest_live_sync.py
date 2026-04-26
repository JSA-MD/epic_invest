import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_regime_live as prl


def _make_args(execute: bool = True, mode: str = "primary") -> object:
    import argparse
    args = argparse.Namespace(
        execute=execute,
        mode=mode,
        state_path=str(ROOT_DIR / "models" / "pairwise_state.json"),
        summary_path=str(ROOT_DIR / "models" / "pairwise_summary.json"),
        model_path=str(ROOT_DIR / "models" / "pairwise_model.json"),
        promotion_report=str(ROOT_DIR / "models" / "pairwise_promotion_report.json"),
        decision_log_path=str(ROOT_DIR / "models" / "pairwise_decision_log.jsonl"),
        refresh_live_data=False,
        force_execute=False,
        force_note="",
    )
    return args


class TestLatestLiveSyncPostTrade(unittest.TestCase):
    """Defect 1: state['latest_live_sync'] must reflect the POST-trade positions."""

    def test_post_trade_snapshot_overwrites_pre_trade(self):
        pre_trade_positions = {"BNBUSDT": {"qty": -10.0}}
        post_trade_positions = {"BNBUSDT": {"qty": 0.0}}

        bridge = MagicMock()
        bridge.fetch_equity.return_value = 1000.0
        # First call returns pre-trade, second call (post-reconcile) returns post-trade
        bridge.fetch_open_position_map.side_effect = [
            pre_trade_positions,
            post_trade_positions,
        ]
        bridge.reconcile_target_positions.return_value = []
        bridge.install_shutdown_protection.return_value = {}

        fake_plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {},
            "promotion_report_path": None,
        }
        fake_gate = {"requested_gate_ready": True, "gate_passes": True}

        state = {"position_open_since_ts": {}}

        with patch.object(prl, "load_execution_bridge", return_value=bridge), \
             patch.object(prl, "load_state", return_value=state), \
             patch.object(prl, "save_state"), \
             patch.object(prl, "build_pairwise_plan", return_value=fake_plan), \
             patch.object(prl, "persist_runtime_plan_state"), \
             patch.object(prl, "load_promotion_gate", return_value=fake_gate), \
             patch.object(prl, "promotion_gate_allows_execution", return_value=True), \
             patch.object(prl, "record_runtime_success"), \
             patch.object(prl, "append_jsonl"), \
             patch.object(prl, "sync_shadow_paper_from_live_positions"), \
             patch.object(prl, "sync_position_loss_notifications"), \
             patch.object(prl, "load_notification_bridge", return_value=MagicMock()):
            prl.run_live_once(_make_args(execute=True))

        # The final latest_live_sync must be the post-trade snapshot
        sync = state.get("latest_live_sync", {})
        self.assertEqual(sync.get("source"), "run_live_once_post_trade",
                         "source should be 'run_live_once_post_trade' after reconcile")
        self.assertEqual(sync.get("positions"), post_trade_positions,
                         "positions should reflect post-trade exchange state")

    def test_pre_trade_snapshot_written_initially(self):
        """The pre-trade write still happens (crash-safety) — verify it uses 'run_live_once' source."""
        pre_trade_positions = {"BNBUSDT": {"qty": -10.0}}
        post_trade_positions = {"BNBUSDT": {"qty": 0.0}}

        bridge = MagicMock()
        bridge.fetch_equity.return_value = 1000.0
        bridge.fetch_open_position_map.side_effect = [
            pre_trade_positions,
            post_trade_positions,
        ]
        bridge.reconcile_target_positions.return_value = []
        bridge.install_shutdown_protection.return_value = {}

        fake_plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {},
            "promotion_report_path": None,
        }
        fake_gate = {"requested_gate_ready": True, "gate_passes": True}

        state = {"position_open_since_ts": {}}
        sync_snapshots = []

        original_setitem = dict.__setitem__

        class TrackingState(dict):
            def __setitem__(self, key, value):
                if key == "latest_live_sync":
                    sync_snapshots.append(dict(value))
                super().__setitem__(key, value)

        tracking_state = TrackingState(state)

        with patch.object(prl, "load_execution_bridge", return_value=bridge), \
             patch.object(prl, "load_state", return_value=tracking_state), \
             patch.object(prl, "save_state"), \
             patch.object(prl, "build_pairwise_plan", return_value=fake_plan), \
             patch.object(prl, "persist_runtime_plan_state"), \
             patch.object(prl, "load_promotion_gate", return_value=fake_gate), \
             patch.object(prl, "promotion_gate_allows_execution", return_value=True), \
             patch.object(prl, "record_runtime_success"), \
             patch.object(prl, "append_jsonl"), \
             patch.object(prl, "sync_shadow_paper_from_live_positions"), \
             patch.object(prl, "sync_position_loss_notifications"), \
             patch.object(prl, "load_notification_bridge", return_value=MagicMock()):
            prl.run_live_once(_make_args(execute=True))

        self.assertGreaterEqual(len(sync_snapshots), 2, "expected at least 2 latest_live_sync writes")
        self.assertEqual(sync_snapshots[0]["source"], "run_live_once")
        self.assertEqual(sync_snapshots[0]["positions"], pre_trade_positions)
        self.assertEqual(sync_snapshots[-1]["source"], "run_live_once_post_trade")
        self.assertEqual(sync_snapshots[-1]["positions"], post_trade_positions)


if __name__ == "__main__":
    unittest.main()
