import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from event_orderflow_specialist import build_event_orderflow_trace


class EventOrderflowSpecialistTests(unittest.TestCase):
    def test_long_reclaim_in_uptrend_triggers_when_flat(self) -> None:
        context = {
            "close": [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
            "order_imbalance": [0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
            "buy_volume_share": [0.5, 0.5, 0.5, 0.6, 0.8, 0.8],
            "close_location_value": [0.0] * 6,
            "body_to_range": [0.0] * 6,
            "wick_skew": [0.0] * 6,
            "oi_rel": [1.0] * 6,
            "basis_rate": [0.0] * 6,
            "top_pos_log_ratio": [0.0] * 6,
            "taker_buy_sell_log_ratio": [0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
            "dc_trend_05": [0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
            "dc_run_05": [0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
            "range_bps": [10.0, 10.0, 10.0, 20.0, 40.0, 40.0],
            "volume_ratio": [1.0, 1.0, 1.0, 1.0, 1.3, 1.1],
        }
        baseline = {"target_weight": [0.0] * 6}
        out = build_event_orderflow_trace(
            context=context,
            baseline_trace=baseline,
            trigger_weight=0.75,
            fast_span=2,
            slow_span=4,
            retest_lookback=3,
            support_floor=0.0,
            volume_ratio_floor=1.0,
            activation_mode="flat_only",
            hold_bars=2,
        )
        self.assertEqual(float(out["trigger_side"][4]), 1.0)
        self.assertEqual(float(out["target_weight"][4]), 0.75)

    def test_existing_baseline_same_side_blocks_flat_only(self) -> None:
        context = {
            "close": [100.0, 101.0, 102.0, 101.0, 103.0, 104.0],
            "order_imbalance": [0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
            "buy_volume_share": [0.5, 0.5, 0.5, 0.6, 0.8, 0.8],
            "close_location_value": [0.0] * 6,
            "body_to_range": [0.0] * 6,
            "wick_skew": [0.0] * 6,
            "oi_rel": [1.0] * 6,
            "basis_rate": [0.0] * 6,
            "top_pos_log_ratio": [0.0] * 6,
            "taker_buy_sell_log_ratio": [0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
            "dc_trend_05": [0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
            "dc_run_05": [0.0, 0.0, 0.0, 0.1, 0.2, 0.2],
            "range_bps": [10.0, 10.0, 10.0, 20.0, 40.0, 40.0],
            "volume_ratio": [1.0, 1.0, 1.0, 1.0, 1.3, 1.1],
        }
        baseline = {"target_weight": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]}
        out = build_event_orderflow_trace(
            context=context,
            baseline_trace=baseline,
            trigger_weight=0.75,
            fast_span=2,
            slow_span=4,
            retest_lookback=3,
            support_floor=0.0,
            volume_ratio_floor=1.0,
            activation_mode="flat_only",
            hold_bars=2,
        )
        self.assertEqual(float(out["target_weight"][4]), 0.5)


if __name__ == "__main__":
    unittest.main()
