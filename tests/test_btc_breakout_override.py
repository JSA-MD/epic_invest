import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from btc_breakout_override import build_breakout_override_trace


class BTCBreakoutOverrideTests(unittest.TestCase):
    def test_build_breakout_override_trace_triggers_long_breakout_when_flat(self) -> None:
        close = [100.0, 101.0, 102.0, 103.5]
        zeros = [0.0] * 4
        ones = [1.5] * 4
        context = {
            "close": close,
            "volume_ratio": [1.0, 1.0, 1.0, 1.6],
            "order_imbalance": zeros,
            "buy_volume_share": [0.5, 0.5, 0.5, 0.9],
            "close_location_value": zeros,
            "body_to_range": zeros,
            "wick_skew": zeros,
            "oi_rel": ones,
            "basis_rate": zeros,
            "top_pos_log_ratio": zeros,
            "taker_buy_sell_log_ratio": zeros,
            "dc_trend_05": zeros,
            "dc_run_05": [0.0, 0.0, 0.0, 1.0],
            "range_bps": [10.0, 10.0, 10.0, 80.0],
        }
        baseline_trace = {"target_weight": [0.0, 0.0, 0.0, 0.0]}
        result = build_breakout_override_trace(
            context=context,
            baseline_trace=baseline_trace,
            alpha=1.0,
            breakout_weight=0.75,
            activation_mode="flat_only",
            lookback_bars=3,
            breakout_buffer_bps=0.0,
            support_floor=0.1,
            volume_ratio_floor=1.0,
        )
        self.assertEqual(float(result["target_weight"][-1]), 0.75)
        self.assertEqual(float(result["breakout_side"][-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
