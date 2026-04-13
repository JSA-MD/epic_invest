import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from btc_convex_blend import (
    blend_runtime_weight,
    build_blended_target_trace,
    get_btc_convex_blend,
)


class BTCConvexBlendTests(unittest.TestCase):
    def test_get_btc_convex_blend_filters_non_btc_pair(self) -> None:
        candidate = {
            "btc_convex_blend": {
                "alpha": 0.39,
                "mode": "always",
                "pair": "BTCUSDT",
                "specialist_pair_config": {"mapping_indices": [1] * 12, "route_breadth_threshold": 0.5},
            }
        }
        self.assertIsNotNone(get_btc_convex_blend(candidate, "BTCUSDT"))
        self.assertIsNone(get_btc_convex_blend(candidate, "BNBUSDT"))

    def test_blend_runtime_weight_respects_mode(self) -> None:
        self.assertAlmostEqual(
            blend_runtime_weight(
                baseline_weight=0.4,
                specialist_weight=1.0,
                route_state_name="equity_up_narrow",
                alpha=0.5,
                mode="always",
            ),
            0.7,
        )
        self.assertAlmostEqual(
            blend_runtime_weight(
                baseline_weight=0.4,
                specialist_weight=-1.0,
                route_state_name="equity_up_broad",
                alpha=0.5,
                mode="disagree_narrow_only",
            ),
            0.4,
        )

    def test_build_blended_target_trace_narrow_only(self) -> None:
        context = {
            "route_state_mode": "equity_corr",
            "bucket_codes": {
                0.5: np.asarray([1, 0, 3, 2], dtype="int64"),
            },
        }
        baseline_trace = {"target_weight": np.asarray([0.0, 0.5, 0.5, -0.5], dtype="float64")}
        specialist_trace = {"target_weight": np.asarray([1.0, 1.0, -1.0, -1.0], dtype="float64")}
        blended = build_blended_target_trace(
            context=context,
            route_breadth_threshold=0.5,
            baseline_trace=baseline_trace,
            specialist_trace=specialist_trace,
            alpha=0.4,
            mode="narrow_only",
        )
        expected = np.asarray([0.0, 0.7, 0.5, -0.7], dtype="float64")
        np.testing.assert_allclose(blended, expected)


if __name__ == "__main__":
    unittest.main()
