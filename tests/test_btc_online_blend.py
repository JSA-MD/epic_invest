import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from btc_online_blend import (
    build_online_expert_blend_trace,
    runtime_online_blend_alpha,
    update_runtime_online_score,
)


class BTCOnlineBlendTests(unittest.TestCase):
    def test_online_blend_is_causal(self) -> None:
        context = {
            "route_state_mode": "equity_corr",
            "bucket_codes": {0.5: np.asarray([11, 11, 11], dtype="int64")},
        }
        baseline = {
            "target_weight": np.asarray([0.0, 0.0, 0.0], dtype="float64"),
            "bar_net": np.asarray([0.01, -0.01, 0.0], dtype="float64"),
        }
        specialist = {
            "target_weight": np.asarray([1.0, 1.0, 1.0], dtype="float64"),
            "bar_net": np.asarray([0.03, -0.03, 0.0], dtype="float64"),
        }
        out = build_online_expert_blend_trace(
            context=context,
            route_breadth_threshold=0.5,
            baseline_trace=baseline,
            specialist_trace=specialist,
            alpha_cap=0.4,
            eta=0.1,
            decay=0.95,
            activation_mode="always",
            reward_scale=100.0,
        )
        self.assertAlmostEqual(float(out["alpha"][0]), 0.0)
        self.assertGreater(float(out["alpha"][1]), 0.0)

    def test_online_blend_respects_activation_mask(self) -> None:
        context = {
            "route_state_mode": "equity_corr",
            "bucket_codes": {0.5: np.asarray([10, 11, 9], dtype="int64")},
        }
        baseline = {
            "target_weight": np.asarray([0.0, 1.0, -1.0], dtype="float64"),
            "bar_net": np.asarray([0.0, 0.0, 0.0], dtype="float64"),
        }
        specialist = {
            "target_weight": np.asarray([1.0, 1.0, 1.0], dtype="float64"),
            "bar_net": np.asarray([0.02, 0.02, 0.02], dtype="float64"),
        }
        out = build_online_expert_blend_trace(
            context=context,
            route_breadth_threshold=0.5,
            baseline_trace=baseline,
            specialist_trace=specialist,
            alpha_cap=0.4,
            eta=0.2,
            decay=0.98,
            activation_mode="disagree_narrow_only",
        )
        self.assertTrue(bool(out["active"][0]))
        self.assertFalse(bool(out["active"][1]))
        self.assertFalse(bool(out["active"][2]))

    def test_runtime_online_alpha_uses_previous_score_and_activation(self) -> None:
        alpha = runtime_online_blend_alpha(
            previous_score=2.0,
            alpha_cap=0.1,
            eta=0.1,
            activation_mode="disagree_only",
            route_state_name="equity_aligned:bull_broad",
            baseline_weight=-1.0,
            specialist_weight=1.0,
        )
        self.assertGreater(alpha, 0.0)
        self.assertLessEqual(alpha, 0.1)

    def test_runtime_online_score_updates_causally(self) -> None:
        score = update_runtime_online_score(
            previous_score=0.5,
            baseline_weight=0.0,
            specialist_weight=1.0,
            previous_price=100.0,
            current_price=101.0,
            decay=0.98,
            reward_scale=100.0,
        )
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()
