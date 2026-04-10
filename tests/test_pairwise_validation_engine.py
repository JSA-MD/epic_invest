import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_validation_engine as engine


class PairwiseValidationEngineTests(unittest.TestCase):
    def test_compute_dsr_proxy_rewards_stable_positive_returns(self) -> None:
        stable = np.asarray([0.0038, 0.0042] * 60, dtype="float64")
        noisy = np.asarray([0.02, -0.018] * 60, dtype="float64")
        self.assertGreater(engine.compute_dsr_proxy(stable, trial_count=20), engine.compute_dsr_proxy(noisy, trial_count=20))

    def test_summarize_cpcv_lite_builds_splits(self) -> None:
        index = pd.date_range("2025-01-01", periods=90, freq="D", tz="UTC")
        frame = pd.DataFrame({"net_return": np.full(90, 0.003, dtype="float64")}, index=index)
        result = engine.summarize_cpcv_lite(frame, n_blocks=6, test_blocks=2, embargo_days=1)
        self.assertGreater(result["n_splits"], 0)
        self.assertGreaterEqual(result["pass_rate"], 0.0)
        self.assertLessEqual(result["pass_rate"], 1.0)

    def test_build_candidate_validation_bundle_includes_gate(self) -> None:
        index = pd.date_range("2025-01-01", periods=120, freq="D", tz="UTC")
        returns_a = np.full(120, 0.0035, dtype="float64")
        returns_b = np.full(120, 0.0015, dtype="float64")
        frames = {
            "a": engine.build_return_frame(returns_a, index),
            "b": engine.build_return_frame(returns_b, index),
        }
        bundle = engine.build_candidate_validation_bundle(
            "a",
            returns_a,
            index,
            trial_count=8,
            peer_frames_by_key=frames,
            cpcv_blocks=6,
            cpcv_test_blocks=2,
            cpcv_embargo_days=1,
        )
        self.assertIn("gate", bundle)
        self.assertIn("profile", bundle)
        self.assertIn("dsr_proxy", bundle)
        self.assertIn("candidate_selection_pbo", bundle)


if __name__ == "__main__":
    unittest.main()
