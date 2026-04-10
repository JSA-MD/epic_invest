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

    def test_split_market_os_frames_reserves_recent_adaptation_and_final_oos(self) -> None:
        index = pd.date_range("2024-01-01", periods=420, freq="D", tz="UTC")
        frame = pd.DataFrame({"net_return": np.full(420, 0.002, dtype="float64")}, index=index)
        stages = engine.split_market_os_frames(frame)
        self.assertEqual(len(stages["final_oos"]), engine.FINAL_OOS_DAYS)
        self.assertEqual(
            len(stages["recent_adaptation"]),
            engine.RECENT_VALIDATION_DAYS - engine.FINAL_OOS_DAYS,
        )
        self.assertEqual(
            len(stages["structure_train"]) + len(stages["recent_adaptation"]) + len(stages["final_oos"]),
            len(frame),
        )

    def test_market_operating_system_rewards_corr_state_robustness(self) -> None:
        index = pd.date_range("2024-01-01", periods=240, freq="D", tz="UTC")
        returns = np.full(240, 0.0032, dtype="float64")
        frame = engine.build_return_frame(returns, index)
        cpcv = engine.summarize_cpcv_lite(frame, n_blocks=6, test_blocks=2, embargo_days=1)
        robust = engine.build_market_operating_system(
            frame,
            trial_count=8,
            cpcv=cpcv,
            pbo_profile={"selection_share": 0.8},
            state_payload={
                "route_state_returns": {
                    "equity_aligned:bull_broad": [0.0030, 0.0032, 0.0031],
                    "equity_mixed:bull_narrow": [0.0027, 0.0028],
                    "equity_inverse:bear_broad": [0.0018, 0.0020],
                },
                "corr_bucket_returns": {
                    "equity_aligned": [0.0030, 0.0031],
                    "equity_mixed": [0.0028, 0.0027],
                    "equity_inverse": [0.0018, 0.0020],
                },
                "total_route_states": 12,
                "total_corr_buckets": 3,
            },
            cost_reference={"mean_cost_ratio": 0.02, "mean_n_trades": 40},
        )
        fragile = engine.build_market_operating_system(
            frame,
            trial_count=8,
            cpcv=cpcv,
            pbo_profile={"selection_share": 0.2},
            state_payload={
                "route_state_returns": {
                    "equity_aligned:bull_broad": [0.0040, 0.0038],
                    "equity_inverse:bear_broad": [-0.0045, -0.0042],
                },
                "corr_bucket_returns": {
                    "equity_aligned": [0.0040, 0.0038],
                    "equity_inverse": [-0.0045, -0.0042],
                },
                "total_route_states": 12,
                "total_corr_buckets": 3,
            },
            cost_reference={"mean_cost_ratio": 0.18, "mean_n_trades": 140},
        )
        self.assertGreater(
            robust["fitness"]["raw"]["corr_state_robustness"],
            fragile["fitness"]["raw"]["corr_state_robustness"],
        )
        self.assertGreater(robust["fitness"]["score"], fragile["fitness"]["score"])

    def test_market_operating_system_keeps_final_oos_as_audit_only(self) -> None:
        index = pd.date_range("2024-01-01", periods=420, freq="D", tz="UTC")
        returns = np.concatenate(
            [
                np.full(359, 0.0030, dtype="float64"),
                np.full(61, -0.0020, dtype="float64"),
            ]
        )
        frame = engine.build_return_frame(returns, index)
        cpcv = engine.summarize_cpcv_lite(frame, n_blocks=6, test_blocks=2, embargo_days=1)
        market_os = engine.build_market_operating_system(
            frame,
            trial_count=8,
            cpcv=cpcv,
            pbo_profile={"selection_share": 0.8},
            state_payload={
                "route_state_returns": {
                    "equity_aligned:bull_broad": [0.0030, 0.0031, 0.0032],
                    "equity_mixed:bull_narrow": [0.0028, 0.0029],
                    "equity_inverse:bear_broad": [0.0018, 0.0019],
                },
                "corr_bucket_returns": {
                    "equity_aligned": [0.0030, 0.0031],
                    "equity_mixed": [0.0028, 0.0029],
                    "equity_inverse": [0.0018, 0.0019],
                },
                "total_route_states": 12,
                "total_corr_buckets": 3,
            },
            cost_reference={"mean_cost_ratio": 0.02, "mean_n_trades": 40},
        )

        self.assertIn("audit", market_os)
        self.assertFalse(market_os["audit"]["passes_final_oos_total_return"])
        self.assertNotIn("passes_final_oos_total_return", market_os["gate"])

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
            state_payload={
                "route_state_returns": {
                    "equity_aligned:bull_broad": [0.0035, 0.0034],
                    "equity_mixed:bull_narrow": [0.0028, 0.0026],
                    "equity_inverse:bear_narrow": [0.0017, 0.0016],
                },
                "corr_bucket_returns": {
                    "equity_aligned": [0.0035, 0.0034],
                    "equity_mixed": [0.0028, 0.0026],
                    "equity_inverse": [0.0017, 0.0016],
                },
                "total_route_states": 12,
                "total_corr_buckets": 3,
            },
            cost_reference={"mean_cost_ratio": 0.02, "mean_n_trades": 48},
            cpcv_blocks=6,
            cpcv_test_blocks=2,
            cpcv_embargo_days=1,
        )
        self.assertIn("gate", bundle)
        self.assertIn("profile", bundle)
        self.assertIn("dsr_proxy", bundle)
        self.assertIn("candidate_selection_pbo", bundle)
        self.assertIn("market_operating_system", bundle)
        self.assertIn("passes_market_os_fitness", bundle["gate"])
        self.assertIn("final_oos", bundle["market_operating_system"]["stages"])
        self.assertIn("audit", bundle["market_operating_system"])


if __name__ == "__main__":
    unittest.main()
