import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_pair_subset_fractal_genome as fractal
from fractal_genome_core import collect_specs, LeafNode


class FractalObservationModesTests(unittest.TestCase):
    def test_build_feature_specs_filters_by_observation_mode(self) -> None:
        volume_specs = fractal.build_feature_specs(("BTCUSDT", "BNBUSDT"), observation_mode="volume")
        volume_features = {name for name, _, _ in volume_specs}
        self.assertIn("btc_regime", volume_features)
        self.assertIn("btc_volume_rel_1h", volume_features)
        self.assertIn("volume_rel_spread_btc_minus_bnb_1h", volume_features)
        self.assertNotIn("btc_dc_event_05_1h", volume_features)
        self.assertNotIn("btc_order_imbalance_1h", volume_features)

        imbalance_specs = fractal.build_feature_specs(("BTCUSDT", "BNBUSDT"), observation_mode="imbalance")
        imbalance_features = {name for name, _, _ in imbalance_specs}
        self.assertIn("btc_regime", imbalance_features)
        self.assertIn("btc_order_imbalance_1h", imbalance_features)
        self.assertIn("imbalance_spread_btc_minus_bnb_1h", imbalance_features)
        self.assertNotIn("btc_volume_rel_1h", imbalance_features)
        self.assertNotIn("btc_dc_event_05_1h", imbalance_features)

    def test_candidate_tree_key_is_unique_per_observation_mode(self) -> None:
        tree = LeafNode(0)
        time_key = fractal.candidate_tree_key("time", "5m", tree)
        dc_key = fractal.candidate_tree_key("directional_change", "5m", tree)
        self.assertNotEqual(time_key, dc_key)
        self.assertNotEqual(
            fractal.candidate_tree_key("time", "5m", tree),
            fractal.candidate_tree_key("time", "4h", tree),
        )

    def test_seed_trees_respect_mode_feature_universe(self) -> None:
        condition_options = fractal.build_condition_options(
            fractal.build_feature_specs(("BTCUSDT", "BNBUSDT"), observation_mode="imbalance")
        )
        allowed = {spec.feature for spec in condition_options}
        expert_pool = [
            {"pair_configs": {"BTCUSDT": {}, "BNBUSDT": {}}},
            {"pair_configs": {"BTCUSDT": {}, "BNBUSDT": {}}},
            {"pair_configs": {"BTCUSDT": {}, "BNBUSDT": {}}},
            {"pair_configs": {"BTCUSDT": {}, "BNBUSDT": {}}},
        ]

        seeds = fractal.build_seed_trees(expert_pool, condition_options, ("BTCUSDT", "BNBUSDT"))

        for seed in seeds:
            for spec in collect_specs(seed):
                self.assertIn(spec.feature, allowed)

    def test_project_feature_arrays_changes_common_representation_by_observation_mode(self) -> None:
        base = {
            "btc_regime": np.asarray([0.10, 0.10], dtype="float64"),
            "bnb_regime": np.asarray([-0.05, -0.05], dtype="float64"),
            "regime_spread_btc_minus_bnb": np.asarray([0.15, 0.15], dtype="float64"),
            "breadth": np.asarray([0.50, 0.50], dtype="float64"),
            "breadth_change_1d": np.asarray([0.05, -0.05], dtype="float64"),
            "btc_volume_rel_1h": np.asarray([4.0, 1.0], dtype="float64"),
            "bnb_volume_rel_1h": np.asarray([1.0, 0.5], dtype="float64"),
            "btc_order_imbalance_1h": np.asarray([1.0, 0.0], dtype="float64"),
            "bnb_order_imbalance_1h": np.asarray([-1.0, 0.0], dtype="float64"),
            "imbalance_spread_btc_minus_bnb_1h": np.asarray([1.0, 0.0], dtype="float64"),
            "btc_dc_trend_05_1h": np.asarray([1.0, 0.0], dtype="float64"),
            "bnb_dc_trend_05_1h": np.asarray([-1.0, 0.0], dtype="float64"),
            "dc_trend_spread_btc_minus_bnb_1h": np.asarray([2.0, 0.0], dtype="float64"),
            "btc_dc_event_05_1h": np.asarray([1.0, 0.0], dtype="float64"),
            "bnb_dc_event_05_1h": np.asarray([-1.0, 0.0], dtype="float64"),
            "dc_event_spread_btc_minus_bnb_1h": np.asarray([2.0, 0.0], dtype="float64"),
            "btc_dc_run_05_1h": np.asarray([1.0, 0.0], dtype="float64"),
            "bnb_dc_run_05_1h": np.asarray([-1.0, 0.0], dtype="float64"),
        }

        volume = fractal.project_feature_arrays_by_observation_mode(base, "volume")
        imbalance = fractal.project_feature_arrays_by_observation_mode(base, "imbalance")
        directional_change = fractal.project_feature_arrays_by_observation_mode(base, "directional_change")

        self.assertGreater(volume["btc_regime"][0], base["btc_regime"][0])
        self.assertGreater(imbalance["btc_regime"][0], base["btc_regime"][0])
        self.assertGreater(directional_change["btc_regime"][0], base["btc_regime"][0])
        self.assertFalse(np.allclose(volume["breadth"], base["breadth"]))
        self.assertFalse(np.allclose(imbalance["breadth"], base["breadth"]))
        self.assertFalse(np.allclose(directional_change["breadth"], base["breadth"]))
        self.assertLessEqual(float(volume["breadth"][0]), 1.0)

    def test_apply_label_horizon_uses_decision_cadence(self) -> None:
        base = {
            "btc_intraday_return_1h": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float64"),
        }
        cadenced = fractal.apply_label_horizon_to_feature_arrays(base, "30m")
        np.testing.assert_allclose(
            cadenced["btc_intraday_return_1h"],
            np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0], dtype="float64"),
        )


if __name__ == "__main__":
    unittest.main()
