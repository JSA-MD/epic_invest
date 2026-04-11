import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fractal_genome_core import LeafGene
from pairwise_regime_mixture_shadow_live import (
    apply_leaf_gene_to_overlay_params,
    detect_candidate_kind,
    load_promotion_gate,
)
from run_pair_subset_promotion_pipeline import build_fractal_stress_report, build_fractal_validation_report
from search_gp_drawdown_overlay import OverlayParams
from search_pair_subset_fractal_genome import apply_leaf_gene_to_pair_config


class FractalRuntimeBridgeTests(unittest.TestCase):
    def test_apply_leaf_gene_to_pair_config_shifts_route_and_mapping(self) -> None:
        pair_config = {
            "route_breadth_threshold": 0.50,
            "mapping_indices": [10, 20, 30, 40],
            "route_state_mode": "base",
        }
        adjusted = apply_leaf_gene_to_pair_config(
            pair_config,
            LeafGene(route_threshold_bias=1, mapping_shift=-5),
            (0.35, 0.50, 0.65, 0.80),
            100,
        )
        self.assertEqual(adjusted["route_breadth_threshold"], 0.65)
        self.assertEqual(adjusted["mapping_indices"], [5, 15, 25, 35])
        self.assertEqual(adjusted["route_state_mode"], "base")

    def test_apply_leaf_gene_to_overlay_params_scales_risk_controls(self) -> None:
        params = OverlayParams(
            signal_span=24,
            rebalance_bars=1,
            regime_threshold=0.05,
            breadth_threshold=0.50,
            target_vol_ann=0.20,
            gross_cap=1.50,
            kill_switch_pct=0.10,
            cooldown_days=3,
        )
        adjusted = apply_leaf_gene_to_overlay_params(
            params,
            LeafGene(
                target_vol_scale=1.25,
                gross_cap_scale=0.80,
                kill_switch_scale=0.50,
                cooldown_scale=2.0,
            ),
        )
        self.assertAlmostEqual(adjusted.target_vol_ann, 0.25)
        self.assertAlmostEqual(adjusted.gross_cap, 1.20)
        self.assertAlmostEqual(adjusted.kill_switch_pct, 0.05)
        self.assertEqual(adjusted.cooldown_days, 6)

    def test_detect_candidate_kind_handles_tree_and_pairwise(self) -> None:
        self.assertEqual(detect_candidate_kind({"tree": {"type": "leaf", "expert_idx": 0}}), "fractal_tree")
        self.assertEqual(detect_candidate_kind({"pair_configs": {"BTCUSDT": {}}}), "pairwise_candidate")

    def test_load_promotion_gate_accepts_pipeline_decision_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pipeline_report.json"
            path.write_text(json.dumps({"decision": {"status": "ready_for_live", "ready_for_merge": True}}))
            gate = load_promotion_gate(path)
        self.assertEqual(gate["status"], "ready_for_live")
        self.assertTrue(gate["selected_candidate_ready_for_merge"])

    def test_fractal_pipeline_reports_embed_runtime_metadata(self) -> None:
        summary = {
            "pairs": ["BTCUSDT", "BNBUSDT"],
            "selected_candidate": {
                "candidate_kind": "fractal_tree",
                "tree_key": "abc",
                "observation_mode": "directional_change",
                "label_horizon": "4h",
                "tree_depth": 3,
                "logic_depth": 2,
                "validation": {
                    "profiles": {
                        "progressive_improvement": {"passed": False},
                        "target_060": {"passed": False},
                        "final_oos": {"passed": False},
                    }
                },
                "robustness": {
                    "gate_passed": False,
                    "wf_1": {"passed": True},
                    "stress_survival_rate_mean": 0.5,
                    "stress_survival_rate_min": 0.0,
                    "stress_survival_threshold": 0.67,
                    "latest_fold_stress_reserve_score": 123.0,
                },
            },
        }
        validation_report = build_fractal_validation_report(summary)
        stress_report = build_fractal_stress_report(summary)
        self.assertEqual(validation_report["selected_candidate"]["observation_mode"], "directional_change")
        self.assertEqual(validation_report["market_os_gate"]["passed"], False)
        self.assertEqual(stress_report["promotion_decision"]["status"], "shadow_ready_only")
        self.assertTrue(stress_report["promotion_decision"]["ready_for_live"])
        self.assertFalse(stress_report["promotion_decision"]["ready_for_merge"])


if __name__ == "__main__":
    unittest.main()
