import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fractal_genome_core import LeafGene
from pairwise_regime_mixture_shadow_live import (
    apply_leaf_gene_to_overlay_params,
    build_shadow_snapshot,
    detect_candidate_kind,
    load_live_frame,
    load_shadow_state,
    load_strategy_bundle,
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

    def test_load_strategy_bundle_prefers_embedded_model_and_baseline_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            nested = tmp / "artifacts"
            nested.mkdir()
            summary_path = nested / "summary.json"
            embedded_base = nested / "embedded_base.json"
            embedded_model = nested / "embedded_model.dill"
            fallback_base = nested / "fallback_base.json"
            fallback_model = nested / "fallback_model.dill"
            embedded_base.write_text(json.dumps({"selected_candidate": {"id": "embedded"}}))
            fallback_base.write_text(json.dumps({"selected_candidate": {"id": "fallback"}}))
            embedded_model.write_text("embedded")
            fallback_model.write_text("fallback")
            summary_path.write_text(
                json.dumps(
                    {
                        "baseline_summary_path": embedded_base.name,
                        "model_path": embedded_model.name,
                        "selected_candidate": {"pair_configs": {"BTCUSDT": {}}},
                    }
                )
            )

            with (
                patch("pairwise_regime_mixture_shadow_live.iter_params", return_value=[{"dummy": True}]),
                patch("pairwise_regime_mixture_shadow_live.load_model", return_value=(object(), None)),
                patch(
                    "pairwise_regime_mixture_shadow_live.gp.toolbox.compile",
                    return_value=lambda *args: [0.0],
                ),
            ):
                bundle = load_strategy_bundle(
                    summary_path,
                    fallback_base,
                    fallback_model,
                )

        self.assertEqual(Path(bundle["base_summary_path"]).name, "embedded_base.json")
        self.assertEqual(Path(bundle["model_path"]).name, "embedded_model.dill")

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

    def test_load_shadow_state_upgrades_legacy_journal_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state_path = tmp / "state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "decision_journal": [
                            {"at": "2026-04-11T00:00:00+00:00", "session_type": "flat"},
                        ]
                    }
                )
            )
            state = load_shadow_state(state_path, tmp / "decisions.jsonl")

        self.assertIsInstance(state["decision_journal"], dict)
        self.assertEqual(len(state["decision_journal"]["history"]), 1)
        self.assertIn("shadow_paper", state)
        self.assertIn("latest_decision_snapshot", state)

    def test_build_shadow_snapshot_keeps_watchdog_shadow_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            state = load_shadow_state(tmp / "missing_state.json", tmp / "decisions.jsonl")

        plan = {
            "equity": 10000.0,
            "pairs": ["BTCUSDT", "BNBUSDT"],
            "pair_plans": [
                {"pair": "BTCUSDT"},
                {"pair": "BNBUSDT"},
            ],
            "selected_candidate": {"candidate_kind": "pairwise_candidate"},
            "promotion_gate": {"status": "shadow_ready_only"},
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -1.5},
            "gross_target_weight": 1.5,
            "session_type": "pairwise",
            "signal_timestamp": "2026-04-11T00:00:00+00:00",
        }
        build_shadow_snapshot(state, plan, None)

        self.assertEqual(state["shadow_paper"]["last_signal_timestamp"], "2026-04-11T00:00:00+00:00")
        self.assertEqual(state["shadow_paper"]["current_weights"]["BNBUSDT"], -1.5)
        self.assertEqual(state["latest_decision_snapshot"]["session_type"], "pairwise")

    def test_load_live_frame_merges_recent_pair_bars_and_drops_incomplete_tail(self) -> None:
        base_index = pd.date_range("2026-04-10 10:00", periods=25, freq="5min", tz="UTC")
        base = pd.DataFrame(
            {
                "BTCUSDT_close": [float(i) for i in range(25)],
                "BNBUSDT_close": [float(i + 100) for i in range(25)],
            },
            index=base_index,
        )
        recent_index = pd.date_range("2026-04-10 12:00", periods=2, freq="5min", tz="UTC")
        recent_btc = pd.DataFrame({"close": [2.1, 2.2]}, index=recent_index)
        recent_bnb = pd.DataFrame({"close": [4.1, 4.2]}, index=recent_index)

        with (
            patch("pairwise_regime_mixture_shadow_live.gp.load_all_pairs", return_value=base),
            patch(
                "pairwise_regime_mixture_shadow_live.gp.fetch_klines",
                side_effect=[recent_btc, recent_bnb],
            ),
            patch(
                "pairwise_regime_mixture_shadow_live.utc_now",
                return_value=datetime(2026, 4, 10, 12, 7, tzinfo=timezone.utc),
            ),
        ):
            df = load_live_frame(("BTCUSDT", "BNBUSDT"), refresh_live_data=True, recent_days=1)

        self.assertEqual(df.index.max().isoformat(), "2026-04-10T12:00:00+00:00")
        self.assertAlmostEqual(float(df.iloc[-1]["BTCUSDT_close"]), 2.1)
        self.assertAlmostEqual(float(df.iloc[-1]["BNBUSDT_close"]), 4.1)


if __name__ == "__main__":
    unittest.main()
