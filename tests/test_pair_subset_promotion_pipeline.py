import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_pair_subset_promotion_pipeline as pipeline


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


class PairSubsetPromotionPipelineTests(unittest.TestCase):
    def test_pairwise_mode_defaults_seed_multiple_candidate_sources(self) -> None:
        with patch.object(sys, "argv", ["run_pair_subset_promotion_pipeline.py"]):
            args = pipeline.parse_args()

        paths = pipeline.resolve_mode_paths(args)

        self.assertIn("gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json", paths["candidate_summaries"])
        self.assertIn("gp_regime_mixture_btc_bnb_pairwise_repair_summary.json", paths["candidate_summaries"])
        self.assertIn("gp_regime_mixture_btc_bnb_pairwise_nsga3_summary.json", paths["candidate_summaries"])

    def test_pairwise_mode_aggregates_validation_engine_and_stress_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            search_summary_out = tmp / "pairwise_validated_summary.json"
            validation_report_out = tmp / "pairwise_validation_report.json"
            stress_report_out = tmp / "pairwise_stress_report.json"
            pipeline_report_out = tmp / "pairwise_pipeline_report.json"

            write_json(
                search_summary_out,
                {
                    "pairs": ["BTCUSDT", "BNBUSDT"],
                    "selection": {"reason": "target_060_plus_validation"},
                    "selected_candidate": {
                        "route_breadth_threshold": 0.65,
                        "mapping_indices": [1, 2, 3, 4],
                        "pair_configs": {
                            "BTCUSDT": {"route_breadth_threshold": 0.65, "mapping_indices": [1, 2, 3, 4]},
                            "BNBUSDT": {"route_breadth_threshold": 0.65, "mapping_indices": [1, 2, 3, 4]},
                        },
                        "candidate_id": "pairwise-1",
                        "validation": {
                            "profiles": {
                                "final_oos": {
                                    "passed": True,
                                    "checks": [{"name": "recent_2m_worst_pair_positive", "passed": True}],
                                }
                            }
                        },
                        "validation_engine": {
                            "gate": {"passed": True, "failed_checks": []},
                            "market_operating_system": {
                                "gate": {"passed": True, "failed_checks": []},
                                "audit": {"passed": True, "failed_checks": []},
                            },
                        },
                    },
                },
            )
            write_json(
                stress_report_out,
                {
                    "promotion_decision": {
                        "selected_candidate_ready_for_merge": True,
                        "selected_passes_progressive_stress": True,
                        "selected_passes_target_060_stress": True,
                        "status": "target_060_stress_pass",
                    }
                },
            )

            args = Namespace(
                pipeline_mode=pipeline.PIPELINE_MODE_PAIRWISE_MARKET_OS,
                pairs="BTCUSDT,BNBUSDT",
                candidate_summaries=str(tmp / "pairwise_candidates.json"),
                baseline_summary=str(tmp / "pairwise_baseline.json"),
                search_summary_out=search_summary_out,
                validation_report_out=validation_report_out,
                stress_report_out=stress_report_out,
                pipeline_report_out=pipeline_report_out,
                skip_search=True,
                skip_validation=False,
                skip_stress=False,
            )

            with (
                patch.object(pipeline, "parse_args", return_value=args),
                patch.object(pipeline, "run_step") as run_step,
            ):
                pipeline.main()

            report = json.loads(pipeline_report_out.read_text())
            validation_report = json.loads(validation_report_out.read_text())

            self.assertEqual(report["pipeline_mode"], pipeline.PIPELINE_MODE_PAIRWISE_MARKET_OS)
            self.assertTrue(report["validation_gate"]["passed"])
            self.assertTrue(report["market_os_gate"]["passed"])
            self.assertTrue(report["final_oos_audit"]["passed"])
            self.assertTrue(report["stress_gate"]["passed"])
            self.assertTrue(report["ready_for_live"])
            self.assertTrue(report["ready_for_merge"])
            self.assertEqual(report["status"], "ready_for_live")
            self.assertEqual(validation_report["validation_engine"]["gate"]["passed"], True)
            self.assertEqual(validation_report["market_os_gate"]["passed"], True)
            self.assertEqual(run_step.call_count, 1)

    def test_legacy_mode_skip_search_keeps_compatibility_report_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            search_summary_out = tmp / "legacy_search_summary.json"
            validation_report_out = tmp / "legacy_validation_report.json"
            stress_report_out = tmp / "legacy_stress_report.json"
            pipeline_report_out = tmp / "legacy_pipeline_report.json"

            write_json(
                search_summary_out,
                {
                    "pairs": ["BTCUSDT", "BNBUSDT"],
                    "selection": {"reason": "target_060_pass"},
                    "selected_candidate": {
                        "route_breadth_threshold": 0.5,
                        "mapping_indices": [0, 1, 2, 3],
                    },
                },
            )
            write_json(
                validation_report_out,
                {
                    "pairs": ["BTCUSDT", "BNBUSDT"],
                    "summary_path": str(search_summary_out),
                    "selected_candidate": {"route_breadth_threshold": 0.5, "mapping_indices": [0, 1, 2, 3]},
                    "profiles": {
                        "progressive_improvement": {
                            "selected": {"passed": True, "checks": [{"name": "progressive", "passed": True}]}
                        },
                        "final_oos": {
                            "selected": {"passed": True, "checks": [{"name": "final_oos", "passed": True}]}
                        },
                    },
                },
            )
            write_json(
                stress_report_out,
                {
                    "promotion_decision": {
                        "selected_candidate_ready_for_merge": True,
                        "status": "target_060_stress_pass",
                    }
                },
            )

            args = Namespace(
                pipeline_mode=pipeline.PIPELINE_MODE_LEGACY_SHARED,
                pairs="BTCUSDT,BNBUSDT",
                candidate_summaries=None,
                baseline_summary=None,
                search_summary_out=search_summary_out,
                validation_report_out=validation_report_out,
                stress_report_out=stress_report_out,
                pipeline_report_out=pipeline_report_out,
                skip_search=True,
                skip_validation=False,
                skip_stress=False,
            )

            with (
                patch.object(pipeline, "parse_args", return_value=args),
                patch.object(pipeline, "run_step") as run_step,
            ):
                pipeline.main()

            report = json.loads(pipeline_report_out.read_text())

            self.assertEqual(report["pipeline_mode"], pipeline.PIPELINE_MODE_LEGACY_SHARED)
            self.assertTrue(report["validation_gate"]["passed"])
            self.assertTrue(report["final_oos_audit"]["passed"])
            self.assertTrue(report["stress_gate"]["passed"])
            self.assertTrue(report["ready_for_live"])
            self.assertTrue(report["ready_for_merge"])
            self.assertEqual(report["status"], "ready_for_live")
            self.assertIsNone(report["market_os_gate"])
            self.assertEqual(run_step.call_count, 2)

    def test_pairwise_mode_surfaces_shadow_ready_only_when_target_stress_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            search_summary_out = tmp / "pairwise_validated_summary.json"
            validation_report_out = tmp / "pairwise_validation_report.json"
            stress_report_out = tmp / "pairwise_stress_report.json"
            pipeline_report_out = tmp / "pairwise_pipeline_report.json"

            write_json(
                search_summary_out,
                {
                    "pairs": ["BTCUSDT", "BNBUSDT"],
                    "selection": {"reason": "progressive_plus_validation"},
                    "selected_candidate": {
                        "pair_configs": {
                            "BTCUSDT": {"route_breadth_threshold": 0.5, "mapping_indices": [1, 2, 3, 4]},
                            "BNBUSDT": {"route_breadth_threshold": 0.5, "mapping_indices": [1, 2, 3, 4]},
                        },
                        "candidate_id": "pairwise-shadow",
                        "validation": {
                            "profiles": {
                                "final_oos": {
                                    "passed": True,
                                    "checks": [{"name": "recent_2m_worst_pair_positive", "passed": True}],
                                }
                            }
                        },
                        "validation_engine": {
                            "gate": {"passed": True, "failed_checks": []},
                            "market_operating_system": {
                                "gate": {"passed": True, "failed_checks": []},
                                "audit": {"passed": True, "failed_checks": []},
                            },
                        },
                    },
                },
            )
            write_json(
                stress_report_out,
                {
                    "promotion_decision": {
                        "ready_for_live": True,
                        "ready_for_merge": False,
                        "selected_candidate_ready_for_merge": False,
                        "status": "ready_for_live",
                    }
                },
            )

            args = Namespace(
                pipeline_mode=pipeline.PIPELINE_MODE_PAIRWISE_MARKET_OS,
                pairs="BTCUSDT,BNBUSDT",
                candidate_summaries=None,
                baseline_summary=None,
                search_summary_out=search_summary_out,
                validation_report_out=validation_report_out,
                stress_report_out=stress_report_out,
                pipeline_report_out=pipeline_report_out,
                skip_search=True,
                skip_validation=False,
                skip_stress=False,
            )

            with (
                patch.object(pipeline, "parse_args", return_value=args),
                patch.object(pipeline, "run_step") as run_step,
            ):
                pipeline.main()

            report = json.loads(pipeline_report_out.read_text())

            self.assertTrue(report["ready_for_shadow_live"])
            self.assertFalse(report["ready_for_live"])
            self.assertFalse(report["ready_for_merge"])
            self.assertEqual(report["status"], "shadow_ready_only")
            self.assertTrue(report["stress_gate"]["ready_for_shadow_live"])
            self.assertFalse(report["stress_gate"]["ready_for_merge"])
            self.assertEqual(run_step.call_count, 1)


if __name__ == "__main__":
    unittest.main()
