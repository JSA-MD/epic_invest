import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_fractal_market_os_campaign as campaign


class FractalCampaignRunnerTests(unittest.TestCase):
    def test_build_campaign_entry_extracts_key_metrics(self) -> None:
        summary = {
            "selected_candidate": {
                "candidate_kind": "fractal_tree",
                "tree_key": "seed-tree",
                "observation_mode": "imbalance",
                "label_horizon": "30m",
                "tree_depth": 3,
                "logic_depth": 2,
                "windows": {
                    "recent_2m": {"aggregate": {"worst_pair_avg_daily_return": 0.005, "worst_max_drawdown": -0.08}},
                    "recent_6m": {"aggregate": {"worst_pair_avg_daily_return": 0.0065, "worst_max_drawdown": -0.10}},
                    "full_4y": {"aggregate": {"worst_pair_avg_daily_return": 0.0045, "worst_max_drawdown": -0.12}},
                },
                "robustness": {
                    "wf_1": {"passed": True},
                    "stress_survival_rate_mean": 0.55,
                    "stress_survival_rate_min": 0.33,
                    "stress_survival_threshold": 0.67,
                    "latest_fold_stress_reserve_score": 120.0,
                },
                "validation": {
                    "profiles": {
                        "target_060": {"passed": False},
                        "progressive_improvement": {"passed": True},
                        "joint_repair_market_os": {"passed": False},
                        "pair_repair_1y": {"passed": True},
                        "final_oos": {"passed": True},
                    }
                },
                "repair_metrics": {
                    "repair_pair": "BNBUSDT",
                    "repair_pair_avg_daily_return": 0.0003,
                    "repair_pair_total_return": 0.05,
                    "repair_pair_max_drawdown": -0.12,
                    "positive_pair_count": 2,
                    "pair_count": 2,
                },
            }
        }
        pipeline = {
            "decision": {
                "status": "validation_gate_blocked",
                "validation_gate_passed": True,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": True,
                "stress_gate_passed": False,
                "ready_for_live": False,
                "ready_for_merge": False,
            }
        }

        entry = campaign.build_campaign_entry(
            seed=20260415,
            artifact_paths=campaign.build_artifact_paths("/tmp/fractal_campaign", 20260415),
            search_summary=summary,
            pipeline_report=pipeline,
            elapsed_seconds=9.5,
        )

        self.assertEqual(entry["seed"], 20260415)
        self.assertEqual(entry["observation_mode"], "imbalance")
        self.assertEqual(entry["label_horizon"], "30m")
        self.assertAlmostEqual(entry["recent_6m_worst_daily"], 0.0065)
        self.assertAlmostEqual(entry["full_4y_mdd"], 0.12)
        self.assertTrue(entry["gate_flags"]["validation_gate_passed"])
        self.assertTrue(entry["gate_flags"]["wf1_passed"])
        self.assertTrue(entry["pair_repair_1y_passed"])
        self.assertFalse(entry["final_hard_gate_passed"])
        self.assertEqual(entry["repair_pair"], "BNBUSDT")
        self.assertEqual(entry["gate_pass_count"], 3)

    def test_rank_campaign_entries_prefers_more_gate_progress(self) -> None:
        stronger = {
            "status": "ok",
            "seed": 1,
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": True,
            "pair_repair_1y_passed": True,
            "gate_flags": {
                "ready_for_merge": False,
                "stress_gate_passed": False,
                "market_os_gate_passed": True,
                "final_oos_audit_passed": True,
                "validation_gate_passed": True,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.003,
            "recent_6m_worst_daily": 0.004,
            "recent_1y_repair_pair_daily": 0.001,
            "recent_2m_worst_daily": 0.005,
            "stress_survival_mean": 0.55,
            "latest_fold_stress_reserve_score": 50.0,
            "full_4y_mdd": 0.12,
            "recent_6m_mdd": 0.10,
            "recent_2m_mdd": 0.08,
            "elapsed_seconds": 10.0,
        }
        weaker = {
            "status": "ok",
            "seed": 2,
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": False,
            "pair_repair_1y_passed": True,
            "gate_flags": {
                "ready_for_merge": False,
                "stress_gate_passed": False,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": False,
                "validation_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.005,
            "recent_6m_worst_daily": 0.007,
            "recent_1y_repair_pair_daily": 0.002,
            "recent_2m_worst_daily": 0.006,
            "stress_survival_mean": 0.70,
            "latest_fold_stress_reserve_score": 100.0,
            "full_4y_mdd": 0.20,
            "recent_6m_mdd": 0.15,
            "recent_2m_mdd": 0.10,
            "elapsed_seconds": 9.0,
        }

        ranked = campaign.rank_campaign_entries([weaker, stronger])

        self.assertEqual(ranked[0]["seed"], 1)

    def test_rank_campaign_entries_prefers_repair_hard_gate_over_stress_or_floor_only(self) -> None:
        repair_hard = {
            "status": "ok",
            "seed": 100,
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": False,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": False,
            "repair_hard_gate_passed": True,
            "gate_flags": {
                "ready_for_merge": False,
                "stress_gate_passed": False,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": False,
                "validation_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.001,
            "recent_6m_worst_daily": 0.003,
            "recent_1y_repair_pair_daily": 0.0005,
            "recent_2m_worst_daily": 0.004,
            "stress_survival_mean": 0.45,
            "latest_fold_stress_reserve_score": 20.0,
            "full_4y_mdd": 0.10,
            "recent_6m_mdd": 0.08,
            "recent_2m_mdd": 0.06,
            "elapsed_seconds": 6.0,
        }
        stress_only = {
            "status": "ok",
            "seed": 101,
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": True,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": True,
            "repair_hard_gate_passed": False,
            "gate_flags": {
                "ready_for_merge": False,
                "stress_gate_passed": True,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": False,
                "validation_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.008,
            "recent_6m_worst_daily": 0.009,
            "recent_1y_repair_pair_daily": 0.002,
            "recent_2m_worst_daily": 0.010,
            "stress_survival_mean": 0.70,
            "latest_fold_stress_reserve_score": 140.0,
            "full_4y_mdd": 0.15,
            "recent_6m_mdd": 0.12,
            "recent_2m_mdd": 0.10,
            "elapsed_seconds": 7.0,
        }
        floor_only = {
            "status": "ok",
            "seed": 102,
            "joint_repair_min_floor_passed": True,
            "joint_repair_market_os_passed": True,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": False,
            "repair_hard_gate_passed": False,
            "gate_flags": {
                "ready_for_merge": False,
                "stress_gate_passed": False,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": False,
                "validation_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.005,
            "recent_6m_worst_daily": 0.007,
            "recent_1y_repair_pair_daily": 0.0015,
            "recent_2m_worst_daily": 0.008,
            "stress_survival_mean": 0.58,
            "latest_fold_stress_reserve_score": 60.0,
            "full_4y_mdd": 0.13,
            "recent_6m_mdd": 0.11,
            "recent_2m_mdd": 0.09,
            "elapsed_seconds": 7.5,
        }

        ranked = campaign.rank_campaign_entries([stress_only, floor_only, repair_hard])

        self.assertEqual(ranked[0]["seed"], 100)

    def test_rank_campaign_entries_prefers_final_hard_gate_over_repair_hard_only(self) -> None:
        final_hard = {
            "status": "ok",
            "seed": 110,
            "final_hard_gate_passed": True,
            "repair_hard_gate_passed": True,
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": False,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": False,
            "gate_flags": {
                "ready_for_merge": False,
                "stress_gate_passed": True,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": False,
                "validation_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.0062,
            "recent_6m_worst_daily": 0.0065,
            "recent_1y_repair_pair_daily": 0.0008,
            "recent_2m_worst_daily": 0.005,
            "stress_survival_mean": 0.70,
            "stress_survival_min": 0.68,
            "latest_fold_stress_reserve_score": 60.0,
            "latest_fold_non_nominal_survival": 0.70,
            "latest_non_nominal_stress_reserve_score": 55.0,
            "full_4y_mdd": 0.14,
            "recent_6m_mdd": 0.12,
            "recent_2m_mdd": 0.08,
            "elapsed_seconds": 8.0,
        }
        repair_hard_only = {
            "status": "ok",
            "seed": 111,
            "final_hard_gate_passed": False,
            "repair_hard_gate_passed": True,
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": False,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": False,
            "gate_flags": {
                "ready_for_merge": False,
                "stress_gate_passed": False,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": True,
                "validation_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.0064,
            "recent_6m_worst_daily": 0.0068,
            "recent_1y_repair_pair_daily": 0.0015,
            "recent_2m_worst_daily": 0.006,
            "stress_survival_mean": 0.60,
            "stress_survival_min": 0.55,
            "latest_fold_stress_reserve_score": 120.0,
            "latest_fold_non_nominal_survival": 0.60,
            "latest_non_nominal_stress_reserve_score": 110.0,
            "full_4y_mdd": 0.12,
            "recent_6m_mdd": 0.10,
            "recent_2m_mdd": 0.07,
            "elapsed_seconds": 7.0,
        }

        ranked = campaign.rank_campaign_entries([repair_hard_only, final_hard])

        self.assertEqual(ranked[0]["seed"], 110)

    def test_build_campaign_report_counts_stress_ready_gate_and_keeps_it_first(self) -> None:
        repair_hard = {
            "status": "ok",
            "seed": 201,
            "observation_mode": "imbalance",
            "label_horizon": "4h",
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": False,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": False,
            "repair_hard_gate_passed": True,
            "gate_flags": {
                "ready_for_merge": False,
                "ready_for_live": False,
                "validation_gate_passed": False,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": False,
                "stress_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.004,
            "recent_6m_worst_daily": 0.006,
            "recent_1y_repair_pair_daily": 0.001,
            "recent_2m_worst_daily": 0.005,
            "stress_survival_mean": 0.60,
            "latest_fold_stress_reserve_score": 90.0,
            "full_4y_mdd": 0.11,
            "recent_6m_mdd": 0.09,
            "recent_2m_mdd": 0.07,
            "elapsed_seconds": 8.0,
        }
        stress_only = {
            "status": "ok",
            "seed": 202,
            "observation_mode": "imbalance",
            "label_horizon": "4h",
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": True,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": True,
            "repair_hard_gate_passed": False,
            "gate_flags": {
                "ready_for_merge": False,
                "ready_for_live": False,
                "validation_gate_passed": True,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": True,
                "stress_gate_passed": True,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.007,
            "recent_6m_worst_daily": 0.008,
            "recent_1y_repair_pair_daily": 0.002,
            "recent_2m_worst_daily": 0.006,
            "stress_survival_mean": 0.55,
            "latest_fold_stress_reserve_score": 40.0,
            "full_4y_mdd": 0.14,
            "recent_6m_mdd": 0.12,
            "recent_2m_mdd": 0.10,
            "elapsed_seconds": 7.0,
        }
        floor_only = {
            "status": "ok",
            "seed": 203,
            "observation_mode": "imbalance",
            "label_horizon": "4h",
            "joint_repair_min_floor_passed": True,
            "joint_repair_market_os_passed": True,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": False,
            "repair_hard_gate_passed": False,
            "gate_flags": {
                "ready_for_merge": False,
                "ready_for_live": False,
                "validation_gate_passed": True,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": True,
                "stress_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.007,
            "recent_6m_worst_daily": 0.008,
            "recent_1y_repair_pair_daily": 0.002,
            "recent_2m_worst_daily": 0.006,
            "stress_survival_mean": 0.55,
            "latest_fold_stress_reserve_score": 40.0,
            "full_4y_mdd": 0.14,
            "recent_6m_mdd": 0.12,
            "recent_2m_mdd": 0.10,
            "elapsed_seconds": 7.2,
        }

        report = campaign.build_campaign_report(
            config={"seeds": [201, 202, 203]},
            entries=[stress_only, floor_only, repair_hard],
            top_n_report=2,
        )

        self.assertEqual(report["best_candidate"]["seed"], 201)
        self.assertEqual(report["gate_counts"]["stress_gate_passed"], 1)
        self.assertEqual(report["gate_counts"]["joint_repair_min_floor_passed"], 1)
        self.assertEqual(report["gate_counts"]["joint_repair_market_os_passed"], 2)
        self.assertEqual(report["gate_counts"]["joint_repair_stress_passed"], 1)
        self.assertEqual(report["gate_counts"]["repair_hard_gate_passed"], 1)

    def test_build_campaign_report_counts_final_hard_gate(self) -> None:
        final_hard = {
            "status": "ok",
            "seed": 301,
            "observation_mode": "imbalance",
            "label_horizon": "4h",
            "final_hard_gate_passed": True,
            "repair_hard_gate_passed": True,
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": False,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": False,
            "gate_flags": {
                "ready_for_merge": False,
                "ready_for_live": False,
                "validation_gate_passed": False,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": False,
                "stress_gate_passed": True,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.0062,
            "recent_6m_worst_daily": 0.0064,
            "recent_1y_repair_pair_daily": 0.001,
            "recent_2m_worst_daily": 0.005,
            "stress_survival_mean": 0.70,
            "stress_survival_min": 0.68,
            "latest_fold_stress_reserve_score": 80.0,
            "latest_fold_non_nominal_survival": 0.70,
            "latest_non_nominal_stress_reserve_score": 75.0,
            "full_4y_mdd": 0.14,
            "recent_6m_mdd": 0.11,
            "recent_2m_mdd": 0.07,
            "elapsed_seconds": 8.0,
        }
        repair_hard_only = {
            "status": "ok",
            "seed": 302,
            "observation_mode": "time",
            "label_horizon": "4h",
            "final_hard_gate_passed": False,
            "repair_hard_gate_passed": True,
            "joint_repair_min_floor_passed": False,
            "joint_repair_market_os_passed": False,
            "pair_repair_1y_passed": True,
            "joint_repair_stress_passed": False,
            "gate_flags": {
                "ready_for_merge": False,
                "ready_for_live": False,
                "validation_gate_passed": False,
                "market_os_gate_passed": False,
                "final_oos_audit_passed": False,
                "stress_gate_passed": False,
                "wf1_passed": True,
            },
            "full_4y_worst_daily": 0.0048,
            "recent_6m_worst_daily": 0.0060,
            "recent_1y_repair_pair_daily": 0.0012,
            "recent_2m_worst_daily": 0.0052,
            "stress_survival_mean": 0.55,
            "stress_survival_min": 0.40,
            "latest_fold_stress_reserve_score": 10.0,
            "latest_fold_non_nominal_survival": 0.55,
            "latest_non_nominal_stress_reserve_score": 8.0,
            "full_4y_mdd": 0.13,
            "recent_6m_mdd": 0.10,
            "recent_2m_mdd": 0.07,
            "elapsed_seconds": 8.5,
        }

        report = campaign.build_campaign_report(
            config={"seeds": [301, 302]},
            entries=[repair_hard_only, final_hard],
            top_n_report=2,
        )

        self.assertEqual(report["best_candidate"]["seed"], 301)
        self.assertEqual(report["gate_counts"]["final_hard_gate_passed"], 1)
        self.assertEqual(report["gate_counts"]["repair_hard_gate_passed"], 2)

    def test_build_campaign_report_groups_best_by_mode_and_horizon(self) -> None:
        entries = [
            {
                "status": "ok",
                "seed": 11,
                "observation_mode": "volume",
                "label_horizon": "30m",
                "joint_repair_min_floor_passed": False,
                "joint_repair_market_os_passed": False,
                "pair_repair_1y_passed": False,
                "joint_repair_stress_passed": False,
                "repair_hard_gate_passed": False,
                "gate_flags": {
                    "ready_for_merge": False,
                    "ready_for_live": False,
                    "validation_gate_passed": True,
                    "market_os_gate_passed": False,
                    "final_oos_audit_passed": True,
                    "stress_gate_passed": False,
                    "wf1_passed": True,
                },
                "full_4y_worst_daily": 0.003,
                "recent_6m_worst_daily": 0.006,
                "recent_1y_repair_pair_daily": -0.0002,
                "recent_2m_worst_daily": 0.005,
                "stress_survival_mean": 0.50,
                "latest_fold_stress_reserve_score": 20.0,
                "full_4y_mdd": 0.12,
                "recent_6m_mdd": 0.10,
                "recent_2m_mdd": 0.09,
                "elapsed_seconds": 12.0,
            },
            {
                "status": "ok",
                "seed": 12,
                "observation_mode": "imbalance",
                "label_horizon": "4h",
                "joint_repair_min_floor_passed": False,
                "joint_repair_market_os_passed": True,
                "pair_repair_1y_passed": True,
                "joint_repair_stress_passed": True,
                "repair_hard_gate_passed": False,
                "gate_flags": {
                    "ready_for_merge": False,
                    "ready_for_live": False,
                    "validation_gate_passed": False,
                    "market_os_gate_passed": False,
                    "final_oos_audit_passed": False,
                    "stress_gate_passed": False,
                    "wf1_passed": True,
                },
                "full_4y_worst_daily": 0.002,
                "recent_6m_worst_daily": 0.004,
                "recent_1y_repair_pair_daily": 0.003,
                "recent_2m_worst_daily": 0.004,
                "stress_survival_mean": 0.40,
                "latest_fold_stress_reserve_score": 10.0,
                "full_4y_mdd": 0.11,
                "recent_6m_mdd": 0.09,
                "recent_2m_mdd": 0.08,
                "elapsed_seconds": 11.0,
            },
            {
                "status": "failed",
                "seed": 13,
                "error": "boom",
            },
        ]

        report = campaign.build_campaign_report(
            config={"seeds": [11, 12, 13]},
            entries=entries,
            top_n_report=2,
        )

        self.assertEqual(report["completed_count"], 2)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["best_candidate"]["seed"], 12)
        self.assertEqual(report["best_by_observation_mode"]["volume"]["seed"], 11)
        self.assertEqual(report["best_by_label_horizon"]["4h"]["seed"], 12)
        self.assertEqual(report["gate_counts"]["joint_repair_min_floor_passed"], 0)
        self.assertEqual(report["gate_counts"]["joint_repair_market_os_passed"], 1)
        self.assertEqual(report["gate_counts"]["joint_repair_stress_passed"], 1)
        self.assertEqual(report["gate_counts"]["repair_hard_gate_passed"], 0)


if __name__ == "__main__":
    unittest.main()
