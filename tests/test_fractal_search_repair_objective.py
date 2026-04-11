import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_pair_subset_fractal_genome as fractal
import validate_pair_subset_summary as summary_validation
from fractal_genome_core import LeafNode


def make_window(
    btc_daily: float,
    bnb_daily: float,
    *,
    btc_total: float,
    bnb_total: float,
    btc_mdd: float,
    bnb_mdd: float,
    btc_win_rate: float = 0.52,
    bnb_win_rate: float = 0.52,
    btc_trades: int = 20,
    bnb_trades: int = 20,
) -> dict[str, object]:
    per_pair = {
        "BTCUSDT": {
            "avg_daily_return": btc_daily,
            "total_return": btc_total,
            "max_drawdown": btc_mdd,
            "daily_win_rate": btc_win_rate,
            "n_trades": btc_trades,
        },
        "BNBUSDT": {
            "avg_daily_return": bnb_daily,
            "total_return": bnb_total,
            "max_drawdown": bnb_mdd,
            "daily_win_rate": bnb_win_rate,
            "n_trades": bnb_trades,
        },
    }
    worst_daily = min(btc_daily, bnb_daily)
    worst_total = min(btc_total, bnb_total)
    worst_mdd = min(btc_mdd, bnb_mdd)
    return {
        "per_pair": per_pair,
        "aggregate": {
            "mean_avg_daily_return": (btc_daily + bnb_daily) / 2.0,
            "worst_pair_avg_daily_return": worst_daily,
            "worst_pair_total_return": worst_total,
            "mean_total_return": (btc_total + bnb_total) / 2.0,
            "worst_max_drawdown": worst_mdd,
            "pair_return_dispersion": abs(btc_daily - bnb_daily),
            "positive_pair_count": int(btc_daily >= 0.0) + int(bnb_daily >= 0.0),
        },
    }


def make_robustness(
    *,
    latest_fold_delta_worst_pair_total_return: float = 0.0,
    latest_fold_delta_worst_max_drawdown: float = 0.0,
    latest_fold_delta_worst_daily_win_rate: float = 0.0,
    latest_fold_stress_reserve_score: float = 0.0,
    latest_non_nominal_stress_reserve_score: float = 0.0,
    stress_survival_rate_mean: float = 0.67,
    stress_survival_rate_min: float = 0.67,
    non_nominal_stress_survival_rate_mean: float = 0.67,
    min_fold_non_nominal_stress_survival_rate: float = 0.67,
    latest_non_nominal_stress_survival_rate: float = 0.67,
    stress_survival_threshold: float = 0.67,
    fold_pass_rate: float = 1.0,
    latest_fold_trade_count_ratio: float = 0.5,
    mean_fold_trade_count_ratio: float = 0.5,
) -> dict[str, float]:
    return {
        "worst_fold_delta_worst_pair_avg_daily_return": 0.0,
        "worst_fold_delta_worst_pair_total_return": 0.0,
        "worst_fold_delta_worst_max_drawdown": 0.0,
        "worst_fold_delta_worst_daily_win_rate": 0.0,
        "latest_fold_delta_worst_pair_avg_daily_return": 0.0,
        "latest_fold_delta_mean_avg_daily_return": 0.0,
        "latest_fold_delta_worst_pair_total_return": latest_fold_delta_worst_pair_total_return,
        "latest_fold_delta_worst_max_drawdown": latest_fold_delta_worst_max_drawdown,
        "latest_fold_delta_worst_daily_win_rate": latest_fold_delta_worst_daily_win_rate,
        "latest_fold_stress_reserve_score": latest_fold_stress_reserve_score,
        "latest_non_nominal_stress_reserve_score": latest_non_nominal_stress_reserve_score,
        "stress_survival_rate_mean": stress_survival_rate_mean,
        "stress_survival_rate_min": stress_survival_rate_min,
        "non_nominal_stress_survival_rate_mean": non_nominal_stress_survival_rate_mean,
        "min_fold_non_nominal_stress_survival_rate": min_fold_non_nominal_stress_survival_rate,
        "latest_non_nominal_stress_survival_rate": latest_non_nominal_stress_survival_rate,
        "stress_survival_threshold": stress_survival_threshold,
        "fold_pass_rate": fold_pass_rate,
        "latest_fold_trade_count_ratio": latest_fold_trade_count_ratio,
        "mean_fold_trade_count_ratio": mean_fold_trade_count_ratio,
    }


def get_repair_hard_gate_helper():
    return getattr(fractal, "candidate_repair_hard_gate_pass", None) or getattr(fractal, "repair_hard_gate_passed", None)


def get_final_hard_gate_helper():
    return getattr(fractal, "candidate_final_hard_gate_pass", None) or getattr(fractal, "final_hard_gate_passed", None)


class FractalSearchRepairObjectiveTests(unittest.TestCase):
    def test_compute_dynamic_windows_includes_recent_1y(self) -> None:
        index = fractal.pd.date_range("2025-01-01", "2026-04-10 23:55:00", freq="5min", tz="UTC")
        windows = fractal.compute_dynamic_windows(fractal.pd.DatetimeIndex(index))
        keys = [item["key"] for item in windows]
        self.assertIn("recent_1y", keys)

    def test_build_pair_repair_metrics_tracks_bnb_gap(self) -> None:
        windows = {
            "recent_1y": make_window(
                0.0040,
                -0.0002,
                btc_total=1.0,
                bnb_total=-0.05,
                btc_mdd=-0.10,
                bnb_mdd=-0.12,
            )
        }

        metrics = fractal.build_pair_repair_metrics(windows, "BNBUSDT")

        self.assertEqual(metrics["repair_pair"], "BNBUSDT")
        self.assertEqual(metrics["positive_pair_count"], 1)
        self.assertLess(metrics["repair_pair_avg_daily_return"], 0.0)

    def test_load_warm_start_trees_imports_matching_mode_and_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "warm_summary.json"
            payload = {
                "selected_candidate": {
                    "observation_mode": "imbalance",
                    "label_horizon": "4h",
                    "tree": fractal.serialize_tree(LeafNode(1)),
                },
                "top_candidates": [
                    {
                        "observation_mode": "imbalance",
                        "label_horizon": "4h",
                        "tree": fractal.serialize_tree(LeafNode(2)),
                    },
                    {
                        "observation_mode": "time",
                        "label_horizon": "4h",
                        "tree": fractal.serialize_tree(LeafNode(3)),
                    },
                ],
            }
            path.write_text(json.dumps(payload))

            trees = fractal.load_warm_start_trees(
                [path],
                observation_mode="imbalance",
                label_horizon="4h",
                limit=8,
            )

            self.assertEqual([fractal.tree_key(tree) for tree in trees], [fractal.tree_key(LeafNode(1)), fractal.tree_key(LeafNode(2))])

    def test_build_local_variant_population_keeps_incumbent_and_expands_neighbors(self) -> None:
        feature_specs = fractal.build_feature_specs(("BTCUSDT", "BNBUSDT"), observation_mode="time")
        condition_options = fractal.build_condition_options(feature_specs)
        base_tree = LeafNode(0)

        variants = fractal.build_local_variant_population(
            [base_tree],
            rng=random.Random(7),
            condition_options=condition_options,
            expert_count=4,
            max_depth=2,
            logic_max_depth=1,
            variant_budget=6,
        )

        variant_keys = [fractal.tree_key(tree) for tree in variants]
        self.assertIn(fractal.tree_key(base_tree), variant_keys)
        self.assertGreaterEqual(len(variants), 2)

    def test_load_warm_start_trees_prioritizes_repair_hard_lineage(self) -> None:
        def make_candidate(tree_value: int, *, passed: bool) -> dict[str, object]:
            windows = {
                "recent_1y": make_window(
                    0.0040,
                    0.0004 if passed else -0.0002,
                    btc_total=1.20,
                    bnb_total=0.08 if passed else -0.04,
                    btc_mdd=-0.10,
                    bnb_mdd=-0.11 if passed else -0.12,
                )
            }
            return {
                "observation_mode": "imbalance",
                "label_horizon": "4h",
                "tree": fractal.serialize_tree(LeafNode(tree_value)),
                "validation": {
                    "profiles": {
                        "pair_repair_1y": {
                            "passed": passed,
                            "repair_pair": "BNBUSDT",
                        }
                    }
                },
                "repair_metrics": fractal.build_pair_repair_metrics(windows, "BNBUSDT"),
                "robustness": make_robustness(),
                "windows": windows,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "warm_summary.json"
            payload = {
                "selected_candidate": make_candidate(1, passed=False),
                "top_candidates": [
                    make_candidate(2, passed=True),
                    make_candidate(3, passed=True),
                    make_candidate(4, passed=False),
                ],
            }
            path.write_text(json.dumps(payload))

            trees = fractal.load_warm_start_trees(
                [path],
                observation_mode="imbalance",
                label_horizon="4h",
                limit=8,
            )

            keys = [fractal.tree_key(tree) for tree in trees]
            self.assertEqual(keys[:2], [fractal.tree_key(LeafNode(2)), fractal.tree_key(LeafNode(3))])
            self.assertEqual(len(keys), 2)

    def test_build_exploit_pool_preserves_repair_hard_majority(self) -> None:
        def make_item(tree_value: int, *, passed: bool) -> dict[str, object]:
            windows = {
                "recent_1y": make_window(
                    0.0040,
                    0.0004 if passed else -0.0002,
                    btc_total=1.20,
                    bnb_total=0.08 if passed else -0.04,
                    btc_mdd=-0.10,
                    bnb_mdd=-0.11 if passed else -0.12,
                )
            }
            return {
                "tree": LeafNode(tree_value),
                "tree_key": fractal.tree_key(LeafNode(tree_value)),
                "validation": {
                    "profiles": {
                        "pair_repair_1y": {
                            "passed": passed,
                            "repair_pair": "BNBUSDT",
                        }
                    }
                },
                "repair_metrics": fractal.build_pair_repair_metrics(windows, "BNBUSDT"),
                "robustness": make_robustness(),
                "windows": windows,
            }

        parent_pool = [
            make_item(1, passed=True),
            make_item(2, passed=True),
            make_item(3, passed=False),
            make_item(4, passed=False),
            make_item(5, passed=False),
            make_item(6, passed=False),
        ]

        exploit_pool = fractal.build_exploit_pool(parent_pool)

        repair_count = sum(1 for item in exploit_pool if get_repair_hard_gate_helper()(item))
        non_repair_count = len(exploit_pool) - repair_count
        self.assertEqual(repair_count, 2)
        self.assertEqual(non_repair_count, 0)

    def test_build_exploit_pool_falls_back_to_parent_pool_when_no_repair_hard_candidate_exists(self) -> None:
        def make_item(tree_value: int, *, stronger: bool) -> dict[str, object]:
            windows = {
                "recent_2m": make_window(
                    0.0045,
                    -0.0001,
                    btc_total=0.30,
                    bnb_total=-0.02,
                    btc_mdd=-0.10,
                    bnb_mdd=-0.12,
                ),
                "recent_1y": make_window(
                    0.0040,
                    -0.0002,
                    btc_total=1.20,
                    bnb_total=-0.04,
                    btc_mdd=-0.10,
                    bnb_mdd=-0.12,
                ),
                "recent_6m": make_window(
                    0.0062 if stronger else 0.0055,
                    0.0061 if stronger else 0.0054,
                    btc_total=1.10,
                    bnb_total=1.05,
                    btc_mdd=-0.11,
                    bnb_mdd=-0.11,
                ),
                "full_4y": make_window(
                    0.0046 if stronger else 0.0038,
                    0.0045 if stronger else 0.0037,
                    btc_total=5.40,
                    bnb_total=5.30,
                    btc_mdd=-0.12,
                    bnb_mdd=-0.12,
                ),
            }
            robustness = make_robustness(
                latest_fold_delta_worst_pair_total_return=0.10 if stronger else -0.10,
                latest_fold_delta_worst_max_drawdown=0.01 if stronger else -0.01,
                latest_fold_delta_worst_daily_win_rate=0.01 if stronger else -0.01,
            )
            return {
                "tree": LeafNode(tree_value),
                "tree_key": fractal.tree_key(LeafNode(tree_value)),
                "validation": summary_validation.build_validation_bundle(windows, windows, repair_pair="BNBUSDT"),
                "repair_metrics": fractal.build_pair_repair_metrics(windows, "BNBUSDT"),
                "robustness": robustness,
                "windows": windows,
            }

        parent_pool = [
            make_item(1, stronger=True),
            make_item(2, stronger=False),
            make_item(3, stronger=False),
        ]

        exploit_pool = fractal.build_exploit_pool(parent_pool)

        self.assertEqual(
            [item["tree_key"] for item in exploit_pool],
            [fractal.tree_key(LeafNode(1)), fractal.tree_key(LeafNode(2)), fractal.tree_key(LeafNode(3))],
        )

    def test_repair_objective_lifts_candidate_with_better_bnb_one_year_profile(self) -> None:
        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0050, 0.0050, btc_total=0.90, bnb_total=0.90, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0046, 0.0046, btc_total=5.0, bnb_total=5.0, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        negative_bnb = {
            **baseline,
            "recent_1y": make_window(0.0040, -0.0002, btc_total=1.20, bnb_total=-0.05, btc_mdd=-0.10, bnb_mdd=-0.12),
        }
        repaired_bnb = {
            **baseline,
            "recent_1y": make_window(0.0040, 0.0003, btc_total=1.20, bnb_total=0.08, btc_mdd=-0.10, bnb_mdd=-0.11),
        }

        bad_score, _ = fractal.fractal_fast_scalar_score(
            negative_bnb,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            make_robustness(),
            0.0,
            repair_pair="BNBUSDT",
        )
        repaired_score, _ = fractal.fractal_fast_scalar_score(
            repaired_bnb,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            make_robustness(),
            0.0,
            repair_pair="BNBUSDT",
        )

        self.assertGreater(repaired_score, bad_score)

    def test_joint_objective_prefers_repair_with_wf1_and_long_horizon_support(self) -> None:
        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0060, 0.0060, btc_total=0.90, bnb_total=0.90, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0046, 0.0046, btc_total=5.0, bnb_total=5.0, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        repair_only = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0038, 0.0039, btc_total=3.8, bnb_total=3.9, btc_mdd=-0.16, bnb_mdd=-0.16),
        }
        balanced = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0048, 0.0047, btc_total=5.2, bnb_total=5.1, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        weak_robustness = make_robustness()
        weak_robustness["latest_fold_delta_worst_pair_total_return"] = -0.01
        strong_robustness = make_robustness()

        repair_only_score, _ = fractal.fractal_fast_scalar_score(
            repair_only,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            weak_robustness,
            0.0,
            repair_pair="BNBUSDT",
        )
        balanced_score, _ = fractal.fractal_fast_scalar_score(
            balanced,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            strong_robustness,
            0.0,
            repair_pair="BNBUSDT",
        )

        self.assertGreater(balanced_score, repair_only_score)

    def test_candidate_joint_repair_stress_pass_requires_repair_wf1_long_horizon_stress_and_cost_reserve(self) -> None:
        helper = getattr(fractal, "candidate_joint_repair_stress_pass", None)
        if helper is None:
            self.skipTest("candidate_joint_repair_stress_pass is not implemented yet")

        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0061, 0.0062, btc_total=0.90, bnb_total=0.95, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0048, 0.0047, btc_total=5.2, bnb_total=5.1, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        candidate = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
        }
        stress_ready = make_robustness(
            latest_fold_stress_reserve_score=0.25,
            latest_non_nominal_stress_reserve_score=0.20,
            stress_survival_rate_mean=0.72,
            stress_survival_rate_min=0.70,
            non_nominal_stress_survival_rate_mean=0.71,
            min_fold_non_nominal_stress_survival_rate=0.70,
            latest_non_nominal_stress_survival_rate=0.71,
        )
        candidate_item = {
            "windows": candidate,
            "validation": summary_validation.build_validation_bundle(candidate, baseline, repair_pair="BNBUSDT"),
            "robustness": stress_ready,
        }

        self.assertTrue(fractal.candidate_joint_repair_balance_pass(candidate_item))
        self.assertTrue(fractal.candidate_wf1_pass(candidate_item, tolerance=1e-12))
        self.assertTrue(fractal.candidate_stress_pass({"robustness": stress_ready}))
        self.assertTrue(fractal.candidate_cost_reserve_pass({"robustness": stress_ready}))
        self.assertTrue(helper(candidate_item))

        cases = [
            (
                "repair",
                {**candidate, "recent_1y": make_window(0.0038, -0.0002, btc_total=1.10, bnb_total=-0.05, btc_mdd=-0.10, bnb_mdd=-0.11)},
                stress_ready,
            ),
            (
                "wf1",
                candidate,
                make_robustness(
                    latest_fold_delta_worst_pair_total_return=-0.01,
                    latest_fold_delta_worst_max_drawdown=-0.01,
                    latest_fold_delta_worst_daily_win_rate=-0.01,
                    latest_fold_stress_reserve_score=0.25,
                    latest_non_nominal_stress_reserve_score=0.20,
                    stress_survival_rate_mean=0.72,
                    stress_survival_rate_min=0.70,
                    non_nominal_stress_survival_rate_mean=0.71,
                    min_fold_non_nominal_stress_survival_rate=0.70,
                    latest_non_nominal_stress_survival_rate=0.71,
                ),
            ),
            (
                "long_horizon",
                {**candidate, "recent_6m": make_window(0.0050, 0.0051, btc_total=0.90, bnb_total=0.95, btc_mdd=-0.11, bnb_mdd=-0.11)},
                stress_ready,
            ),
            (
                "stress",
                candidate,
                make_robustness(
                    latest_fold_stress_reserve_score=-0.10,
                    latest_non_nominal_stress_reserve_score=0.20,
                    stress_survival_rate_mean=0.60,
                    stress_survival_rate_min=0.60,
                    non_nominal_stress_survival_rate_mean=0.60,
                    min_fold_non_nominal_stress_survival_rate=0.60,
                    latest_non_nominal_stress_survival_rate=0.60,
                ),
            ),
            (
                "stress_min",
                candidate,
                make_robustness(
                    latest_fold_stress_reserve_score=0.25,
                    latest_non_nominal_stress_reserve_score=0.20,
                    stress_survival_rate_mean=0.72,
                    stress_survival_rate_min=0.0,
                    non_nominal_stress_survival_rate_mean=0.71,
                    min_fold_non_nominal_stress_survival_rate=0.70,
                    latest_non_nominal_stress_survival_rate=0.71,
                ),
            ),
            (
                "full_4y_floor",
                {
                    **candidate,
                    "full_4y": make_window(0.0042, 0.0043, btc_total=5.0, bnb_total=5.0, btc_mdd=-0.12, bnb_mdd=-0.12),
                },
                stress_ready,
            ),
            (
                "cost_reserve",
                candidate,
                make_robustness(
                    latest_fold_stress_reserve_score=0.25,
                    latest_non_nominal_stress_reserve_score=-0.10,
                    stress_survival_rate_mean=0.72,
                    stress_survival_rate_min=0.70,
                    non_nominal_stress_survival_rate_mean=0.60,
                    min_fold_non_nominal_stress_survival_rate=0.60,
                    latest_non_nominal_stress_survival_rate=0.60,
                ),
            ),
        ]

        for name, mutated_candidate, mutated_robustness in cases:
            with self.subTest(name=name):
                mutated_item = {
                    "windows": mutated_candidate,
                    "validation": summary_validation.build_validation_bundle(mutated_candidate, baseline, repair_pair="BNBUSDT"),
                    "robustness": mutated_robustness,
                }
                self.assertFalse(helper(mutated_item))

    def test_joint_stress_objective_prefers_positive_stress_min_and_stronger_four_year_floor(self) -> None:
        helper = getattr(fractal, "candidate_joint_repair_stress_pass", None)
        if helper is None:
            self.skipTest("candidate_joint_repair_stress_pass is not implemented yet")

        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0061, 0.0062, btc_total=0.90, bnb_total=0.95, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0048, 0.0048, btc_total=5.2, bnb_total=5.2, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        strong_floor = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0049, 0.0048, btc_total=5.3, bnb_total=5.2, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        weak_floor = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0042, 0.0041, btc_total=4.8, bnb_total=4.7, btc_mdd=-0.14, bnb_mdd=-0.14),
        }
        strong_min = make_robustness(
            latest_fold_stress_reserve_score=0.25,
            latest_non_nominal_stress_reserve_score=0.20,
            stress_survival_rate_mean=0.72,
            stress_survival_rate_min=0.70,
            non_nominal_stress_survival_rate_mean=0.71,
            min_fold_non_nominal_stress_survival_rate=0.70,
            latest_non_nominal_stress_survival_rate=0.71,
        )
        weak_min = make_robustness(
            latest_fold_stress_reserve_score=0.25,
            latest_non_nominal_stress_reserve_score=0.20,
            stress_survival_rate_mean=0.72,
            stress_survival_rate_min=0.0,
            non_nominal_stress_survival_rate_mean=0.71,
            min_fold_non_nominal_stress_survival_rate=0.70,
            latest_non_nominal_stress_survival_rate=0.71,
        )

        strong_score, _ = fractal.fractal_fast_scalar_score(
            strong_floor,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            strong_min,
            0.0,
            repair_pair="BNBUSDT",
        )
        weak_min_score, _ = fractal.fractal_fast_scalar_score(
            strong_floor,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            weak_min,
            0.0,
            repair_pair="BNBUSDT",
        )
        weak_floor_score, _ = fractal.fractal_fast_scalar_score(
            weak_floor,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            strong_min,
            0.0,
            repair_pair="BNBUSDT",
        )

        self.assertTrue(helper({"windows": strong_floor, "validation": summary_validation.build_validation_bundle(strong_floor, baseline, repair_pair="BNBUSDT"), "robustness": strong_min}))
        self.assertGreater(strong_score, weak_min_score)
        self.assertGreater(strong_score, weak_floor_score)

    def test_joint_stress_objective_prefers_stress_ready_candidate_over_joint_repair_only_candidate(self) -> None:
        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0061, 0.0062, btc_total=0.90, bnb_total=0.95, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0048, 0.0047, btc_total=5.2, bnb_total=5.1, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        joint_repair_only = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0042, 0.0041, btc_total=4.8, bnb_total=4.7, btc_mdd=-0.14, bnb_mdd=-0.14),
        }
        stress_ready = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0049, 0.0048, btc_total=5.3, bnb_total=5.2, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        joint_repair_only_robustness = make_robustness(
            latest_fold_stress_reserve_score=-0.15,
            latest_non_nominal_stress_reserve_score=-0.10,
            stress_survival_rate_mean=0.60,
            stress_survival_rate_min=0.0,
            non_nominal_stress_survival_rate_mean=0.60,
            min_fold_non_nominal_stress_survival_rate=0.60,
            latest_non_nominal_stress_survival_rate=0.60,
        )
        stress_ready_robustness = make_robustness(
            latest_fold_stress_reserve_score=0.25,
            latest_non_nominal_stress_reserve_score=0.20,
            stress_survival_rate_mean=0.72,
            stress_survival_rate_min=0.70,
            non_nominal_stress_survival_rate_mean=0.71,
            min_fold_non_nominal_stress_survival_rate=0.70,
            latest_non_nominal_stress_survival_rate=0.71,
        )

        joint_only_score, _ = fractal.fractal_fast_scalar_score(
            joint_repair_only,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            joint_repair_only_robustness,
            0.0,
            repair_pair="BNBUSDT",
        )
        stress_ready_score, _ = fractal.fractal_fast_scalar_score(
            stress_ready,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            stress_ready_robustness,
            0.0,
            repair_pair="BNBUSDT",
        )

        self.assertGreater(stress_ready_score, joint_only_score)

    def test_build_validation_bundle_flags_one_year_pair_repair_profile(self) -> None:
        candidate = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0061, 0.0062, btc_total=0.90, bnb_total=0.95, btc_mdd=-0.11, bnb_mdd=-0.11),
            "recent_1y": make_window(0.0035, -0.0001, btc_total=1.0, bnb_total=-0.03, btc_mdd=-0.10, bnb_mdd=-0.12),
            "full_4y": make_window(0.0048, 0.0047, btc_total=5.0, bnb_total=4.8, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        baseline = {
            "recent_2m": make_window(0.0040, 0.0040, btc_total=0.25, bnb_total=0.25, btc_mdd=-0.11, bnb_mdd=-0.11),
            "recent_6m": make_window(0.0058, 0.0058, btc_total=0.80, bnb_total=0.80, btc_mdd=-0.12, bnb_mdd=-0.12),
            "full_4y": make_window(0.0045, 0.0045, btc_total=4.5, bnb_total=4.5, btc_mdd=-0.13, bnb_mdd=-0.13),
        }

        validation = summary_validation.build_validation_bundle(candidate, baseline, repair_pair="BNBUSDT")

        self.assertFalse(validation["profiles"]["pair_repair_1y"]["passed"])

    def test_build_validation_bundle_flags_joint_repair_market_os_profile(self) -> None:
        candidate = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0062, 0.0063, btc_total=0.95, bnb_total=0.96, btc_mdd=-0.11, bnb_mdd=-0.11),
            "recent_1y": make_window(0.0038, 0.0002, btc_total=1.1, bnb_total=0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0046, 0.0047, btc_total=5.0, bnb_total=5.1, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        baseline = {
            "recent_2m": make_window(0.0040, 0.0040, btc_total=0.25, bnb_total=0.25, btc_mdd=-0.11, bnb_mdd=-0.11),
            "recent_6m": make_window(0.0058, 0.0058, btc_total=0.80, bnb_total=0.80, btc_mdd=-0.12, bnb_mdd=-0.12),
            "full_4y": make_window(0.0045, 0.0045, btc_total=4.5, bnb_total=4.5, btc_mdd=-0.13, bnb_mdd=-0.13),
        }

        validation = summary_validation.build_validation_bundle(candidate, baseline, repair_pair="BNBUSDT")

        self.assertTrue(validation["profiles"]["joint_repair_market_os"]["passed"])

    def test_candidate_repair_hard_gate_requires_positive_bnb_one_year_daily_total_pair_count_and_max_drawdown(self) -> None:
        helper = get_repair_hard_gate_helper()
        if helper is None:
            self.skipTest("candidate_repair_hard_gate_pass / repair_hard_gate_passed is not implemented yet")

        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0061, 0.0062, btc_total=0.90, bnb_total=0.95, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0048, 0.0047, btc_total=5.2, bnb_total=5.1, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        passing = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0003, btc_total=1.10, bnb_total=0.08, btc_mdd=-0.10, bnb_mdd=-0.11),
        }
        candidate_item = {
            "windows": passing,
            "validation": summary_validation.build_validation_bundle(passing, baseline, repair_pair="BNBUSDT"),
            "robustness": make_robustness(
                latest_fold_stress_reserve_score=0.25,
                latest_non_nominal_stress_reserve_score=0.20,
                stress_survival_rate_mean=0.72,
                stress_survival_rate_min=0.70,
                non_nominal_stress_survival_rate_mean=0.71,
                min_fold_non_nominal_stress_survival_rate=0.70,
                latest_non_nominal_stress_survival_rate=0.71,
            ),
        }

        self.assertTrue(helper(candidate_item))

        cases = [
            (
                "negative_daily_return",
                {
                    **passing,
                    "recent_1y": make_window(0.0038, -0.0002, btc_total=1.10, bnb_total=0.08, btc_mdd=-0.10, bnb_mdd=-0.11),
                },
            ),
            (
                "negative_total_return",
                {
                    **passing,
                    "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=-0.05, btc_mdd=-0.10, bnb_mdd=-0.11),
                },
            ),
            (
                "positive_pair_count_shortfall",
                {
                    **passing,
                    "recent_1y": {
                        **make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.08, btc_mdd=-0.10, bnb_mdd=-0.11),
                        "aggregate": {
                            **make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.08, btc_mdd=-0.10, bnb_mdd=-0.11)["aggregate"],
                            "positive_pair_count": 1,
                        },
                    },
                },
            ),
            (
                "max_drawdown_too_deep",
                {
                    **passing,
                    "recent_1y": make_window(0.0038, 0.0002, btc_total=1.10, bnb_total=0.08, btc_mdd=-0.10, bnb_mdd=-0.16),
                },
            ),
        ]

        for name, mutated_windows in cases:
            with self.subTest(name=name):
                mutated_item = {
                    "windows": mutated_windows,
                    "validation": summary_validation.build_validation_bundle(mutated_windows, baseline, repair_pair="BNBUSDT"),
                    "robustness": candidate_item["robustness"],
                }
                self.assertFalse(helper(mutated_item))

    def test_candidate_repair_hard_gate_score_prefers_hard_pass_over_hard_fail_even_when_fail_has_better_stress_and_four_year_floor(self) -> None:
        helper = get_repair_hard_gate_helper()
        if helper is None:
            self.skipTest("candidate_repair_hard_gate_pass / repair_hard_gate_passed is not implemented yet")

        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0061, 0.0062, btc_total=0.90, bnb_total=0.95, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0048, 0.0047, btc_total=5.2, bnb_total=5.1, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        hard_pass = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0003, btc_total=1.10, bnb_total=0.08, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0047, 0.0046, btc_total=5.1, bnb_total=5.0, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        hard_fail = {
            **baseline,
            "recent_1y": make_window(0.0038, 0.0003, btc_total=1.10, bnb_total=-0.02, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0049, 0.0048, btc_total=5.3, bnb_total=5.2, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        hard_pass_robustness = make_robustness(
            latest_fold_stress_reserve_score=0.25,
            latest_non_nominal_stress_reserve_score=0.20,
            stress_survival_rate_mean=0.72,
            stress_survival_rate_min=0.70,
            non_nominal_stress_survival_rate_mean=0.71,
            min_fold_non_nominal_stress_survival_rate=0.70,
            latest_non_nominal_stress_survival_rate=0.71,
        )
        hard_fail_robustness = make_robustness(
            latest_fold_stress_reserve_score=0.30,
            latest_non_nominal_stress_reserve_score=0.25,
            stress_survival_rate_mean=0.76,
            stress_survival_rate_min=0.74,
            non_nominal_stress_survival_rate_mean=0.75,
            min_fold_non_nominal_stress_survival_rate=0.73,
            latest_non_nominal_stress_survival_rate=0.74,
        )

        hard_pass_score, _ = fractal.fractal_fast_scalar_score(
            hard_pass,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            hard_pass_robustness,
            0.0,
            repair_pair="BNBUSDT",
        )
        hard_fail_score, _ = fractal.fractal_fast_scalar_score(
            hard_fail,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            hard_fail_robustness,
            0.0,
            repair_pair="BNBUSDT",
        )

        self.assertTrue(
            helper(
                {
                    "windows": hard_pass,
                    "validation": summary_validation.build_validation_bundle(hard_pass, baseline, repair_pair="BNBUSDT"),
                    "robustness": hard_pass_robustness,
                }
            )
        )
        self.assertFalse(
            helper(
                {
                    "windows": hard_fail,
                    "validation": summary_validation.build_validation_bundle(hard_fail, baseline, repair_pair="BNBUSDT"),
                    "robustness": hard_fail_robustness,
                }
            )
        )
        self.assertGreater(hard_pass_score, hard_fail_score)

    def test_candidate_final_hard_gate_requires_repair_stress_cost_reserve_and_target_060(self) -> None:
        helper = get_final_hard_gate_helper()
        if helper is None:
            self.skipTest("candidate_final_hard_gate_pass / final_hard_gate_passed is not implemented yet")

        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0058, 0.0058, btc_total=0.90, bnb_total=0.90, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0045, 0.0045, btc_total=4.8, bnb_total=4.8, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        passing = {
            "recent_2m": make_window(0.0048, 0.0049, btc_total=0.32, bnb_total=0.33, btc_mdd=-0.09, bnb_mdd=-0.09),
            "recent_6m": make_window(0.0063, 0.0064, btc_total=1.00, bnb_total=1.05, btc_mdd=-0.11, bnb_mdd=-0.11),
            "recent_1y": make_window(0.0040, 0.0004, btc_total=1.20, bnb_total=0.09, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0061, 0.0062, btc_total=5.9, bnb_total=6.0, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        strong_robustness = make_robustness(
            latest_fold_stress_reserve_score=0.30,
            latest_non_nominal_stress_reserve_score=0.25,
            stress_survival_rate_mean=0.72,
            stress_survival_rate_min=0.70,
            non_nominal_stress_survival_rate_mean=0.72,
            min_fold_non_nominal_stress_survival_rate=0.70,
            latest_non_nominal_stress_survival_rate=0.71,
        )
        candidate_item = {
            "windows": passing,
            "validation": summary_validation.build_validation_bundle(passing, baseline, repair_pair="BNBUSDT"),
            "robustness": strong_robustness,
        }

        self.assertTrue(helper(candidate_item))

        cases = [
            (
                "repair_hard_failed",
                {
                    **passing,
                    "recent_1y": make_window(0.0040, -0.0001, btc_total=1.20, bnb_total=0.09, btc_mdd=-0.10, bnb_mdd=-0.11),
                },
                strong_robustness,
            ),
            (
                "target_060_failed",
                {
                    **passing,
                    "full_4y": make_window(0.0057, 0.0058, btc_total=5.7, bnb_total=5.8, btc_mdd=-0.12, bnb_mdd=-0.12),
                },
                strong_robustness,
            ),
            (
                "wf1_failed",
                passing,
                make_robustness(
                    latest_fold_delta_worst_pair_total_return=-0.01,
                    latest_fold_delta_worst_max_drawdown=-0.01,
                    latest_fold_delta_worst_daily_win_rate=-0.01,
                    latest_fold_stress_reserve_score=0.30,
                    latest_non_nominal_stress_reserve_score=0.25,
                    stress_survival_rate_mean=0.72,
                    stress_survival_rate_min=0.70,
                    non_nominal_stress_survival_rate_mean=0.72,
                    min_fold_non_nominal_stress_survival_rate=0.70,
                    latest_non_nominal_stress_survival_rate=0.71,
                ),
            ),
            (
                "stress_failed",
                passing,
                make_robustness(
                    latest_fold_stress_reserve_score=-0.10,
                    latest_non_nominal_stress_reserve_score=0.25,
                    stress_survival_rate_mean=0.60,
                    stress_survival_rate_min=0.60,
                    non_nominal_stress_survival_rate_mean=0.72,
                    min_fold_non_nominal_stress_survival_rate=0.70,
                    latest_non_nominal_stress_survival_rate=0.71,
                ),
            ),
            (
                "cost_reserve_failed",
                passing,
                make_robustness(
                    latest_fold_stress_reserve_score=0.30,
                    latest_non_nominal_stress_reserve_score=-0.10,
                    stress_survival_rate_mean=0.72,
                    stress_survival_rate_min=0.70,
                    non_nominal_stress_survival_rate_mean=0.60,
                    min_fold_non_nominal_stress_survival_rate=0.60,
                    latest_non_nominal_stress_survival_rate=0.60,
                ),
            ),
        ]

        for name, mutated_windows, mutated_robustness in cases:
            with self.subTest(name=name):
                mutated_item = {
                    "windows": mutated_windows,
                    "validation": summary_validation.build_validation_bundle(mutated_windows, baseline, repair_pair="BNBUSDT"),
                    "robustness": mutated_robustness,
                }
                self.assertFalse(helper(mutated_item))

    def test_final_hard_score_prefers_final_hard_candidate_over_repair_only_candidate(self) -> None:
        helper = get_final_hard_gate_helper()
        if helper is None:
            self.skipTest("candidate_final_hard_gate_pass / final_hard_gate_passed is not implemented yet")

        baseline = {
            "recent_2m": make_window(0.0045, 0.0045, btc_total=0.30, bnb_total=0.30, btc_mdd=-0.10, bnb_mdd=-0.10),
            "recent_6m": make_window(0.0058, 0.0058, btc_total=0.90, bnb_total=0.90, btc_mdd=-0.11, bnb_mdd=-0.11),
            "full_4y": make_window(0.0045, 0.0045, btc_total=4.8, bnb_total=4.8, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        final_hard = {
            "recent_2m": make_window(0.0049, 0.0050, btc_total=0.35, bnb_total=0.36, btc_mdd=-0.09, bnb_mdd=-0.09),
            "recent_6m": make_window(0.0064, 0.0065, btc_total=1.05, bnb_total=1.08, btc_mdd=-0.11, bnb_mdd=-0.11),
            "recent_1y": make_window(0.0040, 0.0004, btc_total=1.20, bnb_total=0.09, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0062, 0.0061, btc_total=6.0, bnb_total=5.9, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        repair_only = {
            "recent_2m": make_window(0.0051, 0.0052, btc_total=0.36, bnb_total=0.37, btc_mdd=-0.09, bnb_mdd=-0.09),
            "recent_6m": make_window(0.0065, 0.0066, btc_total=1.08, bnb_total=1.10, btc_mdd=-0.11, bnb_mdd=-0.11),
            "recent_1y": make_window(0.0040, 0.0005, btc_total=1.22, bnb_total=0.10, btc_mdd=-0.10, bnb_mdd=-0.11),
            "full_4y": make_window(0.0058, 0.0057, btc_total=5.8, bnb_total=5.7, btc_mdd=-0.12, bnb_mdd=-0.12),
        }
        final_hard_robustness = make_robustness(
            latest_fold_stress_reserve_score=0.30,
            latest_non_nominal_stress_reserve_score=0.25,
            stress_survival_rate_mean=0.72,
            stress_survival_rate_min=0.70,
            non_nominal_stress_survival_rate_mean=0.72,
            min_fold_non_nominal_stress_survival_rate=0.70,
            latest_non_nominal_stress_survival_rate=0.71,
        )
        repair_only_robustness = make_robustness(
            latest_fold_stress_reserve_score=0.32,
            latest_non_nominal_stress_reserve_score=0.28,
            stress_survival_rate_mean=0.74,
            stress_survival_rate_min=0.72,
            non_nominal_stress_survival_rate_mean=0.74,
            min_fold_non_nominal_stress_survival_rate=0.72,
            latest_non_nominal_stress_survival_rate=0.73,
        )

        final_hard_score, _ = fractal.fractal_fast_scalar_score(
            final_hard,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            final_hard_robustness,
            0.0,
            repair_pair="BNBUSDT",
        )
        repair_only_score, _ = fractal.fractal_fast_scalar_score(
            repair_only,
            baseline,
            fractal.FilterDecision(accepted=True, source="test", reason="ok"),
            LeafNode(0),
            repair_only_robustness,
            0.0,
            repair_pair="BNBUSDT",
        )

        self.assertTrue(
            helper(
                {
                    "windows": final_hard,
                    "validation": summary_validation.build_validation_bundle(final_hard, baseline, repair_pair="BNBUSDT"),
                    "robustness": final_hard_robustness,
                }
            )
        )
        self.assertFalse(
            helper(
                {
                    "windows": repair_only,
                    "validation": summary_validation.build_validation_bundle(repair_only, baseline, repair_pair="BNBUSDT"),
                    "robustness": repair_only_robustness,
                }
            )
        )
        self.assertGreater(final_hard_score, repair_only_score)


if __name__ == "__main__":
    unittest.main()
