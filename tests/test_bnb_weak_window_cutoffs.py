from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from search_bnb_weak_window_cutoffs import (
    ADAPTATION_LABEL,
    BNB_PAIR,
    STATIC_PAIR,
    build_bnb_neighbor_variants,
    candidate_score,
    evaluate_guard,
)


class BnbWeakWindowCutoffTests(unittest.TestCase):
    def test_neighbor_variants_only_touch_bnb_execution_gene(self) -> None:
        candidate = {
            "candidate_kind": "pairwise_candidate",
            "pair_configs": {
                STATIC_PAIR: {
                    "mapping_indices": [0, 1, 2, 3],
                    "route_breadth_threshold": 0.5,
                },
                BNB_PAIR: {
                    "mapping_indices": [0, 1, 2, 3],
                    "route_breadth_threshold": 0.5,
                    "execution_gene": {
                        "signal_gate_pct": 0.15,
                        "range_signal_gate_mult": 1.0,
                        "range_regime_buffer_mult": 1.0,
                    },
                },
            },
        }
        baseline_btc = copy.deepcopy(candidate["pair_configs"][STATIC_PAIR])
        variants = build_bnb_neighbor_variants(candidate)
        self.assertTrue(variants)
        for variant in variants:
            self.assertEqual(variant["pair_configs"][STATIC_PAIR], baseline_btc)
            self.assertIn("execution_gene", variant["pair_configs"][BNB_PAIR])

    def test_guard_and_score_reward_target_improvement_without_durable_damage(self) -> None:
        compare = {
            ADAPTATION_LABEL: {"delta_worst_total_return": 0.02, "delta_worst_max_drawdown": 0.0, "delta_worst_win_rate": 0.01, "delta_mean_win_rate": 0.02},
            "recent_6m": {"delta_worst_total_return": 0.05, "delta_worst_max_drawdown": 0.001, "delta_worst_win_rate": 0.01},
            "recent_1y": {"delta_worst_total_return": 0.08, "delta_worst_max_drawdown": 0.0005, "delta_worst_win_rate": 0.01},
            "full_4y": {"delta_worst_total_return": 0.10, "delta_worst_max_drawdown": 0.0, "delta_worst_win_rate": 0.02},
        }
        weak_compare = {
            "weak_2024_02": {
                "candidate_total_return": -0.08,
                "delta_total_return": 0.04,
                "delta_daily_win_rate": 0.10,
                "delta_max_drawdown": 0.02,
            },
            "weak_2024_03": {
                "candidate_total_return": -0.02,
                "delta_total_return": 0.02,
                "delta_daily_win_rate": 0.03,
                "delta_max_drawdown": 0.01,
            },
            "weak_2025_05": {
                "candidate_total_return": 0.01,
                "delta_total_return": 0.01,
                "delta_daily_win_rate": 0.0,
                "delta_max_drawdown": 0.005,
            },
        }
        guard = evaluate_guard(compare, weak_compare)
        self.assertTrue(guard["guard_pass"])
        self.assertTrue(guard["weak_pass"])
        self.assertEqual(guard["traffic_light"], "green")
        score = candidate_score(compare, weak_compare, guard)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
