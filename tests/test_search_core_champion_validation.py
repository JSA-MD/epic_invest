import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_core_champion as champion


def make_candidate(
    *,
    dsr: float = 0.45,
    generalization_gap: float = 0.05,
    return_stability_gap: float = 0.10,
    cpcv_pass_rate: float = 0.80,
    cpcv_positive_rate: float = 0.80,
    cpcv_min_test_return: float = -0.03,
    cpcv_splits: list[dict[str, float]] | None = None,
    cpcv_pbo: dict[str, float] | None = None,
    fold_positive_rate: float = 0.80,
    stress_survival_rate: float = 1.0,
    regime_positive_rate: float = 2.0 / 3.0,
    corr_positive_rate: float = 2.0 / 3.0,
) -> dict[str, object]:
    splits = cpcv_splits or [
        {"train_total_return": 0.10, "test_total_return": 0.04},
        {"train_total_return": 0.08, "test_total_return": 0.03},
        {"train_total_return": 0.06, "test_total_return": 0.02},
    ]
    pbo_profile = cpcv_pbo or {
        "selected_count": 1,
        "selection_share": 0.10,
        "selected_below_median_rate": 0.0,
        "avg_selected_test_percentile": 0.80,
        "worst_selected_test_percentile": 0.80,
        "avg_selected_test_score": 12.0,
        "avg_selected_test_return": 0.04,
    }
    return {
        "test": {"total_return": 0.08},
        "oos": {"total_return": 0.14, "max_drawdown": -0.12, "sharpe": 1.2},
        "objective_metrics": {
            "oos_dsr_proxy": dsr,
            "generalization_gap": generalization_gap,
            "return_stability_gap": return_stability_gap,
            "oos_calmar": 1.1,
            "oos_cvar": -0.02,
        },
        "fold_robustness": {
            "fold_positive_rate": fold_positive_rate,
            "fold_avg_return": 0.03,
            "fold_min_return": -0.02,
        },
        "cpcv": {
            "pass_rate": cpcv_pass_rate,
            "test_positive_rate": cpcv_positive_rate,
            "avg_test_return": 0.03,
            "min_test_return": cpcv_min_test_return,
            "splits": splits,
        },
        "cpcv_pbo": pbo_profile,
        "stress": {"stress_survival_rate": stress_survival_rate},
        "regime_breakdown": {
            "summary": {
                "positive_rate": regime_positive_rate,
                "avg_total_return": 0.02,
                "min_total_return": -0.01,
            }
        },
        "corr_state_robustness": {
            "summary": {
                "positive_rate": corr_positive_rate,
                "avg_total_return": 0.02,
                "min_total_return": -0.01,
            }
        },
        "parameter_stability": {
            "neighbor_positive_rate": 0.75,
            "neighbor_avg_oos_return": 0.05,
            "neighbor_min_oos_return": -0.01,
            "neighbor_oos_std": 0.02,
        },
        "pareto": {"rank": 1, "crowding_sort_value": 1.0},
        "selection_score": 0.0,
    }


class SearchCoreChampionValidationTests(unittest.TestCase):
    def test_candidate_selection_pbo_reports_split_level_overfit(self) -> None:
        summary = champion.summarize_candidate_selection_pbo(
            [
                {
                    "test_blocks": [0, 1],
                    "candidates": [
                        {"key": "a", "train_score": 10.0, "test_score": 1.0, "train_total_return": 0.10, "test_total_return": -0.02},
                        {"key": "b", "train_score": 8.0, "test_score": 3.0, "train_total_return": 0.08, "test_total_return": 0.03},
                        {"key": "c", "train_score": 7.0, "test_score": 2.0, "train_total_return": 0.07, "test_total_return": 0.01},
                    ],
                },
                {
                    "test_blocks": [2, 3],
                    "candidates": [
                        {"key": "a", "train_score": 11.0, "test_score": 0.5, "train_total_return": 0.11, "test_total_return": -0.01},
                        {"key": "b", "train_score": 9.0, "test_score": 2.0, "train_total_return": 0.09, "test_total_return": 0.02},
                        {"key": "c", "train_score": 6.0, "test_score": 1.5, "train_total_return": 0.06, "test_total_return": 0.01},
                    ],
                },
            ]
        )

        self.assertEqual(summary["n_splits"], 2)
        self.assertEqual(summary["pbo"], 1.0)
        self.assertEqual(summary["profiles"]["a"]["selected_count"], 2)
        self.assertLess(summary["profiles"]["a"]["avg_selected_test_percentile"], 0.5)

    def test_low_dsr_candidate_is_blocked(self) -> None:
        candidate = make_candidate(dsr=0.05)
        candidate["validation_profile"] = champion.build_validation_profile(candidate)
        gate = champion.build_promotion_gate(candidate)

        self.assertFalse(gate["passes_dsr_hard_gate"])
        self.assertFalse(gate["passed"])
        self.assertIn("passes_dsr_hard_gate", gate["failed_checks"])

    def test_large_generalization_gap_is_blocked(self) -> None:
        candidate = make_candidate(generalization_gap=0.14, return_stability_gap=0.21)
        candidate["validation_profile"] = champion.build_validation_profile(candidate)
        gate = champion.build_promotion_gate(candidate)

        self.assertFalse(gate["passes_generalization_gap"])
        self.assertFalse(gate["passes_return_stability_gap"])
        self.assertFalse(gate["passed"])

    def test_cpcv_overfit_rate_flags_train_test_breakdown(self) -> None:
        candidate = make_candidate(
            cpcv_splits=[
                {"train_total_return": 0.12, "test_total_return": -0.01},
                {"train_total_return": 0.08, "test_total_return": 0.01},
                {"train_total_return": 0.07, "test_total_return": 0.00},
            ],
            cpcv_pbo={
                "selected_count": 2,
                "selection_share": 0.20,
                "selected_below_median_rate": 0.50,
                "avg_selected_test_percentile": 0.40,
                "worst_selected_test_percentile": 0.20,
                "avg_selected_test_score": 2.0,
                "avg_selected_test_return": -0.01,
            },
        )
        candidate["validation_profile"] = champion.build_validation_profile(candidate)
        gate = champion.build_promotion_gate(candidate)

        self.assertGreaterEqual(candidate["validation_profile"]["cpcv_overfit_rate"], 0.80)
        self.assertFalse(gate["passes_cpcv_overfit_rate"])
        self.assertFalse(gate["passes_pbo_selected_below_median_rate"])
        self.assertFalse(gate["passes_pbo_avg_selected_test_percentile"])
        self.assertFalse(gate["passed"])

    def test_balanced_candidate_passes_hard_gate(self) -> None:
        candidate = make_candidate()
        candidate["validation_profile"] = champion.build_validation_profile(candidate)
        gate = champion.build_promotion_gate(candidate)

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failed_checks"], [])


if __name__ == "__main__":
    unittest.main()
