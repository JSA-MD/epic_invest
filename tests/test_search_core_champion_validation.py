import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


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


def make_selection_row(
    key: str,
    *,
    total_return: float,
    dsr: float,
    quality: float,
    cpcv_pass_rate: float,
    stress_survival_rate: float,
    corr_positive_rate: float,
    drawdown_abs: float,
    cvar_abs: float,
    promotion_score: float,
    gate_passed: bool = True,
) -> dict[str, object]:
    return {
        "key": key,
        "promotion_gate": {"passed": gate_passed},
        "promotion_score": promotion_score,
        "selection_score": 0.0,
        "pareto_vector": {
            "oos_total_return": total_return,
            "oos_dsr_proxy": dsr,
            "validation_quality_score": quality,
            "cpcv_pass_rate": cpcv_pass_rate,
            "stress_survival_rate": stress_survival_rate,
            "corr_positive_rate": corr_positive_rate,
            "oos_max_drawdown_abs": drawdown_abs,
            "oos_cvar_abs": cvar_abs,
        },
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

    def test_market_operating_system_gate_blocks_promotion(self) -> None:
        candidate = make_candidate()
        candidate["validation_profile"] = champion.build_validation_profile(candidate)
        candidate["market_operating_system"] = {
            "gate": {
                "passes_market_os_fitness": False,
                "passes_corr_state_robustness": True,
                "passes_regime_coverage": True,
                "passes_parameter_instability": True,
            }
        }
        gate = champion.build_promotion_gate(candidate)

        self.assertFalse(gate["passes_market_os_fitness"])
        self.assertFalse(gate["passed"])
        self.assertIn("passes_market_os_fitness", gate["failed_checks"])

    def test_market_operating_system_audit_does_not_block_promotion(self) -> None:
        candidate = make_candidate()
        candidate["validation_profile"] = champion.build_validation_profile(candidate)
        candidate["market_operating_system"] = {
            "gate": {
                "passes_market_os_fitness": True,
                "passes_corr_state_robustness": True,
                "passes_regime_coverage": True,
                "passes_parameter_instability": True,
            },
            "audit": {
                "passes_final_oos_total_return": False,
                "passes_final_oos_max_drawdown": False,
            },
        }
        gate = champion.build_promotion_gate(candidate)

        self.assertTrue(gate["passed"])
        self.assertNotIn("passes_final_oos_total_return", gate)

    def test_market_os_state_payload_uses_route_and_corr_states(self) -> None:
        index = pd.date_range("2025-01-01", periods=90, freq="D", tz="UTC")
        base = np.linspace(100.0, 145.0, len(index))
        close = pd.DataFrame(
            {
                "BTCUSDT": base,
                "ETHUSDT": base * 0.8 + np.sin(np.arange(len(index))) * 2.0,
                "SOLUSDT": base * 0.5 + np.cos(np.arange(len(index))) * 3.0,
                "XRPUSDT": base * 0.2 + np.linspace(0.0, 8.0, len(index)),
            },
            index=index,
        )
        market_context = pd.DataFrame({"QQQ": base * 1.1 + np.sin(np.arange(len(index)))}, index=index)
        portfolio_frame = pd.DataFrame(
            {
                "net_return": np.full(len(index), 0.002, dtype="float64"),
                "turnover": np.full(len(index), 0.15, dtype="float64"),
            },
            index=index,
        )
        payload = champion.build_market_os_state_payload(
            close,
            market_context,
            portfolio_frame,
            {
                "family": champion.LONG_ONLY_FAMILY,
                "params": {
                    "lookback_fast": 5,
                    "lookback_slow": 14,
                    "vol_window": 5,
                    "regime_threshold": 0.02,
                    "breadth_threshold": 0.50,
                },
            },
        )

        self.assertTrue(payload["route_state_returns"])
        self.assertGreaterEqual(payload["total_route_states"], 4)

    def test_final_selection_uses_nsga2_front_before_scalar_score(self) -> None:
        dominated_high_score = make_selection_row(
            "dominated",
            total_return=0.05,
            dsr=0.40,
            quality=0.65,
            cpcv_pass_rate=0.70,
            stress_survival_rate=0.80,
            corr_positive_rate=0.60,
            drawdown_abs=0.20,
            cvar_abs=0.04,
            promotion_score=999.0,
        )
        dominant_low_score = make_selection_row(
            "dominant",
            total_return=0.06,
            dsr=0.50,
            quality=0.75,
            cpcv_pass_rate=0.80,
            stress_survival_rate=0.90,
            corr_positive_rate=0.70,
            drawdown_abs=0.10,
            cvar_abs=0.02,
            promotion_score=0.0,
        )

        ranked = champion.rank_candidates_for_selection([dominated_high_score, dominant_low_score])

        self.assertEqual(ranked[0]["key"], "dominant")
        self.assertEqual(ranked[0]["selection_nsga2"]["rank"], 1)
        self.assertGreater(ranked[1]["selection_nsga2"]["rank"], ranked[0]["selection_nsga2"]["rank"])

    def test_final_selection_keeps_promotion_gate_priority(self) -> None:
        failed_gate_dominant = make_selection_row(
            "failed",
            total_return=0.20,
            dsr=0.90,
            quality=0.95,
            cpcv_pass_rate=1.00,
            stress_survival_rate=1.00,
            corr_positive_rate=1.00,
            drawdown_abs=0.02,
            cvar_abs=0.01,
            promotion_score=999.0,
            gate_passed=False,
        )
        passed_gate_weaker = make_selection_row(
            "passed",
            total_return=0.03,
            dsr=0.35,
            quality=0.60,
            cpcv_pass_rate=0.70,
            stress_survival_rate=0.80,
            corr_positive_rate=0.60,
            drawdown_abs=0.15,
            cvar_abs=0.03,
            promotion_score=0.0,
            gate_passed=True,
        )

        ranked = champion.rank_candidates_for_selection([failed_gate_dominant, passed_gate_weaker])

        self.assertEqual(ranked[0]["key"], "passed")
        self.assertFalse(ranked[1]["promotion_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
