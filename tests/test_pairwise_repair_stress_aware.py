import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import repair_pair_subset_pairwise_candidate as repair_pairwise


def make_windows(
    *,
    recent_2m: float,
    recent_6m: float,
    full_4y: float,
    full_4y_mean: float,
    bnb_full_4y: float,
    recent_6m_mdd: float = -0.10,
    full_4y_mdd: float = -0.14,
    hit_rate: float = 0.58,
    win_rate: float = 0.60,
    sharpe: float = 1.6,
    worst_day: float = -0.018,
    n_trades: int = 48,
    dispersion: float = 0.003,
) -> dict[str, object]:
    return {
        "recent_2m": {
            "aggregate": {
                "worst_pair_avg_daily_return": recent_2m,
                "mean_avg_daily_return": recent_2m + 0.0003,
                "worst_max_drawdown": recent_6m_mdd,
                "pair_return_dispersion": dispersion,
                "positive_pair_count": 2,
            },
            "per_pair": {
                "BTCUSDT": {
                    "avg_daily_return": recent_2m + 0.0004,
                    "daily_target_hit_rate": hit_rate,
                    "daily_win_rate": win_rate,
                    "sharpe": sharpe,
                    "worst_day": worst_day,
                    "n_trades": n_trades,
                },
                "BNBUSDT": {
                    "avg_daily_return": recent_2m,
                    "daily_target_hit_rate": hit_rate,
                    "daily_win_rate": win_rate,
                    "sharpe": sharpe,
                    "worst_day": worst_day,
                    "n_trades": n_trades,
                },
            },
        },
        "recent_6m": {
            "aggregate": {
                "worst_pair_avg_daily_return": recent_6m,
                "mean_avg_daily_return": recent_6m + 0.0004,
                "worst_max_drawdown": recent_6m_mdd,
                "pair_return_dispersion": dispersion,
                "positive_pair_count": 2,
            },
            "per_pair": {
                "BTCUSDT": {
                    "avg_daily_return": recent_6m + 0.0005,
                    "daily_target_hit_rate": hit_rate,
                    "daily_win_rate": win_rate,
                    "sharpe": sharpe,
                    "worst_day": worst_day,
                    "n_trades": n_trades,
                },
                "BNBUSDT": {
                    "avg_daily_return": recent_6m,
                    "daily_target_hit_rate": hit_rate,
                    "daily_win_rate": win_rate,
                    "sharpe": sharpe,
                    "worst_day": worst_day,
                    "n_trades": n_trades,
                },
            },
        },
        "full_4y": {
            "aggregate": {
                "worst_pair_avg_daily_return": full_4y,
                "mean_avg_daily_return": full_4y_mean,
                "worst_max_drawdown": full_4y_mdd,
                "pair_return_dispersion": dispersion,
                "positive_pair_count": 2,
            },
            "per_pair": {
                "BTCUSDT": {
                    "avg_daily_return": full_4y_mean,
                    "daily_target_hit_rate": hit_rate,
                    "daily_win_rate": win_rate,
                    "sharpe": sharpe,
                    "worst_day": worst_day,
                    "n_trades": n_trades,
                },
                "BNBUSDT": {
                    "avg_daily_return": bnb_full_4y,
                    "daily_target_hit_rate": hit_rate,
                    "daily_win_rate": win_rate,
                    "sharpe": sharpe,
                    "worst_day": worst_day,
                    "n_trades": n_trades,
                },
            },
        },
    }


class PairwiseRepairStressAwareTests(unittest.TestCase):
    def test_build_pairwise_pareto_vector_exposes_expected_objectives(self) -> None:
        item = {
            "candidate_id": "pairwise-a",
            "windows": make_windows(
                recent_2m=0.0072,
                recent_6m=0.0076,
                full_4y=0.0068,
                full_4y_mean=0.0070,
                bnb_full_4y=0.0067,
                recent_6m_mdd=-0.08,
                full_4y_mdd=-0.12,
                dispersion=0.002,
            ),
            "validation_engine": {
                "market_operating_system": {
                    "fitness": {"score": 0.7, "raw": {"turnover_cost": 0.02}},
                    "state_summary": {"corr_state_robustness": 0.61},
                },
                "profile": {
                    "validation_quality_score": 0.88,
                    "false_positive_risk": 0.12,
                },
            },
        }

        pareto_vector = repair_pairwise.build_pairwise_pareto_vector(item)

        self.assertEqual(
            set(pareto_vector.keys()),
            {
                "market_os_fitness",
                "validation_quality_score",
                "recent_2m_worst_pair_avg_daily_return",
                "recent_6m_worst_pair_avg_daily_return",
                "full_4y_worst_pair_avg_daily_return",
                "bnb_full_4y_avg_daily_return",
                "corr_state_robustness",
                "target_060_shortfall",
                "bnb_full_4y_target_shortfall",
                "full_4y_worst_max_drawdown_abs",
                "false_positive_risk",
                "turnover_cost",
            },
        )
        self.assertGreater(pareto_vector["market_os_fitness"], 0.0)
        self.assertLess(pareto_vector["target_060_shortfall"], repair_pairwise.TARGET_060_DAILY_RETURN)

    def test_assign_pairwise_pareto_metadata_marks_dominant_candidate_first_front(self) -> None:
        dominant = {
            "candidate_id": "dominant",
            "windows": make_windows(
                recent_2m=0.0075,
                recent_6m=0.0079,
                full_4y=0.0070,
                full_4y_mean=0.0072,
                bnb_full_4y=0.0069,
                recent_6m_mdd=-0.07,
                full_4y_mdd=-0.10,
                hit_rate=0.65,
                win_rate=0.67,
                sharpe=2.0,
                worst_day=-0.015,
                n_trades=32,
                dispersion=0.0018,
            ),
            "validation_engine": {
                "market_operating_system": {
                    "fitness": {"score": 0.82, "raw": {"turnover_cost": 0.01}},
                    "state_summary": {"corr_state_robustness": 0.72},
                },
                "profile": {
                    "validation_quality_score": 0.93,
                    "false_positive_risk": 0.08,
                },
            },
        }
        dominated = {
            "candidate_id": "dominated",
            "windows": make_windows(
                recent_2m=0.0062,
                recent_6m=0.0064,
                full_4y=0.0058,
                full_4y_mean=0.0060,
                bnb_full_4y=0.0056,
                recent_6m_mdd=-0.15,
                full_4y_mdd=-0.22,
                hit_rate=0.48,
                win_rate=0.50,
                sharpe=1.1,
                worst_day=-0.028,
                n_trades=80,
                dispersion=0.006,
            ),
            "validation_engine": {
                "market_operating_system": {
                    "fitness": {"score": 0.40, "raw": {"turnover_cost": 0.08}},
                    "state_summary": {"corr_state_robustness": 0.31},
                },
                "profile": {
                    "validation_quality_score": 0.44,
                    "false_positive_risk": 0.34,
                },
            },
        }

        for item in (dominant, dominated):
            item["pareto_vector"] = repair_pairwise.build_pairwise_pareto_vector(item)

        metadata = repair_pairwise.assign_pairwise_pareto_metadata([dominant, dominated])

        self.assertEqual(metadata["dominant"]["rank"], 1)
        self.assertTrue(metadata["dominant"]["is_nondominated"])
        self.assertEqual(metadata["dominated"]["rank"], 2)
        self.assertFalse(metadata["dominated"]["is_nondominated"])

    def test_candidate_score_penalizes_bnb_full_4y_shortfall(self) -> None:
        stronger = make_windows(
            recent_2m=0.0068,
            recent_6m=0.0071,
            full_4y=0.0055,
            full_4y_mean=0.0062,
            bnb_full_4y=0.0061,
        )
        weaker = make_windows(
            recent_2m=0.0068,
            recent_6m=0.0071,
            full_4y=0.0055,
            full_4y_mean=0.0062,
            bnb_full_4y=0.0032,
        )

        stronger_score = repair_pairwise.candidate_score(
            stronger,
            0.0056,
            0.0065,
            0.0048,
            0.0050,
            0.0049,
            repair_pairwise.TARGET_060_DAILY_RETURN,
        )
        weaker_score = repair_pairwise.candidate_score(
            weaker,
            0.0056,
            0.0065,
            0.0048,
            0.0050,
            0.0049,
            repair_pairwise.TARGET_060_DAILY_RETURN,
        )

        self.assertGreater(stronger_score, weaker_score)

    def test_stress_aware_fitness_penalizes_validation_gate_failure(self) -> None:
        windows = make_windows(
            recent_2m=0.0069,
            recent_6m=0.0072,
            full_4y=0.0057,
            full_4y_mean=0.0064,
            bnb_full_4y=0.0060,
        )
        passed = {
            "score": 1.0,
            "windows": windows,
            "validation": {
                "profiles": {
                    "target_060": {"passed": True},
                    "final_oos": {"passed": True},
                }
            },
            "validation_engine": {
                "gate": {"passed": True},
                "profile": {"false_positive_risk": 0.08, "validation_quality_score": 0.92},
                "market_operating_system": {
                    "gate": {"passed": True},
                    "fitness": {"score": 0.68},
                },
            },
            "stress_proxy": {
                "evaluated": True,
                "passed": True,
                "reserve": 0.0002,
                "recent_2m_worst_pair_avg_daily_return": 0.0063,
                "full_4y_worst_pair_avg_daily_return": 0.0061,
                "bnb_full_4y_avg_daily_return": 0.0060,
                "recent_6m_worst_mdd_abs": 0.12,
                "full_4y_worst_mdd_abs": 0.16,
            },
            "pareto": {"rank": 1, "crowding_sort_value": 2.0},
        }
        failed = {
            **passed,
            "validation_engine": {
                **passed["validation_engine"],
                "gate": {"passed": False},
            },
        }

        self.assertGreater(
            repair_pairwise.stress_aware_fitness(passed),
            repair_pairwise.stress_aware_fitness(failed),
        )

    def test_fast_validation_proxy_penalizes_fragile_tail_profile(self) -> None:
        robust = make_windows(
            recent_2m=0.0069,
            recent_6m=0.0072,
            full_4y=0.0063,
            full_4y_mean=0.0067,
            bnb_full_4y=0.0062,
            hit_rate=0.61,
            win_rate=0.63,
            sharpe=1.8,
            worst_day=-0.019,
            n_trades=40,
            dispersion=0.0025,
        )
        fragile = make_windows(
            recent_2m=0.0069,
            recent_6m=0.0072,
            full_4y=0.0063,
            full_4y_mean=0.0067,
            bnb_full_4y=0.0062,
            full_4y_mdd=-0.27,
            hit_rate=0.41,
            win_rate=0.44,
            sharpe=0.5,
            worst_day=-0.058,
            n_trades=180,
            dispersion=0.0130,
        )

        self.assertGreater(
            repair_pairwise.fast_validation_robustness_proxy(
                robust,
                target_daily_return=repair_pairwise.TARGET_060_DAILY_RETURN,
            ),
            repair_pairwise.fast_validation_robustness_proxy(
                fragile,
                target_daily_return=repair_pairwise.TARGET_060_DAILY_RETURN,
            ),
        )


if __name__ == "__main__":
    unittest.main()
