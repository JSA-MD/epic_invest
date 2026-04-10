import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_pair_subset_pairwise_nsga3 as pairwise_nsga3


def build_window(
    worst_pair_avg_daily_return: float,
    mean_avg_daily_return: float,
    worst_max_drawdown: float,
    pair_return_dispersion: float,
) -> dict[str, dict[str, float]]:
    return {
        "aggregate": {
            "worst_pair_avg_daily_return": worst_pair_avg_daily_return,
            "mean_avg_daily_return": mean_avg_daily_return,
            "worst_max_drawdown": worst_max_drawdown,
            "pair_return_dispersion": pair_return_dispersion,
        }
    }


def build_aggregate(
    worst_pair_avg_daily_return: float,
    mean_avg_daily_return: float,
    worst_max_drawdown: float,
    pair_return_dispersion: float,
) -> dict[str, float]:
    return {
        "worst_pair_avg_daily_return": worst_pair_avg_daily_return,
        "mean_avg_daily_return": mean_avg_daily_return,
        "worst_max_drawdown": worst_max_drawdown,
        "pair_return_dispersion": pair_return_dispersion,
    }


def build_report(
    recent_2m: tuple[float, float, float, float],
    recent_6m: tuple[float, float, float, float],
    full_4y: tuple[float, float, float, float],
) -> dict[str, object]:
    return {
        "windows": {
            "recent_2m": build_window(*recent_2m),
            "recent_6m": build_window(*recent_6m),
            "full_4y": build_window(*full_4y),
        }
    }


def stress_proxy_reserve(report: dict[str, object]) -> float:
    windows = report["windows"]  # type: ignore[index]
    candidates = []
    for key in ("recent_2m", "recent_6m", "full_4y"):
        aggregate = windows[key]["aggregate"]  # type: ignore[index]
        candidates.append(float(aggregate["worst_pair_avg_daily_return"]) - 0.006)
    return min(candidates)


class PairwiseNsga3StressAwareFitnessTests(unittest.TestCase):
    def test_fast_scalar_score_prefers_higher_stress_proxy_reserve(self) -> None:
        weaker = {
            "recent_2m": build_aggregate(0.0063, 0.0066, -0.04, 0.0030),
            "recent_6m": build_aggregate(0.0065, 0.0069, -0.08, 0.0035),
            "full_4y": build_aggregate(0.0062, 0.0064, -0.10, 0.0040),
        }
        stronger = {
            "recent_2m": build_aggregate(0.0084, 0.0087, -0.03, 0.0020),
            "recent_6m": build_aggregate(0.0082, 0.0086, -0.07, 0.0025),
            "full_4y": build_aggregate(0.0081, 0.0085, -0.09, 0.0030),
        }

        self.assertGreater(
            stress_proxy_reserve(build_report(
                recent_2m=(0.0084, 0.0087, -0.03, 0.0020),
                recent_6m=(0.0082, 0.0086, -0.07, 0.0025),
                full_4y=(0.0081, 0.0085, -0.09, 0.0030),
            )),
            stress_proxy_reserve(build_report(
                recent_2m=(0.0063, 0.0066, -0.04, 0.0030),
                recent_6m=(0.0065, 0.0069, -0.08, 0.0035),
                full_4y=(0.0062, 0.0064, -0.10, 0.0040),
            )),
        )
        self.assertGreater(pairwise_nsga3.fast_scalar_score(stronger), pairwise_nsga3.fast_scalar_score(weaker))

    def test_selection_penalizes_full_4y_drag_in_realistic_score(self) -> None:
        bnb_dragged = build_report(
            recent_2m=(0.0072, 0.0075, -0.05, 0.0025),
            recent_6m=(0.0078, 0.0080, -0.09, 0.0030),
            full_4y=(0.0046, 0.0050, -0.24, 0.0100),
        )
        reserve_favored = build_report(
            recent_2m=(0.0073, 0.0076, -0.04, 0.0020),
            recent_6m=(0.0081, 0.0084, -0.08, 0.0025),
            full_4y=(0.0068, 0.0071, -0.13, 0.0030),
        )

        ranked = max([bnb_dragged, reserve_favored], key=pairwise_nsga3.score_realistic_candidate)

        self.assertIs(ranked, reserve_favored)
        self.assertGreater(
            pairwise_nsga3.score_realistic_candidate(reserve_favored),
            pairwise_nsga3.score_realistic_candidate(bnb_dragged),
        )


if __name__ == "__main__":
    unittest.main()
