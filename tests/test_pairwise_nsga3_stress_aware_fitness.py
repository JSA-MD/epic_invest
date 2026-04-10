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


def build_report_with_pairs(
    *,
    recent_2m: tuple[float, float, float, float],
    recent_6m: tuple[float, float, float, float],
    full_4y: tuple[float, float, float, float],
    hit_rate: float,
    win_rate: float,
    sharpe: float,
    worst_day: float,
    n_trades: int,
) -> dict[str, object]:
    report = build_report(recent_2m=recent_2m, recent_6m=recent_6m, full_4y=full_4y)
    for key in ("recent_2m", "recent_6m", "full_4y"):
        report["windows"][key]["per_pair"] = {  # type: ignore[index]
            "BTCUSDT": {
                "daily_target_hit_rate": hit_rate,
                "daily_win_rate": win_rate,
                "sharpe": sharpe,
                "worst_day": worst_day,
                "n_trades": n_trades,
                "avg_daily_return": report["windows"][key]["aggregate"]["worst_pair_avg_daily_return"],  # type: ignore[index]
            },
            "BNBUSDT": {
                "daily_target_hit_rate": hit_rate,
                "daily_win_rate": win_rate,
                "sharpe": sharpe,
                "worst_day": worst_day,
                "n_trades": n_trades,
                "avg_daily_return": report["windows"][key]["aggregate"]["worst_pair_avg_daily_return"],  # type: ignore[index]
            },
        }
        report["windows"][key]["aggregate"]["positive_pair_count"] = 2  # type: ignore[index]
    return report


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

    def test_fast_scalar_score_penalizes_fragile_validation_proxy(self) -> None:
        robust = build_report_with_pairs(
            recent_2m=(0.0068, 0.0070, -0.05, 0.0020),
            recent_6m=(0.0069, 0.0071, -0.08, 0.0025),
            full_4y=(0.0067, 0.0069, -0.12, 0.0030),
            hit_rate=0.60,
            win_rate=0.63,
            sharpe=1.8,
            worst_day=-0.018,
            n_trades=42,
        )
        fragile = build_report_with_pairs(
            recent_2m=(0.0068, 0.0070, -0.05, 0.0110),
            recent_6m=(0.0069, 0.0071, -0.08, 0.0120),
            full_4y=(0.0067, 0.0069, -0.24, 0.0140),
            hit_rate=0.42,
            win_rate=0.45,
            sharpe=0.4,
            worst_day=-0.055,
            n_trades=180,
        )

        self.assertGreater(
            pairwise_nsga3.fast_scalar_score(robust["windows"]),  # type: ignore[index]
            pairwise_nsga3.fast_scalar_score(fragile["windows"]),  # type: ignore[index]
        )


if __name__ == "__main__":
    unittest.main()
