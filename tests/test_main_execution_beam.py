import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_main_execution_beam as beam


def make_windows(
    recent_2m_mean_win: float,
    recent_4m_mean_win: float,
    recent_6m_mean_win: float,
    recent_6m_worst_win: float,
    recent_6m_worst_ret: float,
    recent_6m_mdd: float,
    recent_1y_mean_win: float,
    recent_1y_worst_win: float,
    recent_1y_worst_ret: float,
    recent_1y_mdd: float,
    full_4y_mean_win: float,
    full_4y_worst_win: float,
    full_4y_worst_ret: float,
    full_4y_mdd: float,
) -> dict[str, dict[str, dict[str, float]]]:
    template = {
        "mean_daily_win_rate": 0.50,
        "worst_pair_daily_win_rate": 0.49,
        "worst_pair_avg_daily_return": 0.005,
        "worst_pair_total_return": 1.00,
        "worst_max_drawdown": -0.10,
    }
    windows = {
        label: {"aggregate": dict(template)}
        for label, *_ in beam.SEARCH_WINDOWS
    }
    windows["recent_2m"]["aggregate"]["mean_daily_win_rate"] = recent_2m_mean_win
    windows["recent_4m"]["aggregate"]["mean_daily_win_rate"] = recent_4m_mean_win
    windows["recent_6m"]["aggregate"].update(
        {
            "mean_daily_win_rate": recent_6m_mean_win,
            "worst_pair_daily_win_rate": recent_6m_worst_win,
            "worst_pair_total_return": recent_6m_worst_ret,
            "worst_max_drawdown": recent_6m_mdd,
        }
    )
    windows["recent_1y"]["aggregate"].update(
        {
            "mean_daily_win_rate": recent_1y_mean_win,
            "worst_pair_daily_win_rate": recent_1y_worst_win,
            "worst_pair_total_return": recent_1y_worst_ret,
            "worst_max_drawdown": recent_1y_mdd,
        }
    )
    windows["full_4y"]["aggregate"].update(
        {
            "mean_daily_win_rate": full_4y_mean_win,
            "worst_pair_daily_win_rate": full_4y_worst_win,
            "worst_pair_total_return": full_4y_worst_ret,
            "worst_max_drawdown": full_4y_mdd,
        }
    )
    return windows


class MainExecutionBeamTests(unittest.TestCase):
    def test_evaluate_guard_flags_retention_and_traffic_light(self) -> None:
        baseline = make_windows(0.49, 0.50, 0.49, 0.48, 2.50, -0.10, 0.48, 0.46, 4.00, -0.10, 0.49, 0.45, 900.0, -0.12)
        candidate = make_windows(0.51, 0.51, 0.50, 0.49, 2.52, -0.10, 0.49, 0.47, 4.05, -0.10, 0.50, 0.46, 905.0, -0.12)
        compare = beam.compare_to_baseline(candidate, baseline)

        guard = beam.evaluate_guard(
            candidate,
            baseline,
            compare,
            return_retention_floor=0.995,
            drawdown_ratio_cap=1.02,
        )

        self.assertTrue(guard["guard_pass"])
        self.assertTrue(guard["strict_nonworse_pass"])
        self.assertEqual(guard["traffic_light"], "green")

    def test_winrate_guarded_score_penalizes_tail_damage(self) -> None:
        baseline = make_windows(0.49, 0.50, 0.49, 0.48, 2.50, -0.10, 0.48, 0.46, 4.00, -0.10, 0.49, 0.45, 900.0, -0.12)
        guarded = make_windows(0.52, 0.52, 0.50, 0.49, 2.52, -0.10, 0.49, 0.47, 4.02, -0.10, 0.50, 0.46, 905.0, -0.12)
        tail_broken = make_windows(0.54, 0.54, 0.53, 0.52, 2.20, -0.11, 0.52, 0.50, 3.70, -0.11, 0.52, 0.48, 850.0, -0.13)

        guarded_score = beam.candidate_score(
            guarded,
            baseline,
            objective="winrate_guarded",
            return_retention_floor=0.995,
            drawdown_ratio_cap=1.02,
        )
        tail_broken_score = beam.candidate_score(
            tail_broken,
            baseline,
            objective="winrate_guarded",
            return_retention_floor=0.995,
            drawdown_ratio_cap=1.02,
        )

        self.assertGreater(guarded_score, tail_broken_score)

    def test_build_seed_candidates_loads_extra_seed_results(self) -> None:
        baseline_candidate = {
            "candidate_kind": "pairwise_candidate",
            "pair_configs": {
                "BTCUSDT": {"route_breadth_threshold": 0.5, "route_state_mode": "equity_corr", "mapping_indices": [10] * 12},
                "BNBUSDT": {"route_breadth_threshold": 0.5, "route_state_mode": "equity_corr", "mapping_indices": [20] * 12},
            },
        }
        extra_candidate = {
            "candidate_kind": "pairwise_candidate",
            "pair_configs": {
                "BTCUSDT": {
                    "route_breadth_threshold": 0.5,
                    "route_state_mode": "equity_corr",
                    "mapping_indices": [10] * 12,
                    "execution_gene": {"maker_priority": 0.85},
                },
                "BNBUSDT": {"route_breadth_threshold": 0.5, "route_state_mode": "equity_corr", "mapping_indices": [20] * 12},
            },
        }
        with TemporaryDirectory() as tmpdir:
            extra_path = Path(tmpdir) / "extra.json"
            extra_path.write_text(
                beam.json.dumps(
                    {
                        "top_guard_passed": [
                            {"candidate": extra_candidate}
                        ]
                    }
                )
            )
            seeds = beam.build_seed_candidates(
                baseline_candidate,
                pairs=("BTCUSDT", "BNBUSDT"),
                pair_gate_grid=Path(tmpdir) / "missing_pair.json",
                role_gate_grid=Path(tmpdir) / "missing_role.json",
                global_gate_grid=Path(tmpdir) / "missing_global.json",
                extra_seed_results=(extra_path,),
                top_k=2,
            )

        self.assertTrue(
            any(
                seed["pair_configs"]["BTCUSDT"].get("execution_gene", {}).get("maker_priority") == 0.85
                for seed in seeds
            )
        )


if __name__ == "__main__":
    unittest.main()
