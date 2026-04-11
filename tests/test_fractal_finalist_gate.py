import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_fractal_market_os_finalist_gate as finalist_gate


class FractalFinalistGateTests(unittest.TestCase):
    def make_record(
        self,
        *,
        one_year_worst_daily: float,
        full_4y_worst_daily: float,
        full_4y_mdd: float,
        latest_fold_stress_reserve_score: float,
    ) -> dict[str, object]:
        return {
            "decision": {"ready_for_merge": False},
            "metrics": {
                "wf1_passed": True,
                "latest_fold_stress_reserve_score": latest_fold_stress_reserve_score,
            },
            "blockers": [],
            "period_audit": {
                "one_year_worst_daily": one_year_worst_daily,
                "one_year_positive_pair_count": 1,
                "one_year_weakest_pair": "BNBUSDT",
                "full_4y_worst_daily": full_4y_worst_daily,
                "full_4y_mdd": full_4y_mdd,
            },
        }

    def test_extract_period_snapshot_picks_one_year_weakest_pair(self) -> None:
        report = {
            "periods": [
                {
                    "label": "1y",
                    "aggregate": {
                        "worst_pair_avg_daily_return": -0.0002,
                        "positive_pair_count": 1,
                        "worst_max_drawdown": -0.12,
                    },
                    "per_pair": {
                        "BTCUSDT": {"avg_daily_return": 0.003, "total_return": 1.0, "max_drawdown": -0.08},
                        "BNBUSDT": {"avg_daily_return": -0.0002, "total_return": -0.05, "max_drawdown": -0.12},
                    },
                },
                {
                    "label": "4y",
                    "aggregate": {
                        "worst_pair_avg_daily_return": 0.0047,
                        "positive_pair_count": 2,
                        "worst_max_drawdown": -0.13,
                    },
                },
            ]
        }

        snapshot = finalist_gate.extract_period_snapshot(report)

        self.assertEqual(snapshot["one_year_weakest_pair"], "BNBUSDT")
        self.assertEqual(snapshot["one_year_positive_pair_count"], 1)
        self.assertAlmostEqual(snapshot["full_4y_worst_daily"], 0.0047)

    def test_classify_finalist_record_returns_watch_for_near_frontier_one_year_gap(self) -> None:
        record = {
            "decision": {"ready_for_merge": False},
            "metrics": {
                "wf1_passed": True,
                "latest_fold_stress_reserve_score": -100.0,
            },
            "blockers": ["target_060_failed"],
            "period_audit": {
                "one_year_worst_daily": -0.0002,
                "one_year_positive_pair_count": 1,
                "one_year_weakest_pair": "BNBUSDT",
                "full_4y_worst_daily": 0.0047,
                "full_4y_mdd": 0.12,
            },
        }

        band, repair_theme, blockers = finalist_gate.classify_finalist_record(record)

        self.assertEqual(band, "watch")
        self.assertEqual(repair_theme, "bnb_1y_repair")
        self.assertIn("one_year_worst_daily_failed", blockers)

    def test_rank_finalist_records_prefers_bnb_1y_repair_track(self) -> None:
        bnb_repair = self.make_record(
            one_year_worst_daily=-0.00018,
            full_4y_worst_daily=0.0046,
            full_4y_mdd=0.12,
            latest_fold_stress_reserve_score=-25.0,
        )
        long_horizon = self.make_record(
            one_year_worst_daily=-0.00042,
            full_4y_worst_daily=0.0054,
            full_4y_mdd=0.10,
            latest_fold_stress_reserve_score=250.0,
        )

        ranked = finalist_gate.rank_finalist_records([bnb_repair, long_horizon])
        top_band, top_theme, _ = finalist_gate.classify_finalist_record(ranked[0])

        self.assertEqual(top_band, "watch")
        self.assertEqual(top_theme, "bnb_1y_repair")
        self.assertGreater(
            float(ranked[0]["period_audit"]["one_year_worst_daily"]),
            float(ranked[1]["period_audit"]["one_year_worst_daily"]),
        )

    def test_classify_finalist_record_returns_drop_for_long_horizon_gap(self) -> None:
        record = {
            "decision": {"ready_for_merge": False},
            "metrics": {
                "wf1_passed": True,
                "latest_fold_stress_reserve_score": 500.0,
            },
            "blockers": [],
            "period_audit": {
                "one_year_worst_daily": -0.00019,
                "one_year_positive_pair_count": 1,
                "one_year_weakest_pair": "BNBUSDT",
                "full_4y_worst_daily": 0.0009,
                "full_4y_mdd": 0.18,
            },
        }

        band, repair_theme, blockers = finalist_gate.classify_finalist_record(record)

        self.assertEqual(band, "drop")
        self.assertEqual(repair_theme, "bnb_1y_repair")
        self.assertIn("one_year_pair_breadth_failed", blockers)


if __name__ == "__main__":
    unittest.main()
