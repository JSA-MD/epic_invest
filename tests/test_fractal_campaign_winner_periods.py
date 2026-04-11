import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest_fractal_campaign_winner_periods as winner_periods


class FractalCampaignWinnerPeriodsTests(unittest.TestCase):
    def test_select_candidate_entry_prefers_best_when_seed_missing(self) -> None:
        campaign_report = {"best_candidate": {"seed": 10}}
        selected = winner_periods.select_candidate_entry(campaign_report, seed=None)
        self.assertEqual(selected["seed"], 10)

    def test_select_candidate_entry_finds_requested_seed(self) -> None:
        campaign_report = {
            "jobs": [
                {"seed": 11, "artifacts": {"search_summary": "/tmp/a.json"}},
                {"seed": 12, "artifacts": {"search_summary": "/tmp/b.json"}},
            ]
        }
        selected = winner_periods.select_candidate_entry(campaign_report, seed=12)
        self.assertEqual(selected["artifacts"]["search_summary"], "/tmp/b.json")

    def test_infer_last_complete_day_skips_partial_tail(self) -> None:
        full_day = winner_periods.FULL_DAY_BARS_5M
        dates = (
            list(
                winner_periods.pd.date_range("2026-04-09", periods=full_day, freq="5min", tz="UTC")
            )
            + list(winner_periods.pd.date_range("2026-04-10", periods=full_day // 2, freq="5min", tz="UTC"))
        )
        resolved = winner_periods.infer_last_complete_day(winner_periods.pd.DatetimeIndex(dates))
        self.assertEqual(str(resolved.date()), "2026-04-09")


if __name__ == "__main__":
    unittest.main()
