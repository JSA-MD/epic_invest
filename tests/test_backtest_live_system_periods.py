import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backtest_live_system_periods as live_periods


class LiveSystemPeriodBacktestTests(unittest.TestCase):
    def test_infer_last_complete_day_skips_partial_tail(self) -> None:
        full_day = pd.date_range("2026-04-09 00:00:00", periods=288, freq="5min", tz="UTC")
        partial_day = pd.date_range("2026-04-10 00:00:00", periods=37, freq="5min", tz="UTC")
        frame = pd.DataFrame({"BTCUSDT_close": 1.0}, index=full_day.append(partial_day))
        frame.index = frame.index.tz_convert(None)

        end_day = live_periods.infer_last_complete_day(frame)

        self.assertEqual(end_day, pd.Timestamp("2026-04-09"))

    def test_build_live_overlay_signal_map_uses_prior_day_and_respects_core_activity(self) -> None:
        index = pd.date_range("2026-01-01", periods=6, freq="D")
        close = pd.DataFrame(
            {
                "BTCUSDT": [100.0, 101.0, 102.0, 103.0, 100.0, 99.0],
                "ETHUSDT": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
                "SOLUSDT": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                "XRPUSDT": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            },
            index=index,
        )
        core_weights = pd.DataFrame(0.0, index=index, columns=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
        core_weights.loc[pd.Timestamp("2026-01-06"), "BTCUSDT"] = 1.0

        signal_map = live_periods.build_live_overlay_signal_map(
            close,
            core_weights,
            pd.Timestamp("2026-01-01"),
            pd.Timestamp("2026-01-06"),
        )

        self.assertEqual(signal_map[pd.Timestamp("2026-01-01")], 0.0)
        self.assertEqual(signal_map[pd.Timestamp("2026-01-05")], 100.0)
        self.assertEqual(signal_map[pd.Timestamp("2026-01-06")], 0.0)


if __name__ == "__main__":
    unittest.main()
