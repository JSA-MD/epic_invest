import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import refresh_strategy_inputs as refresh_inputs


class RefreshStrategyInputsTests(unittest.TestCase):
    def test_build_refresh_report_includes_market_context_and_derivatives(self) -> None:
        timestamp = pd.Timestamp("2026-04-13T04:30:00Z")
        derivative_frame = pd.DataFrame({"timestamp": [timestamp], "value": [1.0]})
        with (
            patch.object(
                refresh_inputs,
                "refresh_market_context",
                return_value={"status": "ok", "usable_columns": ["QQQ", "SPY"]},
            ),
            patch.object(
                refresh_inputs,
                "update_derivative_metric_cache",
                return_value=derivative_frame,
            ),
        ):
            report = refresh_inputs.build_refresh_report(
                pairs=("BTCUSDT", "BNBUSDT"),
                context_names=("QQQ", "SPY"),
                lookback_days=7,
            )
        self.assertEqual(report["market_context"]["status"], "ok")
        self.assertEqual(report["derivatives"]["status"], "ok")
        self.assertIn("BTCUSDT", report["derivatives"]["per_pair"])
        sample = report["derivatives"]["per_pair"]["BTCUSDT"]["open_interest"]
        self.assertEqual(sample["rows"], 1)
        self.assertEqual(sample["last_timestamp"], "2026-04-13T04:30:00+00:00")

    def test_build_refresh_report_can_include_ohlcv(self) -> None:
        timestamp = pd.Timestamp("2026-04-13T04:45:00Z")
        ohlcv_frame = pd.DataFrame({"BTCUSDT_close": [1.0]}, index=pd.DatetimeIndex([timestamp], tz="UTC"))
        derivative_frame = pd.DataFrame({"timestamp": [timestamp], "value": [1.0]})
        with (
            patch.object(refresh_inputs, "refresh_market_context", return_value={"status": "ok"}),
            patch.object(refresh_inputs, "update_derivative_metric_cache", return_value=derivative_frame),
            patch.object(refresh_inputs.gp, "load_pair", return_value=ohlcv_frame),
        ):
            report = refresh_inputs.build_refresh_report(
                pairs=("BTCUSDT",),
                context_names=("QQQ",),
                lookback_days=7,
                refresh_ohlcv_inputs=True,
                ohlcv_start="2022-04-06",
            )
        self.assertEqual(report["ohlcv"]["status"], "ok")
        self.assertTrue(report["refresh_ohlcv"])
        self.assertEqual(report["ohlcv"]["per_pair"]["BTCUSDT"]["5m"]["last_timestamp"], "2026-04-13T04:45:00+00:00")

    def test_build_refresh_report_can_include_funding(self) -> None:
        timestamp = pd.Timestamp("2026-04-13T04:48:00Z")
        derivative_frame = pd.DataFrame({"timestamp": [timestamp], "value": [1.0]})
        funding_frame = pd.DataFrame({"fundingTime": [timestamp], "fundingRate": [0.001]})
        with (
            patch.object(refresh_inputs, "refresh_market_context", return_value={"status": "ok"}),
            patch.object(refresh_inputs, "update_derivative_metric_cache", return_value=derivative_frame),
            patch.object(refresh_inputs, "fetch_funding_rates", return_value=funding_frame),
        ):
            report = refresh_inputs.build_refresh_report(
                pairs=("BTCUSDT",),
                context_names=("QQQ",),
                lookback_days=7,
                refresh_funding_inputs=True,
                funding_start="2022-04-06",
            )
        self.assertEqual(report["funding"]["status"], "ok")
        self.assertTrue(report["refresh_funding"])
        self.assertEqual(report["funding"]["per_pair"]["BTCUSDT"]["last_timestamp"], "2026-04-13T04:48:00+00:00")


if __name__ == "__main__":
    unittest.main()
