import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import equity_corr_regime as corr_regime


class EquityCorrRegimeTests(unittest.TestCase):
    def test_resolve_corr_risk_scales(self) -> None:
        self.assertEqual(corr_regime.resolve_corr_risk_scales(0.35), (1.0, 1.0, "equity_aligned"))
        self.assertEqual(corr_regime.resolve_corr_risk_scales(0.0), (1.0, 1.0, "equity_mixed"))
        self.assertEqual(corr_regime.resolve_corr_risk_scales(-0.35), (0.8, 1.1, "equity_inverse"))
        self.assertEqual(corr_regime.resolve_corr_risk_scales(None), (1.0, 1.0, "equity_unknown"))

    def test_build_btc_equity_corr_overlay_prefers_qqq(self) -> None:
        index = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
        close = pd.DataFrame(
            {
                "BTCUSDT": 100.0 + pd.Series(range(40), index=index, dtype="float64"),
                "BNBUSDT": 50.0 + pd.Series(range(40), index=index, dtype="float64"),
            },
            index=index,
        )
        market_context = pd.DataFrame(
            {
                "QQQ": 300.0 + pd.Series(range(40), index=index, dtype="float64"),
                "SPY": 400.0 + pd.Series(range(40), index=index, dtype="float64"),
            },
            index=index,
        )
        status = {"status": "ok", "usable_columns": ["QQQ", "SPY"]}

        with patch.object(corr_regime, "load_market_context_dataset", return_value=(market_context, status)):
            overlay = corr_regime.build_btc_equity_corr_overlay(close)

        self.assertEqual(overlay["equity_corr_context"], "QQQ")
        self.assertEqual(overlay["equity_corr_source_mode"], "market_context")
        self.assertIn("equity_corr_daily", overlay)
        self.assertIn("equity_corr_bucket_daily", overlay)
        self.assertEqual(len(overlay["equity_corr_daily"]), len(close.resample("1D").last().dropna()))


if __name__ == "__main__":
    unittest.main()
