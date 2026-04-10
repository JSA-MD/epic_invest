import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import core_market_profile as market_profile


def sample_close() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=8, freq="D")
    return pd.DataFrame(
        {
            "BTCUSDT": [100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 108.0, 110.0],
            "ETHUSDT": [50.0, 50.5, 51.5, 52.0, 53.0, 54.0, 55.5, 56.0],
            "SOLUSDT": [20.0, 19.8, 20.2, 20.8, 21.2, 22.0, 22.5, 23.0],
        },
        index=index,
    )


class CoreMarketProfileTests(unittest.TestCase):
    def test_build_core_market_profile_returns_shared_features(self) -> None:
        close = sample_close()
        context = pd.DataFrame(
            {"QQQ": [400.0, 401.0, 403.0, 402.0, 404.0, 405.0, 407.0, 408.0]},
            index=close.index,
        )
        profile = market_profile.build_core_market_profile(
            close,
            context,
            fast_lookback=1,
            slow_lookback=2,
            vol_window=2,
            corr_window=2,
            regime_threshold=0.01,
            breadth_threshold=0.50,
        )

        self.assertIn("btc_regime", profile["feature_frame"].columns)
        self.assertIn("breadth", profile["feature_frame"].columns)
        self.assertIn("regime_label", profile["feature_frame"].columns)
        self.assertIn("route_bucket", profile["feature_frame"].columns)
        route_state = market_profile.build_route_state_snapshot(profile["feature_frame"], close.index[-1])
        self.assertIn(route_state["route_bucket"], market_profile.ROUTE_BUCKET_LABELS.values())
        corr_snapshot = market_profile.build_context_corr_snapshot(profile["corr_state_profiles"], close.index[-1])
        self.assertEqual(corr_snapshot["source_mode"], "market_context")
        self.assertIn("QQQ", corr_snapshot["contexts"])

    def test_build_corr_state_profiles_uses_internal_fallback(self) -> None:
        close = sample_close()
        profile = market_profile.build_core_market_profile(
            close,
            pd.DataFrame(),
            fast_lookback=1,
            slow_lookback=2,
            vol_window=2,
            corr_window=2,
            regime_threshold=0.01,
            breadth_threshold=0.50,
        )

        corr_profiles = profile["corr_state_profiles"]
        self.assertEqual(corr_profiles["source_mode"], "internal_fallback")
        self.assertIn("internal_alt_basket", corr_profiles["profiles"])


if __name__ == "__main__":
    unittest.main()
