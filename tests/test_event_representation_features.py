import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gp_crypto_evolution as gp
import search_pair_subset_regime_mixture as regime
import search_pair_subset_fractal_genome as fractal
import train_daily_participation_filter as participation


def build_ohlcv_frame(price_scale: float, taker_share: float) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=60, freq="5min", tz="UTC")
    pattern = np.asarray(
        [100.0, 100.25, 100.55, 100.10, 99.40, 98.85, 99.10, 99.60, 100.20, 101.15, 100.35, 99.55],
        dtype="float64",
    )
    close = np.tile(pattern * price_scale, 5)[: len(index)]
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = np.linspace(1_000.0, 1_600.0, len(index), dtype="float64")
    taker_base = volume * taker_share
    taker_quote = taker_base * close
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "taker_base": taker_base,
            "taker_quote": taker_quote,
        },
        index=index,
    )


class EventRepresentationFeatureTests(unittest.TestCase):
    def test_enrich_features_adds_multi_threshold_dc_and_order_imbalance(self) -> None:
        enriched = gp.enrich_features(build_ohlcv_frame(price_scale=1.0, taker_share=0.75))

        required = {
            "buy_volume_share",
            "order_imbalance",
            "dc_event_015",
            "dc_event_03",
            "dc_event_05",
            "dc_event_10",
            "dc_trend_015",
            "dc_trend_10",
        }
        self.assertTrue(required.issubset(set(enriched.columns)))
        self.assertAlmostEqual(float(enriched["buy_volume_share"].iloc[-1]), 0.75, places=6)
        self.assertAlmostEqual(float(enriched["order_imbalance"].iloc[-1]), 0.50, places=6)
        self.assertTrue((enriched["dc_event_015"].abs() > 0.0).any())
        self.assertTrue((enriched["dc_event_10"].abs() > 0.0).any())

    def test_enrich_features_defaults_missing_taker_flow_to_neutral(self) -> None:
        raw = build_ohlcv_frame(price_scale=1.0, taker_share=0.75).drop(columns=["taker_base", "taker_quote"])
        enriched = gp.enrich_features(raw)

        self.assertTrue(np.allclose(enriched["buy_volume_share"], 0.5))
        self.assertTrue(np.allclose(enriched["order_imbalance"], 0.0))
        self.assertTrue(np.allclose(enriched["taker_base"], enriched["volume"] * 0.5))
        self.assertTrue(np.allclose(enriched["taker_quote"], enriched["taker_base"] * enriched["close"]))

    def test_build_market_features_exposes_event_and_imbalance_family(self) -> None:
        btc = gp.enrich_features(build_ohlcv_frame(price_scale=1.0, taker_share=0.75)).add_prefix("BTCUSDT_")
        bnb = gp.enrich_features(build_ohlcv_frame(price_scale=2.0, taker_share=0.35)).add_prefix("BNBUSDT_")
        merged = pd.concat([btc, bnb], axis=1)

        features = fractal.build_market_features(merged, ("BTCUSDT", "BNBUSDT"))

        required = {
            "btc_order_imbalance_1h",
            "bnb_order_imbalance_1h",
            "imbalance_spread_btc_minus_bnb_1h",
            "btc_dc_event_015_1h",
            "btc_dc_event_10_1h",
            "bnb_dc_event_015_1h",
            "bnb_dc_event_10_1h",
        }
        self.assertTrue(required.issubset(set(features)))
        self.assertAlmostEqual(float(features["btc_order_imbalance_1h"].iloc[-1]), 0.50, places=6)
        self.assertAlmostEqual(float(features["bnb_order_imbalance_1h"].iloc[-1]), -0.30, places=6)
        self.assertAlmostEqual(float(features["imbalance_spread_btc_minus_bnb_1h"].iloc[-1]), 0.80, places=6)
        self.assertTrue((features["btc_dc_event_015_1h"].abs() > 0.0).any())
        self.assertTrue((features["bnb_dc_event_10_1h"].abs() > 0.0).any())

    def test_materialize_feature_arrays_uses_feature_specific_neutrals(self) -> None:
        index = pd.date_range("2026-02-01", periods=3, freq="5min", tz="UTC")
        features = {
            "btc_dc_event_05_1h": pd.Series([np.nan], index=[index[1]]),
            "btc_volume_rel_1h": pd.Series([np.nan], index=[index[1]]),
            "btc_rsi_14_1h": pd.Series([np.nan], index=[index[1]]),
            "btc_bb_p_1h": pd.Series([np.nan], index=[index[1]]),
            "session_asia_flag": pd.Series([np.nan], index=[index[1]]),
            "btc_vol_rel": pd.Series([np.nan], index=[index[0]]),
            "rsi_spread_btc_minus_bnb_1h": pd.Series([np.nan], index=[index[1]]),
            "mfi_spread_btc_minus_bnb_1h": pd.Series([np.nan], index=[index[1]]),
            "bb_pct_b_spread_btc_minus_bnb_20_2": pd.Series([np.nan], index=[index[0].normalize()]),
            "return_spread_btc_minus_bnb_1d": pd.Series([np.nan], index=[index[0].normalize()]),
        }

        arrays = fractal.materialize_feature_arrays(features, index)

        self.assertTrue(np.allclose(arrays["btc_dc_event_05_1h"], 0.0))
        self.assertTrue(np.allclose(arrays["btc_volume_rel_1h"], 1.0))
        self.assertTrue(np.allclose(arrays["btc_rsi_14_1h"], 50.0))
        self.assertTrue(np.allclose(arrays["btc_bb_p_1h"], 0.5))
        self.assertTrue(np.allclose(arrays["session_asia_flag"], 0.0))
        self.assertTrue(np.allclose(arrays["btc_vol_rel"], 1.0))
        self.assertTrue(np.allclose(arrays["rsi_spread_btc_minus_bnb_1h"], 0.0))
        self.assertTrue(np.allclose(arrays["mfi_spread_btc_minus_bnb_1h"], 0.0))
        self.assertTrue(np.allclose(arrays["bb_pct_b_spread_btc_minus_bnb_20_2"], 0.0))
        self.assertTrue(np.allclose(arrays["return_spread_btc_minus_bnb_1d"], 0.0))

    def test_participation_filter_adds_volume_relative_context(self) -> None:
        merged_frames = []
        for idx, pair in enumerate(gp.PAIRS):
            frame = gp.enrich_features(build_ohlcv_frame(price_scale=1.0 + idx, taker_share=0.55)).add_prefix(f"{pair}_")
            merged_frames.append(frame)
        row = pd.concat(merged_frames, axis=1).iloc[-1]

        names, values = participation.build_feature_vector(row, signal=25.0)

        volume_rel_name = f"{gp.PAIRS[0]}_volume_rel"
        self.assertIn(volume_rel_name, names)
        feature_idx = names.index(volume_rel_name)
        expected = float(row[f"{gp.PAIRS[0]}_volume"]) / max(abs(float(row[f"{gp.PAIRS[0]}_vol_sma"])), 1e-8)
        self.assertAlmostEqual(values[feature_idx], expected, places=6)

    def test_participation_filter_clamps_volume_relative_context_when_sma_missing(self) -> None:
        merged_frames = []
        for idx, pair in enumerate(gp.PAIRS):
            frame = gp.enrich_features(build_ohlcv_frame(price_scale=1.0 + idx, taker_share=0.55)).add_prefix(f"{pair}_")
            merged_frames.append(frame)
        row = pd.concat(merged_frames, axis=1).iloc[-1].copy()
        row[f"{gp.PAIRS[0]}_vol_sma"] = 0.0

        names, values = participation.build_feature_vector(row, signal=10.0)

        feature_idx = names.index(f"{gp.PAIRS[0]}_volume_rel")
        self.assertEqual(values[feature_idx], 1.0)

    def test_fast_context_defaults_equity_corr_inputs_for_single_asset_paths(self) -> None:
        btc = gp.enrich_features(build_ohlcv_frame(price_scale=1.0, taker_share=0.60)).add_prefix("BTCUSDT_")
        overlay_inputs = {
            "btc_regime_daily": pd.Series(0.10, index=btc.index.normalize().unique(), dtype="float64"),
            "breadth_daily": pd.Series(0.60, index=btc.index.normalize().unique(), dtype="float64"),
            "vol_ann_bar": pd.Series(0.20, index=btc.index, dtype="float64"),
        }
        context = regime.build_fast_context(
            btc,
            "BTCUSDT",
            raw_signal=pd.Series(np.linspace(0.0, 1.0, len(btc)), index=btc.index),
            overlay_inputs=overlay_inputs,
            route_thresholds=(0.5,),
            library_lookup={"spans": (2,)},
            route_state_mode=regime.ROUTE_STATE_MODE_BASE,
        )

        self.assertTrue(np.allclose(context["equity_corr"], 0.0))
        self.assertTrue(np.allclose(context["equity_corr_gross_scale"], 1.0))
        self.assertTrue(np.allclose(context["equity_corr_regime_mult"], 1.0))
        self.assertTrue(np.isnan(context["btc_qqq_corr_5d"]).all())
        self.assertTrue(np.isnan(context["btc_qqq_corr_20d"]).all())
        self.assertTrue(np.isnan(context["btc_spy_beta_20d"]).all())


if __name__ == "__main__":
    unittest.main()
