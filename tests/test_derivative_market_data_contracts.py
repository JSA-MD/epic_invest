import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import gp_crypto_evolution as gp
import derivative_market_data as derivatives
import search_pair_subset_fractal_genome as fractal
import search_pair_subset_regime_mixture as regime
from fractal_genome_core import ConditionNode, ConditionSpec, LeafNode, ThresholdCell


def build_price_frame(price_scale: float, taker_share: float) -> pd.DataFrame:
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


def build_multi_day_price_frame(first_day_price: float, second_day_price: float, taker_share: float) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=16 * 288, freq="5min", tz="UTC")
    close = np.concatenate(
        [
            np.full(15 * 288, first_day_price, dtype="float64"),
            np.full(288, second_day_price, dtype="float64"),
        ]
    )
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = np.linspace(1_000.0, 2_000.0, len(index), dtype="float64")
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


class DerivativeMarketDataContractTests(unittest.TestCase):
    def test_fetch_derivative_metric_pages_by_period_window(self) -> None:
        captured_params: list[dict[str, object]] = []
        start_dt = pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime()
        end_dt = pd.Timestamp("2026-04-04T00:00:00Z").to_pydatetime()
        end_ms = int(pd.Timestamp(end_dt).timestamp() * 1000)

        def fake_http_get_json(path: str, params: dict[str, object]) -> list[dict[str, object]]:
            captured_params.append(dict(params))
            timestamp = int(params["endTime"])
            if timestamp > end_ms:
                return []
            return [
                {
                    "symbol": "BTCUSDT",
                    "sumOpenInterest": "100.0",
                    "sumOpenInterestValue": "200.0",
                    "timestamp": timestamp,
                }
            ]

        with mock.patch.object(derivatives, "_http_get_json", side_effect=fake_http_get_json):
            with mock.patch.object(derivatives.time, "sleep"):
                loaded = derivatives.fetch_derivative_metric(
                    "BTCUSDT",
                    "open_interest",
                    start_dt=start_dt,
                    end_dt=end_dt,
                    lookback_days=7,
                )

        self.assertGreaterEqual(len(captured_params), 2)
        self.assertLess(int(captured_params[0]["endTime"]), end_ms)
        self.assertEqual(len(loaded), len(captured_params))

    def test_normalize_metric_frame_preserves_canonical_columns_and_millisecond_timestamps(self) -> None:
        normalized = derivatives._normalize_metric_frame(
            "open_interest",
            pd.DataFrame(
                {
                    "timestamp": ["1775893800000", "2026-04-10T23:55:00Z"],
                    "open_interest": ["96708.35600000", "96864.86100000"],
                    "open_interest_value": ["7026983880.32960000", "7037958437.00733100"],
                }
            ),
        )

        self.assertEqual(list(normalized.columns), ["timestamp", "open_interest", "open_interest_value"])
        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["timestamp"].iloc[0], pd.Timestamp("2026-04-10 23:55:00+00:00"))
        self.assertEqual(normalized["timestamp"].iloc[1], pd.Timestamp("2026-04-11 07:50:00+00:00"))
        self.assertAlmostEqual(float(normalized["open_interest"].iloc[1]), 96708.356, places=6)
        self.assertAlmostEqual(float(normalized["open_interest_value"].iloc[0]), 7037958437.007331, places=6)

    def test_normalize_metric_frame_preserves_existing_datetime_timestamps(self) -> None:
        normalized = derivatives._normalize_metric_frame(
            "open_interest",
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-10T23:55:00Z", "2026-04-11T00:00:00Z"],
                        utc=True,
                    ),
                    "open_interest": [96864.861, 97125.236],
                    "open_interest_value": [7037958437.007331, 7082119938.028627],
                }
            ),
        )

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["timestamp"].iloc[0], pd.Timestamp("2026-04-10 23:55:00+00:00"))
        self.assertEqual(normalized["timestamp"].iloc[1], pd.Timestamp("2026-04-11 00:00:00+00:00"))
        self.assertAlmostEqual(float(normalized["open_interest"].iloc[0]), 96864.861, places=6)

    def test_summarize_derivative_feature_coverage_tracks_non_null_ratio(self) -> None:
        index = pd.date_range("2026-04-10", periods=4, freq="5min", tz="UTC")
        coverage = fractal.summarize_derivative_feature_coverage(
            {
                "btc_top_pos_log_ratio_1h": pd.Series([np.nan, 0.04, 0.05, np.nan], index=index),
                "basis_rate_spread_btc_minus_bnb_1h": pd.Series([0.0, 0.001, np.nan, -0.001], index=index),
                "btc_return_1d": pd.Series([0.0, 0.0, 0.0, 0.0], index=index),
            },
            index,
        )

        self.assertEqual(coverage["aggregate"]["feature_count"], 2)
        self.assertEqual(coverage["aggregate"]["features_with_signal_count"], 2)
        self.assertAlmostEqual(
            coverage["per_feature"]["btc_top_pos_log_ratio_1h"]["non_null_ratio"],
            0.5,
            places=6,
        )
        self.assertEqual(
            coverage["per_feature"]["basis_rate_spread_btc_minus_bnb_1h"]["last_valid_timestamp"],
            pd.Timestamp(index[-1]).isoformat(),
        )

    def test_summarize_tree_condition_activity_tracks_derivative_branches(self) -> None:
        tree = ConditionNode(
            condition=ThresholdCell(
                ConditionSpec(
                    feature="btc_top_pos_log_ratio_1h",
                    comparator=">=",
                    threshold=0.04,
                )
            ),
            if_true=LeafNode(0),
            if_false=LeafNode(1),
        )
        activity = fractal.summarize_tree_condition_activity(
            tree,
            {
                "btc_top_pos_log_ratio_1h": np.asarray([0.03, 0.04, 0.07], dtype="float64"),
            },
        )

        self.assertEqual(activity["condition_count"], 1)
        self.assertEqual(activity["derivative_condition_count"], 1)
        self.assertEqual(activity["derivative_feature_names"], ["btc_top_pos_log_ratio_1h"])
        self.assertEqual(activity["conditions"][0]["true_count"], 2)
        self.assertEqual(activity["conditions"][0]["false_count"], 1)

    def test_summarize_derivative_selection_profile_rewards_live_coverage(self) -> None:
        activity = {
            "condition_count": 1,
            "derivative_condition_count": 1,
            "derivative_feature_names": ["btc_top_pos_log_ratio_1h"],
            "conditions": [
                {
                    "derivative_condition": True,
                    "derivative_feature_names": ["btc_top_pos_log_ratio_1h"],
                    "active_ratio": 0.65,
                    "true_ratio_within_active": 0.55,
                }
            ],
        }
        full_coverage = {
            "per_feature": {
                "btc_top_pos_log_ratio_1h": {"non_null_ratio": 0.05},
            }
        }
        latest_coverage = {
            "per_feature": {
                "btc_top_pos_log_ratio_1h": {"non_null_ratio": 0.45},
            }
        }

        profile = fractal.summarize_derivative_selection_profile(
            activity,
            full_coverage,
            latest_coverage,
        )

        self.assertEqual(profile["condition_count"], 1)
        self.assertEqual(profile["feature_count"], 1)
        self.assertGreater(profile["score"], 1.0)
        self.assertAlmostEqual(profile["latest_feature_coverage"], 0.45, places=6)

    def test_derivative_features_remain_time_mode_inputs(self) -> None:
        self.assertEqual(
            fractal.classify_feature_observation_mode("btc_top_pos_log_ratio_1h"),
            fractal.OBSERVATION_MODE_TIME,
        )
        volume_specs = fractal.filter_feature_specs_by_observation_mode(
            (
                ("btc_top_pos_log_ratio_1h", ">=", (0.0, 0.04)),
                ("btc_volume_rel_1h", ">=", (1.0,)),
            ),
            fractal.OBSERVATION_MODE_VOLUME,
        )
        self.assertEqual([spec[0] for spec in volume_specs], ["btc_volume_rel_1h"])

    def test_select_near_frontier_structural_winner_prefers_derivative_profile_when_frontier_is_close(self) -> None:
        base_item = {
            "validation": {},
            "robustness": {
                "stress_survival_threshold": 0.67,
                "min_fold_non_nominal_stress_survival_rate": 0.0,
                "min_fold_stress_survival_rate": 0.0,
                "latest_non_nominal_stress_reserve_score": 0.0,
                "latest_fold_stress_reserve_score": 0.0,
                "latest_fold_delta_worst_pair_total_return": 0.0,
                "latest_fold_delta_worst_max_drawdown": 0.0,
                "latest_fold_delta_worst_daily_win_rate": 0.0,
            },
            "structural_score": 1.0,
            "logic_depth": 1,
            "tree_depth": 1,
            "leaf_cardinality": 2,
            "fitness": -1000.0,
        }
        stronger_derivative = {
            **base_item,
            "tree_key": "derivative",
            "performance_score": -1000.0,
            "derivative_profile": {"score": 2.0, "latest_feature_coverage": 0.4, "condition_count": 1},
        }
        weaker_derivative = {
            **base_item,
            "tree_key": "plain",
            "performance_score": -1010.0,
            "derivative_profile": {"score": 0.0, "latest_feature_coverage": 0.0, "condition_count": 0},
        }

        winner, diagnostics = fractal.select_near_frontier_structural_winner(
            [weaker_derivative, stronger_derivative],
            frontier_ratio=0.05,
            frontier_floor=20.0,
        )

        self.assertIsNotNone(winner)
        self.assertEqual(winner["tree_key"], "derivative")
        self.assertEqual(diagnostics["frontier_count"], 2)

    def test_fetch_derivative_metric_rejects_error_object_payloads(self) -> None:
        with mock.patch.object(
            derivatives,
            "_http_get_json",
            return_value={"code": -1130, "msg": "parameter 'startTime' is invalid."},
        ):
            with self.assertRaisesRegex(RuntimeError, "Unexpected Binance derivatives response"):
                derivatives.fetch_derivative_metric(
                    "BTCUSDT",
                    "open_interest",
                    start_dt=pd.Timestamp("2026-03-01T00:00:00Z").to_pydatetime(),
                    end_dt=pd.Timestamp("2026-03-15T00:00:00Z").to_pydatetime(),
                )

    def test_fetch_derivative_metric_retries_rate_limit_responses(self) -> None:
        row = {
            "symbol": "BTCUSDT",
            "sumOpenInterest": "96708.35600000",
            "sumOpenInterestValue": "7026983880.32960000",
            "timestamp": 1775893800000,
        }
        with mock.patch.object(
            derivatives,
            "_http_get_json",
            side_effect=[
                {"code": -1003, "msg": "Too many requests"},
                [row],
                [],
            ],
        ):
            with mock.patch.object(derivatives.time, "sleep") as sleep_mock:
                loaded = derivatives.fetch_derivative_metric(
                    "BTCUSDT",
                    "open_interest",
                    start_dt=pd.Timestamp("2026-04-10T00:00:00Z").to_pydatetime(),
                    end_dt=pd.Timestamp("2026-04-11T00:00:00Z").to_pydatetime(),
                )

        self.assertEqual(len(loaded), 1)
        self.assertAlmostEqual(float(loaded["open_interest"].iloc[0]), 96708.356, places=6)
        self.assertGreaterEqual(sleep_mock.call_count, 1)

    def test_update_derivative_metric_cache_backfills_requested_lookback_when_cache_is_too_short(self) -> None:
        existing = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-04-10T00:00:00Z", "2026-04-11T00:00:00Z"],
                    utc=True,
                ),
                "open_interest": [1.0, 2.0],
                "open_interest_value": [10.0, 20.0],
            }
        )
        fetched = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-04-04T00:00:00Z", "2026-04-11T00:00:00Z"],
                    utc=True,
                ),
                "open_interest": [3.0, 4.0],
                "open_interest_value": [30.0, 40.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = derivatives.DERIVATIVE_CACHE_DIR
            derivatives.DERIVATIVE_CACHE_DIR = Path(tmpdir)
            try:
                with mock.patch.object(derivatives, "load_derivative_metric_cache", return_value=existing):
                    with mock.patch.object(derivatives, "fetch_derivative_metric", return_value=fetched) as fetch_mock:
                        merged = derivatives.update_derivative_metric_cache(
                            "BTCUSDT",
                            "open_interest",
                            end_dt=pd.Timestamp("2026-04-11T00:00:00Z").to_pydatetime(),
                            lookback_days=7,
                        )
            finally:
                derivatives.DERIVATIVE_CACHE_DIR = original_dir

        self.assertEqual(fetch_mock.call_args.kwargs["start_dt"], pd.Timestamp("2026-04-04T00:00:00Z").to_pydatetime())
        self.assertEqual(len(merged), 3)

    def test_load_or_fetch_funding_normalizes_mixed_timestamps_and_numeric_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            original_data_dir = gp.DATA_DIR
            gp.DATA_DIR = data_dir
            try:
                path = data_dir / "BTCUSDT_funding_2026-01-01_2026-01-02.csv"
                pd.DataFrame(
                    {
                        "fundingTime": [
                            "2026-01-02T00:00:00Z",
                            "2026-01-01T08:00:00-08:00",
                            "2026-01-01T16:00:00+00:00",
                        ],
                        "fundingRate": ["0.0002", "0.0001", "bad"],
                        "openInterest": [10, 11, 12],
                    }
                ).to_csv(path, index=False)

                loaded = regime.load_or_fetch_funding("BTCUSDT", "2026-01-01", "2026-01-02")
            finally:
                gp.DATA_DIR = original_data_dir

        self.assertEqual(list(loaded.columns), ["fundingTime", "fundingRate", "openInterest"])
        self.assertEqual(len(loaded), 2)
        self.assertTrue(loaded["fundingTime"].is_monotonic_increasing)
        self.assertTrue(all(ts.tz is not None for ts in loaded["fundingTime"]))
        self.assertTrue(np.allclose(loaded["fundingRate"].to_numpy(dtype="float64"), [0.0001, 0.0002]))

    def test_load_funding_from_cache_or_empty_uses_matching_candidate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            original_data_dir = gp.DATA_DIR
            gp.DATA_DIR = data_dir
            try:
                path = data_dir / "BTCUSDT_funding_2026-01-01_2026-01-03.csv"
                pd.DataFrame(
                    {
                        "fundingTime": [
                            "2026-01-01T00:00:00Z",
                            "2026-01-02T00:00:00Z",
                            "2026-01-03T00:00:00Z",
                        ],
                        "fundingRate": [0.0001, 0.0002, 0.0003],
                    }
                ).to_csv(path, index=False)

                loaded = fractal.load_funding_from_cache_or_empty("BTCUSDT", "2026-01-01", "2026-01-02")
            finally:
                gp.DATA_DIR = original_data_dir

        self.assertEqual(list(loaded.columns), ["fundingTime", "fundingRate"])
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded["fundingTime"].iloc[0], pd.Timestamp("2026-01-01T00:00:00Z"))
        self.assertEqual(loaded["fundingTime"].iloc[1], pd.Timestamp("2026-01-02T00:00:00Z"))
        self.assertEqual(loaded["fundingTime"].iloc[2], pd.Timestamp("2026-01-03T00:00:00Z"))

    def test_extra_derivative_columns_do_not_change_existing_feature_paths(self) -> None:
        base_btc = gp.enrich_features(build_price_frame(price_scale=1.0, taker_share=0.75)).add_prefix("BTCUSDT_")
        base_bnb = gp.enrich_features(build_price_frame(price_scale=2.0, taker_share=0.35)).add_prefix("BNBUSDT_")
        augmented = pd.concat([base_btc, base_bnb], axis=1)
        augmented["BTCUSDT_open_interest"] = 123.0
        augmented["BTCUSDT_top_trader_position_ratio"] = 0.58
        augmented["BTCUSDT_top_trader_account_ratio"] = 0.61
        augmented["BTCUSDT_global_long_short_ratio"] = 1.12
        augmented["BTCUSDT_taker_buy_sell_volume"] = 0.47
        augmented["BTCUSDT_basis"] = 0.0025
        augmented["BNBUSDT_open_interest"] = 98.0
        augmented["BNBUSDT_top_trader_position_ratio"] = 0.44
        augmented["BNBUSDT_top_trader_account_ratio"] = 0.42
        augmented["BNBUSDT_global_long_short_ratio"] = 0.93
        augmented["BNBUSDT_taker_buy_sell_volume"] = 0.53
        augmented["BNBUSDT_basis"] = -0.0015

        features = fractal.build_market_features(augmented, ("BTCUSDT", "BNBUSDT"))

        self.assertIn("btc_order_imbalance_1h", features)
        self.assertIn("bnb_order_imbalance_1h", features)
        self.assertAlmostEqual(float(features["btc_order_imbalance_1h"].iloc[-1]), 0.50, places=6)
        self.assertAlmostEqual(float(features["bnb_order_imbalance_1h"].iloc[-1]), -0.30, places=6)
        self.assertAlmostEqual(float(features["imbalance_spread_btc_minus_bnb_1h"].iloc[-1]), 0.80, places=6)

        overlay_inputs = regime.build_overlay_inputs(augmented, ("BTCUSDT", "BNBUSDT"), "BTCUSDT")
        overlay_augmented = dict(overlay_inputs)
        overlay_augmented.update(
            {
                "funding_rate_daily": pd.Series(0.0001, index=overlay_inputs["btc_regime_daily"].index, dtype="float64"),
                "open_interest_daily": pd.Series(1.0, index=overlay_inputs["btc_regime_daily"].index, dtype="float64"),
                "top_trader_position_ratio_daily": pd.Series(0.5, index=overlay_inputs["btc_regime_daily"].index, dtype="float64"),
                "top_trader_account_ratio_daily": pd.Series(0.5, index=overlay_inputs["btc_regime_daily"].index, dtype="float64"),
                "global_long_short_ratio_daily": pd.Series(1.0, index=overlay_inputs["btc_regime_daily"].index, dtype="float64"),
                "taker_buy_sell_ratio_daily": pd.Series(0.5, index=overlay_inputs["btc_regime_daily"].index, dtype="float64"),
                "basis_daily": pd.Series(0.0, index=overlay_inputs["btc_regime_daily"].index, dtype="float64"),
            }
        )
        raw_signal = pd.Series(np.linspace(0.0, 1.0, len(augmented)), index=augmented.index)
        library_lookup = {"spans": (2,)}

        context_base = regime.build_fast_context(
            augmented,
            "BTCUSDT",
            raw_signal=raw_signal,
            overlay_inputs=overlay_inputs,
            route_thresholds=(0.5,),
            library_lookup=library_lookup,
        )
        context_augmented = regime.build_fast_context(
            augmented,
            "BTCUSDT",
            raw_signal=raw_signal,
            overlay_inputs=overlay_augmented,
            route_thresholds=(0.5,),
            library_lookup=library_lookup,
        )

        np.testing.assert_allclose(context_base["equity_corr"], context_augmented["equity_corr"])
        np.testing.assert_allclose(context_base["breadth"], context_augmented["breadth"])
        np.testing.assert_allclose(context_base["vol_ann"], context_augmented["vol_ann"])
        self.assertEqual(context_base["route_state_mode"], context_augmented["route_state_mode"])

    def test_daily_market_features_are_not_pre_reflected_into_same_day_five_minute_bars(self) -> None:
        btc = gp.enrich_features(build_multi_day_price_frame(100.0, 110.0, taker_share=0.75)).add_prefix("BTCUSDT_")
        bnb = gp.enrich_features(build_multi_day_price_frame(200.0, 220.0, taker_share=0.35)).add_prefix("BNBUSDT_")
        merged = pd.concat([btc, bnb], axis=1)

        features = fractal.build_market_features(merged, ("BTCUSDT", "BNBUSDT"))
        arrays = fractal.materialize_feature_arrays(
            features,
            merged.index,
            strict_external_asof=True,
        )

        day15_close = pd.Timestamp("2026-01-15 23:55:00+00:00")
        day16_open = pd.Timestamp("2026-01-16 00:00:00+00:00")
        day16_midday = pd.Timestamp("2026-01-16 12:00:00+00:00")

        self.assertAlmostEqual(float(features["btc_return_1d"].loc[pd.Timestamp("2026-01-16", tz="UTC")]), 0.10, places=6)
        self.assertAlmostEqual(float(arrays["btc_return_1d"][merged.index.get_loc(day15_close)]), 0.0, places=6)
        self.assertAlmostEqual(float(arrays["btc_return_1d"][merged.index.get_loc(day16_open)]), 0.0, places=6)
        self.assertAlmostEqual(float(arrays["btc_return_1d"][merged.index.get_loc(day16_midday)]), 0.0, places=6)

    def test_daily_market_features_use_main_legacy_alignment_by_default(self) -> None:
        btc = gp.enrich_features(build_multi_day_price_frame(100.0, 110.0, taker_share=0.75)).add_prefix("BTCUSDT_")
        bnb = gp.enrich_features(build_multi_day_price_frame(200.0, 220.0, taker_share=0.35)).add_prefix("BNBUSDT_")
        merged = pd.concat([btc, bnb], axis=1)

        features = fractal.build_market_features(merged, ("BTCUSDT", "BNBUSDT"))
        arrays = fractal.materialize_feature_arrays(features, merged.index)

        day16_midday = pd.Timestamp("2026-01-16 12:00:00+00:00")
        self.assertAlmostEqual(float(arrays["btc_return_1d"][merged.index.get_loc(day16_midday)]), 0.10, places=6)


if __name__ == "__main__":
    unittest.main()
