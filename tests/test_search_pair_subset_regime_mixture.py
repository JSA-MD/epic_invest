import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from search_pair_subset_regime_mixture import realistic_overlay_replay_from_context, score_realistic_candidate


class SearchPairSubsetRegimeMixtureTests(unittest.TestCase):
    def test_score_realistic_candidate_prefers_recently_active_profiles(self) -> None:
        aggregate = {
            "worst_pair_avg_daily_return": 0.01,
            "mean_avg_daily_return": 0.015,
            "worst_max_drawdown": -0.05,
            "pair_return_dispersion": 0.01,
            "mean_n_trades": 80.0,
        }
        quiet = {
            "windows": {
                "recent_2m": {"aggregate": {**aggregate, "mean_n_trades": 2.0}},
                "recent_4m": {"aggregate": {**aggregate, "mean_n_trades": 4.0}},
                "recent_6m": {"aggregate": {**aggregate, "mean_n_trades": 8.0}},
                "full_4y": {"aggregate": {**aggregate, "mean_n_trades": 24.0}},
            }
        }
        active = {
            "windows": {
                "recent_2m": {"aggregate": {**aggregate, "mean_n_trades": 48.0}},
                "recent_4m": {"aggregate": {**aggregate, "mean_n_trades": 96.0}},
                "recent_6m": {"aggregate": {**aggregate, "mean_n_trades": 192.0}},
                "full_4y": {"aggregate": {**aggregate, "mean_n_trades": 960.0}},
            }
        }

        self.assertGreater(score_realistic_candidate(active), score_realistic_candidate(quiet))

    def test_python_trace_replay_preserves_fee_and_funding_breakdown(self) -> None:
        open_p = np.array([100.0, 101.0, 102.0, 103.0, 102.5, 103.5, 104.0], dtype="float64")
        close_p = np.array([100.4, 101.5, 102.4, 103.1, 103.0, 103.8, 104.4], dtype="float64")
        context = {
            "route_state_mode": "base",
            "open": open_p,
            "close": close_p,
            "funding_rates": np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0], dtype="float64"),
            "bucket_codes": {0.5: np.zeros_like(open_p, dtype="int64")},
            "regime": np.ones_like(open_p, dtype="float64"),
            "breadth": np.ones_like(open_p, dtype="float64"),
            "vol_ann": np.full_like(open_p, 0.35, dtype="float64"),
            "equity_corr_gross_scale": np.ones_like(open_p, dtype="float64"),
            "equity_corr_regime_mult": np.ones_like(open_p, dtype="float64"),
            "smooth_signal_matrix": np.array([[40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]], dtype="float64"),
            "order_imbalance": np.full_like(open_p, 0.2, dtype="float64"),
            "buy_volume_share": np.full_like(open_p, 0.6, dtype="float64"),
            "dc_trend_05": np.full_like(open_p, 0.2, dtype="float64"),
            "dc_run_05": np.full_like(open_p, 0.15, dtype="float64"),
        }
        library_lookup = {
            "signal_pos": np.array([0], dtype="int64"),
            "rebalance_bars": np.array([12], dtype="int64"),
            "regime_threshold": np.array([0.0], dtype="float64"),
            "breadth_threshold": np.array([0.0], dtype="float64"),
            "target_vol_ann": np.array([1.0], dtype="float64"),
            "gross_cap": np.array([1.5], dtype="float64"),
            "kill_switch_pct": np.array([1.0], dtype="float64"),
            "cooldown_days": np.array([0], dtype="int64"),
        }
        mapping = (0, 0, 0, 0)

        baseline = realistic_overlay_replay_from_context(
            context,
            library_lookup,
            mapping,
            0.5,
            engine="python",
            return_trace=False,
        )
        traced = realistic_overlay_replay_from_context(
            context,
            library_lookup,
            mapping,
            0.5,
            engine="python",
            return_trace=True,
        )

        self.assertGreater(float(baseline["fee_paid"]), 0.0)
        self.assertGreater(float(baseline["slippage_paid"]), 0.0)
        self.assertNotEqual(int(baseline["funding_events"]), 0)
        self.assertAlmostEqual(float(traced["fee_paid"]), float(baseline["fee_paid"]), places=12)
        self.assertAlmostEqual(float(traced["slippage_paid"]), float(baseline["slippage_paid"]), places=12)
        self.assertAlmostEqual(float(traced["funding_paid"]), float(baseline["funding_paid"]), places=12)
        self.assertEqual(int(traced["funding_events"]), int(baseline["funding_events"]))

    def test_execution_gene_countertrend_override_can_unlock_mild_bull_short(self) -> None:
        open_p = np.array([100.0, 99.8, 99.2, 98.9, 98.4, 98.1, 97.9], dtype="float64")
        close_p = np.array([99.9, 99.4, 99.0, 98.6, 98.2, 97.95, 97.8], dtype="float64")
        context = {
            "route_state_mode": "base",
            "open": open_p,
            "close": close_p,
            "funding_rates": np.zeros_like(open_p, dtype="float64"),
            "bucket_codes": {0.5: np.full_like(open_p, 3, dtype="int64")},
            "regime": np.full_like(open_p, 0.04, dtype="float64"),
            "breadth": np.full_like(open_p, 0.5, dtype="float64"),
            "vol_ann": np.full_like(open_p, 0.40, dtype="float64"),
            "equity_corr_gross_scale": np.ones_like(open_p, dtype="float64"),
            "equity_corr_regime_mult": np.ones_like(open_p, dtype="float64"),
            "smooth_signal_matrix": np.array([[-200.0] * len(open_p)], dtype="float64"),
            "order_imbalance": np.full_like(open_p, 0.05, dtype="float64"),
            "buy_volume_share": np.full_like(open_p, 0.52, dtype="float64"),
            "close_location_value": np.full_like(open_p, 0.55, dtype="float64"),
            "body_to_range": np.full_like(open_p, 0.70, dtype="float64"),
            "wick_skew": np.full_like(open_p, -0.15, dtype="float64"),
            "candle_micro_score": np.full_like(open_p, 0.20, dtype="float64"),
            "oi_rel": np.full_like(open_p, 1.0, dtype="float64"),
            "basis_rate": np.full_like(open_p, -0.0007, dtype="float64"),
            "top_pos_log_ratio": np.full_like(open_p, -0.2, dtype="float64"),
            "taker_buy_sell_log_ratio": np.full_like(open_p, -1.0, dtype="float64"),
            "range_bps": np.full_like(open_p, 3.0, dtype="float64"),
            "volume_ratio": np.full_like(open_p, 0.33, dtype="float64"),
            "dc_trend_05": np.full_like(open_p, -1.0, dtype="float64"),
            "dc_run_05": np.zeros_like(open_p, dtype="float64"),
        }
        library_lookup = {
            "signal_pos": np.array([0], dtype="int64"),
            "rebalance_bars": np.array([1], dtype="int64"),
            "regime_threshold": np.array([0.0], dtype="float64"),
            "breadth_threshold": np.array([0.65], dtype="float64"),
            "target_vol_ann": np.array([0.8], dtype="float64"),
            "gross_cap": np.array([0.75], dtype="float64"),
            "kill_switch_pct": np.array([1.0], dtype="float64"),
            "cooldown_days": np.array([0], dtype="int64"),
        }
        mapping = (0, 0, 0, 0)

        baseline = realistic_overlay_replay_from_context(
            context,
            library_lookup,
            mapping,
            0.5,
            engine="python",
            return_trace=True,
        )
        specialist = realistic_overlay_replay_from_context(
            context,
            library_lookup,
            mapping,
            0.5,
            execution_gene={
                "confirm_bars": 1,
                "countertrend_weight_scale": 0.2,
                "countertrend_rebalance_bypass": True,
                "countertrend_signal_floor_pct": 150.0,
                "countertrend_regime_score_cap": 0.05,
                "countertrend_breadth_slack": 0.2,
                "countertrend_microstructure_floor": -0.25,
                "countertrend_positioning_floor": 0.15,
                "countertrend_dc_floor": 0.5,
                "countertrend_range_bps_floor": 2.0,
                "countertrend_volume_ratio_floor": 0.25,
            },
            engine="python",
            return_trace=True,
        )

        self.assertTrue(np.allclose(baseline["trace"]["requested_weight"], 0.0))
        self.assertLess(np.min(specialist["trace"]["requested_weight"]), -0.01)
        self.assertTrue(np.allclose(baseline["trace"]["target_weight"], 0.0))
        self.assertLess(np.min(specialist["trace"]["target_weight"]), -0.01)
        self.assertGreater(int(specialist["n_trades"]), int(baseline["n_trades"]))


if __name__ == "__main__":
    unittest.main()
