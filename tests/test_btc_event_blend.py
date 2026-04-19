from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from btc_event_blend import (
    apply_runtime_event_blend,
    build_runtime_event_context_from_frame,
    get_btc_event_blend,
)


class BtcEventBlendTests(unittest.TestCase):
    def test_get_btc_event_blend_normalizes_payload(self) -> None:
        payload = get_btc_event_blend(
            {
                "btc_event_blend": {
                    "pair": "BTCUSDT",
                    "alpha": "0.05",
                    "mode": "disagree_narrow_only",
                    "trigger_weight": "0.4",
                }
            },
            "BTCUSDT",
        )
        assert payload is not None
        self.assertEqual(payload["pair"], "BTCUSDT")
        self.assertAlmostEqual(payload["alpha"], 0.05)
        self.assertEqual(payload["mode"], "disagree_narrow_only")
        self.assertAlmostEqual(payload["trigger_weight"], 0.4)

    def test_runtime_blend_activates_on_breakout_when_baseline_flat(self) -> None:
        idx = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
        close = np.array([100.0, 99.5, 99.0, 99.3, 99.8, 100.1, 100.6, 101.2])
        high = close + 0.2
        low = close - 0.2
        frame = pd.DataFrame(
            {
                "BTCUSDT_close": close,
                "BTCUSDT_high": high,
                "BTCUSDT_low": low,
                "BTCUSDT_volume": np.array([10, 9, 8, 8, 9, 10, 15, 18], dtype=float),
                "BTCUSDT_vol_sma": np.full(8, 10.0),
                "BTCUSDT_order_imbalance": np.array([0.0, -0.1, -0.2, -0.1, 0.1, 0.2, 0.4, 0.6]),
                "BTCUSDT_buy_volume_share": np.array([0.45, 0.4, 0.38, 0.45, 0.55, 0.6, 0.65, 0.7]),
                "BTCUSDT_close_location_value": np.array([0.1, -0.2, -0.3, -0.1, 0.2, 0.4, 0.6, 0.8]),
                "BTCUSDT_body_to_range": np.array([0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.7, 0.8]),
                "BTCUSDT_wick_skew": np.array([0.0, -0.1, -0.2, -0.1, 0.1, 0.15, 0.2, 0.3]),
                "BTCUSDT_oi_rel": np.array([0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.05, 1.08]),
                "BTCUSDT_basis_rate": np.array([-0.0002, -0.0001, 0.0, 0.0001, 0.0002, 0.0004, 0.0007, 0.001]),
                "BTCUSDT_top_pos_log_ratio": np.array([-0.1, -0.08, -0.05, -0.02, 0.05, 0.09, 0.12, 0.18]),
                "BTCUSDT_taker_buy_sell_log_ratio": np.array([-0.08, -0.05, -0.03, -0.01, 0.03, 0.05, 0.08, 0.11]),
                "BTCUSDT_dc_trend_05": np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float),
                "BTCUSDT_dc_run_05": np.array([0.0, -0.2, -0.3, -0.1, 0.1, 0.2, 0.3, 0.5]),
            },
            index=idx,
        )
        context = build_runtime_event_context_from_frame(frame, "BTCUSDT")
        self.assertTrue(context["derivative_inputs_ready"])
        baseline_plan = {
            "requested_weight": 0.0,
            "target_weight": 0.0,
            "route_state_name": "equity_aligned:bull_narrow",
        }
        event = {
            "alpha": 0.05,
            "mode": "disagree_narrow_only",
            "trigger_weight": 0.4,
            "fast_span": 3,
            "slow_span": 5,
            "retest_lookback": 3,
            "support_floor": 0.0,
            "volume_ratio_floor": 1.0,
            "activation_mode": "flat_only",
            "hold_bars": 2,
        }
        final_plan, next_state = apply_runtime_event_blend(
            context=context,
            baseline_plan=baseline_plan,
            pair_state={"active_side": 1.0, "hold_left": 1},
            event=event,
        )
        self.assertGreater(final_plan["target_weight"], 0.0)
        self.assertEqual(next_state["active_side"], 1.0)

    def test_runtime_blend_allows_broad_state_disagreement_when_specialist_is_active(self) -> None:
        idx = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
        close = np.array([101.2, 100.9, 100.6, 100.3, 99.9, 99.4, 98.9, 98.3])
        high = close + 0.2
        low = close - 0.2
        frame = pd.DataFrame(
            {
                "BTCUSDT_close": close,
                "BTCUSDT_high": high,
                "BTCUSDT_low": low,
                "BTCUSDT_volume": np.array([11, 12, 12, 13, 14, 15, 16, 18], dtype=float),
                "BTCUSDT_vol_sma": np.full(8, 10.0),
                "BTCUSDT_order_imbalance": np.array([-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.45]),
                "BTCUSDT_buy_volume_share": np.array([0.46, 0.44, 0.42, 0.4, 0.38, 0.36, 0.34, 0.32]),
                "BTCUSDT_close_location_value": np.array([-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4, -0.45]),
                "BTCUSDT_body_to_range": np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]),
                "BTCUSDT_wick_skew": np.array([-0.05, -0.08, -0.1, -0.12, -0.15, -0.18, -0.2, -0.25]),
                "BTCUSDT_oi_rel": np.array([1.0, 1.01, 1.02, 1.03, 1.05, 1.06, 1.08, 1.1]),
                "BTCUSDT_basis_rate": np.array([-0.0001, -0.0002, -0.0003, -0.0004, -0.0005, -0.0006, -0.0008, -0.001]),
                "BTCUSDT_top_pos_log_ratio": np.array([-0.02, -0.03, -0.04, -0.05, -0.07, -0.09, -0.11, -0.14]),
                "BTCUSDT_taker_buy_sell_log_ratio": np.array([-0.01, -0.03, -0.04, -0.05, -0.07, -0.09, -0.11, -0.13]),
                "BTCUSDT_dc_trend_05": np.full(8, -1.0, dtype=float),
                "BTCUSDT_dc_run_05": np.array([-0.05, -0.08, -0.1, -0.12, -0.15, -0.18, -0.22, -0.28]),
            },
            index=idx,
        )
        context = build_runtime_event_context_from_frame(frame, "BTCUSDT")
        baseline_plan = {
            "requested_weight": 0.0,
            "target_weight": 0.0,
            "route_state_name": "equity_mixed:bull_broad",
        }
        event = {
            "alpha": 0.25,
            "mode": "disagree_only",
            "trigger_weight": 0.05,
            "fast_span": 3,
            "slow_span": 5,
            "retest_lookback": 3,
            "support_floor": 0.2,
            "volume_ratio_floor": 1.2,
            "activation_mode": "flat_only",
            "hold_bars": 1,
        }
        final_plan, next_state = apply_runtime_event_blend(
            context=context,
            baseline_plan=baseline_plan,
            pair_state={"active_side": -1.0, "hold_left": 1},
            event=event,
        )
        self.assertLess(final_plan["target_weight"], 0.0)
        self.assertEqual(next_state["active_side"], -1.0)

    def test_runtime_blend_stays_disabled_when_derivative_inputs_are_missing(self) -> None:
        idx = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
        close = np.array([100.0, 99.8, 99.6, 99.7, 99.9, 100.2, 100.4, 100.7])
        frame = pd.DataFrame(
            {
                "BTCUSDT_close": close,
                "BTCUSDT_high": close + 0.15,
                "BTCUSDT_low": close - 0.15,
                "BTCUSDT_volume": np.full(8, 10.0),
                "BTCUSDT_vol_sma": np.full(8, 10.0),
                "BTCUSDT_order_imbalance": np.linspace(-0.1, 0.4, 8),
                "BTCUSDT_buy_volume_share": np.linspace(0.45, 0.65, 8),
                "BTCUSDT_close_location_value": np.linspace(-0.2, 0.6, 8),
                "BTCUSDT_body_to_range": np.linspace(0.2, 0.7, 8),
                "BTCUSDT_wick_skew": np.linspace(-0.1, 0.2, 8),
                "BTCUSDT_dc_trend_05": np.array([-1, -1, -1, -1, 1, 1, 1, 1], dtype=float),
                "BTCUSDT_dc_run_05": np.linspace(-0.2, 0.4, 8),
            },
            index=idx,
        )
        context = build_runtime_event_context_from_frame(frame, "BTCUSDT")
        self.assertFalse(context["derivative_inputs_ready"])
        baseline_plan = {
            "requested_weight": 0.0,
            "target_weight": 0.0,
            "route_state_name": "equity_aligned:bull_narrow",
        }
        event = {
            "alpha": 0.05,
            "mode": "disagree_narrow_only",
            "trigger_weight": 0.4,
            "fast_span": 3,
            "slow_span": 5,
            "retest_lookback": 3,
            "support_floor": 0.0,
            "volume_ratio_floor": 1.0,
            "activation_mode": "flat_only",
            "hold_bars": 2,
        }
        final_plan, next_state = apply_runtime_event_blend(
            context=context,
            baseline_plan=baseline_plan,
            pair_state={"active_side": 1.0, "hold_left": 1},
            event=event,
        )
        self.assertEqual(final_plan["target_weight"], baseline_plan["target_weight"])
        self.assertFalse(final_plan["event_blend"]["enabled"])
        self.assertEqual(final_plan["event_blend"]["reason"], "missing_derivative_inputs")
        self.assertEqual(next_state["active_side"], 0.0)
        self.assertEqual(next_state["hold_left"], 0)

    def test_runtime_context_uses_derivative_bundle_when_live_frame_lacks_columns(self) -> None:
        idx = pd.date_range("2026-01-01", periods=300, freq="5min", tz="UTC")
        close = np.linspace(100.0, 102.0, len(idx))
        frame = pd.DataFrame(
            {
                "BTCUSDT_close": close,
                "BTCUSDT_high": close + 0.2,
                "BTCUSDT_low": close - 0.2,
                "BTCUSDT_volume": np.full(len(idx), 12.0),
                "BTCUSDT_vol_sma": np.full(len(idx), 10.0),
                "BTCUSDT_order_imbalance": np.linspace(-0.05, 0.35, len(idx)),
                "BTCUSDT_buy_volume_share": np.linspace(0.48, 0.62, len(idx)),
                "BTCUSDT_close_location_value": np.linspace(-0.1, 0.7, len(idx)),
                "BTCUSDT_body_to_range": np.linspace(0.2, 0.8, len(idx)),
                "BTCUSDT_wick_skew": np.linspace(-0.05, 0.25, len(idx)),
                "BTCUSDT_dc_trend_05": np.concatenate([np.full(150, -1.0), np.full(150, 1.0)]),
                "BTCUSDT_dc_run_05": np.linspace(-0.3, 0.5, len(idx)),
            },
            index=idx,
        )
        derivative_bundle = {
            "open_interest": pd.DataFrame(
                {
                    "timestamp": idx,
                    "open_interest": np.linspace(1_000_000.0, 1_500_000.0, len(idx)),
                }
            ),
            "basis_perpetual": pd.DataFrame(
                {
                    "timestamp": idx,
                    "basis_rate": np.linspace(-0.0005, 0.0012, len(idx)),
                }
            ),
            "top_trader_position_ratio": pd.DataFrame(
                {
                    "timestamp": idx,
                    "long_short_ratio": np.linspace(0.95, 1.25, len(idx)),
                }
            ),
            "taker_buy_sell_ratio": pd.DataFrame(
                {
                    "timestamp": idx,
                    "buy_sell_ratio": np.linspace(0.92, 1.18, len(idx)),
                }
            ),
        }

        context = build_runtime_event_context_from_frame(
            frame,
            "BTCUSDT",
            derivative_bundle=derivative_bundle,
        )

        self.assertTrue(context["derivative_inputs_ready"])
        self.assertGreater(float(context["oi_rel"][-1]), 1.0)
        self.assertGreater(float(context["basis_rate"][-1]), 0.0)
        self.assertGreater(float(context["top_pos_log_ratio"][-1]), 0.0)
        self.assertGreater(float(context["taker_buy_sell_log_ratio"][-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
