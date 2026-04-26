import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import search_pair_subset_regime_mixture as pairwise_search


class PairwiseCorrRouteStateTests(unittest.TestCase):
    def test_build_route_bucket_codes_base_and_equity_corr(self) -> None:
        index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
        overlay_inputs = {
            "btc_regime_daily": pd.Series([-0.2, -0.1, 0.1, 0.2], index=index, dtype="float64"),
            "breadth_daily": pd.Series([0.2, 0.8, 0.2, 0.8], index=index, dtype="float64"),
            "equity_corr_bucket_daily": pd.Series(
                ["equity_inverse", "equity_mixed", "equity_aligned", "equity_unknown"],
                index=index,
                dtype="object",
            ),
        }

        base_codes = pairwise_search.build_route_bucket_codes(index, overlay_inputs, 0.5)
        strict_base_codes = pairwise_search.build_route_bucket_codes(
            index,
            overlay_inputs,
            0.5,
            strict_external_asof=True,
        )
        corr_codes = pairwise_search.build_route_bucket_codes(
            index,
            overlay_inputs,
            0.5,
            route_state_mode=pairwise_search.ROUTE_STATE_MODE_EQUITY_CORR,
        )
        strict_corr_codes = pairwise_search.build_route_bucket_codes(
            index,
            overlay_inputs,
            0.5,
            route_state_mode=pairwise_search.ROUTE_STATE_MODE_EQUITY_CORR,
            strict_external_asof=True,
        )

        self.assertEqual(base_codes.tolist(), [0, 1, 2, 3])
        self.assertEqual(corr_codes.tolist(), [0, 5, 10, 7])
        self.assertEqual(strict_base_codes.tolist(), [2, 0, 1, 2])
        self.assertEqual(strict_corr_codes.tolist(), [6, 0, 5, 10])

    def test_normalize_mapping_indices_expands_and_compresses(self) -> None:
        base_mapping = (0, 1, 2, 3)
        expanded = pairwise_search.normalize_mapping_indices(
            base_mapping,
            pairwise_search.ROUTE_STATE_MODE_EQUITY_CORR,
        )
        compressed = pairwise_search.normalize_mapping_indices(
            expanded,
            pairwise_search.ROUTE_STATE_MODE_BASE,
        )

        self.assertEqual(expanded, (0, 1, 2, 3) * 3)
        self.assertEqual(compressed, base_mapping)
        self.assertEqual(len(pairwise_search.route_state_names(pairwise_search.ROUTE_STATE_MODE_EQUITY_CORR)), 12)

    def test_realistic_score_ignores_recent_2m_noise(self) -> None:
        base_report = {
            "windows": {
                "recent_2m": {
                    "aggregate": {
                        "worst_pair_avg_daily_return": 0.0200,
                        "mean_avg_daily_return": 0.0210,
                        "worst_max_drawdown": -0.01,
                        "pair_return_dispersion": 0.0010,
                    }
                },
                "recent_6m": {
                    "aggregate": {
                        "worst_pair_avg_daily_return": 0.0064,
                        "mean_avg_daily_return": 0.0068,
                        "worst_max_drawdown": -0.11,
                        "pair_return_dispersion": 0.0020,
                    }
                },
                "full_4y": {
                    "aggregate": {
                        "worst_pair_avg_daily_return": 0.0048,
                        "mean_avg_daily_return": 0.0052,
                        "worst_max_drawdown": -0.14,
                        "pair_return_dispersion": 0.0030,
                    }
                },
            }
        }
        noisy_recent_report = {
            "windows": {
                "recent_2m": {
                    "aggregate": {
                        "worst_pair_avg_daily_return": -0.0300,
                        "mean_avg_daily_return": -0.0280,
                        "worst_max_drawdown": -0.30,
                        "pair_return_dispersion": 0.0200,
                    }
                },
                "recent_6m": base_report["windows"]["recent_6m"],
                "full_4y": base_report["windows"]["full_4y"],
            }
        }

        self.assertEqual(
            pairwise_search.score_realistic_candidate(base_report),
            pairwise_search.score_realistic_candidate(noisy_recent_report),
        )


class InitialCooldownBarsTests(unittest.TestCase):
    """Verify initial_cooldown_bars carry-in suppresses trades for exactly N bars."""

    def _make_kernel_args(self, n_bars: int = 60):
        """Build minimal arrays for _realistic_overlay_replay_kernel_impl."""
        import numpy as np
        rng = np.random.default_rng(42)
        price = 30000.0 + np.cumsum(rng.normal(0, 10, n_bars + 1))
        funding = np.zeros(n_bars + 1)
        # All bars in regime bucket 0 (bull), breadth above threshold
        bucket_codes = np.zeros(n_bars + 1, dtype=np.int64)
        regime = np.full(n_bars + 1, 0.15)
        breadth = np.full(n_bars + 1, 0.8)
        vol_ann = np.full(n_bars + 1, 0.6)
        ones = np.ones(n_bars + 1)
        # Signal: always 50 (maps to weight=0.5 long); shape=(n_signal_rows, n_bars+1)
        smooth_signal_matrix = np.full((1, n_bars + 1), 50.0)
        # Library: single param set, never kill-switch (kill_switch_pct=0)
        lib_signal_pos = np.array([0], dtype=np.int64)
        lib_rebalance_bars = np.array([1], dtype=np.int64)
        lib_regime_threshold = np.array([0.0])
        lib_breadth_threshold = np.array([0.0])
        lib_target_vol_ann = np.array([0.3])
        lib_gross_cap = np.array([1.0])
        lib_kill_switch_pct = np.array([0.99])  # never triggers (99% drawdown threshold)
        lib_cooldown_days = np.array([1], dtype=np.int64)
        mapping = np.array([0], dtype=np.int64)
        state_specialists = np.array([0], dtype=np.int64)
        role_gate = np.array([1.0])
        role_regime = np.array([0.0])
        zero_arr = np.zeros(n_bars + 1)
        return dict(
            open_p=price,
            close_p=price,
            funding_rates=funding,
            bucket_codes=bucket_codes,
            regime=regime,
            breadth=breadth,
            vol_ann=vol_ann,
            equity_corr_gross_scale=ones,
            equity_corr_regime_mult=ones,
            smooth_signal_matrix=smooth_signal_matrix,
            library_signal_pos=lib_signal_pos,
            library_rebalance_bars=lib_rebalance_bars,
            library_regime_threshold=lib_regime_threshold,
            library_breadth_threshold=lib_breadth_threshold,
            library_target_vol_ann=lib_target_vol_ann,
            library_gross_cap=lib_gross_cap,
            library_kill_switch_pct=lib_kill_switch_pct,
            library_cooldown_days=lib_cooldown_days,
            order_imbalance=zero_arr,
            buy_volume_share=ones * 0.5,
            close_location_value=zero_arr,
            body_to_range=zero_arr,
            wick_skew=zero_arr,
            candle_micro_score=zero_arr,
            oi_rel=ones,
            basis_rate=zero_arr,
            top_pos_log_ratio=zero_arr,
            taker_buy_sell_log_ratio=zero_arr,
            range_bps=ones * 100.0,
            volume_ratio=ones,
            dc_trend_05=zero_arr,
            dc_run_05=zero_arr,
            mapping=mapping,
            initial_cash=10000.0,
            fee_rate=0.0004,
            slippage=0.0002,
            amount_step=0.001,
            min_qty=0.001,
            no_trade_band_pct=0.001,
            signal_gate_pct=0.0,
            regime_buffer_mult=0.0,
            confirm_bars=1,
            state_specialists=state_specialists,
            role_signal_gate_mults=role_gate,
            role_regime_buffer_mults=role_regime,
            abstain_edge_pct=0.0,
            specialist_isolation_mult=0.0,
            liquidity_range_gate_bp=0.0,
            liquidity_volume_ratio_floor=0.0,
            short_horizon_abstain_mult=0.0,
            countertrend_weight_scale=0.0,
            countertrend_rebalance_bypass=False,
            countertrend_signal_floor_pct=0.0,
            countertrend_regime_score_cap=0.0,
            countertrend_breadth_slack=0.0,
            countertrend_microstructure_floor=0.0,
            countertrend_positioning_floor=0.0,
            countertrend_dc_floor=0.0,
            countertrend_range_bps_floor=0.0,
            countertrend_volume_ratio_floor=0.0,
            microstructure_align_gate_pct=0.0,
            dc_align_gate_pct=0.0,
            min_alignment_votes=0,
            bars_per_day=288,
            daily_target=0.003,
            bar_factor=1.0,
        )

    def test_initial_cooldown_zero_allows_trades(self) -> None:
        """default initial_cooldown_bars=0 must allow the kernel to take positions (regression guard)."""
        args = self._make_kernel_args(n_bars=40)
        result = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=0,
        )
        self.assertIsInstance(result, dict)
        trace = result.get("trace", {})
        target_weight_arr = trace.get("target_weight")
        self.assertIsNotNone(target_weight_arr, "trace must contain target_weight")
        # With cooldown=0 at start, the kernel must reach a non-zero target weight at some bar
        import numpy as np
        self.assertTrue(np.any(target_weight_arr != 0.0),
                        "expected at least one non-zero target weight with initial_cooldown_bars=0")

    def test_initial_cooldown_blocks_first_n_bars(self) -> None:
        """initial_cooldown_bars=N must keep cooldown_bars_left > 0 for first N bars."""
        n_bars = 40
        cooldown_in = 10
        args = self._make_kernel_args(n_bars=n_bars)
        result = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=cooldown_in,
        )
        self.assertIsInstance(result, dict)
        trace = result.get("trace", {})
        cooldown_arr = trace.get("cooldown_bars_left")
        self.assertIsNotNone(cooldown_arr, "trace must contain cooldown_bars_left")
        # First bar should have cooldown_in - 1 (decremented once at bar 0)
        self.assertEqual(int(cooldown_arr[0]), cooldown_in - 1,
                         "first trace bar should show initial_cooldown_bars decremented by 1")
        # All bars 0..(cooldown_in-2) should be non-zero (still cooling down)
        for i in range(cooldown_in - 1):
            self.assertGreater(int(cooldown_arr[i]), 0,
                               f"bar {i} should still be in cooldown")

    def test_initial_cooldown_result_equals_zero_default(self) -> None:
        """Calling with initial_cooldown_bars=0 must produce identical result to omitting it."""
        args = self._make_kernel_args(n_bars=30)
        result_default = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=False,
            min_notional_usd=0.0,
            max_hold_bars=288,
        )
        result_zero = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=False,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=0,
        )
        self.assertEqual(result_default, result_zero,
                         "initial_cooldown_bars=0 must be identical to the default")


class FinalDecisionCooldownOverrideTests(unittest.TestCase):
    """Verify final_decision_cooldown_override forces cooldown on the last decision bar only."""

    def _make_kernel_args(self, n_bars: int = 40):
        """Reuse same minimal array setup as InitialCooldownBarsTests."""
        import numpy as np
        rng = np.random.default_rng(7)
        price = 30000.0 + np.cumsum(rng.normal(0, 10, n_bars + 1))
        funding = np.zeros(n_bars + 1)
        bucket_codes = np.zeros(n_bars + 1, dtype=np.int64)
        regime = np.full(n_bars + 1, 0.15)
        breadth = np.full(n_bars + 1, 0.8)
        vol_ann = np.full(n_bars + 1, 0.6)
        ones = np.ones(n_bars + 1)
        smooth_signal_matrix = np.full((1, n_bars + 1), 50.0)
        lib_signal_pos = np.array([0], dtype=np.int64)
        lib_rebalance_bars = np.array([1], dtype=np.int64)
        lib_regime_threshold = np.array([0.0])
        lib_breadth_threshold = np.array([0.0])
        lib_target_vol_ann = np.array([0.3])
        lib_gross_cap = np.array([1.0])
        lib_kill_switch_pct = np.array([0.99])
        lib_cooldown_days = np.array([1], dtype=np.int64)
        mapping = np.array([0], dtype=np.int64)
        state_specialists = np.array([0], dtype=np.int64)
        role_gate = np.array([1.0])
        role_regime = np.array([0.0])
        zero_arr = np.zeros(n_bars + 1)
        return dict(
            open_p=price,
            close_p=price,
            funding_rates=funding,
            bucket_codes=bucket_codes,
            regime=regime,
            breadth=breadth,
            vol_ann=vol_ann,
            equity_corr_gross_scale=ones,
            equity_corr_regime_mult=ones,
            smooth_signal_matrix=smooth_signal_matrix,
            library_signal_pos=lib_signal_pos,
            library_rebalance_bars=lib_rebalance_bars,
            library_regime_threshold=lib_regime_threshold,
            library_breadth_threshold=lib_breadth_threshold,
            library_target_vol_ann=lib_target_vol_ann,
            library_gross_cap=lib_gross_cap,
            library_kill_switch_pct=lib_kill_switch_pct,
            library_cooldown_days=lib_cooldown_days,
            order_imbalance=zero_arr,
            buy_volume_share=ones * 0.5,
            close_location_value=zero_arr,
            body_to_range=zero_arr,
            wick_skew=zero_arr,
            candle_micro_score=zero_arr,
            oi_rel=ones,
            basis_rate=zero_arr,
            top_pos_log_ratio=zero_arr,
            taker_buy_sell_log_ratio=zero_arr,
            range_bps=ones * 100.0,
            volume_ratio=ones,
            dc_trend_05=zero_arr,
            dc_run_05=zero_arr,
            mapping=mapping,
            initial_cash=10000.0,
            fee_rate=0.0004,
            slippage=0.0002,
            amount_step=0.001,
            min_qty=0.001,
            no_trade_band_pct=0.001,
            signal_gate_pct=0.0,
            regime_buffer_mult=0.0,
            confirm_bars=1,
            state_specialists=state_specialists,
            role_signal_gate_mults=role_gate,
            role_regime_buffer_mults=role_regime,
            abstain_edge_pct=0.0,
            specialist_isolation_mult=0.0,
            liquidity_range_gate_bp=0.0,
            liquidity_volume_ratio_floor=0.0,
            short_horizon_abstain_mult=0.0,
            countertrend_weight_scale=0.0,
            countertrend_rebalance_bypass=False,
            countertrend_signal_floor_pct=0.0,
            countertrend_regime_score_cap=0.0,
            countertrend_breadth_slack=0.0,
            countertrend_microstructure_floor=0.0,
            countertrend_positioning_floor=0.0,
            countertrend_dc_floor=0.0,
            countertrend_range_bps_floor=0.0,
            countertrend_volume_ratio_floor=0.0,
            microstructure_align_gate_pct=0.0,
            dc_align_gate_pct=0.0,
            min_alignment_votes=0,
            bars_per_day=288,
            daily_target=0.003,
            bar_factor=1.0,
        )

    def test_override_blocks_last_bar(self) -> None:
        """With a large final_decision_cooldown_override, the last decision bar's target_weight must be 0."""
        import numpy as np
        n_bars = 20
        args = self._make_kernel_args(n_bars=n_bars)
        result = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=0,
            final_decision_cooldown_override=n_bars * 10,  # large: will be 1 after decrement
        )
        self.assertIsInstance(result, dict)
        trace = result.get("trace", {})
        target_weight_arr = trace.get("target_weight")
        self.assertIsNotNone(target_weight_arr)
        # Last trace bar (index -1) corresponds to the last decision: must be 0.0
        self.assertEqual(float(target_weight_arr[-1]), 0.0,
                         "last decision bar must have target_weight=0 when override is large")

    def test_no_override_allows_last_bar(self) -> None:
        """With final_decision_cooldown_override=None the last bar behaves naturally (no forced zero)."""
        import numpy as np
        args = self._make_kernel_args(n_bars=20)
        result = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=0,
            final_decision_cooldown_override=None,
        )
        self.assertIsInstance(result, dict)
        trace = result.get("trace", {})
        target_weight_arr = trace.get("target_weight")
        self.assertIsNotNone(target_weight_arr)
        # Signal is always 50 (weight 0.5), regime/breadth permissive: at least one bar must be nonzero
        self.assertTrue(any(float(w) != 0.0 for w in target_weight_arr),
                        "without override, at least one bar must produce a nonzero target_weight")

    def test_override_takes_precedence_over_natural_decay(self) -> None:
        """Even if cooldown would have naturally reached 0, override re-blocks the last bar."""
        import numpy as np
        # Use n_bars=5 with initial_cooldown_bars=3: by bar 4 (last), cooldown would be 0 naturally.
        # But with override=999, last bar is forced blocked.
        n_bars = 5
        args = self._make_kernel_args(n_bars=n_bars)
        result_with = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=3,
            final_decision_cooldown_override=999,
        )
        result_without = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=3,
            final_decision_cooldown_override=None,
        )
        trace_with = result_with.get("trace", {})
        trace_without = result_without.get("trace", {})
        tw_with = trace_with.get("target_weight", [])
        tw_without = trace_without.get("target_weight", [])
        self.assertIsNotNone(tw_with)
        self.assertIsNotNone(tw_without)
        # Last bar with override must be 0
        self.assertEqual(float(tw_with[-1]), 0.0,
                         "override=999 must block last bar even if natural cooldown decayed to 0")
        # Without override the last bar is free (natural cooldown expired at bar ~3)
        self.assertTrue(any(float(w) != 0.0 for w in tw_without),
                        "without override some bars after cooldown expiry should be nonzero")


    def test_override_one_still_blocks_last_bar(self) -> None:
        """Regression: override=1 must block (off-by-one guard).

        Live convention: shadow.cooldown_bars_left[pair]==1 at the start of bar
        T means the gate must still block (one bar remaining). If the kernel
        applied the override before the per-bar decrement it would burn off
        to 0 and let the trade through — exactly the bug Codex flagged as
        "cooldown carry-over unblocks one bar early".
        """
        n_bars = 5
        args = self._make_kernel_args(n_bars=n_bars)
        result = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=0,
            final_decision_cooldown_override=1,
        )
        trace = result.get("trace", {})
        tw = trace.get("target_weight", [])
        self.assertIsNotNone(tw)
        self.assertEqual(
            float(tw[-1]), 0.0,
            "override=1 must still block the last decision bar; if not, "
            "the override is being applied before the per-bar decrement "
            "(off-by-one)",
        )

    def test_override_persistence_decrements_each_call(self) -> None:
        """Regression (Codex #16): trace cooldown at last bar must be override-1.

        If the raw override value is stored in the trace, live will read it back
        next cycle and cooldown never expires.  The trace value must be
        max(0, override - 1) so each successive live bar decrements normally.
        """
        n_bars = 5
        override = 3
        args = self._make_kernel_args(n_bars=n_bars)
        result = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=0,
            final_decision_cooldown_override=override,
        )
        trace = result.get("trace", {})
        cd = trace.get("cooldown_bars_left")
        self.assertIsNotNone(cd)
        self.assertEqual(
            int(cd[-1]), override - 1,
            f"trace cooldown at last bar should be override-1={override - 1}, "
            f"got {int(cd[-1])}; raw override in trace means live cooldown never expires",
        )

    def test_override_zero_persists_zero(self) -> None:
        """override=0 means cooldown is already expired; trace must record 0."""
        n_bars = 5
        args = self._make_kernel_args(n_bars=n_bars)
        result = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=0,
            final_decision_cooldown_override=0,
        )
        trace = result.get("trace", {})
        cd = trace.get("cooldown_bars_left")
        self.assertIsNotNone(cd)
        self.assertEqual(
            int(cd[-1]), 0,
            "override=0: trace cooldown must be 0 (already expired)",
        )

    def test_override_one_persists_zero(self) -> None:
        """override=1 blocks last bar AND persists 0 so next bar is unblocked."""
        n_bars = 5
        args = self._make_kernel_args(n_bars=n_bars)
        result = pairwise_search._realistic_overlay_replay_kernel_impl(
            **args,
            entry_keep_flags=None,
            return_trace=True,
            min_notional_usd=0.0,
            max_hold_bars=288,
            initial_cooldown_bars=0,
            final_decision_cooldown_override=1,
        )
        trace = result.get("trace", {})
        tw = trace.get("target_weight", [])
        cd = trace.get("cooldown_bars_left")
        self.assertIsNotNone(tw)
        self.assertIsNotNone(cd)
        # Gate still blocks
        self.assertEqual(float(tw[-1]), 0.0,
                         "override=1 must block last bar")
        # But persisted value is 0 so next bar starts unblocked
        self.assertEqual(int(cd[-1]), 0,
                         "override=1 must persist 0 so next cycle is unblocked")


if __name__ == "__main__":
    unittest.main()
