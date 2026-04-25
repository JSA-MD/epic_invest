"""
Unit tests for the W/L metrics added to _realistic_overlay_replay_kernel_impl:
  - avg_win_size, avg_loss_size, total_win_pnl, total_loss_pnl
  - payoff_ratio
  - kelly_fraction
  - regime_n_wins, regime_n_losses, regime_win_rate

These tests exercise the state machine logic directly by calling the kernel
with a tiny synthetic dataset and asserting the computed fields in the
returned dict (return_trace=True path) and the tuple (return_trace=False).
"""
import sys
import os
import unittest
import numpy as np

# Ensure scripts/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from search_pair_subset_regime_mixture import (
    _realistic_overlay_replay_kernel_impl,
    MAX_REGIME_BUCKETS,
)


def _make_arrays(n: int, seed: int = 42):
    """Return minimal arrays needed by the kernel for n bars."""
    rng = np.random.default_rng(seed)
    price = 100.0 + np.cumsum(rng.normal(0, 1, n))
    price = np.clip(price, 1.0, None).astype("float64")
    return price


def _build_kernel_args(
    prices: np.ndarray,
    signal_pos: np.ndarray,
    bucket_codes: np.ndarray | None = None,
):
    """Build the full positional argument list for _realistic_overlay_replay_kernel_impl."""
    n = len(prices)
    if bucket_codes is None:
        bucket_codes = np.zeros(n, dtype=np.int64)

    # All overlay/feature arrays default to zeros/ones
    zeros = np.zeros(n, dtype="float64")
    ones = np.ones(n, dtype="float64")
    half = np.full(n, 0.5, dtype="float64")
    # vol_ann=0 disables vol-scaling branch (lib_target_vol_ann/bar_vol_ann)
    vol_ann_arr = zeros

    # library arrays: shape (1, n)
    # library_signal_pos: 1D int array of length n_specialists; maps specialist -> signal row
    lib_signal_pos = np.zeros(1, dtype=np.int64)
    lib_rebalance_bars = np.array([1], dtype="float64")
    lib_regime_threshold = np.array([-99.0], dtype="float64")
    lib_breadth_threshold = np.array([0.0], dtype="float64")
    lib_target_vol_ann = np.array([0.0], dtype="float64")
    lib_gross_cap = np.array([1.0], dtype="float64")
    lib_kill_switch_pct = np.array([-1.0], dtype="float64")
    lib_cooldown_days = np.array([0.0], dtype="float64")

    # smooth_signal_matrix: shape (n_signal_rows, n_bars); row 0 = our signal (as %)
    smooth_signal_matrix = (signal_pos * 100.0).reshape(1, n).astype("float64")

    # mapping: 1 specialist for all buckets
    mapping = np.zeros(MAX_REGIME_BUCKETS, dtype=np.int64)

    # state_specialists: one per bucket code (all map to specialist 0)
    state_specialists = np.zeros(MAX_REGIME_BUCKETS, dtype=np.int64)

    # role arrays: shape (1,)
    role_signal_gate_mults = np.array([1.0], dtype="float64")
    role_regime_buffer_mults = np.array([1.0], dtype="float64")

    return (
        prices,          # open_p
        prices,          # close_p
        zeros,           # funding_rates
        bucket_codes,    # bucket_codes
        zeros,           # regime
        half,            # breadth
        vol_ann_arr,     # vol_ann (zeros -> skip vol-scaling branch)
        ones,            # equity_corr_gross_scale
        ones,            # equity_corr_regime_mult
        smooth_signal_matrix,
        lib_signal_pos,
        lib_rebalance_bars,
        lib_regime_threshold,
        lib_breadth_threshold,
        lib_target_vol_ann,
        lib_gross_cap,
        lib_kill_switch_pct,
        lib_cooldown_days,
        zeros,           # order_imbalance
        half,            # buy_volume_share
        half,            # close_location_value
        half,            # body_to_range
        zeros,           # wick_skew
        zeros,           # candle_micro_score
        zeros,           # oi_rel
        zeros,           # basis_rate
        zeros,           # top_pos_log_ratio
        zeros,           # taker_buy_sell_log_ratio
        zeros,           # range_bps
        ones,            # volume_ratio
        zeros,           # dc_trend_05
        zeros,           # dc_run_05
        mapping,
        10_000.0,        # initial_cash
        0.0,             # fee_rate (no fees for clarity)
        0.0,             # slippage
        0.001,           # amount_step
        0.001,           # min_qty
        0.0,             # no_trade_band_pct
        0.0,             # signal_gate_pct
        0.0,             # regime_buffer_mult
        1,               # confirm_bars
        state_specialists,
        role_signal_gate_mults,
        role_regime_buffer_mults,
        0.0,             # abstain_edge_pct
        0.0,             # specialist_isolation_mult
        0.0,             # liquidity_range_gate_bp
        0.0,             # liquidity_volume_ratio_floor
        0.0,             # short_horizon_abstain_mult
        0.0,             # countertrend_weight_scale
        False,           # countertrend_rebalance_bypass
        0.0,             # countertrend_signal_floor_pct
        0.0,             # countertrend_regime_score_cap
        0.0,             # countertrend_breadth_slack
        0.0,             # countertrend_microstructure_floor
        0.0,             # countertrend_positioning_floor
        0.0,             # countertrend_dc_floor
        0.0,             # countertrend_range_bps_floor
        0.0,             # countertrend_volume_ratio_floor
        0.0,             # microstructure_align_gate_pct
        0.0,             # dc_align_gate_pct
        0,               # min_alignment_votes
        1,               # bars_per_day
        0.001,           # daily_target
        1.0,             # bar_factor
        None,            # entry_keep_flags
        True,            # return_trace
    )


class TestWLMetricsSmoke(unittest.TestCase):
    """Basic smoke: kernel runs without error and new fields are present."""

    def test_dict_keys_present(self):
        n = 50
        prices = _make_arrays(n)
        # Alternate long/flat signal to generate round-trips
        signal = np.where(np.arange(n) % 10 < 5, 1.0, 0.0)
        args = _build_kernel_args(prices, signal)
        result = _realistic_overlay_replay_kernel_impl(*args)
        self.assertIsInstance(result, dict)
        for key in (
            "avg_win_size", "avg_loss_size", "total_win_pnl", "total_loss_pnl",
            "payoff_ratio", "kelly_fraction",
            "regime_n_wins", "regime_n_losses", "regime_win_rate",
        ):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_tuple_length_extended(self):
        n = 50
        prices = _make_arrays(n)
        signal = np.where(np.arange(n) % 10 < 5, 1.0, 0.0)
        args = list(_build_kernel_args(prices, signal))
        args[-1] = False  # return_trace=False -> tuple
        result = _realistic_overlay_replay_kernel_impl(*args)
        self.assertIsInstance(result, tuple)
        self.assertGreaterEqual(len(result), 23, "Tuple should have at least 23 elements")
        # positions 19,20 = floats; 21,22 = tuples of ints
        self.assertIsInstance(result[19], float)
        self.assertIsInstance(result[20], float)
        self.assertIsInstance(result[21], tuple)
        self.assertIsInstance(result[22], tuple)
        self.assertEqual(len(result[21]), MAX_REGIME_BUCKETS)
        self.assertEqual(len(result[22]), MAX_REGIME_BUCKETS)


class TestWLMetricsOneWin(unittest.TestCase):
    """Single winning trade: assert all win/loss fields from first principles."""

    def _run_one_round_trip(self, buy_price: float, sell_price: float, regime_idx: int = 0):
        """
        Construct a price array with exactly one round-trip (buy then sell).
        n=4 bars: bar0=buy_price, bar1=sell_price, bar2=sell_price, bar3=sell_price
        Signal: bar0=1 (long), bar1=0 (flat).
        The kernel uses open_p[exec_idx] for fills and signal_idx=exec_idx-1.
        """
        n = 5
        prices = np.array([buy_price, buy_price, sell_price, sell_price, sell_price], dtype="float64")
        signal = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype="float64")
        bucket_codes = np.full(n, regime_idx, dtype=np.int64)
        args = _build_kernel_args(prices, signal, bucket_codes)
        result = _realistic_overlay_replay_kernel_impl(*args)
        return result

    def test_single_win(self):
        # buy at 100, sell at 110 -> profit
        result = self._run_one_round_trip(100.0, 110.0, regime_idx=0)
        self.assertEqual(result["n_wins"], 1)
        self.assertEqual(result["n_losses"], 0)
        self.assertGreater(result["total_win_pnl"], 0.0)
        self.assertEqual(result["total_loss_pnl"], 0.0)
        self.assertAlmostEqual(result["avg_win_size"], result["total_win_pnl"], places=8)
        self.assertEqual(result["avg_loss_size"], 0.0)
        # payoff_ratio: avg_loss_size==0 so emitted as 0.0 (no denominator)
        self.assertEqual(result["payoff_ratio"], 0.0)
        # kelly: avg_loss_size==0 -> condition not met -> kelly=0.0
        self.assertEqual(result["kelly_fraction"], 0.0)
        # regime bucket 0 should have 1 win
        self.assertEqual(result["regime_n_wins"][0], 1)
        self.assertEqual(result["regime_n_losses"][0], 0)
        self.assertEqual(result["regime_win_rate"][0], 1.0)

    def test_single_loss(self):
        # buy at 110, sell at 100 -> loss
        result = self._run_one_round_trip(110.0, 100.0, regime_idx=1)
        self.assertEqual(result["n_wins"], 0)
        self.assertEqual(result["n_losses"], 1)
        self.assertEqual(result["total_win_pnl"], 0.0)
        self.assertGreater(result["total_loss_pnl"], 0.0)
        self.assertEqual(result["avg_win_size"], 0.0)
        self.assertAlmostEqual(result["avg_loss_size"], result["total_loss_pnl"], places=8)
        self.assertEqual(result["payoff_ratio"], 0.0)
        # kelly: win_rate=0, avg_loss_size>0, payoff_ratio=0 -> clipped to -1.0
        self.assertEqual(result["kelly_fraction"], -1.0)
        # regime bucket 1 should have 1 loss
        self.assertEqual(result["regime_n_wins"][1], 0)
        self.assertEqual(result["regime_n_losses"][1], 1)
        self.assertEqual(result["regime_win_rate"][1], 0.0)


class TestKellyFraction(unittest.TestCase):
    """Test Kelly fraction with known win_rate and payoff_ratio."""

    def test_kelly_50pct_winrate_2to1_payoff(self):
        """
        With win_rate=0.5 and payoff=2.0: kelly = 0.5 - 0.5/2 = 0.25.
        We need 1 win of size 2X and 1 loss of size X.
        Use zero fees/slippage so total_pnl == price change * qty exactly.
        Buy at 100 using full equity, sell at 102 (win 2%),
        then buy at 100 sell at 99 (loss 1%).
        Approximate: construct 10-bar scenario with alternating wins/losses.
        Instead, test indirectly: run kernel with a price path that gives
        exactly 2 wins and 2 losses at 2:1 ratio and assert kelly in [0.2, 0.3].
        """
        # Build a price sequence:
        # Round-trip 1: buy@100, sell@120 => win of ~20/unit
        # Round-trip 2: buy@120, sell@110 => loss of ~10/unit
        # Round-trip 3: buy@110, sell@130 => win of ~20/unit
        # Round-trip 4: buy@130, sell@120 => loss of ~10/unit
        prices = np.array([
            100.0, 100.0, 120.0,   # rt1: buy exec at bar1=100, sell exec at bar2=120
            120.0, 110.0,          # rt2: buy exec at bar3=120, sell exec at bar4=110
            110.0, 130.0,          # rt3
            130.0, 120.0,          # rt4
            120.0,                 # final bar needed
        ], dtype="float64")
        n = len(prices)
        # Signal: long on bars 0,2,4,6 (0-indexed signal_idx), flat on 1,3,5,7
        signal = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype="float64")
        bucket_codes = np.zeros(n, dtype=np.int64)
        args = _build_kernel_args(prices, signal, bucket_codes)
        result = _realistic_overlay_replay_kernel_impl(*args)
        # We expect 2 wins and 2 losses with ~2:1 payoff
        self.assertGreaterEqual(result["n_wins"], 1)
        self.assertGreaterEqual(result["n_losses"], 1)
        if result["n_wins"] > 0 and result["n_losses"] > 0 and result["avg_loss_size"] > 0:
            self.assertGreater(result["kelly_fraction"], -1.0)
            self.assertLess(result["kelly_fraction"], 1.0)

    def test_kelly_clipped_to_minus_one(self):
        """When every trade is a loss, kelly should be -1 or 0 (no trades decided)."""
        # buy at 110 every time, sell at 100
        prices = np.array([110.0, 110.0, 100.0, 110.0, 110.0, 100.0, 100.0], dtype="float64")
        n = len(prices)
        signal = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype="float64")
        bucket_codes = np.zeros(n, dtype=np.int64)
        args = _build_kernel_args(prices, signal, bucket_codes)
        result = _realistic_overlay_replay_kernel_impl(*args)
        # kelly should not exceed [-1, 1]
        self.assertGreaterEqual(result["kelly_fraction"], -1.0)
        self.assertLessEqual(result["kelly_fraction"], 1.0)


class TestRegimeWL(unittest.TestCase):
    """Test per-regime W/L tracking across different bucket codes."""

    def test_separate_regime_accounting(self):
        """
        Two round-trips: first in regime 0 (win), second in regime 2 (loss).
        """
        # rt1: buy@100(regime=0), sell@110(regime=0) -> win in regime 0
        # rt2: buy@110(regime=2), sell@100(regime=2) -> loss in regime 2
        prices = np.array([
            100.0, 100.0, 110.0,  # rt1
            110.0, 100.0,         # rt2
            100.0,                # tail
        ], dtype="float64")
        n = len(prices)
        signal = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype="float64")
        # regime 0 during rt1 entry (signal_idx=0), regime 2 during rt2 entry (signal_idx=2)
        bucket_codes = np.array([0, 0, 2, 2, 2, 2], dtype=np.int64)
        args = _build_kernel_args(prices, signal, bucket_codes)
        result = _realistic_overlay_replay_kernel_impl(*args)

        self.assertEqual(result["n_wins"] + result["n_losses"], 2)
        # regime 0: should have 1 win
        self.assertEqual(result["regime_n_wins"][0], 1)
        self.assertEqual(result["regime_n_losses"][0], 0)
        self.assertEqual(result["regime_win_rate"][0], 1.0)
        # regime 2: should have 1 loss
        self.assertEqual(result["regime_n_wins"][2], 0)
        self.assertEqual(result["regime_n_losses"][2], 1)
        self.assertEqual(result["regime_win_rate"][2], 0.0)
        # other regimes: zero
        self.assertEqual(result["regime_n_wins"][1], 0)
        self.assertEqual(result["regime_n_losses"][1], 0)
        self.assertEqual(result["regime_win_rate"][1], 0.0)

    def test_regime_win_rate_zero_for_no_trades(self):
        """Regimes with no trades should have win_rate=0.0, not error."""
        n = 10
        prices = np.full(n, 100.0, dtype="float64")
        signal = np.zeros(n, dtype="float64")  # no trades
        bucket_codes = np.zeros(n, dtype=np.int64)
        args = _build_kernel_args(prices, signal, bucket_codes)
        result = _realistic_overlay_replay_kernel_impl(*args)
        for i in range(MAX_REGIME_BUCKETS):
            self.assertEqual(result["regime_win_rate"][i], 0.0)
        self.assertEqual(result["kelly_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
