import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from execution_gene_utils import (
    blended_entry_quality_score,
    blended_microstructure_score,
    candle_microstructure_proxy_score,
    dc_alignment_score,
    derive_execution_profile,
    derivative_positioning_score,
    extract_pair_execution_gene,
    legacy_execution_profile,
    microstructure_alignment_score,
    should_abstain_for_alignment,
    should_abstain_for_liquidity,
    should_allow_countertrend_entry,
    should_abstain_for_weak_tape,
)


class ExecutionGeneUtilsTests(unittest.TestCase):
    def test_derive_execution_profile_penalizes_urgent_execution(self) -> None:
        patient = derive_execution_profile(
            {
                "maker_priority": 0.85,
                "max_wait_bars": 2,
                "chase_distance_bp": 1.0,
                "cancel_replace_interval_bars": 1,
                "partial_fill_tolerance": 0.90,
                "emergency_market_threshold_bp": 35.0,
                "flow_alignment_threshold": 0.20,
                "dc_alignment_threshold": 0.15,
                "min_alignment_votes": 3,
            }
        )
        urgent = derive_execution_profile(
            {
                "maker_priority": 0.20,
                "max_wait_bars": 0,
                "chase_distance_bp": 6.0,
                "cancel_replace_interval_bars": 3,
                "partial_fill_tolerance": 0.25,
                "emergency_market_threshold_bp": 8.0,
                "microstructure_align_gate_pct": 0.35,
                "dc_align_gate_pct": 0.30,
                "min_alignment_votes": 2,
            }
        )

        self.assertGreater(urgent["fee_rate"], patient["fee_rate"])
        self.assertGreater(urgent["slippage"], patient["slippage"])
        self.assertLess(urgent["fill_confidence"], patient["fill_confidence"])
        self.assertIn("abstain_edge_pct", patient)
        self.assertEqual(patient["flow_alignment_threshold"], 0.20)
        self.assertEqual(patient["microstructure_align_gate_pct"], 0.20)
        self.assertEqual(patient["dc_alignment_threshold"], 0.15)
        self.assertEqual(patient["dc_align_gate_pct"], 0.15)
        self.assertEqual(patient["min_alignment_votes"], 2)
        self.assertEqual(patient["liquidity_range_gate_bp"], 0.0)
        self.assertEqual(patient["liquidity_volume_ratio_floor"], 0.0)
        self.assertIn("microstructure_align_gate_pct", patient)
        self.assertIn("dc_align_gate_pct", patient)
        self.assertEqual(len(patient["role_signal_gate_mults"]), 4)
        self.assertEqual(len(patient["role_regime_buffer_mults"]), 4)
        self.assertEqual(urgent["flow_alignment_threshold"], 0.35)
        self.assertEqual(urgent["dc_alignment_threshold"], 0.30)
        self.assertEqual(urgent["min_alignment_votes"], 2)

    def test_liquidity_helper_blocks_only_stressed_entries(self) -> None:
        self.assertFalse(should_abstain_for_liquidity(0, 0.0, 80.0, 0.2, 35.0, 0.5))
        self.assertFalse(should_abstain_for_liquidity(1, 1.0, 80.0, 0.2, 35.0, 0.5))
        self.assertFalse(should_abstain_for_liquidity(1, 0.0, 30.0, 0.6, 35.0, 0.5))
        self.assertTrue(should_abstain_for_liquidity(1, 0.0, 80.0, 0.2, 35.0, 0.5))
        self.assertTrue(should_abstain_for_liquidity(-1, 0.0, 80.0, 1.0, 35.0, 0.0))
        self.assertTrue(should_abstain_for_liquidity(-1, 0.0, 10.0, 0.2, 0.0, 0.5))

    def test_extract_pair_execution_gene_prefers_pair_config(self) -> None:
        candidate = {
            "pair_configs": {
                "BTCUSDT": {
                    "execution_gene": {
                        "maker_priority": 0.75,
                        "max_wait_bars": 1,
                    }
                }
            },
            "execution_genes": {
                "BTCUSDT": {
                    "maker_priority": 0.20,
                    "max_wait_bars": 0,
                }
            },
        }

        gene = extract_pair_execution_gene(candidate, "BTCUSDT")

        self.assertIsNotNone(gene)
        self.assertEqual(gene["maker_priority"], 0.75)
        self.assertEqual(gene["max_wait_bars"], 1)

    def test_alignment_helpers_abstain_on_mismatch(self) -> None:
        bullish_micro = microstructure_alignment_score(0.6, 0.8)
        bearish_micro = microstructure_alignment_score(-0.6, 0.2)
        bullish_dc = dc_alignment_score(1.0, 0.5)
        bearish_dc = dc_alignment_score(-1.0, -0.5)

        self.assertGreater(bullish_micro, 0.0)
        self.assertLess(bearish_micro, 0.0)
        self.assertGreater(bullish_dc, 0.0)
        self.assertLess(bearish_dc, 0.0)
        self.assertTrue(should_abstain_for_alignment(1, bearish_micro, bullish_dc, 0.10, 0.10))
        self.assertTrue(should_abstain_for_alignment(-1, bullish_micro, bearish_dc, 0.10, 0.10))
        self.assertFalse(should_abstain_for_alignment(1, bullish_micro, bullish_dc, 0.10, 0.10))
        self.assertFalse(should_abstain_for_alignment(1, bullish_micro, bearish_dc, 0.10, 0.10, 1))

    def test_candle_microstructure_proxy_and_blend_capture_bullish_tape(self) -> None:
        bullish_candle = candle_microstructure_proxy_score(0.8, 0.7, 0.2)
        bearish_candle = candle_microstructure_proxy_score(-0.8, 0.7, -0.2)
        blended = blended_microstructure_score(0.4, 0.7, 0.8, 0.7, 0.2)

        self.assertGreater(bullish_candle, 0.0)
        self.assertLess(bearish_candle, 0.0)
        self.assertGreater(blended, bullish_candle * 0.5)

    def test_derivative_positioning_and_entry_quality_capture_supportive_tape(self) -> None:
        supportive_positioning = derivative_positioning_score(
            oi_rel=1.08,
            basis_rate=0.0005,
            top_pos_log_ratio=0.08,
            taker_buy_sell_log_ratio=0.60,
        )
        adverse_positioning = derivative_positioning_score(
            oi_rel=0.94,
            basis_rate=-0.0006,
            top_pos_log_ratio=-0.10,
            taker_buy_sell_log_ratio=-0.70,
        )
        entry_quality = blended_entry_quality_score(
            order_imbalance=0.4,
            buy_volume_share=0.7,
            close_location_value=0.8,
            body_to_range=0.7,
            wick_skew=0.2,
            oi_rel=1.08,
            basis_rate=0.0005,
            top_pos_log_ratio=0.08,
            taker_buy_sell_log_ratio=0.60,
        )

        self.assertGreater(supportive_positioning, 0.0)
        self.assertLess(adverse_positioning, 0.0)
        self.assertGreater(entry_quality, supportive_positioning * 0.5)

    def test_legacy_execution_profile_disables_new_alignment_and_abstain_gates(self) -> None:
        legacy = legacy_execution_profile()
        bullish_micro = microstructure_alignment_score(0.6, 0.8)
        bearish_dc = dc_alignment_score(-1.0, -0.5)

        self.assertEqual(legacy["signal_gate_pct"], 0.0)
        self.assertEqual(legacy["regime_buffer_mult"], 0.0)
        self.assertEqual(legacy["abstain_edge_pct"], 0.0)
        self.assertEqual(legacy["specialist_isolation_mult"], 0.0)
        self.assertEqual(legacy["min_alignment_votes"], 0)
        self.assertEqual(legacy["microstructure_align_gate_pct"], 0.0)
        self.assertEqual(legacy["dc_align_gate_pct"], 0.0)
        self.assertEqual(legacy["countertrend_weight_scale"], 0.0)
        self.assertEqual(legacy["countertrend_signal_floor_pct"], 0.0)
        self.assertFalse(
            should_abstain_for_alignment(
                1,
                bullish_micro,
                bearish_dc,
                legacy["microstructure_align_gate_pct"],
                legacy["dc_align_gate_pct"],
                legacy["min_alignment_votes"],
            )
        )
        self.assertFalse(
            should_allow_countertrend_entry(
                -1,
                signal_pct=-220.0,
                regime_score=0.03,
                breadth_score=0.5,
                breadth_threshold=0.65,
                microstructure_score=-0.2,
                positioning_score=-0.4,
                dc_score=-0.8,
                range_bps=12.0,
                volume_ratio=0.8,
                signal_floor_pct=legacy["countertrend_signal_floor_pct"],
                regime_score_cap=legacy["countertrend_regime_score_cap"],
                breadth_slack=legacy["countertrend_breadth_slack"],
                microstructure_floor=legacy["countertrend_microstructure_floor"],
                positioning_floor=legacy["countertrend_positioning_floor"],
                dc_floor=legacy["countertrend_dc_floor"],
                range_bps_floor=legacy["countertrend_range_bps_floor"],
                volume_ratio_floor=legacy["countertrend_volume_ratio_floor"],
            )
        )

    def test_countertrend_helper_allows_mild_bull_short_when_signal_is_extreme_and_aligned(self) -> None:
        self.assertTrue(
            should_allow_countertrend_entry(
                -1,
                signal_pct=-196.0,
                regime_score=0.04,
                breadth_score=0.5,
                breadth_threshold=0.65,
                microstructure_score=0.20,
                positioning_score=-0.48,
                dc_score=-0.75,
                range_bps=3.0,
                volume_ratio=0.33,
                signal_floor_pct=150.0,
                regime_score_cap=0.05,
                breadth_slack=0.2,
                microstructure_floor=-0.25,
                positioning_floor=0.15,
                dc_floor=0.5,
                range_bps_floor=2.0,
                volume_ratio_floor=0.25,
            )
        )
        self.assertFalse(
            should_allow_countertrend_entry(
                -1,
                signal_pct=-196.0,
                regime_score=0.08,
                breadth_score=0.5,
                breadth_threshold=0.65,
                microstructure_score=0.20,
                positioning_score=-0.48,
                dc_score=-0.75,
                range_bps=3.0,
                volume_ratio=0.33,
                signal_floor_pct=150.0,
                regime_score_cap=0.05,
                breadth_slack=0.2,
                microstructure_floor=-0.25,
                positioning_floor=0.15,
                dc_floor=0.5,
                range_bps_floor=2.0,
                volume_ratio_floor=0.25,
            )
        )

    def test_weak_tape_helper_blocks_new_entries_only_when_multiple_signals_are_weak(self) -> None:
        self.assertFalse(
            should_abstain_for_weak_tape(
                1,
                1.0,
                microstructure_score=0.0,
                dc_score=0.0,
                range_bps=10.0,
                volume_ratio=0.8,
                equity_corr_gross_scale=0.7,
                short_horizon_abstain_mult=1.0,
            )
        )

        self.assertFalse(
            should_abstain_for_weak_tape(
                1,
                0.0,
                microstructure_score=0.4,
                dc_score=0.3,
                range_bps=80.0,
                volume_ratio=1.5,
                equity_corr_gross_scale=1.0,
                short_horizon_abstain_mult=0.0,
            )
        )
        self.assertTrue(
            should_abstain_for_weak_tape(
                1,
                0.0,
                microstructure_score=-0.1,
                dc_score=-0.1,
                range_bps=20.0,
                volume_ratio=0.7,
                equity_corr_gross_scale=0.7,
                short_horizon_abstain_mult=1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
