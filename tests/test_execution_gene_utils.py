import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from execution_gene_utils import (
    dc_alignment_score,
    derive_execution_profile,
    extract_pair_execution_gene,
    legacy_execution_profile,
    microstructure_alignment_score,
    should_abstain_for_alignment,
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
        self.assertIn("microstructure_align_gate_pct", patient)
        self.assertIn("dc_align_gate_pct", patient)
        self.assertEqual(len(patient["role_signal_gate_mults"]), 4)
        self.assertEqual(len(patient["role_regime_buffer_mults"]), 4)
        self.assertEqual(urgent["flow_alignment_threshold"], 0.35)
        self.assertEqual(urgent["dc_alignment_threshold"], 0.30)
        self.assertEqual(urgent["min_alignment_votes"], 2)

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


if __name__ == "__main__":
    unittest.main()
