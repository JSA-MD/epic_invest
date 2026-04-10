import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_pair_subset_pairwise_stress as stress


VALIDATED_SUMMARY_NAME = "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"


def make_summary(*, validation_pass: bool = True, market_os_pass: bool = True, audit_pass: bool = True) -> dict:
    return {
        "selected_candidate": {
            "validation_engine": {
                "gate": {
                    "passed": validation_pass,
                    "failed_checks": [] if validation_pass else ["passes_validation_quality"],
                },
                "market_operating_system": {
                    "gate": {
                        "passed": market_os_pass,
                        "failed_checks": [] if market_os_pass else ["passes_market_os_fitness"],
                    },
                    "audit": {
                        "passed": audit_pass,
                        "failed_checks": [] if audit_pass else ["passes_final_oos_total_return"],
                    },
                },
            }
        }
    }


def make_report(*, target_pass: bool = True, progressive_pass: bool = True) -> dict:
    return {
        "profiles": {
            "target_060_stress": {
                "selected": {
                    "passed": target_pass,
                    "checks": [
                        {
                            "name": "recent_2m_worst_pair_target_060_all",
                            "passed": target_pass,
                        }
                    ],
                }
            },
            "progressive_stress": {
                "selected": {
                    "passed": progressive_pass,
                    "checks": [
                        {
                            "name": "recent_2m_worst_pair_positive_all",
                            "passed": progressive_pass,
                        }
                    ],
                }
            },
        }
    }


class PairwiseStressReportTests(unittest.TestCase):
    def test_default_summary_path_uses_validated_pairwise_summary(self) -> None:
        with patch.object(sys, "argv", ["run_pair_subset_pairwise_stress.py"]):
            args = stress.parse_args()
        self.assertEqual(args.summary, str(ROOT_DIR / "models" / VALIDATED_SUMMARY_NAME))

    def test_validation_failure_blocks_ready(self) -> None:
        decision = stress.build_promotion_decision(
            make_summary(validation_pass=False, market_os_pass=True, audit_pass=True),
            make_report(target_pass=True, progressive_pass=True),
        )

        self.assertEqual(decision["status"], "validation_fail")
        self.assertFalse(decision["ready_for_merge"])
        self.assertFalse(decision["ready_for_live"])
        self.assertIn("validation_gate.passes_validation_quality", decision["failed_checks"])

    def test_stress_failure_blocks_ready(self) -> None:
        decision = stress.build_promotion_decision(
            make_summary(validation_pass=True, market_os_pass=True, audit_pass=True),
            make_report(target_pass=False, progressive_pass=False),
        )

        self.assertEqual(decision["status"], "stress_fail")
        self.assertFalse(decision["ready_for_merge"])
        self.assertFalse(decision["ready_for_live"])
        self.assertIn("target_060_stress.recent_2m_worst_pair_target_060_all", decision["failed_checks"])
        self.assertIn("progressive_stress.recent_2m_worst_pair_positive_all", decision["failed_checks"])


if __name__ == "__main__":
    unittest.main()
