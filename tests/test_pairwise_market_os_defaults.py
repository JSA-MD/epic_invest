import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import repair_pair_subset_pairwise_candidate as repair_pairwise


class PairwiseMarketOperatingDefaultsTests(unittest.TestCase):
    def test_repair_defaults_to_equity_corr_validated_output(self) -> None:
        with patch.object(sys, "argv", ["repair_pair_subset_pairwise_candidate.py", "--candidate-summaries", "a.json"]):
            args = repair_pairwise.parse_args()

        self.assertEqual(args.route_state_mode, "equity_corr")
        self.assertEqual(args.start_candidate_count, 6)
        self.assertEqual(args.stress_proxy_candidate_count, 6)
        self.assertEqual(
            Path(args.summary_out).name,
            "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_candidate_summary.json",
        )


if __name__ == "__main__":
    unittest.main()
