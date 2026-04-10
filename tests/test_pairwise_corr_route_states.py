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
        corr_codes = pairwise_search.build_route_bucket_codes(
            index,
            overlay_inputs,
            0.5,
            route_state_mode=pairwise_search.ROUTE_STATE_MODE_EQUITY_CORR,
        )

        self.assertEqual(base_codes.tolist(), [0, 1, 2, 3])
        self.assertEqual(corr_codes.tolist(), [0, 5, 10, 7])

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


if __name__ == "__main__":
    unittest.main()
