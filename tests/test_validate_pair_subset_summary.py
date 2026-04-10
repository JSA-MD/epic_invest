import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_pair_subset_summary as pairwise_validation


def make_windows(
    *,
    recent_2m_return: float,
    recent_2m_mdd: float,
    recent_6m_return: float,
    recent_6m_mdd: float,
    full_4y_return: float,
    full_4y_mean: float,
    full_4y_mdd: float,
) -> dict[str, object]:
    return {
        "recent_2m": {
            "aggregate": {
                "worst_pair_avg_daily_return": recent_2m_return,
                "mean_avg_daily_return": recent_2m_return + 0.0003,
                "worst_max_drawdown": recent_2m_mdd,
            }
        },
        "recent_6m": {
            "aggregate": {
                "worst_pair_avg_daily_return": recent_6m_return,
                "mean_avg_daily_return": recent_6m_return + 0.0004,
                "worst_max_drawdown": recent_6m_mdd,
            }
        },
        "full_4y": {
            "aggregate": {
                "worst_pair_avg_daily_return": full_4y_return,
                "mean_avg_daily_return": full_4y_mean,
                "worst_max_drawdown": full_4y_mdd,
            }
        },
    }


class ValidatePairSubsetSummaryTests(unittest.TestCase):
    def test_progressive_profile_ignores_recent_2m_when_core_windows_are_strong(self) -> None:
        baseline = make_windows(
            recent_2m_return=0.0060,
            recent_2m_mdd=-0.10,
            recent_6m_return=0.0061,
            recent_6m_mdd=-0.12,
            full_4y_return=0.0042,
            full_4y_mean=0.0045,
            full_4y_mdd=-0.16,
        )
        candidate = make_windows(
            recent_2m_return=-0.0020,
            recent_2m_mdd=-0.11,
            recent_6m_return=0.0065,
            recent_6m_mdd=-0.11,
            full_4y_return=0.0048,
            full_4y_mean=0.0049,
            full_4y_mdd=-0.14,
        )
        profile = pairwise_validation.build_progressive_profile_for_windows(candidate, baseline)
        self.assertTrue(profile["passed"])
        self.assertFalse(any("recent_2m" in check.name for check in profile["checks"]))

    def test_final_oos_profile_blocks_recent_2m_breakdown(self) -> None:
        baseline = make_windows(
            recent_2m_return=0.0060,
            recent_2m_mdd=-0.10,
            recent_6m_return=0.0061,
            recent_6m_mdd=-0.12,
            full_4y_return=0.0042,
            full_4y_mean=0.0045,
            full_4y_mdd=-0.16,
        )
        candidate = make_windows(
            recent_2m_return=-0.0010,
            recent_2m_mdd=-0.22,
            recent_6m_return=0.0065,
            recent_6m_mdd=-0.11,
            full_4y_return=0.0048,
            full_4y_mean=0.0049,
            full_4y_mdd=-0.14,
        )
        profile = pairwise_validation.build_final_oos_profile_for_windows(candidate, baseline)
        self.assertFalse(profile["passed"])
        failed = [check.name for check in profile["checks"] if not check.passed]
        self.assertIn("recent_2m_worst_pair_positive", failed)
        self.assertIn("recent_2m_worst_mdd_cap", failed)

    def test_target_060_profile_ignores_recent_2m_when_core_windows_hit_target(self) -> None:
        candidate = make_windows(
            recent_2m_return=-0.0040,
            recent_2m_mdd=-0.25,
            recent_6m_return=0.0064,
            recent_6m_mdd=-0.11,
            full_4y_return=0.0062,
            full_4y_mean=0.0065,
            full_4y_mdd=-0.14,
        )
        profile = pairwise_validation.build_target_060_profile_for_windows(candidate)

        self.assertTrue(profile["passed"])
        self.assertFalse(any("recent_2m" in check.name for check in profile["checks"]))


if __name__ == "__main__":
    unittest.main()
