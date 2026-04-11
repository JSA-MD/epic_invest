import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_fractal_market_os_deep_validation as deep_validation


class FractalDeepValidationTests(unittest.TestCase):
    def test_collect_candidate_entries_deduplicates_across_views(self) -> None:
        campaign_report = {
            "top_candidates": [
                {"seed": 1, "artifacts": {"search_summary": "/tmp/a.json", "pipeline_report": "/tmp/a.pipe"}},
            ],
            "best_by_observation_mode": {
                "time": {"seed": 1, "artifacts": {"search_summary": "/tmp/a.json", "pipeline_report": "/tmp/a.pipe"}},
                "imbalance": {"seed": 2, "artifacts": {"search_summary": "/tmp/b.json", "pipeline_report": "/tmp/b.pipe"}},
            },
            "best_by_label_horizon": {
                "4h": {"seed": 2, "artifacts": {"search_summary": "/tmp/b.json", "pipeline_report": "/tmp/b.pipe"}},
                "5m": {"seed": 3, "artifacts": {"search_summary": "/tmp/c.json", "pipeline_report": "/tmp/c.pipe"}},
            },
        }

        entries = deep_validation.collect_candidate_entries(campaign_report, top_n=2)

        self.assertEqual([entry["seed"] for entry in entries], [1, 2, 3])

    def test_classify_candidate_returns_watch_for_wf1_and_near_frontier_profile(self) -> None:
        record = {
            "decision": {"ready_for_merge": False},
            "metrics": {
                "wf1_passed": True,
                "target_060_passed": False,
                "final_oos_passed": False,
                "stress_gate_passed": False,
                "recent_6m_worst_daily": 0.0068,
                "full_4y_worst_daily": 0.0047,
                "full_4y_mdd": 0.12,
                "latest_fold_stress_reserve_score": -100.0,
            },
        }

        band, blockers = deep_validation.classify_candidate(record)

        self.assertEqual(band, "watch")
        self.assertIn("target_060_failed", blockers)

    def test_classify_candidate_returns_drop_when_gate_progress_is_weak(self) -> None:
        record = {
            "decision": {"ready_for_merge": False},
            "metrics": {
                "wf1_passed": False,
                "target_060_passed": False,
                "final_oos_passed": False,
                "stress_gate_passed": False,
                "recent_6m_worst_daily": 0.004,
                "full_4y_worst_daily": 0.001,
                "full_4y_mdd": 0.20,
                "latest_fold_stress_reserve_score": -500.0,
            },
        }

        band, blockers = deep_validation.classify_candidate(record)

        self.assertEqual(band, "drop")
        self.assertIn("wf1_failed", blockers)


if __name__ == "__main__":
    unittest.main()
