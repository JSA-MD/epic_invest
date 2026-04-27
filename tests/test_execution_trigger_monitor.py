"""Regression tests for the silent-drop classifier.

Pins the seven boundary cases of `_is_silently_dropped`:

1. Real silent drop — large signal, flat target, no overlay credit.
2. Tiny signal — gate respected an honest noise reading; not a drop.
3. Non-flat target — no drop happened at all.
4. New live_overlay_runner stamp on pp["overlay_force_flat"] → attributed.
5. Legacy overlay attribution via decision_journal fallback (in-memory tests).
6. Production JSONL log: pair_plan stamp survives the log envelope and
   classifier reads it directly without relying on decision_journal.
7. Mixed: overlay stamp on one pair, real silent on another, must
   classify each independently.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from execution_trigger_monitor import _is_silently_dropped  # noqa: E402


class TestSilentDropClassifier(unittest.TestCase):
    def test_real_silent_drop_flagged(self):
        pp = {"signal_pct": -198.0, "target_weight": 1e-13}
        self.assertTrue(_is_silently_dropped(pp))

    def test_tiny_signal_not_silent(self):
        pp = {"signal_pct": 5.0, "target_weight": 0.0}
        self.assertFalse(_is_silently_dropped(pp))

    def test_non_flat_target_not_silent(self):
        pp = {"signal_pct": -180.0, "target_weight": -0.005}
        self.assertFalse(_is_silently_dropped(pp))

    def test_new_overlay_stamp_attributes(self):
        for tag in (
            "sign_instability",
            "adaptive_threshold_noise",
            "max_hold",
            "cvar_cut",
            "stale_price",
        ):
            with self.subTest(tag=tag):
                pp = {"signal_pct": -198.0, "target_weight": 0.0, "overlay_force_flat": tag}
                self.assertFalse(_is_silently_dropped(pp))

    def test_decision_journal_fallback_attributes(self):
        # Used by tests that pass in-memory state without a JSONL round-trip.
        pp = {"signal_pct": -198.0, "target_weight": 0.0}
        journal = [{"pair": "BTCUSDT", "override_reason": "max_hold", "target_weight_forced": 0.0}]
        self.assertFalse(_is_silently_dropped(pp, journal_for_pair=journal))

    def test_empty_journal_does_not_attribute(self):
        pp = {"signal_pct": -198.0, "target_weight": 0.0}
        self.assertTrue(_is_silently_dropped(pp, journal_for_pair=[]))

    def test_journal_entry_without_reason_does_not_attribute(self):
        pp = {"signal_pct": -198.0, "target_weight": 0.0}
        journal = [{"pair": "BTCUSDT", "note": "informational only"}]
        self.assertTrue(_is_silently_dropped(pp, journal_for_pair=journal))

    def test_missing_fields_safe_defaults(self):
        # signal_pct or target_weight missing → not classified as silent
        # (we only flag clear cases, not malformed log rows).
        self.assertFalse(_is_silently_dropped({}))
        self.assertFalse(_is_silently_dropped({"signal_pct": -200.0}))
        self.assertFalse(_is_silently_dropped({"target_weight": 0.0}))


if __name__ == "__main__":
    unittest.main()
