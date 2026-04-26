import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_regime_live as prl


class TestMinutesSince(unittest.TestCase):
    """minutes_since returns inf on parse failure (staleness sentinel)."""

    def test_none_returns_inf(self):
        import math
        self.assertTrue(math.isinf(prl.minutes_since(None)))

    def test_invalid_string_returns_inf(self):
        import math
        self.assertTrue(math.isinf(prl.minutes_since("not-a-date")))

    def test_valid_timestamp_returns_expected_minutes(self):
        frozen_now = datetime.now(UTC)
        ts = (frozen_now - timedelta(minutes=10)).isoformat()
        with patch.object(prl, "utc_now", return_value=frozen_now):
            result = prl.minutes_since(ts)
        self.assertAlmostEqual(result, 10.0, delta=0.01)


class TestMinutesSinceOrZero(unittest.TestCase):
    """minutes_since_or_zero returns 0.0 on parse failure (safe default)."""

    def test_none_returns_zero(self):
        self.assertEqual(prl.minutes_since_or_zero(None), 0.0)

    def test_invalid_string_returns_zero(self):
        self.assertEqual(prl.minutes_since_or_zero("not-a-date"), 0.0)

    def test_valid_timestamp_returns_expected_minutes(self):
        frozen_now = datetime.now(UTC)
        ts = (frozen_now - timedelta(minutes=5)).isoformat()
        with patch.object(prl, "utc_now", return_value=frozen_now):
            result = prl.minutes_since_or_zero(ts)
        self.assertAlmostEqual(result, 5.0, delta=0.01)

    def test_none_does_not_trigger_stale_gate(self):
        # 0.0 is less than any positive max_stale_minutes — missing data appears fresh
        self.assertFalse(prl.minutes_since_or_zero(None) > 30.0)

    def test_contrast_with_minutes_since_on_none(self):
        # minutes_since(None) is inf — fails staleness gate; minutes_since_or_zero(None) is 0.0 — passes
        import math
        self.assertTrue(math.isinf(prl.minutes_since(None)))
        self.assertEqual(prl.minutes_since_or_zero(None), 0.0)


if __name__ == "__main__":
    unittest.main()
