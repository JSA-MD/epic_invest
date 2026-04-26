"""
Tests verifying that module-load env reads in pairwise_regime_live survive
invalid values by falling back to defaults instead of raising.
"""
import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"


def reimport_with_env(env_overrides: dict):
    """Force a fresh import of pairwise_regime_live with env overrides applied."""
    sys.modules.pop("pairwise_regime_live", None)
    for k, v in env_overrides.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module("pairwise_regime_live")


class TestModuleSafeLoads(unittest.TestCase):

    def tearDown(self):
        # Clean up injected env vars after each test
        for key in ("PAIRWISE_GROSS_CAP", "PAIRWISE_MAX_HOLD_BARS", "PAIRWISE_CVAR_CUT_HOLD_HOURS"):
            os.environ.pop(key, None)
        sys.modules.pop("pairwise_regime_live", None)

    def test_module_loads_with_invalid_pairwise_gross_cap(self):
        """PAIRWISE_GROSS_CAP=abc must not raise; module loads with default 1.0."""
        mod = reimport_with_env({"PAIRWISE_GROSS_CAP": "abc"})
        self.assertAlmostEqual(mod.PAIRWISE_GROSS_CAP, 1.0)

    def test_module_loads_with_invalid_max_hold_bars(self):
        """PAIRWISE_MAX_HOLD_BARS=foo must not raise; module loads with default 288."""
        mod = reimport_with_env({"PAIRWISE_MAX_HOLD_BARS": "foo"})
        self.assertEqual(mod.PAIRWISE_MAX_HOLD_BARS, 288)

    def test_module_loads_with_invalid_cvar_cut_hold_hours(self):
        """PAIRWISE_CVAR_CUT_HOLD_HOURS=xyz must not raise; module loads with default 24."""
        mod = reimport_with_env({"PAIRWISE_CVAR_CUT_HOLD_HOURS": "xyz"})
        self.assertEqual(mod.PAIRWISE_CVAR_CUT_HOLD_HOURS, 24)

    def test_max_hold_seconds_correct_with_default(self):
        """_MAX_HOLD_SECONDS must equal 288 * 5 * 60 when PAIRWISE_MAX_HOLD_BARS is invalid."""
        mod = reimport_with_env({"PAIRWISE_MAX_HOLD_BARS": "bad"})
        self.assertEqual(mod._MAX_HOLD_SECONDS, 288 * 5 * 60)

    def test_valid_values_still_accepted(self):
        """Valid env values must still be parsed correctly (no regression)."""
        mod = reimport_with_env({
            "PAIRWISE_GROSS_CAP": "0.03",
            "PAIRWISE_MAX_HOLD_BARS": "144",
            "PAIRWISE_CVAR_CUT_HOLD_HOURS": "12",
        })
        self.assertAlmostEqual(mod.PAIRWISE_GROSS_CAP, 0.03)
        self.assertEqual(mod.PAIRWISE_MAX_HOLD_BARS, 144)
        self.assertEqual(mod.PAIRWISE_CVAR_CUT_HOLD_HOURS, 12)


if __name__ == "__main__":
    unittest.main()
