import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from safety_guards import (
    check_position_divergence,
    clip_candidate_gross_cap,
    enforce_runtime_gross_cap_ceiling,
    update_stale_price_tracker,
    validate_safety_switches,
    validate_state_alphas_coverage,
)


def _make_candidate(bnb_gross_cap=None, btc_gross_cap=None, mode="state_alphas", state_alphas=None):
    pair_configs = {}
    if btc_gross_cap is not None:
        pair_configs["BTCUSDT"] = {"gross_cap": btc_gross_cap}
    if bnb_gross_cap is not None:
        pair_configs["BNBUSDT"] = {"gross_cap": bnb_gross_cap}
    blend = {"pair": "BNBUSDT", "alpha": 0.2, "mode": mode}
    if state_alphas is not None:
        blend["state_alphas"] = state_alphas
    elif mode == "state_alphas":
        blend["state_alphas"] = {"equity_mixed:bull_broad": 0.2}
    return {
        "selected_candidate": {
            "pair_configs": pair_configs,
            "pair_convex_blends": {"BNBUSDT": blend},
        }
    }


# ---------------------------------------------------------------------------
# TestEnforceRuntimeGrossCapCeiling
# ---------------------------------------------------------------------------

class TestEnforceRuntimeGrossCapCeiling(unittest.TestCase):

    def test_safe_default_clamps_valid_runtime(self):
        # Stage 0 lockdown: SAFE_DEFAULT=0.01 acts as an absolute hard ceiling.
        # Even valid runtime/ceiling above SAFE_DEFAULT are clipped down. This
        # is what makes lockdown durable across watchdog restarts.
        env = {"PAIRWISE_GROSS_CAP": "0.03", "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("Stage 0 lockdown", warning)

    def test_clip_when_over_ceiling(self):
        # PAIRWISE_GROSS_CAP=1.5 exceeds HARD_MAX=1.0 → invalid; ceiling=0.05 valid.
        # effective = min(SAFE_DEFAULT=0.01, ceiling=0.05) = 0.01
        env = {"PAIRWISE_GROSS_CAP": "1.5", "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("1.5", warning)
        self.assertIn("0.01", warning)

    def test_default_ceiling_when_unset(self):
        env: dict = {}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        # Stage 0 lockdown: defaults are 0.01 / 0.01 → effective = 0.01
        self.assertAlmostEqual(effective, 0.01)

    def test_unparseable_runtime_warns_now(self):
        env = {"PAIRWISE_GROSS_CAP": "abc"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("PAIRWISE_GROSS_CAP", warning)

    def test_negative_runtime_falls_back_to_safe_default(self):
        env = {"PAIRWISE_GROSS_CAP": "-1.0"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("PAIRWISE_GROSS_CAP", warning)
        self.assertIn("negative", warning)

    def test_negative_ceiling_falls_back_to_safe_default(self):
        env = {"PAIRWISE_LIVE_MAX_GROSS_CAP": "-0.5"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("PAIRWISE_LIVE_MAX_GROSS_CAP", warning)
        self.assertIn("negative", warning)

    def test_nan_runtime_falls_back(self):
        env = {"PAIRWISE_GROSS_CAP": "nan"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("NaN/inf", warning)

    def test_inf_runtime_falls_back(self):
        env = {"PAIRWISE_GROSS_CAP": "inf"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("NaN/inf", warning)

    def test_above_hard_max_falls_back(self):
        env = {"PAIRWISE_GROSS_CAP": "5.0"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("HARD_MAX", warning)

    def test_both_invalid_combines_warnings(self):
        # Both invalid → effective = SAFE_DEFAULT = 0.01 (only candidate)
        env = {"PAIRWISE_GROSS_CAP": "-1", "PAIRWISE_LIVE_MAX_GROSS_CAP": "abc"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("PAIRWISE_GROSS_CAP", warning)
        self.assertIn("PAIRWISE_LIVE_MAX_GROSS_CAP", warning)

    # --- New invariant: invalid env var never widens exposure ---

    def test_invalid_ceiling_does_not_widen_strict_runtime(self):
        # User set a strict 1% cap; ceiling has a typo.
        # Invalid ceiling must NOT let effective fall back to the wider SAFE_DEFAULT.
        env = {"PAIRWISE_GROSS_CAP": "0.01", "PAIRWISE_LIVE_MAX_GROSS_CAP": "abc"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("PAIRWISE_LIVE_MAX_GROSS_CAP", warning)
        self.assertIn("0.01", warning)

    def test_invalid_runtime_does_not_widen_strict_ceiling(self):
        # Strict ceiling 0.005 (tighter than SAFE_DEFAULT=0.01); runtime invalid.
        # Effective must honour the strict ceiling, not fall back to SAFE_DEFAULT.
        env = {"PAIRWISE_GROSS_CAP": "-1", "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.005"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.005)
        self.assertIsNotNone(warning)
        self.assertIn("PAIRWISE_GROSS_CAP", warning)

    def test_both_invalid_uses_safe_default(self):
        # No valid inputs at all → only candidate is SAFE_DEFAULT=0.01
        env = {"PAIRWISE_GROSS_CAP": "-1", "PAIRWISE_LIVE_MAX_GROSS_CAP": "abc"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)

    def test_invalid_ceiling_with_loose_runtime_falls_back_to_safe_default(self):
        # Runtime=0.5 (valid but loose); ceiling invalid.
        # effective = min(SAFE_DEFAULT=0.01, runtime=0.5) = 0.01
        env = {"PAIRWISE_GROSS_CAP": "0.5", "PAIRWISE_LIVE_MAX_GROSS_CAP": "abc"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("PAIRWISE_LIVE_MAX_GROSS_CAP", warning)

    def test_zero_runtime_with_invalid_ceiling(self):
        # Zero runtime is the strictest possible cap; ceiling is invalid.
        # The zero value must be honoured — no widening to SAFE_DEFAULT.
        env = {"PAIRWISE_GROSS_CAP": "0.0", "PAIRWISE_LIVE_MAX_GROSS_CAP": "abc"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.0)
        self.assertIsNotNone(warning)

    def test_hard_max_boundary_exact(self):
        # 1.0 is valid (== HARD_MAX); but 1.0 > SAFE_DEFAULT 0.01 → clips, not fallback
        env = {"PAIRWISE_GROSS_CAP": "1.0"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        self.assertIn("clipping", warning)
        self.assertNotIn("HARD_MAX", warning)

    def test_zero_runtime_is_valid(self):
        env = {"PAIRWISE_GROSS_CAP": "0.0", "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.0)
        self.assertIsNone(warning)

    def test_default_pairwise_gross_cap_is_safe_default(self):
        # No PAIRWISE_GROSS_CAP set → defaults to 0.01 (SAFE_DEFAULT post-Stage 0)
        # Ceiling=0.05 explicit, runtime=0.01 default. effective = min(0.01, 0.01, 0.05) = 0.01
        env = {"PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNone(warning)

    def test_at_ceiling_returns_no_warning(self):
        # Both runtime and ceiling at 0.01 (Stage 0 lockdown); no clip warning expected.
        env = {"PAIRWISE_GROSS_CAP": "0.01", "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.01"}
        effective, warning = enforce_runtime_gross_cap_ceiling(env)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNone(warning)

    def test_reads_os_environ_when_env_is_none(self):
        with patch.dict("os.environ", {"PAIRWISE_GROSS_CAP": "1.5", "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05"}):
            effective, warning = enforce_runtime_gross_cap_ceiling(None)
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)


# ---------------------------------------------------------------------------
# TestClipCandidateGrossCapDeprecated
# ---------------------------------------------------------------------------

class TestClipCandidateGrossCapDeprecated(unittest.TestCase):

    def test_clip_candidate_gross_cap_is_deprecated_noop(self):
        """clip_candidate_gross_cap is superseded — it must return the input unchanged."""
        candidate = _make_candidate(bnb_gross_cap=1.5, btc_gross_cap=1.5)
        result = clip_candidate_gross_cap(candidate)
        self.assertIs(result, candidate)
        # Values must be untouched — the function no longer clips anything
        sc = result["selected_candidate"]["pair_configs"]
        self.assertAlmostEqual(sc["BNBUSDT"]["gross_cap"], 1.5)
        self.assertAlmostEqual(sc["BTCUSDT"]["gross_cap"], 1.5)

    def test_no_error_on_empty_candidate(self):
        result = clip_candidate_gross_cap({})
        self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# TestCheckPositionDivergence
# ---------------------------------------------------------------------------

class TestCheckPositionDivergence(unittest.TestCase):

    def _post_trade_state(self, positions: dict) -> dict:
        return {
            "latest_live_sync": {
                "source": "run_live_once_post_trade",
                "positions": positions,
            }
        }

    def test_no_divergence_returns_none(self):
        prev = {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}}
        curr = {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}}
        state = self._post_trade_state(prev)
        result = check_position_divergence(state, curr, equity=1000.0)
        self.assertIsNone(result)

    def test_divergence_above_threshold_returns_message(self):
        # prev BNB qty=-10, current qty=0, equity=$1000, price=$600
        # notional diff = |0 - (-10)| * 600 = 6000 > 1000*0.01=10
        prev = {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}}
        curr = {"BNBUSDT": {"qty": 0.0, "mark_price": 600.0}}
        state = self._post_trade_state(prev)
        result = check_position_divergence(state, curr, equity=1000.0)
        self.assertIsNotNone(result)
        self.assertIn("BNBUSDT", result)

    def test_missing_post_trade_snapshot_returns_none(self):
        state = {}
        curr = {"BNBUSDT": {"qty": 0.0, "mark_price": 600.0}}
        result = check_position_divergence(state, curr, equity=1000.0)
        self.assertIsNone(result)

    def test_wrong_source_returns_none(self):
        state = {
            "latest_live_sync": {
                "source": "run_live_once",
                "positions": {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}},
            }
        }
        curr = {"BNBUSDT": {"qty": 0.0, "mark_price": 600.0}}
        result = check_position_divergence(state, curr, equity=1000.0)
        self.assertIsNone(result)

    def test_missing_equity_returns_none(self):
        prev = {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}}
        curr = {"BNBUSDT": {"qty": 0.0, "mark_price": 600.0}}
        state = self._post_trade_state(prev)
        result = check_position_divergence(state, curr, equity=None)
        self.assertIsNone(result)

    def test_small_divergence_below_threshold_returns_none(self):
        # diff = |(-10.001) - (-10.0)| * 600 = 0.6 < 1000*0.01=10
        prev = {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}}
        curr = {"BNBUSDT": {"qty": -10.001, "mark_price": 600.0}}
        state = self._post_trade_state(prev)
        result = check_position_divergence(state, curr, equity=1000.0)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# TestUpdateStalePriceTracker
# ---------------------------------------------------------------------------

class TestUpdateStalePriceTracker(unittest.TestCase):

    def test_first_call_initializes(self):
        state = {}
        stale = update_stale_price_tracker(state, {"BNBUSDT": 600.0})
        self.assertIn("stale_price_counts", state)
        self.assertIn("last_known_prices", state)
        self.assertEqual(state["last_known_prices"]["BNBUSDT"], 600.0)
        self.assertEqual(stale, [])

    def test_increments_on_unchanged_price(self):
        state = {
            "stale_price_counts": {"BNBUSDT": 2},
            "last_known_prices": {"BNBUSDT": 600.0},
        }
        stale = update_stale_price_tracker(state, {"BNBUSDT": 600.0})
        self.assertEqual(state["stale_price_counts"]["BNBUSDT"], 3)
        self.assertEqual(stale, [])

    def test_returns_pair_when_stale_count_exceeds(self):
        state = {
            "stale_price_counts": {"BNBUSDT": 12},
            "last_known_prices": {"BNBUSDT": 600.0},
        }
        stale = update_stale_price_tracker(state, {"BNBUSDT": 600.0})
        self.assertEqual(state["stale_price_counts"]["BNBUSDT"], 13)
        self.assertIn("BNBUSDT", stale)

    def test_not_stale_at_exactly_max(self):
        state = {
            "stale_price_counts": {"BNBUSDT": 11},
            "last_known_prices": {"BNBUSDT": 600.0},
        }
        stale = update_stale_price_tracker(state, {"BNBUSDT": 600.0}, max_stale_cycles=12)
        self.assertEqual(state["stale_price_counts"]["BNBUSDT"], 12)
        self.assertEqual(stale, [])

    def test_resets_on_price_change(self):
        state = {
            "stale_price_counts": {"BNBUSDT": 5},
            "last_known_prices": {"BNBUSDT": 600.0},
        }
        stale = update_stale_price_tracker(state, {"BNBUSDT": 605.0})
        self.assertEqual(state["stale_price_counts"]["BNBUSDT"], 0)
        self.assertEqual(stale, [])

    def test_handles_none_prices(self):
        state = {}
        stale = update_stale_price_tracker(state, {"BNBUSDT": None})
        self.assertEqual(stale, [])

    def test_handles_missing_state_keys(self):
        state = {"stale_price_counts": None, "last_known_prices": None}
        stale = update_stale_price_tracker(state, {"BNBUSDT": 600.0})
        self.assertEqual(stale, [])

    def test_multiple_pairs(self):
        state = {
            "stale_price_counts": {"BNBUSDT": 12, "BTCUSDT": 2},
            "last_known_prices": {"BNBUSDT": 600.0, "BTCUSDT": 90000.0},
        }
        stale = update_stale_price_tracker(state, {"BNBUSDT": 600.0, "BTCUSDT": 90000.0})
        self.assertIn("BNBUSDT", stale)
        self.assertNotIn("BTCUSDT", stale)


# ---------------------------------------------------------------------------
# TestValidateSafetySwitches
# ---------------------------------------------------------------------------

class TestValidateSafetySwitches(unittest.TestCase):

    def test_all_safe_returns_empty(self):
        env = {
            "PAIRWISE_MAX_HOLD_BARS": "288",
            "PAIRWISE_CVAR_CUT": "1",
            "PAIRWISE_FORCE_EXECUTE": "0",
            "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05",
        }
        result = validate_safety_switches(env)
        self.assertEqual(result, [])

    def test_force_execute_warns(self):
        env = {"PAIRWISE_FORCE_EXECUTE": "1"}
        result = validate_safety_switches(env)
        self.assertTrue(any("PAIRWISE_FORCE_EXECUTE" in w for w in result))

    def test_max_hold_too_low_warns(self):
        env = {"PAIRWISE_MAX_HOLD_BARS": "1"}
        result = validate_safety_switches(env)
        self.assertTrue(any("PAIRWISE_MAX_HOLD_BARS" in w for w in result))

    def test_max_hold_too_high_warns(self):
        env = {"PAIRWISE_MAX_HOLD_BARS": "5000"}
        result = validate_safety_switches(env)
        self.assertTrue(any("PAIRWISE_MAX_HOLD_BARS" in w for w in result))

    def test_max_hold_at_boundary_ok(self):
        result = validate_safety_switches({"PAIRWISE_MAX_HOLD_BARS": "12"})
        self.assertEqual(result, [])
        result = validate_safety_switches({"PAIRWISE_MAX_HOLD_BARS": "2880"})
        self.assertEqual(result, [])

    def test_cvar_cut_disabled_warns(self):
        env = {"PAIRWISE_CVAR_CUT": "0"}
        result = validate_safety_switches(env)
        self.assertTrue(any("PAIRWISE_CVAR_CUT" in w for w in result))

    def test_cvar_cut_false_warns(self):
        env = {"PAIRWISE_CVAR_CUT": "false"}
        result = validate_safety_switches(env)
        self.assertTrue(any("PAIRWISE_CVAR_CUT" in w for w in result))

    def test_gross_cap_too_high_warns(self):
        env = {"PAIRWISE_LIVE_MAX_GROSS_CAP": "0.5"}
        result = validate_safety_switches(env)
        self.assertTrue(any("PAIRWISE_LIVE_MAX_GROSS_CAP" in w for w in result))

    def test_gross_cap_at_stage_a_ceiling_ok(self):
        # Post Stage 0 lockdown, the no-warn ceiling is 0.05 (Stage A).
        result = validate_safety_switches({"PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05"})
        self.assertEqual(result, [])

    def test_gross_cap_above_stage_a_ceiling_warns(self):
        # 0.20 is the legacy defensive ceiling but post Stage 0 lockdown it
        # exceeds the Stage A 0.05 cap and must warn.
        result = validate_safety_switches({"PAIRWISE_LIVE_MAX_GROSS_CAP": "0.20"})
        self.assertTrue(any("Stage A ceiling" in w for w in result))
        # Above 0.20 it also trips the legacy defensive warning.
        result_high = validate_safety_switches({"PAIRWISE_LIVE_MAX_GROSS_CAP": "0.50"})
        self.assertTrue(any("Stage A ceiling" in w for w in result_high))
        self.assertTrue(any("defensive ceiling 0.20" in w for w in result_high))

    def test_missing_env_vars_return_empty(self):
        result = validate_safety_switches({})
        self.assertEqual(result, [])

    def test_reads_os_environ_by_default(self):
        with patch.dict("os.environ", {"PAIRWISE_FORCE_EXECUTE": "1"}):
            result = validate_safety_switches()
        self.assertTrue(any("PAIRWISE_FORCE_EXECUTE" in w for w in result))

    def test_max_hold_unparseable_warns(self):
        result = validate_safety_switches({"PAIRWISE_MAX_HOLD_BARS": "foo"})
        self.assertTrue(any("PAIRWISE_MAX_HOLD_BARS" in w for w in result))
        self.assertTrue(any("valid integer" in w for w in result))

    def test_gross_cap_unparseable_warns(self):
        result = validate_safety_switches({"PAIRWISE_LIVE_MAX_GROSS_CAP": "abc"})
        self.assertTrue(any("PAIRWISE_LIVE_MAX_GROSS_CAP" in w for w in result))
        self.assertTrue(any("valid number" in w for w in result))

    def test_cvar_cut_unparseable_warns(self):
        result = validate_safety_switches({"PAIRWISE_CVAR_CUT": "maybe"})
        self.assertTrue(any("PAIRWISE_CVAR_CUT" in w for w in result))
        self.assertTrue(any("valid boolean" in w for w in result))

    def test_force_execute_unparseable_warns(self):
        result = validate_safety_switches({"PAIRWISE_FORCE_EXECUTE": "oui"})
        self.assertTrue(any("PAIRWISE_FORCE_EXECUTE" in w for w in result))
        self.assertTrue(any("valid boolean" in w for w in result))

    def test_cvar_hold_unparseable_warns(self):
        result = validate_safety_switches({"PAIRWISE_CVAR_CUT_HOLD_HOURS": "foo"})
        self.assertTrue(any("PAIRWISE_CVAR_CUT_HOLD_HOURS" in w for w in result))
        self.assertTrue(any("valid integer" in w for w in result))

    def test_cvar_hold_out_of_range_low_warns(self):
        result = validate_safety_switches({"PAIRWISE_CVAR_CUT_HOLD_HOURS": "0"})
        self.assertTrue(any("PAIRWISE_CVAR_CUT_HOLD_HOURS" in w for w in result))
        self.assertTrue(any("[1, 168]" in w for w in result))

    def test_cvar_hold_out_of_range_high_warns(self):
        result = validate_safety_switches({"PAIRWISE_CVAR_CUT_HOLD_HOURS": "500"})
        self.assertTrue(any("PAIRWISE_CVAR_CUT_HOLD_HOURS" in w for w in result))
        self.assertTrue(any("[1, 168]" in w for w in result))

    def test_cvar_hold_in_range_no_warning(self):
        result = validate_safety_switches({"PAIRWISE_CVAR_CUT_HOLD_HOURS": "48"})
        self.assertFalse(any("PAIRWISE_CVAR_CUT_HOLD_HOURS" in w for w in result))


# ---------------------------------------------------------------------------
# TestValidateStateAlphasCoverage
# ---------------------------------------------------------------------------

class TestValidateStateAlphasCoverage(unittest.TestCase):

    def _full_candidate(self):
        return {
            "selected_candidate": {
                "pair_configs": {},
                "pair_convex_blends": {
                    "BNBUSDT": {
                        "mode": "state_alphas",
                        "state_alphas": {
                            "equity_aligned:bull_broad": 0.2,
                            "equity_aligned:bull_narrow": 0.2,
                            "equity_aligned:bear_broad": 0.2,
                            "equity_aligned:bear_narrow": 0.2,
                            "equity_mixed:bull_broad": 0.2,
                            "equity_mixed:bull_narrow": 0.2,
                            "equity_mixed:bear_broad": 0.2,
                            "equity_mixed:bear_narrow": 0.2,
                            "equity_unknown": 0.2,
                        },
                    }
                },
            }
        }

    def test_all_routes_covered_returns_empty(self):
        candidate = self._full_candidate()
        observed = {"equity_mixed:bull_broad", "equity_aligned:bear_narrow", "equity_unknown"}
        result = validate_state_alphas_coverage(candidate, observed)
        self.assertEqual(result, [])

    def test_missing_route_returns_warning(self):
        candidate = _make_candidate(mode="state_alphas", state_alphas={"equity_mixed:bull_broad": 0.2})
        observed = {"equity_mixed:bull_broad", "equity_mixed:bull_narrow"}
        result = validate_state_alphas_coverage(candidate, observed)
        self.assertTrue(any("bull_narrow" in w for w in result))
        self.assertTrue(any("BNBUSDT" in w for w in result))

    def test_no_state_alphas_mode_returns_empty(self):
        candidate = _make_candidate(mode="always")
        observed = {"equity_mixed:bull_narrow"}
        result = validate_state_alphas_coverage(candidate, observed)
        self.assertEqual(result, [])

    def test_empty_observed_returns_empty(self):
        candidate = _make_candidate(mode="state_alphas", state_alphas={"equity_mixed:bull_broad": 0.2})
        result = validate_state_alphas_coverage(candidate, [])
        self.assertEqual(result, [])

    def test_no_pair_convex_blends_returns_empty(self):
        candidate = {"selected_candidate": {"pair_configs": {}, "pair_convex_blends": {}}}
        result = validate_state_alphas_coverage(candidate, ["equity_mixed:bull_narrow"])
        self.assertEqual(result, [])

    def test_empty_candidate_returns_empty(self):
        result = validate_state_alphas_coverage({}, ["equity_mixed:bull_narrow"])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
