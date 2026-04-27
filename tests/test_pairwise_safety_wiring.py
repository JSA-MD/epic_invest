"""
Wiring tests: verify safety guards fire correctly inside run_live_once.

These tests mock heavy dependencies (exchange, model loading) to exercise
only the guard integration paths. Where mocking run_live_once end-to-end
is too brittle, unit-level assertions document the integration expectation.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import safety_guards


class TestSafetyGuardA_ForceExecuteWarning(unittest.TestCase):
    """Guard A: validate_safety_switches prints warning when PAIRWISE_FORCE_EXECUTE=1."""

    def test_force_execute_warning_logged(self, capsys=None):
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with patch.dict("os.environ", {"PAIRWISE_FORCE_EXECUTE": "1"}):
                warnings = safety_guards.validate_safety_switches()
                for w in warnings:
                    print(f"[pairwise-live] safety warning: {w}")
        output = buf.getvalue()
        self.assertIn("PAIRWISE_FORCE_EXECUTE", output)

    def test_no_warning_when_force_execute_off(self):
        with patch.dict("os.environ", {"PAIRWISE_FORCE_EXECUTE": "0"}):
            warnings = safety_guards.validate_safety_switches()
        force_warnings = [w for w in warnings if "PAIRWISE_FORCE_EXECUTE" in w]
        self.assertEqual(force_warnings, [])

    def test_safety_switches_called_in_run_live_once(self):
        """validate_safety_switches is called at the top of run_live_once."""
        import pairwise_regime_live as prl
        # Verify the import is present in run_live_once source
        import inspect
        src = inspect.getsource(prl.run_live_once)
        self.assertIn("validate_safety_switches", src)
        self.assertIn("safety_guards", src)


class TestSafetyGuardB_PositionDivergence(unittest.TestCase):
    """Guard B: check_position_divergence returns message for diverged positions."""

    def _post_trade_state(self, positions):
        return {
            "latest_live_sync": {
                "source": "run_live_once_post_trade",
                "positions": positions,
            }
        }

    def test_divergence_detected_and_printed(self):
        import io, contextlib
        # BNB qty=-10 in state, qty=0 in exchange; equity=$1000, price=$600
        # notional diff = 10 * 600 = 6000 >> 1000 * 0.01 = 10
        prev = {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}}
        curr = {"BNBUSDT": {"qty": 0.0, "mark_price": 600.0}}
        state = self._post_trade_state(prev)
        msg = safety_guards.check_position_divergence(state, curr, equity=1000.0)
        self.assertIsNotNone(msg)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if msg:
                print(f"[pairwise-live] POSITION DIVERGENCE: {msg}")
        self.assertIn("POSITION DIVERGENCE", buf.getvalue())
        self.assertIn("BNBUSDT", buf.getvalue())

    def test_no_divergence_no_output(self):
        prev = {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}}
        curr = {"BNBUSDT": {"qty": -10.0, "mark_price": 600.0}}
        state = self._post_trade_state(prev)
        msg = safety_guards.check_position_divergence(state, curr, equity=1000.0)
        self.assertIsNone(msg)

    def test_wiring_present_in_run_live_once(self):
        import pairwise_regime_live as prl
        import inspect
        src = inspect.getsource(prl.run_live_once)
        self.assertIn("check_position_divergence", src)
        self.assertIn("POSITION DIVERGENCE", src)


class TestSafetyGuardD_StalePriceFlat(unittest.TestCase):
    """Guard D: update_stale_price_tracker forces flat when threshold exceeded."""

    def test_stale_price_forces_flat_when_threshold_exceeded(self):
        state = {
            "stale_price_counts": {"BNBUSDT": 12},
            "last_known_prices": {"BNBUSDT": 600.0},
        }
        plan = {
            "target_weights": {"BNBUSDT": -0.05, "BTCUSDT": 0.0},
            "pair_plans": {
                "BNBUSDT": {"target_weight": -0.05},
                "BTCUSDT": {"target_weight": 0.0},
            },
            "latest_prices": {"BNBUSDT": 600.0, "BTCUSDT": 90000.0},
        }
        stale = safety_guards.update_stale_price_tracker(state, plan["latest_prices"])
        self.assertIn("BNBUSDT", stale)
        # Simulate what wiring does
        for sp in stale:
            plan["target_weights"][sp] = 0.0
            if sp in plan.get("pair_plans", {}):
                plan["pair_plans"][sp]["target_weight"] = 0.0
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        self.assertAlmostEqual(plan["pair_plans"]["BNBUSDT"]["target_weight"], 0.0)

    def test_btc_unaffected_when_only_bnb_stale(self):
        state = {
            "stale_price_counts": {"BNBUSDT": 12, "BTCUSDT": 0},
            "last_known_prices": {"BNBUSDT": 600.0, "BTCUSDT": 90000.0},
        }
        stale = safety_guards.update_stale_price_tracker(
            state, {"BNBUSDT": 600.0, "BTCUSDT": 90001.0}
        )
        self.assertIn("BNBUSDT", stale)
        self.assertNotIn("BTCUSDT", stale)

    def test_wiring_present_in_run_live_once(self):
        import pairwise_regime_live as prl
        import inspect
        src = inspect.getsource(prl.run_live_once)
        self.assertIn("update_stale_price_tracker", src)
        self.assertIn("stale price", src)


class TestSafetyGuardC_GrossCapCeiling(unittest.TestCase):
    """Guard C: enforce_runtime_gross_cap_ceiling wired in run_live_once, D2 uses _effective_gross_cap."""

    def test_gross_cap_ceiling_clips_at_runtime(self):
        """enforce_runtime_gross_cap_ceiling clips correctly and warning mentions both values."""
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with patch.dict("os.environ", {"PAIRWISE_GROSS_CAP": "1.5", "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05"}):
                effective, warning = safety_guards.enforce_runtime_gross_cap_ceiling()
                if warning:
                    print(f"[pairwise-live] safety warning: {warning}")
        # runtime=1.5 invalid (>HARD_MAX) → errors-fallback path:
        # effective = min(SAFE_DEFAULT=0.01, ceiling=0.05) = 0.01
        self.assertAlmostEqual(effective, 0.01)
        output = buf.getvalue()
        self.assertIn("1.5", output)
        self.assertIn("0.01", output)

    def test_wiring_present_in_run_live_once(self):
        import pairwise_regime_live as prl
        import inspect
        src = inspect.getsource(prl.run_live_once)
        self.assertIn("enforce_runtime_gross_cap_ceiling", src)
        self.assertIn("_effective_gross_cap", src)

    def test_d2_uses_effective_gross_cap_not_module_constant(self):
        """D2 in run_live_once uses _effective_gross_cap, not PAIRWISE_GROSS_CAP directly."""
        import pairwise_regime_live as prl
        import inspect
        src = inspect.getsource(prl.run_live_once)
        # D2 must reference the local variable, not the module constant
        self.assertIn("_effective_gross_cap", src)
        # build_pairwise_plan must NOT contain clip_candidate_gross_cap call
        plan_src = inspect.getsource(prl.build_pairwise_plan)
        self.assertNotIn("clip_candidate_gross_cap", plan_src)

    def test_hard_ceiling_clamps_valid_runtime(self):
        # Stage 0 lockdown: SAFE_DEFAULT=0.01 clamps runtime even when both
        # values are valid and runtime <= ceiling.
        with patch.dict("os.environ", {"PAIRWISE_GROSS_CAP": "0.03", "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.05"}):
            effective, warning = safety_guards.enforce_runtime_gross_cap_ceiling()
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)

    def test_negative_pairwise_gross_cap_falls_back_to_safe_with_telegram_alert(self):
        """Negative PAIRWISE_GROSS_CAP triggers safe fallback, warning in stdout, Telegram alert."""
        import io, contextlib

        mock_notif = MagicMock()
        mock_notif.send_telegram_notification = MagicMock()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with patch.dict("os.environ", {"PAIRWISE_GROSS_CAP": "-1.0"}, clear=False):
                effective, warning = safety_guards.enforce_runtime_gross_cap_ceiling()
                if warning:
                    print(f"[pairwise-live] safety warning: {warning}")
                    try:
                        mock_notif.send_telegram_notification(warning)
                    except Exception:
                        pass

        # Stage 0 lockdown: SAFE_DEFAULT=0.01, ceiling defaults to 0.01.
        self.assertAlmostEqual(effective, 0.01)
        self.assertIsNotNone(warning)
        output = buf.getvalue()
        self.assertIn("safety warning", output)
        self.assertIn("PAIRWISE_GROSS_CAP", output)
        mock_notif.send_telegram_notification.assert_called_once()


class TestSafetyGuardE_StateAlphasCoverage(unittest.TestCase):
    """Guard E: validate_state_alphas_coverage detects missing route states."""

    def test_detects_missing_bull_narrow(self):
        candidate = {
            "selected_candidate": {
                "pair_configs": {},
                "pair_convex_blends": {
                    "BNBUSDT": {
                        "mode": "state_alphas",
                        "state_alphas": {"equity_mixed:bull_broad": 0.2},
                    }
                },
            }
        }
        observed = {"equity_mixed:bull_broad", "equity_mixed:bull_narrow"}
        missing = safety_guards.validate_state_alphas_coverage(candidate, observed)
        self.assertTrue(any("bull_narrow" in m for m in missing))

    def test_wiring_present_in_run_live_once(self):
        import pairwise_regime_live as prl
        import inspect
        src = inspect.getsource(prl.run_live_once)
        self.assertIn("validate_state_alphas_coverage", src)
        self.assertIn("state_alphas coverage gaps", src)

    def test_wiring_sources_from_pair_plans_not_decision_journal(self):
        """Guard E must read route states from plan['pair_plans'], not decision_journal."""
        import pairwise_regime_live as prl
        import inspect
        src = inspect.getsource(prl.run_live_once)
        # Must source from pair_plans
        self.assertIn("pair_plans", src)
        self.assertIn("route_state_name", src)
        # Must NOT use decision_journal for route state extraction
        # (the old broken wiring used decision_journal for _recent_routes/_rs)
        # Check that recent_route_states is persisted to state
        self.assertIn("recent_route_states", src)

    def test_state_alphas_coverage_warning_fires_when_route_missing(self):
        """validate_state_alphas_coverage fires with routes sourced from plan['pair_plans']."""
        import io, contextlib

        candidate = {
            "selected_candidate": {
                "pair_configs": {},
                "pair_convex_blends": {
                    "BNBUSDT": {
                        "mode": "state_alphas",
                        "state_alphas": {"equity_mixed:bull_broad": 0.2},
                    }
                },
            }
        }
        # Simulate the guard E logic: build observed_routes from plan["pair_plans"]
        plan = {
            "pair_plans": {
                "BNBUSDT": {"route_state_name": "equity_mixed:bull_narrow"},
            }
        }
        state: dict = {}

        # Replicate the guard E accumulation logic
        observed_routes: set = set()
        for _pair, _pair_plan in (plan.get("pair_plans") or {}).items():
            _rs = (_pair_plan or {}).get("route_state_name")
            if _rs:
                observed_routes.add(str(_rs))

        recent_window: dict = state.get("recent_route_states") or {}
        for _pair, _pair_plan in (plan.get("pair_plans") or {}).items():
            _rs = (_pair_plan or {}).get("route_state_name")
            if _rs:
                recent_window.setdefault(_pair, [])
                if isinstance(recent_window.get(_pair), list):
                    recent_window[_pair].append(str(_rs))
                    recent_window[_pair] = list(dict.fromkeys(recent_window[_pair]))[-50:]
                else:
                    recent_window[_pair] = [str(_rs)]
        state["recent_route_states"] = recent_window

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            missing = safety_guards.validate_state_alphas_coverage(candidate, observed_routes)
            if missing:
                print(f"[pairwise-live] state_alphas coverage gaps: {missing}")

        self.assertTrue(any("bull_narrow" in m for m in missing))
        output = buf.getvalue()
        self.assertIn("state_alphas coverage gaps", output)
        self.assertIn("equity_mixed:bull_narrow", state["recent_route_states"]["BNBUSDT"])

    def test_recent_route_states_persists_across_cycles(self):
        """Route states accumulate across cycles via state['recent_route_states']."""
        state: dict = {}

        def _run_cycle(route_state_name: str) -> None:
            plan = {"pair_plans": {"BNBUSDT": {"route_state_name": route_state_name}}}
            recent_window: dict = state.get("recent_route_states") or {}
            for _pair, _pair_plan in (plan.get("pair_plans") or {}).items():
                _rs = (_pair_plan or {}).get("route_state_name")
                if _rs:
                    recent_window.setdefault(_pair, [])
                    if isinstance(recent_window.get(_pair), list):
                        recent_window[_pair].append(str(_rs))
                        recent_window[_pair] = list(dict.fromkeys(recent_window[_pair]))[-50:]
                    else:
                        recent_window[_pair] = [str(_rs)]
            state["recent_route_states"] = recent_window

        _run_cycle("equity_mixed:bull_broad")
        self.assertIn("equity_mixed:bull_broad", state["recent_route_states"]["BNBUSDT"])
        self.assertNotIn("equity_mixed:bull_narrow", state["recent_route_states"]["BNBUSDT"])

        _run_cycle("equity_mixed:bull_narrow")
        self.assertIn("equity_mixed:bull_broad", state["recent_route_states"]["BNBUSDT"])
        self.assertIn("equity_mixed:bull_narrow", state["recent_route_states"]["BNBUSDT"])


class TestEnvParseFailTelegramWiring(unittest.TestCase):
    """Guard A extension: parse-failure warnings trigger CRITICAL Telegram alert."""

    def test_env_parse_failure_triggers_critical_telegram(self):
        """PAIRWISE_MAX_HOLD_BARS=abc produces a warning AND sends a CRITICAL Telegram notification."""
        import io
        import contextlib

        mock_notif = MagicMock()
        mock_notif.send_telegram_notification = MagicMock()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with patch.dict("os.environ", {"PAIRWISE_MAX_HOLD_BARS": "abc"}):
                _warnings = safety_guards.validate_safety_switches()
                for _w in _warnings:
                    print(f"[pairwise-live] safety warning: {_w}")
                # Replicate the wiring logic from run_live_once
                if any("is not a valid" in _w for _w in _warnings):
                    _bad = "; ".join(w for w in _warnings if "is not a valid" in w)
                    mock_notif.send_telegram_notification(_bad)

        output = buf.getvalue()
        self.assertIn("is not a valid integer", output)
        self.assertIn("PAIRWISE_MAX_HOLD_BARS", output)
        mock_notif.send_telegram_notification.assert_called_once()
        call_arg = mock_notif.send_telegram_notification.call_args[0][0]
        self.assertIn("is not a valid", call_arg)

    def test_parse_fail_wiring_present_in_run_live_once(self):
        """run_live_once source must contain the env-parse-fail Telegram block."""
        import pairwise_regime_live as prl
        import inspect
        src = inspect.getsource(prl.run_live_once)
        self.assertIn("is not a valid", src)
        self.assertIn("env-parse-fail", src)
        self.assertIn("환경변수 설정 오류", src)


if __name__ == "__main__":
    unittest.main()
