import sys
import unittest
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_regime_live as prl


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _run_r3(
    plan: dict,
    state: dict,
    positions: dict,
    now: datetime | None = None,
    *,
    cvar_enabled: bool = True,
    positions_fetched: bool = True,
    should_cut: bool = False,
) -> None:
    """Run the R3 block with frozen time, mocked cvar thresholds, and no Telegram side-effects.

    positions_fetched=False means _exch_active is always False (no ground truth).
    Plan-tw enforcement still runs — cut intent is "be flat" regardless of fetch status.
    The retry-close journal only fires when _exch_active is True (requires positions_fetched).
    The new-cut trigger (should_apply_cvar_cut) fires regardless of positions_fetched.
    """
    _now = now or datetime.now(UTC)
    notif = MagicMock()

    with patch.object(prl, "utc_now", return_value=_now), \
         patch.object(prl, "load_notification_bridge", return_value=notif), \
         patch.object(prl, "PAIRWISE_CVAR_CUT_ENABLED", cvar_enabled), \
         patch.object(prl, "load_tail_risk_thresholds", return_value={}), \
         patch.object(prl, "should_apply_cvar_cut", return_value=should_cut):

        if not prl.PAIRWISE_CVAR_CUT_ENABLED:
            return

        cvar_cut_until = dict(state.get("cvar_cut_until_ts") or {})
        for _pair in list(prl.PAIRS):
            _cut_until_raw = cvar_cut_until.get(_pair)
            _cut_until_dt = prl.parse_utc_datetime(_cut_until_raw)
            if _cut_until_dt is not None and _now < _cut_until_dt:
                _tw = float(plan["target_weights"].get(_pair, 0.0))
                # _exch_active requires ground truth — False when fetch failed
                _exch_active = positions_fetched and prl._exchange_position_is_open(positions, _pair)
                if abs(_tw) > prl.TARGET_WEIGHT_EPS or _exch_active:
                    plan["target_weights"][_pair] = 0.0
                    if _pair in plan.get("pair_plans", {}):
                        plan["pair_plans"][_pair]["target_weight"] = 0.0
                    if _exch_active and abs(_tw) <= prl.TARGET_WEIGHT_EPS:
                        print(f"[pairwise-live] R3 cut still active for {_pair}; exchange position not yet flat — forcing tw=0 to retry close")
                        state.setdefault("decision_journal", []).append(
                            {
                                "at": _now.isoformat(),
                                "pair": _pair,
                                "override_reason": "cvar_cut_retry_close",
                                "cvar_cut_until": _cut_until_dt.isoformat() if _cut_until_dt else None,
                                "exchange_position_active": _exch_active,
                                "target_weight_forced": 0.0,
                            }
                        )
                continue
            else:
                if _cut_until_dt is not None and _now >= _cut_until_dt:
                    cvar_cut_until[_pair] = None
            if prl.should_apply_cvar_cut(_pair, {}):
                from datetime import timedelta as _td
                _resume_dt = _now + _td(hours=prl.PAIRWISE_CVAR_CUT_HOLD_HOURS)
                cvar_cut_until[_pair] = _resume_dt.isoformat()
                plan["target_weights"][_pair] = 0.0
                if _pair in plan.get("pair_plans", {}):
                    plan["pair_plans"][_pair]["target_weight"] = 0.0
                state.setdefault("decision_journal", []).append(
                    {
                        "at": _now.isoformat(),
                        "pair": _pair,
                        "override_reason": "cvar_cut",
                        "cvar_threshold": None,
                        "cvar_cut_until": _resume_dt.isoformat(),
                        "target_weight_forced": 0.0,
                    }
                )
        state["cvar_cut_until_ts"] = cvar_cut_until


class TestR3ExchangeAware(unittest.TestCase):
    """Defect 4: R3 active-cut branch must enforce flat even when plan tw is already 0."""

    def test_active_cut_plan_tw_nonzero_forces_flat(self):
        """Standard case: plan tw is non-zero, cut active — forced to 0."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=4))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.05}, "pair_plans": {}}
        positions = {}

        _run_r3(plan, state, positions, now=frozen_now)

        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)

    def test_active_cut_plan_tw_zero_exchange_open_forces_flat_and_warns(self):
        """Plan tw already 0 but exchange still open — must keep tw=0 and print warning."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=4))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {"BNBUSDT": {"target_weight": 0.0}},
        }
        positions = {"BNBUSDT": {"qty": -5.0}}

        import io
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            _run_r3(plan, state, positions, now=frozen_now)
            output = mock_out.getvalue()

        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        self.assertAlmostEqual(plan["pair_plans"]["BNBUSDT"]["target_weight"], 0.0)
        self.assertIn("exchange position not yet flat", output)
        self.assertIn("BNBUSDT", output)

    def test_active_cut_plan_tw_zero_exchange_flat_no_warn(self):
        """Plan tw=0 AND exchange flat — no warning, no change."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=4))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0}, "pair_plans": {}}
        positions = {}

        import io
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            _run_r3(plan, state, positions, now=frozen_now)
            output = mock_out.getvalue()

        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        self.assertNotIn("exchange position not yet flat", output)

    def test_active_cut_pair_plans_target_weight_also_zeroed(self):
        """pair_plans[pair]['target_weight'] is zeroed when exchange still open."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=4))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {"BNBUSDT": {"target_weight": 0.0}},
        }
        positions = {"BNBUSDT": {"qty": -5.0}}

        _run_r3(plan, state, positions, now=frozen_now)

        self.assertAlmostEqual(plan["pair_plans"]["BNBUSDT"]["target_weight"], 0.0)

    def test_expired_cut_does_not_enforce(self):
        """Expired cut should not enforce flat (original behavior)."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now - timedelta(hours=1))  # already expired
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.05}, "pair_plans": {}}
        positions = {"BNBUSDT": {"qty": -5.0}}

        _run_r3(plan, state, positions, now=frozen_now)

        # Expired cut should NOT force flat (no new cut was triggered in our mock)
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.05)

    def test_btc_unaffected_by_bnb_cut(self):
        """R3 cut on BNBUSDT must not affect BTCUSDT."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=4))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {"target_weights": {"BTCUSDT": -0.1, "BNBUSDT": 0.05}, "pair_plans": {}}
        positions = {}

        _run_r3(plan, state, positions, now=frozen_now)

        self.assertAlmostEqual(plan["target_weights"]["BTCUSDT"], -0.1)
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)

    def test_active_cut_retry_close_appends_decision_journal(self):
        """R3 active-cut retry-close (tw already 0, exchange still open) appends journal entry."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=4))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {"BNBUSDT": {"target_weight": 0.0}},
        }
        positions = {"BNBUSDT": {"qty": -5.0}}

        _run_r3(plan, state, positions, now=frozen_now)

        journal = state.get("decision_journal", [])
        retry_entries = [e for e in journal if e.get("override_reason") == "cvar_cut_retry_close"]
        self.assertEqual(len(retry_entries), 1)
        entry = retry_entries[0]
        self.assertEqual(entry["pair"], "BNBUSDT")
        self.assertAlmostEqual(entry["target_weight_forced"], 0.0)
        self.assertTrue(entry["exchange_position_active"])
        self.assertIn("cvar_cut_until", entry)

    def test_active_cut_retry_close_journal_entry_shape(self):
        """decision_journal entry for retry-close has all required keys."""
        frozen_now = datetime.now(UTC)
        cut_until_dt = frozen_now + timedelta(hours=2)
        cut_until = _iso(cut_until_dt)
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0}, "pair_plans": {}}
        positions = {"BNBUSDT": {"qty": -1.0}}

        _run_r3(plan, state, positions, now=frozen_now)

        journal = state.get("decision_journal", [])
        self.assertTrue(len(journal) >= 1)
        entry = journal[-1]
        for key in ("at", "pair", "override_reason", "cvar_cut_until", "exchange_position_active", "target_weight_forced"):
            self.assertIn(key, entry, f"Missing key: {key}")
        self.assertEqual(entry["override_reason"], "cvar_cut_retry_close")


class TestR3PositionsUnfetched(unittest.TestCase):
    """R3 active-cut plan-tw enforcement always runs; _exch_active is False when fetch failed."""

    def test_r3_active_cut_no_op_when_tw_already_flat_and_positions_unfetched(self):
        """When positions_fetched=False, tw=0, and no ground truth: plan stays 0, no journal entry, cut state unchanged."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=12))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {},
        }
        # positions has an open entry but positions_fetched=False means _exch_active=False,
        # so the guard (abs(tw)>eps or _exch_active) is False — no-op since plan already flat.
        positions = {"BNBUSDT": {"qty": -5.0}}

        _run_r3(plan, state, positions, now=frozen_now, positions_fetched=False)

        # No retry-close journal entry (no ground truth)
        journal = state.get("decision_journal", [])
        retry_entries = [e for e in journal if e.get("override_reason") == "cvar_cut_retry_close"]
        self.assertEqual(len(retry_entries), 0)
        # Plan tw stays 0 (already at safe state)
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        # Cut state persists untouched
        self.assertEqual(state["cvar_cut_until_ts"]["BNBUSDT"], cut_until)

    def test_r3_active_cut_forces_tw_when_positions_unfetched(self):
        """When positions_fetched=False and tw is non-zero, R3 DOES force tw=0 — cut intent enforced regardless of fetch status."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=12))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -0.1},
            "pair_plans": {"BNBUSDT": {"target_weight": -0.1}},
        }
        positions = {}

        _run_r3(plan, state, positions, now=frozen_now, positions_fetched=False)

        # tw forced to 0 — cut intent always enforced
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        self.assertAlmostEqual(plan["pair_plans"]["BNBUSDT"]["target_weight"], 0.0)
        # No retry-close journal entry (no ground truth to confirm exchange open)
        journal = state.get("decision_journal", [])
        retry_entries = [e for e in journal if e.get("override_reason") == "cvar_cut_retry_close"]
        self.assertEqual(len(retry_entries), 0)

    def test_r3_new_cut_triggers_even_when_positions_unfetched(self):
        """New-cut trigger fires even when positions_fetched=False (trigger is return-based, no position dependency)."""
        frozen_now = datetime.now(UTC)
        # No active cut
        state = {"cvar_cut_until_ts": {}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.05},
            "pair_plans": {},
        }
        positions = {}

        # should_cut=True so the trigger fires for every pair evaluated
        _run_r3(plan, state, positions, now=frozen_now, positions_fetched=False, should_cut=True)

        # Cut must have fired for BNBUSDT
        self.assertIsNotNone(state["cvar_cut_until_ts"].get("BNBUSDT"))
        # Plan tw forced to 0
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        # decision_journal has a cvar_cut entry
        journal = state.get("decision_journal", [])
        cut_entries = [e for e in journal if e.get("override_reason") == "cvar_cut"]
        bnb_entries = [e for e in cut_entries if e.get("pair") == "BNBUSDT"]
        self.assertEqual(len(bnb_entries), 1)

    def test_r3_active_cut_resumes_after_fetch_recovers(self):
        """Multi-cycle: fetch fail is no-op when tw already flat; fetch recovery triggers retry-close journal."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=4))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}

        # Cycle 1: positions_fetched=False, exchange has open position (invisible — _exch_active=False).
        # tw=0 and _exch_active=False → guard is False → no-op.
        plan1 = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {"BNBUSDT": {"target_weight": 0.0}},
        }
        positions_real = {"BNBUSDT": {"qty": -5.0}}

        _run_r3(plan1, state, positions_real, now=frozen_now, positions_fetched=False)

        # Cut state unchanged, no journal entry (plan already at safe state)
        self.assertEqual(state["cvar_cut_until_ts"]["BNBUSDT"], cut_until)
        journal = state.get("decision_journal", [])
        self.assertEqual(len(journal), 0)
        # Plan tw stays 0
        self.assertAlmostEqual(plan1["target_weights"]["BNBUSDT"], 0.0)

        # Cycle 2: 5 minutes later, positions_fetched=True, exchange still open
        now2 = frozen_now + timedelta(minutes=5)
        plan2 = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {"BNBUSDT": {"target_weight": 0.0}},
        }

        _run_r3(plan2, state, positions_real, now=now2, positions_fetched=True)

        # Retry-close fires — journal entry written
        journal2 = state.get("decision_journal", [])
        retry_entries = [e for e in journal2 if e.get("override_reason") == "cvar_cut_retry_close"]
        self.assertEqual(len(retry_entries), 1)
        self.assertEqual(retry_entries[0]["pair"], "BNBUSDT")
        self.assertTrue(retry_entries[0]["exchange_position_active"])

    def test_r3_cut_expiry_cleanup_runs_without_positions(self):
        """Cut expiry cleanup runs even when positions_fetched=False."""
        frozen_now = datetime.now(UTC)
        # Cut already expired
        cut_until = _iso(frozen_now - timedelta(hours=1))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.05},
            "pair_plans": {},
        }
        positions = {}

        _run_r3(plan, state, positions, now=frozen_now, positions_fetched=False)

        # Expired cut cleared to None
        self.assertIsNone(state["cvar_cut_until_ts"]["BNBUSDT"])

    def test_r3_active_cut_no_skip_warning_when_positions_unfetched(self):
        """Regression guard: old 'R3 active-cut enforcement skipped' warning must NOT appear."""
        import io
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=4))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {},
        }
        positions = {}

        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            _run_r3(plan, state, positions, now=frozen_now, positions_fetched=False)
            output = mock_out.getvalue()

        self.assertNotIn("R3 active-cut enforcement skipped", output)


if __name__ == "__main__":
    unittest.main()
