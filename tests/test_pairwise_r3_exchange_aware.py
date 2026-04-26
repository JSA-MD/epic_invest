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
) -> None:
    """Run the R3 block with frozen time, mocked cvar thresholds, and no Telegram side-effects.

    positions_fetched=False mirrors the fail-safe path: _exch_active is forced False
    so unknown-state cycles don't write spurious retry-close journal entries.
    """
    _now = now or datetime.now(UTC)
    notif = MagicMock()

    with patch.object(prl, "utc_now", return_value=_now), \
         patch.object(prl, "load_notification_bridge", return_value=notif), \
         patch.object(prl, "PAIRWISE_CVAR_CUT_ENABLED", cvar_enabled), \
         patch.object(prl, "load_tail_risk_thresholds", return_value={}), \
         patch.object(prl, "should_apply_cvar_cut", return_value=False):

        if not prl.PAIRWISE_CVAR_CUT_ENABLED:
            return

        cvar_cut_until = dict(state.get("cvar_cut_until_ts") or {})
        for _pair in list(prl.PAIRS):
            _cut_until_raw = cvar_cut_until.get(_pair)
            _cut_until_dt = prl.parse_utc_datetime(_cut_until_raw)
            if _cut_until_dt is not None and _now < _cut_until_dt:
                _tw = float(plan["target_weights"].get(_pair, 0.0))
                _exch_active = (
                    positions_fetched and prl._exchange_position_is_open(positions, _pair)
                )
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
    """Fail-safe: R3 active-cut must fall back to plan-tw-only when positions_fetched=False."""

    def test_r3_active_cut_falls_back_to_plan_when_positions_unfetched(self):
        """When positions_fetched=False and tw=0, no retry-close journal entry is written."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=12))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {},
        }
        # Even though positions has an open entry, positions_fetched=False means
        # _exch_active is treated as False — no retry-close fires.
        positions = {"BNBUSDT": {"qty": -5.0}}

        _run_r3(plan, state, positions, now=frozen_now, positions_fetched=False)

        journal = state.get("decision_journal", [])
        retry_entries = [e for e in journal if e.get("override_reason") == "cvar_cut_retry_close"]
        self.assertEqual(len(retry_entries), 0)

    def test_r3_active_cut_still_enforces_when_tw_nonzero_and_positions_unfetched(self):
        """When positions_fetched=False but tw is non-zero, plan tw is still forced to 0."""
        frozen_now = datetime.now(UTC)
        cut_until = _iso(frozen_now + timedelta(hours=12))
        state = {"cvar_cut_until_ts": {"BNBUSDT": cut_until}}
        plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -0.1},
            "pair_plans": {},
        }
        positions = {}

        _run_r3(plan, state, positions, now=frozen_now, positions_fetched=False)

        # tw-based enforcement still fires (abs(-0.1) > TARGET_WEIGHT_EPS)
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        # But no retry-close journal entry (exch_active forced False)
        journal = state.get("decision_journal", [])
        retry_entries = [e for e in journal if e.get("override_reason") == "cvar_cut_retry_close"]
        self.assertEqual(len(retry_entries), 0)


if __name__ == "__main__":
    unittest.main()
