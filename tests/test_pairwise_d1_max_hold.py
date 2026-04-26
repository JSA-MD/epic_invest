import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pairwise_regime_live as prl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _make_plan(btc_tw: float = 0.0, bnb_tw: float = 0.0) -> dict:
    return {
        "target_weights": {"BTCUSDT": btc_tw, "BNBUSDT": bnb_tw},
        "pair_plans": {},
    }


def _make_state(position_open_since: dict | None = None) -> dict:
    return {"position_open_since_ts": position_open_since or {}}


def _run_d1(
    plan: dict,
    state: dict,
    positions: dict,
    now: datetime | None = None,
    *,
    positions_fetched: bool = True,
) -> None:
    """Run the D1 block with frozen time and mocked notification bridge.

    Patching utc_now covers both the _now variable and compute_position_age_seconds,
    which also calls utc_now() internally, so age calculations are deterministic.

    Mirrors production logic: timer is exchange-state-aware. When the exchange is
    flat the timer resets for a fresh signal or clears for fully flat. This prevents
    a closed position's age from suppressing a new genuine signal.

    positions_fetched=False mirrors the fail-safe path: D1 is skipped entirely and
    existing timers are preserved unchanged.
    """
    _now = now or datetime.now(UTC)
    notif = MagicMock()

    with patch.object(prl, "utc_now", return_value=_now), \
         patch.object(prl, "load_notification_bridge", return_value=notif):
        position_open_since: dict = dict(state.get("position_open_since_ts") or {})
        if not positions_fetched:
            print("[pairwise-live] D1 skipped: positions unknown (fetch unavailable)")
            return
        for _pair in list(prl.PAIRS):
            _tw = float(plan["target_weights"].get(_pair, 0.0))
            _prev_since = position_open_since.get(_pair)
            tw_active = abs(_tw) > prl.TARGET_WEIGHT_EPS
            exch_active = prl._exchange_position_is_open(positions, _pair)
            if exch_active:
                # Exchange has a real open position — track continuously from first appearance.
                if _prev_since is None:
                    position_open_since[_pair] = _now.isoformat()
                else:
                    age_secs = prl.compute_position_age_seconds(_prev_since)
                    if age_secs > prl._MAX_HOLD_SECONDS:
                        _msg = (
                            f"[pairwise-live] D1 max_hold override: {_pair} position "
                            f"open {age_secs/3600:.1f}h >= {prl.PAIRWISE_MAX_HOLD_BARS} bars — forcing flat"
                        )
                        print(_msg)
                        _notif = notif
                        try:
                            from telegram_format import AlertLevel as _AL, format_alert as _fa, should_send as _ss
                            if _ss(_AL.HIGH, f"d1-max-hold-{_pair}"):
                                _payload = _fa(
                                    _AL.HIGH,
                                    title="D1 max_hold 자동 청산",
                                    body=f"{_pair} 포지션 {age_secs/3600:.1f}h 경과 ({prl.PAIRWISE_MAX_HOLD_BARS}바 한도) — 강제 플랫",
                                    pair=_pair,
                                )
                                _notif.send_telegram_notification(_payload["text"])
                        except Exception:
                            _notif.send_telegram_notification(_msg)
                        plan["target_weights"][_pair] = 0.0
                        if _pair in plan.get("pair_plans", {}):
                            plan["pair_plans"][_pair]["target_weight"] = 0.0
                        # Timer NOT cleared — nag-until-fixed while exchange is open.
                        state.setdefault("decision_journal", []).append(
                            {
                                "at": _now.isoformat(),
                                "pair": _pair,
                                "override_reason": "max_hold",
                                "age_seconds": age_secs,
                                "max_hold_seconds": prl._MAX_HOLD_SECONDS,
                                "target_weight_forced": 0.0,
                                "exchange_position_active": exch_active,
                            }
                        )
            else:
                # Exchange is flat — reset for fresh signal or clear if fully flat.
                if tw_active:
                    position_open_since[_pair] = _now.isoformat()
                else:
                    position_open_since[_pair] = None
        state["position_open_since_ts"] = position_open_since


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestD1MaxHold(unittest.TestCase):

    # 1. New entry: tw 0 -> active, positions empty — timer set, D1 does not fire
    def test_new_entry_sets_timer_no_fire(self):
        plan = _make_plan(btc_tw=-0.1)
        state = _make_state()
        _run_d1(plan, state, positions={})

        self.assertIsNotNone(state["position_open_since_ts"].get("BTCUSDT"))
        self.assertAlmostEqual(plan["target_weights"]["BTCUSDT"], -0.1)
        self.assertNotIn("decision_journal", state)

    # 2. 24h elapsed, plan-driven (tw active, exchange flat) — under new exchange-aware
    #    logic the exchange is flat so the timer resets to now for the fresh signal;
    #    D1 does NOT fire (no stale age to trigger it on the new position).
    def test_plan_driven_24h_exch_flat_no_fire(self):
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(seconds=prl._MAX_HOLD_SECONDS + 3600))
        plan = _make_plan(btc_tw=-0.1)
        state = _make_state({"BTCUSDT": open_since})

        _run_d1(plan, state, positions={}, now=frozen_now)

        # Exchange is flat -> timer resets to now for the fresh signal, D1 does not fire.
        self.assertAlmostEqual(plan["target_weights"]["BTCUSDT"], -0.1)
        self.assertEqual(state["position_open_since_ts"]["BTCUSDT"], frozen_now.isoformat())
        self.assertNotIn("decision_journal", state)

    # 2b. 24h elapsed AND exchange is open — D1 must fire
    def test_plan_driven_24h_exch_open_fires(self):
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(seconds=prl._MAX_HOLD_SECONDS + 3600))
        plan = _make_plan(btc_tw=-0.1)
        state = _make_state({"BTCUSDT": open_since})
        positions = {"BTCUSDT": {"qty": -10.0}}

        _run_d1(plan, state, positions=positions, now=frozen_now)

        self.assertAlmostEqual(plan["target_weights"]["BTCUSDT"], 0.0)
        # Timer must NOT be cleared at fire time — it persists until exchange confirms flat.
        self.assertEqual(state["position_open_since_ts"]["BTCUSDT"], open_since)
        journal = state.get("decision_journal", [])
        self.assertEqual(len(journal), 1)
        entry = journal[0]
        self.assertEqual(entry["override_reason"], "max_hold")
        self.assertAlmostEqual(entry["target_weight_forced"], 0.0)
        self.assertTrue(entry["exchange_position_active"])

    # 3. Defect-1 regression: tw drops to 0 but exchange still open — timer must NOT reset
    def test_defect1_tw_flicker_exchange_open_timer_preserved(self):
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(hours=1))
        state = _make_state({"BTCUSDT": open_since})
        plan = _make_plan(btc_tw=0.0)  # tw flickered to ~0
        positions = {"BTCUSDT": {"qty": -10.0}}

        _run_d1(plan, state, positions=positions, now=frozen_now)

        # Timer must NOT have been cleared
        self.assertEqual(state["position_open_since_ts"]["BTCUSDT"], open_since)
        self.assertNotIn("decision_journal", state)

    # 4. Defect-2 regression: tw=0 entire time, exchange open — timer set on first sight
    def test_defect2_exchange_only_sets_timer(self):
        frozen_now = datetime.now(UTC)
        plan = _make_plan(btc_tw=0.0)
        state = _make_state()
        positions = {"BTCUSDT": {"qty": -10.0}}

        _run_d1(plan, state, positions=positions, now=frozen_now)

        self.assertIsNotNone(state["position_open_since_ts"].get("BTCUSDT"))
        self.assertNotIn("decision_journal", state)

    def test_defect2_exchange_only_fires_after_24h(self):
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(seconds=prl._MAX_HOLD_SECONDS + 3600))
        state = _make_state({"BTCUSDT": open_since})
        plan = _make_plan(btc_tw=0.0)
        positions = {"BTCUSDT": {"qty": -10.0}}

        _run_d1(plan, state, positions=positions, now=frozen_now)

        # Timer must NOT be cleared at fire time — persists until exchange confirms flat.
        self.assertEqual(state["position_open_since_ts"]["BTCUSDT"], open_since)
        journal = state.get("decision_journal", [])
        self.assertEqual(len(journal), 1)
        entry = journal[0]
        self.assertEqual(entry["override_reason"], "max_hold")
        self.assertTrue(entry["exchange_position_active"])

    # 5. Genuine flat: tw=0 AND exchange flat — timer cleared
    def test_genuine_flat_clears_timer(self):
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(hours=1))
        state = _make_state({"BTCUSDT": open_since})
        plan = _make_plan(btc_tw=0.0)

        _run_d1(plan, state, positions={}, now=frozen_now)

        self.assertIsNone(state["position_open_since_ts"]["BTCUSDT"])

    def test_genuine_flat_zero_qty_clears_timer(self):
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(hours=1))
        state = _make_state({"BTCUSDT": open_since})
        plan = _make_plan(btc_tw=0.0)
        positions = {"BTCUSDT": {"qty": 0.0}}

        _run_d1(plan, state, positions=positions, now=frozen_now)

        self.assertIsNone(state["position_open_since_ts"]["BTCUSDT"])

    # 6. Boundary: exactly MAX_HOLD_SECONDS does NOT fire (> is strict); +1s fires
    def test_boundary_exactly_max_hold_does_not_fire(self):
        # Both _now and open_since are derived from the same frozen_now, so
        # compute_position_age_seconds returns exactly _MAX_HOLD_SECONDS — no drift.
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(seconds=prl._MAX_HOLD_SECONDS))
        state = _make_state({"BTCUSDT": open_since})
        plan = _make_plan(btc_tw=-0.1)

        _run_d1(plan, state, positions={}, now=frozen_now)

        self.assertAlmostEqual(plan["target_weights"]["BTCUSDT"], -0.1)
        self.assertNotIn("decision_journal", state)

    def test_boundary_one_second_over_exch_flat_no_fire(self):
        # Exchange flat: timer resets to now for fresh signal regardless of old age.
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(seconds=prl._MAX_HOLD_SECONDS + 1))
        state = _make_state({"BTCUSDT": open_since})
        plan = _make_plan(btc_tw=-0.1)

        _run_d1(plan, state, positions={}, now=frozen_now)

        self.assertAlmostEqual(plan["target_weights"]["BTCUSDT"], -0.1)
        self.assertEqual(state["position_open_since_ts"]["BTCUSDT"], frozen_now.isoformat())
        self.assertNotIn("decision_journal", state)

    def test_boundary_one_second_over_exch_open_fires(self):
        # Exchange open: age > MAX_HOLD → D1 fires.
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(seconds=prl._MAX_HOLD_SECONDS + 1))
        state = _make_state({"BTCUSDT": open_since})
        plan = _make_plan(btc_tw=-0.1)
        positions = {"BTCUSDT": {"qty": -10.0}}

        _run_d1(plan, state, positions=positions, now=frozen_now)

        self.assertAlmostEqual(plan["target_weights"]["BTCUSDT"], 0.0)
        journal = state.get("decision_journal", [])
        self.assertEqual(len(journal), 1)

    # 7. Multi-pair independence: BTC exch open 25h → D1 fires; BNB exch flat with fresh signal → timer reset
    def test_multi_pair_independence(self):
        frozen_now = datetime.now(UTC)
        btc_open_since = _iso(frozen_now - timedelta(seconds=prl._MAX_HOLD_SECONDS + 3600))
        bnb_open_since = _iso(frozen_now - timedelta(hours=1))
        state = _make_state({"BTCUSDT": btc_open_since, "BNBUSDT": bnb_open_since})
        plan = _make_plan(btc_tw=-0.1, bnb_tw=0.08)
        # BTC exchange position is open; BNB exchange is flat
        positions = {"BTCUSDT": {"qty": -10.0}}

        _run_d1(plan, state, positions=positions, now=frozen_now)

        # BTC: D1 fired; timer persists until exchange confirms flat
        self.assertAlmostEqual(plan["target_weights"]["BTCUSDT"], 0.0)
        self.assertEqual(state["position_open_since_ts"]["BTCUSDT"], btc_open_since)
        # BNB: exch flat + fresh tw signal → timer reset to now, tw unchanged
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.08)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], frozen_now.isoformat())
        journal = state.get("decision_journal", [])
        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["pair"], "BTCUSDT")


class TestD1TimerPreservationOnFire(unittest.TestCase):
    """Tests for the dry-run / reconcile-failure fix: timer must not be cleared at D1 fire time."""

    def test_d1_fire_does_not_clear_timer(self):
        """After D1 fires the position_open_since timestamp is unchanged."""
        frozen_now = datetime.now(UTC)
        original_open_since = _iso(frozen_now - timedelta(hours=25))
        state = _make_state({"BNBUSDT": original_open_since})
        plan = _make_plan(bnb_tw=-0.1)
        positions = {"BNBUSDT": {"qty": -10.0}}

        _run_d1(plan, state, positions=positions, now=frozen_now)

        # tw must be zeroed
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        # timer must be UNCHANGED — still the original open timestamp
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], original_open_since)
        # journal entry present
        journal = state.get("decision_journal", [])
        max_hold_entries = [e for e in journal if e.get("override_reason") == "max_hold"]
        self.assertEqual(len(max_hold_entries), 1)
        self.assertEqual(max_hold_entries[0]["pair"], "BNBUSDT")

    def test_d1_re_fires_when_exchange_position_persists(self):
        """D1 fires on consecutive cycles when exchange position is never closed."""
        frozen_now = datetime.now(UTC)
        original_open_since = _iso(frozen_now - timedelta(hours=25))
        state = _make_state({"BNBUSDT": original_open_since})
        plan = _make_plan(bnb_tw=-0.1)
        positions = {"BNBUSDT": {"qty": -10.0}}

        # First cycle — D1 fires
        _run_d1(plan, state, positions=positions, now=frozen_now)
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        self.assertEqual(len(state.get("decision_journal", [])), 1)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], original_open_since)

        # Second cycle 5 minutes later — exchange still open, tw still 0 (carried over)
        frozen_now2 = frozen_now + timedelta(minutes=5)
        # plan tw is already 0 from first cycle; exchange position still open
        _run_d1(plan, state, positions=positions, now=frozen_now2)

        # D1 must fire again (age is now 25h05m, still > 24h, exch_active=True)
        journal = state.get("decision_journal", [])
        max_hold_entries = [e for e in journal if e.get("override_reason") == "max_hold"]
        self.assertEqual(len(max_hold_entries), 2)
        # Timer still unchanged — the ORIGINAL open timestamp
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], original_open_since)

    def test_d1_timer_clears_only_when_both_inactive(self):
        """Timer clears only after exchange confirms close AND tw is 0."""
        frozen_now = datetime.now(UTC)
        original_open_since = _iso(frozen_now - timedelta(hours=25))
        state = _make_state({"BNBUSDT": original_open_since})
        plan = _make_plan(bnb_tw=-0.1)
        positions = {"BNBUSDT": {"qty": -10.0}}

        # Cycle 1: D1 fires, timer stays
        _run_d1(plan, state, positions=positions, now=frozen_now)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], original_open_since)

        # Cycle 2: exchange closed (qty=0), tw=0 (already zeroed by D1)
        positions_closed = {"BNBUSDT": {"qty": 0.0}}
        _run_d1(plan, state, positions=positions_closed, now=frozen_now + timedelta(minutes=1))

        # Now both tw and exchange are flat — else branch fires, timer cleared
        self.assertIsNone(state["position_open_since_ts"]["BNBUSDT"])


class TestD1TimerResetOnClose(unittest.TestCase):
    """Regression tests for the Codex-reported bug: timer survives a successful close
    and suppresses fresh signals (D1 fires when exch is flat but prev_since is old)."""

    def test_close_then_fresh_signal_resets_timer(self):
        """After exchange closes, a new tw signal must reset the timer — not inherit the old age."""
        frozen_now = datetime.now(UTC)
        # Simulate: position was open 25h ago, but exchange has since closed.
        old_open_since = _iso(frozen_now - timedelta(hours=25))
        state = _make_state({"BNBUSDT": old_open_since})
        plan = _make_plan(bnb_tw=-0.05)   # fresh genuine signal
        positions = {}  # exchange is flat — position was successfully closed

        _run_d1(plan, state, positions=positions, now=frozen_now)

        # Timer must be reset to now (not the dead 25h-old timestamp)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], frozen_now.isoformat())
        # Plan tw must be unchanged — D1 must NOT fire on a fresh entry
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], -0.05)
        self.assertNotIn("decision_journal", state)

    def test_close_with_no_signal_clears_timer(self):
        """After exchange closes with no active tw, timer must be cleared to None."""
        frozen_now = datetime.now(UTC)
        old_open_since = _iso(frozen_now - timedelta(hours=25))
        state = _make_state({"BNBUSDT": old_open_since})
        plan = _make_plan(bnb_tw=0.0)   # no signal
        positions = {}  # exchange flat

        _run_d1(plan, state, positions=positions, now=frozen_now)

        self.assertIsNone(state["position_open_since_ts"]["BNBUSDT"])
        self.assertNotIn("decision_journal", state)

    def test_open_to_close_to_open_lifecycle(self):
        """Full 5-cycle lifecycle: open → continue → D1 fire → close → fresh entry."""
        t1 = datetime.now(UTC)

        # Cycle 1: positions empty, tw=-0.1. Timer set to t1.
        state = _make_state()
        plan = _make_plan(bnb_tw=-0.1)
        _run_d1(plan, state, positions={}, now=t1)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], t1.isoformat())
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], -0.1)

        # Cycle 2: 5m later, exchange shows open position. Timer must stay at t1.
        t2 = t1 + timedelta(minutes=5)
        positions_open = {"BNBUSDT": {"qty": -10.0}}
        _run_d1(plan, state, positions=positions_open, now=t2)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], t1.isoformat())
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], -0.1)

        # Cycle 3: 25h after t1, exchange still open → D1 fires.
        t3 = t1 + timedelta(seconds=prl._MAX_HOLD_SECONDS + 60)
        _run_d1(plan, state, positions=positions_open, now=t3)
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], t1.isoformat())
        journal = state.get("decision_journal", [])
        self.assertEqual(len([e for e in journal if e["override_reason"] == "max_hold"]), 1)

        # Cycle 4: 5m after cycle 3, exchange closed and tw=0 (D1 forced it). Timer cleared.
        t4 = t3 + timedelta(minutes=5)
        _run_d1(plan, state, positions={}, now=t4)
        self.assertIsNone(state["position_open_since_ts"]["BNBUSDT"])

        # Cycle 5: 5m later, brand-new signal. Timer set to t5, tw unchanged, no D1 fire.
        t5 = t4 + timedelta(minutes=5)
        plan["target_weights"]["BNBUSDT"] = -0.07
        _run_d1(plan, state, positions={}, now=t5)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], t5.isoformat())
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], -0.07)
        # No additional journal entries since cycle 3
        journal2 = state.get("decision_journal", [])
        self.assertEqual(len([e for e in journal2 if e["override_reason"] == "max_hold"]), 1)


class TestD1PositionsUnfetched(unittest.TestCase):
    """Fail-safe: D1 must preserve timers when positions_fetched=False."""

    def test_d1_preserves_timer_when_positions_unfetched(self):
        """Timer must be unchanged and no D1 fire when positions_fetched=False."""
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(hours=12))
        plan = _make_plan(bnb_tw=-0.05)
        state = _make_state({"BNBUSDT": open_since})

        _run_d1(plan, state, positions={}, now=frozen_now, positions_fetched=False)

        # Timer unchanged
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], open_since)
        # No D1 fire
        self.assertNotIn("decision_journal", state)
        # Plan tw unchanged
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], -0.05)

    def test_d1_does_not_fire_when_positions_unfetched_even_if_age_exceeds(self):
        """Timer must NOT be cleared even when age > 24h and tw=0 if positions_fetched=False."""
        frozen_now = datetime.now(UTC)
        open_since = _iso(frozen_now - timedelta(hours=25))
        plan = _make_plan(bnb_tw=0.0)
        state = _make_state({"BNBUSDT": open_since})

        _run_d1(plan, state, positions={}, now=frozen_now, positions_fetched=False)

        # Timer unchanged — NOT cleared
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], open_since)
        self.assertNotIn("decision_journal", state)

    def test_d1_resumes_after_fetch_recovers(self):
        """Multi-cycle: skip with positions_fetched=False, then fire correctly when fetch recovers."""
        frozen_now = datetime.now(UTC)
        t0 = frozen_now - timedelta(hours=25)
        open_since = _iso(t0)

        state = _make_state({"BNBUSDT": open_since})
        plan = _make_plan(bnb_tw=0.0)
        positions_open = {"BNBUSDT": {"qty": -10.0}}

        # Cycle 1: fetch failed — D1 skipped, timer stays at t0
        _run_d1(plan, state, positions={}, now=frozen_now, positions_fetched=False)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], open_since)
        self.assertNotIn("decision_journal", state)

        # Cycle 2: 5m later, fetch succeeds — D1 sees exch_active=True, age=25h05m → fires
        frozen_now2 = frozen_now + timedelta(minutes=5)
        _run_d1(plan, state, positions=positions_open, now=frozen_now2, positions_fetched=True)

        # Timer preserved (nag-until-fixed), tw forced to 0, journal entry present
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], open_since)
        journal = state.get("decision_journal", [])
        max_hold_entries = [e for e in journal if e.get("override_reason") == "max_hold"]
        self.assertEqual(len(max_hold_entries), 1)
        self.assertEqual(max_hold_entries[0]["pair"], "BNBUSDT")
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], 0.0)


class TestExchangePositionIsOpen(unittest.TestCase):

    def test_missing_pair_returns_false(self):
        self.assertFalse(prl._exchange_position_is_open({}, "BTCUSDT"))

    def test_zero_qty_returns_false(self):
        self.assertFalse(prl._exchange_position_is_open({"BTCUSDT": {"qty": 0.0}}, "BTCUSDT"))

    def test_dust_qty_returns_false(self):
        self.assertFalse(prl._exchange_position_is_open({"BTCUSDT": {"qty": 1e-10}}, "BTCUSDT"))

    def test_real_short_qty_returns_true(self):
        self.assertTrue(prl._exchange_position_is_open({"BTCUSDT": {"qty": -10.0}}, "BTCUSDT"))

    def test_real_long_qty_returns_true(self):
        self.assertTrue(prl._exchange_position_is_open({"BTCUSDT": {"qty": 0.001}}, "BTCUSDT"))

    def test_bad_qty_returns_false(self):
        self.assertFalse(prl._exchange_position_is_open({"BTCUSDT": {"qty": None}}, "BTCUSDT"))


class TestComputePositionAgeSeconds(unittest.TestCase):

    def test_none_returns_zero(self):
        self.assertEqual(prl.compute_position_age_seconds(None), 0.0)

    def test_invalid_string_returns_zero(self):
        self.assertEqual(prl.compute_position_age_seconds("not-a-date"), 0.0)

    def test_valid_timestamp_returns_expected_delta(self):
        frozen_now = datetime.now(UTC)
        open_since = (frozen_now - timedelta(seconds=300)).isoformat()
        with patch.object(prl, "utc_now", return_value=frozen_now):
            age = prl.compute_position_age_seconds(open_since)
        self.assertAlmostEqual(age, 300.0, delta=0.01)

    def test_zero_does_not_trigger_d1(self):
        # None open_since returns 0.0 which is <= _MAX_HOLD_SECONDS — D1 must not fire
        self.assertFalse(prl.compute_position_age_seconds(None) > prl._MAX_HOLD_SECONDS)


if __name__ == "__main__":
    unittest.main()
