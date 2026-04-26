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


def _make_args(mode: str = "primary") -> object:
    import argparse
    return argparse.Namespace(
        execute=False,
        mode=mode,
        state_path=str(ROOT_DIR / "models" / "pairwise_state.json"),
        summary_path=str(ROOT_DIR / "models" / "pairwise_summary.json"),
        model_path=str(ROOT_DIR / "models" / "pairwise_model.json"),
        promotion_report=str(ROOT_DIR / "models" / "pairwise_promotion_report.json"),
        decision_log_path=str(ROOT_DIR / "models" / "pairwise_decision_log.jsonl"),
        refresh_live_data=False,
        force_execute=False,
        force_note="",
    )


class TestDryRunPositionFetch(unittest.TestCase):
    """Defect 3: dry-run should fetch positions for D1/R3 but never call reconcile."""

    def _common_patches(self, state=None):
        if state is None:
            state = {"position_open_since_ts": {}}
        fake_plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0},
            "pair_plans": {},
            "promotion_report_path": None,
        }
        fake_gate = {"requested_gate_ready": False}
        return state, fake_plan, fake_gate

    def test_dryrun_success_fetches_positions(self):
        """When bridge.fetch_open_position_map succeeds, it is called exactly once in dry-run."""
        state, fake_plan, fake_gate = self._common_patches()

        bridge = MagicMock()
        bridge.fetch_open_position_map.return_value = {"BTCUSDT": {"qty": -5.0}}

        with patch.object(prl, "load_execution_bridge", return_value=bridge), \
             patch.object(prl, "load_state", return_value=state), \
             patch.object(prl, "save_state"), \
             patch.object(prl, "build_pairwise_plan", return_value=fake_plan), \
             patch.object(prl, "persist_runtime_plan_state"), \
             patch.object(prl, "load_promotion_gate", return_value=fake_gate), \
             patch.object(prl, "promotion_gate_allows_execution", return_value=False), \
             patch.object(prl, "record_runtime_success"), \
             patch.object(prl, "append_jsonl"), \
             patch.object(prl, "load_notification_bridge", return_value=MagicMock()):
            prl.run_live_once(_make_args())

        # Bridge was called to fetch positions (dry-run position fetch for D1/R3 visibility)
        bridge.fetch_open_position_map.assert_called_once()
        # reconcile must never be called in dry-run
        bridge.reconcile_target_positions.assert_not_called()

    def test_dryrun_credential_failure_uses_empty_positions(self):
        """When bridge raises, dry-run proceeds with positions={} and does not crash."""
        state, fake_plan, fake_gate = self._common_patches()

        with patch.object(prl, "load_execution_bridge", side_effect=RuntimeError("no creds")), \
             patch.object(prl, "load_state", return_value=state), \
             patch.object(prl, "save_state"), \
             patch.object(prl, "build_pairwise_plan", return_value=fake_plan), \
             patch.object(prl, "persist_runtime_plan_state"), \
             patch.object(prl, "load_promotion_gate", return_value=fake_gate), \
             patch.object(prl, "promotion_gate_allows_execution", return_value=False), \
             patch.object(prl, "record_runtime_success"), \
             patch.object(prl, "append_jsonl"), \
             patch.object(prl, "load_notification_bridge", return_value=MagicMock()):
            # Must not raise
            result = prl.run_live_once(_make_args())

        # run_live_once returns 0 in dry-run preview path
        self.assertEqual(result, 0)

    def test_dryrun_never_calls_reconcile(self):
        """Regardless of whether bridge is available, reconcile must never be called."""
        state, fake_plan, fake_gate = self._common_patches()

        bridge = MagicMock()
        bridge.fetch_open_position_map.return_value = {"BNBUSDT": {"qty": 3.0}}

        with patch.object(prl, "load_execution_bridge", return_value=bridge), \
             patch.object(prl, "load_state", return_value=state), \
             patch.object(prl, "save_state"), \
             patch.object(prl, "build_pairwise_plan", return_value=fake_plan), \
             patch.object(prl, "persist_runtime_plan_state"), \
             patch.object(prl, "load_promotion_gate", return_value=fake_gate), \
             patch.object(prl, "promotion_gate_allows_execution", return_value=True), \
             patch.object(prl, "record_runtime_success"), \
             patch.object(prl, "append_jsonl"), \
             patch.object(prl, "load_notification_bridge", return_value=MagicMock()):
            prl.run_live_once(_make_args())

        bridge.reconcile_target_positions.assert_not_called()

    def test_dryrun_equity_fetched_when_bridge_available(self):
        """Dry-run fetches equity from bridge when credentials are available."""
        state, fake_plan, fake_gate = self._common_patches()

        bridge = MagicMock()
        bridge.fetch_open_position_map.return_value = {}
        bridge.fetch_equity.return_value = 5000.0

        with patch.object(prl, "load_execution_bridge", return_value=bridge), \
             patch.object(prl, "load_state", return_value=state), \
             patch.object(prl, "save_state"), \
             patch.object(prl, "build_pairwise_plan", return_value=fake_plan), \
             patch.object(prl, "persist_runtime_plan_state"), \
             patch.object(prl, "load_promotion_gate", return_value=fake_gate), \
             patch.object(prl, "promotion_gate_allows_execution", return_value=False), \
             patch.object(prl, "record_runtime_success"), \
             patch.object(prl, "append_jsonl"), \
             patch.object(prl, "load_notification_bridge", return_value=MagicMock()):
            result = prl.run_live_once(_make_args())

        bridge.fetch_equity.assert_called_once()
        self.assertEqual(result, 0)

    def test_dryrun_equity_failure_does_not_crash(self):
        """Equity fetch failure in dry-run is tolerated; run continues."""
        state, fake_plan, fake_gate = self._common_patches()

        bridge = MagicMock()
        bridge.fetch_open_position_map.return_value = {}
        bridge.fetch_equity.side_effect = RuntimeError("no equity endpoint")

        with patch.object(prl, "load_execution_bridge", return_value=bridge), \
             patch.object(prl, "load_state", return_value=state), \
             patch.object(prl, "save_state"), \
             patch.object(prl, "build_pairwise_plan", return_value=fake_plan), \
             patch.object(prl, "persist_runtime_plan_state"), \
             patch.object(prl, "load_promotion_gate", return_value=fake_gate), \
             patch.object(prl, "promotion_gate_allows_execution", return_value=False), \
             patch.object(prl, "record_runtime_success"), \
             patch.object(prl, "append_jsonl"), \
             patch.object(prl, "load_notification_bridge", return_value=MagicMock()):
            result = prl.run_live_once(_make_args())

        self.assertEqual(result, 0)


class TestD1RunsInDryrun(unittest.TestCase):
    """Acceptance criterion: D1 overlay runs and mutates plan in dry-run."""

    def test_d1_runs_in_dryrun(self):
        """D1 fires in dry-run when position has been open > 24h; no reconcile called."""
        frozen_now = datetime.now(UTC)
        open_since = (frozen_now - timedelta(seconds=prl._MAX_HOLD_SECONDS + 3600)).isoformat()

        state = {
            "position_open_since_ts": {"BNBUSDT": open_since},
        }
        fake_plan = {
            "target_weights": {"BTCUSDT": 0.0, "BNBUSDT": -0.1},
            "pair_plans": {"BNBUSDT": {"target_weight": -0.1}},
            "promotion_report_path": None,
        }
        fake_gate = {"requested_gate_ready": False}

        bridge = MagicMock()
        bridge.fetch_open_position_map.return_value = {"BNBUSDT": {"qty": -10.0}}
        bridge.fetch_equity.return_value = 1000.0

        args = _make_args()  # execute=False

        with patch.object(prl, "utc_now", return_value=frozen_now), \
             patch.object(prl, "load_execution_bridge", return_value=bridge), \
             patch.object(prl, "load_state", return_value=state), \
             patch.object(prl, "save_state"), \
             patch.object(prl, "build_pairwise_plan", return_value=fake_plan), \
             patch.object(prl, "persist_runtime_plan_state"), \
             patch.object(prl, "load_promotion_gate", return_value=fake_gate), \
             patch.object(prl, "promotion_gate_allows_execution", return_value=False), \
             patch.object(prl, "record_runtime_success"), \
             patch.object(prl, "append_jsonl"), \
             patch.object(prl, "load_notification_bridge", return_value=MagicMock()):
            prl.run_live_once(args)

        # D1 must have zeroed the BNBUSDT weight
        self.assertAlmostEqual(fake_plan["target_weights"]["BNBUSDT"], 0.0)
        # decision_journal must contain a max_hold entry
        journal = state.get("decision_journal", [])
        max_hold_entries = [e for e in journal if e.get("override_reason") == "max_hold"]
        self.assertEqual(len(max_hold_entries), 1)
        self.assertEqual(max_hold_entries[0]["pair"], "BNBUSDT")
        # No orders placed
        bridge.reconcile_target_positions.assert_not_called()


class TestD1DryrunTimerPreservation(unittest.TestCase):
    """D1 timer must not be erased by dry-run cycles; 'nag-until-fixed' behaviour."""

    def _run_cycle(
        self,
        state: dict,
        plan: dict,
        positions: dict,
        frozen_now: datetime,
        *,
        positions_fetched: bool = True,
    ) -> None:
        """Single D1 cycle mirroring production logic; mutates state and plan in place.

        Exchange-state-aware: timer resets for fresh signal when exchange is flat,
        or clears when fully flat. D1 only fires while the exchange position is open.

        positions_fetched=False mirrors the fail-safe path: D1 is skipped entirely
        and existing timers are preserved unchanged.
        """
        notif = MagicMock()
        with patch.object(prl, "utc_now", return_value=frozen_now), \
             patch.object(prl, "load_notification_bridge", return_value=notif):
            position_open_since: dict = dict(state.get("position_open_since_ts") or {})
            if not positions_fetched:
                return
            for _pair in list(prl.PAIRS):
                _tw = float(plan["target_weights"].get(_pair, 0.0))
                _prev_since = position_open_since.get(_pair)
                tw_active = abs(_tw) > prl.TARGET_WEIGHT_EPS
                exch_active = prl._exchange_position_is_open(positions, _pair)
                if exch_active:
                    # Exchange has a real open position — track continuously.
                    if _prev_since is None:
                        position_open_since[_pair] = frozen_now.isoformat()
                    else:
                        age_secs = prl.compute_position_age_seconds(_prev_since)
                        if age_secs > prl._MAX_HOLD_SECONDS:
                            plan["target_weights"][_pair] = 0.0
                            if _pair in plan.get("pair_plans", {}):
                                plan["pair_plans"][_pair]["target_weight"] = 0.0
                            # Timer NOT cleared — nag-until-fixed while exchange is open.
                            state.setdefault("decision_journal", []).append(
                                {
                                    "at": frozen_now.isoformat(),
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
                        position_open_since[_pair] = frozen_now.isoformat()
                    else:
                        position_open_since[_pair] = None
            state["position_open_since_ts"] = position_open_since

    def test_d1_dryrun_repeated_fire_preserves_age(self):
        """3 consecutive dry-run cycles with exchange still open: timer unchanged, 3 journal entries."""
        base_now = datetime.now(UTC)
        original_open_since = (base_now - timedelta(hours=25)).isoformat()

        state = {"position_open_since_ts": {"BNBUSDT": original_open_since}}
        plan = {"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0}, "pair_plans": {}}
        # Exchange position stays open throughout (dry-run never closes it)
        positions = {"BNBUSDT": {"qty": -10.0}}

        for i in range(3):
            cycle_now = base_now + timedelta(minutes=i * 5)
            self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], original_open_since,
                             f"Timer must not change before cycle {i}")
            self._run_cycle(state, plan, positions, cycle_now)

        # After 3 cycles timer is STILL the original open timestamp
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], original_open_since)
        # 3 max_hold journal entries accumulated
        journal = state.get("decision_journal", [])
        max_hold_entries = [e for e in journal if e.get("override_reason") == "max_hold"]
        self.assertEqual(len(max_hold_entries), 3)


class TestD1DryrunCloseResetsTimer(unittest.TestCase):
    """Codex repro: dry-run cycle after successful close must reset timer for fresh signal."""

    def test_dryrun_close_then_fresh_signal_resets_timer(self):
        """Cycle A: exch open, age>24h, D1 re-fires (nag), timer kept.
        Cycle B: exch just closed, fresh tw signal — timer resets to cycle-B now, tw unchanged."""
        base_now = datetime.now(UTC)
        original_open_since = (base_now - timedelta(hours=25)).isoformat()

        state = {"position_open_since_ts": {"BNBUSDT": original_open_since}}
        plan = {"target_weights": {"BTCUSDT": 0.0, "BNBUSDT": 0.0}, "pair_plans": {}}
        positions_open = {"BNBUSDT": {"qty": -10.0}}

        # Cycle A: exchange still open, tw=0 (D1 previously zeroed it). Nag fires again.
        t_a = base_now
        _run_cycle = TestD1DryrunTimerPreservation()._run_cycle
        _run_cycle(state, plan, positions_open, t_a)

        # Nag fired — timer unchanged, journal entry added
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], original_open_since)
        journal_a = state.get("decision_journal", [])
        self.assertEqual(len([e for e in journal_a if e["override_reason"] == "max_hold"]), 1)

        # Cycle B: exchange has just closed; fresh tw signal arrives.
        t_b = base_now + timedelta(minutes=5)
        plan["target_weights"]["BNBUSDT"] = -0.05  # fresh planner signal
        positions_closed = {}  # exchange flat

        _run_cycle(state, plan, positions_closed, t_b)

        # Timer must be reset to t_b (not the dead 25h timestamp)
        self.assertEqual(state["position_open_since_ts"]["BNBUSDT"], t_b.isoformat())
        # tw must be unchanged at -0.05 — D1 must NOT suppress the fresh entry
        self.assertAlmostEqual(plan["target_weights"]["BNBUSDT"], -0.05)
        # No new journal entries on cycle B
        journal_b = state.get("decision_journal", [])
        self.assertEqual(len([e for e in journal_b if e["override_reason"] == "max_hold"]), 1)


if __name__ == "__main__":
    unittest.main()
