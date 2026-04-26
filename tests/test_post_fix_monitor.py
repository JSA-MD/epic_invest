"""Unit tests for monitor_post_fix_evolution.py.

Tests:
  - cooldown trend aggregation (snapshot collection from synthetic state)
  - trade count aggregation accuracy from synthetic slippage log
  - breadth stats aggregation from synthetic decisions log
  - alert threshold logic (BNB never-zero, consecutive no-trade, drift)
  - mock Telegram send path
"""

from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers — patch paths before importing the module under test
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _make_module(
    shadow_state_path: Path,
    decisions_log_path: Path,
    slippage_log_path: Path,
    models_dir: Path,
    monitoring_state_path: Path,
    docs_dir: Path,
) -> object:
    """Import module with patched paths."""
    import importlib
    import monitor_post_fix_evolution as m

    m.SHADOW_STATE_PATH = shadow_state_path
    m.DECISIONS_LOG_PATH = decisions_log_path
    m.SLIPPAGE_LOG_PATH = slippage_log_path
    m.MODELS_DIR = models_dir
    m.MONITORING_STATE_PATH = monitoring_state_path
    m.DOCS_DIR = docs_dir
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp(tmp_path):
    return tmp_path


@pytest.fixture()
def module(tmp):
    """Patched module instance."""
    import monitor_post_fix_evolution as m
    m.SHADOW_STATE_PATH = tmp / "shadow_state.json"
    m.DECISIONS_LOG_PATH = tmp / "pairwise_regime_decisions.jsonl"
    m.SLIPPAGE_LOG_PATH = tmp / "pairwise_slippage.jsonl"
    m.MODELS_DIR = tmp / "models"
    m.MONITORING_STATE_PATH = tmp / "monitor_state.json"
    m.DOCS_DIR = tmp / "docs"
    m.MODELS_DIR.mkdir()
    return m


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _write_shadow_state(path: Path, btc_cd: int, bnb_cd: int) -> None:
    state = {
        "shadow_paper": {
            "cooldown_bars_left": {"BTCUSDT": btc_cd, "BNBUSDT": bnb_cd}
        }
    }
    path.write_text(json.dumps(state))


def _write_decisions_log(path: Path, entries: list[dict]) -> None:
    lines = [json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n")


def _write_slippage_log(path: Path, entries: list[dict]) -> None:
    lines = [json.dumps(e) for e in entries]
    path.write_text("\n".join(lines) + "\n")


def _make_decision_entry(at_kst_date: str, pair: str, breadth: float, cooldown: int) -> dict:
    """Build a synthetic decisions JSONL entry."""
    at = f"{at_kst_date}T02:00:00+00:00"  # UTC time, maps to same KST date +9h
    return {
        "at": at,
        "mode": "live",
        "plan": {
            "pair_plans": {
                pair: {
                    "breadth_score": breadth,
                    "cooldown_bars_left_after": cooldown,
                }
            }
        },
    }


def _make_slippage_entry(at_kst_date: str, symbol: str) -> dict:
    at = f"{at_kst_date}T02:00:00+00:00"
    return {"at": at, "symbol": symbol, "side": "buy", "qty": 0.1}


def _snapshot(date: str, btc_cd: int, bnb_cd: int, btc_trades: int, bnb_trades: int) -> dict:
    return {
        "date": date,
        "cooldown": {"BTCUSDT": btc_cd, "BNBUSDT": bnb_cd},
        "trade_counts": {"BTCUSDT": btc_trades, "BNBUSDT": bnb_trades},
        "drift_bps": None,
        "alerts": [],
    }


# ---------------------------------------------------------------------------
# Tests: collect_cooldown_snapshot
# ---------------------------------------------------------------------------

class TestCooldownSnapshot:
    def test_returns_correct_values(self, module, tmp):
        _write_shadow_state(module.SHADOW_STATE_PATH, btc_cd=144, bnb_cd=271)
        result = module.collect_cooldown_snapshot()
        assert result["BTCUSDT"] == 144
        assert result["BNBUSDT"] == 271

    def test_missing_file_returns_minus_one(self, module):
        # File does not exist
        result = module.collect_cooldown_snapshot()
        assert result["BTCUSDT"] == -1
        assert result["BNBUSDT"] == -1

    def test_zero_cooldown(self, module):
        _write_shadow_state(module.SHADOW_STATE_PATH, btc_cd=0, bnb_cd=0)
        result = module.collect_cooldown_snapshot()
        assert result["BTCUSDT"] == 0
        assert result["BNBUSDT"] == 0


# ---------------------------------------------------------------------------
# Tests: collect_breadth_stats
# ---------------------------------------------------------------------------

class TestBreadthStats:
    def test_correct_mean_and_max(self, module):
        date = "2026-04-26"
        entries = [
            _make_decision_entry(date, "BTCUSDT", 0.3, 0),
            _make_decision_entry(date, "BTCUSDT", 0.7, 0),
            _make_decision_entry(date, "BTCUSDT", 0.5, 0),
        ]
        _write_decisions_log(module.DECISIONS_LOG_PATH, entries)
        result = module.collect_breadth_stats(date)
        btc = result["BTCUSDT"]
        assert btc["count"] == 3
        assert abs(btc["mean"] - 0.5) < 0.001
        assert abs(btc["max"] - 0.7) < 0.001

    def test_no_log_returns_none(self, module):
        result = module.collect_breadth_stats("2026-04-26")
        assert result["BTCUSDT"]["mean"] is None
        assert result["BTCUSDT"]["count"] == 0

    def test_filters_to_date(self, module):
        entries = [
            _make_decision_entry("2026-04-25", "BTCUSDT", 0.9, 0),
            _make_decision_entry("2026-04-26", "BTCUSDT", 0.2, 0),
        ]
        _write_decisions_log(module.DECISIONS_LOG_PATH, entries)
        result = module.collect_breadth_stats("2026-04-26")
        btc = result["BTCUSDT"]
        assert btc["count"] == 1
        assert abs(btc["mean"] - 0.2) < 0.001

    def test_p50_single_value(self, module):
        date = "2026-04-26"
        entries = [_make_decision_entry(date, "BNBUSDT", 0.65, 0)]
        _write_decisions_log(module.DECISIONS_LOG_PATH, entries)
        result = module.collect_breadth_stats(date)
        bnb = result["BNBUSDT"]
        assert bnb["p50"] == pytest.approx(0.65, abs=0.001)


# ---------------------------------------------------------------------------
# Tests: collect_trade_counts
# ---------------------------------------------------------------------------

class TestTradeCounts:
    def test_counts_per_pair(self, module):
        date = "2026-04-26"
        entries = [
            _make_slippage_entry(date, "BTCUSDT"),
            _make_slippage_entry(date, "BTCUSDT"),
            _make_slippage_entry(date, "BNBUSDT"),
        ]
        _write_slippage_log(module.SLIPPAGE_LOG_PATH, entries)
        result = module.collect_trade_counts(date)
        assert result["BTCUSDT"] == 2
        assert result["BNBUSDT"] == 1

    def test_zero_if_no_log(self, module):
        result = module.collect_trade_counts("2026-04-26")
        assert result["BTCUSDT"] == 0
        assert result["BNBUSDT"] == 0

    def test_filters_to_date(self, module):
        entries = [
            _make_slippage_entry("2026-04-25", "BTCUSDT"),
            _make_slippage_entry("2026-04-26", "BNBUSDT"),
        ]
        _write_slippage_log(module.SLIPPAGE_LOG_PATH, entries)
        result = module.collect_trade_counts("2026-04-26")
        assert result["BTCUSDT"] == 0
        assert result["BNBUSDT"] == 1

    def test_unknown_symbol_ignored(self, module):
        entries = [_make_slippage_entry("2026-04-26", "ETHUSDT")]
        _write_slippage_log(module.SLIPPAGE_LOG_PATH, entries)
        result = module.collect_trade_counts("2026-04-26")
        assert result["BTCUSDT"] == 0
        assert result["BNBUSDT"] == 0


# ---------------------------------------------------------------------------
# Tests: evaluate_alerts
# ---------------------------------------------------------------------------

class TestEvaluateAlerts:
    def test_no_alerts_when_ok(self, module):
        # BNB hits 0 in history, has trades, drift ok
        history = [_snapshot(f"2026-04-{20+i:02d}", 0, 0, 1, 1) for i in range(6)]
        today = _snapshot("2026-04-26", 0, 0, 1, 1)
        today["drift_bps"] = -50.0
        alerts = module.evaluate_alerts(today, history + [today])
        assert alerts == []

    def test_bnb_never_zero_triggers(self, module):
        # 7 snapshots where BNB cooldown is always > 0
        history = [_snapshot(f"2026-04-{20+i:02d}", 0, 100, 1, 1) for i in range(6)]
        today = _snapshot("2026-04-26", 0, 50, 1, 1)
        today["drift_bps"] = -10.0
        alerts = module.evaluate_alerts(today, history + [today])
        bnb_alert = [a for a in alerts if "BNB" in a and "0" in a]
        assert bnb_alert, f"Expected BNB cooldown alert, got: {alerts}"

    def test_consecutive_zero_trades_triggers(self, module):
        # 3 consecutive days with 0 total trades
        history = [_snapshot(f"2026-04-{20+i:02d}", 0, 0, 0, 0) for i in range(2)]
        today = _snapshot("2026-04-26", 0, 0, 0, 0)
        today["drift_bps"] = None
        alerts = module.evaluate_alerts(today, history + [today])
        trade_alert = [a for a in alerts if "거래" in a or "trade" in a.lower() or "0건" in a or "연속" in a]
        assert trade_alert, f"Expected no-trade alert, got: {alerts}"

    def test_drift_threshold_triggers(self, module):
        history = []
        today = _snapshot("2026-04-26", 0, 0, 1, 1)
        today["drift_bps"] = -150.0  # below -100 threshold
        alerts = module.evaluate_alerts(today, [today])
        drift_alert = [a for a in alerts if "bps" in a.lower() or "drift" in a.lower() or "Drift" in a]
        assert drift_alert, f"Expected drift alert, got: {alerts}"

    def test_drift_ok_no_alert(self, module):
        today = _snapshot("2026-04-26", 0, 0, 1, 1)
        today["drift_bps"] = -99.9
        alerts = module.evaluate_alerts(today, [today])
        drift_alert = [a for a in alerts if "bps" in a.lower() or "Drift" in a]
        assert drift_alert == []

    def test_partial_history_no_false_alarm(self, module):
        # Only 2 days of history — BNB-never-zero rule needs 7, should not fire
        history = [_snapshot(f"2026-04-{24+i:02d}", 0, 200, 1, 1) for i in range(1)]
        today = _snapshot("2026-04-26", 0, 180, 1, 1)
        today["drift_bps"] = None
        alerts = module.evaluate_alerts(today, history + [today])
        bnb_alert = [a for a in alerts if "BNB" in a]
        assert bnb_alert == [], f"Should not alert with only 2 days: {alerts}"


# ---------------------------------------------------------------------------
# Tests: Telegram send (mock)
# ---------------------------------------------------------------------------

class TestTelegramSend:
    def test_send_called_on_alert(self, module, tmp):
        """run() calls _send_telegram when alerts are present."""
        _write_shadow_state(module.SHADOW_STATE_PATH, btc_cd=0, bnb_cd=0)

        # Write 3 days of zero-trade history snapshots
        for i in range(2):
            snap = _snapshot(f"2026-04-{24+i:02d}", 0, 0, 0, 0)
            (module.MODELS_DIR / f"post_fix_monitoring_2026-04-{24+i:02d}.json").write_text(
                json.dumps(snap)
            )

        # No slippage today = 0 trades, triggering 3-consecutive alert
        date = "2026-04-26"

        KST = timezone(timedelta(hours=9))
        fake_now = datetime(2026, 4, 26, 9, 15, tzinfo=KST)

        sent_messages: list[str] = []

        def fake_send(text: str) -> None:
            sent_messages.append(text)

        with (
            patch.object(module, "now_kst", return_value=fake_now),
            patch.object(module, "_send_telegram", side_effect=fake_send),
        ):
            module.run(dry_run=False)

        assert sent_messages, "Expected at least one Telegram message"

    def test_dry_run_no_send(self, module, tmp):
        """run(dry_run=True) must not call _send_telegram."""
        _write_shadow_state(module.SHADOW_STATE_PATH, btc_cd=0, bnb_cd=0)

        KST = timezone(timedelta(hours=9))
        fake_now = datetime(2026, 4, 26, 9, 15, tzinfo=KST)

        with (
            patch.object(module, "now_kst", return_value=fake_now),
            patch.object(module, "_send_telegram") as mock_send,
        ):
            module.run(dry_run=True)

        mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: auto-stop after 7 days
# ---------------------------------------------------------------------------

class TestAutoStop:
    def test_window_detection(self, module):
        # Write 7 snapshot files — should detect window complete
        for i in range(7):
            snap = _snapshot(f"2026-04-{20+i:02d}", 0, 0, 0, 0)
            (module.MODELS_DIR / f"post_fix_monitoring_2026-04-{20+i:02d}.json").write_text(
                json.dumps(snap)
            )
        history = module._load_history()
        assert module._is_past_window(history) is True

    def test_window_not_complete(self, module):
        for i in range(6):
            snap = _snapshot(f"2026-04-{20+i:02d}", 0, 0, 0, 0)
            (module.MODELS_DIR / f"post_fix_monitoring_2026-04-{20+i:02d}.json").write_text(
                json.dumps(snap)
            )
        history = module._load_history()
        assert module._is_past_window(history) is False
