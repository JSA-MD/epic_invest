"""Unit tests for scripts/calibrate_slippage.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import calibrate_slippage as cal


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


class TestLoadSlippageRecords(unittest.TestCase):
    def test_empty_file_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            p = Path(f.name)
        records = cal.load_slippage_records(p, datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(records, [])
        p.unlink()

    def test_missing_file_returns_empty(self) -> None:
        p = Path("/nonexistent/does_not_exist.jsonl")
        records = cal.load_slippage_records(p, datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(records, [])

    def test_filters_by_cutoff(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, dir="/tmp") as f:
            p = Path(f.name)
        _write_jsonl(p, [
            {"ts": "2026-04-20T00:00:00+00:00", "symbol": "BTC/USDT:USDT", "slippage_bps": 1.0, "mode": "live"},
            {"ts": "2026-04-18T00:00:00+00:00", "symbol": "BTC/USDT:USDT", "slippage_bps": 2.0, "mode": "live"},
        ])
        cutoff = datetime(2026, 4, 19, tzinfo=timezone.utc)
        records = cal.load_slippage_records(p, cutoff)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["slippage_bps"], 1.0)
        p.unlink()

    def test_skips_malformed_lines(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False, dir="/tmp") as f:
            p = Path(f.name)
        with p.open("w") as fh:
            fh.write("not json\n")
            fh.write(json.dumps({"ts": "2026-04-25T00:00:00+00:00", "symbol": "BTC/USDT:USDT", "slippage_bps": 3.0, "mode": "live"}) + "\n")
        cutoff = datetime(2026, 4, 1, tzinfo=timezone.utc)
        records = cal.load_slippage_records(p, cutoff)
        self.assertEqual(len(records), 1)
        p.unlink()


class TestComputeSymbolStats(unittest.TestCase):
    def _make_records(self, values: list[float], symbol: str = "BTC/USDT:USDT", mode: str = "live") -> list[dict]:
        return [{"symbol": symbol, "slippage_bps": v, "mode": mode} for v in values]

    def test_percentiles_correct(self) -> None:
        # Use a known distribution: 10 values 1..10
        records = self._make_records(list(range(1, 11)))
        stats = cal.compute_symbol_stats(records)
        self.assertIn("BTCUSDT", stats)
        s = stats["BTCUSDT"]
        self.assertEqual(s["n"], 10)
        import numpy as np
        arr = list(range(1, 11))
        self.assertAlmostEqual(s["p50"], float(np.percentile(arr, 50)), places=5)
        self.assertAlmostEqual(s["p75"], float(np.percentile(arr, 75)), places=5)
        self.assertAlmostEqual(s["p90"], float(np.percentile(arr, 90)), places=5)
        self.assertAlmostEqual(s["p99"], float(np.percentile(arr, 99)), places=5)

    def test_mode_filter_excludes_demo(self) -> None:
        records = (
            self._make_records([5.0, 6.0], mode="live")
            + self._make_records([100.0], mode="demo")
        )
        live_stats = cal.compute_symbol_stats(records, mode_filter="live")
        self.assertEqual(live_stats["BTCUSDT"]["n"], 2)
        self.assertLess(live_stats["BTCUSDT"]["p99"], 10.0)

    def test_mode_filter_demo_only(self) -> None:
        records = (
            self._make_records([1.0], mode="live")
            + self._make_records([50.0, 60.0], mode="demo")
        )
        demo_stats = cal.compute_symbol_stats(records, mode_filter="demo")
        self.assertEqual(demo_stats["BTCUSDT"]["n"], 2)

    def test_empty_records_returns_empty(self) -> None:
        stats = cal.compute_symbol_stats([])
        self.assertEqual(stats, {})

    def test_single_record_no_std_error(self) -> None:
        records = self._make_records([3.5])
        stats = cal.compute_symbol_stats(records)
        self.assertAlmostEqual(stats["BTCUSDT"]["p50"], 3.5)
        self.assertEqual(stats["BTCUSDT"]["std"], 0.0)


class TestVerdictForSymbol(unittest.TestCase):
    def _stats(self, p75: float) -> dict:
        return {"n": 10, "mean": p75, "std": 1.0, "p50": p75 * 0.8, "p75": p75, "p90": p75 * 1.2, "p99": p75 * 1.5}

    def test_reasonable(self) -> None:
        # p75 == assumption → REASONABLE
        line, is_alert = cal.verdict_for_symbol("BTCUSDT", self._stats(2.0), 2.0, 1.5)
        self.assertFalse(is_alert)
        self.assertIn("REASONABLE", line)

    def test_underestimated_below_threshold(self) -> None:
        # p75 = 2.5bp vs 2.0bp assumption → ratio 1.25 < 1.5 → UNDERESTIMATED (no alert)
        line, is_alert = cal.verdict_for_symbol("BTCUSDT", self._stats(2.5), 2.0, 1.5)
        self.assertFalse(is_alert)
        self.assertIn("UNDERESTIMATED", line)

    def test_alert_triggered(self) -> None:
        # p75 = 4.0bp vs 2.0bp assumption → ratio 2.0 >= 1.5 → ALERT
        line, is_alert = cal.verdict_for_symbol("BTCUSDT", self._stats(4.0), 2.0, 1.5)
        self.assertTrue(is_alert)
        self.assertIn("ALERT", line)

    def test_overestimated(self) -> None:
        # p75 = 1.0bp vs 2.0bp assumption → ratio 0.5 < 0.9 → OVERESTIMATED
        line, is_alert = cal.verdict_for_symbol("BTCUSDT", self._stats(1.0), 2.0, 1.5)
        self.assertFalse(is_alert)
        self.assertIn("OVERESTIMATED", line)


class TestRunEndToEnd(unittest.TestCase):
    def _make_log(self, records: list[dict]) -> Path:
        p = Path(tempfile.mktemp(suffix=".jsonl", dir="/tmp"))
        _write_jsonl(p, records)
        return p

    def test_run_no_data(self) -> None:
        """Empty log produces a result without errors."""
        log = self._make_log([])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.json"
            now = datetime(2026, 4, 26, tzinfo=timezone.utc)
            result = cal.run(days=7, log_path=log, alert=False, output_path=out, _now=now)
        self.assertEqual(result["live_count"], 0)
        self.assertEqual(result["demo_count"], 0)
        self.assertFalse(result["has_alert"])
        log.unlink(missing_ok=True)

    def test_run_alert_fires_for_high_slippage(self) -> None:
        """P75 >> assumption generates alert_symbols."""
        now = datetime(2026, 4, 26, tzinfo=timezone.utc)
        ts = "2026-04-25T12:00:00+00:00"
        records = [
            {"ts": ts, "symbol": "BTC/USDT:USDT", "slippage_bps": float(v), "mode": "live", "side": "buy"}
            for v in [8.0] * 20  # P75 = 8.0 >> 2.0bp assumption; ratio = 4.0
        ]
        log = self._make_log(records)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.json"
            result = cal.run(days=7, log_path=log, alert=False, output_path=out, _now=now)
        self.assertTrue(result["has_alert"])
        self.assertIn("BTCUSDT", result["alert_symbols"])
        log.unlink(missing_ok=True)

    def test_run_json_output_written(self) -> None:
        now = datetime(2026, 4, 26, tzinfo=timezone.utc)
        ts = "2026-04-25T12:00:00+00:00"
        records = [
            {"ts": ts, "symbol": "BTC/USDT:USDT", "slippage_bps": 2.0, "mode": "live", "side": "buy"}
        ]
        log = self._make_log(records)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.json"
            cal.run(days=7, log_path=log, alert=False, output_path=out, _now=now)
            self.assertTrue(out.exists())
            with out.open() as fh:
                data = json.load(fh)
            self.assertEqual(data["live_count"], 1)
        log.unlink(missing_ok=True)

    def test_run_demo_vs_live_separation(self) -> None:
        now = datetime(2026, 4, 26, tzinfo=timezone.utc)
        ts = "2026-04-25T12:00:00+00:00"
        records = [
            {"ts": ts, "symbol": "BTC/USDT:USDT", "slippage_bps": 2.0, "mode": "live"},
            {"ts": ts, "symbol": "BTC/USDT:USDT", "slippage_bps": 2.0, "mode": "demo"},
            {"ts": ts, "symbol": "BTC/USDT:USDT", "slippage_bps": 2.0, "mode": "demo"},
        ]
        log = self._make_log(records)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.json"
            result = cal.run(days=7, log_path=log, alert=False, output_path=out, _now=now)
        self.assertEqual(result["live_count"], 1)
        self.assertEqual(result["demo_count"], 2)
        log.unlink(missing_ok=True)

    def test_run_alert_flag_no_telegram_when_no_token(self) -> None:
        """--alert with empty env vars should not raise; just log a warning."""
        import os
        orig = os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        try:
            now = datetime(2026, 4, 26, tzinfo=timezone.utc)
            ts = "2026-04-25T12:00:00+00:00"
            records = [
                {"ts": ts, "symbol": "BTC/USDT:USDT", "slippage_bps": 10.0, "mode": "live"}
                for _ in range(20)
            ]
            log = self._make_log(records)
            with tempfile.TemporaryDirectory() as tmpdir:
                out = Path(tmpdir) / "out.json"
                # Should not raise even with alert=True and no token
                result = cal.run(days=7, log_path=log, alert=True, output_path=out, _now=now)
            self.assertTrue(result["has_alert"])
            log.unlink(missing_ok=True)
        finally:
            if orig is not None:
                os.environ["TELEGRAM_BOT_TOKEN"] = orig


if __name__ == "__main__":
    unittest.main()
