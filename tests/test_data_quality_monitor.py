#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import data_quality_monitor as monitor


def write_csv(path: Path, header: str, *rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join((header, *rows)) + "\n")


class DataQualityMonitorTests(unittest.TestCase):
    def test_classify_freshness(self) -> None:
        threshold = monitor.FreshnessThreshold(10.0, 20.0)
        self.assertEqual(monitor.classify_freshness(None, threshold), "missing")
        self.assertEqual(monitor.classify_freshness(5.0, threshold), "fresh")
        self.assertEqual(monitor.classify_freshness(15.0, threshold), "aging")
        self.assertEqual(monitor.classify_freshness(25.0, threshold), "stale")

    def test_read_last_csv_timestamp_supports_trade_time_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agg_trades.csv"
            write_csv(path, "a,p,q,T", "1,100,1,2026-04-12T13:59:00Z")
            self.assertEqual(monitor._read_last_csv_timestamp(path), "2026-04-12T13:59:00+00:00")

    def test_build_snapshot_flags_missing_lob_and_daily_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            futures = data_dir / "binance_futures"
            derivatives = futures / "derivatives"
            market_context = data_dir / "market_context" / "daily"
            now = datetime(2026, 4, 12, 14, 0, tzinfo=timezone.utc)

            write_csv(futures / "BTCUSDT_5m.csv", "open_time,close", "2026-04-12T13:55:00Z,1")
            write_csv(futures / "BNBUSDT_5m.csv", "open_time,close", "2026-04-12T13:55:00Z,1")
            write_csv(futures / "BTCUSDT_1d.csv", "date,close", "2026-04-11,1")
            write_csv(futures / "BTCUSDT_funding_2026-04-01_2026-04-12.csv", "fundingTime,fundingRate", "2026-04-12T08:00:00Z,0.001")
            write_csv(futures / "BNBUSDT_funding_2026-04-01_2026-04-12.csv", "fundingTime,fundingRate", "2026-04-12T08:00:00Z,0.001")

            for pair in ("BTCUSDT", "BNBUSDT"):
                for metric in monitor.DERIVATIVE_METRICS:
                    write_csv(derivatives / f"{pair}_{metric}_5m.csv", "timestamp,value", "2026-04-12T08:00:00Z,1")
            for name in monitor.DEFAULT_MARKET_CONTEXT:
                write_csv(market_context / f"{name}.csv", "date,close", "2026-04-11,100")

            with (
                patch.object(monitor, "ROOT_DIR", root),
                patch.object(monitor, "DATA_DIR", data_dir),
                patch.object(monitor, "MODELS_DIR", root / "models"),
                patch.object(monitor, "BINANCE_FUTURES_DIR", futures),
                patch.object(monitor, "DERIVATIVE_DIR", derivatives),
                patch.object(monitor, "MARKET_CONTEXT_DIR", market_context),
                patch.object(monitor, "LOB_DIR", data_dir / "lob" / "binance_futures"),
            ):
                snapshot = monitor.build_data_quality_snapshot(pairs=("BTCUSDT", "BNBUSDT"), now=now)

            self.assertEqual(snapshot["ohlcv"]["status"], "warning")
            self.assertEqual(snapshot["lob"]["status"], "warning")
            self.assertTrue(any("LOB" in item for item in snapshot["recommendations"]))

    def test_market_context_treats_us_weekend_as_fresh(self) -> None:
        now = datetime(2026, 4, 13, 4, 40, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            market_context = root / "data" / "market_context" / "daily"
            write_csv(market_context / "QQQ.csv", "date,close", "2026-04-10,100")
            with patch.object(monitor, "MARKET_CONTEXT_DIR", market_context):
                payload = monitor.summarize_market_context_feed("QQQ", now=now)
        self.assertEqual(payload["freshness"], "fresh")
        self.assertEqual(payload["expected_latest_date"], "2026-04-10")
        self.assertEqual(payload["trading_day_lag"], 0)

    def test_market_context_flags_real_trading_day_lag(self) -> None:
        now = datetime(2026, 4, 15, 22, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            market_context = root / "data" / "market_context" / "daily"
            write_csv(market_context / "QQQ.csv", "date,close", "2026-04-10,100")
            with patch.object(monitor, "MARKET_CONTEXT_DIR", market_context):
                payload = monitor.summarize_market_context_feed("QQQ", now=now)
        self.assertEqual(payload["freshness"], "stale")
        self.assertGreaterEqual(payload["trading_day_lag"], 2)


if __name__ == "__main__":
    unittest.main()
