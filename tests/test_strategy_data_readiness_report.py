#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import report_strategy_data_readiness as report


def write_csv(path: Path, header: str, *rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join((header, *rows)) + "\n")


class StrategyDataReadinessReportTests(unittest.TestCase):
    def test_parse_timestamp_handles_milliseconds_and_iso(self) -> None:
        self.assertEqual(report._parse_timestamp("1714953600000"), "2024-05-06T00:00:00+00:00")
        self.assertEqual(report._parse_timestamp("2026-04-06T00:00:00Z"), "2026-04-06T00:00:00+00:00")

    def test_read_last_csv_timestamp_prefers_known_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            write_csv(path, "timestamp,value", "1714953600000,1", "1714953900000,2")
            self.assertEqual(report._read_last_csv_timestamp(path), "2024-05-06T00:05:00+00:00")

    def test_build_report_marks_missing_strategic_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            data_dir = project_root / "data"
            futures_dir = data_dir / "binance_futures"
            derivatives_dir = futures_dir / "derivatives"
            market_context_dir = data_dir / "market_context" / "daily"
            manifest_path = data_dir / "market_context" / "manifest.json"

            write_csv(futures_dir / "BTCUSDT_5m.csv", "open_time,close", "1714953600000,1", "1714953900000,2")
            write_csv(futures_dir / "BTCUSDT_1d.csv", "open_time,close", "1714867200000,1")
            write_csv(futures_dir / "BTCUSDT_funding_2026-01-01_2026-04-06.csv", "fundingTime,fundingRate", "1714953600000,0.001")
            write_csv(futures_dir / "BNBUSDT_5m.csv", "open_time,close", "1714953600000,1")
            write_csv(futures_dir / "BNBUSDT_1d.csv", "open_time,close", "1714867200000,1")
            write_csv(futures_dir / "BNBUSDT_funding_2026-01-01_2026-04-06.csv", "fundingTime,fundingRate", "1714953600000,0.001")
            for pair in ("BTCUSDT", "BNBUSDT"):
                for metric in report.DERIVATIVE_METRICS:
                    write_csv(derivatives_dir / f"{pair}_{metric}_5m.csv", "timestamp,value", "1714953600000,1")
            for name in report.MARKET_CONTEXT_NAMES:
                write_csv(market_context_dir / f"{name}.csv", "date,close", "2026-04-06,100")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"QQQ": {"updated_at": "2026-04-06T00:00:00Z"}}))

            with patch.object(report, "ROOT_DIR", project_root), patch.object(report, "DATA_DIR", data_dir), patch.object(
                report, "MODELS_DIR", project_root / "models"
            ), patch.object(report, "BINANCE_FUTURES_DIR", futures_dir), patch.object(
                report, "DERIVATIVE_DIR", derivatives_dir
            ), patch.object(
                report, "MARKET_CONTEXT_DIR", market_context_dir
            ), patch.object(
                report, "MARKET_CONTEXT_MANIFEST", manifest_path
            ), patch.object(
                report,
                "STRATEGIC_LAYER_SPECS",
                (
                    {
                        "key": "lob_microstructure",
                        "priority": 1,
                        "path": data_dir / "lob",
                        "status_if_missing": "missing",
                        "decision_role": "나쁜 진입 차단",
                        "why": "test",
                        "required_sources": ("depth",),
                        "derived_features": ("spread",),
                        "source_links": ("https://example.com",),
                    },
                ),
            ):
                payload = report.build_strategy_data_report(project_root, ("BTCUSDT", "BNBUSDT"))

            self.assertEqual(payload["local_layers"]["ohlcv"]["status"], "ready")
            self.assertEqual(payload["local_layers"]["derivatives"]["status"], "ready")
            self.assertEqual(payload["local_layers"]["market_context"]["status"], "ready")
            self.assertEqual(payload["strategic_layers"][0]["status"], "missing")

    def test_summarize_ohlcv_layer_uses_freshest_funding_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            futures_dir = Path(tmp) / "data" / "binance_futures"
            write_csv(futures_dir / "BTCUSDT_5m.csv", "open_time,close", "1714953600000,1")
            write_csv(futures_dir / "BTCUSDT_1d.csv", "open_time,close", "1714867200000,1")
            write_csv(
                futures_dir / "BTCUSDT_funding_2025-10-01_2026-04-18.csv",
                "fundingTime,fundingRate",
                "2026-04-18T08:00:00Z,0.001",
            )
            write_csv(
                futures_dir / "BTCUSDT_funding_2026-02-15_2026-04-14.csv",
                "fundingTime,fundingRate",
                "2026-04-14T08:00:00Z,0.001",
            )

            with patch.object(report, "BINANCE_FUTURES_DIR", futures_dir):
                payload = report.summarize_ohlcv_layer(("BTCUSDT",))

            funding = payload["per_pair"]["BTCUSDT"]["funding"]
            self.assertTrue(funding["path"].endswith("BTCUSDT_funding_2025-10-01_2026-04-18.csv"))
            self.assertEqual(funding["last_timestamp"], "2026-04-18T08:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
