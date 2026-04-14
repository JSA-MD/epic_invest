#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import lob_market_data as lob


class LobMarketDataTests(unittest.TestCase):
    def test_compute_microstructure_features(self) -> None:
        row = lob.compute_microstructure_features(
            "BTCUSDT",
            collected_at="2026-04-12T14:00:00+00:00",
            depth_snapshot={
                "lastUpdateId": 123,
                "bids": [["100", "5"], ["99", "5"]],
                "asks": [["101", "4"], ["102", "6"]],
            },
            book_ticker={
                "bidPrice": "100",
                "bidQty": "5",
                "askPrice": "101",
                "askQty": "4",
                "time": 1776002400000,
            },
            agg_trades=[
                {"a": 1, "q": "2", "m": False, "T": 1776002400000},
                {"a": 2, "q": "1", "m": True, "T": 1776002401000},
            ],
        )
        self.assertAlmostEqual(row["spread"], 1.0, places=6)
        self.assertAlmostEqual(row["agg_trade_buy_share"], 2 / 3, places=6)
        self.assertGreater(row["microprice"], 100.0)

    def test_collect_symbol_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(lob, "LOB_DIR", root),
                patch.object(lob, "SNAPSHOT_DIR", root / "snapshots"),
                patch.object(lob, "BOOK_TICKER_DIR", root / "book_ticker"),
                patch.object(lob, "AGG_TRADES_DIR", root / "agg_trades"),
                patch.object(lob, "FEATURES_DIR", root / "features"),
                patch.object(
                    lob,
                    "fetch_depth_snapshot",
                    return_value={"lastUpdateId": 1, "E": 1776002400000, "T": 1776002400000, "bids": [["100", "5"]], "asks": [["101", "4"]]},
                ),
                patch.object(
                    lob,
                    "fetch_book_ticker",
                    return_value={"bidPrice": "100", "bidQty": "5", "askPrice": "101", "askQty": "4", "time": 1776002400000},
                ),
                patch.object(
                    lob,
                    "fetch_agg_trades",
                    return_value=[{"a": 1, "p": "100.5", "q": "2", "m": False, "T": 1776002400000}],
                ),
            ):
                row = lob.collect_symbol("BTCUSDT", depth_limit=20, agg_limit=200)
            self.assertEqual(row["symbol"], "BTCUSDT")
            self.assertTrue((root / "features" / "BTCUSDT_microstructure.csv").exists())
            self.assertTrue((root / "snapshots" / "BTCUSDT_depth_top20.jsonl").exists())
            trades = pd.read_csv(root / "agg_trades" / "BTCUSDT.csv")
            self.assertEqual(len(trades), 1)
            payload = json.loads((root / "snapshots" / "BTCUSDT_depth_top20.jsonl").read_text().strip())
            self.assertEqual(payload["symbol"], "BTCUSDT")


if __name__ == "__main__":
    unittest.main()
