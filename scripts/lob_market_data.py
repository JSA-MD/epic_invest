#!/usr/bin/env python3
"""Collect Binance futures LOB snapshots and derive microstructure features."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
LOB_DIR = ROOT_DIR / "data" / "lob" / "binance_futures"
SNAPSHOT_DIR = LOB_DIR / "snapshots"
BOOK_TICKER_DIR = LOB_DIR / "book_ticker"
AGG_TRADES_DIR = LOB_DIR / "agg_trades"
FEATURES_DIR = LOB_DIR / "features"
API_BASE = "https://fapi.binance.com"
UTC = timezone.utc
DEFAULT_SYMBOLS = ("BTCUSDT", "BNBUSDT")


def iso_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Binance futures LOB snapshots and derived features.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_once = sub.add_parser("run-once", help="Collect one LOB sample.")
    run_once.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    run_once.add_argument("--depth-limit", type=int, default=20)
    run_once.add_argument("--agg-limit", type=int, default=200)

    loop = sub.add_parser("loop", help="Collect LOB samples repeatedly.")
    loop.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    loop.add_argument("--depth-limit", type=int, default=20)
    loop.add_argument("--agg-limit", type=int, default=200)
    loop.add_argument("--interval-seconds", type=int, default=30)
    return parser.parse_args()


def _http_get_json(path: str, params: dict[str, Any]) -> Any:
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        response = requests.get(f"{API_BASE}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception:
        completed = subprocess.run(
            ["curl", "-sS", "--max-time", "20", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)


def fetch_depth_snapshot(symbol: str, *, limit: int) -> dict[str, Any]:
    return _http_get_json("/fapi/v1/depth", {"symbol": symbol, "limit": int(limit)})


def fetch_book_ticker(symbol: str) -> dict[str, Any]:
    return _http_get_json("/fapi/v1/ticker/bookTicker", {"symbol": symbol})


def fetch_agg_trades(symbol: str, *, limit: int) -> list[dict[str, Any]]:
    payload = _http_get_json("/fapi/v1/aggTrades", {"symbol": symbol, "limit": int(limit)})
    if isinstance(payload, list):
        return payload
    raise RuntimeError(f"Unexpected aggTrades payload for {symbol}: {payload}")


def _levels_to_frame(levels: list[list[str]]) -> pd.DataFrame:
    frame = pd.DataFrame(levels, columns=["price", "qty"])
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["qty"] = pd.to_numeric(frame["qty"], errors="coerce")
    return frame.dropna()


def compute_microstructure_features(
    symbol: str,
    *,
    collected_at: str,
    depth_snapshot: dict[str, Any],
    book_ticker: dict[str, Any],
    agg_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    bids = _levels_to_frame(depth_snapshot.get("bids") or [])
    asks = _levels_to_frame(depth_snapshot.get("asks") or [])
    if bids.empty or asks.empty:
        raise ValueError(f"LOB snapshot for {symbol} is empty")

    bid_price = float(pd.to_numeric(book_ticker.get("bidPrice"), errors="coerce"))
    ask_price = float(pd.to_numeric(book_ticker.get("askPrice"), errors="coerce"))
    bid_qty = float(pd.to_numeric(book_ticker.get("bidQty"), errors="coerce"))
    ask_qty = float(pd.to_numeric(book_ticker.get("askQty"), errors="coerce"))
    mid = (bid_price + ask_price) / 2.0
    spread = ask_price - bid_price
    spread_bps = (spread / mid) * 10_000.0 if mid > 0 else float("nan")
    qty_sum = bid_qty + ask_qty
    microprice = ((ask_price * bid_qty) + (bid_price * ask_qty)) / qty_sum if qty_sum > 0 else mid

    def depth_imbalance(levels: int) -> float:
        bid_depth = float(bids.head(levels)["qty"].sum())
        ask_depth = float(asks.head(levels)["qty"].sum())
        total = bid_depth + ask_depth
        return (bid_depth - ask_depth) / total if total > 0 else 0.0

    taker_buy = 0.0
    taker_sell = 0.0
    latest_trade_ts = None
    for trade in agg_trades:
        qty = float(pd.to_numeric(trade.get("q"), errors="coerce"))
        latest_trade_ts = trade.get("T") or latest_trade_ts
        if bool(trade.get("m")):
            taker_sell += qty
        else:
            taker_buy += qty
    total_taker = taker_buy + taker_sell
    buy_share = taker_buy / total_taker if total_taker > 0 else 0.5
    ofi = 2.0 * buy_share - 1.0

    return {
        "collected_at": collected_at,
        "symbol": symbol,
        "last_update_id": int(depth_snapshot.get("lastUpdateId", 0) or 0),
        "book_event_time": int(book_ticker.get("time", 0) or 0),
        "latest_trade_time": int(latest_trade_ts or 0),
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "mid_price": mid,
        "spread": spread,
        "spread_bps": spread_bps,
        "microprice": microprice,
        "depth_imbalance_5": depth_imbalance(5),
        "depth_imbalance_10": depth_imbalance(10),
        "depth_imbalance_20": depth_imbalance(20),
        "agg_trade_buy_share": buy_share,
        "agg_trade_ofi": ofi,
        "agg_trade_count": int(len(agg_trades)),
        "agg_trade_buy_qty": taker_buy,
        "agg_trade_sell_qty": taker_sell,
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _merge_trade_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return
    if path.exists():
        existing = pd.read_csv(path)
        merged = pd.concat([existing, incoming], axis=0, ignore_index=True)
    else:
        merged = incoming
    if "a" in merged.columns:
        merged = merged.drop_duplicates(subset=["a"], keep="last")
    elif "T" in merged.columns:
        merged = merged.drop_duplicates(subset=["T", "p", "q"], keep="last")
    merged = merged.sort_values(["T", "a"] if "a" in merged.columns else ["T"])
    merged.to_csv(path, index=False)


def _append_feature_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        merged = pd.concat([existing, frame], axis=0, ignore_index=True)
        merged = merged.drop_duplicates(subset=["collected_at"], keep="last").sort_values("collected_at")
    else:
        merged = frame
    merged.to_csv(path, index=False)


def collect_symbol(symbol: str, *, depth_limit: int, agg_limit: int) -> dict[str, Any]:
    collected_at = iso_now()
    depth = fetch_depth_snapshot(symbol, limit=depth_limit)
    ticker = fetch_book_ticker(symbol)
    trades = fetch_agg_trades(symbol, limit=agg_limit)

    feature_row = compute_microstructure_features(
        symbol,
        collected_at=collected_at,
        depth_snapshot=depth,
        book_ticker=ticker,
        agg_trades=trades,
    )

    depth_payload = {
        "collected_at": collected_at,
        "symbol": symbol,
        "lastUpdateId": depth.get("lastUpdateId"),
        "event_time": depth.get("E"),
        "transaction_time": depth.get("T"),
        "bids": depth.get("bids"),
        "asks": depth.get("asks"),
    }
    _append_jsonl(SNAPSHOT_DIR / f"{symbol}_depth_top20.jsonl", depth_payload)

    ticker_row = {
        "collected_at": collected_at,
        "symbol": symbol,
        "bidPrice": ticker.get("bidPrice"),
        "bidQty": ticker.get("bidQty"),
        "askPrice": ticker.get("askPrice"),
        "askQty": ticker.get("askQty"),
        "time": ticker.get("time"),
    }
    _append_feature_row(BOOK_TICKER_DIR / f"{symbol}.csv", ticker_row)
    _merge_trade_rows(AGG_TRADES_DIR / f"{symbol}.csv", trades)
    _append_feature_row(FEATURES_DIR / f"{symbol}_microstructure.csv", feature_row)
    return feature_row


def run_once(symbols: tuple[str, ...], *, depth_limit: int, agg_limit: int) -> list[dict[str, Any]]:
    results = []
    for symbol in symbols:
        results.append(collect_symbol(symbol, depth_limit=depth_limit, agg_limit=agg_limit))
    return results


def run_loop(symbols: tuple[str, ...], *, depth_limit: int, agg_limit: int, interval_seconds: int) -> None:
    while True:
        payload = {
            "generated_at": iso_now(),
            "rows": run_once(symbols, depth_limit=depth_limit, agg_limit=agg_limit),
        }
        print(json.dumps(payload, ensure_ascii=False))
        time.sleep(max(5, int(interval_seconds)))


def main() -> None:
    args = parse_args()
    if args.command == "run-once":
        rows = run_once(tuple(args.symbols), depth_limit=int(args.depth_limit), agg_limit=int(args.agg_limit))
        print(json.dumps({"generated_at": iso_now(), "rows": rows}, ensure_ascii=False))
        return
    run_loop(
        tuple(args.symbols),
        depth_limit=int(args.depth_limit),
        agg_limit=int(args.agg_limit),
        interval_seconds=int(args.interval_seconds),
    )


if __name__ == "__main__":
    main()
