#!/usr/bin/env python3
"""Idempotent backfill: fetch 5m candles + funding rates for ETH/SOL/XRP/DOGE.

Checks max(timestamp) per table; only fetches rows newer than what is already
stored. Safe to re-run — uses ON CONFLICT DO NOTHING for candles and
ON CONFLICT DO UPDATE for funding (to refresh mark_price).

Usage:
    python scripts/backfill_multi_pair_5m.py [--symbols ETHUSDT,SOLUSDT,...]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gp_crypto_evolution as gp  # noqa: E402

UTC = timezone.utc

# Map symbol -> (candle_table, timeframe)
SYMBOL_TABLE_MAP: dict[str, str] = {
    "ETHUSDT":  "candles_eth",
    "SOLUSDT":  "candles_sol",
    "XRPUSDT":  "candles_xrp",
    "DOGEUSDT": "candles_doge",
}

# Funding: Binance publishes every 8h; fetch from this start for new symbols
FUNDING_FETCH_START = "2022-04-06"
FUNDING_API_BASE = "https://fapi.binance.com"
CONTAINER = "epic_trading_db"
PSQL_CMD = ["docker", "exec", "-i", CONTAINER, "psql", "-U", "epic", "-d", "epic_trading", "-v", "ON_ERROR_STOP=1"]


def _run_psql(sql: str, *, timeout: int = 300) -> None:
    proc = subprocess.run(
        PSQL_CMD,
        input=sql,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print("STDERR:", proc.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"psql failed (rc={proc.returncode})")


def _query_psql(sql: str) -> str:
    """Run a psql query and return stdout."""
    proc = subprocess.run(
        ["docker", "exec", "-i", CONTAINER, "psql", "-U", "epic", "-d", "epic_trading",
         "-t", "-A", "-v", "ON_ERROR_STOP=1"],
        input=sql,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.strip()


def _get_candle_max_ts(table: str) -> datetime | None:
    """Return max(timestamp) for 5m candles in table, or None if empty."""
    try:
        result = _query_psql(f"SELECT MAX(timestamp) FROM {table} WHERE timeframe='5m';")
        if result and result.lower() not in ("", "null"):
            return datetime.fromisoformat(result.replace("+00", "+00:00")).replace(tzinfo=UTC)
    except Exception as exc:
        print(f"  WARNING: could not query {table}: {exc}", file=sys.stderr)
    return None


def _get_funding_max_ts(symbol: str) -> datetime | None:
    """Return max(timestamp) for symbol in funding_rates, or None."""
    try:
        result = _query_psql(f"SELECT MAX(timestamp) FROM funding_rates WHERE symbol='{symbol}';")
        if result and result.lower() not in ("", "null"):
            return datetime.fromisoformat(result.replace("+00", "+00:00")).replace(tzinfo=UTC)
    except Exception as exc:
        print(f"  WARNING: could not query funding_rates for {symbol}: {exc}", file=sys.stderr)
    return None


def backfill_candles(symbol: str, table: str, interval: str = "5m") -> None:
    """Fetch and insert 5m candles for symbol into table, from after max existing ts."""
    max_ts = _get_candle_max_ts(table)
    now = datetime.now(tz=UTC)
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)  # today 00:00 UTC

    if max_ts is not None and max_ts >= cutoff:
        print(f"  {symbol}: candles up to date (max={max_ts.strftime('%Y-%m-%d %H:%M')} UTC), skipping")
        return

    if max_ts is not None:
        # Start from the bar after the last stored one
        start = max_ts
        print(f"  {symbol}: fetching candles from {start.strftime('%Y-%m-%d %H:%M')} UTC → now")
    else:
        start = datetime(2022, 4, 6, tzinfo=UTC)
        print(f"  {symbol}: no existing candles; fetching full history from {start.date()}")

    df = gp.fetch_klines(symbol, interval, start, now)
    if df.empty:
        print(f"  {symbol}: no rows fetched; skipping")
        return

    df = df.copy()
    df["timestamp"] = df.index
    needed = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[needed]
    print(f"  {symbol}: fetched {len(df)} rows; "
          f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} → {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")

    csv_path = Path(f"/tmp/_backfill_{symbol}_5m.csv")
    container_path = f"/tmp/_backfill_{symbol}_5m.csv"
    with csv_path.open("w") as fh:
        writer = csv.writer(fh)
        for _, row in df.iterrows():
            writer.writerow([
                row["timestamp"].strftime("%Y-%m-%d %H:%M:%S+00"),
                interval,
                row["open"], row["high"], row["low"], row["close"],
                row["volume"],
                symbol,
            ])

    subprocess.run(
        ["docker", "cp", str(csv_path), f"{CONTAINER}:{container_path}"],
        check=True,
    )

    sql = (
        "BEGIN;\n"
        f"CREATE TEMP TABLE tmp_candle_load_{symbol.lower()} "
        "(timestamp timestamptz, timeframe text, "
        "open double precision, high double precision, low double precision, "
        "close double precision, volume double precision, symbol text);\n"
        f"COPY tmp_candle_load_{symbol.lower()} FROM '{container_path}' WITH (FORMAT csv);\n"
        f"INSERT INTO {table} (timestamp, timeframe, open, high, low, close, volume, symbol) "
        f"SELECT timestamp, timeframe, open, high, low, close, volume, symbol "
        f"FROM tmp_candle_load_{symbol.lower()} "
        "ON CONFLICT (timestamp, timeframe) DO NOTHING;\n"
        f"SELECT MIN(timestamp) AS min_t, MAX(timestamp) AS max_t, COUNT(*) AS n "
        f"FROM {table} WHERE timeframe='5m';\n"
        "COMMIT;\n"
    )
    _run_psql(sql)
    subprocess.run(["docker", "exec", CONTAINER, "rm", "-f", container_path], check=False)
    csv_path.unlink(missing_ok=True)


def _fetch_funding_from_api(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch funding rate history from Binance futures API."""
    import requests  # type: ignore
    import time

    url = f"{FUNDING_API_BASE}/fapi/v1/fundingRate"
    results: list[dict] = []
    cur = start_ms
    while cur < end_ms:
        params = {"symbol": symbol, "startTime": cur, "endTime": end_ms, "limit": 1000}
        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(min(30, 2 * (attempt + 1)))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                if attempt < 3:
                    time.sleep(1)
                    continue
                print(f"  WARNING: funding API error for {symbol}: {exc}", file=sys.stderr)
                data = []
                break
        else:
            data = []

        if not data:
            break
        results.extend(data)
        last_ts = data[-1]["fundingTime"]
        if last_ts <= cur or len(data) < 1000:
            break
        cur = last_ts + 1

    return results


def backfill_funding(symbol: str) -> None:
    """Fetch and insert funding rates for symbol into funding_rates table."""
    max_ts = _get_funding_max_ts(symbol)
    now = datetime.now(tz=UTC)
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if max_ts is not None and max_ts >= cutoff:
        print(f"  {symbol}: funding up to date (max={max_ts.strftime('%Y-%m-%d %H:%M')} UTC), skipping")
        return

    if max_ts is not None:
        start_dt = max_ts
        print(f"  {symbol}: fetching funding from {start_dt.strftime('%Y-%m-%d %H:%M')} UTC → now")
    else:
        start_dt = datetime(2022, 4, 6, tzinfo=UTC)
        print(f"  {symbol}: no existing funding; fetching full history from {start_dt.date()}")

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    rows = _fetch_funding_from_api(symbol, start_ms, end_ms)
    if not rows:
        print(f"  {symbol}: no funding rows fetched; skipping")
        return

    print(f"  {symbol}: fetched {len(rows)} funding rows")

    csv_path = Path(f"/tmp/_backfill_{symbol}_funding.csv")
    container_path = f"/tmp/_backfill_{symbol}_funding.csv"
    with csv_path.open("w") as fh:
        writer = csv.writer(fh)
        for row in rows:
            ts_ms = int(row["fundingTime"])
            ts_dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
            ts_str = ts_dt.strftime("%Y-%m-%d %H:%M:%S.%f+00")
            rate_raw = row.get("fundingRate", 0.0)
            mark_raw = row.get("markPrice", 0.0)
            rate = float(rate_raw) if rate_raw != "" else 0.0
            mark = float(mark_raw) if mark_raw != "" else 0.0
            writer.writerow([ts_str, symbol, rate, mark])

    subprocess.run(
        ["docker", "cp", str(csv_path), f"{CONTAINER}:{container_path}"],
        check=True,
    )

    sql = (
        "BEGIN;\n"
        f"CREATE TEMP TABLE tmp_funding_load_{symbol.lower()} "
        "(timestamp timestamptz, symbol text, funding_rate double precision, mark_price double precision);\n"
        f"COPY tmp_funding_load_{symbol.lower()} FROM '{container_path}' WITH (FORMAT csv);\n"
        "INSERT INTO funding_rates (timestamp, symbol, funding_rate, mark_price) "
        f"SELECT timestamp, symbol, funding_rate, mark_price FROM tmp_funding_load_{symbol.lower()} "
        "ON CONFLICT (timestamp, symbol) DO UPDATE SET "
        "funding_rate = EXCLUDED.funding_rate, "
        "mark_price = CASE WHEN EXCLUDED.mark_price > 0.0 THEN EXCLUDED.mark_price "
        "ELSE funding_rates.mark_price END;\n"
        f"SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM funding_rates WHERE symbol='{symbol}';\n"
        "COMMIT;\n"
    )
    _run_psql(sql)
    subprocess.run(["docker", "exec", CONTAINER, "rm", "-f", container_path], check=False)
    csv_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Idempotent 5m candle + funding backfill for multi-pair expansion."
    )
    parser.add_argument(
        "--symbols",
        default=",".join(SYMBOL_TABLE_MAP.keys()),
        help="Comma-separated Binance futures symbols to backfill (default: all new pairs).",
    )
    parser.add_argument(
        "--skip-funding",
        action="store_true",
        help="Skip funding rate backfill.",
    )
    parser.add_argument(
        "--skip-candles",
        action="store_true",
        help="Skip candle backfill.",
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"Backfilling {len(symbols)} symbol(s): {symbols}")

    for symbol in symbols:
        print(f"\n--- {symbol} ---")

        if not args.skip_candles:
            table = SYMBOL_TABLE_MAP.get(symbol)
            if table is None:
                print(f"  {symbol}: no candle table mapping; skipping candles")
            else:
                try:
                    backfill_candles(symbol, table)
                except Exception as exc:
                    print(f"  {symbol}: candle backfill FAILED: {exc}", file=sys.stderr)

        if not args.skip_funding:
            try:
                backfill_funding(symbol)
            except Exception as exc:
                print(f"  {symbol}: funding backfill FAILED: {exc}", file=sys.stderr)

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
