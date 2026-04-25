#!/usr/bin/env python3
"""Backfill funding_rates table from CSV caches for BTCUSDT and BNBUSDT.

The DB previously had only ~100 rows of BTC funding (one month) which the
backtest treated as authoritative, ignoring the 4-year CSV cache. This script
loads the full CSV history into the DB so DB queries return complete data.

Missing markPrice in CSV is filled with 0.0 (backtest does not use it).
"""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "binance_futures"


def load_csv(symbol: str) -> pd.DataFrame:
    pattern = f"{symbol}_funding_2022-04-06_*.csv"
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No funding CSV for {symbol} matching {pattern}")
    latest = files[-1]
    print(f"  Loading {latest.name}")
    df = pd.read_csv(latest)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, format="mixed")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["markPrice"] = pd.to_numeric(df.get("markPrice"), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["fundingTime", "fundingRate"]).sort_values("fundingTime")
    df = df.drop_duplicates(subset=["fundingTime"])
    return df


def backfill_symbol(symbol: str) -> None:
    print(f"Backfilling {symbol}")
    df = load_csv(symbol)
    print(f"  {len(df)} rows; {df['fundingTime'].min()} -> {df['fundingTime'].max()}")

    csv_path = Path(f"/tmp/_funding_{symbol}_backfill.csv")
    with csv_path.open("w") as fh:
        writer = csv.writer(fh)
        for _, row in df.iterrows():
            ts = row["fundingTime"].strftime("%Y-%m-%d %H:%M:%S.%f+00")
            writer.writerow([ts, symbol, float(row["fundingRate"]), float(row["markPrice"])])

    container_path = f"/tmp/_funding_{symbol}_backfill.csv"
    subprocess.run(
        ["docker", "cp", str(csv_path), f"epic_trading_db:{container_path}"],
        check=True,
    )

    sql = (
        "BEGIN;\n"
        "CREATE TEMP TABLE tmp_funding_load (timestamp timestamptz, symbol text, "
        "funding_rate double precision, mark_price double precision);\n"
        f"COPY tmp_funding_load FROM '{container_path}' WITH (FORMAT csv);\n"
        "INSERT INTO funding_rates (timestamp, symbol, funding_rate, mark_price) "
        "SELECT timestamp, symbol, funding_rate, mark_price FROM tmp_funding_load "
        "ON CONFLICT (timestamp, symbol) DO UPDATE SET "
        "funding_rate = EXCLUDED.funding_rate, "
        "mark_price = CASE WHEN EXCLUDED.mark_price > 0.0 THEN EXCLUDED.mark_price "
        "ELSE funding_rates.mark_price END;\n"
        f"SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM funding_rates WHERE symbol = '{symbol}';\n"
        "COMMIT;\n"
    )
    proc = subprocess.run(
        ["docker", "exec", "-i", "epic_trading_db", "psql", "-U", "epic", "-d", "epic_trading", "-v", "ON_ERROR_STOP=1"],
        input=sql,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr)
        sys.exit(proc.returncode)
    subprocess.run(["docker", "exec", "epic_trading_db", "rm", "-f", container_path], check=False)


def main() -> None:
    for symbol in ("BTCUSDT", "BNBUSDT"):
        backfill_symbol(symbol)


if __name__ == "__main__":
    main()
