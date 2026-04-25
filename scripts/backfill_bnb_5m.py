#!/usr/bin/env python3
"""One-off backfill: fetch BNBUSDT 5m candles from Binance and load into candles_bnb."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gp_crypto_evolution as gp  # noqa: E402


def main() -> None:
    symbol = "BNBUSDT"
    interval = "5m"
    start = datetime(2026, 4, 5, 16, 10, tzinfo=timezone.utc)
    end = datetime.now(tz=timezone.utc)
    print(f"Fetching {symbol} {interval} from {start.isoformat()} to {end.isoformat()}")
    df = gp.fetch_klines(symbol, interval, start, end)
    if df.empty:
        print("No rows fetched; exiting")
        return
    df = df.copy()
    df["timestamp"] = df.index
    needed = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[needed]
    print(f"Fetched {len(df)} rows; first={df['timestamp'].iloc[0]}, last={df['timestamp'].iloc[-1]}")

    csv_path = Path("/tmp/_bnb_5m_backfill.csv")
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
    print(f"CSV written to {csv_path}, {csv_path.stat().st_size} bytes")

    container_path = "/tmp/_bnb_5m_backfill.csv"
    subprocess.run(
        ["docker", "cp", str(csv_path), f"epic_trading_db:{container_path}"],
        check=True,
    )

    sql = (
        "BEGIN;\n"
        "CREATE TEMP TABLE tmp_bnb_load (timestamp timestamptz, timeframe text, "
        "open double precision, high double precision, low double precision, close double precision, "
        "volume double precision, symbol text);\n"
        f"COPY tmp_bnb_load FROM '{container_path}' WITH (FORMAT csv);\n"
        "INSERT INTO candles_bnb (timestamp, timeframe, open, high, low, close, volume, symbol) "
        "SELECT timestamp, timeframe, open, high, low, close, volume, symbol FROM tmp_bnb_load "
        "ON CONFLICT (timestamp, timeframe) DO NOTHING;\n"
        "SELECT MIN(timestamp) AS min_t, MAX(timestamp) AS max_t, COUNT(*) AS n "
        "FROM candles_bnb WHERE timeframe='5m' AND symbol='BNBUSDT';\n"
        "COMMIT;\n"
    )
    proc = subprocess.run(
        ["docker", "exec", "-i", "epic_trading_db", "psql", "-U", "epic", "-d", "epic_trading", "-v", "ON_ERROR_STOP=1"],
        input=sql,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print("STDOUT:", proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr)
        sys.exit(proc.returncode)
    subprocess.run(["docker", "exec", "epic_trading_db", "rm", "-f", container_path], check=False)


if __name__ == "__main__":
    main()
