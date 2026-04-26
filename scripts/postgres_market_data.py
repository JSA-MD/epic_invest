"""PostgreSQL-backed Binance market data loader.

The project historically used CSV caches for backtests. This module keeps the
same dataframe contract while allowing the shared GP loaders to read candles
and funding rates from the local Timescale/PostgreSQL Docker containers.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd


RAW_CACHE_COLUMNS = ("open", "high", "low", "close", "volume", "taker_base", "taker_quote")
DEFAULT_SYMBOL_CANDLE_TABLES = {
    "BTCUSDT": "candles",
    "BNBUSDT": "candles_bnb",
    "ETHUSDT": "candles_eth",
    "SOLUSDT": "candles_sol",
    "XRPUSDT": "candles_xrp",
    "DOGEUSDT": "candles_doge",
}
POSTGRES_SOURCE_VALUES = {"postgres", "postgresql", "pg"}
AUTO_SOURCE_VALUES = {"auto", "postgres-auto", "pg-auto"}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9_:-]+$")
_TIMEFRAME_RE = re.compile(r"^[0-9]+[mhdwM]$")
_DOCKER_CANDIDATES = (
    "/usr/local/bin/docker",
    "/opt/homebrew/bin/docker",
    "/Applications/Docker.app/Contents/Resources/bin/docker",
)


@dataclass(frozen=True)
class PostgresMarketDataConfig:
    user: str = "epic"
    dbname: str = "epic_trading"
    container: str = "epic_trading_db"
    schema: str = "public"
    candles_table: str | None = None
    funding_table: str = "funding_rates"

    @classmethod
    def from_env(cls) -> "PostgresMarketDataConfig":
        return cls(
            user=os.getenv("EPIC_POSTGRES_USER", os.getenv("POSTGRES_USER", cls.user)),
            dbname=os.getenv("EPIC_POSTGRES_DB", os.getenv("POSTGRES_DB", cls.dbname)),
            container=os.getenv("EPIC_POSTGRES_CONTAINER", cls.container),
            schema=os.getenv("EPIC_POSTGRES_SCHEMA", cls.schema),
            candles_table=os.getenv("EPIC_POSTGRES_CANDLES_TABLE") or None,
            funding_table=os.getenv("EPIC_POSTGRES_FUNDING_TABLE", cls.funding_table),
        )


def market_data_source() -> str:
    return os.getenv("EPIC_MARKET_DATA_SOURCE", os.getenv("MARKET_DATA_SOURCE", "csv")).strip().lower()


def postgres_source_enabled() -> bool:
    return market_data_source() in POSTGRES_SOURCE_VALUES


def postgres_auto_enabled() -> bool:
    return market_data_source() in AUTO_SOURCE_VALUES


def _validate_identifier(value: str, *, label: str) -> str:
    if not _IDENT_RE.match(value):
        raise ValueError(f"Invalid PostgreSQL {label}: {value!r}")
    return value


def _qualified_table(schema: str, table: str) -> str:
    return f"{_validate_identifier(schema, label='schema')}.{_validate_identifier(table, label='table')}"


def _validate_symbol(symbol: str) -> str:
    symbol = symbol.upper()
    if not _SYMBOL_RE.match(symbol):
        raise ValueError(f"Invalid symbol: {symbol!r}")
    return symbol


def _validate_timeframe(interval: str) -> str:
    if not _TIMEFRAME_RE.match(interval):
        raise ValueError(f"Invalid timeframe: {interval!r}")
    return interval


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _normalize_boundary(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.isoformat()


def _run_psql_csv(sql: str, config: PostgresMarketDataConfig) -> str:
    copy_sql = f"COPY ({sql}) TO STDOUT WITH CSV HEADER"
    docker_bin = os.getenv("EPIC_DOCKER_BIN") or shutil.which("docker")
    if not docker_bin:
        docker_bin = next((path for path in _DOCKER_CANDIDATES if os.path.exists(path)), "docker")
    proc = subprocess.run(
        [
            docker_bin,
            "exec",
            "-i",
            config.container,
            "psql",
            "-U",
            config.user,
            "-d",
            config.dbname,
            "-X",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            copy_sql,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "psql query failed")
    return proc.stdout


def _read_csv_frame(csv_text: str, time_column: str) -> pd.DataFrame:
    if not csv_text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(csv_text))
    if df.empty:
        return df
    df[time_column] = pd.to_datetime(df[time_column], utc=True, format="mixed")
    return df


def _interval_minutes(interval: str) -> int | None:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 1440
    return None


def _filter_candle_quality(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out[
        (out["open"] > 0.0)
        & (out["high"] > 0.0)
        & (out["low"] > 0.0)
        & (out["close"] > 0.0)
        & (out["high"] >= out[["open", "close", "low"]].max(axis=1))
        & (out["low"] <= out[["open", "close", "high"]].min(axis=1))
    ]
    minutes = _interval_minutes(interval)
    if minutes is not None:
        index = pd.DatetimeIndex(out.index)
        aligned = (index.second == 0) & (index.microsecond == 0) & (index.nanosecond == 0)
        if minutes < 60:
            aligned &= (index.minute % minutes) == 0
        elif minutes < 1440:
            hours = max(1, minutes // 60)
            aligned &= (index.minute == 0) & ((index.hour % hours) == 0)
        else:
            aligned &= (index.minute == 0) & (index.hour == 0)
        out = out.loc[aligned]
    if len(out) < 20:
        return out
    close = out["close"].astype("float64")
    bars_per_day = 288 if interval == "5m" else max(24, min(288, len(out) // 4))
    window = max(24, min(int(bars_per_day), len(out)))
    rolling_median = close.rolling(window, center=True, min_periods=max(12, min(window // 4, 48))).median()
    rolling_median = rolling_median.fillna(close.median())
    ratio = close / rolling_median.replace(0.0, np.nan)
    return out.loc[ratio.between(0.20, 5.0).fillna(False)]


def _candle_table_for(symbol: str, config: PostgresMarketDataConfig) -> str:
    if config.candles_table:
        return config.candles_table
    return DEFAULT_SYMBOL_CANDLE_TABLES.get(symbol.upper(), "candles")


def _build_candle_sql(
    table_name: str,
    *,
    config: PostgresMarketDataConfig,
    symbol: str,
    interval: str,
    start: str | datetime | None,
    end: str | datetime | None,
) -> str:
    clauses = [
        f"symbol = {_sql_literal(symbol)}",
        f"timeframe = {_sql_literal(interval)}",
    ]
    start_ts = _normalize_boundary(start)
    end_ts = _normalize_boundary(end)
    if start_ts is not None:
        clauses.append(f"timestamp >= {_sql_literal(start_ts)}::timestamptz")
    if end_ts is not None:
        clauses.append(f"timestamp <= {_sql_literal(end_ts)}::timestamptz")
    where_sql = " AND ".join(clauses)
    table_sql = _qualified_table(config.schema, table_name)
    return f"""
        SELECT
            timestamp AS open_time,
            open,
            high,
            low,
            close,
            volume,
            volume * 0.5 AS taker_base,
            volume * close * 0.5 AS taker_quote
        FROM {table_sql}
        WHERE {where_sql}
        ORDER BY timestamp
    """


def load_ohlcv(
    symbol: str,
    interval: str,
    start: str | datetime | None,
    end: str | datetime | None,
    *,
    config: PostgresMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Load OHLCV from PostgreSQL and return the raw cache dataframe shape."""
    config = config or PostgresMarketDataConfig.from_env()
    symbol = _validate_symbol(symbol)
    interval = _validate_timeframe(interval)
    primary_table = _candle_table_for(symbol, config)
    candidate_tables = [primary_table]
    if primary_table != "candles":
        candidate_tables.append("candles")

    last_error: Exception | None = None
    saw_successful_query = False
    for table_name in candidate_tables:
        try:
            csv_text = _run_psql_csv(
                _build_candle_sql(table_name, config=config, symbol=symbol, interval=interval, start=start, end=end),
                config,
            )
        except Exception as exc:
            last_error = exc
            continue
        saw_successful_query = True
        df = _read_csv_frame(csv_text, "open_time")
        if df.empty:
            continue
        for column in RAW_CACHE_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = (
            df.dropna(subset=["open_time", "open", "high", "low", "close"])
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .set_index("open_time")
        )
        df["volume"] = df["volume"].fillna(0.0)
        df["taker_base"] = df["taker_base"].fillna(df["volume"] * 0.5)
        df["taker_quote"] = df["taker_quote"].fillna(df["taker_base"] * df["close"])
        df = _filter_candle_quality(df, interval)
        return df[list(RAW_CACHE_COLUMNS)]

    if last_error is not None and not saw_successful_query:
        raise RuntimeError(f"PostgreSQL OHLCV load failed for {symbol} {interval}: {last_error}") from last_error
    return pd.DataFrame(columns=list(RAW_CACHE_COLUMNS))


def _build_funding_sql(
    table_name: str,
    *,
    config: PostgresMarketDataConfig,
    symbol: str,
    start: str | datetime | None,
    end: str | datetime | None,
) -> str:
    rate_column = "funding_rate" if table_name == "funding_rates" else "rate"
    clauses = [f"symbol = {_sql_literal(symbol)}"]
    start_ts = _normalize_boundary(start)
    end_ts = _normalize_boundary(end)
    if start_ts is not None:
        clauses.append(f"timestamp >= {_sql_literal(start_ts)}::timestamptz")
    if end_ts is not None:
        clauses.append(f"timestamp <= {_sql_literal(end_ts)}::timestamptz")
    where_sql = " AND ".join(clauses)
    table_sql = _qualified_table(config.schema, table_name)
    return f"""
        SELECT
            timestamp AS "fundingTime",
            {rate_column} AS "fundingRate"
        FROM {table_sql}
        WHERE {where_sql}
        ORDER BY timestamp
    """


def load_funding_rates(
    symbol: str,
    start: str | datetime | None,
    end: str | datetime | None,
    *,
    config: PostgresMarketDataConfig | None = None,
) -> pd.DataFrame:
    """Load funding rates from PostgreSQL in the backtester column shape."""
    config = config or PostgresMarketDataConfig.from_env()
    symbol = _validate_symbol(symbol)
    candidate_tables = [config.funding_table]
    if config.funding_table != "funding_rate_snapshots":
        candidate_tables.append("funding_rate_snapshots")

    last_error: Exception | None = None
    saw_successful_query = False
    for table_name in candidate_tables:
        try:
            csv_text = _run_psql_csv(
                _build_funding_sql(table_name, config=config, symbol=symbol, start=start, end=end),
                config,
            )
        except Exception as exc:
            last_error = exc
            continue
        saw_successful_query = True
        df = _read_csv_frame(csv_text, "fundingTime")
        if df.empty:
            continue
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return (
            df.dropna(subset=["fundingTime", "fundingRate"])
            .drop_duplicates(subset=["fundingTime"])
            .sort_values("fundingTime")
            .reset_index(drop=True)
        )[["fundingTime", "fundingRate"]]

    if last_error is not None and not saw_successful_query:
        raise RuntimeError(f"PostgreSQL funding load failed for {symbol}: {last_error}") from last_error
    return pd.DataFrame(columns=["fundingTime", "fundingRate"])
