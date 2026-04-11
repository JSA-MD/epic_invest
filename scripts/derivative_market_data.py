#!/usr/bin/env python3
"""Fetch, cache, and slice Binance derivatives market-data side channels."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pandas.api.types import is_datetime64_any_dtype

import gp_crypto_evolution as gp


API_BASE = "https://fapi.binance.com"
UTC = timezone.utc
DERIVATIVE_CACHE_DIR = gp.DATA_DIR / "derivatives"
DEFAULT_PERIOD = "5m"
MAX_RECENT_LOOKBACK_DAYS = 30
RATE_LIMIT_ERROR_CODE = -1003
MAX_RATE_LIMIT_RETRIES = 5
_PERIOD_UNITS_IN_MS = {
    "m": 60_000,
    "h": 3_600_000,
    "d": 86_400_000,
    "w": 604_800_000,
}

METRIC_SPECS: dict[str, dict[str, Any]] = {
    "open_interest": {
        "path": "/futures/data/openInterestHist",
        "timestamp_field": "timestamp",
        "columns": (
            ("sumOpenInterest", "open_interest"),
            ("sumOpenInterestValue", "open_interest_value"),
        ),
        "uses_pair": False,
        "needs_period": True,
        "default_limit": 500,
    },
    "top_trader_position_ratio": {
        "path": "/futures/data/topLongShortPositionRatio",
        "timestamp_field": "timestamp",
        "columns": (
            ("longShortRatio", "long_short_ratio"),
            ("longAccount", "long_account"),
            ("shortAccount", "short_account"),
        ),
        "uses_pair": False,
        "needs_period": True,
        "default_limit": 500,
    },
    "top_trader_account_ratio": {
        "path": "/futures/data/topLongShortAccountRatio",
        "timestamp_field": "timestamp",
        "columns": (
            ("longShortRatio", "long_short_ratio"),
            ("longAccount", "long_account"),
            ("shortAccount", "short_account"),
        ),
        "uses_pair": False,
        "needs_period": True,
        "default_limit": 500,
    },
    "global_long_short_ratio": {
        "path": "/futures/data/globalLongShortAccountRatio",
        "timestamp_field": "timestamp",
        "columns": (
            ("longShortRatio", "long_short_ratio"),
            ("longAccount", "long_account"),
            ("shortAccount", "short_account"),
        ),
        "uses_pair": False,
        "needs_period": True,
        "default_limit": 500,
    },
    "taker_buy_sell_ratio": {
        "path": "/futures/data/takerlongshortRatio",
        "timestamp_field": "timestamp",
        "columns": (
            ("buySellRatio", "buy_sell_ratio"),
            ("buyVol", "buy_volume"),
            ("sellVol", "sell_volume"),
        ),
        "uses_pair": False,
        "needs_period": True,
        "default_limit": 500,
    },
    "basis_perpetual": {
        "path": "/futures/data/basis",
        "timestamp_field": "timestamp",
        "columns": (
            ("basisRate", "basis_rate"),
            ("basis", "basis"),
            ("futuresPrice", "futures_price"),
            ("indexPrice", "index_price"),
            ("annualizedBasisRate", "annualized_basis_rate"),
        ),
        "uses_pair": True,
        "needs_period": True,
        "default_limit": 500,
        "extra_params": {"contractType": "PERPETUAL"},
    },
}


def derivative_cache_path(symbol: str, metric_key: str, period: str = DEFAULT_PERIOD) -> Path:
    return DERIVATIVE_CACHE_DIR / f"{symbol}_{metric_key}_{period}.csv"


def _period_to_milliseconds(period: str) -> int:
    raw = str(period).strip().lower()
    if len(raw) < 2 or raw[-1] not in _PERIOD_UNITS_IN_MS:
        raise ValueError(f"Unsupported Binance period: {period}")
    return int(raw[:-1]) * _PERIOD_UNITS_IN_MS[raw[-1]]


def _empty_metric_frame(metric_key: str) -> pd.DataFrame:
    columns = ["timestamp"] + [canonical for _, canonical in METRIC_SPECS[metric_key]["columns"]]
    return pd.DataFrame(columns=columns)


def _normalize_metric_frame(metric_key: str, frame: pd.DataFrame) -> pd.DataFrame:
    spec = METRIC_SPECS[metric_key]
    if frame.empty:
        return _empty_metric_frame(metric_key)

    timestamp_field = spec["timestamp_field"]
    if timestamp_field not in frame.columns:
        return _empty_metric_frame(metric_key)

    out = pd.DataFrame(index=frame.index)
    out[timestamp_field] = frame[timestamp_field]
    for source, canonical in spec["columns"]:
        if canonical in frame.columns:
            out[canonical] = frame[canonical]
        elif source in frame.columns:
            out[canonical] = frame[source]
        else:
            out[canonical] = pd.NA

    timestamp = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    raw_timestamp = out[timestamp_field]
    if is_datetime64_any_dtype(raw_timestamp):
        timestamp = pd.to_datetime(raw_timestamp, utc=True, errors="coerce")
    else:
        numeric_timestamp = pd.to_numeric(raw_timestamp, errors="coerce")
        numeric_mask = numeric_timestamp.notna()
        if numeric_mask.any():
            timestamp.loc[numeric_mask] = pd.to_datetime(
                numeric_timestamp.loc[numeric_mask],
                utc=True,
                unit="ms",
                errors="coerce",
            )
        if (~numeric_mask).any():
            timestamp.loc[~numeric_mask] = pd.to_datetime(
                raw_timestamp.loc[~numeric_mask],
                utc=True,
                errors="coerce",
                format="mixed",
            )
    out["timestamp"] = timestamp
    if timestamp_field != "timestamp":
        out = out.drop(columns=[timestamp_field], errors="ignore")
    for _, canonical in spec["columns"]:
        out[canonical] = pd.to_numeric(out[canonical], errors="coerce")
    out = (
        out.dropna(subset=["timestamp"])
        .loc[lambda df_: df_["timestamp"] >= pd.Timestamp("2017-01-01T00:00:00Z")]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return out[["timestamp", *[canonical for _, canonical in spec["columns"]]]]


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


def load_derivative_metric_cache(symbol: str, metric_key: str, period: str = DEFAULT_PERIOD) -> pd.DataFrame:
    path = derivative_cache_path(symbol, metric_key, period=period)
    if not path.exists():
        return _empty_metric_frame(metric_key)
    return _normalize_metric_frame(metric_key, pd.read_csv(path))


def fetch_derivative_metric(
    symbol: str,
    metric_key: str,
    *,
    start_dt: datetime,
    end_dt: datetime,
    period: str = DEFAULT_PERIOD,
    lookback_days: int = MAX_RECENT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    spec = METRIC_SPECS[metric_key]
    now_utc = datetime.now(tz=UTC)
    capped_end = min(end_dt.astimezone(UTC), now_utc)
    capped_start = max(start_dt.astimezone(UTC), capped_end - timedelta(days=lookback_days))
    if capped_start >= capped_end:
        return _empty_metric_frame(metric_key)

    rows_all: list[dict[str, Any]] = []
    cur = int(capped_start.timestamp() * 1000)
    end_ms = int(capped_end.timestamp() * 1000)
    page_span_ms = None
    if spec.get("needs_period", False):
        page_span_ms = _period_to_milliseconds(period) * max(int(spec.get("default_limit", 500)) - 1, 1)
    retry_count = 0
    while cur <= end_ms:
        request_end_ms = end_ms if page_span_ms is None else min(end_ms, cur + page_span_ms)
        params: dict[str, Any] = {
            "startTime": cur,
            "endTime": request_end_ms,
            "limit": int(spec.get("default_limit", 500)),
        }
        if spec.get("uses_pair", False):
            params["pair"] = symbol
        else:
            params["symbol"] = symbol
        if spec.get("needs_period", False):
            params["period"] = period
        params.update(spec.get("extra_params") or {})

        rows = _http_get_json(spec["path"], params)
        if not rows:
            break
        if not isinstance(rows, list):
            code = rows.get("code") if isinstance(rows, dict) else None
            msg = rows.get("msg") if isinstance(rows, dict) else str(rows)
            if code == RATE_LIMIT_ERROR_CODE:
                retry_count += 1
                if retry_count > MAX_RATE_LIMIT_RETRIES:
                    if rows_all:
                        break
                    raise RuntimeError(
                        f"Binance derivatives rate-limited {metric_key} with no cached page "
                        f"(msg={msg}, params={params})"
                    )
                time.sleep(min(15.0, 1.5 * retry_count))
                continue
            raise RuntimeError(
                f"Unexpected Binance derivatives response for {metric_key} "
                f"(code={code}, msg={msg}, params={params})"
            )
        retry_count = 0
        rows_all.extend(rows)
        last_timestamp = int(rows[-1][spec["timestamp_field"]])
        nxt = max(last_timestamp + 1, request_end_ms + 1)
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.10)
    return _normalize_metric_frame(metric_key, pd.DataFrame(rows_all))


def update_derivative_metric_cache(
    symbol: str,
    metric_key: str,
    *,
    end_dt: datetime,
    period: str = DEFAULT_PERIOD,
    lookback_days: int = MAX_RECENT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    existing = load_derivative_metric_cache(symbol, metric_key, period=period)
    lookback_window = timedelta(days=lookback_days)
    desired_start = end_dt - lookback_window
    if existing.empty:
        refresh_start = desired_start
    else:
        existing_first = pd.Timestamp(existing["timestamp"].iloc[0]).to_pydatetime()
        existing_last = pd.Timestamp(existing["timestamp"].iloc[-1]).to_pydatetime()
        if existing_first > desired_start:
            refresh_start = desired_start
        else:
            refresh_start = max(desired_start, existing_last - timedelta(days=1))
    fresh = fetch_derivative_metric(
        symbol,
        metric_key,
        start_dt=refresh_start,
        end_dt=end_dt,
        period=period,
        lookback_days=lookback_days,
    )
    if existing.empty:
        merged = fresh
    elif fresh.empty:
        merged = existing
    else:
        merged = _normalize_metric_frame(metric_key, pd.concat([existing, fresh], ignore_index=True))
    path = derivative_cache_path(symbol, metric_key, period=period)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def slice_metric_frame(
    frame: pd.DataFrame,
    *,
    start_dt: datetime,
    end_dt: datetime,
    history: timedelta | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    start_ts = pd.Timestamp(start_dt).tz_convert("UTC") if pd.Timestamp(start_dt).tzinfo else pd.Timestamp(start_dt, tz="UTC")
    end_ts = pd.Timestamp(end_dt).tz_convert("UTC") if pd.Timestamp(end_dt).tzinfo else pd.Timestamp(end_dt, tz="UTC")
    history_start = start_ts - history if history is not None else start_ts
    anchor = frame[(frame["timestamp"] < start_ts) & (frame["timestamp"] >= history_start)].copy()
    if history is None:
        anchor = anchor.tail(1)
    within = frame[(frame["timestamp"] >= start_ts) & (frame["timestamp"] <= end_ts)].copy()
    return (
        pd.concat([anchor, within], ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def load_derivative_bundle(
    symbol: str,
    *,
    start_dt: datetime,
    end_dt: datetime,
    period: str = DEFAULT_PERIOD,
    fetch: bool = False,
    lookback_days: int = MAX_RECENT_LOOKBACK_DAYS,
) -> dict[str, pd.DataFrame]:
    bundle: dict[str, pd.DataFrame] = {}
    for metric_key in METRIC_SPECS:
        frame = (
            update_derivative_metric_cache(
                symbol,
                metric_key,
                end_dt=end_dt,
                period=period,
                lookback_days=lookback_days,
            )
            if fetch
            else load_derivative_metric_cache(symbol, metric_key, period=period)
        )
        bundle[metric_key] = slice_metric_frame(frame, start_dt=start_dt, end_dt=end_dt)
    return bundle


def slice_derivative_bundle(
    bundle: dict[str, pd.DataFrame],
    *,
    start_dt: datetime,
    end_dt: datetime,
    history: timedelta | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        metric_key: slice_metric_frame(frame, start_dt=start_dt, end_dt=end_dt, history=history)
        for metric_key, frame in (bundle or {}).items()
    }
