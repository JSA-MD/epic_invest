#!/usr/bin/env python3
"""Cached daily market-context loader for macro/cross-asset regime features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timezone
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKET_CONTEXT_DIR = PROJECT_ROOT / "data" / "market_context" / "daily"
MARKET_CONTEXT_MANIFEST = PROJECT_ROOT / "data" / "market_context" / "manifest.json"
DEFAULT_MAX_FFILL_DAYS = 3
DEFAULT_MIN_CONTEXT_DAYS = 60


@dataclass(frozen=True)
class MarketContextSeries:
    name: str
    label: str
    source: str
    symbol: str
    close: pd.Series


MARKET_CONTEXT_SPECS: dict[str, dict[str, Any]] = {
    "QQQ": {
        "label": "QQQ",
        "candidates": (("yahoo", "QQQ"),),
    },
    "SPY": {
        "label": "SPY",
        "candidates": (("yahoo", "SPY"),),
    },
    "GLD": {
        "label": "GLD",
        "candidates": (("yahoo", "GLD"),),
    },
    "DXY": {
        "label": "DXY",
        "candidates": (
            ("yahoo", "DX-Y.NYB"),
            ("yahoo", "UUP"),
        ),
    },
}


def context_cache_path(name: str) -> Path:
    return MARKET_CONTEXT_DIR / f"{name}.csv"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def fetch_stooq_daily(symbol: str) -> pd.DataFrame:
    resp = requests.get(
        "https://stooq.com/q/d/l/",
        params={"s": symbol, "i": "d"},
        timeout=20,
    )
    resp.raise_for_status()
    text = resp.text.strip()
    if not text or text.lower().startswith("no data"):
        raise ValueError(f"No data returned for Stooq symbol {symbol}")
    frame = pd.read_csv(StringIO(text))
    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    frame = frame.rename(columns=rename_map)
    required = {"date", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected Stooq schema for {symbol}: {sorted(frame.columns)}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["adj_close"] = frame["close"]
    frame["source"] = "stooq"
    frame["fetched_at"] = pd.Timestamp.now(tz=timezone.utc).isoformat()
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return frame[["date", "open", "high", "low", "close", "adj_close", "volume", "source", "fetched_at"]]


def fetch_yahoo_daily(symbol: str) -> pd.DataFrame:
    resp = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "interval": "1d",
            "range": "10y",
            "includeAdjustedClose": "true",
            "events": "div,splits",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    resp.raise_for_status()
    payload = resp.json()
    result = ((payload.get("chart") or {}).get("result") or [None])[0]
    if result is None:
        error = ((payload.get("chart") or {}).get("error") or {}).get("description")
        raise ValueError(error or f"No Yahoo chart result for {symbol}")

    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    adjclose_rows = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(timestamps, dtype="int64"), unit="s", utc=True, errors="coerce").dt.normalize(),
            "open": pd.to_numeric(pd.Series(quote.get("open", [])), errors="coerce"),
            "high": pd.to_numeric(pd.Series(quote.get("high", [])), errors="coerce"),
            "low": pd.to_numeric(pd.Series(quote.get("low", [])), errors="coerce"),
            "close": pd.to_numeric(pd.Series(quote.get("close", [])), errors="coerce"),
            "adj_close": pd.to_numeric(pd.Series(adjclose_rows.get("adjclose", [])), errors="coerce"),
            "volume": pd.to_numeric(pd.Series(quote.get("volume", [])), errors="coerce"),
        }
    )
    frame["source"] = "yahoo"
    frame["fetched_at"] = pd.Timestamp.now(tz=timezone.utc).isoformat()
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if frame["adj_close"].isna().all():
        frame["adj_close"] = frame["close"]
    return frame[["date", "open", "high", "low", "close", "adj_close", "volume", "source", "fetched_at"]]


def fetch_market_context_series(name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = MARKET_CONTEXT_SPECS[name]
    last_error: str | None = None
    for source, symbol in spec["candidates"]:
        try:
            if source == "stooq":
                frame = fetch_stooq_daily(symbol)
            elif source == "yahoo":
                frame = fetch_yahoo_daily(symbol)
            else:
                raise ValueError(f"Unsupported source {source}")
            meta = {
                "name": name,
                "label": spec["label"],
                "source": source,
                "resolved_symbol": symbol,
                "proxy_used": bool(name == "DXY" and symbol != "DX-Y.NYB"),
                "rows": int(len(frame)),
            }
            return frame, meta
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    raise RuntimeError(f"Failed to fetch {name}: {last_error or 'unknown error'}")


def load_cached_market_context_series(name: str) -> MarketContextSeries | None:
    path = context_cache_path(name)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "date" not in frame.columns or "close" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if frame.empty:
        return None

    spec = MARKET_CONTEXT_SPECS[name]
    source = str(frame["source"].dropna().iloc[-1]) if "source" in frame.columns and not frame["source"].dropna().empty else "cache"
    symbol = str(frame.get("resolved_symbol", pd.Series(dtype="object")).dropna().iloc[-1]) if "resolved_symbol" in frame else name
    return MarketContextSeries(
        name=name,
        label=spec["label"],
        source=source,
        symbol=symbol,
        close=frame.set_index("date")["close"].sort_index(),
    )


def save_market_context_series(name: str, frame: pd.DataFrame, meta: dict[str, Any]) -> None:
    output = frame.copy()
    output["resolved_symbol"] = meta["resolved_symbol"]
    path = context_cache_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)

    manifest = _read_json(MARKET_CONTEXT_MANIFEST)
    manifest[name] = {
        "label": meta["label"],
        "source": meta["source"],
        "resolved_symbol": meta["resolved_symbol"],
        "proxy_used": bool(meta.get("proxy_used", False)),
        "rows": int(meta["rows"]),
        "updated_at": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "path": str(path),
    }
    _write_json(MARKET_CONTEXT_MANIFEST, manifest)


def align_market_context_to_index(
    market_close: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    max_ffill_days: int = DEFAULT_MAX_FFILL_DAYS,
) -> pd.DataFrame:
    if market_close.empty:
        return market_close
    aligned = market_close.sort_index().reindex(target_index)
    if max_ffill_days > 0:
        aligned = aligned.ffill(limit=max_ffill_days)
    return aligned.dropna(axis=1, how="all")


def load_market_context_dataset(
    *,
    names: tuple[str, ...] = tuple(MARKET_CONTEXT_SPECS.keys()),
    refresh: bool = False,
    allow_fetch_on_miss: bool = False,
    target_index: pd.DatetimeIndex | None = None,
    max_ffill_days: int = DEFAULT_MAX_FFILL_DAYS,
    min_context_days: int = DEFAULT_MIN_CONTEXT_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    series_rows: list[pd.Series] = []
    manifest_rows: list[dict[str, Any]] = []

    for name in names:
        loaded = None if refresh else load_cached_market_context_series(name)
        status = "loaded_cache" if loaded is not None else "missing"

        if loaded is None and allow_fetch_on_miss:
            try:
                fetched, meta = fetch_market_context_series(name)
                save_market_context_series(name, fetched, meta)
                loaded = load_cached_market_context_series(name)
                status = "fetched"
            except Exception as exc:  # noqa: BLE001
                manifest_rows.append(
                    {
                        "name": name,
                        "label": MARKET_CONTEXT_SPECS[name]["label"],
                        "status": "fetch_failed",
                        "error": str(exc),
                    }
                )
                continue

        if loaded is None:
            manifest_rows.append(
                {
                    "name": name,
                    "label": MARKET_CONTEXT_SPECS[name]["label"],
                    "status": status,
                }
            )
            continue

        series_rows.append(loaded.close.rename(name))
        manifest_rows.append(
            {
                "name": name,
                "label": loaded.label,
                "status": status,
                "source": loaded.source,
                "resolved_symbol": loaded.symbol,
                "rows": int(len(loaded.close)),
            }
        )

    if not series_rows:
        return pd.DataFrame(), {
            "status": "missing",
            "series": manifest_rows,
            "max_ffill_days": int(max_ffill_days),
            "min_context_days": int(min_context_days),
        }

    market_close = pd.concat(series_rows, axis=1).sort_index()
    if target_index is not None:
        market_close = align_market_context_to_index(
            market_close,
            target_index,
            max_ffill_days=max_ffill_days,
        )

    usable_columns = [col for col in market_close.columns if int(market_close[col].notna().sum()) >= int(min_context_days)]
    market_close = market_close[usable_columns].copy()

    for row in manifest_rows:
        if row.get("name") not in usable_columns and row.get("status") in {"loaded_cache", "fetched"}:
            row["status"] = "insufficient_history"

    status = "ok" if usable_columns else "insufficient_history"
    return market_close, {
        "status": status,
        "series": manifest_rows,
        "usable_columns": usable_columns,
        "max_ffill_days": int(max_ffill_days),
        "min_context_days": int(min_context_days),
    }
