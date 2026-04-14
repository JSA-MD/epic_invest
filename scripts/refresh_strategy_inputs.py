#!/usr/bin/env python3
"""Refresh market-context, derivatives, and OHLCV caches used by pairwise routing."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from derivative_market_data import METRIC_SPECS, update_derivative_metric_cache
import gp_crypto_evolution as gp
from market_context import load_market_context_dataset
from replay_regime_mixture_realistic import fetch_funding_rates


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_OUTPUT_PATH = MODELS_DIR / "strategy_input_refresh_report.json"
DEFAULT_PAIRS = ("BTCUSDT", "BNBUSDT")
DEFAULT_CONTEXT = ("QQQ", "SPY", "GLD", "DXY")
UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh pairwise strategy input caches.")
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--context", nargs="+", default=list(DEFAULT_CONTEXT))
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--refresh-ohlcv", action="store_true")
    parser.add_argument("--ohlcv-start", default=gp.TRAIN_START)
    parser.add_argument("--refresh-funding", action="store_true")
    parser.add_argument("--funding-start", default=gp.TRAIN_START)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def _frame_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "last_timestamp": None}
    for column in ("timestamp", "fundingTime", "date"):
        if column in frame.columns:
            last_value = frame[column].iloc[-1]
            break
    else:
        last_value = frame.index[-1]
    if hasattr(last_value, "isoformat"):
        last_timestamp = last_value.isoformat()
    else:
        last_timestamp = str(last_value)
    return {
        "rows": int(len(frame)),
        "last_timestamp": last_timestamp,
    }


def refresh_derivatives(pairs: tuple[str, ...], *, lookback_days: int, now: datetime) -> dict[str, Any]:
    per_pair: dict[str, Any] = {}
    for pair in pairs:
        metrics: dict[str, Any] = {}
        for metric_key in METRIC_SPECS:
            frame = update_derivative_metric_cache(
                pair,
                metric_key,
                end_dt=now,
                lookback_days=lookback_days,
            )
            metrics[metric_key] = _frame_summary(frame)
        per_pair[pair] = metrics
    return {"status": "ok", "per_pair": per_pair}


def refresh_market_context(names: tuple[str, ...]) -> dict[str, Any]:
    _, status = load_market_context_dataset(
        names=names,
        refresh=True,
        allow_fetch_on_miss=True,
    )
    return status


def refresh_ohlcv(
    pairs: tuple[str, ...],
    *,
    start: str | None,
) -> dict[str, Any]:
    per_pair: dict[str, Any] = {}
    for pair in pairs:
        frame_5m = gp.load_pair(
            pair,
            interval="5m",
            start=start,
            end=None,
            refresh_cache=True,
        )
        frame_1d = gp.load_pair(
            pair,
            interval="1d",
            start=start,
            end=None,
            refresh_cache=True,
        )
        per_pair[pair] = {
            "5m": _frame_summary(frame_5m),
            "1d": _frame_summary(frame_1d),
        }
    return {"status": "ok", "per_pair": per_pair}


def refresh_funding(
    pairs: tuple[str, ...],
    *,
    start: str,
    now: datetime,
) -> dict[str, Any]:
    end_date = now.date().isoformat()
    per_pair: dict[str, Any] = {}
    for pair in pairs:
        frame = fetch_funding_rates(
            pair,
            datetime.fromisoformat(start).replace(tzinfo=UTC),
            now,
        )
        path = gp.DATA_DIR / f"{pair}_funding_{start}_{end_date}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        per_pair[pair] = {
            **_frame_summary(frame),
            "path": str(path),
        }
    return {"status": "ok", "per_pair": per_pair}


def build_refresh_report(
    *,
    pairs: tuple[str, ...],
    context_names: tuple[str, ...],
    lookback_days: int,
    refresh_ohlcv_inputs: bool = False,
    ohlcv_start: str | None = gp.TRAIN_START,
    refresh_funding_inputs: bool = False,
    funding_start: str = gp.TRAIN_START,
) -> dict[str, Any]:
    now = utc_now()
    market_context_status = refresh_market_context(context_names)
    derivative_status = refresh_derivatives(pairs, lookback_days=lookback_days, now=now)
    ohlcv_status = (
        refresh_ohlcv(pairs, start=ohlcv_start)
        if refresh_ohlcv_inputs
        else {"status": "skipped", "per_pair": {}}
    )
    funding_status = (
        refresh_funding(pairs, start=funding_start, now=now)
        if refresh_funding_inputs
        else {"status": "skipped", "per_pair": {}}
    )
    return {
        "generated_at": now.isoformat(),
        "pairs": list(pairs),
        "context_names": list(context_names),
        "lookback_days": int(lookback_days),
        "refresh_ohlcv": bool(refresh_ohlcv_inputs),
        "ohlcv_start": ohlcv_start,
        "ohlcv": ohlcv_status,
        "refresh_funding": bool(refresh_funding_inputs),
        "funding_start": funding_start,
        "funding": funding_status,
        "market_context": market_context_status,
        "derivatives": derivative_status,
    }


def main() -> None:
    args = parse_args()
    report = build_refresh_report(
        pairs=tuple(args.pairs),
        context_names=tuple(args.context),
        lookback_days=int(args.lookback_days),
        refresh_ohlcv_inputs=bool(args.refresh_ohlcv),
        ohlcv_start=args.ohlcv_start,
        refresh_funding_inputs=bool(args.refresh_funding),
        funding_start=args.funding_start,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"status": "ok", "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
