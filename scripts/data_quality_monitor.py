#!/usr/bin/env python3
"""Assess freshness and availability of strategy data layers."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
BINANCE_FUTURES_DIR = DATA_DIR / "binance_futures"
DERIVATIVE_DIR = BINANCE_FUTURES_DIR / "derivatives"
MARKET_CONTEXT_DIR = DATA_DIR / "market_context" / "daily"
LOB_DIR = DATA_DIR / "lob" / "binance_futures"
DEFAULT_OUTPUT_PATH = MODELS_DIR / "data_quality_report.json"
DEFAULT_PAIRS = ("BTCUSDT", "BNBUSDT")
DEFAULT_MARKET_CONTEXT = ("QQQ", "SPY", "GLD", "DXY")
DERIVATIVE_METRICS = (
    "open_interest",
    "top_trader_position_ratio",
    "top_trader_account_ratio",
    "global_long_short_ratio",
    "taker_buy_sell_ratio",
    "basis_perpetual",
)
UTC = timezone.utc
US_EASTERN = ZoneInfo("America/New_York")
US_MARKET_CLOSE_GRACE = time(hour=16, minute=30)


@dataclass(frozen=True)
class FreshnessThreshold:
    warn_after_seconds: float
    stale_after_seconds: float


THRESHOLDS = {
    "ohlcv_5m": FreshnessThreshold(warn_after_seconds=6 * 3600, stale_after_seconds=24 * 3600),
    "ohlcv_1d": FreshnessThreshold(warn_after_seconds=48 * 3600, stale_after_seconds=96 * 3600),
    "funding": FreshnessThreshold(warn_after_seconds=18 * 3600, stale_after_seconds=48 * 3600),
    "derivatives": FreshnessThreshold(warn_after_seconds=18 * 3600, stale_after_seconds=48 * 3600),
    "market_context": FreshnessThreshold(warn_after_seconds=72 * 3600, stale_after_seconds=120 * 3600),
    "lob": FreshnessThreshold(warn_after_seconds=30 * 60, stale_after_seconds=2 * 3600),
}


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess strategy data freshness and coverage.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    return parser.parse_args()


def _parse_timestamp(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        if value.isdigit():
            numeric = int(value)
            unit = "ms" if numeric > 10_000_000_000 else "s"
            return datetime.fromtimestamp(numeric / (1000 if unit == "ms" else 1), tz=UTC).isoformat()
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _read_last_csv_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return None
            index = None
            for candidate in ("timestamp", "fundingTime", "open_time", "openTime", "date", "collected_at", "T"):
                if candidate in header:
                    index = header.index(candidate)
                    break
            if index is None:
                return None
            last_row: list[str] | None = None
            for row in reader:
                if row:
                    last_row = row
            if last_row is None or index >= len(last_row):
                return None
            return _parse_timestamp(last_row[index])
    except OSError:
        return None


def _read_last_jsonl_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        last_line = ""
        with open(path, "r") as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return None
        payload = json.loads(last_line)
    except (OSError, json.JSONDecodeError):
        return None
    for candidate in ("collected_at", "timestamp", "event_time"):
        value = _parse_timestamp(payload.get(candidate))
        if value is not None:
            return value
    return None


def _mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return None


def _age_seconds(iso_value: str | None, *, now: datetime) -> float | None:
    if iso_value is None:
        return None
    try:
        parsed = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (now - parsed.astimezone(UTC)).total_seconds())


def classify_freshness(age_seconds: float | None, threshold: FreshnessThreshold) -> str:
    if age_seconds is None:
        return "missing"
    if age_seconds > threshold.stale_after_seconds:
        return "stale"
    if age_seconds > threshold.warn_after_seconds:
        return "aging"
    return "fresh"


def summarize_feed(path: Path, threshold: FreshnessThreshold, *, now: datetime, timestamp_reader: str = "csv") -> dict[str, Any]:
    last_timestamp = (
        _read_last_jsonl_timestamp(path) if timestamp_reader == "jsonl" else _read_last_csv_timestamp(path)
    )
    age_seconds = _age_seconds(last_timestamp, now=now)
    return {
        "path": str(path),
        "exists": path.exists(),
        "last_timestamp": last_timestamp,
        "age_seconds": None if age_seconds is None else round(age_seconds, 1),
        "freshness": classify_freshness(age_seconds, threshold),
        "file_modified_at": _mtime_iso(path),
    }


def _rollup_status(values: list[str]) -> str:
    if not values:
        return "missing"
    if any(value in {"critical", "stale"} for value in values):
        return "critical"
    if any(value in {"warning", "aging", "missing"} for value in values):
        return "warning"
    return "ok"


def summarize_ohlcv(pairs: tuple[str, ...], *, now: datetime) -> dict[str, Any]:
    per_pair: dict[str, Any] = {}
    statuses: list[str] = []
    for pair in pairs:
        feed_5m = summarize_feed(BINANCE_FUTURES_DIR / f"{pair}_5m.csv", THRESHOLDS["ohlcv_5m"], now=now)
        feed_1d = summarize_feed(BINANCE_FUTURES_DIR / f"{pair}_1d.csv", THRESHOLDS["ohlcv_1d"], now=now)
        funding_files = sorted(BINANCE_FUTURES_DIR.glob(f"{pair}_funding_*.csv"))
        funding_path = funding_files[-1] if funding_files else Path()
        feed_funding = summarize_feed(funding_path, THRESHOLDS["funding"], now=now) if funding_files else {
            "path": str(funding_path),
            "exists": False,
            "last_timestamp": None,
            "age_seconds": None,
            "freshness": "missing",
            "file_modified_at": None,
        }
        per_pair[pair] = {"5m": feed_5m, "1d": feed_1d, "funding": feed_funding}
        statuses.extend([feed_5m["freshness"], feed_1d["freshness"], feed_funding["freshness"]])
    return {
        "status": _rollup_status(statuses),
        "per_pair": per_pair,
    }


def summarize_derivatives(pairs: tuple[str, ...], *, now: datetime) -> dict[str, Any]:
    per_pair: dict[str, Any] = {}
    statuses: list[str] = []
    for pair in pairs:
        metrics: dict[str, Any] = {}
        for metric in DERIVATIVE_METRICS:
            payload = summarize_feed(DERIVATIVE_DIR / f"{pair}_{metric}_5m.csv", THRESHOLDS["derivatives"], now=now)
            metrics[metric] = payload
            statuses.append(payload["freshness"])
        per_pair[pair] = metrics
    return {
        "status": _rollup_status(statuses),
        "per_pair": per_pair,
    }


def summarize_market_context(*, now: datetime) -> dict[str, Any]:
    per_series: dict[str, Any] = {}
    statuses: list[str] = []
    for name in DEFAULT_MARKET_CONTEXT:
        payload = summarize_market_context_feed(name, now=now)
        per_series[name] = payload
        statuses.append(payload["freshness"])
    return {
        "status": _rollup_status(statuses),
        "per_series": per_series,
    }


def summarize_lob(pairs: tuple[str, ...], *, now: datetime) -> dict[str, Any]:
    per_pair: dict[str, Any] = {}
    statuses: list[str] = []
    for pair in pairs:
        features = summarize_feed(LOB_DIR / "features" / f"{pair}_microstructure.csv", THRESHOLDS["lob"], now=now)
        snapshots = summarize_feed(
            LOB_DIR / "snapshots" / f"{pair}_depth_top20.jsonl",
            THRESHOLDS["lob"],
            now=now,
            timestamp_reader="jsonl",
        )
        trades = summarize_feed(LOB_DIR / "agg_trades" / f"{pair}.csv", THRESHOLDS["lob"], now=now)
        per_pair[pair] = {"features": features, "snapshots": snapshots, "agg_trades": trades}
        statuses.extend([features["freshness"], snapshots["freshness"], trades["freshness"]])
    return {
        "status": _rollup_status(statuses),
        "per_pair": per_pair,
    }


def build_recommendations(snapshot: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    ohlcv = snapshot["ohlcv"]
    derivatives = snapshot["derivatives"]
    market_context = snapshot["market_context"]
    lob = snapshot["lob"]

    if lob["status"] != "ok":
        recommendations.append("LOB 미시구조 수집기를 상시화해 spread·depth imbalance·microprice를 확보해야 합니다.")
    if ohlcv["status"] != "ok":
        recommendations.append("OHLCV 캐시 결손 또는 stale 상태로 가격·변동성·장기 검증 품질이 약화되고 있습니다.")
    if any(not item["1d"]["exists"] for item in ohlcv["per_pair"].values()):
        recommendations.append("일봉 캐시 결손이 있어 교차자산·장기 검증 레이어 신뢰도가 떨어집니다.")
    if derivatives["status"] != "ok":
        recommendations.append("파생 포지셔닝 캐시가 오래돼 OI·basis·long/short 신호가 stale 상태입니다.")
    if market_context["status"] != "ok":
        recommendations.append("QQQ/SPY/GLD/DXY 문맥 캐시가 낡아 corr-state gating 품질이 떨어질 수 있습니다.")
    return recommendations


def build_data_quality_snapshot(*, pairs: tuple[str, ...] = DEFAULT_PAIRS, now: datetime | None = None) -> dict[str, Any]:
    now = now or utc_now()
    ohlcv = summarize_ohlcv(pairs, now=now)
    derivatives = summarize_derivatives(pairs, now=now)
    market_context = summarize_market_context(now=now)
    lob = summarize_lob(pairs, now=now)
    overall = _rollup_status([ohlcv["status"], derivatives["status"], market_context["status"], lob["status"]])
    payload = {
        "generated_at": now.isoformat(),
        "status": overall,
        "ohlcv": ohlcv,
        "derivatives": derivatives,
        "market_context": market_context,
        "lob": lob,
    }
    payload["recommendations"] = build_recommendations(payload)
    return payload


def _previous_us_weekday(day: date) -> date:
    current = day
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _last_completed_us_trading_day(now: datetime) -> date:
    eastern_now = now.astimezone(US_EASTERN)
    current_day = eastern_now.date()
    if eastern_now.weekday() >= 5:
        return _previous_us_weekday(current_day)
    if eastern_now.time() < US_MARKET_CLOSE_GRACE:
        return _previous_us_weekday(current_day - timedelta(days=1))
    return current_day


def _count_us_trading_days_between(start_day: date, end_day: date) -> int:
    if start_day >= end_day:
        return 0
    count = 0
    cursor = start_day + timedelta(days=1)
    while cursor <= end_day:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


def summarize_market_context_feed(name: str, *, now: datetime) -> dict[str, Any]:
    path = MARKET_CONTEXT_DIR / f"{name}.csv"
    payload = summarize_feed(path, THRESHOLDS["market_context"], now=now)
    last_timestamp = payload["last_timestamp"]
    if last_timestamp is None:
        payload["expected_latest_date"] = None
        payload["trading_day_lag"] = None
        return payload
    try:
        last_dt = datetime.fromisoformat(last_timestamp.replace("Z", "+00:00"))
    except ValueError:
        payload["expected_latest_date"] = None
        payload["trading_day_lag"] = None
        return payload
    expected_day = _last_completed_us_trading_day(now)
    last_day = last_dt.astimezone(UTC).date()
    lag = _count_us_trading_days_between(last_day, expected_day)
    if lag <= 0:
        freshness = "fresh"
    elif lag == 1:
        freshness = "aging"
    else:
        freshness = "stale"
    payload["expected_latest_date"] = expected_day.isoformat()
    payload["trading_day_lag"] = lag
    payload["freshness"] = freshness
    return payload


def main() -> None:
    args = parse_args()
    snapshot = build_data_quality_snapshot(pairs=tuple(args.pairs))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    print(json.dumps({"status": "ok", "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
