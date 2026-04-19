#!/usr/bin/env python3
"""Audit local strategy data layers and recommend next collection upgrades."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
BINANCE_FUTURES_DIR = DATA_DIR / "binance_futures"
DERIVATIVE_DIR = BINANCE_FUTURES_DIR / "derivatives"
MARKET_CONTEXT_DIR = DATA_DIR / "market_context" / "daily"
MARKET_CONTEXT_MANIFEST = DATA_DIR / "market_context" / "manifest.json"
DEFAULT_OUTPUT_PATH = MODELS_DIR / "strategy_data_readiness_report.json"
UTC = timezone.utc

DEFAULT_TARGETS = {
    "daily_win_rate_min": 0.70,
    "daily_return_min": 0.007,
    "max_drawdown_floor": -0.05,
}

DEFAULT_PAIRS = ("BTCUSDT", "BNBUSDT")
DERIVATIVE_METRICS = (
    "open_interest",
    "top_trader_position_ratio",
    "top_trader_account_ratio",
    "global_long_short_ratio",
    "taker_buy_sell_ratio",
    "basis_perpetual",
)
MARKET_CONTEXT_NAMES = ("QQQ", "SPY", "GLD", "DXY")

STRATEGIC_LAYER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "lob_microstructure",
        "priority": 1,
        "path": DATA_DIR / "lob",
        "status_if_missing": "missing",
        "decision_role": "나쁜 진입 차단",
        "why": "초단기 손실 대부분은 방향 오류보다 진입 품질 저하에서 발생한다.",
        "required_sources": (
            "Binance futures depth snapshot",
            "Binance futures diff depth stream",
            "book ticker",
            "agg trades",
        ),
        "derived_features": (
            "spread",
            "top-N depth imbalance",
            "microprice",
            "queue imbalance",
            "cancel ratio",
            "order flow imbalance",
        ),
        "source_links": (
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Order-Book",
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Diff-Book-Depth-Streams",
            "https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/How-to-manage-a-local-order-book-correctly",
        ),
    },
    {
        "key": "event_and_dc_bars",
        "priority": 2,
        "path": DATA_DIR / "event_bars",
        "status_if_missing": "missing",
        "decision_role": "레짐 전환 조기 감지",
        "why": "고정 시간봉보다 이벤트 기반 샘플링이 노이즈를 줄이고 구조 변화를 더 빨리 드러낸다.",
        "required_sources": (
            "5m OHLCV",
            "trade count proxy",
            "taker flow",
        ),
        "derived_features": (
            "directional-change event clocks",
            "run length",
            "overshoot",
            "volume bars",
            "dollar bars",
        ),
        "source_links": (
            "https://arxiv.org/abs/2506.05764",
        ),
    },
    {
        "key": "onchain_exchange_flow",
        "priority": 3,
        "path": DATA_DIR / "onchain",
        "status_if_missing": "missing",
        "decision_role": "현물 수급·참여 강도 확인",
        "why": "온체인과 거래소 흐름은 파생 포지셔닝과 별개로 현물 기반 수급을 보여준다.",
        "required_sources": (
            "exchange balances",
            "exchange net position change",
            "miner/whale flow",
            "stablecoin exchange inflow",
        ),
        "derived_features": (
            "spot supply absorption",
            "exchange inventory pressure",
            "miner sell pressure",
            "whale inflow shock",
        ),
        "source_links": (
            "https://docs.glassnode.com/basic-api/endpoints/indicators",
            "https://docs.glassnode.com/basic-api/endpoints/fees",
            "https://docs.glassnode.com/further-information/changelog/2025",
            "https://arxiv.org/abs/2506.21246",
        ),
    },
    {
        "key": "options_surface",
        "priority": 4,
        "path": DATA_DIR / "options",
        "status_if_missing": "missing",
        "decision_role": "변동성 체제·꼬리 위험 선행 감지",
        "why": "옵션 IV term structure와 skew는 현물·선물보다 앞서 위험 선호 변화를 반영하는 경우가 많다.",
        "required_sources": (
            "BTC option book summary",
            "mark IV / volatility index",
            "option OI by strike/tenor",
        ),
        "derived_features": (
            "ATM IV level",
            "front-back IV slope",
            "risk reversal / skew proxy",
            "OI concentration",
        ),
        "source_links": (
            "https://docs.deribit.com/api-reference/upcoming/market-data/public-get_book_summary_by_currency",
            "https://docs.deribit.com/subscriptions/market-data/markpriceoptionsindex_name",
        ),
    },
    {
        "key": "macro_event_calendar",
        "priority": 5,
        "path": DATA_DIR / "event_calendar",
        "status_if_missing": "missing",
        "decision_role": "예정된 이벤트 회피·재진입 타이밍",
        "why": "거시 데이터는 값 자체보다 발표 시점과 서프라이즈가 더 중요하다.",
        "required_sources": (
            "FRED/ALFRED release-aware macro series",
            "CPI/NFP/FOMC release timestamps",
            "VIX / yields / DXY daily context",
        ),
        "derived_features": (
            "time_to_next_release",
            "post_release_shock",
            "surprise regime",
            "event blackout windows",
        ),
        "source_links": (
            "https://fred.stlouisfed.org/docs/api/fred/fred/realtime_period.html",
            "https://fred.stlouisfed.org/docs/api/fred/series/release_tables.html",
        ),
    },
)


def iso_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report local strategy data readiness and next collection priorities.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    return parser.parse_args()


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


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
            return pd.to_datetime(numeric, unit=unit, utc=True).isoformat()
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _read_last_csv_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if not header:
                return None
            timestamp_idx = None
            for candidate in ("timestamp", "fundingTime", "open_time", "openTime", "date"):
                if candidate in header:
                    timestamp_idx = header.index(candidate)
                    break
            if timestamp_idx is None:
                return None
            last_row: list[str] | None = None
            for row in reader:
                if row:
                    last_row = row
            if last_row is None or timestamp_idx >= len(last_row):
                return None
            return _parse_timestamp(last_row[timestamp_idx])
    except OSError:
        return None


def _mtime_iso(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
    except OSError:
        return None


def _select_latest_csv_path(paths: list[Path]) -> Path:
    if not paths:
        return Path()
    ranked: list[tuple[tuple[bool, str, str, str], Path]] = []
    for path in paths:
        ranked.append(
            (
                (
                    _read_last_csv_timestamp(path) is not None,
                    _read_last_csv_timestamp(path) or "",
                    _mtime_iso(path) or "",
                    str(path),
                ),
                path,
            )
        )
    return max(ranked, key=lambda item: item[0])[1]


def summarize_csv_feed(path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "last_timestamp": _read_last_csv_timestamp(path) if exists else None,
        "file_modified_at": _mtime_iso(path) if exists else None,
    }


def summarize_ohlcv_layer(pairs: tuple[str, ...]) -> dict[str, Any]:
    per_pair: dict[str, Any] = {}
    pair_ready_count = 0
    for pair in pairs:
        funding_path = _select_latest_csv_path(sorted(BINANCE_FUTURES_DIR.glob(f"{pair}_funding_*.csv")))
        feeds = {
            "5m": summarize_csv_feed(BINANCE_FUTURES_DIR / f"{pair}_5m.csv"),
            "1d": summarize_csv_feed(BINANCE_FUTURES_DIR / f"{pair}_1d.csv"),
            "funding": summarize_csv_feed(funding_path),
        }
        feeds["pair_ready"] = bool(feeds["5m"]["exists"] and feeds["1d"]["exists"])
        pair_ready_count += int(feeds["pair_ready"])
        per_pair[pair] = feeds
    status = "ready" if pair_ready_count == len(pairs) else "partial" if pair_ready_count else "missing"
    return {
        "status": status,
        "pairs_ready": pair_ready_count,
        "pair_count": len(pairs),
        "per_pair": per_pair,
    }


def summarize_derivatives_layer(pairs: tuple[str, ...]) -> dict[str, Any]:
    per_pair: dict[str, Any] = {}
    total_expected = len(pairs) * len(DERIVATIVE_METRICS)
    total_present = 0
    for pair in pairs:
        metrics: dict[str, Any] = {}
        for metric in DERIVATIVE_METRICS:
            payload = summarize_csv_feed(DERIVATIVE_DIR / f"{pair}_{metric}_5m.csv")
            metrics[metric] = payload
            total_present += int(payload["exists"])
        per_pair[pair] = metrics
    coverage = float(total_present / total_expected) if total_expected else 0.0
    status = "ready" if coverage >= 0.99 else "partial" if coverage > 0.0 else "missing"
    return {
        "status": status,
        "coverage": coverage,
        "present_feeds": total_present,
        "expected_feeds": total_expected,
        "per_pair": per_pair,
    }


def summarize_market_context_layer() -> dict[str, Any]:
    manifest = _safe_json(MARKET_CONTEXT_MANIFEST)
    per_series: dict[str, Any] = {}
    present_count = 0
    for name in MARKET_CONTEXT_NAMES:
        path = MARKET_CONTEXT_DIR / f"{name}.csv"
        payload = summarize_csv_feed(path)
        payload["manifest_entry"] = manifest.get(name)
        per_series[name] = payload
        present_count += int(payload["exists"])
    status = "ready" if present_count == len(MARKET_CONTEXT_NAMES) else "partial" if present_count else "missing"
    return {
        "status": status,
        "series_ready": present_count,
        "series_expected": len(MARKET_CONTEXT_NAMES),
        "manifest_path": str(MARKET_CONTEXT_MANIFEST),
        "per_series": per_series,
    }


def summarize_strategic_layers() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in STRATEGIC_LAYER_SPECS:
        path = Path(spec["path"])
        exists = path.exists()
        row = {
            "key": spec["key"],
            "priority": int(spec["priority"]),
            "status": "present" if exists else str(spec["status_if_missing"]),
            "path": str(path),
            "decision_role": spec["decision_role"],
            "why": spec["why"],
            "required_sources": list(spec["required_sources"]),
            "derived_features": list(spec["derived_features"]),
            "source_links": list(spec["source_links"]),
        }
        rows.append(row)
    return rows


def build_executive_summary(
    ohlcv: dict[str, Any],
    derivatives: dict[str, Any],
    market_context: dict[str, Any],
    strategic_layers: list[dict[str, Any]],
) -> dict[str, Any]:
    strengths: list[str] = []
    gaps: list[str] = []

    if ohlcv["status"] != "missing":
        strengths.append("기본 OHLCV와 funding은 운영 수준으로 이미 존재한다.")
    if derivatives["coverage"] >= 0.95:
        strengths.append("BTC/BNB 파생 포지셔닝 6종은 이미 확보돼 있어 다음 단계는 확장보다 활용 고도화다.")
    if market_context["status"] != "missing":
        strengths.append("QQQ/SPY/GLD/DXY 교차자산 문맥은 이미 들어와 있다.")

    for row in strategic_layers:
        if row["status"] != "present":
            gaps.append(f"{row['priority']}. {row['key']} 레이어가 비어 있다.")

    return {
        "strengths": strengths,
        "critical_gaps": gaps,
        "primary_upgrade_thesis": (
            "현재 성과 상한은 검증 엔진보다 데이터 레이어 부족에서 올 가능성이 높다. "
            "특히 초단기 false entry 차단용 미시구조 데이터와 현물 수급 확인용 PiT 흐름 데이터가 가장 직접적이다."
        ),
    }


def build_strategy_data_report(project_root: Path, pairs: tuple[str, ...]) -> dict[str, Any]:
    ohlcv = summarize_ohlcv_layer(pairs)
    derivatives = summarize_derivatives_layer(pairs)
    market_context = summarize_market_context_layer()
    strategic_layers = summarize_strategic_layers()
    return {
        "generated_at": iso_now(),
        "project_root": str(project_root),
        "targets": DEFAULT_TARGETS,
        "local_layers": {
            "ohlcv": ohlcv,
            "derivatives": derivatives,
            "market_context": market_context,
        },
        "strategic_layers": strategic_layers,
        "executive_summary": build_executive_summary(ohlcv, derivatives, market_context, strategic_layers),
    }


def main() -> None:
    args = parse_args()
    report = build_strategy_data_report(ROOT_DIR, tuple(args.pairs))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"status": "ok", "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
