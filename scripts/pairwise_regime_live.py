#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv

import gp_crypto_evolution as gp
from btc_convex_blend import blend_runtime_weight, get_btc_convex_blend
from btc_event_blend import apply_runtime_event_blend, build_runtime_event_context_from_frame, get_btc_event_blend
from btc_online_blend import get_btc_online_blend, runtime_online_blend_alpha, update_runtime_online_score
from execution_gene_utils import normalize_execution_gene
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    _load_derivative_bundle,
    build_fast_context,
    build_overlay_inputs,
    build_library_lookup,
    build_route_bucket_codes,
    normalize_mapping_indices,
    normalize_route_state_mode,
    realistic_overlay_replay_from_context,
    route_state_names,
)

load_dotenv(ROOT / ".env")

PAIRS = ("BTCUSDT", "BNBUSDT")
# If PAIRWISE_PAIR_OVERRIDE is set (e.g. "ETHUSDT"), restrict trading to that single pair.
_pair_override = os.environ.get("PAIRWISE_PAIR_OVERRIDE", "").strip()
if _pair_override:
    PAIRS = (_pair_override,)
PAIR_TO_MARKET = {
    "BTCUSDT": "BTC/USDT:USDT",
    "BNBUSDT": "BNB/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
    "SOLUSDT": "SOL/USDT:USDT",
    "XRPUSDT": "XRP/USDT:USDT",
    "DOGEUSDT": "DOGE/USDT:USDT",
}

DEFAULT_SUMMARY_PATH = ROOT / "models" / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"
DEFAULT_MARKET_OS_SUMMARY_PATH = ROOT / "models" / "gp_regime_mixture_btc_bnb_pairwise_market_os_candidate_summary.json"
DEFAULT_VALIDATED_STRESS_REPORT_PATH = ROOT / "models" / "gp_regime_mixture_btc_bnb_pairwise_validated_stress_report.json"
DEFAULT_MODEL_PATH = ROOT / "models" / "recent_6m_gp_vectorized_big_capped_rerun.dill"
DEFAULT_PROMOTION_REPORT_PATH = ROOT / "models" / "gp_regime_mixture_btc_bnb_pairwise_market_os_pipeline_report.json"
DEFAULT_STATE_PATH = ROOT / "models" / "pairwise_regime_live_state.json"
DEFAULT_DECISION_LOG_PATH = ROOT / "logs" / "pairwise_regime_decisions.jsonl"
DEFAULT_SHADOW_STATE_PATH = ROOT / "models" / "pairwise_regime_shadow_state.json"
DEFAULT_SHADOW_DECISION_LOG_PATH = ROOT / "logs" / "pairwise_regime_shadow_decisions.jsonl"
PAIRWISE_HISTORY_START = "2022-04-06"

SHADOW_DEFAULT_EQUITY = 100_000.0
SHADOW_TRADING_COST_RATE = 0.0006
PROMOTION_STAGE_SPECS = (
    {"key": "day_1", "label": "1-day observe", "min_observations": 288},
    {"key": "day_3", "label": "3-day confirm", "min_observations": 864},
    {"key": "day_7", "label": "7-day promote", "min_observations": 2016},
)
SHADOW_PROMOTION_MIN_OBSERVATIONS = PROMOTION_STAGE_SPECS[-1]["min_observations"]
SHADOW_PROMOTION_MAX_DRAWDOWN = 0.18
SHADOW_PROMOTION_MIN_RETURN = 0.0
SHADOW_PROMOTION_MAX_STALE_MINUTES = 20
TARGET_WEIGHT_EPS = 1e-6
DEFAULT_POLL_SECONDS = 300
PAIRWISE_EQUITY_CORR_RISK_ENABLED = os.getenv("PAIRWISE_EQUITY_CORR_RISK", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DIRECTIONAL_GA_OVERLAY_ENABLED = os.getenv("PAIRWISE_DIRECTIONAL_GA_OVERLAY", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
DIRECTIONAL_GA_OVERLAY_MODE = os.getenv("PAIRWISE_DIRECTIONAL_GA_MODE", "gate").strip().lower() or "gate"
DIRECTIONAL_GA_OVERLAY_PATH = Path(
    os.getenv("PAIRWISE_DIRECTIONAL_GA_PATH", str(ROOT / "models" / "directional_genetic_overlay_candidate.json"))
)
# D2: per-pair gross weight cap (1.0 = no cap; set PAIRWISE_GROSS_CAP=0.01 for Stage A)
PAIRWISE_GROSS_CAP = float(os.getenv("PAIRWISE_GROSS_CAP", "1.0"))

# D1: max-hold-bars auto-flatten (24 h = 288 × 5-min bars)
PAIRWISE_MAX_HOLD_BARS = int(os.getenv("PAIRWISE_MAX_HOLD_BARS", "288"))
_MAX_HOLD_SECONDS = PAIRWISE_MAX_HOLD_BARS * 5 * 60  # 288 bars × 300 s = 86 400 s

# R3: CVaR-99 cut overlay
PAIRWISE_CVAR_CUT_ENABLED = os.getenv("PAIRWISE_CVAR_CUT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
PAIRWISE_CVAR_CUT_HOLD_HOURS = int(os.getenv("PAIRWISE_CVAR_CUT_HOLD_HOURS", "24"))

DEFAULT_TAIL_RISK_PATH = ROOT / "models" / "tail_risk_report.json"
DEFAULT_LIVE_PNL_PATH = ROOT / "models" / "live_actual_pnl_30d.json"


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_utc_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def minutes_since(value: Any) -> float:
    parsed = parse_utc_datetime(value)
    if parsed is None:
        return math.inf
    return max(0.0, (utc_now() - parsed).total_seconds() / 60.0)


def minutes_since_or_zero(value: Any) -> float:
    """Like minutes_since but returns 0.0 on parse failure instead of inf.

    Use where missing/unparseable input should be treated as 'just happened'
    (safe default) rather than 'infinitely stale' (gate-fail default).
    """
    parsed = parse_utc_datetime(value)
    if parsed is None:
        return 0.0
    return max(0.0, (utc_now() - parsed).total_seconds() / 60.0)


# ---------------------------------------------------------------------------
# D1 / R3 overlay helpers (unit-testable, no side effects)
# ---------------------------------------------------------------------------

def compute_position_age_seconds(open_since_ts: Any) -> float:
    """Return how many seconds a position has been open; 0.0 if no valid open timestamp."""
    parsed = parse_utc_datetime(open_since_ts)
    if parsed is None:
        return 0.0
    return max(0.0, (utc_now() - parsed).total_seconds())


def _exchange_position_is_open(positions: Mapping[str, Any], pair: str) -> bool:
    """True if the exchange reports a non-dust position for pair.

    Threshold 1e-9: sub-satoshi dust left after a near-complete fill should not
    keep the D1 timer alive; anything >= 1e-9 contracts is a real position.
    """
    pos = positions.get(pair)
    if pos is None:
        return False
    qty = pos.get("qty", 0.0)
    try:
        return abs(float(qty)) > 1e-9
    except (TypeError, ValueError):
        return False


def load_tail_risk_thresholds(path: Path = DEFAULT_TAIL_RISK_PATH) -> Dict[str, float]:
    """Return {pair: CVaR_99} from models/tail_risk_report.json.

    Returns an empty dict if the file is missing or malformed.
    """
    try:
        payload = json.loads(path.read_text())
        per_pair = payload.get("per_pair") or {}
        return {pair: float(per_pair[pair]["CVaR_99"]) for pair in per_pair if "CVaR_99" in per_pair[pair]}
    except Exception:
        return {}


def compute_rolling_30d_return(
    pair: str,
    path: Path = DEFAULT_LIVE_PNL_PATH,
    initial_equity: float = 100_000.0,
) -> float | None:
    """Return rolling 30-day realised return for *pair* (fraction, e.g. -0.03 = -3%).

    Uses models/live_actual_pnl_30d.json ``daily_pnl_live`` rows.
    Returns None if the file is missing / pair has no rows.
    """
    try:
        payload = json.loads(path.read_text())
        rows = payload.get("daily_pnl_live") or []
        equity_ref = float(payload.get("initial_equity_estimate") or initial_equity) or initial_equity
        cutoff = (utc_now() - timedelta(days=30)).date()
        total_pnl = 0.0
        found = False
        for row in rows:
            if str(row.get("pair")) != pair:
                continue
            try:
                row_date = datetime.fromisoformat(str(row["date"])).date()
            except (KeyError, ValueError):
                continue
            if row_date < cutoff:
                continue
            total_pnl += float(row.get("total", 0.0) or 0.0)
            found = True
        if not found:
            return None
        return total_pnl / equity_ref
    except Exception:
        return None


def should_apply_cvar_cut(
    pair: str,
    cvar_thresholds: Mapping[str, float],
    live_pnl_path: Path = DEFAULT_LIVE_PNL_PATH,
) -> bool:
    """Return True if 30-day realised return is below the pair's CVaR-99 threshold."""
    threshold = cvar_thresholds.get(pair)
    if threshold is None:
        return False
    rolling_return = compute_rolling_30d_return(pair, live_pnl_path)
    if rolling_return is None:
        return False
    # CVaR_99 from tail_risk_report is typically negative (loss); compare directly
    return rolling_return < threshold


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.reset_index().to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _state_tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def _state_backup_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.bak")


def _default_state() -> Dict[str, Any]:
    return {
        "strategy_class": "pairwise_regime_live",
        "created_at": iso_now(),
        "updated_at": iso_now(),
        "shadow_paper": {
            "enabled": True,
            "observations": 0,
            "source_mode": "shadow",
            "baseline_equity": SHADOW_DEFAULT_EQUITY,
            "equity": SHADOW_DEFAULT_EQUITY,
            "peak_equity": SHADOW_DEFAULT_EQUITY,
            "max_drawdown": 0.0,
            "return_pct": 0.0,
            "last_prices": {},
            "current_weights": {},
            "cooldown_bars_left": {},
            "turnover_cost_paid": 0.0,
        },
        "runtime_health": {
            "status": "idle",
            "consecutive_errors": 0,
            "last_error": None,
            "last_success_at": None,
        },
        "notification_state": {
            "position_loss_alerted": {},
        },
        "latest_runtime_snapshot": {},
        "latest_decision_snapshot": {},
        "promotion_gate": {},
        "decision_journal": [],
        "position_open_since_ts": {},
        "cvar_cut_until_ts": {},
    }


def ensure_shadow_paper_defaults(shadow: Dict[str, Any]) -> Dict[str, Any]:
    shadow.setdefault("enabled", True)
    shadow.setdefault("observations", 0)
    shadow.setdefault("source_mode", "shadow")
    shadow.setdefault("baseline_equity", SHADOW_DEFAULT_EQUITY)
    shadow.setdefault("equity", SHADOW_DEFAULT_EQUITY)
    shadow.setdefault("peak_equity", shadow["equity"])
    shadow.setdefault("max_drawdown", 0.0)
    shadow.setdefault("return_pct", 0.0)
    shadow.setdefault("last_prices", {})
    shadow.setdefault("current_weights", {})
    shadow.setdefault("cooldown_bars_left", {})
    shadow.setdefault("turnover_cost_paid", 0.0)
    return shadow


def _load_state_payload(path: Path) -> Dict[str, Any]:
    candidates = (path, _state_backup_path(path), _state_tmp_path(path))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
    recovered = _default_state()
    recovered["runtime_health"]["status"] = "recovered"
    recovered["runtime_health"]["last_error"] = "state_json_corrupted_recovered"
    recovered["runtime_health"]["last_success_at"] = iso_now()
    return recovered


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _default_state()
    state = _load_state_payload(path)
    notification_state = state.setdefault("notification_state", {})
    notification_state.setdefault("position_loss_alerted", {})
    ensure_shadow_paper_defaults(state.setdefault("shadow_paper", {}))
    return state


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    ensure_parent(path)
    payload = copy.deepcopy(dict(state))
    payload["updated_at"] = iso_now()
    encoded = json.dumps(json_ready(payload), indent=2, sort_keys=True)
    tmp_path = _state_tmp_path(path)
    backup_path = _state_backup_path(path)
    if path.exists():
        backup_path.write_text(path.read_text())
    tmp_path.write_text(encoded)
    tmp_path.replace(path)


def append_jsonl(path: Path, item: Mapping[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(item), sort_keys=True))
        handle.write("\n")


def load_selected_candidate(summary_path: Path) -> Dict[str, Any]:
    payload = json.loads(summary_path.read_text())
    selected = payload.get("selected_candidate")
    if not selected:
        raise ValueError(f"No selected_candidate found in {summary_path}")
    return payload


def _promotion_report_runtime_ready(report: Mapping[str, Any]) -> bool:
    decision = report.get("decision") if isinstance(report.get("decision"), Mapping) else {}
    readiness_flags = (
        report.get("ready_for_demo"),
        report.get("ready_for_shadow_live"),
        report.get("ready_for_live"),
        report.get("ready_for_merge"),
        decision.get("ready_for_demo"),
        decision.get("ready_for_shadow_live"),
        decision.get("ready_for_live"),
        decision.get("ready_for_merge"),
        decision.get("selected_candidate_ready_for_live"),
        decision.get("selected_candidate_ready_for_merge"),
    )
    return any(bool(flag) for flag in readiness_flags)


def resolve_runtime_summary_path(summary_path: Path, promotion_report_path: Path) -> Path:
    requested = Path(summary_path)
    if requested != DEFAULT_SUMMARY_PATH:
        return requested
    if not promotion_report_path.exists() or not DEFAULT_MARKET_OS_SUMMARY_PATH.exists():
        return requested
    try:
        report = json.loads(promotion_report_path.read_text())
    except json.JSONDecodeError:
        return requested
    if not isinstance(report, Mapping):
        return requested
    if not report.get("selected_candidate"):
        return requested
    if not _promotion_report_runtime_ready(report):
        return requested
    return DEFAULT_MARKET_OS_SUMMARY_PATH


def extract_strategy_artifact_reference(summary_payload: Mapping[str, Any], *field_names: str) -> str | None:
    containers = [
        summary_payload,
        summary_payload.get("search") or {},
        summary_payload.get("artifacts") or {},
    ]
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for field_name in field_names:
            value = container.get(field_name)
            if value:
                return str(value)
    return None


def resolve_strategy_artifact_path(path: str | Path, anchor_file: Path) -> Path:
    candidate = Path(path)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                anchor_file.parent / candidate,
                ROOT / candidate,
            ]
        )
    seen: set[str] = set()
    for option in candidates:
        key = str(option)
        if key in seen:
            continue
        seen.add(key)
        if option.exists():
            return option
    return candidates[0] if candidates else anchor_file.parent / candidate


def _same_resolved_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve(strict=False) == Path(right).resolve(strict=False)


def extract_report_summary_references(report_payload: Mapping[str, Any]) -> List[str]:
    references: List[str] = []
    containers = [
        report_payload,
        report_payload.get("artifacts") or {},
        report_payload.get("selected_candidate") or {},
        report_payload.get("selection") or {},
        report_payload.get("decision") or {},
        report_payload.get("promotion_decision") or {},
    ]
    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for field_name in (
            "summary_path",
            "search_summary",
            "selected_summary_path",
            "candidate_summary_path",
        ):
            value = container.get(field_name)
            if value:
                references.append(str(value))
    return references


def promotion_report_matches_summary(report_path: Path, summary_path: Path) -> bool:
    if not report_path.exists():
        return False
    try:
        payload = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return False
    if not isinstance(payload, Mapping):
        return False
    for raw_reference in extract_report_summary_references(payload):
        resolved_reference = resolve_strategy_artifact_path(raw_reference, report_path)
        if _same_resolved_path(resolved_reference, summary_path):
            return True
    return False


def resolve_runtime_promotion_report_path(summary_path: Path, promotion_report_path: Path) -> Path:
    requested = Path(promotion_report_path)
    candidate_reports: List[Path] = []
    for candidate in (
        requested,
        DEFAULT_PROMOTION_REPORT_PATH if _same_resolved_path(summary_path, DEFAULT_MARKET_OS_SUMMARY_PATH) else None,
        DEFAULT_VALIDATED_STRESS_REPORT_PATH if _same_resolved_path(summary_path, DEFAULT_SUMMARY_PATH) else None,
    ):
        if candidate is None:
            continue
        if any(_same_resolved_path(candidate, existing) for existing in candidate_reports):
            continue
        candidate_reports.append(candidate)
    for candidate in candidate_reports:
        if promotion_report_matches_summary(candidate, summary_path):
            return candidate
    return requested


def pandas_timeframe(interval: str) -> str:
    if interval.endswith("m"):
        return f"{interval[:-1]}min"
    if interval.endswith("h"):
        return f"{interval[:-1]}h"
    return interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise BTC/BNB demo/live runner")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
        cmd.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
        cmd.add_argument("--promotion-report", type=Path, default=DEFAULT_PROMOTION_REPORT_PATH)
        cmd.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
        cmd.add_argument("--decision-log-path", type=Path, default=DEFAULT_DECISION_LOG_PATH)
        cmd.add_argument("--equity", type=float, default=SHADOW_DEFAULT_EQUITY)
        cmd.add_argument("--refresh-live-data", dest="refresh_live_data", action="store_true")
        cmd.add_argument("--no-refresh-live-data", dest="refresh_live_data", action="store_false")
        cmd.set_defaults(refresh_live_data=True)

    add_common(sub.add_parser("status"))

    run_once = sub.add_parser("run-once")
    add_common(run_once)
    run_once.add_argument("--execute", action="store_true")
    run_once.add_argument("--force-execute", action="store_true")
    run_once.add_argument("--force-note", default="manual_primary_switch")
    run_once.add_argument("--mode", choices=("demo", "live"), default="demo")

    loop = sub.add_parser("loop")
    add_common(loop)
    loop.add_argument("--execute", action="store_true")
    loop.add_argument("--force-execute", action="store_true")
    loop.add_argument("--force-note", default="manual_primary_switch")
    loop.add_argument("--mode", choices=("demo", "live"), default="demo")
    loop.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)

    sync_state = sub.add_parser("sync-state")
    add_common(sync_state)
    sync_state.add_argument("--mode", choices=("demo", "live"), default="demo")
    sync_state.add_argument("--execute", action="store_true")

    shutdown_protect = sub.add_parser("shutdown-protect")
    add_common(shutdown_protect)
    shutdown_protect.add_argument("--mode", choices=("demo", "live"), default="demo")
    shutdown_protect.add_argument("--execute", action="store_true")

    close_all = sub.add_parser("close-all")
    add_common(close_all)
    close_all.add_argument("--mode", choices=("demo", "live"), default="demo")
    close_all.add_argument("--execute", action="store_true")

    return parser.parse_args()


def _drop_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize(UTC)
        df = df.copy()
        df.index = index
    else:
        index = index.tz_convert(UTC)
        df = df.copy()
        df.index = index
    current_bar_open = pd.Timestamp.utcnow().tz_convert(UTC).floor(pandas_timeframe(gp.TIMEFRAME))
    if index[-1] >= current_bar_open:
        return df.iloc[:-1].copy()
    return df


def _align_complete_common_pair_frame(
    df: pd.DataFrame,
    pairs: Iterable[str],
    *,
    max_stale_bars: int = 2,
) -> pd.DataFrame:
    required_cols = [f"{pair}_close" for pair in pairs]
    aligned = df.dropna(subset=required_cols).sort_index()
    if aligned.empty:
        raise RuntimeError("No complete common pair bars available after alignment.")
    latest_common = pd.Timestamp(aligned.index[-1])
    if latest_common.tzinfo is None:
        latest_common = latest_common.tz_localize(UTC)
    else:
        latest_common = latest_common.tz_convert(UTC)
    interval_delta = pd.Timedelta(pandas_timeframe(gp.TIMEFRAME))
    current_bar_open = pd.Timestamp(utc_now()).tz_convert(UTC).floor(pandas_timeframe(gp.TIMEFRAME))
    latest_completed_bar = current_bar_open - interval_delta
    stale_cap = latest_completed_bar - interval_delta * max(int(max_stale_bars), 0)
    if latest_common < stale_cap:
        raise RuntimeError(
            "Common pair frame is stale. "
            f"latest_common={latest_common.isoformat()} latest_completed_bar={latest_completed_bar.isoformat()}"
        )
    return aligned


def _merge_recent_pair_frame(df: pd.DataFrame, pair: str, recent: pd.DataFrame) -> pd.DataFrame:
    if recent.empty:
        return df
    pair_prefix = f"{pair}_"
    renamed = recent.rename(columns={column: f"{pair_prefix}{column}" for column in recent.columns})
    renamed = renamed.sort_index()
    existing_pair_cols = [column for column in df.columns if column.startswith(pair_prefix)]
    other = df.drop(columns=existing_pair_cols)
    current_pair = df[existing_pair_cols].copy()
    merged_pair = pd.concat([current_pair, renamed]).sort_index()
    merged_pair = merged_pair[~merged_pair.index.duplicated(keep="last")]
    merged = pd.concat([other, merged_pair], axis=1).sort_index()
    return merged


def load_live_frame(
    pairs: Iterable[str],
    refresh_live_data: bool,
    recent_days: int = 10,
) -> pd.DataFrame:
    df = gp.load_all_pairs(
        pairs=list(pairs),
        start=PAIRWISE_HISTORY_START,
        end=None,
        refresh_cache=False,
    )
    if refresh_live_data:
        start_dt = utc_now() - timedelta(days=recent_days)
        end_dt = utc_now()
        for pair in pairs:
            try:
                recent = gp.fetch_klines(pair, gp.TIMEFRAME, start_dt, end_dt)
                df = _merge_recent_pair_frame(df, pair, recent)
            except Exception as exc:
                print(f"  {pair}: live refresh skipped ({exc})")
                continue
    df = _drop_incomplete_bar(df)
    df = _align_complete_common_pair_frame(df, pairs)
    if len(df) < 20:
        raise RuntimeError("Not enough bars available for pairwise live planning")
    return df


def compute_requested_weight(
    raw_signal: np.ndarray,
    params: Any,
    regime_score: float,
    breadth_score: float,
    bar_vol_ann: float,
    *,
    equity_corr_gross_scale: float = 1.0,
    equity_corr_regime_mult: float = 1.0,
) -> float:
    smoothed = pd.Series(raw_signal).ewm(span=max(int(params.signal_span), 1), adjust=False).mean().to_numpy()
    signal_pct = float(np.nan_to_num(smoothed[-1], nan=0.0))
    requested_weight = signal_pct / 100.0
    effective_regime_threshold = float(params.regime_threshold) * float(equity_corr_regime_mult)
    effective_gross_cap = float(params.gross_cap) * float(equity_corr_gross_scale)
    long_ok = regime_score >= effective_regime_threshold and breadth_score >= float(params.breadth_threshold)
    short_ok = regime_score <= -effective_regime_threshold and breadth_score <= (1.0 - float(params.breadth_threshold))
    if requested_weight > 0.0 and not long_ok:
        signal_pct = 0.0
        requested_weight = 0.0
    elif requested_weight < 0.0 and not short_ok:
        signal_pct = 0.0
        requested_weight = 0.0
    if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
        vol_scale = min(
            float(params.target_vol_ann) / float(bar_vol_ann),
            float(effective_gross_cap) / max(abs(requested_weight), 1e-8),
        )
        requested_weight *= float(vol_scale)
    return float(np.clip(requested_weight, -float(effective_gross_cap), float(effective_gross_cap)))


def _append_synthetic_planning_bar(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    interval_delta = pd.Timedelta(pandas_timeframe(gp.TIMEFRAME))
    tail = df.iloc[[-1]].copy()
    next_index = pd.Timestamp(df.index[-1]) + interval_delta
    while next_index in df.index:
        next_index += interval_delta
    tail.index = pd.DatetimeIndex([next_index])
    return pd.concat([df, tail]).sort_index()


def _resolve_pair_state_specialists(pair_config: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw = pair_config.get("state_specialists")
    if not isinstance(raw, (list, tuple)):
        return None
    values = tuple(int(value) for value in raw)
    return values or None


def _trace_scalar(trace: Mapping[str, Any], key: str, index: int, default: float = 0.0) -> float:
    values = trace.get(key)
    if values is None:
        return float(default)
    arr = np.asarray(values)
    if index < 0 or index >= len(arr):
        return float(default)
    value = arr[index]
    return float(value) if np.isfinite(value) else float(default)


def _trace_int(trace: Mapping[str, Any], key: str, index: int, default: int = 0) -> int:
    values = trace.get(key)
    if values is None:
        return int(default)
    arr = np.asarray(values)
    if index < 0 or index >= len(arr):
        return int(default)
    return int(arr[index])


def _latest_trace_signal_index(trace: Mapping[str, Any], planning_frame_length: int) -> int:
    values = trace.get("target_weight")
    if values is not None:
        arr = np.asarray(values)
        if arr.size > 0:
            return max(0, int(arr.size - 1))
    # The planning frame includes one synthetic bar so the latest actionable
    # signal sits one bar before the last completed bar.
    return max(int(planning_frame_length) - 3, 0)


def _build_trace_driven_pair_plan(
    *,
    df: pd.DataFrame,
    pair: str,
    pair_config: Mapping[str, Any],
    raw_signal: np.ndarray,
    overlay_inputs: Mapping[str, Any],
    library: list[Any],
    library_lookup: Mapping[str, Any],
    current_weight: float,
    derivative_bundle: Mapping[str, pd.DataFrame] | None,
) -> Dict[str, Any]:
    route_state_mode = normalize_route_state_mode(pair_config.get("route_state_mode"))
    route_threshold = float(pair_config["route_breadth_threshold"])
    mapping = normalize_mapping_indices(pair_config["mapping_indices"], route_state_mode)
    raw_signal_series = pd.Series(np.asarray(raw_signal, dtype="float64"), index=pd.DatetimeIndex(df.index))
    fast_context = build_fast_context(
        df=df,
        pair=pair,
        raw_signal=raw_signal_series,
        overlay_inputs=dict(overlay_inputs),
        route_thresholds=(route_threshold,),
        library_lookup=dict(library_lookup),
        derivative_bundle=None if derivative_bundle is None else dict(derivative_bundle),
        route_state_mode=route_state_mode,
    )
    execution_gene = None
    if isinstance(pair_config.get("execution_gene"), Mapping):
        execution_gene = normalize_execution_gene(pair_config.get("execution_gene"))
    result = realistic_overlay_replay_from_context(
        fast_context,
        dict(library_lookup),
        mapping,
        route_threshold,
        use_equity_corr_risk=PAIRWISE_EQUITY_CORR_RISK_ENABLED,
        execution_gene=execution_gene,
        state_specialists=_resolve_pair_state_specialists(pair_config),
        engine="python",
        return_trace=True,
    )
    trace = result.get("trace") or {}
    signal_index = _latest_trace_signal_index(trace, len(df))
    bucket_codes = np.asarray(fast_context["bucket_codes"][route_threshold], dtype="int64")
    bucket_code = int(bucket_codes[signal_index])
    active_index = int(mapping[bucket_code])
    params = library[active_index]
    day_index = pd.DatetimeIndex(df.index).normalize()
    theoretical_current_weight = _trace_scalar(trace, "target_weight", signal_index - 1, default=0.0)
    equity_corr_value = float(fast_context["equity_corr"][signal_index])
    return {
        "price": float(df[f"{pair}_close"].iloc[signal_index]),
        "current_weight": float(current_weight),
        "policy_current_weight": float(theoretical_current_weight),
        "state_drift": float(current_weight - theoretical_current_weight),
        "requested_weight": _trace_scalar(trace, "requested_weight", signal_index, default=0.0),
        "target_weight": _trace_scalar(trace, "target_weight", signal_index, default=0.0),
        "rebalance_due": bool(signal_index % max(int(params.rebalance_bars), 1) == 0),
        "cooldown_bars_left_after": _trace_int(trace, "cooldown_bars_left", signal_index, default=0),
        "route_bucket": bucket_code,
        "route_state_mode": route_state_mode,
        "route_state_name": route_state_names(route_state_mode)[bucket_code],
        "route_mapping_index": active_index,
        "params": asdict(params),
        "regime_score": float(fast_context["regime"][signal_index]),
        "breadth_score": float(fast_context["breadth"][signal_index]),
        "bar_vol_ann": float(fast_context["vol_ann"][signal_index]),
        "equity_corr_value": equity_corr_value if np.isfinite(equity_corr_value) else None,
        "equity_corr_bucket": str(
            overlay_inputs["equity_corr_bucket_daily"].reindex(day_index, method="ffill").fillna("equity_unknown").iloc[signal_index]
        ),
        "equity_corr_quantile_state": str(
            overlay_inputs["equity_corr_quantile_state_daily"].reindex(day_index, method="ffill").fillna("missing").iloc[signal_index]
        ),
        "equity_corr_context": overlay_inputs.get("equity_corr_context"),
        "equity_corr_source_mode": overlay_inputs.get("equity_corr_source_mode"),
        "equity_corr_gross_scale": float(fast_context["equity_corr_gross_scale"][signal_index]),
        "equity_corr_regime_threshold_mult": float(fast_context["equity_corr_regime_mult"][signal_index]),
        "signal_value": float(np.nan_to_num(raw_signal[signal_index], nan=0.0)),
        "signal_pct": _trace_scalar(trace, "signal_pct", signal_index, default=0.0),
        "role_idx": _trace_int(trace, "role_idx", signal_index, default=0),
    }


def build_pairwise_plan(
    summary_path: Path,
    model_path: Path,
    promotion_report_path: Path,
    refresh_live_data: bool,
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved_summary_path = resolve_runtime_summary_path(summary_path, promotion_report_path)
    resolved_promotion_report_path = resolve_runtime_promotion_report_path(
        resolved_summary_path,
        promotion_report_path,
    )
    summary = load_selected_candidate(resolved_summary_path)
    embedded_model_ref = extract_strategy_artifact_reference(summary, "model_path")
    resolved_model_path = resolve_strategy_artifact_path(embedded_model_ref or model_path, resolved_summary_path)
    config = summary["selected_candidate"]["pair_configs"]
    library = list(iter_params())
    library_lookup = build_library_lookup(library)
    model_tree, _ = load_signal_model(resolved_model_path)
    compiled = gp.toolbox.compile(expr=model_tree)
    df = load_live_frame(PAIRS, refresh_live_data=refresh_live_data)
    df_planning = _append_synthetic_planning_bar(df)
    signal_index = len(df) - 1
    shadow = state.get("shadow_paper", {})
    current_weights = shadow.get("current_weights", {})

    pair_plans: Dict[str, Any] = {}
    target_weights: Dict[str, float] = {}
    latest_prices: Dict[str, float] = {}
    online_blend_state = state.setdefault("btc_online_blend_state", {})
    event_blend_state = state.setdefault("btc_event_blend_state", {})
    derivative_bundles: Dict[str, Mapping[str, pd.DataFrame] | None] = {}

    for pair in PAIRS:
        raw_signal = np.asarray(compiled(*gp.get_feature_arrays(df_planning, pair)), dtype=float)
        overlay_inputs = build_overlay_inputs(df_planning, PAIRS, regime_pair=pair)
        if pair not in derivative_bundles:
            try:
                derivative_bundles[pair] = _load_derivative_bundle(pair)
            except Exception:
                derivative_bundles[pair] = None

        baseline_plan = _build_trace_driven_pair_plan(
            df=df_planning,
            pair=pair,
            pair_config=config[pair],
            raw_signal=raw_signal,
            overlay_inputs=overlay_inputs,
            library=library,
            library_lookup=library_lookup,
            current_weight=float(current_weights.get(pair, 0.0)),
            derivative_bundle=derivative_bundles[pair],
        )
        final_plan = baseline_plan
        blend = get_btc_convex_blend(summary["selected_candidate"], pair)
        specialist_plan: Dict[str, Any] | None = None
        if blend is not None:
            specialist_plan = _build_trace_driven_pair_plan(
                df=df_planning,
                pair=pair,
                pair_config=blend["specialist_pair_config"],
                raw_signal=raw_signal,
                overlay_inputs=overlay_inputs,
                library=library,
                library_lookup=library_lookup,
                current_weight=float(current_weights.get(pair, 0.0)),
                derivative_bundle=derivative_bundles[pair],
            )
            final_plan = dict(baseline_plan)
            final_plan["requested_weight"] = blend_runtime_weight(
                baseline_weight=float(baseline_plan["requested_weight"]),
                specialist_weight=float(specialist_plan["requested_weight"]),
                route_state_name=str(baseline_plan["route_state_name"]),
                alpha=float(blend["alpha"]),
                mode=str(blend["mode"]),
                state_alphas=blend.get("state_alphas"),
            )
            final_plan["target_weight"] = blend_runtime_weight(
                baseline_weight=float(baseline_plan["target_weight"]),
                specialist_weight=float(specialist_plan["target_weight"]),
                route_state_name=str(baseline_plan["route_state_name"]),
                alpha=float(blend["alpha"]),
                mode=str(blend["mode"]),
                state_alphas=blend.get("state_alphas"),
            )
            final_plan["blend"] = {
                "alpha": float(blend["alpha"]),
                "mode": str(blend["mode"]),
                "state_alphas": dict(blend.get("state_alphas") or {}),
                "specialist_target_weight": float(specialist_plan["target_weight"]),
                "specialist_requested_weight": float(specialist_plan["requested_weight"]),
                "specialist_route_state_name": str(specialist_plan["route_state_name"]),
            }
        online = get_btc_online_blend(summary["selected_candidate"], pair)
        if online is not None and specialist_plan is not None:
            pair_state = dict(online_blend_state.get(pair) or {})
            baseline_requested_weight = float(final_plan["requested_weight"])
            baseline_target_weight = float(final_plan["target_weight"])
            current_price = float(final_plan["price"])
            score = update_runtime_online_score(
                previous_score=float(pair_state.get("score", 0.0)),
                baseline_weight=float(pair_state.get("baseline_target_weight", 0.0)),
                specialist_weight=float(pair_state.get("specialist_target_weight", 0.0)),
                previous_price=pair_state.get("price"),
                current_price=current_price,
                decay=float(online["decay"]),
                reward_scale=float(online["reward_scale"]),
            )
            online_alpha = runtime_online_blend_alpha(
                previous_score=score,
                alpha_cap=float(online["alpha_cap"]),
                eta=float(online["eta"]),
                activation_mode=str(online["activation_mode"]),
                route_state_name=str(final_plan["route_state_name"]),
                baseline_weight=baseline_requested_weight,
                specialist_weight=float(specialist_plan["requested_weight"]),
            )
            if online_alpha > 0.0:
                final_plan["requested_weight"] = (1.0 - float(online_alpha)) * baseline_requested_weight + float(online_alpha) * float(
                    specialist_plan["requested_weight"]
                )
                final_plan["target_weight"] = (1.0 - float(online_alpha)) * baseline_target_weight + float(online_alpha) * float(
                    specialist_plan["target_weight"]
                )
            final_plan["online_blend"] = {
                "alpha": float(online_alpha),
                "score": float(score),
                "activation_mode": str(online["activation_mode"]),
                "alpha_cap": float(online["alpha_cap"]),
                "eta": float(online["eta"]),
                "decay": float(online["decay"]),
                "baseline_target_weight": baseline_target_weight,
                "specialist_target_weight": float(specialist_plan["target_weight"]),
                "specialist_requested_weight": float(specialist_plan["requested_weight"]),
            }
            online_blend_state[pair] = {
                "score": float(score),
                "alpha": float(online_alpha),
                "price": current_price,
                "baseline_target_weight": baseline_target_weight,
                "specialist_target_weight": float(specialist_plan["target_weight"]),
                "baseline_requested_weight": baseline_requested_weight,
                "specialist_requested_weight": float(specialist_plan["requested_weight"]),
                "route_state_name": str(final_plan["route_state_name"]),
                "updated_at": iso_now(),
            }
        event = get_btc_event_blend(summary["selected_candidate"], pair)
        if event is not None:
            event_context = build_runtime_event_context_from_frame(
                df,
                pair,
                derivative_bundle=derivative_bundles.get(pair),
            )
            final_plan, next_event_state = apply_runtime_event_blend(
                context=event_context,
                baseline_plan=final_plan,
                pair_state=event_blend_state.get(pair),
                event=event,
            )
            next_event_state["updated_at"] = iso_now()
            event_blend_state[pair] = next_event_state

        close_price = float(df[f"{pair}_close"].iloc[signal_index])
        final_plan["price"] = close_price
        latest_prices[pair] = close_price
        target_weights[pair] = float(final_plan["target_weight"])
        pair_plans[pair] = {
            **final_plan,
        }

    directional_ga_overlay: Dict[str, Any] | None = None
    if DIRECTIONAL_GA_OVERLAY_ENABLED:
        try:
            from directional_genetic_overlay import apply_overlay_to_targets, compute_overlay_targets, load_candidate

            overlay_candidate = load_candidate(DIRECTIONAL_GA_OVERLAY_PATH)
            overlay_targets = compute_overlay_targets(df, overlay_candidate, PAIRS)
            adjusted_targets, overlay_details = apply_overlay_to_targets(
                target_weights,
                overlay_targets,
                mode=DIRECTIONAL_GA_OVERLAY_MODE,
            )
            for pair in PAIRS:
                target_weights[pair] = float(adjusted_targets.get(pair, target_weights.get(pair, 0.0)))
                pair_plans[pair]["directional_ga_overlay"] = overlay_details["pairs"].get(pair, {})
                pair_plans[pair]["target_weight"] = target_weights[pair]
            directional_ga_overlay = {
                "enabled": True,
                "status": "applied",
                "path": str(DIRECTIONAL_GA_OVERLAY_PATH),
                **overlay_details,
            }
        except Exception as exc:
            directional_ga_overlay = {
                "enabled": True,
                "status": "error",
                "path": str(DIRECTIONAL_GA_OVERLAY_PATH),
                "error": str(exc),
            }

    gross = float(sum(abs(weight) for weight in target_weights.values()))
    net = float(sum(target_weights.values()))
    return {
        "generated_at": iso_now(),
        "strategy_class": "pairwise_regime_live",
        "session_type": "pairwise" if gross > TARGET_WEIGHT_EPS else "flat",
        "bars_seen": int(len(df)),
        "signal_timestamp": str(pd.Timestamp(df.index[signal_index]).isoformat()),
        "target_weights": target_weights,
        "pair_plans": pair_plans,
        "gross_leverage": gross,
        "net_exposure": net,
        "latest_prices": latest_prices,
        "equity_corr_risk_enabled": PAIRWISE_EQUITY_CORR_RISK_ENABLED,
        "summary_path": str(resolved_summary_path),
        "promotion_report_path": str(resolved_promotion_report_path),
        "model_path": str(resolved_model_path),
        "directional_ga_overlay": directional_ga_overlay,
    }


def sync_shadow_paper_from_live_positions(
    state: Dict[str, Any],
    positions_by_pair: Mapping[str, Mapping[str, Any]],
    equity: float,
) -> None:
    shadow = ensure_shadow_paper_defaults(state.setdefault("shadow_paper", {}))

    current_weights: Dict[str, float] = {}
    last_prices: Dict[str, float] = {}
    for pair in PAIRS:
        position = positions_by_pair.get(pair) or {}
        qty = float(position.get("qty", 0.0) or 0.0)
        price = float(position.get("mark_price") or position.get("entry_price") or 0.0)
        if price > 0.0:
            last_prices[pair] = price
        else:
            last_prices[pair] = float(shadow.get("last_prices", {}).get(pair, 0.0) or 0.0)
        if abs(equity) > TARGET_WEIGHT_EPS and last_prices[pair] > 0.0:
            current_weights[pair] = float(qty * last_prices[pair] / equity)
        else:
            current_weights[pair] = 0.0

    source_mode = str(shadow.get("source_mode") or "shadow").strip().lower()
    if source_mode != "live":
        baseline_equity = float(equity)
        peak_equity = float(equity)
        max_drawdown = 0.0
        shadow["turnover_cost_paid"] = 0.0
    else:
        baseline_equity = float(shadow.get("baseline_equity") or 0.0)
        if abs(baseline_equity) <= TARGET_WEIGHT_EPS:
            baseline_equity = float(equity)
        peak_equity = max(float(shadow.get("peak_equity", equity) or equity), float(equity))
        drawdown = 0.0 if peak_equity <= TARGET_WEIGHT_EPS else max(0.0, 1.0 - float(equity) / peak_equity)
        max_drawdown = max(float(shadow.get("max_drawdown", 0.0) or 0.0), drawdown)

    shadow["source_mode"] = "live"
    shadow["baseline_equity"] = baseline_equity
    shadow["equity"] = float(equity)
    shadow["peak_equity"] = peak_equity
    shadow["max_drawdown"] = max_drawdown
    shadow["return_pct"] = (
        0.0
        if abs(baseline_equity) <= TARGET_WEIGHT_EPS
        else (float(equity) / baseline_equity - 1.0) * 100.0
    )
    shadow["current_weights"] = current_weights
    shadow["last_prices"] = last_prices
    shadow["last_updated_at"] = iso_now()


def persist_runtime_plan_state(state: Dict[str, Any], plan: Mapping[str, Any]) -> None:
    shadow = ensure_shadow_paper_defaults(state.setdefault("shadow_paper", {}))
    pair_plans = plan.get("pair_plans") or {}
    cooldown_state = dict(shadow.get("cooldown_bars_left") or {})
    for pair in PAIRS:
        pair_plan = pair_plans.get(pair)
        if not isinstance(pair_plan, Mapping):
            continue
        cooldown_state[pair] = int(pair_plan.get("cooldown_bars_left_after", cooldown_state.get(pair, 0)) or 0)
    shadow["cooldown_bars_left"] = cooldown_state

    latest_prices = plan.get("latest_prices") or {}
    if isinstance(latest_prices, Mapping):
        merged_prices = dict(shadow.get("last_prices") or {})
        for pair in PAIRS:
            price = latest_prices.get(pair)
            if price in (None, ""):
                continue
            merged_prices[pair] = float(price)
        shadow["last_prices"] = merged_prices

    signal_timestamp = plan.get("signal_timestamp")
    if signal_timestamp:
        shadow["last_signal_timestamp"] = str(signal_timestamp)
    shadow["last_updated_at"] = iso_now()


def apply_shadow_mark_to_market(state: Dict[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
    shadow = ensure_shadow_paper_defaults(state.setdefault("shadow_paper", {}))
    shadow["source_mode"] = "shadow"
    shadow["baseline_equity"] = float(shadow.get("baseline_equity") or SHADOW_DEFAULT_EQUITY)

    last_prices = shadow["last_prices"]
    current_weights = shadow["current_weights"]
    current_equity = float(shadow["equity"])
    peak_equity = float(shadow["peak_equity"])
    latest_prices = plan["latest_prices"]

    if last_prices:
        weighted_return = 0.0
        for pair in PAIRS:
            prev_price = float(last_prices.get(pair, latest_prices[pair]))
            current_price = float(latest_prices[pair])
            if prev_price > 0:
                pair_return = current_price / prev_price - 1.0
                weighted_return += float(current_weights.get(pair, 0.0)) * pair_return
        current_equity *= 1.0 + weighted_return

    target_weights = plan["target_weights"]
    turnover = float(sum(abs(float(target_weights.get(pair, 0.0)) - float(current_weights.get(pair, 0.0))) for pair in PAIRS))
    cost = current_equity * turnover * SHADOW_TRADING_COST_RATE
    current_equity -= cost
    peak_equity = max(peak_equity, current_equity)
    drawdown = 0.0 if peak_equity <= 0 else 1.0 - current_equity / peak_equity

    shadow["observations"] = int(shadow["observations"]) + 1
    shadow["equity"] = current_equity
    shadow["peak_equity"] = peak_equity
    shadow["max_drawdown"] = max(float(shadow["max_drawdown"]), drawdown)
    shadow["return_pct"] = (current_equity / float(shadow["baseline_equity"]) - 1.0) * 100.0
    shadow["turnover_cost_paid"] = float(shadow["turnover_cost_paid"]) + cost
    shadow["current_weights"] = {pair: float(target_weights.get(pair, 0.0)) for pair in PAIRS}
    shadow["cooldown_bars_left"] = {
        pair: int(plan["pair_plans"][pair]["cooldown_bars_left_after"]) for pair in PAIRS
    }
    shadow["last_prices"] = {pair: float(latest_prices[pair]) for pair in PAIRS}
    shadow["last_signal_timestamp"] = plan["signal_timestamp"]
    shadow["last_updated_at"] = iso_now()

    return {
        "equity": current_equity,
        "peak_equity": peak_equity,
        "max_drawdown": drawdown,
        "turnover_cost": cost,
        "turnover": turnover,
        "return_pct": shadow["return_pct"],
    }


def build_shadow_evaluation(state: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    shadow = state.get("shadow_paper", {})
    observations = int(shadow.get("observations", 0))
    max_drawdown = float(shadow.get("max_drawdown", 0.0))
    return_pct = float(shadow.get("return_pct", 0.0))
    last_updated_at_raw = shadow.get("last_updated_at") or state.get("updated_at")
    last_updated_at = parse_utc_datetime(last_updated_at_raw)
    stale_minutes = minutes_since(last_updated_at_raw)
    last_signal_timestamp_raw = shadow.get("last_signal_timestamp")
    signal_stale_minutes = minutes_since(last_signal_timestamp_raw)
    shadow_feed_stale = bool(stale_minutes > args.max_stale_minutes)
    shadow_signal_stale = bool(signal_stale_minutes > args.max_stale_minutes)
    runtime_health = state.get("runtime_health", {})

    def evaluate_gate(min_observations: int) -> Dict[str, Any]:
        passed = True
        reasons: List[str] = []
        if observations < min_observations:
            passed = False
            reasons.append(f"shadow observations {observations} < required {min_observations}")
        if max_drawdown > args.max_drawdown:
            passed = False
            reasons.append(f"max drawdown {max_drawdown:.2%} > cap {args.max_drawdown:.2%}")
        if return_pct < args.min_return:
            passed = False
            reasons.append(f"shadow return {return_pct:.2f}% < floor {args.min_return:.2f}%")
        if shadow_feed_stale:
            passed = False
            reasons.append(f"shadow feed stale {stale_minutes:.1f}m > cap {args.max_stale_minutes:.1f}m")
        if shadow_signal_stale:
            passed = False
            reasons.append(f"shadow signal stale {signal_stale_minutes:.1f}m > cap {args.max_stale_minutes:.1f}m")
        if int(runtime_health.get("consecutive_errors", 0)) > 0:
            passed = False
            reasons.append("runtime health has consecutive errors")
        return {
            "passed": passed,
            "reasons": reasons,
            "remaining_observations": max(0, int(min_observations) - observations),
            "min_observations": int(min_observations),
        }

    requested_gate = evaluate_gate(int(args.min_observations))
    stages: Dict[str, Any] = {}
    completed_stages: List[str] = []
    next_stage: Optional[Dict[str, Any]] = None
    for spec in PROMOTION_STAGE_SPECS:
        gate = evaluate_gate(int(spec["min_observations"]))
        stage_result = {
            "key": spec["key"],
            "label": spec["label"],
            **gate,
        }
        stages[spec["key"]] = stage_result
        if gate["passed"]:
            completed_stages.append(spec["key"])
        elif next_stage is None:
            next_stage = {
                "key": spec["key"],
                "label": spec["label"],
                "remaining_observations": gate["remaining_observations"],
            }

    if completed_stages:
        latest_key = completed_stages[-1]
        current_stage = {
            "key": latest_key,
            "label": stages[latest_key]["label"],
        }
    else:
        current_stage = {
            "key": "collecting",
            "label": "Collecting evidence",
        }

    final_stage_key = PROMOTION_STAGE_SPECS[-1]["key"]
    promotion_ready = bool(stages[final_stage_key]["passed"])
    return {
        "promotion_ready": promotion_ready,
        "reasons": stages[final_stage_key]["reasons"],
        "requested_gate_ready": requested_gate["passed"],
        "requested_gate": requested_gate,
        "current_stage": current_stage,
        "next_stage": next_stage,
        "completed_stages": completed_stages,
        "stages": stages,
        "observations": observations,
        "max_drawdown": max_drawdown,
        "return_pct": return_pct,
        "stale_minutes": stale_minutes,
        "signal_stale_minutes": signal_stale_minutes,
        "shadow_feed_stale": shadow_feed_stale,
        "shadow_signal_stale": shadow_signal_stale,
        "last_updated_at": last_updated_at.isoformat() if last_updated_at else None,
        "last_signal_timestamp": str(last_signal_timestamp_raw) if last_signal_timestamp_raw else None,
        "runtime_health": runtime_health,
    }


def default_promotion_eval_args(state_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        state_path=state_path,
        min_observations=SHADOW_PROMOTION_MIN_OBSERVATIONS,
        max_drawdown=SHADOW_PROMOTION_MAX_DRAWDOWN,
        min_return=SHADOW_PROMOTION_MIN_RETURN,
        max_stale_minutes=SHADOW_PROMOTION_MAX_STALE_MINUTES,
    )


def load_execution_bridge():
    import rotation_target_050_live as rotation

    rotation.PAIR_TO_MARKET = dict(PAIR_TO_MARKET)
    rotation.MARKET_TO_PAIR = {market: pair for pair, market in rotation.PAIR_TO_MARKET.items()}
    rotation.PAIRS = list(PAIRS)
    return rotation


def load_notification_bridge():
    import rotation_target_050_live as rotation

    return rotation


def load_promotion_gate(report_path: Path) -> Dict[str, Any]:
    if not report_path.exists():
        return {
            "status": "missing",
            "ready_for_demo": False,
            "ready_for_shadow_live": False,
            "ready_for_live": False,
            "ready_for_merge": False,
            "shadow_required": True,
            "manual_override_active": False,
            "manual_live_override_blocked": False,
            "failed_checks": ["promotion_report_missing"],
            "path": str(report_path),
            "decision": {},
        }
    payload = json.loads(report_path.read_text())
    decision = payload.get("decision")
    if not isinstance(decision, Mapping):
        decision = payload.get("promotion_decision")
    if not isinstance(decision, Mapping):
        decision = {}
    manual_override = payload.get("manual_override")
    if not isinstance(manual_override, Mapping):
        manual_override = decision.get("manual_override")
    if not isinstance(manual_override, Mapping):
        manual_override = {}
    manual_override_active = bool(manual_override.get("enabled", False))
    effective_decision: Dict[str, Any] = dict(decision)
    base_ready_for_merge = bool(
        decision.get("ready_for_merge", decision.get("selected_candidate_ready_for_merge", False))
    )
    base_ready_for_live = bool(
        decision.get("ready_for_live", decision.get("selected_candidate_ready_for_live", base_ready_for_merge))
    )
    base_ready_for_demo = bool(
        decision.get("ready_for_demo", decision.get("ready_for_shadow_live", base_ready_for_live))
    )
    manual_live_override_blocked = False
    if manual_override_active:
        manual_ready_for_demo = bool(
            manual_override.get(
                "ready_for_demo",
                manual_override.get("ready_for_shadow_live", base_ready_for_demo),
            )
        )
        requested_live_ready = bool(manual_override.get("ready_for_live", base_ready_for_live))
        requested_merge_ready = bool(manual_override.get("ready_for_merge", base_ready_for_merge))
        effective_decision["ready_for_demo"] = manual_ready_for_demo
        effective_decision["ready_for_shadow_live"] = manual_ready_for_demo
        effective_decision["ready_for_live"] = bool(requested_live_ready and base_ready_for_live)
        effective_decision["ready_for_merge"] = bool(requested_merge_ready and base_ready_for_merge)
        manual_live_override_blocked = bool(
            (requested_live_ready and not base_ready_for_live)
            or (requested_merge_ready and not base_ready_for_merge)
        )
        if effective_decision["ready_for_live"]:
            effective_decision["status"] = str(manual_override.get("status", "manually_promoted_ready_for_live"))
        elif effective_decision["ready_for_shadow_live"]:
            effective_decision["status"] = "demo_ready_only"
        else:
            effective_decision["status"] = str(effective_decision.get("status", "blocked"))
        effective_decision["manual_override"] = dict(manual_override)
        effective_decision["manual_live_override_blocked"] = manual_live_override_blocked
    ready_for_demo = bool(
        effective_decision.get(
            "ready_for_demo",
            effective_decision.get("ready_for_shadow_live", base_ready_for_demo),
        )
    )
    ready_for_merge = bool(
        effective_decision.get("ready_for_merge", effective_decision.get("selected_candidate_ready_for_merge", False))
    )
    ready_for_live = bool(
        effective_decision.get("ready_for_live", effective_decision.get("selected_candidate_ready_for_live", ready_for_merge))
    )
    ready_for_shadow_live = bool(
        effective_decision.get(
            "ready_for_shadow_live",
            effective_decision.get("selected_candidate_ready_for_live", ready_for_live),
        )
    )
    shadow_required = False
    return {
        "status": str(effective_decision.get("status", "unknown")),
        "ready_for_demo": ready_for_demo,
        "ready_for_shadow_live": ready_for_shadow_live,
        "ready_for_live": ready_for_live,
        "ready_for_merge": ready_for_merge,
        "shadow_required": shadow_required,
        "manual_override_active": manual_override_active,
        "manual_live_override_blocked": manual_live_override_blocked,
        "failed_checks": list(effective_decision.get("failed_checks") or []),
        "path": str(report_path),
        "decision": effective_decision,
    }


def promotion_gate_allows_execution(gate: Mapping[str, Any], mode: str) -> bool:
    mode_name = str(mode or "demo").lower()
    if mode_name == "demo":
        return bool(
            gate.get(
                "ready_for_demo",
                gate.get(
                    "ready_for_shadow_live",
                    gate.get("ready_for_live", gate.get("ready_for_merge", False)),
                ),
            )
        )
    return bool(gate.get("ready_for_live", gate.get("ready_for_merge", False)))


def sync_position_loss_notifications(state: Dict[str, Any], positions_by_pair: Mapping[str, Mapping[str, Any]]) -> None:
    snapshot = state.setdefault("latest_runtime_snapshot", {})
    snapshot["positions"] = [dict(position) for position in positions_by_pair.values()]
    notification_state = state.setdefault("notification_state", {})
    notification_state.setdefault("position_loss_alerted", {})
    rotation = load_notification_bridge()
    messages = rotation.collect_position_loss_notifications(state)
    rotation.dispatch_notifications(state, messages)


def record_runtime_success(state: Dict[str, Any], plan: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None) -> None:
    generated_at = iso_now()
    state.setdefault("runtime_health", {})
    state["runtime_health"].update(
        {
            "status": "ok",
            "consecutive_errors": 0,
            "last_error": None,
            "pid": os.getpid(),
            "last_loop_started_at": state.get("runtime_health", {}).get("last_loop_started_at") or generated_at,
            "last_loop_completed_at": generated_at,
            "last_success_at": generated_at,
        }
    )
    state.update(
        {
            "generated_at": generated_at,
            "summary_path": plan.get("summary_path"),
            "model_path": plan.get("model_path"),
            "pid": os.getpid(),
            "last_signal_timestamp": plan.get("signal_timestamp"),
            "last_loop_started_at": state.get("runtime_health", {}).get("last_loop_started_at"),
            "last_loop_completed_at": generated_at,
            "last_success_at": generated_at,
        }
    )
    state["latest_runtime_snapshot"] = {
        "generated_at": generated_at,
        "summary_path": plan.get("summary_path"),
        "model_path": plan.get("model_path"),
        "plan": plan,
        "extra": extra or {},
    }
    state["latest_decision_snapshot"] = {
        "generated_at": generated_at,
        "strategy_class": "pairwise_regime_live",
        "session_type": plan.get("session_type"),
        "target_weights": plan.get("target_weights", {}),
        "pair_plans": plan.get("pair_plans", {}),
        "rationale": {
            "mode": "pairwise",
            "gross_leverage": plan.get("gross_leverage", 0.0),
            "net_exposure": plan.get("net_exposure", 0.0),
            "signal_timestamp": plan.get("signal_timestamp"),
        },
    }
    journal = state.setdefault("decision_journal", [])
    journal.append(
        {
            "at": generated_at,
            "session_type": plan.get("session_type"),
            "target_weights": plan.get("target_weights"),
        }
    )
    if len(journal) > 200:
        del journal[:-200]


def record_runtime_error(state: Dict[str, Any], exc: Exception) -> None:
    runtime_health = state.setdefault("runtime_health", {})
    runtime_health["status"] = "error"
    runtime_health["consecutive_errors"] = int(runtime_health.get("consecutive_errors", 0)) + 1
    runtime_health["last_error"] = f"{type(exc).__name__}: {exc}"
    runtime_health["pid"] = os.getpid()
    runtime_health["last_error_at"] = iso_now()


def render_status(state: Mapping[str, Any], plan: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "strategy_class": state.get("strategy_class", "pairwise_regime_live"),
        "runtime_health": state.get("runtime_health", {}),
        "shadow_paper": state.get("shadow_paper", {}),
        "promotion_gate": state.get("promotion_gate", {}),
        "latest_runtime_snapshot": state.get("latest_runtime_snapshot", {}),
        "latest_decision_snapshot": state.get("latest_decision_snapshot", {}),
    }
    if plan is not None:
        payload["preview_plan"] = plan
    return payload


def run_status(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    plan = build_pairwise_plan(args.summary_path, args.model_path, args.promotion_report, args.refresh_live_data, state)
    print(json.dumps(json_ready(render_status(state, plan)), indent=2, sort_keys=True))
    return 0


def run_shadow_once(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    try:
        plan = build_pairwise_plan(args.summary_path, args.model_path, args.promotion_report, args.refresh_live_data, state)
        shadow_update = apply_shadow_mark_to_market(state, plan)
        promotion_gate = build_shadow_evaluation(state, default_promotion_eval_args(args.state_path))
        state["promotion_gate"] = promotion_gate
        record_runtime_success(state, plan, extra={"shadow_update": shadow_update})
        append_jsonl(
            args.decision_log_path,
            {
                "at": iso_now(),
                "mode": "shadow",
                "plan": plan,
                "shadow_update": shadow_update,
                "promotion_gate": promotion_gate,
            },
        )
        save_state(args.state_path, state)
        print(
            json.dumps(
                json_ready({"plan": plan, "shadow_update": shadow_update, "promotion_gate": promotion_gate}),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        record_runtime_error(state, exc)
        save_state(args.state_path, state)
        raise


def run_shadow_loop(args: argparse.Namespace) -> int:
    while True:
        try:
            run_shadow_once(args)
        except Exception as exc:
            print(f"[pairwise-shadow] {type(exc).__name__}: {exc}", file=sys.stderr)
        time.sleep(max(args.poll_seconds, 1))


def run_evaluate_shadow(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    evaluation = build_shadow_evaluation(state, args)
    state["promotion_gate"] = evaluation
    save_state(args.state_path, state)
    print(json.dumps(json_ready(evaluation), indent=2, sort_keys=True))
    return 0 if evaluation["requested_gate_ready"] else 2


def run_live_once(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    bridge = None
    exchange = None
    equity = None
    positions: Dict[str, Any] = {}
    positions_fetched: bool = False
    if args.execute:
        bridge = load_execution_bridge()
        exchange = bridge.get_exchange(args.mode)
        equity = float(bridge.fetch_equity(exchange))
        positions = bridge.fetch_open_position_map(exchange)
        positions_fetched = True
        sync_shadow_paper_from_live_positions(state, positions, equity)
        # Persist a fresh exchange snapshot every cycle so consumers like
        # position_reconciliation.py see current ground truth. Without this,
        # latest_live_sync remained stale across restarts because only the
        # explicit `sync-state` subcommand updated it, producing phantom
        # mismatch alerts (recon 2026-04-26 found state=-0.12 / exchange=0).
        state["latest_live_sync"] = {
            "at": iso_now(),
            "mode": args.mode,
            "execute": True,
            "equity": equity,
            "positions": positions,
            "source": "run_live_once",
        }
    else:
        # Dry-run: fetch positions and equity so D2/D1/R3 overlays (moved below)
        # can detect stale exchange state. No orders are placed regardless.
        # Tolerate missing credentials gracefully.
        try:
            bridge = load_execution_bridge()
            exchange = bridge.get_exchange(args.mode)
            positions = bridge.fetch_open_position_map(exchange)
            positions_fetched = True
            try:
                equity = float(bridge.fetch_equity(exchange))
            except Exception as eq_exc:
                print(f"[pairwise-live] dry-run equity fetch skipped: {eq_exc}")
                equity = None
        except Exception as exc:
            print(f"[pairwise-live] dry-run position fetch skipped: {exc}")
            bridge = None
            exchange = None
            positions = {}
            # positions_fetched remains False — D1/R3 will skip position-dependent logic
            equity = None

    plan = build_pairwise_plan(args.summary_path, args.model_path, args.promotion_report, args.refresh_live_data, state)
    persist_runtime_plan_state(state, plan)
    promotion_report_path = Path(plan.get("promotion_report_path") or args.promotion_report)
    promotion_gate = load_promotion_gate(promotion_report_path)
    state["promotion_gate"] = promotion_gate

    # ------------------------------------------------------------------
    # Safety overlays D2 / D1 / R3 — run unconditionally (execute AND dry-run).
    # Order matters: D2 clips tw first, D1 reads tw that D2 may have reduced,
    # R3 reads tw that D1 may have zeroed.  Mutations are auditable in dry-run
    # via decision_journal and decision_log even when no orders are placed.
    # ------------------------------------------------------------------

    # D2: apply per-pair gross cap
    if PAIRWISE_GROSS_CAP < 1.0:
        plan["target_weights"] = {
            pair: max(-PAIRWISE_GROSS_CAP, min(PAIRWISE_GROSS_CAP, float(w)))
            for pair, w in plan["target_weights"].items()
        }

    # D1: max-hold-bars auto-flatten (24 h)
    # Timer tracks the *current continuous* exchange position start time.
    # When the exchange is flat the timer either resets (fresh tw signal) or
    # clears (fully flat), so a new entry never inherits a dead position's age.
    position_open_since: Dict[str, Any] = dict(state.get("position_open_since_ts") or {})
    _now = utc_now()
    if not positions_fetched:
        # Position fetch unavailable — skip D1 entirely. Preserve existing timers
        # so a later successful-fetch cycle can resume tracking. The dry-run fetch
        # already printed a warning at fetch time; one line here is sufficient.
        print("[pairwise-live] D1 skipped: positions unknown (fetch unavailable)")
    else:
        for _pair in list(PAIRS):
            _tw = float(plan["target_weights"].get(_pair, 0.0))
            _prev_since = position_open_since.get(_pair)
            tw_active = abs(_tw) > TARGET_WEIGHT_EPS
            exch_active = _exchange_position_is_open(positions, _pair)
            if exch_active:
                # Exchange has a real open position — track continuously from first appearance.
                if _prev_since is None:
                    # First cycle this position is visible — start the clock.
                    position_open_since[_pair] = _now.isoformat()
                else:
                    age_secs = compute_position_age_seconds(_prev_since)
                    if age_secs > _MAX_HOLD_SECONDS:
                        _msg = (
                            f"[pairwise-live] D1 max_hold override: {_pair} position "
                            f"open {age_secs/3600:.1f}h >= {PAIRWISE_MAX_HOLD_BARS} bars — forcing flat"
                        )
                        print(_msg)
                        _notif = load_notification_bridge()
                        try:
                            from telegram_format import AlertLevel as _AL, format_alert as _fa, should_send as _ss
                            if _ss(_AL.HIGH, f"d1-max-hold-{_pair}"):
                                _payload = _fa(
                                    _AL.HIGH,
                                    title="D1 max_hold 자동 청산",
                                    body=f"{_pair} 포지션 {age_secs/3600:.1f}h 경과 ({PAIRWISE_MAX_HOLD_BARS}바 한도) — 강제 플랫",
                                    pair=_pair,
                                )
                                _notif.send_telegram_notification(_payload["text"])
                        except Exception:
                            _notif.send_telegram_notification(_msg)
                        plan["target_weights"][_pair] = 0.0
                        if _pair in plan.get("pair_plans", {}):
                            plan["pair_plans"][_pair]["target_weight"] = 0.0
                        # NOTE: do NOT clear position_open_since here. The timer must
                        # persist until reconcile actually closes the exchange position.
                        # D1 re-fires every cycle while exch_active=True AND age>24h
                        # (nag-until-fixed). The else branch below handles clean-up.
                        # Log to decision_journal
                        state.setdefault("decision_journal", []).append(
                            {
                                "at": _now.isoformat(),
                                "pair": _pair,
                                "override_reason": "max_hold",
                                "age_seconds": age_secs,
                                "max_hold_seconds": _MAX_HOLD_SECONDS,
                                "target_weight_forced": 0.0,
                                "exchange_position_active": exch_active,
                            }
                        )
            else:
                # Exchange is flat — the previous position (if any) has closed.
                if tw_active:
                    # Fresh signal will open a new position; reset timer to now so
                    # the new entry is not penalised by the dead position's age.
                    position_open_since[_pair] = _now.isoformat()
                else:
                    # Fully flat — clear the timer.
                    position_open_since[_pair] = None
        state["position_open_since_ts"] = position_open_since

    # R3: CVaR-99 cut overlay
    if PAIRWISE_CVAR_CUT_ENABLED:
        cvar_thresholds = load_tail_risk_thresholds()
        cvar_cut_until: Dict[str, Any] = dict(state.get("cvar_cut_until_ts") or {})
        for _pair in list(PAIRS):
            _cut_until_raw = cvar_cut_until.get(_pair)
            _cut_until_dt = parse_utc_datetime(_cut_until_raw)
            # Check if an existing cut is still active
            if _cut_until_dt is not None and _now < _cut_until_dt:
                _tw = float(plan["target_weights"].get(_pair, 0.0))
                _exch_active = (
                    positions_fetched and _exchange_position_is_open(positions, _pair)
                )
                if abs(_tw) > TARGET_WEIGHT_EPS or _exch_active:
                    plan["target_weights"][_pair] = 0.0
                    if _pair in plan.get("pair_plans", {}):
                        plan["pair_plans"][_pair]["target_weight"] = 0.0
                    # Exchange position still open while cut active — keep forcing tw=0 to retry close
                    if _exch_active and abs(_tw) <= TARGET_WEIGHT_EPS:
                        print(f"[pairwise-live] R3 cut still active for {_pair}; exchange position not yet flat — forcing tw=0 to retry close")
                        state.setdefault("decision_journal", []).append(
                            {
                                "at": _now.isoformat(),
                                "pair": _pair,
                                "override_reason": "cvar_cut_retry_close",
                                "cvar_cut_until": _cut_until_dt.isoformat() if _cut_until_dt else None,
                                "exchange_position_active": _exch_active,
                                "target_weight_forced": 0.0,
                            }
                        )
                continue
            else:
                # Cut expired — clear it
                if _cut_until_dt is not None and _now >= _cut_until_dt:
                    cvar_cut_until[_pair] = None
            # Evaluate whether a new cut should trigger
            if should_apply_cvar_cut(_pair, cvar_thresholds):
                _resume_dt = _now + timedelta(hours=PAIRWISE_CVAR_CUT_HOLD_HOURS)
                cvar_cut_until[_pair] = _resume_dt.isoformat()
                _msg = (
                    f"[pairwise-live] R3 CVaR-99 cut: {_pair} 30d return below "
                    f"CVaR-99 threshold ({cvar_thresholds.get(_pair, 'N/A'):.4f}). "
                    f"Forcing flat until {_resume_dt.strftime('%Y-%m-%dT%H:%MZ')}"
                )
                print(_msg)
                _notif = load_notification_bridge()
                try:
                    from telegram_format import AlertLevel as _AL, format_alert as _fa, should_send as _ss, format_kst as _fkst
                    if _ss(_AL.HIGH, f"r3-cvar-cut-{_pair}"):
                        _thresh_val = cvar_thresholds.get(_pair)
                        _thresh_str = f"{_thresh_val:.4f}" if _thresh_val is not None else "N/A"
                        _payload = _fa(
                            _AL.HIGH,
                            title="R3 CVaR-99 컷 발동",
                            body=f"{_pair} 30일 수익률이 CVaR-99 임계({_thresh_str}) 미만 — 재개 {_fkst(_resume_dt)}",
                            pair=_pair,
                            buttons=[
                                {"label": "📈 차트", "callback_data": "chart"},
                                {"label": "🔓 재개", "callback_data": "resume"},
                            ],
                        )
                        _notif.send_telegram_notification(_payload["text"])
                except Exception:
                    _notif.send_telegram_notification(_msg)
                plan["target_weights"][_pair] = 0.0
                if _pair in plan.get("pair_plans", {}):
                    plan["pair_plans"][_pair]["target_weight"] = 0.0
                state.setdefault("decision_journal", []).append(
                    {
                        "at": _now.isoformat(),
                        "pair": _pair,
                        "override_reason": "cvar_cut",
                        "cvar_threshold": cvar_thresholds.get(_pair),
                        "cvar_cut_until": _resume_dt.isoformat(),
                        "target_weight_forced": 0.0,
                    }
                )
        state["cvar_cut_until_ts"] = cvar_cut_until

    if args.execute:
        force_execute = bool(getattr(args, "force_execute", False))
        force_note = str(getattr(args, "force_note", "manual_primary_switch")).strip() or "manual_primary_switch"
        gate_ready = promotion_gate_allows_execution(promotion_gate, args.mode)
        if not gate_ready and not force_execute:
            record_runtime_success(
                state,
                plan,
                extra={
                    "execution": {
                        "enabled": False,
                        "blocked": True,
                        "mode": args.mode,
                        "equity": equity,
                        "promotion_gate": promotion_gate,
                        "force_requested": force_execute,
                    }
                },
            )
            sync_position_loss_notifications(state, positions)
            append_jsonl(
                args.decision_log_path,
                {
                    "at": iso_now(),
                    "mode": f"{args.mode}-blocked",
                    "execute": True,
                    "plan": plan,
                    "promotion_gate": promotion_gate,
                    "force_requested": force_execute,
                },
            )
            save_state(args.state_path, state)
            print(
                json.dumps(
                    json_ready(
                        {
                            "plan": plan,
                            "promotion_gate": promotion_gate,
                            "force_requested": force_execute,
                        }
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

        actions = bridge.reconcile_target_positions(
            exchange,
            equity,
            plan["target_weights"],
            execute=True,
            pairs=list(PAIRS),
        )
        protection_report = bridge.install_shutdown_protection(exchange, state, execute=True)
        positions = bridge.fetch_open_position_map(exchange)
        sync_shadow_paper_from_live_positions(state, positions, equity)
        # Post-trade snapshot overwrites the pre-trade one so consumers like
        # position_reconciliation.py see the settled exchange state, not the
        # pre-reconcile view that would generate phantom mismatch alerts.
        state["latest_live_sync"] = {
            "at": iso_now(),
            "mode": args.mode,
            "execute": True,
            "equity": equity,
            "positions": positions,
            "source": "run_live_once_post_trade",
        }
        execution_mode = f"{args.mode}-executed"
        execution_override: Dict[str, Any] | None = None
        if force_execute and not gate_ready:
            execution_mode = f"{args.mode}-forced"
            execution_override = {
                "force_execute": True,
                "force_note": force_note,
                "promotion_gate_bypassed": True,
            }
        record_runtime_success(
            state,
            plan,
            extra={
                "execution": {
                    "enabled": True,
                    "mode": args.mode,
                    "equity": equity,
                    "actions": actions,
                    "shutdown_protection": protection_report,
                    "promotion_gate": promotion_gate,
                    "override": execution_override,
                }
            },
        )
        sync_position_loss_notifications(state, positions)
        append_jsonl(
            args.decision_log_path,
            {
                "at": iso_now(),
                "mode": execution_mode,
                "execute": True,
                "plan": plan,
                "equity": equity,
                "actions": actions,
                "shutdown_protection": protection_report,
                "promotion_gate": promotion_gate,
                "override": execution_override,
            },
        )
        save_state(args.state_path, state)
        print(
            json.dumps(
                json_ready(
                    {
                        "plan": plan,
                        "equity": equity,
                        "actions": actions,
                        "shutdown_protection": protection_report,
                        "promotion_gate": promotion_gate,
                        "override": execution_override,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    record_runtime_success(
        state,
        plan,
        extra={
            "execution": {
                "enabled": bool(args.execute),
                "mode": args.mode,
                "promotion_gate": promotion_gate,
                "note": "promotion path prepared; order routing intentionally left disabled until shadow gate passes",
            }
        },
    )
    append_jsonl(
        args.decision_log_path,
        {
            "at": iso_now(),
            "mode": "live-preview",
            "execute": bool(args.execute),
            "plan": plan,
        },
    )
    save_state(args.state_path, state)
    print(
        json.dumps(
            json_ready(
                {
                    "plan": plan,
                    "execution": {
                        "enabled": bool(args.execute),
                        "mode": args.mode,
                        "promotion_gate_required": True,
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_live_loop(args: argparse.Namespace) -> int:
    while True:
        try:
            state = load_state(args.state_path)
            state.setdefault("runtime_health", {})
            state["runtime_health"]["pid"] = os.getpid()
            state["runtime_health"]["last_loop_started_at"] = iso_now()
            save_state(args.state_path, state)
            run_live_once(args)
        except Exception as exc:
            state = load_state(args.state_path)
            record_runtime_error(state, exc)
            save_state(args.state_path, state)
            print(f"[pairwise-live] {type(exc).__name__}: {exc}", file=sys.stderr)
        time.sleep(max(args.poll_seconds, 1))


def run_sync_state(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    bridge = load_execution_bridge()
    exchange = bridge.get_exchange(args.mode)
    cleanup_report = None
    if bool(args.execute):
        cleanup_report = bridge.install_shutdown_protection(exchange, state, execute=True)
    equity = float(bridge.fetch_equity(exchange))
    positions = bridge.fetch_open_position_map(exchange)
    protections = bridge.fetch_strategy_protection_orders(exchange)
    sync_shadow_paper_from_live_positions(state, positions, equity)
    snapshot = {
        "at": iso_now(),
        "mode": args.mode,
        "execute": bool(args.execute),
        "equity": equity,
        "positions": positions,
        "protection_orders": protections,
    }
    if cleanup_report is not None:
        snapshot["protection_cleanup"] = cleanup_report
    state["latest_live_sync"] = snapshot
    save_state(args.state_path, state)
    print(json.dumps(json_ready(snapshot), indent=2, sort_keys=True))
    return 0


def run_shutdown_protect(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    bridge = load_execution_bridge()
    exchange = bridge.get_exchange(args.mode)
    report = bridge.install_shutdown_protection(exchange, state, execute=bool(args.execute))
    state["latest_shutdown_protection_report"] = {
        "at": iso_now(),
        "mode": args.mode,
        "execute": bool(args.execute),
        "report": report,
    }
    save_state(args.state_path, state)
    print(json.dumps(json_ready(report), indent=2, sort_keys=True))
    return 0


def run_close_all(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    bridge = load_execution_bridge()
    exchange = bridge.get_exchange(args.mode)
    actions = bridge.flatten_pairs(exchange, list(PAIRS), execute=bool(args.execute))
    state["latest_close_all_report"] = {
        "at": iso_now(),
        "mode": args.mode,
        "execute": bool(args.execute),
        "actions": actions,
    }
    save_state(args.state_path, state)
    print(json.dumps(json_ready({"actions": actions}), indent=2, sort_keys=True))
    return 0


def main() -> int:
    # R4: override NO_TRADE_BAND in-memory before any kernel use
    _no_trade_band = os.getenv("PAIRWISE_NO_TRADE_BAND_PCT")
    if _no_trade_band:
        gp.NO_TRADE_BAND = float(_no_trade_band)
        print(f"[pairwise-live] NO_TRADE_BAND overridden to {gp.NO_TRADE_BAND} (from env)")
    print(f"[pairwise-live] PAIRWISE_GROSS_CAP={PAIRWISE_GROSS_CAP}  NO_TRADE_BAND={gp.NO_TRADE_BAND}")
    args = parse_args()
    if args.command == "status":
        return run_status(args)
    if args.command == "run-once":
        return run_live_once(args)
    if args.command == "loop":
        return run_live_loop(args)
    if args.command == "sync-state":
        return run_sync_state(args)
    if args.command == "shutdown-protect":
        return run_shutdown_protect(args)
    if args.command == "close-all":
        return run_close_all(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
