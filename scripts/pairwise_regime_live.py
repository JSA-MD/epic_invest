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
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    build_overlay_inputs,
    build_route_bucket_codes,
    normalize_mapping_indices,
    normalize_route_state_mode,
    route_state_names,
)

load_dotenv(ROOT / ".env")

PAIRS = ("BTCUSDT", "BNBUSDT")
PAIR_TO_MARKET = {
    "BTCUSDT": "BTC/USDT:USDT",
    "BNBUSDT": "BNB/USDT:USDT",
}

DEFAULT_SUMMARY_PATH = ROOT / "models" / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"
DEFAULT_MODEL_PATH = ROOT / "models" / "recent_6m_gp_vectorized_big_capped_rerun.dill"
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


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "strategy_class": "pairwise_regime_live",
            "created_at": iso_now(),
            "updated_at": iso_now(),
            "shadow_paper": {
                "enabled": True,
                "observations": 0,
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
            "latest_runtime_snapshot": {},
            "latest_decision_snapshot": {},
            "promotion_gate": {},
            "decision_journal": [],
        }
    return json.loads(path.read_text())


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    ensure_parent(path)
    payload = copy.deepcopy(dict(state))
    payload["updated_at"] = iso_now()
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True))


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


def pandas_timeframe(interval: str) -> str:
    if interval.endswith("m"):
        return f"{interval[:-1]}min"
    if interval.endswith("h"):
        return f"{interval[:-1]}h"
    return interval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise BTC/BNB live and shadow runner")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
        cmd.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
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
    run_once.add_argument("--shadow-state-path", type=Path, default=DEFAULT_SHADOW_STATE_PATH)

    shadow_once = sub.add_parser("shadow-once")
    add_common(shadow_once)
    shadow_once.set_defaults(state_path=DEFAULT_SHADOW_STATE_PATH, decision_log_path=DEFAULT_SHADOW_DECISION_LOG_PATH)

    loop = sub.add_parser("loop")
    add_common(loop)
    loop.add_argument("--execute", action="store_true")
    loop.add_argument("--force-execute", action="store_true")
    loop.add_argument("--force-note", default="manual_primary_switch")
    loop.add_argument("--mode", choices=("demo", "live"), default="demo")
    loop.add_argument("--shadow-state-path", type=Path, default=DEFAULT_SHADOW_STATE_PATH)
    loop.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)

    shadow_loop = sub.add_parser("shadow-loop")
    add_common(shadow_loop)
    shadow_loop.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    shadow_loop.set_defaults(state_path=DEFAULT_SHADOW_STATE_PATH, decision_log_path=DEFAULT_SHADOW_DECISION_LOG_PATH)

    eval_shadow = sub.add_parser("evaluate-shadow")
    add_common(eval_shadow)
    eval_shadow.add_argument("--min-observations", type=int, default=SHADOW_PROMOTION_MIN_OBSERVATIONS)
    eval_shadow.add_argument("--max-drawdown", type=float, default=SHADOW_PROMOTION_MAX_DRAWDOWN)
    eval_shadow.add_argument("--min-return", type=float, default=SHADOW_PROMOTION_MIN_RETURN)
    eval_shadow.add_argument("--max-stale-minutes", type=float, default=SHADOW_PROMOTION_MAX_STALE_MINUTES)
    eval_shadow.set_defaults(state_path=DEFAULT_SHADOW_STATE_PATH, decision_log_path=DEFAULT_SHADOW_DECISION_LOG_PATH)

    sync_state = sub.add_parser("sync-state")
    add_common(sync_state)
    sync_state.add_argument("--mode", choices=("demo", "live"), default="demo")

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


def build_pairwise_plan(
    summary_path: Path,
    model_path: Path,
    refresh_live_data: bool,
    state: Mapping[str, Any],
) -> Dict[str, Any]:
    summary = load_selected_candidate(summary_path)
    embedded_model_ref = extract_strategy_artifact_reference(summary, "model_path")
    resolved_model_path = resolve_strategy_artifact_path(embedded_model_ref or model_path, summary_path)
    config = summary["selected_candidate"]["pair_configs"]
    library = list(iter_params())
    model_tree, _ = load_signal_model(resolved_model_path)
    compiled = gp.toolbox.compile(expr=model_tree)
    df = load_live_frame(PAIRS, refresh_live_data=refresh_live_data)
    signal_index = len(df) - 1
    shadow = state.get("shadow_paper", {})
    current_weights = shadow.get("current_weights", {})
    cooldown_state = shadow.get("cooldown_bars_left", {})
    current_equity = float(shadow.get("equity", SHADOW_DEFAULT_EQUITY))
    peak_equity = float(shadow.get("peak_equity", current_equity))

    pair_plans: Dict[str, Any] = {}
    target_weights: Dict[str, float] = {}
    latest_prices: Dict[str, float] = {}

    for pair in PAIRS:
        raw_signal = np.asarray(compiled(*gp.get_feature_arrays(df, pair)), dtype=float)
        overlay_inputs = build_overlay_inputs(df, PAIRS, regime_pair=pair)
        route_state_mode = normalize_route_state_mode(config[pair].get("route_state_mode"))
        bucket_codes = build_route_bucket_codes(
            df.index,
            overlay_inputs,
            config[pair]["route_breadth_threshold"],
            route_state_mode=route_state_mode,
        )
        mapping = normalize_mapping_indices(config[pair]["mapping_indices"], route_state_mode)
        bucket_code = int(bucket_codes[signal_index])
        active_index = int(mapping[bucket_code])
        route_state_name = route_state_names(route_state_mode)[bucket_code]
        params = library[active_index]
        current_weight = float(current_weights.get(pair, 0.0))
        cooldown_bars_left = max(int(cooldown_state.get(pair, 0)), 0)
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1
        day_index = pd.DatetimeIndex(df.index).normalize()
        regime_score = float(
            overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0).iloc[signal_index]
        )
        breadth_score = float(
            overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0).iloc[signal_index]
        )
        bar_vol_ann = float(overlay_inputs["vol_ann_bar"].fillna(np.nan).iloc[signal_index])
        equity_corr_value = float(
            overlay_inputs["equity_corr_daily"].reindex(day_index, method="ffill").iloc[signal_index]
        )
        equity_corr_bucket = str(
            overlay_inputs["equity_corr_bucket_daily"].reindex(day_index, method="ffill").fillna("equity_unknown").iloc[signal_index]
        )
        equity_corr_quantile_state = str(
            overlay_inputs["equity_corr_quantile_state_daily"].reindex(day_index, method="ffill").fillna("missing").iloc[signal_index]
        )
        equity_corr_gross_scale = float(
            overlay_inputs["equity_corr_gross_scale_daily"].reindex(day_index, method="ffill").fillna(1.0).iloc[signal_index]
        )
        equity_corr_regime_mult = float(
            overlay_inputs["equity_corr_regime_threshold_mult_daily"].reindex(day_index, method="ffill").fillna(1.0).iloc[signal_index]
        )
        if not PAIRWISE_EQUITY_CORR_RISK_ENABLED:
            equity_corr_gross_scale = 1.0
            equity_corr_regime_mult = 1.0
        requested_weight = compute_requested_weight(
            raw_signal=raw_signal,
            params=params,
            regime_score=regime_score,
            breadth_score=breadth_score,
            bar_vol_ann=bar_vol_ann,
            equity_corr_gross_scale=equity_corr_gross_scale,
            equity_corr_regime_mult=equity_corr_regime_mult,
        )
        drawdown = current_equity / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -float(params.kill_switch_pct) and cooldown_bars_left == 0:
            cooldown_bars_left = int(params.cooldown_days) * gp.periods_per_day(gp.TIMEFRAME)
        rebalance_due = signal_index % max(int(params.rebalance_bars), 1) == 0
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif rebalance_due or abs(current_weight) <= TARGET_WEIGHT_EPS:
            target_weight = requested_weight
        else:
            target_weight = current_weight
        if abs(target_weight - current_weight) < gp.NO_TRADE_BAND / 100.0:
            target_weight = current_weight
        close_price = float(df[f"{pair}_close"].iloc[signal_index])
        latest_prices[pair] = close_price
        target_weights[pair] = target_weight
        pair_plans[pair] = {
            "price": close_price,
            "current_weight": current_weight,
            "requested_weight": requested_weight,
            "target_weight": target_weight,
            "rebalance_due": rebalance_due,
            "cooldown_bars_left_after": cooldown_bars_left,
            "route_bucket": bucket_code,
            "route_state_mode": route_state_mode,
            "route_state_name": route_state_name,
            "route_mapping_index": active_index,
            "params": asdict(params),
            "regime_score": regime_score,
            "breadth_score": breadth_score,
            "bar_vol_ann": bar_vol_ann,
            "equity_corr_value": equity_corr_value if np.isfinite(equity_corr_value) else None,
            "equity_corr_bucket": equity_corr_bucket,
            "equity_corr_quantile_state": equity_corr_quantile_state,
            "equity_corr_context": overlay_inputs.get("equity_corr_context"),
            "equity_corr_source_mode": overlay_inputs.get("equity_corr_source_mode"),
            "equity_corr_gross_scale": equity_corr_gross_scale,
            "equity_corr_regime_threshold_mult": equity_corr_regime_mult,
            "signal_value": float(np.nan_to_num(raw_signal[-1], nan=0.0)),
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
        "summary_path": str(summary_path),
        "model_path": str(resolved_model_path),
    }


def apply_shadow_mark_to_market(state: Dict[str, Any], plan: Mapping[str, Any]) -> Dict[str, Any]:
    shadow = state.setdefault("shadow_paper", {})
    shadow.setdefault("enabled", True)
    shadow.setdefault("observations", 0)
    shadow.setdefault("equity", SHADOW_DEFAULT_EQUITY)
    shadow.setdefault("peak_equity", shadow["equity"])
    shadow.setdefault("max_drawdown", 0.0)
    shadow.setdefault("return_pct", 0.0)
    shadow.setdefault("last_prices", {})
    shadow.setdefault("current_weights", {})
    shadow.setdefault("cooldown_bars_left", {})
    shadow.setdefault("turnover_cost_paid", 0.0)

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
    shadow["return_pct"] = (current_equity / SHADOW_DEFAULT_EQUITY - 1.0) * 100.0
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


def record_runtime_success(state: Dict[str, Any], plan: Mapping[str, Any], extra: Optional[Mapping[str, Any]] = None) -> None:
    state.setdefault("runtime_health", {})
    state["runtime_health"].update(
        {
            "status": "ok",
            "consecutive_errors": 0,
            "last_error": None,
            "last_success_at": iso_now(),
        }
    )
    state["latest_runtime_snapshot"] = {
        "generated_at": iso_now(),
        "plan": plan,
        "extra": extra or {},
    }
    state["latest_decision_snapshot"] = {
        "generated_at": iso_now(),
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
            "at": iso_now(),
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
    plan = build_pairwise_plan(args.summary_path, args.model_path, args.refresh_live_data, state)
    print(json.dumps(json_ready(render_status(state, plan)), indent=2, sort_keys=True))
    return 0


def run_shadow_once(args: argparse.Namespace) -> int:
    state = load_state(args.state_path)
    try:
        plan = build_pairwise_plan(args.summary_path, args.model_path, args.refresh_live_data, state)
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
    plan = build_pairwise_plan(args.summary_path, args.model_path, args.refresh_live_data, state)
    if args.execute:
        shadow_state = load_state(args.shadow_state_path)
        evaluation = build_shadow_evaluation(shadow_state, default_promotion_eval_args(args.shadow_state_path))
        state["promotion_gate"] = evaluation
        force_execute = bool(getattr(args, "force_execute", False))
        force_note = str(getattr(args, "force_note", "manual_primary_switch")).strip() or "manual_primary_switch"
        force_blocked_by_stale_shadow = bool(
            force_execute
            and not evaluation["promotion_ready"]
            and (evaluation.get("shadow_feed_stale") or evaluation.get("shadow_signal_stale"))
        )
        if not evaluation["promotion_ready"] and (not force_execute or force_blocked_by_stale_shadow):
            blocked_mode = "live-force-blocked" if force_blocked_by_stale_shadow else "live-blocked"
            record_runtime_success(
                state,
                plan,
                extra={
                    "execution": {
                        "enabled": False,
                        "blocked": True,
                        "mode": args.mode,
                        "promotion_gate": evaluation,
                        "force_requested": force_execute,
                        "force_blocked_by_stale_shadow": force_blocked_by_stale_shadow,
                    }
                },
            )
            append_jsonl(
                args.decision_log_path,
                {
                    "at": iso_now(),
                    "mode": blocked_mode,
                    "execute": True,
                    "plan": plan,
                    "promotion_gate": evaluation,
                    "force_requested": force_execute,
                    "force_blocked_by_stale_shadow": force_blocked_by_stale_shadow,
                },
            )
            save_state(args.state_path, state)
            print(
                json.dumps(
                    json_ready(
                        {
                            "plan": plan,
                            "promotion_gate": evaluation,
                            "force_requested": force_execute,
                            "force_blocked_by_stale_shadow": force_blocked_by_stale_shadow,
                        }
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2

        bridge = load_execution_bridge()
        exchange = bridge.get_exchange(args.mode)
        equity = float(bridge.fetch_equity(exchange))
        actions = bridge.reconcile_target_positions(
            exchange,
            equity,
            plan["target_weights"],
            execute=True,
            pairs=list(PAIRS),
        )
        protection_report = bridge.install_shutdown_protection(exchange, state, execute=True)
        execution_mode = "live-executed"
        execution_override: Dict[str, Any] | None = None
        if force_execute and not evaluation["promotion_ready"]:
            execution_mode = "live-forced"
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
                    "promotion_gate": evaluation,
                    "override": execution_override,
                }
            },
        )
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
                "promotion_gate": evaluation,
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
                        "promotion_gate": evaluation,
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
    equity = float(bridge.fetch_equity(exchange))
    positions = bridge.fetch_open_position_map(exchange)
    protections = bridge.fetch_strategy_protection_orders(exchange)
    snapshot = {
        "at": iso_now(),
        "mode": args.mode,
        "equity": equity,
        "positions": positions,
        "protection_orders": protections,
    }
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
    args = parse_args()
    if args.command == "status":
        return run_status(args)
    if args.command == "shadow-once":
        return run_shadow_once(args)
    if args.command == "shadow-loop":
        return run_shadow_loop(args)
    if args.command == "evaluate-shadow":
        return run_evaluate_shadow(args)
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
