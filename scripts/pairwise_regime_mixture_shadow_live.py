#!/usr/bin/env python3
"""Shadow/live runner for the BTC/BNB pairwise regime-mixture strategy."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import signal
import time
from dataclasses import asdict, replace
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

import gp_crypto_evolution as gp
from fractal_genome_core import LeafGene, collect_specs, deserialize_tree, evaluate_tree_leaf_codes
from replay_regime_mixture_realistic import load_model
from rotation_target_050_live import (
    append_jsonl,
    fetch_equity,
    get_exchange,
    json_ready,
    normalize_day,
    save_state,
    utc_now,
)
from search_pair_subset_fractal_genome import (
    apply_label_horizon_to_feature_arrays,
    apply_leaf_gene_to_pair_config,
    build_expert_pool,
    build_market_features,
    materialize_feature_arrays,
    project_feature_arrays_by_observation_mode,
)
from search_pair_subset_regime_mixture import (
    build_library_lookup,
    build_overlay_inputs,
    build_route_bucket_codes,
    json_safe,
    normalize_mapping_indices,
    normalize_route_state_mode,
    route_state_names,
)
from search_gp_drawdown_overlay import OverlayParams, iter_params


load_dotenv()

PAIR_TO_MARKET = {
    "BTCUSDT": "BTC/USDT:USDT",
    "BNBUSDT": "BNB/USDT:USDT",
}
DEFAULT_PAIRS = ("BTCUSDT", "BNBUSDT")
PAIRWISE_HISTORY_START = "2022-04-06"
SHADOW_DEFAULT_EQUITY = 100_000.0
STATE_PATH = Path(
    os.getenv(
        "PAIRWISE_SHADOW_STATE_PATH",
        str(gp.MODELS_DIR / "pairwise_regime_shadow_state.json"),
    )
)
DECISION_LOG_PATH = Path(
    os.getenv("PAIRWISE_SHADOW_DECISION_LOG_FILE")
    or os.getenv("PAIRWISE_SHADOW_DECISION_LOG_PATH")
    or str(gp.MODELS_DIR.parent / "logs" / "pairwise_regime_shadow_decisions.jsonl")
)
DEFAULT_SUMMARY_PATH = Path(
    os.getenv(
        "PAIRWISE_SHADOW_SUMMARY_PATH",
        str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"),
    )
)
DEFAULT_BASE_SUMMARY_PATH = Path(
    os.getenv(
        "PAIRWISE_SHADOW_BASE_SUMMARY_PATH",
        str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
    )
)
DEFAULT_MODEL_PATH = Path(
    os.getenv(
        "PAIRWISE_SHADOW_MODEL_PATH",
        str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
)
DEFAULT_PROMOTION_REPORT_PATH = Path(
    os.getenv(
        "PAIRWISE_SHADOW_PROMOTION_REPORT_PATH",
        str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_market_os_pipeline_report.json"),
    )
)
DEFAULT_EXCHANGE_LEVERAGE = 5
EPSILON = 1e-12

RUNTIME_CONTEXT: dict[str, Any] = {
    "args": None,
    "state": None,
    "exchange": None,
}


def resolve_pairs(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_PAIRS
    items = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(items) != 2:
        raise ValueError("Pairwise shadow runner expects exactly two pairs.")
    for pair in items:
        if pair not in PAIR_TO_MARKET:
            raise ValueError(f"Unsupported pair {pair!r}. Supported pairs: {', '.join(PAIR_TO_MARKET)}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the BTC/BNB pairwise regime-mixture strategy in shadow mode or promote it to execution.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = dict(
        pairs=DEFAULT_PAIRS,
        summary=str(DEFAULT_SUMMARY_PATH),
        base_summary=str(DEFAULT_BASE_SUMMARY_PATH),
        model=str(DEFAULT_MODEL_PATH),
        promotion_report=str(DEFAULT_PROMOTION_REPORT_PATH),
        state_path=str(STATE_PATH),
        decision_log=str(DECISION_LOG_PATH),
        leverage=DEFAULT_EXCHANGE_LEVERAGE,
    )

    status = subparsers.add_parser("status", help="Show the current shadow plan without placing orders.")
    status.add_argument("--pairs", default=",".join(common["pairs"]))
    status.add_argument("--summary", default=common["summary"])
    status.add_argument("--base-summary", default=common["base_summary"])
    status.add_argument("--model", default=common["model"])
    status.add_argument("--promotion-report", default=common["promotion_report"])
    status.add_argument("--state-path", default=common["state_path"])
    status.add_argument("--decision-log", default=common["decision_log"])
    status.add_argument("--equity", type=float, default=gp.INITIAL_CASH)
    status.add_argument("--mode", choices=["demo", "live"], default=os.getenv("BINANCE_MODE", "demo"))
    status.add_argument("--leverage", type=float, default=common["leverage"])
    status.add_argument("--refresh-live-data", dest="refresh_live_data", action="store_true")
    status.add_argument("--no-refresh-live-data", dest="refresh_live_data", action="store_false")
    status.set_defaults(refresh_live_data=True)

    run_once = subparsers.add_parser("run-once", help="Evaluate the pairwise strategy once.")
    run_once.add_argument("--pairs", default=",".join(common["pairs"]))
    run_once.add_argument("--summary", default=common["summary"])
    run_once.add_argument("--base-summary", default=common["base_summary"])
    run_once.add_argument("--model", default=common["model"])
    run_once.add_argument("--promotion-report", default=common["promotion_report"])
    run_once.add_argument("--state-path", default=common["state_path"])
    run_once.add_argument("--decision-log", default=common["decision_log"])
    run_once.add_argument("--mode", choices=["demo", "live"], default=os.getenv("BINANCE_MODE", "demo"))
    run_once.add_argument("--equity", type=float, default=None)
    run_once.add_argument("--leverage", type=float, default=common["leverage"])
    run_once.add_argument("--execute", action="store_true", help="Actually place orders.")
    run_once.add_argument("--force-execute", action="store_true", help="Bypass the promotion gate.")
    run_once.add_argument("--refresh-live-data", dest="refresh_live_data", action="store_true")
    run_once.add_argument("--no-refresh-live-data", dest="refresh_live_data", action="store_false")
    run_once.set_defaults(refresh_live_data=True)

    loop = subparsers.add_parser("loop", help="Keep evaluating the pairwise strategy on a schedule.")
    loop.add_argument("--pairs", default=",".join(common["pairs"]))
    loop.add_argument("--summary", default=common["summary"])
    loop.add_argument("--base-summary", default=common["base_summary"])
    loop.add_argument("--model", default=common["model"])
    loop.add_argument("--promotion-report", default=common["promotion_report"])
    loop.add_argument("--state-path", default=common["state_path"])
    loop.add_argument("--decision-log", default=common["decision_log"])
    loop.add_argument("--mode", choices=["demo", "live"], default=os.getenv("BINANCE_MODE", "demo"))
    loop.add_argument("--equity", type=float, default=None)
    loop.add_argument("--leverage", type=float, default=common["leverage"])
    loop.add_argument("--execute", action="store_true", help="Actually place orders.")
    loop.add_argument("--force-execute", action="store_true", help="Bypass the promotion gate.")
    loop.add_argument("--poll-seconds", type=int, default=60)
    loop.add_argument("--refresh-live-data", dest="refresh_live_data", action="store_true")
    loop.add_argument("--no-refresh-live-data", dest="refresh_live_data", action="store_false")
    loop.set_defaults(refresh_live_data=True)

    return parser.parse_args()


def load_shadow_state(path: str | Path, decision_log: str | Path) -> dict[str, Any]:
    state_file = Path(path)
    if not state_file.exists():
        return {
            "strategy_class": "pairwise_regime_mixture_shadow_live",
            "created_at": utc_now().isoformat(),
            "selected_candidate": None,
            "promotion_gate": None,
            "latest_plan": None,
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
                "last_signal_timestamp": None,
                "last_updated_at": None,
            },
            "latest_runtime_snapshot": {
                "captured_at": None,
                "equity": None,
                "positions": [],
                "exchange_error": None,
                "plan": None,
            },
            "latest_decision_snapshot": {},
            "decision_journal": {
                "last_fingerprint": None,
                "last_recorded_at": None,
                "log_path": str(decision_log),
            },
            "runtime_health": {
                "process_started_at": None,
                "last_loop_started_at": None,
                "last_loop_completed_at": None,
                "last_success_at": None,
                "last_error_at": None,
                "last_error_message": None,
                "consecutive_errors": 0,
                "loop_success_count": 0,
                "loop_error_count": 0,
                "last_duration_seconds": None,
                "pid": None,
                "poll_seconds": None,
            },
            "updated_at": None,
        }

    with open(state_file, "r") as f:
        state = json.load(f)
    state.setdefault("strategy_class", "pairwise_regime_mixture_shadow_live")
    state.setdefault("created_at", None)
    state.setdefault("selected_candidate", None)
    state.setdefault("promotion_gate", None)
    state.setdefault("latest_plan", None)
    if not isinstance(state.get("shadow_paper"), dict):
        state["shadow_paper"] = {}
    state["shadow_paper"].setdefault("enabled", True)
    state["shadow_paper"].setdefault("observations", 0)
    state["shadow_paper"].setdefault("equity", SHADOW_DEFAULT_EQUITY)
    state["shadow_paper"].setdefault("peak_equity", SHADOW_DEFAULT_EQUITY)
    state["shadow_paper"].setdefault("max_drawdown", 0.0)
    state["shadow_paper"].setdefault("return_pct", 0.0)
    state["shadow_paper"].setdefault("last_prices", {})
    state["shadow_paper"].setdefault("current_weights", {})
    state["shadow_paper"].setdefault("cooldown_bars_left", {})
    state["shadow_paper"].setdefault("turnover_cost_paid", 0.0)
    state["shadow_paper"].setdefault("last_signal_timestamp", None)
    state["shadow_paper"].setdefault("last_updated_at", None)
    if not isinstance(state.get("latest_runtime_snapshot"), dict):
        state["latest_runtime_snapshot"] = {}
    state["latest_runtime_snapshot"].setdefault("captured_at", None)
    state["latest_runtime_snapshot"].setdefault("equity", None)
    state["latest_runtime_snapshot"].setdefault("positions", [])
    state["latest_runtime_snapshot"].setdefault("exchange_error", None)
    state["latest_runtime_snapshot"].setdefault("plan", None)
    if not isinstance(state.get("latest_decision_snapshot"), dict):
        state["latest_decision_snapshot"] = {}
    if not isinstance(state.get("decision_journal"), dict):
        state["decision_journal"] = {
            "history": state.get("decision_journal") if isinstance(state.get("decision_journal"), list) else [],
        }
    state["decision_journal"].setdefault("last_fingerprint", None)
    state["decision_journal"].setdefault("last_recorded_at", None)
    state["decision_journal"].setdefault("log_path", str(decision_log))
    if not isinstance(state.get("runtime_health"), dict):
        state["runtime_health"] = {}
    state["runtime_health"].setdefault("process_started_at", None)
    state["runtime_health"].setdefault("last_loop_started_at", None)
    state["runtime_health"].setdefault("last_loop_completed_at", None)
    state["runtime_health"].setdefault("last_success_at", None)
    state["runtime_health"].setdefault("last_error_at", None)
    state["runtime_health"].setdefault("last_error_message", None)
    state["runtime_health"].setdefault("consecutive_errors", 0)
    state["runtime_health"].setdefault("loop_success_count", 0)
    state["runtime_health"].setdefault("loop_error_count", 0)
    state["runtime_health"].setdefault("last_duration_seconds", None)
    state["runtime_health"].setdefault("pid", None)
    state["runtime_health"].setdefault("poll_seconds", None)
    state.setdefault("updated_at", None)
    return state


def update_runtime_health(state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    runtime_health = state.setdefault("runtime_health", {})
    for key, value in updates.items():
        runtime_health[key] = value
    return runtime_health


def json_stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_stable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_stable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def pandas_timeframe(interval: str) -> str:
    if interval.endswith("m"):
        return f"{interval[:-1]}min"
    if interval.endswith("h"):
        return f"{interval[:-1]}h"
    return interval


def drop_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    index = pd.DatetimeIndex(df.index)
    if index.tz is None:
        index = index.tz_localize(timezone.utc)
    else:
        index = index.tz_convert(timezone.utc)
    df = df.copy()
    df.index = index
    current_bar_open = pd.Timestamp(utc_now()).floor(pandas_timeframe(gp.TIMEFRAME))
    if index[-1] >= current_bar_open:
        return df.iloc[:-1].copy()
    return df


def align_complete_common_pair_frame(
    df: pd.DataFrame,
    pairs: tuple[str, ...],
    *,
    max_stale_bars: int = 2,
) -> pd.DataFrame:
    required_cols = [f"{pair}_close" for pair in pairs]
    aligned = df.dropna(subset=required_cols).sort_index()
    if aligned.empty:
        raise ValueError("No complete common pair bars available for pairwise shadow runner.")
    latest_common = pd.Timestamp(aligned.index[-1])
    if latest_common.tzinfo is None:
        latest_common = latest_common.tz_localize(timezone.utc)
    else:
        latest_common = latest_common.tz_convert(timezone.utc)
    interval_delta = pd.Timedelta(pandas_timeframe(gp.TIMEFRAME))
    current_bar_open = pd.Timestamp(utc_now()).tz_convert(timezone.utc).floor(pandas_timeframe(gp.TIMEFRAME))
    latest_completed_bar = current_bar_open - interval_delta
    stale_cap = latest_completed_bar - interval_delta * max(int(max_stale_bars), 0)
    if latest_common < stale_cap:
        raise ValueError(
            "Common pair frame is stale for pairwise shadow runner. "
            f"latest_common={latest_common.isoformat()} latest_completed_bar={latest_completed_bar.isoformat()}"
        )
    return aligned


def merge_recent_pair_frame(df: pd.DataFrame, pair: str, recent: pd.DataFrame) -> pd.DataFrame:
    if recent.empty:
        return df
    pair_prefix = f"{pair}_"
    renamed = recent.rename(columns={column: f"{pair_prefix}{column}" for column in recent.columns}).sort_index()
    existing_pair_cols = [column for column in df.columns if column.startswith(pair_prefix)]
    other = df.drop(columns=existing_pair_cols)
    current_pair = df[existing_pair_cols].copy()
    merged_pair = pd.concat([current_pair, renamed]).sort_index()
    merged_pair = merged_pair[~merged_pair.index.duplicated(keep="last")]
    return pd.concat([other, merged_pair], axis=1).sort_index()


def load_live_frame(pairs: tuple[str, ...], refresh_live_data: bool, recent_days: int = 10) -> pd.DataFrame:
    df = gp.load_all_pairs(
        pairs=list(pairs),
        start=PAIRWISE_HISTORY_START,
        end=None,
        refresh_cache=False,
    )
    if refresh_live_data:
        now = utc_now()
        start_dt = now - timedelta(days=recent_days)
        end_dt = now
        for pair in pairs:
            try:
                recent = gp.fetch_klines(pair, gp.TIMEFRAME, start_dt, end_dt)
                df = merge_recent_pair_frame(df, pair, recent)
            except Exception as exc:
                print(f"  {pair}: live refresh skipped ({exc})")
                continue
    df = drop_incomplete_bar(df)
    df = align_complete_common_pair_frame(df, pairs)
    if len(df) < 20:
        raise ValueError("No local market data available for pairwise shadow runner.")
    return df


def build_daily_close(df_all: pd.DataFrame, pairs: tuple[str, ...]) -> pd.DataFrame:
    cols = [df_all[f"{pair}_close"].rename(pair) for pair in pairs]
    return pd.concat(cols, axis=1).resample("1D").last().dropna().sort_index()


def quantize_amount(exchange: ccxt.binanceusdm, symbol: str, amount: float) -> float:
    raw_amount = abs(float(amount))
    if raw_amount <= EPSILON:
        return 0.0
    market = exchange.market(symbol)
    amount_precision = market.get("precision", {}).get("amount")
    if amount_precision is not None and raw_amount < float(amount_precision):
        return 0.0
    try:
        precise = float(exchange.amount_to_precision(symbol, raw_amount))
    except ccxt.InvalidOrder:
        return 0.0
    min_amount = (
        market.get("limits", {}).get("amount", {}).get("min")
        if isinstance(market.get("limits"), dict)
        else None
    )
    if min_amount is not None and precise < float(min_amount):
        return 0.0
    return precise


def ensure_symbol_margin_settings(exchange: ccxt.binanceusdm, symbol: str, leverage: int) -> dict[str, Any]:
    result: dict[str, Any] = {"symbol": symbol, "margin_mode": None, "leverage": None, "warnings": []}
    try:
        exchange.set_margin_mode("isolated", symbol)
        result["margin_mode"] = "isolated"
    except Exception as exc:
        result["warnings"].append(f"margin_mode:{exc}")
    try:
        exchange.set_leverage(leverage, symbol)
        result["leverage"] = leverage
    except Exception as exc:
        result["warnings"].append(f"leverage:{exc}")
    return result


def fetch_open_position_map(exchange: ccxt.binanceusdm, pairs: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    positions_by_pair: dict[str, dict[str, Any]] = {}
    try:
        positions = exchange.fetch_positions([PAIR_TO_MARKET[pair] for pair in pairs])
    except Exception:
        return positions_by_pair

    reverse_map = {market: pair for pair, market in PAIR_TO_MARKET.items()}
    for pos in positions:
        symbol = pos.get("symbol")
        pair = reverse_map.get(symbol)
        if pair is None or pair not in pairs:
            continue
        info = pos.get("info", {}) if isinstance(pos.get("info"), dict) else {}
        qty = info.get("positionAmt")
        if qty is None:
            qty = pos.get("contracts", 0.0)
        qty_value = float(qty)
        if abs(qty_value) <= EPSILON:
            continue
        entry_price = info.get("entryPrice")
        if entry_price is None:
            entry_price = pos.get("entryPrice", 0.0)
        mark_price = info.get("markPrice")
        if mark_price is None:
            mark_price = pos.get("markPrice", 0.0)
        positions_by_pair[pair] = {
            "pair": pair,
            "symbol": symbol,
            "qty": qty_value,
            "side": "LONG" if qty_value > 0.0 else "SHORT",
            "entry_price": float(entry_price) if entry_price is not None else 0.0,
            "mark_price": float(mark_price) if mark_price is not None else 0.0,
        }
    return positions_by_pair


def fetch_position_qty_map(exchange: ccxt.binanceusdm, pairs: tuple[str, ...]) -> dict[str, float]:
    qty_map = {pair: 0.0 for pair in pairs}
    for pair, position in fetch_open_position_map(exchange, pairs).items():
        qty_map[pair] = float(position["qty"])
    return qty_map


def reconcile_pairwise_target_positions(
    exchange: ccxt.binanceusdm,
    equity: float,
    target_weights: dict[str, float],
    pairs: tuple[str, ...],
    execute: bool,
    leverage: int,
) -> list[dict[str, Any]]:
    current_qtys = fetch_position_qty_map(exchange, pairs)
    actions: list[dict[str, Any]] = []
    for pair in pairs:
        symbol = PAIR_TO_MARKET[pair]
        price = float(exchange.fetch_ticker(symbol).get("last") or exchange.fetch_ticker(symbol).get("mark") or exchange.fetch_ticker(symbol).get("close"))
        target_notional = float(target_weights.get(pair, 0.0)) * float(equity)
        target_qty = target_notional / price if abs(price) > EPSILON else 0.0
        current_qty = float(current_qtys.get(pair, 0.0))
        diff_qty = target_qty - current_qty
        diff_notional = abs(diff_qty * price)
        amount = quantize_amount(exchange, symbol, diff_qty)
        action = {
            "pair": pair,
            "symbol": symbol,
            "price": price,
            "target_weight": float(target_weights.get(pair, 0.0)),
            "target_notional": target_notional,
            "current_qty": current_qty,
            "target_qty": target_qty,
            "diff_qty": diff_qty,
            "diff_notional": diff_notional,
            "amount": amount,
            "side": "buy" if diff_qty > 0 else "sell",
            "placed": False,
            "margin_action": ensure_symbol_margin_settings(exchange, symbol, leverage),
        }
        if diff_notional < 25.0 or amount <= 0.0:
            action["amount"] = 0.0
            actions.append(action)
            continue
        if execute:
            order = exchange.create_market_order(symbol, action["side"], amount)
            action["placed"] = True
            action["order_id"] = order.get("id")
        actions.append(action)
    return actions


def load_promotion_gate(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {"status": "missing", "selected_candidate_ready_for_merge": False, "path": str(path)}
    payload = json.loads(path.read_text())
    decision = payload.get("promotion_decision")
    if not isinstance(decision, dict):
        decision = payload.get("decision", {})
    return {
        "status": str(decision.get("status", "unknown")),
        "selected_candidate_ready_for_merge": bool(
            decision.get("selected_candidate_ready_for_merge", decision.get("ready_for_merge", False))
        ),
        "path": str(path),
        "decision": decision,
    }


def detect_candidate_kind(candidate: dict[str, Any]) -> str:
    if not isinstance(candidate, dict):
        return "unknown"
    if str(candidate.get("candidate_kind") or "").strip().lower() == "fractal_tree":
        return "fractal_tree"
    if candidate.get("tree"):
        return "fractal_tree"
    if candidate.get("pair_configs"):
        return "pairwise_candidate"
    return "unknown"


def infer_strategy_artifact_root(anchor_file: str | Path) -> Path:
    anchor = Path(anchor_file)
    if anchor.parent.name == "models":
        return anchor.parent.parent
    return anchor.parent


def resolve_strategy_artifact_path(raw_path: str | Path, anchor_file: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    anchor = Path(anchor_file)
    root = infer_strategy_artifact_root(anchor)
    candidates = [root / path, anchor.parent / path, path]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return root / path


def extract_strategy_artifact_reference(summary_payload: dict[str, Any], *field_names: str) -> str | None:
    containers = [
        summary_payload,
        summary_payload.get("search") or {},
        summary_payload.get("artifacts") or {},
    ]
    for container in containers:
        if not isinstance(container, dict):
            continue
        for field_name in field_names:
            value = container.get(field_name)
            if value:
                return str(value)
    return None


def apply_leaf_gene_to_overlay_params(params: OverlayParams, gene: LeafGene | None) -> OverlayParams:
    leaf_gene = gene if isinstance(gene, LeafGene) else LeafGene()
    return replace(
        params,
        target_vol_ann=max(float(params.target_vol_ann) * float(leaf_gene.target_vol_scale), 0.0),
        gross_cap=max(float(params.gross_cap) * float(leaf_gene.gross_cap_scale), 0.0),
        kill_switch_pct=max(float(params.kill_switch_pct) * float(leaf_gene.kill_switch_scale), EPSILON),
        cooldown_days=max(int(round(float(params.cooldown_days) * float(leaf_gene.cooldown_scale))), 0),
    )


def load_strategy_bundle(
    summary_path: str | Path,
    base_summary_path: str | Path,
    model_path: str | Path,
    *,
    candidate_key: str = "selected_candidate",
) -> dict[str, Any]:
    summary_file = Path(summary_path)
    selected_summary = json.loads(summary_file.read_text())
    embedded_base_summary = extract_strategy_artifact_reference(
        selected_summary,
        "baseline_summary_path",
        "base_summary_path",
    )
    embedded_model = extract_strategy_artifact_reference(selected_summary, "model_path")
    base_summary_file = resolve_strategy_artifact_path(embedded_base_summary or base_summary_path, summary_file)
    model_file = resolve_strategy_artifact_path(embedded_model or model_path, summary_file)
    base_summary = json.loads(base_summary_file.read_text())
    library = list(iter_params())
    model, _ = load_model(model_file)
    compiled = gp.toolbox.compile(expr=model)
    selected_candidate = selected_summary[candidate_key]
    candidate_kind = detect_candidate_kind(selected_candidate)
    expert_pool = None
    route_thresholds = tuple(float(v) for v in (selected_summary.get("search", {}).get("route_thresholds") or (0.35, 0.50, 0.65, 0.80)))
    if candidate_kind == "fractal_tree":
        embedded_pool = selected_summary.get("expert_pool")
        if isinstance(embedded_pool, list) and embedded_pool:
            expert_pool = embedded_pool
        else:
            search_meta = selected_summary.get("search") or {}
            expert_summaries = [
                str(resolve_strategy_artifact_path(part, summary_file))
                for part in (search_meta.get("expert_summaries") or [])
                if part
            ]
            pool_size = int(search_meta.get("expert_pool_size", 0) or 0)
            summary_pairs = tuple(selected_summary.get("pairs") or DEFAULT_PAIRS)
            if expert_summaries and pool_size > 0:
                expert_pool, _ = build_expert_pool(expert_summaries, pool_size, summary_pairs)
    payload = {
        "summary_path": str(summary_file),
        "base_summary_path": str(base_summary_file),
        "model_path": str(model_file),
        "library_size": len(library),
        "selected_candidate": selected_candidate,
        "candidate_key": candidate_key,
        "candidate_kind": candidate_kind,
        "summary_metadata": selected_summary,
        "base_summary_metadata": base_summary,
        "expert_pool": expert_pool,
        "route_thresholds": route_thresholds,
        "library": library,
        "compiled_model": compiled,
    }
    return payload


def resolve_runtime_candidate(
    df_all: pd.DataFrame,
    pairs: tuple[str, ...],
    bundle: dict[str, Any],
) -> dict[str, Any]:
    candidate = copy.deepcopy(bundle["selected_candidate"])
    candidate_kind = str(bundle.get("candidate_kind") or detect_candidate_kind(candidate))
    if candidate_kind != "fractal_tree":
        candidate.setdefault("candidate_kind", candidate_kind)
        return candidate

    expert_pool = bundle.get("expert_pool") or []
    if not expert_pool:
        raise ValueError("Fractal runtime candidate is missing expert_pool metadata.")
    tree_payload = candidate.get("tree")
    if not isinstance(tree_payload, dict):
        raise ValueError("Fractal runtime candidate is missing serialized tree payload.")
    route_thresholds = tuple(float(v) for v in (bundle.get("route_thresholds") or (0.35, 0.50, 0.65, 0.80)))
    label_horizon = str(candidate.get("label_horizon") or "5m")
    observation_mode = str(candidate.get("observation_mode") or "time")
    tree = deserialize_tree(tree_payload)
    feature_series = build_market_features(df_all, pairs)
    feature_arrays = materialize_feature_arrays(feature_series, pd.DatetimeIndex(df_all.index))
    projected_arrays = project_feature_arrays_by_observation_mode(feature_arrays, observation_mode)
    horizon_arrays = apply_label_horizon_to_feature_arrays(projected_arrays, label_horizon)
    leaf_codes, leaf_catalog = evaluate_tree_leaf_codes(tree, horizon_arrays)
    if len(leaf_codes) == 0 or not leaf_catalog:
        raise ValueError("Fractal runtime candidate did not produce an active leaf.")
    active_leaf_code = int(leaf_codes[-1])
    active_leaf = leaf_catalog[active_leaf_code]
    active_pair_configs = {
        pair: apply_leaf_gene_to_pair_config(
            expert_pool[int(active_leaf.expert_idx)]["pair_configs"][pair],
            active_leaf.gene,
            route_thresholds,
            int(bundle["library_size"]),
        )
        for pair in pairs
    }
    feature_snapshot = {
        spec.feature: float(horizon_arrays[spec.feature][-1])
        for spec in collect_specs(tree)
        if spec.feature in horizon_arrays and len(horizon_arrays[spec.feature]) > 0
    }
    candidate["candidate_kind"] = "fractal_tree"
    candidate["pair_configs"] = active_pair_configs
    candidate["active_leaf"] = {
        "leaf_index": active_leaf_code,
        "expert_idx": int(active_leaf.expert_idx),
        "gene": json_safe(active_leaf.gene),
    }
    candidate["runtime_context"] = {
        "observation_mode": observation_mode,
        "label_horizon": label_horizon,
        "feature_snapshot": feature_snapshot,
    }
    return candidate


def build_pairwise_plan(
    df_all: pd.DataFrame,
    pairs: tuple[str, ...],
    bundle: dict[str, Any],
    equity: float,
    execute: bool,
    leverage: int,
    exchange: ccxt.binanceusdm | None = None,
) -> dict[str, Any]:
    candidate = resolve_runtime_candidate(df_all, pairs, bundle)
    library = bundle["library"]
    compiled = bundle["compiled_model"]
    library_lookup = build_library_lookup(library)
    daily_close = build_daily_close(df_all, pairs)
    market_day = normalize_day(daily_close.index.max())
    effective_day = market_day + pd.Timedelta(days=1)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }

    pair_plans: list[dict[str, Any]] = []
    target_weights: dict[str, float] = {}
    latest_positions = fetch_open_position_map(exchange, pairs) if (execute and exchange is not None) else {}
    current_qty_map = {pair: float(latest_positions.get(pair, {}).get("qty", 0.0)) for pair in pairs}

    for pair in pairs:
        pair_config = candidate["pair_configs"][pair]
        route_state_mode = normalize_route_state_mode(pair_config.get("route_state_mode"))
        overlay_inputs = build_overlay_inputs(df_all, pairs, regime_pair=pair)
        context = {
            "close": df_all[[f"{asset}_close" for asset in pairs]]
            .rename(columns={f"{asset}_close": asset for asset in pairs})
            .sort_index(),
            "bucket_codes": {
                float(pair_config["route_breadth_threshold"]): build_route_bucket_codes(
                    pd.DatetimeIndex(df_all.index),
                    overlay_inputs,
                    float(pair_config["route_breadth_threshold"]),
                    route_state_mode=route_state_mode,
                ).astype("int64")
            },
            "regime": overlay_inputs["btc_regime_daily"].reindex(df_all.index.normalize(), method="ffill").fillna(0.0).to_numpy(dtype="float64"),
            "breadth": overlay_inputs["breadth_daily"].reindex(df_all.index.normalize(), method="ffill").fillna(0.0).to_numpy(dtype="float64"),
            "vol_ann": overlay_inputs["vol_ann_bar"].reindex(pd.DatetimeIndex(df_all.index)).ffill().bfill().fillna(0.0).to_numpy(dtype="float64"),
            "smooth_signal_matrix": np.vstack(
                [
                    raw_signal_all[pair].ewm(span=span, adjust=False).mean().to_numpy(dtype="float64")
                    for span in library_lookup["spans"]
                ]
            ),
        }

        signal_idx = len(df_all) - 1
        route_threshold = float(pair_config["route_breadth_threshold"])
        bucket_code = int(context["bucket_codes"][route_threshold][signal_idx])
        mapping = normalize_mapping_indices(pair_config["mapping_indices"], route_state_mode)
        active_idx = int(mapping[bucket_code])
        route_state_name = route_state_names(route_state_mode)[bucket_code]
        params = library[active_idx]
        active_leaf = candidate.get("active_leaf") or {}
        if active_leaf:
            params = apply_leaf_gene_to_overlay_params(params, LeafGene(**((active_leaf.get("gene") or {}))))
        signal_pos = int(library_lookup["signal_pos"][active_idx])
        signal_pct = float(np.clip(context["smooth_signal_matrix"][signal_pos, signal_idx], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(context["regime"][signal_idx])
        breadth_score = float(context["breadth"][signal_idx])
        long_ok = regime_score >= params.regime_threshold and breadth_score >= params.breadth_threshold
        short_ok = regime_score <= -params.regime_threshold and breadth_score <= (1.0 - params.breadth_threshold)
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = float(context["vol_ann"][signal_idx])
        if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = min(
                params.target_vol_ann / bar_vol_ann,
                params.gross_cap / max(abs(requested_weight), 1e-8),
            )
            requested_weight *= float(vol_scale)
        target_weight = float(np.clip(requested_weight, -params.gross_cap, params.gross_cap))
        current_weight = 0.0
        if execute and exchange is not None:
            symbol = PAIR_TO_MARKET[pair]
            price = float(exchange.fetch_ticker(symbol).get("last") or exchange.fetch_ticker(symbol).get("mark") or exchange.fetch_ticker(symbol).get("close"))
            if abs(equity) > EPSILON:
                current_weight = float(current_qty_map.get(pair, 0.0) * price / equity)

        pair_plans.append(
            {
                "pair": pair,
                "symbol": PAIR_TO_MARKET[pair],
                "bucket_code": bucket_code,
                "route_state_mode": route_state_mode,
                "route_state_name": route_state_name,
                "active_library_index": active_idx,
                "candidate_kind": candidate.get("candidate_kind", "pairwise_candidate"),
                "active_leaf": active_leaf or None,
                "params": asdict(params),
                "regime_score": regime_score,
                "breadth_score": breadth_score,
                "signal_pct": signal_pct,
                "requested_weight": float(requested_weight),
                "target_weight": target_weight,
                "current_weight": current_weight,
                "current_qty": float(current_qty_map.get(pair, 0.0)),
                "long_gate": bool(long_ok),
                "short_gate": bool(short_ok),
                "bar_vol_ann": bar_vol_ann,
            }
        )
        target_weights[pair] = target_weight

    plan = {
        "strategy_class": "pairwise_regime_mixture_shadow_live",
        "market_day": market_day.date().isoformat(),
        "effective_day": effective_day.date().isoformat(),
        "signal_timestamp": pd.Timestamp(df_all.index.max()).isoformat(),
        "equity": float(equity),
        "leverage": float(leverage),
        "pairs": list(pairs),
        "selected_candidate": candidate,
        "pair_plans": pair_plans,
        "target_weights": target_weights,
        "gross_target_weight": float(sum(abs(v) for v in target_weights.values())),
        "session_type": "flat" if float(sum(abs(v) for v in target_weights.values())) <= EPSILON else "pairwise",
        "promotion_gate": bundle.get("promotion_gate"),
        "summary_path": bundle["summary_path"],
        "base_summary_path": bundle["base_summary_path"],
        "model_path": bundle["model_path"],
    }
    if execute and exchange is not None:
        plan["open_positions"] = latest_positions
    return plan


def decision_fingerprint(decision: dict[str, Any]) -> str:
    stable_payload = {
        "market_day": decision.get("market_day"),
        "effective_day": decision.get("effective_day"),
        "pairs": decision.get("pairs"),
        "selected_candidate": decision.get("selected_candidate"),
        "target_weights": decision.get("target_weights"),
        "promotion_gate": decision.get("promotion_gate"),
    }
    encoded = json.dumps(json_stable(stable_payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def build_shadow_snapshot(state: dict[str, Any], plan: dict[str, Any], exchange_snapshot: dict[str, Any] | None) -> dict[str, Any]:
    captured_at = utc_now().isoformat()
    snapshot = {
        "captured_at": captured_at,
        "equity": plan.get("equity"),
        "positions": exchange_snapshot or [],
        "exchange_error": None,
        "plan": plan,
    }
    state["latest_runtime_snapshot"] = snapshot
    state["latest_plan"] = plan
    state["selected_candidate"] = plan.get("selected_candidate")
    state["promotion_gate"] = plan.get("promotion_gate")
    shadow_paper = state.setdefault("shadow_paper", {})
    shadow_paper["enabled"] = True
    shadow_paper["last_signal_timestamp"] = plan.get("signal_timestamp")
    shadow_paper["last_updated_at"] = captured_at
    shadow_paper["current_weights"] = {
        pair: float(plan.get("target_weights", {}).get(pair, 0.0))
        for pair in plan.get("pairs", [])
    }
    shadow_paper["cooldown_bars_left"] = {
        pair_plan.get("pair"): 0
        for pair_plan in plan.get("pair_plans", [])
        if pair_plan.get("pair")
    }
    shadow_paper["equity"] = float(shadow_paper.get("equity") or plan.get("equity") or SHADOW_DEFAULT_EQUITY)
    shadow_paper["peak_equity"] = float(
        max(float(shadow_paper.get("peak_equity") or 0.0), float(shadow_paper.get("equity") or 0.0))
    )
    state["latest_decision_snapshot"] = {
        "generated_at": captured_at,
        "strategy_class": "pairwise_regime_mixture_shadow_live",
        "session_type": plan.get("session_type"),
        "target_weights": plan.get("target_weights", {}),
        "pair_plans": plan.get("pair_plans", []),
        "rationale": {
            "mode": "pairwise",
            "gross_leverage": plan.get("gross_target_weight", 0.0),
            "net_exposure": float(sum(float(v) for v in (plan.get("target_weights", {}) or {}).values())),
            "signal_timestamp": plan.get("signal_timestamp"),
        },
    }
    return snapshot


def refresh_decision_journal(state: dict[str, Any], decision: dict[str, Any], decision_log_path: str | Path) -> dict[str, Any]:
    journal = state.setdefault("decision_journal", {})
    journal["log_path"] = str(decision_log_path)
    fingerprint = decision_fingerprint(decision)
    if fingerprint != journal.get("last_fingerprint"):
        recorded_at = utc_now().isoformat()
        payload = {"recorded_at": recorded_at, "fingerprint": fingerprint, **decision}
        append_jsonl(decision_log_path, payload)
        journal["last_fingerprint"] = fingerprint
        journal["last_recorded_at"] = recorded_at
    return decision


def build_decision(plan: dict[str, Any], execute: bool, order_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "strategy_class": plan["strategy_class"],
        "market_day": plan["market_day"],
        "effective_day": plan["effective_day"],
        "signal_timestamp": plan.get("signal_timestamp"),
        "session_type": plan.get("session_type"),
        "pairs": plan["pairs"],
        "selected_candidate": plan["selected_candidate"],
        "target_weights": plan["target_weights"],
        "gross_target_weight": plan["gross_target_weight"],
        "promotion_gate": plan["promotion_gate"],
        "execution_mode": "execute" if execute else "shadow",
        "order_actions": order_actions or [],
    }


def run_once(args: argparse.Namespace) -> None:
    pairs = resolve_pairs(args.pairs)
    state = load_shadow_state(args.state_path, args.decision_log)
    exchange = get_exchange(args.mode)
    register = {
        "args": args,
        "state": state,
        "exchange": exchange,
    }
    RUNTIME_CONTEXT.update(register)
    try:
        execute_requested = bool(getattr(args, "execute", False))
        force_execute = bool(getattr(args, "force_execute", False))
        gate = load_promotion_gate(args.promotion_report)
        state["promotion_gate"] = gate
        bundle = load_strategy_bundle(args.summary, args.base_summary, args.model)
        bundle["promotion_gate"] = gate

        if execute_requested and not gate.get("selected_candidate_ready_for_merge", False) and not force_execute:
            raise ValueError(
                "Promotion gate is not open. Regenerate the pairwise stress report or pass --force-execute."
            )

        equity = float(args.equity) if args.equity is not None else float(fetch_equity(exchange) if execute_requested else gp.INITIAL_CASH)
        df_all = load_live_frame(
            pairs=tuple(pairs),
            refresh_live_data=bool(getattr(args, "refresh_live_data", True)),
        )
        if df_all.empty:
            raise ValueError("No local market data available for pairwise shadow runner.")

        plan = build_pairwise_plan(
            df_all=df_all,
            pairs=pairs,
            bundle=bundle,
            equity=equity,
            execute=execute_requested,
            leverage=int(args.leverage),
            exchange=exchange if execute_requested else None,
        )

        order_actions: list[dict[str, Any]] = []
        if execute_requested:
            order_actions = reconcile_pairwise_target_positions(
                exchange=exchange,
                equity=equity,
                target_weights=plan["target_weights"],
                pairs=pairs,
                execute=True,
                leverage=int(args.leverage),
            )
            plan["order_actions"] = order_actions

        snapshot = build_shadow_snapshot(state, plan, None)
        decision = build_decision(plan, execute_requested, order_actions)
        refresh_decision_journal(state, decision, args.decision_log)
        save_state(args.state_path, state)

        print(json.dumps(json_ready({
            "status": "executed" if execute_requested else "shadow",
            "promotion_gate": gate,
            "plan": plan,
            "snapshot": snapshot,
        }), ensure_ascii=False, indent=2))
        print(f"\nState saved: {args.state_path}")
    finally:
        RUNTIME_CONTEXT.update({"args": None, "state": None, "exchange": None})


def loop(args: argparse.Namespace) -> None:
    state = load_shadow_state(args.state_path, args.decision_log)
    state.setdefault("runtime_health", {})["process_started_at"] = utc_now().isoformat()
    save_state(args.state_path, state)
    try:
        while True:
            started = utc_now()
            state = load_shadow_state(args.state_path, args.decision_log)
            update_runtime_health(
                state,
                last_loop_started_at=started.isoformat(),
                pid=os.getpid(),
                poll_seconds=int(args.poll_seconds),
                process_started_at=state.get("runtime_health", {}).get("process_started_at", started.isoformat()),
            )
            save_state(args.state_path, state)
            try:
                run_once(args)
                completed = utc_now()
                state = load_shadow_state(args.state_path, args.decision_log)
                runtime_health = update_runtime_health(
                    state,
                    last_loop_completed_at=completed.isoformat(),
                    last_success_at=completed.isoformat(),
                    last_error_at=None,
                    last_error_message=None,
                    consecutive_errors=0,
                    loop_success_count=int(state.get("runtime_health", {}).get("loop_success_count", 0)) + 1,
                    last_duration_seconds=round((completed - started).total_seconds(), 3),
                    pid=os.getpid(),
                    poll_seconds=int(args.poll_seconds),
                )
                save_state(args.state_path, state)
                print(json.dumps(json_ready({"status": "loop_success", "runtime_health": runtime_health}), indent=2))
            except Exception as exc:
                failed = utc_now()
                state = load_shadow_state(args.state_path, args.decision_log)
                runtime_health = update_runtime_health(
                    state,
                    last_loop_completed_at=failed.isoformat(),
                    last_error_at=failed.isoformat(),
                    last_error_message=str(exc),
                    consecutive_errors=int(state.get("runtime_health", {}).get("consecutive_errors", 0)) + 1,
                    loop_error_count=int(state.get("runtime_health", {}).get("loop_error_count", 0)) + 1,
                    last_duration_seconds=round((failed - started).total_seconds(), 3),
                    pid=os.getpid(),
                    poll_seconds=int(args.poll_seconds),
                )
                save_state(args.state_path, state)
                print(json.dumps(json_safe({"status": "loop_error", "error": str(exc), "runtime_health": runtime_health}), indent=2))
            time.sleep(max(5, int(args.poll_seconds)))
    except KeyboardInterrupt:
        return


def main() -> None:
    args = parse_args()
    if args.command == "status":
        run_once(args)
        return
    if args.command == "run-once":
        run_once(args)
        return
    if args.command == "loop":
        loop(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
