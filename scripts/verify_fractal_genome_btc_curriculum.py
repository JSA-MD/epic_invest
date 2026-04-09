#!/usr/bin/env python3
"""BTC-only independent verifier/backtester for fractal genome curriculum."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from fractal_genome_core import (
    AndCell,
    ConditionNode,
    ConditionSpec,
    collect_leaves,
    collect_leaf_keys,
    deserialize_tree,
    FilterDecision,
    LeafNode,
    NotCell,
    OrCell,
    ThresholdCell,
    collect_specs,
    crossover_tree,
    evaluate_tree_leaf_codes,
    heuristic_semantic_filter,
    mutate_tree,
    random_tree,
    semantic_filter,
    serialize_tree,
    tree_depth,
    tree_key,
    tree_logic_depth,
    tree_logic_size,
    tree_size,
)
from replay_regime_mixture_realistic import load_model
from search_gp_drawdown_overlay import OverlayParams
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_fractal_genome import (
    build_condition_options as shared_build_condition_options,
    build_expert_pool,
    build_feature_specs,
    build_leaf_runtime_arrays_for_pair,
    build_leaf_runtime_arrays_from_pair_configs,
    build_market_features,
    build_seed_trees,
    expert_arrays_for_pair,
    fast_fractal_replay_from_context,
    materialize_feature_arrays as shared_materialize_feature_arrays,
    select_near_frontier_structural_winner,
    structural_bonus_from_metrics,
    tournament_select,
)
from search_pair_subset_regime_mixture import BAR_FACTOR, BARS_PER_DAY, build_fast_context, build_library_lookup, summarize_single_result

try:
    from numba import njit  # noqa: F401
except ImportError:  # pragma: no cover
    njit = None


UTC = timezone.utc
BTC_PAIR = "BTCUSDT"
DEFAULT_EXPERT_SUMMARIES = (
    "models/gp_regime_mixture_btc_bnb_pairwise_repair_summary.json,models/gp_regime_mixture_btc_bnb_pairwise_fullgrid_seed_pool.json"
)
DEFAULT_FUNDING_CACHE = gp.DATA_DIR / "BTCUSDT_funding_2022-04-06_2026-04-06.csv"
DEFAULT_BASELINE_SUMMARY = "models/gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BTC-only fractal-genome search/backtests with a depth curriculum.",
    )
    parser.add_argument(
        "--pairs",
        default=BTC_PAIR,
        help="BTC-only verifier uses a single pair and rejects other values.",
    )
    parser.add_argument(
        "--summary-out",
        default="/tmp/fractal_genome_btc_curriculum_summary.json",
    )
    parser.add_argument(
        "--command-log",
        default="/tmp/fractal_genome_btc_curriculum_command.json",
    )
    parser.add_argument(
        "--depth-curriculum",
        default="1,2,3",
    )
    parser.add_argument(
        "--logic-curriculum",
        default="1,1,2",
    )
    parser.add_argument(
        "--expert-summaries",
        default=DEFAULT_EXPERT_SUMMARIES,
    )
    parser.add_argument("--expert-pool-size", type=int, default=12)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--elite-count", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--route-thresholds",
        default="0.35,0.50,0.65,0.80",
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--funding-cache",
        default=str(DEFAULT_FUNDING_CACHE),
    )
    parser.add_argument(
        "--baseline-summary",
        default=DEFAULT_BASELINE_SUMMARY,
    )
    parser.add_argument(
        "--search-seed-summary",
        default="",
        help="Optional fractal search summary JSON used to seed verifier stages with previously discovered trees.",
    )
    parser.add_argument(
        "--walk-forward-folds",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--walk-forward-test-months",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--commission-stress",
        default="1.0,1.5,2.0",
    )
    parser.add_argument(
        "--stress-survival-threshold",
        type=float,
        default=0.67,
    )
    parser.add_argument(
        "--stress-survival-mean-threshold",
        type=float,
        default=0.67,
    )
    parser.add_argument(
        "--stress-survival-min-threshold",
        type=float,
        default=0.67,
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, FilterDecision):
        return {
            "accepted": value.accepted,
            "source": value.source,
            "reason": value.reason,
            "llm_prompt": value.llm_prompt,
        }
    return value


def parse_csv_tuple(raw: str, cast: type[Any]) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def ensure_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def discover_btc_bounds(data_path: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    bounds = pd.read_csv(data_path, usecols=["open_time"])
    first = pd.to_datetime(bounds["open_time"].iloc[0], utc=True)
    last = pd.to_datetime(bounds["open_time"].iloc[-1], utc=True)
    return first, last


def load_cached_funding(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    df = pd.read_csv(path)
    if "fundingTime" not in df.columns or "fundingRate" not in df.columns:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, format="mixed")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df.dropna(subset=["fundingTime", "fundingRate"]).sort_values("fundingTime").reset_index(drop=True)


def build_btc_feature_frame(df: pd.DataFrame) -> dict[str, pd.Series]:
    return build_market_features(df, (BTC_PAIR,))


def materialize_feature_arrays(features: dict[str, pd.Series], index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return shared_materialize_feature_arrays(features, index)


def build_condition_options() -> list[ConditionSpec]:
    return shared_build_condition_options(build_feature_specs((BTC_PAIR,)))


def build_btc_overlay_inputs(df: pd.DataFrame) -> dict[str, pd.Series]:
    close = df[f"{BTC_PAIR}_close"].sort_index()
    daily_close = close.resample("1D").last().dropna()
    btc_regime = 0.60 * daily_close.pct_change(3) + 0.40 * daily_close.pct_change(14)
    breadth = daily_close.pct_change(3).gt(0.0).astype(float)
    bar_ret = close.pct_change()
    vol_ann = bar_ret.rolling(12 * 24 * 3).std() * BAR_FACTOR
    return {
        "btc_regime_daily": btc_regime,
        "breadth_daily": breadth,
        "vol_ann_bar": vol_ann,
    }


def build_window_ranges(first_ts: pd.Timestamp, last_ts: pd.Timestamp) -> tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...]:
    return (
        ("recent_2m", last_ts - pd.DateOffset(months=2), last_ts),
        ("recent_6m", last_ts - pd.DateOffset(months=6), last_ts),
        ("full_since_first", first_ts, last_ts),
    )


def build_walk_forward_ranges(
    first_ts: pd.Timestamp,
    last_ts: pd.Timestamp,
    folds: int,
    test_months: int,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    current_end = last_ts
    for idx in range(folds):
        start = current_end - pd.DateOffset(months=test_months)
        start = max(start, first_ts)
        if start >= current_end:
            break
        ranges.append((f"wf_{idx + 1}", start, current_end))
        current_end = start
        if current_end <= first_ts:
            break
    ranges.reverse()
    return ranges


def build_btc_expert_pool(summary_paths: list[str], pool_size: int) -> list[dict[str, Any]]:
    raw_pool, _ = build_expert_pool(summary_paths, max(pool_size * 4, pool_size), (BTC_PAIR,))
    if len(raw_pool) < 2:
        raise RuntimeError("Need at least 2 BTC experts to run the BTC-only verifier.")
    return raw_pool[:pool_size]


def load_seed_trees_from_summary(summary_path: str) -> list[Any]:
    if not summary_path:
        return []
    path = Path(summary_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    raw_trees: list[dict[str, Any]] = []
    selected = payload.get("selected_candidate")
    if isinstance(selected, dict) and isinstance(selected.get("tree"), dict):
        raw_trees.append(selected["tree"])
    for item in payload.get("top_candidates", []):
        if isinstance(item, dict) and isinstance(item.get("tree"), dict):
            raw_trees.append(item["tree"])
    out: list[Any] = []
    seen: set[str] = set()
    for raw in raw_trees:
        node = deserialize_tree(raw)
        key = tree_key(node)
        if key in seen:
            continue
        seen.add(key)
        out.append(node)
    return out


def build_baseline_leaf_runtime(
    baseline_summary: dict[str, Any],
    route_thresholds: tuple[float, ...],
    library_size: int,
) -> dict[str, np.ndarray]:
    pair_config = baseline_summary["selected_candidate"]["pair_configs"][BTC_PAIR]
    return build_leaf_runtime_arrays_from_pair_configs(
        [{"pair_config": pair_config}],
        route_thresholds,
        library_size,
    )


def validate_feature_health(feature_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    health = {}
    for name, arr in feature_arrays.items():
        finite = bool(np.isfinite(arr).all())
        health[name] = {
            "finite": finite,
            "std": float(np.nanstd(arr)) if len(arr) else 0.0,
            "min": float(np.nanmin(arr)) if len(arr) else 0.0,
            "max": float(np.nanmax(arr)) if len(arr) else 0.0,
        }
        assert finite, f"feature array contains non-finite values: {name}"
    assert health["btc_regime"]["std"] > 0.0, "btc_regime must vary across the dataset"
    assert health["breadth"]["std"] > 0.0, "breadth proxy must vary across the dataset"
    breadth_values = feature_arrays["breadth"][np.isfinite(feature_arrays["breadth"])]
    assert np.all(np.logical_or(breadth_values == 0.0, breadth_values == 1.0)), "BTC breadth must be binary to match searcher parity"
    return health


def stage_complexity_signature(stage: dict[str, Any]) -> tuple[int, int, int, int, int]:
    selected = stage["selected_candidate"]
    return (
        int(selected["tree_depth"]),
        int(selected["logic_depth"]),
        int(selected["tree_size"]),
        int(selected["logic_size"]),
        int(selected["condition_count"]),
    )


def stage_window_metric(stage: dict[str, Any], window: str, metric: str) -> float:
    return float(stage["selected_candidate"]["windows"][window]["metrics"][metric])


def stage_latest_window_gate(current_stage: dict[str, Any]) -> dict[str, Any]:
    selected = current_stage["selected_candidate"]
    comparison = selected["comparison"]
    recent_2m = comparison["recent_2m"]
    recent_6m = comparison["recent_6m"]
    tree_depth_value = int(selected["tree_depth"])
    required_tree_depth = int(current_stage.get("required_tree_depth", 2))
    if tree_depth_value < required_tree_depth:
        return {
            "passed": bool(required_tree_depth <= 1),
            "reason": "stage_depth_requirement_failed" if required_tree_depth >= 2 else "depth_below_2",
            "tree_depth": tree_depth_value,
            "required_tree_depth": required_tree_depth,
            "recent_2m": {
                "candidate_total_return": float(selected["windows"]["recent_2m"]["metrics"]["total_return"]),
                "candidate_daily_win_rate": float(selected["windows"]["recent_2m"]["metrics"]["daily_win_rate"]),
                "delta_total_return": float(recent_2m["delta_total_return"]),
                "delta_daily_win_rate": float(recent_2m["delta_daily_win_rate"]),
                "delta_max_drawdown": float(recent_2m["delta_max_drawdown"]),
            },
            "recent_6m": {
                "candidate_total_return": float(selected["windows"]["recent_6m"]["metrics"]["total_return"]),
                "candidate_daily_win_rate": float(selected["windows"]["recent_6m"]["metrics"]["daily_win_rate"]),
                "delta_total_return": float(recent_6m["delta_total_return"]),
                "delta_daily_win_rate": float(recent_6m["delta_daily_win_rate"]),
                "delta_max_drawdown": float(recent_6m["delta_max_drawdown"]),
            },
        }

    latest_pass = bool(
        float(recent_2m["delta_total_return"]) >= 0.0
        and float(recent_2m["delta_daily_win_rate"]) >= 0.0
        and float(recent_2m["delta_max_drawdown"]) >= 0.0
    )
    stability_pass = bool(
        float(recent_6m["delta_total_return"]) >= 0.0
        and float(recent_6m["delta_daily_win_rate"]) >= 0.0
        and float(recent_6m["delta_max_drawdown"]) >= 0.0
    )
    passed = bool(latest_pass and stability_pass)
    if not latest_pass:
        reason = "recent_2m_baseline_gate_failed"
    elif not stability_pass:
        reason = "recent_6m_stability_gate_failed"
    else:
        reason = "latest_window_gate_passed"
    return {
        "passed": passed,
        "reason": reason,
        "tree_depth": tree_depth_value,
        "required_tree_depth": required_tree_depth,
        "recent_2m": {
            "candidate_total_return": float(selected["windows"]["recent_2m"]["metrics"]["total_return"]),
            "candidate_daily_win_rate": float(selected["windows"]["recent_2m"]["metrics"]["daily_win_rate"]),
            "delta_total_return": float(recent_2m["delta_total_return"]),
            "delta_daily_win_rate": float(recent_2m["delta_daily_win_rate"]),
            "delta_max_drawdown": float(recent_2m["delta_max_drawdown"]),
        },
        "recent_6m": {
            "candidate_total_return": float(selected["windows"]["recent_6m"]["metrics"]["total_return"]),
            "candidate_daily_win_rate": float(selected["windows"]["recent_6m"]["metrics"]["daily_win_rate"]),
            "delta_total_return": float(recent_6m["delta_total_return"]),
            "delta_daily_win_rate": float(recent_6m["delta_daily_win_rate"]),
            "delta_max_drawdown": float(recent_6m["delta_max_drawdown"]),
        },
    }


def stage_passes_chain(prev_stage: dict[str, Any] | None, current_stage: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    latest_gate = current_stage.get("latest_window_gate", {})
    if prev_stage is None:
        selected = current_stage["selected_candidate"]
        structural_pass = int(selected["condition_count"]) > 0 and int(selected["tree_depth"]) >= 1
        latest_window_pass = bool(latest_gate.get("passed", False))
        result = {
            "structural_pass": structural_pass,
            "performance_pass": structural_pass,
            "complexity_growth_pass": structural_pass,
            "latest_window_pass": latest_window_pass,
            "previous_complexity": None,
            "current_complexity": list(stage_complexity_signature(current_stage)),
            "latest_window_gate": latest_gate,
        }
        return bool(structural_pass and latest_window_pass), result

    prev_signature = stage_complexity_signature(prev_stage)
    current_signature = stage_complexity_signature(current_stage)
    depth_growth_pass = (
        int(current_stage["selected_candidate"]["tree_depth"]) >= int(prev_stage["selected_candidate"]["tree_depth"])
        and int(current_stage["selected_candidate"]["logic_depth"]) >= int(prev_stage["selected_candidate"]["logic_depth"])
    )
    structural_pass = bool(
        depth_growth_pass
        and (
            int(current_stage["selected_candidate"]["tree_depth"]) > int(prev_stage["selected_candidate"]["tree_depth"])
            or int(current_stage["selected_candidate"]["logic_depth"]) > int(prev_stage["selected_candidate"]["logic_depth"])
        )
    )
    performance_pass = True
    tolerances = {"avg_daily_return": 1e-8, "max_drawdown": 5e-4}
    for window in ("recent_2m", "recent_6m", "full_since_first"):
        prev_avg = stage_window_metric(prev_stage, window, "avg_daily_return")
        curr_avg = stage_window_metric(current_stage, window, "avg_daily_return")
        prev_mdd = abs(stage_window_metric(prev_stage, window, "max_drawdown"))
        curr_mdd = abs(stage_window_metric(current_stage, window, "max_drawdown"))
        prev_win = stage_window_metric(prev_stage, window, "daily_win_rate")
        curr_win = stage_window_metric(current_stage, window, "daily_win_rate")
        if curr_avg + tolerances["avg_daily_return"] < prev_avg:
            performance_pass = False
        if curr_mdd > prev_mdd + tolerances["max_drawdown"]:
            performance_pass = False
        if curr_win + tolerances["avg_daily_return"] < prev_win:
            performance_pass = False
    prev_full_return = float(prev_stage["selected_candidate"]["windows"]["full_since_first"]["metrics"]["total_return"])
    curr_full_return = float(current_stage["selected_candidate"]["windows"]["full_since_first"]["metrics"]["total_return"])
    if curr_full_return + 1e-8 < prev_full_return:
        performance_pass = False
    latest_window_pass = bool(latest_gate.get("passed", False))
    result = {
        "structural_pass": structural_pass,
        "performance_pass": performance_pass,
        "complexity_growth_pass": structural_pass,
        "latest_window_pass": latest_window_pass,
        "previous_complexity": list(prev_signature),
        "current_complexity": list(current_signature),
        "latest_window_gate": latest_gate,
    }
    return structural_pass and performance_pass and latest_window_pass, result


def summarize_fold_core_metrics(fold: dict[str, Any]) -> dict[str, Any]:
    return {
        "fold": fold["fold"],
        "start": fold["start"],
        "end": fold["end"],
        "passed": bool(fold["passed"]),
        "candidate": {
            "avg_daily_return": float(fold["candidate"]["avg_daily_return"]),
            "total_return": float(fold["candidate"]["total_return"]),
            "max_drawdown": float(fold["candidate"]["max_drawdown"]),
            "daily_win_rate": float(fold["candidate"]["daily_win_rate"]),
            "n_trades": int(fold["candidate"]["n_trades"]),
        },
        "baseline": {
            "avg_daily_return": float(fold["baseline"]["avg_daily_return"]),
            "total_return": float(fold["baseline"]["total_return"]),
            "max_drawdown": float(fold["baseline"]["max_drawdown"]),
            "daily_win_rate": float(fold["baseline"]["daily_win_rate"]),
            "n_trades": int(fold["baseline"]["n_trades"]),
        },
        "delta_total_return": float(fold["delta_total_return"]),
        "delta_max_drawdown": float(fold["delta_max_drawdown"]),
        "delta_daily_win_rate": float(fold["delta_daily_win_rate"]),
        "stress_survival_rate": float(fold["stress_survival_rate"]),
    }


def build_walk_forward_gate_report(
    folds_report: list[dict[str, Any]],
    stress_survival_mean_threshold: float,
    stress_survival_min_threshold: float,
) -> dict[str, Any]:
    if not folds_report:
        return {
            "folds": [],
            "fold_pass_rate": 0.0,
            "stress_survival_rate_mean": 0.0,
            "min_fold_stress_survival_rate": 0.0,
            "stress_survival_mean_threshold": float(stress_survival_mean_threshold),
            "stress_survival_min_threshold": float(stress_survival_min_threshold),
            "promotion_gate_passed": False,
            "promotion_gate_reason": "no_walk_forward_folds",
            "wf_1": None,
        }

    fold_pass_rate = float(sum(1 for fold in folds_report if fold["passed"]) / len(folds_report))
    stress_survival_rates = [float(fold["stress_survival_rate"]) for fold in folds_report]
    stress_survival_rate_mean = float(sum(stress_survival_rates) / len(stress_survival_rates))
    min_fold_stress_survival_rate = float(min(stress_survival_rates))
    wf_1_fold = next((fold for fold in folds_report if fold["fold"] == "wf_1"), folds_report[-1])
    wf_1_summary = summarize_fold_core_metrics(wf_1_fold)
    promotion_gate_passed = bool(
        all(fold["passed"] for fold in folds_report)
        and stress_survival_rate_mean >= float(stress_survival_mean_threshold)
        and min_fold_stress_survival_rate >= float(stress_survival_min_threshold)
        and float(wf_1_fold["delta_total_return"]) >= 0.0
        and float(wf_1_fold["delta_daily_win_rate"]) >= 0.0
    )
    if float(wf_1_fold["delta_total_return"]) < 0.0 or float(wf_1_fold["delta_daily_win_rate"]) < 0.0:
        promotion_gate_reason = "wf_1_latest_gate_failed"
    elif not all(fold["passed"] for fold in folds_report):
        promotion_gate_reason = "walk_forward_fold_gate_failed"
    elif stress_survival_rate_mean < float(stress_survival_mean_threshold):
        promotion_gate_reason = "stress_survival_mean_below_threshold"
    elif min_fold_stress_survival_rate < float(stress_survival_min_threshold):
        promotion_gate_reason = "stress_survival_min_below_threshold"
    else:
        promotion_gate_reason = "promotion_gate_passed"

    return {
        "folds": folds_report,
        "fold_pass_rate": fold_pass_rate,
        "stress_survival_rate_mean": stress_survival_rate_mean,
        "min_fold_stress_survival_rate": min_fold_stress_survival_rate,
        "stress_survival_mean_threshold": float(stress_survival_mean_threshold),
        "stress_survival_min_threshold": float(stress_survival_min_threshold),
        "promotion_gate_passed": promotion_gate_passed,
        "promotion_gate_reason": promotion_gate_reason,
        "wf_1": wf_1_summary,
        "wf_1_candidate_total_return": float(wf_1_fold["candidate"]["total_return"]),
        "wf_1_candidate_daily_win_rate": float(wf_1_fold["candidate"]["daily_win_rate"]),
        "wf_1_baseline_total_return": float(wf_1_fold["baseline"]["total_return"]),
        "wf_1_baseline_daily_win_rate": float(wf_1_fold["baseline"]["daily_win_rate"]),
        "wf_1_delta_total_return": float(wf_1_fold["delta_total_return"]),
        "wf_1_delta_daily_win_rate": float(wf_1_fold["delta_daily_win_rate"]),
    }


def evaluate_tree_on_windows(
    tree: Any,
    window_cache: dict[str, dict[str, Any]],
    expert_pool: list[dict[str, Any]],
    route_thresholds: tuple[float, ...],
    library_lookup: dict[str, Any],
    baseline_runtime: dict[str, np.ndarray],
    max_depth: int,
    logic_max_depth: int,
) -> dict[str, Any]:
    filter_decision = semantic_filter(tree, expert_pool, max_depth, "heuristic")
    reference_features = next(iter(window_cache.values()))["features"]
    _, leaf_catalog = evaluate_tree_leaf_codes(tree, reference_features)
    leaf_runtime = build_leaf_runtime_arrays_for_pair(BTC_PAIR, leaf_catalog, expert_pool, route_thresholds, len(library_lookup["signal_pos"]))
    windows: dict[str, Any] = {}
    for label, window_state in window_cache.items():
        leaf_codes, _ = evaluate_tree_leaf_codes(tree, window_state["features"])
        window_metrics = fast_fractal_replay_from_context(
            window_state["fast_context"],
            library_lookup,
            route_thresholds,
            leaf_runtime,
            leaf_codes,
        )
        windows[label] = {
            "start": window_state["start"],
            "end": window_state["end"],
            "bars": window_state["bars"],
            "metrics": window_metrics,
        }

    score = 0.0
    baseline_windows: dict[str, Any] = {}
    for label, window_state in window_cache.items():
        baseline_codes = np.zeros(len(next(iter(window_state["features"].values()))), dtype="int16")
        baseline_metrics = fast_fractal_replay_from_context(
            window_state["fast_context"],
            library_lookup,
            route_thresholds,
            baseline_runtime,
            baseline_codes,
        )
        baseline_windows[label] = baseline_metrics

    comparison = {
        label: {
            "delta_avg_daily_return": float(windows[label]["metrics"]["avg_daily_return"] - baseline_windows[label]["avg_daily_return"]),
            "delta_total_return": float(windows[label]["metrics"]["total_return"] - baseline_windows[label]["total_return"]),
            "delta_max_drawdown": float(windows[label]["metrics"]["max_drawdown"] - baseline_windows[label]["max_drawdown"]),
            "delta_daily_win_rate": float(windows[label]["metrics"]["daily_win_rate"] - baseline_windows[label]["daily_win_rate"]),
        }
        for label in windows
    }

    score += comparison["recent_2m"]["delta_avg_daily_return"] * 420000.0
    score += comparison["recent_6m"]["delta_avg_daily_return"] * 320000.0
    score += comparison["full_since_first"]["delta_avg_daily_return"] * 260000.0
    score += comparison["recent_2m"]["delta_total_return"] * 800.0
    score += comparison["recent_6m"]["delta_total_return"] * 450.0
    score += comparison["full_since_first"]["delta_total_return"] * 40.0
    score += (abs(baseline_windows["recent_2m"]["max_drawdown"]) - abs(windows["recent_2m"]["metrics"]["max_drawdown"])) * 20000.0
    score += (abs(baseline_windows["recent_6m"]["max_drawdown"]) - abs(windows["recent_6m"]["metrics"]["max_drawdown"])) * 15000.0
    score += (abs(baseline_windows["full_since_first"]["max_drawdown"]) - abs(windows["full_since_first"]["metrics"]["max_drawdown"])) * 11000.0
    score += (windows["recent_2m"]["metrics"]["daily_win_rate"] - baseline_windows["recent_2m"]["daily_win_rate"]) * 18000.0
    score += (windows["recent_6m"]["metrics"]["daily_win_rate"] - baseline_windows["recent_6m"]["daily_win_rate"]) * 14000.0
    score += (windows["full_since_first"]["metrics"]["daily_win_rate"] - baseline_windows["full_since_first"]["daily_win_rate"]) * 10000.0
    for label in windows:
        trade_ratio = float(windows[label]["metrics"]["n_trades"]) / max(float(baseline_windows[label]["n_trades"]), 1.0)
        if trade_ratio < 0.05:
            score -= (0.05 - trade_ratio) * 150000.0
    score -= tree_size(tree) * 120.0
    score -= max(0, tree_depth(tree) - max_depth) * 180.0
    score -= tree_logic_size(tree) * 45.0
    score -= max(0, tree_logic_depth(tree) - logic_max_depth) * 90.0
    if not filter_decision.accepted:
        score -= 10_000_000.0

    structural_score = structural_bonus_from_metrics(
        int(tree_depth(tree)),
        int(tree_logic_depth(tree)),
        len(set(collect_leaf_keys(tree))),
        int(len(collect_specs(tree))),
    )
    performance_score = float(score)
    search_fitness = float(score) + 30.0 * float(structural_score)

    return {
        "tree": tree,
        "tree_key": tree_key(tree),
        "tree_serialized": serialize_tree(tree),
        "tree_depth": int(tree_depth(tree)),
        "tree_size": int(tree_size(tree)),
        "logic_depth": int(tree_logic_depth(tree)),
        "logic_size": int(tree_logic_size(tree)),
        "condition_count": int(len(collect_specs(tree))),
        "leaf_cardinality": int(len(set(collect_leaf_keys(tree)))),
        "filter": {
            "accepted": filter_decision.accepted,
            "source": filter_decision.source,
            "reason": filter_decision.reason,
        },
        "windows": windows,
        "baseline": baseline_windows,
        "comparison": comparison,
        "fitness": float(score),
        "structural_score": float(structural_score),
        "search_fitness": float(search_fitness),
        "performance_score": float(performance_score),
    }


def run_stage(
    stage_name: str,
    max_depth: int,
    logic_max_depth: int,
    min_tree_depth: int,
    df: pd.DataFrame,
    window_ranges: tuple[tuple[str, pd.Timestamp, pd.Timestamp], ...],
    expert_pool: list[dict[str, Any]],
    route_thresholds: tuple[float, ...],
    library_lookup: dict[str, Any],
    baseline_runtime: dict[str, np.ndarray],
    compiled_model: Any,
    funding_df: pd.DataFrame,
    population: int,
    generations: int,
    elite_count: int,
    top_k: int,
    seed: int,
    previous_best: Any | None = None,
    external_seed_trees: tuple[Any, ...] = (),
) -> tuple[dict[str, Any], Any]:
    rng = random.Random(seed)
    if isinstance(previous_best, dict):
        previous_best = deserialize_tree(previous_best)
    condition_options = build_condition_options()
    stage_seed_trees = [
        tree
        for tree in build_seed_trees(expert_pool, condition_options, (BTC_PAIR,))
        if tree_depth(tree) <= max_depth and tree_logic_depth(tree) <= logic_max_depth
    ]
    population_nodes: list[Any] = [copy.deepcopy(tree) for tree in stage_seed_trees]
    for tree in external_seed_trees:
        if tree_depth(tree) <= max_depth and tree_logic_depth(tree) <= logic_max_depth:
            population_nodes.append(copy.deepcopy(tree))
    if previous_best is not None:
        population_nodes.append(copy.deepcopy(previous_best))
        while len(population_nodes) < max(3, min(population, len(stage_seed_trees) + len(external_seed_trees) + 3)):
            population_nodes.append(
                mutate_tree(
                    previous_best,
                    rng,
                    condition_options,
                    len(expert_pool),
                    max_depth,
                    logic_max_depth=logic_max_depth,
                )
            )
    while len(population_nodes) < population:
        population_nodes.append(
            random_tree(
                rng,
                condition_options,
                len(expert_pool),
                max_depth,
                logic_max_depth=logic_max_depth,
            )
        )

    window_cache: dict[str, dict[str, Any]] = {}
    for label, start, end in window_ranges:
        window_df = df.loc[start:end].copy()
        if window_df.empty:
            raise RuntimeError(f"Window {label} is empty for BTC backtest.")
        feature_arrays = materialize_feature_arrays(build_btc_feature_frame(window_df), pd.DatetimeIndex(window_df.index))
        validate_feature_health(feature_arrays)
        raw_signal = pd.Series(
            compiled_model(*gp.get_feature_arrays(window_df, BTC_PAIR)),
            index=window_df.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        overlay_inputs = build_btc_overlay_inputs(window_df)
        funding_slice = funding_df[
            (funding_df["fundingTime"] >= ensure_utc_timestamp(start))
            & (funding_df["fundingTime"] <= ensure_utc_timestamp(end) + pd.Timedelta(days=1))
        ].copy()
        window_cache[label] = {
            "start": ensure_utc_timestamp(start),
            "end": ensure_utc_timestamp(end),
            "bars": int(len(window_df)),
            "features": feature_arrays,
            "fast_context": build_fast_context(
                df=window_df,
                pair=BTC_PAIR,
                raw_signal=raw_signal,
                overlay_inputs=overlay_inputs,
                route_thresholds=route_thresholds,
                library_lookup=library_lookup,
                funding_df=funding_slice,
            ),
        }

    fast_cache: dict[str, dict[str, Any]] = {}

    def evaluate_tree(tree: Any) -> dict[str, Any]:
        if isinstance(tree, dict):
            tree = deserialize_tree(tree)
        key = tree_key(tree)
        cached = fast_cache.get(key)
        if cached is not None:
            return cached
        evaluated = evaluate_tree_on_windows(
            tree,
            window_cache,
            expert_pool,
            route_thresholds,
            library_lookup,
            baseline_runtime,
            max_depth,
            logic_max_depth,
        )
        fast_cache[key] = evaluated
        return evaluated

    search_started = pd.Timestamp.utcnow()
    for _ in range(generations):
        evaluated = [evaluate_tree(tree) for tree in population_nodes]
        evaluated.sort(key=lambda item: item["search_fitness"], reverse=True)
        elites = [copy.deepcopy(item["tree"]) for item in evaluated[:elite_count]]
        next_population = elites[:]
        while len(next_population) < population:
            parent_a = tournament_select(evaluated, rng)
            if rng.random() < 0.65:
                parent_b = tournament_select(evaluated, rng)
                child_a, child_b = crossover_tree(parent_a, parent_b, rng)
                candidate = child_a if rng.random() < 0.5 else child_b
            else:
                candidate = parent_a
            if rng.random() < 0.70:
                candidate = mutate_tree(
                    candidate,
                    rng,
                    condition_options,
                    len(expert_pool),
                    max_depth,
                    logic_max_depth=logic_max_depth,
                )
            next_population.append(candidate)
        population_nodes = next_population[:population]

    evaluated = [evaluate_tree(tree) for tree in population_nodes]
    ranked = sorted({tree_key(item["tree"]): item for item in evaluated}.values(), key=lambda item: item["search_fitness"], reverse=True)
    ranked = [
        item
        for item in ranked
        if int(item["tree_depth"]) <= max_depth and int(item["logic_depth"]) <= logic_max_depth
    ]
    external_seed_keys = {tree_key(tree) for tree in external_seed_trees}
    stage_depth_candidates = [item for item in ranked if int(item["tree_depth"]) >= int(min_tree_depth)]
    candidate_pool = stage_depth_candidates or ranked
    top_candidates = candidate_pool[:top_k]
    seeded_stage_candidates = [
        item for item in candidate_pool if item["tree_key"] in external_seed_keys
    ]
    latest_gate_candidates = [
        item
        for item in top_candidates
        if stage_latest_window_gate(
            {
                "required_tree_depth": min_tree_depth,
                "selected_candidate": {
                    **item,
                    "windows": item["windows"],
                    "comparison": item["comparison"],
                }
            }
        )["passed"]
    ]
    seeded_latest_gate_candidates = [
        item
        for item in seeded_stage_candidates
        if stage_latest_window_gate(
            {
                "required_tree_depth": min_tree_depth,
                "selected_candidate": {
                    **item,
                    "windows": item["windows"],
                    "comparison": item["comparison"],
                }
            }
        )["passed"]
    ]
    selection_pool = seeded_latest_gate_candidates or latest_gate_candidates or top_candidates
    selected, selection_diagnostics = select_near_frontier_structural_winner(selection_pool)
    assert selected is not None, f"{stage_name} should yield at least one candidate"
    assert selected["tree_depth"] <= max_depth, "selected tree exceeds requested max depth"
    assert selected["logic_depth"] <= logic_max_depth, "selected tree exceeds requested logic depth"

    stage_report = {
        "stage": stage_name,
        "max_depth": int(max_depth),
        "logic_max_depth": int(logic_max_depth),
        "population": int(population),
        "generations": int(generations),
        "elite_count": int(elite_count),
        "top_candidates": [
            {
                "tree": item["tree_serialized"],
                "tree_depth": item["tree_depth"],
                "tree_size": item["tree_size"],
                "logic_depth": item["logic_depth"],
                "logic_size": item["logic_size"],
                "condition_count": item["condition_count"],
                "leaf_cardinality": item["leaf_cardinality"],
                "fitness": item["fitness"],
                "search_fitness": item["search_fitness"],
                "performance_score": item["performance_score"],
                "structural_score": item["structural_score"],
                "filter": item["filter"],
                "windows": item["windows"],
                "comparison": item["comparison"],
            }
            for item in top_candidates
        ],
        "selected_candidate": {
            "tree": selected["tree_serialized"],
            "tree_depth": selected["tree_depth"],
            "tree_size": selected["tree_size"],
            "logic_depth": selected["logic_depth"],
            "logic_size": selected["logic_size"],
            "condition_count": selected["condition_count"],
            "leaf_cardinality": selected["leaf_cardinality"],
            "fitness": selected["fitness"],
            "search_fitness": selected["search_fitness"],
            "performance_score": selected["performance_score"],
            "structural_score": selected["structural_score"],
            "filter": selected["filter"],
            "windows": selected["windows"],
            "comparison": selected["comparison"],
        },
        "latest_window_gate": stage_latest_window_gate(
            {
                "required_tree_depth": min_tree_depth,
                "selected_candidate": {
                    **selected,
                    "windows": selected["windows"],
                    "comparison": selected["comparison"],
                }
            }
        ),
        "window_contract": {
            label: {
                "start": window_cache[label]["start"],
                "end": window_cache[label]["end"],
                "bars": int(window_cache[label]["bars"]),
            }
            for label in window_cache
        },
        "runtime": {
            "evaluated_unique_candidates": len(fast_cache),
            "search_started_at": search_started.isoformat(),
        },
        "selection_pool": {
            "top_candidate_count": len(top_candidates),
            "stage_depth_candidate_count": len(stage_depth_candidates),
            "seeded_stage_candidate_count": len(seeded_stage_candidates),
            "seeded_latest_window_gate_candidate_count": len(seeded_latest_gate_candidates),
            "latest_window_gate_candidate_count": len(latest_gate_candidates),
        },
        "selection_diagnostics": selection_diagnostics,
    }
    return stage_report, selected["tree"]


def run_curriculum_backtest(
    pairs: str,
    summary_out: str,
    command_log: str,
    depth_curriculum: str,
    logic_curriculum: str,
    expert_summaries: str,
    expert_pool_size: int,
    population: int,
    generations: int,
    elite_count: int,
    top_k: int,
    seed: int,
    route_thresholds: str,
    model: str,
    funding_cache: str,
    baseline_summary: str,
    search_seed_summary: str,
    walk_forward_folds: int,
    walk_forward_test_months: int,
    commission_stress: str,
    stress_survival_threshold: float,
    stress_survival_mean_threshold: float,
    stress_survival_min_threshold: float,
) -> dict[str, Any]:
    if pairs != BTC_PAIR:
        raise ValueError("BTC-only verifier only accepts --pairs BTCUSDT")

    summary_path = Path(summary_out)
    command_path = Path(command_log)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.parent.mkdir(parents=True, exist_ok=True)

    depth_values = parse_csv_tuple(depth_curriculum, int)
    logic_values = parse_csv_tuple(logic_curriculum, int)
    if len(depth_values) != len(logic_values):
        raise ValueError("--depth-curriculum and --logic-curriculum must have the same length")

    expert_summary_paths = [part.strip() for part in expert_summaries.split(",") if part.strip()]
    route_threshold_values = parse_csv_tuple(route_thresholds, float)

    btc_csv = gp.DATA_DIR / "BTCUSDT_5m.csv"
    first_ts, last_ts = discover_btc_bounds(btc_csv)
    df = gp.load_all_pairs(pairs=[BTC_PAIR], start=first_ts.isoformat(), end=last_ts.isoformat(), refresh_cache=False)
    assert not df.empty, "BTC dataframe must not be empty"
    assert ensure_utc_timestamp(df.index[0]) == first_ts, "loaded BTC data start must match first collected bar"
    assert ensure_utc_timestamp(df.index[-1]) == last_ts, "loaded BTC data end must match latest collected bar"

    model_tree, _ = load_model(Path(model))
    compiled_model = gp.toolbox.compile(expr=model_tree)
    funding_df = load_cached_funding(Path(funding_cache))
    library = list(iter_params())
    library_lookup = build_library_lookup(library)
    baseline_summary_obj = json.loads(Path(baseline_summary).read_text())
    baseline_runtime = build_baseline_leaf_runtime(baseline_summary_obj, route_threshold_values, len(library))
    expert_pool = build_btc_expert_pool(expert_summary_paths, expert_pool_size)
    external_seed_trees = tuple(load_seed_trees_from_summary(search_seed_summary))
    window_ranges = build_window_ranges(first_ts, last_ts)

    stages: list[dict[str, Any]] = []
    previous_best = None
    for idx, (max_depth, logic_max_depth) in enumerate(zip(depth_values, logic_values, strict=True), start=1):
        stage_report, previous_best = run_stage(
            stage_name=f"depth_{max_depth}",
            max_depth=max_depth,
            logic_max_depth=logic_max_depth,
            min_tree_depth=max_depth,
            df=df,
            window_ranges=window_ranges,
            expert_pool=expert_pool,
            route_thresholds=route_threshold_values,
            library_lookup=library_lookup,
            baseline_runtime=baseline_runtime,
            compiled_model=compiled_model,
            funding_df=funding_df,
            population=population,
            generations=generations,
            elite_count=elite_count,
            top_k=top_k,
            seed=seed + idx - 1,
            previous_best=previous_best,
            external_seed_trees=external_seed_trees,
        )
        stages.append(stage_report)

    selected_stage: dict[str, Any] | None = None
    selected_index: int | None = None
    stage_checks: list[dict[str, Any]] = []
    prev_success_stage: dict[str, Any] | None = None
    for idx, stage in enumerate(stages):
        stage_pass, check = stage_passes_chain(prev_success_stage, stage)
        check["stage"] = stage["stage"]
        check["passed"] = stage_pass
        stage["checks"] = check
        stage_checks.append(check)
        if stage_pass:
            selected_stage = stage
            selected_index = idx
            prev_success_stage = stage

    selected_candidate = None if selected_stage is None else selected_stage["selected_candidate"]
    window_contract = {
        "recent_2m_end_matches_latest_data": bool(stages[-1]["window_contract"]["recent_2m"]["end"] == last_ts),
        "recent_6m_end_matches_latest_data": bool(stages[-1]["window_contract"]["recent_6m"]["end"] == last_ts),
        "full_start_matches_first_data": bool(stages[-1]["window_contract"]["full_since_first"]["start"] == first_ts),
        "full_end_matches_latest_data": bool(stages[-1]["window_contract"]["full_since_first"]["end"] == last_ts),
    }
    assert all(window_contract.values()), "BTC window contract failed"

    overall = {
        "stage_count": len(stages),
        "best_stage": None if selected_stage is None else selected_stage["stage"],
        "selected_candidate_present": selected_candidate is not None,
        "selected_candidate": selected_candidate,
        "curriculum_passed": bool(selected_index is not None and selected_index == len(stages) - 1),
        "stage_checks": stage_checks,
    }

    walk_forward_report = None
    if selected_candidate is not None:
        walk_forward_ranges = build_walk_forward_ranges(first_ts, last_ts, walk_forward_folds, walk_forward_test_months)
        stress_values = parse_csv_tuple(commission_stress, float)
        selected_tree = deserialize_tree(selected_candidate["tree"])
        reference_features = materialize_feature_arrays(build_btc_feature_frame(df), pd.DatetimeIndex(df.index))
        _, selected_leaf_catalog = evaluate_tree_leaf_codes(selected_tree, reference_features)
        selected_leaf_runtime = build_leaf_runtime_arrays_for_pair(
            BTC_PAIR,
            selected_leaf_catalog,
            expert_pool,
            route_threshold_values,
            len(library),
        )
        folds_report: list[dict[str, Any]] = []
        for fold_name, fold_start, fold_end in walk_forward_ranges:
            fold_df = df.loc[fold_start:fold_end].copy()
            if fold_df.empty:
                continue
            fold_features = materialize_feature_arrays(build_btc_feature_frame(fold_df), pd.DatetimeIndex(fold_df.index))
            selected_codes, _ = evaluate_tree_leaf_codes(selected_tree, fold_features)
            baseline_codes = np.zeros(len(next(iter(fold_features.values()))), dtype="int16")
            fold_signal = pd.Series(
                compiled_model(*gp.get_feature_arrays(fold_df, BTC_PAIR)),
                index=fold_df.index,
                dtype="float64",
            ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            fold_overlay_inputs = build_btc_overlay_inputs(fold_df)
            fold_funding = funding_df[
                (funding_df["fundingTime"] >= ensure_utc_timestamp(fold_start))
                & (funding_df["fundingTime"] <= ensure_utc_timestamp(fold_end) + pd.Timedelta(days=1))
            ].copy()
            fold_context = build_fast_context(
                df=fold_df,
                pair=BTC_PAIR,
                raw_signal=fold_signal,
                overlay_inputs=fold_overlay_inputs,
                route_thresholds=route_threshold_values,
                library_lookup=library_lookup,
                funding_df=fold_funding,
            )
            base_metrics = fast_fractal_replay_from_context(
                fold_context,
                library_lookup,
                route_threshold_values,
                baseline_runtime,
                baseline_codes,
            )
            candidate_metrics = fast_fractal_replay_from_context(
                fold_context,
                library_lookup,
                route_threshold_values,
                selected_leaf_runtime,
                selected_codes,
            )
            stress_metrics = []
            for multiplier in stress_values:
                stressed = fast_fractal_replay_from_context(
                    fold_context,
                    library_lookup,
                    route_threshold_values,
                    selected_leaf_runtime,
                    selected_codes,
                    commission_multiplier=multiplier,
                )
                stress_metrics.append(
                    {
                        "commission_multiplier": float(multiplier),
                        "metrics": stressed,
                    }
                )
            delta_return = float(candidate_metrics["total_return"] - base_metrics["total_return"])
            delta_mdd = float(abs(base_metrics["max_drawdown"]) - abs(candidate_metrics["max_drawdown"]))
            delta_win_rate = float(candidate_metrics["daily_win_rate"] - base_metrics["daily_win_rate"])
            fold_pass = bool(delta_return >= 0.0 and delta_mdd >= 0.0 and delta_win_rate >= 0.0)
            stress_survival_rate = float(
                sum(
                    1
                    for item in stress_metrics
                    if (
                        float(item["metrics"]["total_return"]) >= float(base_metrics["total_return"])
                        and abs(float(item["metrics"]["max_drawdown"])) <= abs(float(base_metrics["max_drawdown"])) + 1e-12
                        and float(item["metrics"]["daily_win_rate"]) >= float(base_metrics["daily_win_rate"]) - 1e-12
                    )
                )
                / max(len(stress_metrics), 1)
            )
            folds_report.append(
                {
                    "fold": fold_name,
                    "start": ensure_utc_timestamp(fold_start),
                    "end": ensure_utc_timestamp(fold_end),
                    "candidate": candidate_metrics,
                    "baseline": base_metrics,
                    "delta_total_return": delta_return,
                    "delta_max_drawdown": delta_mdd,
                    "delta_daily_win_rate": delta_win_rate,
                    "passed": fold_pass,
                    "stress_survival_rate": stress_survival_rate,
                    "stress": stress_metrics,
                }
            )
        walk_forward_report = build_walk_forward_gate_report(
            folds_report,
            stress_survival_mean_threshold,
            stress_survival_min_threshold,
        )
        assert walk_forward_report["wf_1"] is not None, "walk-forward summary must expose wf_1 core metrics"
        assert "min_fold_stress_survival_rate" in walk_forward_report, "walk-forward summary must expose min fold stress survival"
        overall["walk_forward"] = walk_forward_report

    report = {
        "pair": BTC_PAIR,
        "data_bounds": {
            "first": first_ts,
            "last": last_ts,
        },
        "window_contract": window_contract,
        "curriculum": {
            "depths": list(depth_values),
            "logic_depths": list(logic_values),
            "population": population,
            "generations": generations,
            "elite_count": elite_count,
            "top_k": top_k,
            "seed": seed,
            "expert_pool_size": len(expert_pool),
            "route_thresholds": list(route_threshold_values),
            "stress_survival_threshold": float(stress_survival_threshold),
        },
        "stages": stages,
        "selection": {
            "reason": "latest_monotonic_stage_pass" if selected_stage is not None else "no_stage_pass",
            "stage": None if selected_stage is None else selected_stage["stage"],
        },
        "overall": overall,
        "walk_forward": walk_forward_report,
        "wf_1_core_metrics": None if walk_forward_report is None else walk_forward_report.get("wf_1"),
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }

    command_log_obj = {
        "command": [
            sys.executable,
            str(Path(__file__).resolve()),
            "--pairs",
            BTC_PAIR,
            "--depth-curriculum",
            depth_curriculum,
            "--logic-curriculum",
            logic_curriculum,
            "--population",
            str(population),
            "--generations",
            str(generations),
            "--elite-count",
            str(elite_count),
            "--top-k",
            str(top_k),
            "--seed",
            str(seed),
        ],
        "summary_path": str(summary_path),
    }
    command_path.write_text(json.dumps(json_safe(command_log_obj), ensure_ascii=False, indent=2) + "\n")
    summary_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe({"pair": BTC_PAIR, "selection": report["selection"], "overall": report["overall"]}), ensure_ascii=False, indent=2))
    return report


def main() -> None:
    args = parse_args()
    run_curriculum_backtest(
        pairs=args.pairs,
        summary_out=args.summary_out,
        command_log=args.command_log,
        depth_curriculum=args.depth_curriculum,
        logic_curriculum=args.logic_curriculum,
        expert_summaries=args.expert_summaries,
        expert_pool_size=args.expert_pool_size,
        population=args.population,
        generations=args.generations,
        elite_count=args.elite_count,
        top_k=args.top_k,
        seed=args.seed,
        route_thresholds=args.route_thresholds,
        model=args.model,
        funding_cache=args.funding_cache,
        baseline_summary=args.baseline_summary,
        search_seed_summary=args.search_seed_summary,
        walk_forward_folds=args.walk_forward_folds,
        walk_forward_test_months=args.walk_forward_test_months,
        commission_stress=args.commission_stress,
        stress_survival_threshold=args.stress_survival_threshold,
        stress_survival_mean_threshold=args.stress_survival_mean_threshold,
        stress_survival_min_threshold=args.stress_survival_min_threshold,
    )


if __name__ == "__main__":
    main()
