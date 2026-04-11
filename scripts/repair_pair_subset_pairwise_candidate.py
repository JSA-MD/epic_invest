#!/usr/bin/env python3
"""Local repair search for pair-specific full-grid BTC/BNB overlay candidates."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from pairwise_validation_engine import (
    build_candidate_validation_bundle,
    build_return_frame,
    build_validation_robustness_profile,
)
from replay_regime_mixture_realistic import load_model
from run_pair_subset_pairwise_stress import build_candidate_metrics as build_stress_candidate_metrics
from run_pair_subset_stress_matrix import SCENARIOS
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    fast_overlay_replay_from_context,
    json_safe,
    load_or_fetch_funding,
    normalize_mapping_indices,
    normalize_route_state_mode,
    parse_csv_tuple,
    realistic_overlay_replay_from_context,
    route_state_count,
    summarize_single_result,
)
from validate_pair_subset_summary import build_validation_bundle


UTC = timezone.utc
TARGET_060_DAILY_RETURN = 0.006
PAIRWISE_PARETO_OBJECTIVES: tuple[tuple[str, bool], ...] = (
    ("market_os_fitness", True),
    ("validation_quality_score", True),
    ("recent_2m_worst_pair_avg_daily_return", True),
    ("recent_6m_worst_pair_avg_daily_return", True),
    ("full_4y_worst_pair_avg_daily_return", True),
    ("bnb_full_4y_avg_daily_return", True),
    ("corr_state_robustness", True),
    ("target_060_shortfall", False),
    ("bnb_full_4y_target_shortfall", False),
    ("full_4y_worst_max_drawdown_abs", False),
    ("false_positive_risk", False),
    ("turnover_cost", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair promising pair-specific full-grid candidates with local search.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--candidate-summaries",
        required=True,
        help="Comma-separated search summary files to mine starting candidates from.",
    )
    parser.add_argument(
        "--baseline-summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_nsga3_summary.json"),
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_candidate_summary.json"),
    )
    parser.add_argument("--random-mutations", type=int, default=6000)
    parser.add_argument("--top-realistic", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--start-candidate-count", type=int, default=6)
    parser.add_argument(
        "--route-state-mode",
        choices=("base", "equity_corr"),
        default="equity_corr",
    )
    parser.add_argument("--cpcv-blocks", type=int, default=6)
    parser.add_argument("--cpcv-test-blocks", type=int, default=2)
    parser.add_argument("--cpcv-embargo-days", type=int, default=2)
    parser.add_argument("--validation-candidate-count", type=int, default=18)
    parser.add_argument("--stress-proxy-candidate-count", type=int, default=6)
    parser.add_argument("--target-daily-return", type=float, default=TARGET_060_DAILY_RETURN)
    return parser.parse_args()


def normalize_pair_config(config: dict[str, Any], route_state_mode: str) -> dict[str, Any]:
    requested_mode = normalize_route_state_mode(route_state_mode)
    return {
        "route_breadth_threshold": float(config["route_breadth_threshold"]),
        "mapping_indices": list(normalize_mapping_indices(config["mapping_indices"], requested_mode)),
        "route_state_mode": requested_mode,
    }


def baseline_pair_configs(
    pairs: tuple[str, ...],
    baseline_summary: dict[str, Any],
    route_state_mode: str,
) -> dict[str, dict[str, Any]]:
    selected = baseline_summary["selected_candidate"]
    base_config = normalize_pair_config(selected, route_state_mode)
    return {
        pair: {
            "route_breadth_threshold": float(base_config["route_breadth_threshold"]),
            "mapping_indices": list(base_config["mapping_indices"]),
            "route_state_mode": str(base_config["route_state_mode"]),
        }
        for pair in pairs
    }


def candidate_key(candidate: dict[str, Any], pairs: tuple[str, ...]) -> tuple[Any, ...]:
    key: list[Any] = []
    for pair in pairs:
        cfg = candidate["pair_configs"][pair]
        key.append(float(cfg["route_breadth_threshold"]))
        key.append(str(cfg.get("route_state_mode", "base")))
        key.extend(int(v) for v in cfg["mapping_indices"])
    return tuple(key)


def clone_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(candidate))


def candidate_id(candidate: dict[str, Any], pairs: tuple[str, ...]) -> str:
    return "|".join(str(part) for part in candidate_key(candidate, pairs))


def replace_corr_block(mapping: list[int], block_index: int, block_values: list[int] | tuple[int, ...]) -> list[int]:
    block_size = 4
    start = int(block_index) * block_size
    updated = list(mapping)
    updated[start:start + block_size] = [int(v) for v in block_values]
    return updated


def replace_base_state_across_blocks(mapping: list[int], base_slot: int, value: int, route_gene_count: int) -> list[int]:
    updated = list(mapping)
    for offset in range(int(base_slot), int(route_gene_count), 4):
        updated[offset] = int(value)
    return updated


def target_shortfall(value: float, target_daily_return: float) -> float:
    return max(0.0, float(target_daily_return) - float(value))


def unwrap_window_aggregate(window_block: dict[str, Any]) -> dict[str, Any]:
    aggregate = window_block.get("aggregate") if isinstance(window_block, dict) else None
    return aggregate if isinstance(aggregate, dict) else window_block


def pair_metric_or_default(
    windows: dict[str, Any],
    *,
    label: str,
    pair: str,
    metric: str,
    default_metric: str,
) -> float:
    window_block = windows[label]
    per_pair = window_block.get("per_pair") if isinstance(window_block, dict) else None
    if isinstance(per_pair, dict) and pair in per_pair and metric in per_pair[pair]:
        return float(per_pair[pair][metric])
    aggregate = unwrap_window_aggregate(window_block)
    return float(aggregate[default_metric])


def aggregate_target_shortfall(
    windows: dict[str, Any],
    *,
    target_daily_return: float,
) -> float:
    recent_2m = target_shortfall(
        float(unwrap_window_aggregate(windows["recent_2m"])["worst_pair_avg_daily_return"]),
        target_daily_return,
    )
    recent_6m = target_shortfall(
        float(unwrap_window_aggregate(windows["recent_6m"])["worst_pair_avg_daily_return"]),
        target_daily_return,
    )
    full_4y = target_shortfall(
        float(unwrap_window_aggregate(windows["full_4y"])["worst_pair_avg_daily_return"]),
        target_daily_return,
    )
    return float(recent_2m * 1.0 + recent_6m * 1.3 + full_4y * 1.7)


def bnb_full_4y_target_shortfall(
    windows: dict[str, Any],
    *,
    target_daily_return: float,
) -> float:
    bnb_full_4y = pair_metric_or_default(
        windows,
        label="full_4y",
        pair="BNBUSDT",
        metric="avg_daily_return",
        default_metric="worst_pair_avg_daily_return",
    )
    return target_shortfall(bnb_full_4y, target_daily_return)


def fast_stress_proxy_reserve(windows: dict[str, Any]) -> float:
    recent_2m = float(unwrap_window_aggregate(windows["recent_2m"])["worst_pair_avg_daily_return"])
    recent_6m = float(unwrap_window_aggregate(windows["recent_6m"])["worst_pair_avg_daily_return"])
    full_4y = float(unwrap_window_aggregate(windows["full_4y"])["worst_pair_avg_daily_return"])
    bnb_full_4y = pair_metric_or_default(
        windows,
        label="full_4y",
        pair="BNBUSDT",
        metric="avg_daily_return",
        default_metric="worst_pair_avg_daily_return",
    )
    return float(min(recent_2m, recent_6m, full_4y, bnb_full_4y) - TARGET_060_DAILY_RETURN)


def fast_validation_robustness_proxy(
    windows: dict[str, Any],
    *,
    target_daily_return: float,
    pair_count: int | None = None,
) -> float:
    labels = ("recent_2m", "recent_6m", "full_4y")
    inferred_pair_count = int(pair_count or 0)
    min_target_hit = 1.0
    min_win_rate = 1.0
    min_sharpe = float("inf")
    positive_ratios: list[float] = []
    mean_trades: list[float] = []
    max_dispersion = 0.0
    max_worst_day_abs = 0.0
    max_drawdown_excess = 0.0

    for label in labels:
        window_block = windows[label]
        aggregate = unwrap_window_aggregate(window_block)
        per_pair = window_block.get("per_pair") if isinstance(window_block, dict) else None
        if isinstance(per_pair, dict) and per_pair:
            inferred_pair_count = max(inferred_pair_count, len(per_pair))
            for metrics in per_pair.values():
                min_target_hit = min(min_target_hit, float(metrics.get("daily_target_hit_rate", 0.0)))
                min_win_rate = min(min_win_rate, float(metrics.get("daily_win_rate", 0.0)))
                min_sharpe = min(min_sharpe, float(metrics.get("sharpe", 0.0)))
                max_worst_day_abs = max(max_worst_day_abs, abs(float(metrics.get("worst_day", 0.0))))
                mean_trades.append(float(metrics.get("n_trades", 0.0)))
        positive_ratio = float(aggregate.get("positive_pair_count", 0.0)) / max(inferred_pair_count, 1)
        positive_ratios.append(positive_ratio)
        max_dispersion = max(max_dispersion, float(aggregate.get("pair_return_dispersion", 0.0)))
        max_drawdown_excess = max(
            max_drawdown_excess,
            max(0.0, abs(float(aggregate.get("worst_max_drawdown", 0.0))) - 0.18),
        )

    if min_sharpe == float("inf"):
        min_sharpe = 0.0
    reserve = float(
        min(
            fast_stress_proxy_reserve(windows),
            pair_metric_or_default(
                windows,
                label="full_4y",
                pair="BNBUSDT",
                metric="avg_daily_return",
                default_metric="worst_pair_avg_daily_return",
            )
            - target_daily_return,
        )
    )
    score = 0.0
    score += reserve * 120.0
    score += min_target_hit * 0.35
    score += min_win_rate * 0.25
    score += float(np.clip(min_sharpe / 3.0, -1.0, 1.0)) * 0.20
    score += float(np.mean(np.asarray(positive_ratios, dtype="float64"))) * 0.20 if positive_ratios else 0.0
    score -= aggregate_target_shortfall(windows, target_daily_return=target_daily_return) * 90.0
    score -= bnb_full_4y_target_shortfall(windows, target_daily_return=target_daily_return) * 120.0
    score -= max_dispersion * 18.0
    score -= max_worst_day_abs * 16.0
    score -= max_drawdown_excess * 12.0
    score -= float(np.mean(np.asarray(mean_trades, dtype="float64"))) / 400.0 if mean_trades else 0.0
    return float(score)


def build_ultra_conservative_stress_proxy(
    *,
    df_all: pd.DataFrame,
    raw_signal_all: dict[str, pd.Series],
    funding_all: dict[str, pd.DataFrame],
    library: list[Any],
    candidate: dict[str, Any],
    pairs: tuple[str, ...],
    target_daily_return: float,
    seed_offset: int,
) -> dict[str, Any]:
    scenario = next(item for item in SCENARIOS if item.name == "ultra_conservative")
    windows = build_stress_candidate_metrics(
        df_all,
        raw_signal_all,
        funding_all,
        library,
        candidate,
        pairs,
        scenario,
        seed_offset=seed_offset,
    )
    bnb_metrics = (windows["full_4y"]["per_pair"] or {}).get("BNBUSDT") or {}
    recent_2m_worst = float(windows["recent_2m"]["aggregate"]["worst_pair_avg_daily_return"])
    recent_6m_worst = float(windows["recent_6m"]["aggregate"]["worst_pair_avg_daily_return"])
    full_4y_worst = float(windows["full_4y"]["aggregate"]["worst_pair_avg_daily_return"])
    bnb_full_4y = float(bnb_metrics.get("avg_daily_return", full_4y_worst))
    recent_6m_mdd = abs(float(windows["recent_6m"]["aggregate"]["worst_max_drawdown"]))
    full_4y_mdd = abs(float(windows["full_4y"]["aggregate"]["worst_max_drawdown"]))
    reserve = min(recent_2m_worst, recent_6m_worst, full_4y_worst, bnb_full_4y) - float(target_daily_return)
    passed = bool(
        reserve >= 0.0
        and recent_6m_mdd <= 0.17
        and full_4y_mdd <= 0.20
    )
    return {
        "scenario": scenario.name,
        "evaluated": True,
        "recent_2m_worst_pair_avg_daily_return": recent_2m_worst,
        "recent_6m_worst_pair_avg_daily_return": recent_6m_worst,
        "full_4y_worst_pair_avg_daily_return": full_4y_worst,
        "bnb_full_4y_avg_daily_return": bnb_full_4y,
        "recent_6m_worst_mdd_abs": recent_6m_mdd,
        "full_4y_worst_mdd_abs": full_4y_mdd,
        "reserve": float(reserve),
        "passed": passed,
    }


def candidate_score(
    windows: dict[str, Any],
    base_2m: float,
    base_6m: float,
    base_4y: float,
    base_4y_mean: float,
    base_bnb_4y: float,
    target_daily_return: float,
) -> float:
    recent_2m = unwrap_window_aggregate(windows["recent_2m"])
    recent_6m = unwrap_window_aggregate(windows["recent_6m"])
    full_4y = unwrap_window_aggregate(windows["full_4y"])
    bnb_full_4y = pair_metric_or_default(
        windows,
        label="full_4y",
        pair="BNBUSDT",
        metric="avg_daily_return",
        default_metric="worst_pair_avg_daily_return",
    )
    score = 0.0
    score += (float(recent_2m["worst_pair_avg_daily_return"]) - base_2m) * 18.0
    score += (float(recent_6m["worst_pair_avg_daily_return"]) - base_6m) * 14.0
    score += (float(full_4y["worst_pair_avg_daily_return"]) - base_4y) * 10.0
    score += (float(full_4y["mean_avg_daily_return"]) - base_4y_mean) * 8.0
    score += (bnb_full_4y - base_bnb_4y) * 18.0
    score -= target_shortfall(float(recent_2m["worst_pair_avg_daily_return"]), target_daily_return) * 28.0
    score -= target_shortfall(float(recent_6m["worst_pair_avg_daily_return"]), target_daily_return) * 32.0
    score -= target_shortfall(float(full_4y["worst_pair_avg_daily_return"]), target_daily_return) * 44.0
    score -= target_shortfall(bnb_full_4y, target_daily_return) * 64.0
    score -= target_shortfall(float(full_4y["mean_avg_daily_return"]), target_daily_return) * 18.0
    score -= abs(float(recent_6m["worst_max_drawdown"])) * 0.45
    score -= abs(float(full_4y["worst_max_drawdown"])) * 0.40
    score += fast_validation_robustness_proxy(
        windows,
        target_daily_return=target_daily_return,
    ) * 1.8
    return float(score)


def candidate_score_with_validation(item: dict[str, Any]) -> float:
    score = float(item["score"])
    validation = item.get("validation") or {}
    validation_engine = item.get("validation_engine") or {}
    validation_robustness = item.get("validation_robustness") or build_validation_robustness_profile(validation_engine)
    profile = validation_engine.get("profile") or {}
    market_os = validation_engine.get("market_operating_system") or {}
    market_fitness = (market_os.get("fitness") or {}).get("score", 0.0)
    gate = validation_engine.get("gate") or {}
    pareto = item.get("pareto") or {}
    target_profile = (validation.get("profiles") or {}).get("target_060") or {}
    final_oos_profile = (validation.get("profiles") or {}).get("final_oos") or {}
    stress_proxy = item.get("stress_proxy") or {}
    aggregate_windows = {
        label: item["windows"][label]["aggregate"]
        for label in ("recent_2m", "recent_6m", "full_4y")
    }
    score += float(market_fitness) * 1.35
    score += float(profile.get("validation_quality_score", 0.0)) * 0.35
    score -= float(profile.get("false_positive_risk", 1.0)) * 0.45
    score += float(validation_robustness.get("score", 0.0)) * 1.10
    score += float(validation_robustness.get("gate_pass_ratio", 0.0)) * 0.20
    score -= aggregate_target_shortfall(aggregate_windows, target_daily_return=TARGET_060_DAILY_RETURN) * 14.0
    if (market_os.get("gate") or {}).get("passed"):
        score += 0.20
    if gate.get("passed"):
        score += 0.25
    if target_profile.get("passed"):
        score += 0.45
    if final_oos_profile.get("passed"):
        score += 0.15
    if stress_proxy.get("evaluated"):
        score += float(stress_proxy.get("reserve", 0.0)) * 24.0
        if stress_proxy.get("passed"):
            score += 0.60
    if pareto:
        score += max(0.0, 5.0 - float(pareto.get("rank", 99.0))) * 0.12
        score += min(float(pareto.get("crowding_sort_value", 0.0)), 10.0) * 0.02
    return float(score)


def stress_aware_fitness(item: dict[str, Any]) -> float:
    score = candidate_score_with_validation(item)
    windows = item.get("windows") or {}
    validation_engine = item.get("validation_engine") or {}
    validation_robustness = item.get("validation_robustness") or build_validation_robustness_profile(validation_engine)
    validation_gate = validation_engine.get("gate") or {}
    market_os_gate = ((validation_engine.get("market_operating_system") or {}).get("gate") or {})
    stress_proxy = item.get("stress_proxy") or {}
    validation = item.get("validation") or {}
    target_profile = (validation.get("profiles") or {}).get("target_060") or {}
    final_oos_profile = (validation.get("profiles") or {}).get("final_oos") or {}

    bnb_shortfall = bnb_full_4y_target_shortfall(windows, target_daily_return=TARGET_060_DAILY_RETURN)
    score -= bnb_shortfall * 120.0
    if not bool(validation_gate.get("passed", False)):
        score -= 1000.0
    if not bool(market_os_gate.get("passed", False)):
        score -= 500.0
    if not bool(target_profile.get("passed", False)):
        score -= 120.0
    if not bool(final_oos_profile.get("passed", False)):
        score -= 80.0
    score += float(validation_robustness.get("score", 0.0)) * 80.0
    score += float(validation_robustness.get("gate_pass_ratio", 0.0)) * 25.0

    if stress_proxy.get("evaluated"):
        score -= target_shortfall(
            float(stress_proxy.get("recent_2m_worst_pair_avg_daily_return", 0.0)),
            TARGET_060_DAILY_RETURN,
        ) * 140.0
        score -= target_shortfall(
            float(stress_proxy.get("full_4y_worst_pair_avg_daily_return", 0.0)),
            TARGET_060_DAILY_RETURN,
        ) * 180.0
        score -= target_shortfall(
            float(stress_proxy.get("bnb_full_4y_avg_daily_return", 0.0)),
            TARGET_060_DAILY_RETURN,
        ) * 220.0
        score -= max(0.0, float(stress_proxy.get("full_4y_worst_mdd_abs", 0.0)) - 0.20) * 90.0
        score -= max(0.0, float(stress_proxy.get("recent_6m_worst_mdd_abs", 0.0)) - 0.17) * 70.0
        if bool(stress_proxy.get("passed", False)):
            score += 200.0
    else:
        score -= 400.0

    return float(score)


def early_validation_aware_score(item: dict[str, Any]) -> float:
    score = float(item.get("score", 0.0))
    validation_engine = item.get("validation_engine") or {}
    validation_robustness = item.get("validation_robustness") or build_validation_robustness_profile(validation_engine)
    validation_gate = validation_engine.get("gate") or {}
    market_os_gate = ((validation_engine.get("market_operating_system") or {}).get("gate") or {})
    score += float(validation_robustness.get("score", 0.0)) * 2.0
    score += float(validation_robustness.get("gate_pass_ratio", 0.0)) * 0.5
    if bool(validation_gate.get("passed", False)):
        score += 1.0
    else:
        score -= 1.4
    if bool(market_os_gate.get("passed", False)):
        score += 0.6
    else:
        score -= 0.8
    return float(score)


def build_pairwise_pareto_vector(item: dict[str, Any]) -> dict[str, float]:
    validation_engine = item.get("validation_engine") or {}
    market_os = validation_engine.get("market_operating_system") or {}
    profile = validation_engine.get("profile") or {}
    windows = item["windows"]
    return {
        "market_os_fitness": float((market_os.get("fitness") or {}).get("score", 0.0)),
        "validation_quality_score": float(profile.get("validation_quality_score", 0.0)),
        "recent_2m_worst_pair_avg_daily_return": float(windows["recent_2m"]["aggregate"]["worst_pair_avg_daily_return"]),
        "recent_6m_worst_pair_avg_daily_return": float(windows["recent_6m"]["aggregate"]["worst_pair_avg_daily_return"]),
        "full_4y_worst_pair_avg_daily_return": float(windows["full_4y"]["aggregate"]["worst_pair_avg_daily_return"]),
        "bnb_full_4y_avg_daily_return": pair_metric_or_default(
            windows,
            label="full_4y",
            pair="BNBUSDT",
            metric="avg_daily_return",
            default_metric="worst_pair_avg_daily_return",
        ),
        "corr_state_robustness": float((market_os.get("state_summary") or {}).get("corr_state_robustness", 0.0)),
        "target_060_shortfall": aggregate_target_shortfall(
            windows,
            target_daily_return=TARGET_060_DAILY_RETURN,
        ),
        "bnb_full_4y_target_shortfall": bnb_full_4y_target_shortfall(
            windows,
            target_daily_return=TARGET_060_DAILY_RETURN,
        ),
        "full_4y_worst_max_drawdown_abs": float(abs(windows["full_4y"]["aggregate"]["worst_max_drawdown"])),
        "false_positive_risk": float(profile.get("false_positive_risk", 1.0)),
        "turnover_cost": float((market_os.get("fitness") or {}).get("raw", {}).get("turnover_cost", 0.0)),
    }


def pairwise_dominates(left: dict[str, Any], right: dict[str, Any], *, eps: float = 1e-12) -> bool:
    better_or_equal = True
    strictly_better = False
    left_vec = left["pareto_vector"]
    right_vec = right["pareto_vector"]
    for name, maximize in PAIRWISE_PARETO_OBJECTIVES:
        lhs = float(left_vec[name])
        rhs = float(right_vec[name])
        if maximize:
            if lhs + eps < rhs:
                better_or_equal = False
                break
            if lhs > rhs + eps:
                strictly_better = True
        else:
            if lhs > rhs + eps:
                better_or_equal = False
                break
            if lhs + eps < rhs:
                strictly_better = True
    return better_or_equal and strictly_better


def assign_pairwise_pareto_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    remaining = list(rows)
    fronts: list[list[dict[str, Any]]] = []
    while remaining:
        front: list[dict[str, Any]] = []
        for row in remaining:
            if not any(pairwise_dominates(other, row) for other in remaining if other["candidate_id"] != row["candidate_id"]):
                front.append(row)
        fronts.append(front)
        front_ids = {row["candidate_id"] for row in front}
        remaining = [row for row in remaining if row["candidate_id"] not in front_ids]

    metadata: dict[str, dict[str, Any]] = {}
    for rank, front in enumerate(fronts, start=1):
        distances = {row["candidate_id"]: 0.0 for row in front}
        if len(front) <= 2:
            for row in front:
                distances[row["candidate_id"]] = float("inf")
        else:
            for name, _ in PAIRWISE_PARETO_OBJECTIVES:
                ordered = sorted(front, key=lambda row: float(row["pareto_vector"][name]))
                min_value = float(ordered[0]["pareto_vector"][name])
                max_value = float(ordered[-1]["pareto_vector"][name])
                distances[ordered[0]["candidate_id"]] = float("inf")
                distances[ordered[-1]["candidate_id"]] = float("inf")
                scale = max(max_value - min_value, 1e-8)
                for idx in range(1, len(ordered) - 1):
                    candidate_id_value = ordered[idx]["candidate_id"]
                    if np.isinf(distances[candidate_id_value]):
                        continue
                    prev_value = float(ordered[idx - 1]["pareto_vector"][name])
                    next_value = float(ordered[idx + 1]["pareto_vector"][name])
                    distances[candidate_id_value] += (next_value - prev_value) / scale

        for row in front:
            crowding_distance = float(distances[row["candidate_id"]])
            metadata[row["candidate_id"]] = {
                "rank": int(rank),
                "is_nondominated": bool(rank == 1),
                "crowding_distance": crowding_distance,
                "crowding_sort_value": float(1e9 if np.isinf(crowding_distance) else crowding_distance),
            }
    return metadata


def build_candidate_validation_input(
    candidate: dict[str, Any],
    *,
    pairs: tuple[str, ...],
    window_cache: dict[str, Any],
    library: list[Any],
    library_lookup: dict[str, Any],
) -> dict[str, Any]:
    pair_payloads: list[dict[str, Any]] = []
    window_pairs = window_cache["full_4y"]["pairs"]
    for pair in pairs:
        cfg = candidate["pair_configs"][pair]
        fast_context = window_pairs[pair]["fast_context"]
        result = fast_overlay_replay_from_context(
            fast_context,
            library,
            library_lookup,
            tuple(int(v) for v in cfg["mapping_indices"]),
            float(cfg["route_breadth_threshold"]),
            "python",
        )
        pair_daily_returns = np.asarray(result["daily_metrics"]["daily_returns"], dtype="float64")
        pair_daily_index = pd.DatetimeIndex(fast_context["validation_daily_index"])
        daily_state_codes = (
            pd.Series(
                np.asarray(fast_context["bucket_codes"][float(cfg["route_breadth_threshold"])], dtype="int64"),
                index=pd.DatetimeIndex(fast_context["bar_day_index"]),
            )
            .groupby(level=0)
            .last()
            .reindex(pair_daily_index, method="ffill")
            .fillna(0)
            .astype("int64")
            .to_numpy(dtype="int64")
        )
        pair_payloads.append(
            {
                "pair": pair,
                "daily_returns": pair_daily_returns,
                "daily_index": pair_daily_index,
                "daily_state_codes": daily_state_codes,
                "route_state_names": list(fast_context["route_state_names"]),
            }
        )
    if not pair_payloads:
        return {
            "daily_returns": np.asarray([], dtype="float64"),
            "daily_index": pd.DatetimeIndex([]),
            "state_payload": {
                "route_state_returns": {},
                "corr_bucket_returns": {},
                "total_route_states": 0,
                "total_corr_buckets": 0,
            },
        }
    common_len = min(len(item["daily_returns"]) for item in pair_payloads)
    daily_index = pd.DatetimeIndex(pair_payloads[0]["daily_index"][:common_len])
    aligned = np.vstack([item["daily_returns"][:common_len] for item in pair_payloads])
    route_state_returns: dict[str, list[float]] = {}
    corr_bucket_returns: dict[str, list[float]] = {}
    total_route_states = max(len(item["route_state_names"]) for item in pair_payloads)
    total_corr_buckets = max(
        len(
            {
                name.split(":", 1)[0]
                for payload in pair_payloads
                for name in payload["route_state_names"]
            }
        ),
        1,
    )
    for item in pair_payloads:
        codes = item["daily_state_codes"][:common_len]
        returns = item["daily_returns"][:common_len]
        names = item["route_state_names"]
        for code, value in zip(codes, returns, strict=False):
            route_state_name = names[int(code)] if 0 <= int(code) < len(names) else f"state_{int(code)}"
            route_state_returns.setdefault(route_state_name, []).append(float(value))
            corr_bucket = route_state_name.split(":", 1)[0] if ":" in route_state_name else "base"
            corr_bucket_returns.setdefault(corr_bucket, []).append(float(value))
    return {
        "daily_returns": np.mean(aligned, axis=0),
        "daily_index": daily_index,
        "state_payload": {
            "route_state_returns": route_state_returns,
            "corr_bucket_returns": corr_bucket_returns,
            "total_route_states": int(total_route_states),
            "total_corr_buckets": int(total_corr_buckets),
        },
    }


def build_candidate_cost_reference(windows: dict[str, Any]) -> dict[str, Any]:
    per_pair = (windows.get("full_4y") or {}).get("per_pair") or {}
    if not per_pair:
        return {
            "mean_cost_ratio": 0.0,
            "max_cost_ratio": 0.0,
            "mean_n_trades": 0.0,
        }
    cost_ratios = []
    trades = []
    for metrics in per_pair.values():
        total_cost = (
            abs(float(metrics.get("fee_paid", 0.0)))
            + abs(float(metrics.get("slippage_paid", 0.0)))
            + abs(float(metrics.get("funding_paid", 0.0)))
        )
        capital_base = max(float(metrics.get("final_equity", gp.INITIAL_CASH)), float(gp.INITIAL_CASH), 1.0)
        cost_ratios.append(total_cost / capital_base)
        trades.append(float(metrics.get("n_trades", 0.0)))
    return {
        "mean_cost_ratio": float(np.mean(np.asarray(cost_ratios, dtype="float64"))),
        "max_cost_ratio": float(np.max(np.asarray(cost_ratios, dtype="float64"))),
        "mean_n_trades": float(np.mean(np.asarray(trades, dtype="float64"))),
    }


def main() -> None:
    args = parse_args()
    started = perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    route_state_mode = normalize_route_state_mode(args.route_state_mode)
    route_gene_count = route_state_count(route_state_mode)

    pairs = parse_csv_tuple(args.pairs, str)
    candidate_summary_paths = parse_csv_tuple(args.candidate_summaries, str)
    route_thresholds = (0.35, 0.5, 0.65, 0.8)

    baseline_summary = json.loads(Path(args.baseline_summary).read_text())
    baseline_configs = baseline_pair_configs(pairs, baseline_summary, route_state_mode)
    base_windows = baseline_summary["selected_candidate"]["windows"]
    base_2m = float(base_windows["recent_2m"]["aggregate"]["worst_pair_avg_daily_return"])
    base_6m = float(base_windows["recent_6m"]["aggregate"]["worst_pair_avg_daily_return"])
    base_4y = float(base_windows["full_4y"]["aggregate"]["worst_pair_avg_daily_return"])
    base_4y_mean = float(base_windows["full_4y"]["aggregate"]["mean_avg_daily_return"])
    base_bnb_4y = float(base_windows["full_4y"]["per_pair"]["BNBUSDT"]["avg_daily_return"])
    target_daily_return = float(args.target_daily_return)

    model, _ = load_model(Path(args.model))
    compiled = gp.toolbox.compile(expr=model)
    library = list(iter_params())
    library_lookup = build_library_lookup(library)

    start_all = DEFAULT_WINDOWS[-1][1]
    end_all = DEFAULT_WINDOWS[-1][2]
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=False)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {pair: load_or_fetch_funding(pair, start_all, end_all) for pair in pairs}

    prepare_started = perf_counter()
    window_cache: dict[str, Any] = {}
    for label, start, end in DEFAULT_WINDOWS:
        df = df_all.loc[start:end].copy()
        pair_cache = {}
        for pair in pairs:
            overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
            signal_slice = raw_signal_all[pair].loc[start:end].copy()
            funding_slice = funding_all[pair]
            if not funding_slice.empty:
                funding_slice = funding_slice[
                    (funding_slice["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                    & (funding_slice["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
                ].copy()
            pair_cache[pair] = {
                "fast_context": build_fast_context(
                    df=df,
                    pair=pair,
                    raw_signal=signal_slice,
                    overlay_inputs=overlay_inputs,
                    route_thresholds=route_thresholds,
                    library_lookup=library_lookup,
                    funding_df=funding_slice,
                    route_state_mode=route_state_mode,
                ),
            }
        window_cache[label] = {"pairs": pair_cache}
    prepare_seconds = perf_counter() - prepare_started

    def eval_fast(candidate: dict[str, Any]) -> dict[str, Any]:
        windows = {}
        for label, _, _ in DEFAULT_WINDOWS:
            per_pair = {}
            for pair in pairs:
                cfg = candidate["pair_configs"][pair]
                per_pair[pair] = summarize_single_result(
                    fast_overlay_replay_from_context(
                        window_cache[label]["pairs"][pair]["fast_context"],
                        library,
                        library_lookup,
                        tuple(int(v) for v in cfg["mapping_indices"]),
                        float(cfg["route_breadth_threshold"]),
                        "numba",
                    )
                )
            windows[label] = {
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
        return windows

    def eval_realistic(candidate: dict[str, Any]) -> dict[str, Any]:
        windows = {}
        for label, start, end in DEFAULT_WINDOWS:
            per_pair = {}
            for pair in pairs:
                cfg = candidate["pair_configs"][pair]
                per_pair[pair] = realistic_overlay_replay_from_context(
                    window_cache[label]["pairs"][pair]["fast_context"],
                    library_lookup,
                    tuple(int(v) for v in cfg["mapping_indices"]),
                    float(cfg["route_breadth_threshold"]),
                )
            windows[label] = {
                "start": start,
                "end": end,
                "bars": int(len(df_all.loc[start:end])),
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
        return windows

    start_candidates: list[dict[str, Any]] = [{"pair_configs": baseline_configs}]
    existing_candidate_summary_paths: list[str] = []
    missing_candidate_summary_paths: list[str] = []
    for raw_path in candidate_summary_paths:
        if Path(raw_path).exists():
            existing_candidate_summary_paths.append(raw_path)
        else:
            missing_candidate_summary_paths.append(raw_path)

    top_pair_configs: list[dict[str, Any]] = []
    for raw_path in existing_candidate_summary_paths:
        obj = json.loads(Path(raw_path).read_text())
        for item in obj.get("realistic_top_candidates", []):
            pair_configs = item.get("pair_configs")
            if not pair_configs:
                continue
            top_pair_configs.append(
                {
                    "pair_configs": {
                        pair: normalize_pair_config(pair_configs[pair], route_state_mode)
                        for pair in pairs
                    }
                }
            )
    start_candidates.extend(top_pair_configs)

    mixed_candidates: list[dict[str, Any]] = []
    best_r2 = max(top_pair_configs, key=lambda c: c["pair_configs"][pairs[0]]["route_breadth_threshold"], default=None)
    del best_r2  # not used directly; mixed pool below is generic.
    for left in top_pair_configs[:8]:
        for right in top_pair_configs[:8]:
            mixed_candidates.append(
                {
                    "pair_configs": {
                        pairs[0]: clone_candidate(left)["pair_configs"][pairs[0]],
                        pairs[1]: clone_candidate(right)["pair_configs"][pairs[1]],
                    }
                }
            )
    start_candidates.extend(mixed_candidates)

    unique_starts: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in start_candidates:
        unique_starts[candidate_key(candidate, pairs)] = candidate
    start_candidates = list(unique_starts.values())

    start_eval = []
    for candidate in start_candidates:
        windows = eval_realistic(candidate)
        start_eval.append(
            {
                "pair_configs": candidate["pair_configs"],
                "windows": windows,
                "score": candidate_score(
                    windows,
                    base_2m,
                    base_6m,
                    base_4y,
                    base_4y_mean,
                    base_bnb_4y,
                    target_daily_return,
                ),
            }
        )
    start_eval.sort(key=lambda item: item["score"], reverse=True)
    chosen_starts = start_eval[: max(int(args.start_candidate_count), 1)]
    chosen_start = chosen_starts[0]

    gene_pool = {
        pair: {
            "thresholds": sorted({float(item["pair_configs"][pair]["route_breadth_threshold"]) for item in start_eval[:12]} | set(route_thresholds)),
            "mappings": [
                sorted({int(item["pair_configs"][pair]["mapping_indices"][bucket]) for item in start_eval[:20]})
                for bucket in range(route_gene_count)
            ],
            "corr_blocks": [
                sorted(
                    {
                        tuple(int(v) for v in item["pair_configs"][pair]["mapping_indices"][block * 4:(block + 1) * 4])
                        for item in start_eval[:20]
                    }
                )
                for block in range(max(route_gene_count // 4, 1))
            ],
            "base_state_values": [
                sorted(
                    {
                        int(item["pair_configs"][pair]["mapping_indices"][idx])
                        for item in start_eval[:20]
                        for idx in range(base_slot, route_gene_count, 4)
                    }
                )
                for base_slot in range(min(4, route_gene_count))
            ],
        }
        for pair in pairs
    }

    candidates: list[dict[str, Any]] = []
    for start_candidate in chosen_starts:
        seed_pair_configs = clone_candidate({"pair_configs": start_candidate["pair_configs"]})["pair_configs"]
        candidates.append({"pair_configs": clone_candidate({"pair_configs": seed_pair_configs})["pair_configs"]})
        for pair in pairs:
            for gene in range(route_gene_count + 1):
                if gene == 0:
                    for value in gene_pool[pair]["thresholds"]:
                        candidate = {"pair_configs": clone_candidate({"pair_configs": seed_pair_configs})["pair_configs"]}
                        candidate["pair_configs"][pair]["route_breadth_threshold"] = float(value)
                        candidates.append(candidate)
                else:
                    for value in gene_pool[pair]["mappings"][gene - 1]:
                        candidate = {"pair_configs": clone_candidate({"pair_configs": seed_pair_configs})["pair_configs"]}
                        candidate["pair_configs"][pair]["mapping_indices"][gene - 1] = int(value)
                        candidates.append(candidate)
            if route_gene_count >= 8:
                for block_index, block_patterns in enumerate(gene_pool[pair]["corr_blocks"]):
                    for block_values in block_patterns:
                        candidate = {"pair_configs": clone_candidate({"pair_configs": seed_pair_configs})["pair_configs"]}
                        candidate["pair_configs"][pair]["mapping_indices"] = replace_corr_block(
                            candidate["pair_configs"][pair]["mapping_indices"],
                            block_index,
                            block_values,
                        )
                        candidates.append(candidate)
                for base_slot, values in enumerate(gene_pool[pair]["base_state_values"]):
                    for value in values:
                        candidate = {"pair_configs": clone_candidate({"pair_configs": seed_pair_configs})["pair_configs"]}
                        candidate["pair_configs"][pair]["mapping_indices"] = replace_base_state_across_blocks(
                            candidate["pair_configs"][pair]["mapping_indices"],
                            base_slot,
                            int(value),
                            route_gene_count,
                        )
                        candidates.append(candidate)

    for _ in range(args.random_mutations):
        seed_candidate = random.choice(chosen_starts)
        candidate = {"pair_configs": clone_candidate({"pair_configs": seed_candidate["pair_configs"]})["pair_configs"]}
        steps = random.choice((1, 1, 1, 2, 2, 3))
        for _ in range(steps):
            pair = random.choice(pairs)
            mutation_mode = random.choice(
                ("threshold", "single_bucket", "single_bucket", "single_bucket", "corr_block", "base_state")
            )
            if mutation_mode == "threshold":
                candidate["pair_configs"][pair]["route_breadth_threshold"] = float(random.choice(route_thresholds))
            elif mutation_mode == "single_bucket":
                gene = random.randrange(route_gene_count)
                bucket = gene
                if random.random() < 0.6 and gene_pool[pair]["mappings"][bucket]:
                    candidate["pair_configs"][pair]["mapping_indices"][bucket] = int(
                        random.choice(gene_pool[pair]["mappings"][bucket])
                    )
                else:
                    candidate["pair_configs"][pair]["mapping_indices"][bucket] = int(random.randrange(len(library)))
            elif mutation_mode == "corr_block" and route_gene_count >= 8:
                block_index = random.randrange(max(route_gene_count // 4, 1))
                patterns = gene_pool[pair]["corr_blocks"][block_index]
                if patterns and random.random() < 0.7:
                    block_values = random.choice(patterns)
                else:
                    block_values = tuple(int(random.randrange(len(library))) for _ in range(4))
                candidate["pair_configs"][pair]["mapping_indices"] = replace_corr_block(
                    candidate["pair_configs"][pair]["mapping_indices"],
                    block_index,
                    block_values,
                )
            elif mutation_mode == "base_state" and route_gene_count >= 8:
                base_slot = random.randrange(4)
                values = gene_pool[pair]["base_state_values"][base_slot]
                value = int(random.choice(values)) if values and random.random() < 0.7 else int(random.randrange(len(library)))
                candidate["pair_configs"][pair]["mapping_indices"] = replace_base_state_across_blocks(
                    candidate["pair_configs"][pair]["mapping_indices"],
                    base_slot,
                    value,
                    route_gene_count,
                )
            else:
                gene = random.randrange(route_gene_count + 1)
                if gene == 0:
                    candidate["pair_configs"][pair]["route_breadth_threshold"] = float(random.choice(route_thresholds))
                    continue
                bucket = gene - 1
                if random.random() < 0.6 and gene_pool[pair]["mappings"][bucket]:
                    candidate["pair_configs"][pair]["mapping_indices"][bucket] = int(
                        random.choice(gene_pool[pair]["mappings"][bucket])
                    )
                else:
                    candidate["pair_configs"][pair]["mapping_indices"][bucket] = int(random.randrange(len(library)))
        candidates.append(candidate)

    fast_ranked = []
    seen: set[tuple[Any, ...]] = set()
    baseline_target_shortfall = aggregate_target_shortfall(
        base_windows,
        target_daily_return=target_daily_return,
    )
    baseline_bnb_target_shortfall = bnb_full_4y_target_shortfall(
        base_windows,
        target_daily_return=target_daily_return,
    )

    for candidate in candidates:
        key = candidate_key(candidate, pairs)
        if key in seen:
            continue
        seen.add(key)
        windows = eval_fast(candidate)
        validation_robustness_proxy = fast_validation_robustness_proxy(
            windows,
            target_daily_return=target_daily_return,
        )
        score = candidate_score(
            windows,
            base_2m,
            base_6m,
            base_4y,
            base_4y_mean,
            base_bnb_4y,
            target_daily_return,
        )
        recent_2m_aggregate = unwrap_window_aggregate(windows["recent_2m"])
        recent_6m_aggregate = unwrap_window_aggregate(windows["recent_6m"])
        full_4y_aggregate = unwrap_window_aggregate(windows["full_4y"])
        bnb_full_4y = pair_metric_or_default(
            windows,
            label="full_4y",
            pair="BNBUSDT",
            metric="avg_daily_return",
            default_metric="worst_pair_avg_daily_return",
        )
        if (
            recent_2m_aggregate["worst_pair_avg_daily_return"] >= base_2m * 0.97
            and aggregate_target_shortfall(windows, target_daily_return=target_daily_return)
            <= baseline_target_shortfall
            and bnb_full_4y_target_shortfall(windows, target_daily_return=target_daily_return)
            <= baseline_bnb_target_shortfall
            and recent_6m_aggregate["worst_pair_avg_daily_return"] >= base_6m * 0.97
            and full_4y_aggregate["mean_avg_daily_return"] >= base_4y_mean * 0.995
            and bnb_full_4y >= base_bnb_4y * 0.97
        ):
            fast_ranked.append(
                {
                    "pair_configs": candidate["pair_configs"],
                    "windows": windows,
                    "score": score,
                    "validation_robustness_proxy": validation_robustness_proxy,
                }
            )
    fast_ranked.sort(
        key=lambda item: (
            item.get("validation_robustness_proxy", float("-inf")),
            -bnb_full_4y_target_shortfall(item["windows"], target_daily_return=target_daily_return),
            -aggregate_target_shortfall(item["windows"], target_daily_return=target_daily_return),
            item["score"],
        ),
        reverse=True,
    )

    validation_candidates = fast_ranked[: max(int(args.validation_candidate_count), args.top_realistic)]
    validation_frames_by_key = {}
    validation_state_payloads = {}
    validation_inputs_by_key = {}
    for item in validation_candidates:
        candidate = {"pair_configs": item["pair_configs"]}
        key = candidate_id(candidate, pairs)
        validation_input = build_candidate_validation_input(
            candidate,
            pairs=pairs,
            window_cache=window_cache,
            library=library,
            library_lookup=library_lookup,
        )
        validation_inputs_by_key[key] = validation_input
        validation_frames_by_key[key] = build_return_frame(validation_input["daily_returns"], validation_input["daily_index"])
        validation_state_payloads[key] = validation_input["state_payload"]
        item["candidate_id"] = key
    for item in validation_candidates:
        key = item["candidate_id"]
        validation_engine = build_candidate_validation_bundle(
            key,
            validation_inputs_by_key[key]["daily_returns"],
            validation_inputs_by_key[key]["daily_index"],
            trial_count=max(len(validation_candidates), 1),
            peer_frames_by_key=validation_frames_by_key,
            state_payload=validation_state_payloads[key],
            cost_reference=build_candidate_cost_reference(item["windows"]),
            cpcv_blocks=args.cpcv_blocks,
            cpcv_test_blocks=args.cpcv_test_blocks,
            cpcv_embargo_days=args.cpcv_embargo_days,
        )
        item["validation_engine"] = validation_engine
        item["validation_robustness"] = (
            validation_engine.get("robustness")
            or build_validation_robustness_profile(validation_engine)
        )
        item["early_validation_score"] = early_validation_aware_score(item)
    validation_candidates.sort(key=early_validation_aware_score, reverse=True)

    realistic_top = []
    for item in validation_candidates[: args.top_realistic]:
        windows = eval_realistic({"pair_configs": item["pair_configs"]})
        key = item["candidate_id"]
        validation_input = validation_inputs_by_key[key]
        validation_engine = build_candidate_validation_bundle(
            key,
            validation_input["daily_returns"],
            validation_input["daily_index"],
            trial_count=max(len(validation_candidates), 1),
            peer_frames_by_key=validation_frames_by_key,
            state_payload=validation_state_payloads.get(key, validation_input["state_payload"]),
            cost_reference=build_candidate_cost_reference(windows),
            cpcv_blocks=args.cpcv_blocks,
            cpcv_test_blocks=args.cpcv_test_blocks,
            cpcv_embargo_days=args.cpcv_embargo_days,
        )
        realistic_top.append(
            {
                "pair_configs": item["pair_configs"],
                "windows": windows,
                "validation_robustness_proxy": item.get("validation_robustness_proxy"),
                "early_validation_score": item.get("early_validation_score"),
                "score": candidate_score(
                    windows,
                    base_2m,
                    base_6m,
                    base_4y,
                    base_4y_mean,
                    base_bnb_4y,
                    target_daily_return,
                ),
                "validation": build_validation_bundle(windows, baseline_summary["selected_candidate"]["windows"]),
                "validation_engine": validation_engine,
                "validation_robustness": (
                    validation_engine.get("robustness")
                    or build_validation_robustness_profile(validation_engine)
                ),
                "candidate_id": key,
            }
        )
    for item in realistic_top:
        item["pareto_vector"] = build_pairwise_pareto_vector(item)
    pareto_metadata = assign_pairwise_pareto_metadata(realistic_top)
    for item in realistic_top:
        item["pareto"] = pareto_metadata[item["candidate_id"]]
    realistic_top.sort(key=stress_aware_fitness, reverse=True)

    stress_proxy_candidates = realistic_top[: max(int(args.stress_proxy_candidate_count), 1)]
    stress_proxy_pass_count = 0
    for stress_idx, item in enumerate(stress_proxy_candidates):
        item["stress_proxy"] = build_ultra_conservative_stress_proxy(
            df_all=df_all,
            raw_signal_all=raw_signal_all,
            funding_all=funding_all,
            library=library,
            candidate={"pair_configs": item["pair_configs"]},
            pairs=pairs,
            target_daily_return=target_daily_return,
            seed_offset=900000 + stress_idx * 1000,
        )
        if item["stress_proxy"]["passed"]:
            stress_proxy_pass_count += 1
    for item in realistic_top:
        if "stress_proxy" not in item:
            item["stress_proxy"] = {
                "scenario": "ultra_conservative",
                "evaluated": False,
                "reserve": None,
                "passed": False,
            }
    realistic_top.sort(key=stress_aware_fitness, reverse=True)
    selected = None
    selection_reason = "no_candidates"
    validation_pass_candidates = [
        item for item in realistic_top if item["validation_engine"]["gate"]["passed"]
    ]
    progressive_candidates = [
        item
        for item in validation_pass_candidates
        if item["validation"]["profiles"]["progressive_improvement"]["passed"]
    ]
    target_060_candidates = [
        item
        for item in validation_pass_candidates
        if item["validation"]["profiles"]["target_060"]["passed"]
    ]
    final_oos_audit_pass_count = sum(
        1 for item in validation_pass_candidates if item["validation"]["profiles"]["final_oos"]["passed"]
    )
    stress_proxy_validation_candidates = [
        item
        for item in validation_pass_candidates
        if bool((item.get("stress_proxy") or {}).get("passed", False))
    ]
    if stress_proxy_validation_candidates:
        selected = max(
            stress_proxy_validation_candidates,
            key=lambda item: (
                float((item.get("stress_proxy") or {}).get("reserve", float("-inf"))),
                stress_aware_fitness(item),
            ),
        )
        selection_reason = "stress_proxy_plus_validation"
    else:
        selected = max(target_060_candidates, key=stress_aware_fitness) if target_060_candidates else None
        selection_reason = "target_060_plus_validation"
    if selected is None and progressive_candidates:
        selected = max(progressive_candidates, key=stress_aware_fitness)
        selection_reason = "progressive_plus_validation"
    if selected is None and validation_pass_candidates:
        selected = max(validation_pass_candidates, key=stress_aware_fitness)
        selection_reason = "validation_only"
    if selected is None and realistic_top:
        selected = realistic_top[0]
        selection_reason = "no_validation_gate_pass"

    report = {
        "search": {
            "algorithm": "pairwise_local_repair",
            "seed": args.seed,
            "random_mutations": args.random_mutations,
            "top_realistic": args.top_realistic,
            "candidate_summary_paths": list(candidate_summary_paths),
            "existing_candidate_summary_paths": list(existing_candidate_summary_paths),
            "missing_candidate_summary_paths": list(missing_candidate_summary_paths),
            "library_source": "full-grid",
            "library_size": len(library),
            "route_state_mode": route_state_mode,
            "route_state_count": route_gene_count,
            "cpcv_blocks": int(args.cpcv_blocks),
            "cpcv_test_blocks": int(args.cpcv_test_blocks),
            "cpcv_embargo_days": int(args.cpcv_embargo_days),
            "target_daily_return": target_daily_return,
        },
        "pairs": list(pairs),
        "model_path": str(args.model),
        "baseline_summary_path": str(args.baseline_summary),
        "baseline_candidate": {"pair_configs": baseline_configs},
        "baseline_realistic": baseline_summary["selected_candidate"]["windows"],
        "chosen_start_candidate": chosen_start,
        "chosen_start_candidates": chosen_starts,
        "runtime": {
            "prepare_context_seconds": prepare_seconds,
            "total_seconds": perf_counter() - started,
            "start_candidate_count": len(start_candidates),
            "chosen_start_count": len(chosen_starts),
            "repair_candidate_count": len(seen),
            "fast_ranked_count": len(fast_ranked),
            "validation_candidate_count": len(validation_candidates),
        },
        "selection": {
            "reason": selection_reason,
            "validation_pass_count": len(validation_pass_candidates),
            "final_oos_audit_pass_count": final_oos_audit_pass_count,
            "progressive_validation_pass_count": len(progressive_candidates),
            "target_060_validation_pass_count": len(target_060_candidates),
            "stress_proxy_pass_count": stress_proxy_pass_count,
            "selected_final_oos_passed": bool(
                selected and selected["validation"]["profiles"]["final_oos"]["passed"]
            ),
        },
        "realistic_top_candidates": realistic_top,
        "selected_candidate": selected,
        "created_at": datetime.now(UTC).isoformat(),
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
