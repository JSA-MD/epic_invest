#!/usr/bin/env python3
"""NSGA-III search with pair-specific route thresholds and bucket mappings."""

from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

import gp_crypto_evolution as gp
from pairwise_validation_engine import (
    build_candidate_validation_bundle,
    build_return_frame,
    build_validation_robustness_profile,
)
from replay_regime_mixture_realistic import resolve_candidate, load_model
from repair_pair_subset_pairwise_candidate import (
    TARGET_060_DAILY_RETURN,
    aggregate_target_shortfall,
    bnb_full_4y_target_shortfall,
    build_candidate_cost_reference,
    build_candidate_validation_input,
    build_ultra_conservative_stress_proxy,
    candidate_id,
    fast_validation_robustness_proxy,
    pair_metric_or_default,
    target_shortfall,
    unwrap_window_aggregate,
)
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    NUMBA_AVAILABLE,
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    fast_overlay_replay_from_context,
    json_safe,
    load_or_fetch_funding,
    parse_csv_tuple,
    realistic_overlay_replay_from_context,
    resolve_fast_engine,
    score_realistic_candidate,
    summarize_single_result,
)
from validate_pair_subset_summary import build_validation_bundle


UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pair-specific NSGA-III search for BTC/BNB overlay mappings.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--base-summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
        help="Summary file containing the overlay library.",
    )
    parser.add_argument(
        "--library-source",
        choices=("summary", "full-grid"),
        default="summary",
        help="Use the compact promoted overlay library or the full overlay parameter grid.",
    )
    parser.add_argument(
        "--baseline-summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_nsga3_summary.json"),
        help="Summary file containing the current shared-mapping baseline.",
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_nsga3_summary.json"),
    )
    parser.add_argument(
        "--subset-indices",
        default=None,
        help="Library indices allowed in searched mappings.",
    )
    parser.add_argument(
        "--route-thresholds",
        default="0.35,0.50,0.65,0.80",
        help="Comma-separated route breadth thresholds to test.",
    )
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--generations", type=int, default=36)
    parser.add_argument("--offspring", type=int, default=128)
    parser.add_argument("--top-k-realistic", type=int, default=10)
    parser.add_argument("--ref-p", type=int, default=2)
    parser.add_argument("--cxpb", type=float, default=0.65)
    parser.add_argument("--mutpb", type=float, default=0.35)
    parser.add_argument("--gene-mutpb", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stress-proxy-candidate-count", type=int, default=6)
    parser.add_argument(
        "--fast-engine",
        choices=("auto", "python", "numba"),
        default="auto",
    )
    return parser.parse_args()


def fast_scalar_score(windows: dict[str, Any]) -> float:
    recent_2m = unwrap_window_aggregate(windows["recent_2m"])
    recent_6m = unwrap_window_aggregate(windows["recent_6m"])
    full_4y = unwrap_window_aggregate(windows["full_4y"])
    reserve = fast_stress_proxy_reserve(windows)
    validation_proxy = fast_validation_robustness_proxy(
        windows,
        target_daily_return=TARGET_060_DAILY_RETURN,
    )
    bnb_full_4y = pair_metric_or_default(
        windows,
        label="full_4y",
        pair="BNBUSDT",
        metric="avg_daily_return",
        default_metric="worst_pair_avg_daily_return",
    )
    score = 0.0
    score += reserve * 520000.0
    score += float(recent_2m["worst_pair_avg_daily_return"]) * 460000.0
    score += float(recent_6m["worst_pair_avg_daily_return"]) * 340000.0
    score += float(full_4y["worst_pair_avg_daily_return"]) * 280000.0
    score += float(bnb_full_4y) * 240000.0
    score += float(full_4y["mean_avg_daily_return"]) * 220000.0
    score += float(validation_proxy) * 2600.0
    score -= target_shortfall(float(full_4y["worst_pair_avg_daily_return"]), TARGET_060_DAILY_RETURN) * 420000.0
    score -= target_shortfall(float(bnb_full_4y), TARGET_060_DAILY_RETURN) * 520000.0
    score -= abs(float(recent_2m["worst_max_drawdown"])) * 22000.0
    score -= abs(float(recent_6m["worst_max_drawdown"])) * 17000.0
    score -= abs(float(full_4y["worst_max_drawdown"])) * 12000.0
    score -= float(recent_2m["pair_return_dispersion"]) * 140000.0
    score -= float(recent_6m["pair_return_dispersion"]) * 100000.0
    score -= float(full_4y["pair_return_dispersion"]) * 80000.0
    return float(score)


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


def realistic_stress_aware_score(item: dict[str, Any]) -> float:
    score = score_realistic_candidate(item)
    windows = item["windows"]
    validation = item.get("validation") or {}
    validation_engine = item.get("validation_engine") or {}
    validation_robustness = item.get("validation_robustness") or build_validation_robustness_profile(validation_engine)
    stress_proxy = item.get("stress_proxy") or {}
    target_profile = (validation.get("profiles") or {}).get("target_060") or {}
    progressive_profile = (validation.get("profiles") or {}).get("progressive_improvement") or {}
    final_oos_profile = (validation.get("profiles") or {}).get("final_oos") or {}
    validation_gate = validation_engine.get("gate") or {}
    market_os_gate = ((validation_engine.get("market_operating_system") or {}).get("gate") or {})

    score += fast_stress_proxy_reserve(windows) * 380000.0
    score -= aggregate_target_shortfall(windows, target_daily_return=TARGET_060_DAILY_RETURN) * 220000.0
    score -= bnb_full_4y_target_shortfall(windows, target_daily_return=TARGET_060_DAILY_RETURN) * 320000.0
    score += float(validation_robustness.get("score", 0.0)) * 2400.0
    score += float(validation_robustness.get("gate_pass_ratio", 0.0)) * 700.0
    if bool(validation_gate.get("passed", False)):
        score += 1800.0
    else:
        score -= 2400.0
    if bool(market_os_gate.get("passed", False)):
        score += 900.0
    if bool(target_profile.get("passed", False)):
        score += 1200.0
    if bool(progressive_profile.get("passed", False)):
        score += 600.0
    if bool(final_oos_profile.get("passed", False)):
        score += 500.0
    if stress_proxy.get("evaluated"):
        score += float(stress_proxy.get("reserve", 0.0)) * 480000.0
        score -= target_shortfall(
            float(stress_proxy.get("bnb_full_4y_avg_daily_return", 0.0)),
            TARGET_060_DAILY_RETURN,
        ) * 720000.0
        score -= target_shortfall(
            float(stress_proxy.get("full_4y_worst_pair_avg_daily_return", 0.0)),
            TARGET_060_DAILY_RETURN,
        ) * 560000.0
        if bool(stress_proxy.get("passed", False)):
            score += 3000.0
    else:
        score -= 4000.0
    return float(score)


def early_validation_aware_score(item: dict[str, Any]) -> float:
    score = float(item.get("scalar_score", 0.0))
    validation_engine = item.get("validation_engine") or {}
    validation_robustness = item.get("validation_robustness") or build_validation_robustness_profile(validation_engine)
    validation_gate = validation_engine.get("gate") or {}
    market_os_gate = ((validation_engine.get("market_operating_system") or {}).get("gate") or {})
    score += float(validation_robustness.get("score", 0.0)) * 5200.0
    score += float(validation_robustness.get("gate_pass_ratio", 0.0)) * 1100.0
    if bool(validation_gate.get("passed", False)):
        score += 1500.0
    else:
        score -= 2200.0
    if bool(market_os_gate.get("passed", False)):
        score += 900.0
    else:
        score -= 1200.0
    return float(score)


def build_baseline_pair_configs(pairs: tuple[str, ...], summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = summary["selected_candidate"]
    return {
        pair: {
            "route_breadth_threshold": float(selected["route_breadth_threshold"]),
            "mapping_indices": [int(v) for v in selected["mapping_indices"]],
        }
        for pair in pairs
    }


def candidate_from_individual(
    individual: list[int],
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    pair_configs: dict[str, dict[str, Any]] = {}
    offset = 0
    for pair in pairs:
        threshold = float(route_thresholds[int(individual[offset])])
        mapping = [int(v) for v in individual[offset + 1 : offset + 5]]
        pair_configs[pair] = {
            "route_breadth_threshold": threshold,
            "mapping_indices": mapping,
        }
        offset += 5
    return {"pair_configs": pair_configs}


def build_deap_toolbox(
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
    subset_indices: tuple[int, ...],
    rng: random.Random,
    ref_points: Any,
) -> base.Toolbox:
    if not hasattr(creator, "FitnessPairSubsetPairwiseNSGA3StressAware"):
        creator.create(
            "FitnessPairSubsetPairwiseNSGA3StressAware",
            base.Fitness,
            weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0),
        )
    if not hasattr(creator, "IndividualPairSubsetPairwiseNSGA3StressAware"):
        creator.create(
            "IndividualPairSubsetPairwiseNSGA3StressAware",
            list,
            fitness=creator.FitnessPairSubsetPairwiseNSGA3StressAware,
        )

    gene_len = len(pairs) * 5
    toolbox = base.Toolbox()

    def random_threshold_gene() -> int:
        return rng.randrange(len(route_thresholds))

    def random_mapping_gene() -> int:
        return int(rng.choice(subset_indices))

    def init_individual() -> creator.IndividualPairSubsetPairwiseNSGA3StressAware:
        genes: list[int] = []
        for _ in pairs:
            genes.append(random_threshold_gene())
            genes.extend(random_mapping_gene() for _ in range(4))
        return creator.IndividualPairSubsetPairwiseNSGA3StressAware(genes)

    def mate(ind1: list[int], ind2: list[int]) -> tuple[list[int], list[int]]:
        tools.cxUniform(ind1, ind2, indpb=0.5)
        return ind1, ind2

    def mutate(individual: list[int], indpb: float) -> tuple[list[int]]:
        for idx in range(gene_len):
            if rng.random() >= indpb:
                continue
            if idx % 5 == 0:
                individual[idx] = random_threshold_gene()
            else:
                individual[idx] = random_mapping_gene()
        return (individual,)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate, indpb=0.25)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    toolbox.register("clone", copy.deepcopy)
    return toolbox


def individual_from_pair_configs(
    pair_configs: dict[str, dict[str, Any]],
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
) -> list[int]:
    genes: list[int] = []
    for pair in pairs:
        cfg = pair_configs[pair]
        genes.append(route_thresholds.index(float(cfg["route_breadth_threshold"])))
        genes.extend(int(v) for v in cfg["mapping_indices"])
    return genes


def main() -> None:
    args = parse_args()
    started = perf_counter()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    pairs = parse_csv_tuple(args.pairs, str)
    subset_indices = parse_csv_tuple(args.subset_indices, int) if args.subset_indices else None
    route_thresholds = parse_csv_tuple(args.route_thresholds, float)
    fast_engine = resolve_fast_engine(args.fast_engine)

    baseline_summary = json.loads(Path(args.baseline_summary).read_text())
    baseline_pair_configs = build_baseline_pair_configs(pairs, baseline_summary)

    _, compact_library, _ = resolve_candidate(Path(args.base_summary), None, None)
    if args.library_source == "full-grid":
        library = list(iter_params())
        full_index_by_params = {params: idx for idx, params in enumerate(library)}
        for cfg in baseline_pair_configs.values():
            cfg["mapping_indices"] = [
                full_index_by_params[compact_library[int(idx)]]
                for idx in cfg["mapping_indices"]
            ]
    else:
        library = compact_library

    if subset_indices is None:
        subset_indices = tuple(range(len(library)))

    model, _ = load_model(Path(args.model))
    compiled = gp.toolbox.compile(expr=model)

    for cfg in baseline_pair_configs.values():
        threshold = float(cfg["route_breadth_threshold"])
        if threshold not in route_thresholds:
            route_thresholds = tuple(sorted(set(route_thresholds + (threshold,))))

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
    window_cache: dict[str, dict[str, Any]] = {}
    for label, start, end in DEFAULT_WINDOWS:
        df = df_all.loc[start:end].copy()
        pair_cache: dict[str, Any] = {}
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
                "signal": signal_slice,
                "funding": funding_slice,
                "fast_context": build_fast_context(
                    df=df,
                    pair=pair,
                    raw_signal=signal_slice,
                    overlay_inputs=overlay_inputs,
                    route_thresholds=route_thresholds,
                    library_lookup=library_lookup,
                    funding_df=funding_slice,
                ),
            }
        window_cache[label] = {
            "start": start,
            "end": end,
            "df": df,
            "pairs": pair_cache,
        }
    prepare_seconds = perf_counter() - prepare_started

    baseline_fast: dict[str, Any] = {}
    baseline_realistic: dict[str, Any] = {}
    baseline_started = perf_counter()
    for label, start, end in DEFAULT_WINDOWS:
        window_data = window_cache[label]
        df = window_data["df"]
        per_pair_fast = {}
        per_pair_realistic = {}
        for pair in pairs:
            cfg = baseline_pair_configs[pair]
            pair_data = window_data["pairs"][pair]
            per_pair_fast[pair] = summarize_single_result(
                fast_overlay_replay_from_context(
                    pair_data["fast_context"],
                    library,
                    library_lookup,
                    tuple(int(v) for v in cfg["mapping_indices"]),
                    float(cfg["route_breadth_threshold"]),
                    fast_engine,
                )
            )
            per_pair_realistic[pair] = realistic_overlay_replay_from_context(
                pair_data["fast_context"],
                library_lookup,
                tuple(int(v) for v in cfg["mapping_indices"]),
                float(cfg["route_breadth_threshold"]),
            )
        baseline_fast[label] = aggregate_metrics(per_pair_fast)
        baseline_realistic[label] = {
            "start": start,
            "end": end,
            "bars": int(len(df)),
            "per_pair": per_pair_realistic,
            "aggregate": aggregate_metrics(per_pair_realistic),
        }
    baseline_seconds = perf_counter() - baseline_started
    baseline_stress_proxy_reserve = fast_stress_proxy_reserve(
        {
            label: {
                "per_pair": baseline_realistic[label]["per_pair"],
                "aggregate": baseline_fast[label],
            }
            for label, _, _ in DEFAULT_WINDOWS
        }
    )

    fast_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def evaluate_fast(individual: list[int]) -> tuple[float, ...]:
        key = tuple(int(v) for v in individual)
        cached = fast_cache.get(key)
        if cached is None:
            candidate = candidate_from_individual(individual, pairs, route_thresholds)
            windows = {}
            for label, _, _ in DEFAULT_WINDOWS:
                window_data = window_cache[label]
                per_pair = {}
                for pair in pairs:
                    cfg = candidate["pair_configs"][pair]
                    pair_data = window_data["pairs"][pair]
                    per_pair[pair] = summarize_single_result(
                        fast_overlay_replay_from_context(
                            pair_data["fast_context"],
                            library,
                            library_lookup,
                            tuple(int(v) for v in cfg["mapping_indices"]),
                            float(cfg["route_breadth_threshold"]),
                            fast_engine,
                        )
                    )
                windows[label] = {
                    "per_pair": per_pair,
                    "aggregate": aggregate_metrics(per_pair),
                }
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
            reserve = fast_stress_proxy_reserve(windows)
            validation_proxy = fast_validation_robustness_proxy(
                windows,
                target_daily_return=TARGET_060_DAILY_RETURN,
            )
            objectives = (
                float(reserve),
                float(recent_2m["worst_pair_avg_daily_return"]),
                float(recent_6m["worst_pair_avg_daily_return"]),
                float(full_4y["worst_pair_avg_daily_return"]),
                float(bnb_full_4y),
                float(validation_proxy),
                float(abs(recent_2m["worst_max_drawdown"])),
                float(abs(recent_6m["worst_max_drawdown"])),
                float(abs(full_4y["worst_max_drawdown"])),
            )
            cached = {
                "candidate": candidate,
                "windows": windows,
                "objectives": objectives,
                "scalar_score": fast_scalar_score(windows),
                "stress_proxy_reserve": reserve,
                "validation_robustness_proxy": validation_proxy,
            }
            fast_cache[key] = cached
        return cached["objectives"]

    ref_points = tools.uniform_reference_points(nobj=9, p=args.ref_p)
    toolbox = build_deap_toolbox(pairs, route_thresholds, subset_indices, rng, ref_points)
    toolbox.register("evaluate", evaluate_fast)

    population = toolbox.population(n=max(1, args.population - 1))
    baseline_individual = creator.IndividualPairSubsetPairwiseNSGA3StressAware(
        individual_from_pair_configs(baseline_pair_configs, pairs, route_thresholds)
    )
    population.append(baseline_individual)

    search_started = perf_counter()
    invalid = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    for _ in range(args.generations):
        offspring = algorithms.varOr(
            population,
            toolbox,
            lambda_=args.offspring,
            cxpb=args.cxpb,
            mutpb=args.mutpb,
        )
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        population = toolbox.select(population + offspring, k=args.population)
    search_seconds = perf_counter() - search_started

    unique_fast_candidates = {}
    for ind in population:
        unique_fast_candidates[tuple(int(v) for v in ind)] = fast_cache[tuple(int(v) for v in ind)]
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    for ind in pareto_front:
        unique_fast_candidates[tuple(int(v) for v in ind)] = fast_cache[tuple(int(v) for v in ind)]

    fast_candidates = list(unique_fast_candidates.values())
    feasible_fast = [
        item
        for item in fast_candidates
        if item["windows"]["recent_2m"]["aggregate"]["positive_pair_count"] == len(pairs)
        and item["windows"]["recent_6m"]["aggregate"]["positive_pair_count"] == len(pairs)
        and item["windows"]["full_4y"]["aggregate"]["positive_pair_count"] == len(pairs)
        and float(item.get("stress_proxy_reserve", float("-inf"))) >= baseline_stress_proxy_reserve * 0.95
    ]
    if not feasible_fast:
        feasible_fast = fast_candidates
    feasible_fast.sort(
        key=lambda item: (
            item.get("validation_robustness_proxy", float("-inf")),
            item["scalar_score"],
        ),
        reverse=True,
    )
    validation_candidate_count = min(
        len(feasible_fast),
        max(int(args.top_k_realistic), int(args.stress_proxy_candidate_count) * 3, 12),
    )
    validation_pool = [
        {
            "pair_configs": item["candidate"]["pair_configs"],
            "objectives": item["objectives"],
            "scalar_score": item["scalar_score"],
            "stress_proxy_reserve": item.get("stress_proxy_reserve"),
            "validation_robustness_proxy": item.get("validation_robustness_proxy"),
            "windows": item["windows"],
        }
        for item in feasible_fast[:validation_candidate_count]
    ]

    validation_frames_by_key: dict[str, Any] = {}
    validation_state_payloads: dict[str, Any] = {}
    validation_inputs_by_key: dict[str, Any] = {}
    for item in validation_pool:
        pair_configs = item["pair_configs"]
        candidate = {"pair_configs": pair_configs}
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

    for item in validation_pool:
        key = item["candidate_id"]
        item["validation_engine"] = build_candidate_validation_bundle(
            key,
            validation_inputs_by_key[key]["daily_returns"],
            validation_inputs_by_key[key]["daily_index"],
            trial_count=max(len(validation_pool), 1),
            peer_frames_by_key=validation_frames_by_key,
            state_payload=validation_state_payloads[key],
            cost_reference=build_candidate_cost_reference(item["windows"]),
        )
        item["validation_robustness"] = (
            item["validation_engine"].get("robustness")
            or build_validation_robustness_profile(item["validation_engine"])
        )
        item["early_validation_score"] = early_validation_aware_score(item)
    validation_pool.sort(key=early_validation_aware_score, reverse=True)

    top_fast = validation_pool[: max(int(args.top_k_realistic), 1)]

    realistic_top = []
    realistic_started = perf_counter()
    for item in top_fast:
        pair_configs = item["pair_configs"]
        windows = {}
        for label, start, end in DEFAULT_WINDOWS:
            window_data = window_cache[label]
            df = window_data["df"]
            per_pair = {}
            for pair in pairs:
                cfg = pair_configs[pair]
                pair_data = window_data["pairs"][pair]
                per_pair[pair] = realistic_overlay_replay_from_context(
                    pair_data["fast_context"],
                    library_lookup,
                    tuple(int(v) for v in cfg["mapping_indices"]),
                    float(cfg["route_breadth_threshold"]),
                )
            windows[label] = {
                "start": start,
                "end": end,
                "bars": int(len(df)),
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
        key = item["candidate_id"]
        validation_input = validation_inputs_by_key[key]
        validation_engine = build_candidate_validation_bundle(
            key,
            validation_input["daily_returns"],
            validation_input["daily_index"],
            trial_count=max(len(top_fast), 1),
            peer_frames_by_key=validation_frames_by_key,
            state_payload=validation_state_payloads[key],
            cost_reference=build_candidate_cost_reference(windows),
        )
        realistic_top.append(
            {
                "pair_configs": pair_configs,
                "candidate_id": key,
                "objectives": item["objectives"],
                "scalar_score": item["scalar_score"],
                "score": item["scalar_score"],
                "stress_proxy_reserve": item.get("stress_proxy_reserve"),
                "validation_robustness_proxy": item.get("validation_robustness_proxy"),
                "early_validation_score": item.get("early_validation_score"),
                "windows": windows,
                "validation": build_validation_bundle(windows, baseline_realistic),
                "validation_engine": validation_engine,
                "validation_robustness": (
                    validation_engine.get("robustness")
                    or build_validation_robustness_profile(validation_engine)
                ),
            }
        )
    realistic_seconds = perf_counter() - realistic_started

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
            target_daily_return=TARGET_060_DAILY_RETURN,
            seed_offset=700000 + stress_idx * 1000,
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
        item["stress_aware_score"] = realistic_stress_aware_score(item)
    realistic_top.sort(key=realistic_stress_aware_score, reverse=True)

    validation_pass_candidates = [
        item for item in realistic_top if bool((item.get("validation_engine") or {}).get("gate", {}).get("passed", False))
    ]
    stress_proxy_validation_candidates = [
        item for item in validation_pass_candidates if bool((item.get("stress_proxy") or {}).get("passed", False))
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
    fallback_best = max(realistic_top, key=realistic_stress_aware_score) if realistic_top else None
    selected = max(stress_proxy_validation_candidates, key=realistic_stress_aware_score) if stress_proxy_validation_candidates else None
    selection_reason = "stress_proxy_plus_validation"
    if selected is None and target_060_candidates:
        selected = max(target_060_candidates, key=realistic_stress_aware_score)
        selection_reason = "target_060_plus_validation"
    if selected is None and progressive_candidates:
        selected = max(progressive_candidates, key=realistic_stress_aware_score)
        selection_reason = "progressive_plus_validation"
    if selected is None and validation_pass_candidates:
        selected = max(validation_pass_candidates, key=realistic_stress_aware_score)
        selection_reason = "validation_only"
    if selected is None:
        selection_reason = "no_gate_pass"

    report = {
        "search": {
            "algorithm": "pairwise_nsga3",
            "population": args.population,
            "generations": args.generations,
            "offspring": args.offspring,
            "seed": args.seed,
            "library_source": args.library_source,
            "library_size": len(library),
            "route_thresholds": list(route_thresholds),
            "subset_indices": list(subset_indices),
            "ref_points": len(ref_points),
            "validation_candidate_count": int(validation_candidate_count),
        },
        "pairs": list(pairs),
        "model_path": str(args.model),
        "baseline_summary_path": str(args.baseline_summary),
        "base_summary_path": str(args.base_summary),
        "baseline_candidate": {
            "pair_configs": baseline_pair_configs,
        },
        "baseline_fast": baseline_fast,
        "baseline_realistic": baseline_realistic,
        "pareto_front": [
            {
                "candidate": fast_cache[tuple(int(v) for v in ind)]["candidate"],
                "objectives": list(fast_cache[tuple(int(v) for v in ind)]["objectives"]),
                "scalar_score": float(fast_cache[tuple(int(v) for v in ind)]["scalar_score"]),
                "stress_proxy_reserve": float(fast_cache[tuple(int(v) for v in ind)]["stress_proxy_reserve"]),
                "windows": fast_cache[tuple(int(v) for v in ind)]["windows"],
            }
            for ind in pareto_front
        ],
        "top_fast_candidates": top_fast,
        "realistic_top_candidates": realistic_top,
        "promotion_candidates": {
            "stress_proxy_validation": [{"pair_configs": item["pair_configs"]} for item in stress_proxy_validation_candidates],
            "target_060": [{"pair_configs": item["pair_configs"]} for item in target_060_candidates],
            "progressive_improvement": [{"pair_configs": item["pair_configs"]} for item in progressive_candidates],
        },
        "selection": {
            "reason": selection_reason,
            "validation_pass_count": len(validation_pass_candidates),
            "stress_proxy_pass_count": stress_proxy_pass_count,
            "stress_proxy_validation_pass_count": len(stress_proxy_validation_candidates),
            "target_060_pass_count": len(target_060_candidates),
            "progressive_pass_count": len(progressive_candidates),
            "realistic_top_count": len(realistic_top),
        },
        "runtime": {
            "fast_engine": fast_engine,
            "numba_available": NUMBA_AVAILABLE,
            "prepare_context_seconds": prepare_seconds,
            "baseline_seconds": baseline_seconds,
            "search_seconds": search_seconds,
            "realistic_seconds": realistic_seconds,
            "total_seconds": perf_counter() - started,
            "evaluated_unique_candidates": len(fast_cache),
        },
        "selected_candidate": selected,
        "fallback_best_candidate": fallback_best,
        "created_at": datetime.now(UTC).isoformat(),
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
