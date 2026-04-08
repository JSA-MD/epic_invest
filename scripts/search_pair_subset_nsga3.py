#!/usr/bin/env python3
"""NSGA-III search for BTC/BNB subset-pair regime-mixture overlays."""

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
from replay_regime_mixture_realistic import load_model, resolve_candidate
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    fast_overlay_replay_from_context,
    json_safe,
    load_or_fetch_funding,
    parse_csv_tuple,
    realistic_overlay_replay,
    resolve_fast_engine,
    score_realistic_candidate,
    summarize_single_result,
)
from validate_pair_subset_summary import build_validation_bundle


UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NSGA-III search for subset-pair regime-mixture candidates.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_nsga3_summary.json"),
    )
    parser.add_argument(
        "--subset-indices",
        default="0,1,2,3,4,5,6,7",
        help="Library indices allowed in searched mappings.",
    )
    parser.add_argument(
        "--route-thresholds",
        default="0.35,0.50,0.65,0.80",
        help="Comma-separated route breadth thresholds to test.",
    )
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--generations", type=int, default=28)
    parser.add_argument("--offspring", type=int, default=96)
    parser.add_argument("--top-k-realistic", type=int, default=8)
    parser.add_argument("--ref-p", type=int, default=2, help="Reference point granularity for NSGA-III.")
    parser.add_argument("--cxpb", type=float, default=0.65)
    parser.add_argument("--mutpb", type=float, default=0.35)
    parser.add_argument("--gene-mutpb", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fast-engine",
        choices=("auto", "python", "numba"),
        default="auto",
        help="Fast replay engine for candidate evaluation.",
    )
    return parser.parse_args()


def fast_scalar_score(windows: dict[str, Any]) -> float:
    recent_2m = windows["recent_2m"]
    recent_6m = windows["recent_6m"]
    full_4y = windows["full_4y"]
    score = 0.0
    score += float(recent_2m["worst_pair_avg_daily_return"]) * 420000.0
    score += float(recent_6m["worst_pair_avg_daily_return"]) * 320000.0
    score += float(full_4y["worst_pair_avg_daily_return"]) * 240000.0
    score += float(full_4y["mean_avg_daily_return"]) * 180000.0
    score -= abs(float(recent_2m["worst_max_drawdown"])) * 18000.0
    score -= abs(float(recent_6m["worst_max_drawdown"])) * 14000.0
    score -= abs(float(full_4y["worst_max_drawdown"])) * 9000.0
    score -= float(recent_6m["pair_return_dispersion"]) * 90000.0
    score -= float(full_4y["pair_return_dispersion"]) * 60000.0
    return float(score)


def candidate_from_individual(individual: list[int], route_thresholds: tuple[float, ...]) -> dict[str, Any]:
    return {
        "route_breadth_threshold": float(route_thresholds[int(individual[0])]),
        "mapping_indices": [int(v) for v in individual[1:]],
    }


def build_deap_toolbox(
    route_thresholds: tuple[float, ...],
    subset_indices: tuple[int, ...],
    rng: random.Random,
    ref_points: Any,
) -> base.Toolbox:
    if not hasattr(creator, "FitnessPairSubsetNSGA3"):
        creator.create(
            "FitnessPairSubsetNSGA3",
            base.Fitness,
            weights=(1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0),
        )
    if not hasattr(creator, "IndividualPairSubsetNSGA3"):
        creator.create("IndividualPairSubsetNSGA3", list, fitness=creator.FitnessPairSubsetNSGA3)

    toolbox = base.Toolbox()

    def random_threshold_gene() -> int:
        return rng.randrange(len(route_thresholds))

    def random_mapping_gene() -> int:
        return int(rng.choice(subset_indices))

    def init_individual() -> creator.IndividualPairSubsetNSGA3:
        return creator.IndividualPairSubsetNSGA3(
            [random_threshold_gene()] + [random_mapping_gene() for _ in range(4)]
        )

    def mate(ind1: list[int], ind2: list[int]) -> tuple[list[int], list[int]]:
        tools.cxUniform(ind1, ind2, indpb=0.5)
        return ind1, ind2

    def mutate(individual: list[int], indpb: float) -> tuple[list[int]]:
        if rng.random() < indpb:
            individual[0] = random_threshold_gene()
        for idx in range(1, 5):
            if rng.random() < indpb:
                individual[idx] = random_mapping_gene()
        return (individual,)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate, indpb=0.25)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    toolbox.register("clone", copy.deepcopy)
    return toolbox


def main() -> None:
    args = parse_args()
    started = perf_counter()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    pairs = parse_csv_tuple(args.pairs, str)
    subset_indices = parse_csv_tuple(args.subset_indices, int)
    route_thresholds = parse_csv_tuple(args.route_thresholds, float)
    fast_engine = resolve_fast_engine(args.fast_engine)

    baseline_candidate, library, _ = resolve_candidate(Path(args.summary), None, None)
    if baseline_candidate.route_breadth_threshold not in route_thresholds:
        route_thresholds = tuple(sorted(set(route_thresholds + (baseline_candidate.route_breadth_threshold,))))

    library_lookup = build_library_lookup(library)
    model, _ = load_model(Path(args.model))
    compiled = gp.toolbox.compile(expr=model)

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
                "overlay_inputs": overlay_inputs,
                "signal": signal_slice,
                "funding": funding_slice,
                "fast_context": build_fast_context(
                    df=df,
                    pair=pair,
                    raw_signal=signal_slice,
                    overlay_inputs=overlay_inputs,
                    route_thresholds=route_thresholds,
                    library_lookup=library_lookup,
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
            pair_data = window_data["pairs"][pair]
            per_pair_fast[pair] = summarize_single_result(
                fast_overlay_replay_from_context(
                    pair_data["fast_context"],
                    library,
                    library_lookup,
                    baseline_candidate.mapping_indices,
                    baseline_candidate.route_breadth_threshold,
                    fast_engine,
                )
            )
            per_pair_realistic[pair] = realistic_overlay_replay(
                df,
                pair,
                pair_data["signal"],
                pair_data["overlay_inputs"],
                pair_data["funding"],
                library,
                baseline_candidate.mapping_indices,
                baseline_candidate.route_breadth_threshold,
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

    fast_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def evaluate_fast(individual: list[int]) -> tuple[float, ...]:
        key = tuple(int(v) for v in individual)
        cached = fast_cache.get(key)
        if cached is None:
            route_threshold = float(route_thresholds[int(individual[0])])
            mapping = tuple(int(v) for v in individual[1:])
            windows = {}
            for label, _, _ in DEFAULT_WINDOWS:
                window_data = window_cache[label]
                per_pair = {}
                for pair in pairs:
                    pair_data = window_data["pairs"][pair]
                    per_pair[pair] = summarize_single_result(
                        fast_overlay_replay_from_context(
                            pair_data["fast_context"],
                            library,
                            library_lookup,
                            mapping,
                            route_threshold,
                            fast_engine,
                        )
                    )
                windows[label] = aggregate_metrics(per_pair)

            objectives = (
                float(windows["recent_2m"]["worst_pair_avg_daily_return"]),
                float(windows["recent_6m"]["worst_pair_avg_daily_return"]),
                float(windows["full_4y"]["worst_pair_avg_daily_return"]),
                float(windows["full_4y"]["mean_avg_daily_return"]),
                float(abs(windows["recent_2m"]["worst_max_drawdown"])),
                float(abs(windows["recent_6m"]["worst_max_drawdown"])),
                float(abs(windows["full_4y"]["worst_max_drawdown"])),
            )
            cached = {
                "candidate": candidate_from_individual(individual, route_thresholds),
                "windows": windows,
                "objectives": objectives,
                "scalar_score": fast_scalar_score(windows),
            }
            fast_cache[key] = cached
        return cached["objectives"]

    ref_points = tools.uniform_reference_points(nobj=7, p=args.ref_p)
    toolbox = build_deap_toolbox(route_thresholds, subset_indices, rng, ref_points)
    toolbox.register("evaluate", evaluate_fast)

    population = toolbox.population(n=max(1, args.population - 1))
    baseline_individual = creator.IndividualPairSubsetNSGA3(
        [route_thresholds.index(baseline_candidate.route_breadth_threshold), *baseline_candidate.mapping_indices]
    )
    population.append(baseline_individual)

    search_started = perf_counter()
    invalid = [ind for ind in population if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit

    for _ in range(args.generations):
        offspring = algorithms.varOr(population, toolbox, lambda_=args.offspring, cxpb=args.cxpb, mutpb=args.mutpb)
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
        if item["windows"]["recent_2m"]["positive_pair_count"] == len(pairs)
        and item["windows"]["recent_6m"]["positive_pair_count"] == len(pairs)
        and item["windows"]["full_4y"]["positive_pair_count"] == len(pairs)
    ]
    if not feasible_fast:
        feasible_fast = fast_candidates

    feasible_fast.sort(key=lambda item: item["scalar_score"], reverse=True)
    top_fast = [
        {
            "route_breadth_threshold": item["candidate"]["route_breadth_threshold"],
            "mapping_indices": item["candidate"]["mapping_indices"],
            "objectives": item["objectives"],
            "scalar_score": item["scalar_score"],
            "windows": item["windows"],
        }
        for item in feasible_fast[: args.top_k_realistic]
    ]

    realistic_top = []
    realistic_started = perf_counter()
    for item in top_fast:
        route_threshold = float(item["route_breadth_threshold"])
        mapping = tuple(int(v) for v in item["mapping_indices"])
        windows = {}
        for label, start, end in DEFAULT_WINDOWS:
            window_data = window_cache[label]
            df = window_data["df"]
            per_pair = {}
            for pair in pairs:
                pair_data = window_data["pairs"][pair]
                per_pair[pair] = realistic_overlay_replay(
                    df,
                    pair,
                    pair_data["signal"],
                    pair_data["overlay_inputs"],
                    pair_data["funding"],
                    library,
                    mapping,
                    route_threshold,
                )
            windows[label] = {
                "start": start,
                "end": end,
                "bars": int(len(df)),
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
        realistic_top.append(
            {
                "route_breadth_threshold": route_threshold,
                "mapping_indices": list(mapping),
                "objectives": item["objectives"],
                "scalar_score": item["scalar_score"],
                "windows": windows,
                "validation": build_validation_bundle(windows, baseline_realistic),
            }
        )
    realistic_seconds = perf_counter() - realistic_started

    progressive_candidates = [
        item
        for item in realistic_top
        if item["validation"]["profiles"]["progressive_improvement"]["passed"]
    ]
    target_060_candidates = [
        item
        for item in realistic_top
        if item["validation"]["profiles"]["target_060"]["passed"]
    ]
    fallback_best = max(realistic_top, key=score_realistic_candidate) if realistic_top else None
    selected = max(target_060_candidates, key=score_realistic_candidate) if target_060_candidates else None
    selection_reason = "target_060_pass"
    if selected is None and progressive_candidates:
        selected = max(progressive_candidates, key=score_realistic_candidate)
        selection_reason = "progressive_pass"
    if selected is None:
        selection_reason = "no_gate_pass"

    report = {
        "search": {
            "algorithm": "nsga3",
            "population": args.population,
            "generations": args.generations,
            "offspring": args.offspring,
            "seed": args.seed,
            "route_thresholds": list(route_thresholds),
            "subset_indices": list(subset_indices),
            "ref_points": len(ref_points),
        },
        "pairs": list(pairs),
        "baseline_candidate": {
            "route_breadth_threshold": baseline_candidate.route_breadth_threshold,
            "mapping_indices": list(baseline_candidate.mapping_indices),
        },
        "baseline_fast": baseline_fast,
        "baseline_realistic": baseline_realistic,
        "pareto_front": [
            {
                "candidate": candidate_from_individual(ind, route_thresholds),
                "objectives": list(fast_cache[tuple(int(v) for v in ind)]["objectives"]),
                "scalar_score": float(fast_cache[tuple(int(v) for v in ind)]["scalar_score"]),
                "windows": fast_cache[tuple(int(v) for v in ind)]["windows"],
            }
            for ind in pareto_front
        ],
        "top_fast_candidates": top_fast,
        "realistic_top_candidates": realistic_top,
        "promotion_candidates": {
            "target_060": [
                {
                    "route_breadth_threshold": item["route_breadth_threshold"],
                    "mapping_indices": item["mapping_indices"],
                }
                for item in target_060_candidates
            ],
            "progressive_improvement": [
                {
                    "route_breadth_threshold": item["route_breadth_threshold"],
                    "mapping_indices": item["mapping_indices"],
                }
                for item in progressive_candidates
            ],
        },
        "selection": {
            "reason": selection_reason,
            "target_060_pass_count": len(target_060_candidates),
            "progressive_pass_count": len(progressive_candidates),
            "realistic_top_count": len(realistic_top),
        },
        "runtime": {
            "fast_engine": fast_engine,
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
