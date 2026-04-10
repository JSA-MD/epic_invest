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
from replay_regime_mixture_realistic import load_model
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
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"),
    )
    parser.add_argument("--random-mutations", type=int, default=6000)
    parser.add_argument("--top-realistic", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument(
        "--route-state-mode",
        choices=("base", "equity_corr"),
        default="base",
    )
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


def candidate_score(
    windows: dict[str, Any],
    base_2m: float,
    base_6m: float,
    base_4y: float,
    base_4y_mean: float,
) -> float:
    recent_2m = windows["recent_2m"]
    recent_6m = windows["recent_6m"]
    full_4y = windows["full_4y"]
    score = 0.0
    score += (float(recent_2m["worst_pair_avg_daily_return"]) - base_2m) * 8.0
    score += (float(recent_6m["worst_pair_avg_daily_return"]) - base_6m) * 12.0
    score += (float(full_4y["worst_pair_avg_daily_return"]) - base_4y) * 6.0
    score += (float(full_4y["mean_avg_daily_return"]) - base_4y_mean) * 6.0
    score -= abs(float(recent_2m["worst_max_drawdown"])) * 0.35
    score -= abs(float(recent_6m["worst_max_drawdown"])) * 0.45
    score -= abs(float(full_4y["worst_max_drawdown"])) * 0.30
    return float(score)


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
            windows[label] = aggregate_metrics(per_pair)
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
    top_pair_configs: list[dict[str, Any]] = []
    for raw_path in candidate_summary_paths:
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
                    {k: v["aggregate"] for k, v in windows.items()},
                    base_2m,
                    base_6m,
                    base_4y,
                    base_4y_mean,
                ),
            }
        )
    start_eval.sort(key=lambda item: item["score"], reverse=True)
    chosen_start = start_eval[0]

    gene_pool = {
        pair: {
            "thresholds": sorted({float(item["pair_configs"][pair]["route_breadth_threshold"]) for item in start_eval[:12]} | set(route_thresholds)),
            "mappings": [
                sorted({int(item["pair_configs"][pair]["mapping_indices"][bucket]) for item in start_eval[:20]})
                for bucket in range(route_gene_count)
            ],
        }
        for pair in pairs
    }

    candidates = [clone_candidate(chosen_start)]
    for pair in pairs:
        for gene in range(route_gene_count + 1):
            if gene == 0:
                for value in gene_pool[pair]["thresholds"]:
                    candidate = {"pair_configs": clone_candidate(chosen_start)["pair_configs"]}
                    candidate["pair_configs"][pair]["route_breadth_threshold"] = float(value)
                    candidates.append(candidate)
            else:
                for value in gene_pool[pair]["mappings"][gene - 1]:
                    candidate = {"pair_configs": clone_candidate(chosen_start)["pair_configs"]}
                    candidate["pair_configs"][pair]["mapping_indices"][gene - 1] = int(value)
                    candidates.append(candidate)

    for _ in range(args.random_mutations):
        candidate = {"pair_configs": clone_candidate(chosen_start)["pair_configs"]}
        steps = random.choice((1, 1, 1, 2, 2, 3))
        for _ in range(steps):
            pair = random.choice(pairs)
            gene = random.randrange(route_gene_count + 1)
            if gene == 0:
                candidate["pair_configs"][pair]["route_breadth_threshold"] = float(random.choice(route_thresholds))
            else:
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
    for candidate in candidates:
        key = candidate_key(candidate, pairs)
        if key in seen:
            continue
        seen.add(key)
        windows = eval_fast(candidate)
        score = candidate_score(windows, base_2m, base_6m, base_4y, base_4y_mean)
        if (
            windows["recent_2m"]["worst_pair_avg_daily_return"] >= base_2m * 0.97
            and windows["full_4y"]["mean_avg_daily_return"] >= base_4y_mean * 0.995
        ):
            fast_ranked.append(
                {
                    "pair_configs": candidate["pair_configs"],
                    "windows": windows,
                    "score": score,
                }
            )
    fast_ranked.sort(key=lambda item: item["score"], reverse=True)

    realistic_top = []
    for item in fast_ranked[: args.top_realistic]:
        windows = eval_realistic({"pair_configs": item["pair_configs"]})
        realistic_top.append(
            {
                "pair_configs": item["pair_configs"],
                "windows": windows,
                "score": candidate_score(
                    {k: v["aggregate"] for k, v in windows.items()},
                    base_2m,
                    base_6m,
                    base_4y,
                    base_4y_mean,
                ),
                "validation": build_validation_bundle(windows, baseline_summary["selected_candidate"]["windows"]),
            }
        )
    realistic_top.sort(key=lambda item: item["score"], reverse=True)
    selected = realistic_top[0]

    report = {
        "search": {
            "algorithm": "pairwise_local_repair",
            "seed": args.seed,
            "random_mutations": args.random_mutations,
            "top_realistic": args.top_realistic,
            "candidate_summary_paths": list(candidate_summary_paths),
            "library_source": "full-grid",
            "library_size": len(library),
            "route_state_mode": route_state_mode,
            "route_state_count": route_gene_count,
        },
        "pairs": list(pairs),
        "baseline_summary_path": str(args.baseline_summary),
        "baseline_candidate": {"pair_configs": baseline_configs},
        "baseline_realistic": baseline_summary["selected_candidate"]["windows"],
        "chosen_start_candidate": chosen_start,
        "runtime": {
            "prepare_context_seconds": prepare_seconds,
            "total_seconds": perf_counter() - started,
            "start_candidate_count": len(start_candidates),
            "repair_candidate_count": len(seen),
            "fast_ranked_count": len(fast_ranked),
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
