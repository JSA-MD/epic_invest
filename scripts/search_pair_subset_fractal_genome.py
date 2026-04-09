#!/usr/bin/env python3
"""First-stage fractal genome search over recursive If-Then-Else trees."""

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

import gp_crypto_evolution as gp
from fractal_genome_core import (
    ConditionNode,
    ConditionSpec,
    FilterDecision,
    LeafNode,
    TreeNode,
    build_llm_prompt,
    collect_specs,
    crossover_tree,
    evaluate_tree_codes,
    load_llm_review_map,
    mutate_tree,
    random_tree,
    semantic_filter,
    serialize_tree,
    tree_depth,
    tree_key,
    tree_size,
)
from replay_regime_mixture_realistic import load_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    BAR_FACTOR,
    BARS_PER_DAY,
    DEFAULT_WINDOWS,
    NUMBA_AVAILABLE,
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    json_safe,
    load_or_fetch_funding,
    parse_csv_tuple,
    resolve_fast_engine,
    score_realistic_candidate,
    summarize_single_result,
)
from validate_pair_subset_summary import build_validation_bundle

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


UTC = timezone.utc

FEATURE_SPECS: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    ("btc_regime", ">=", (-0.05, 0.0, 0.05, 0.10)),
    ("bnb_regime", ">=", (-0.05, 0.0, 0.05, 0.10)),
    ("breadth", ">=", (0.35, 0.50, 0.65, 0.80)),
    ("btc_vol_rel", "<=", (0.80, 1.00, 1.20)),
    ("bnb_vol_rel", "<=", (0.80, 1.00, 1.20)),
    ("btc_momentum_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("bnb_momentum_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("rel_strength_bnb_btc_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search recursive fractal-genome trees that route BTC/BNB expert overlays.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--expert-summaries",
        default="models/gp_regime_mixture_btc_bnb_pairwise_repair_summary.json,models/gp_regime_mixture_btc_bnb_pairwise_fullgrid_seed_pool.json",
    )
    parser.add_argument(
        "--baseline-summary",
        default="models/gp_regime_mixture_btc_bnb_pairwise_repair_summary.json",
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default="models/gp_regime_mixture_btc_bnb_fractal_genome_summary.json",
    )
    parser.add_argument("--expert-pool-size", type=int, default=18)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument(
        "--route-thresholds",
        default="0.35,0.50,0.65,0.80",
    )
    parser.add_argument(
        "--fast-engine",
        choices=("auto", "python", "numba"),
        default="auto",
    )
    parser.add_argument(
        "--filter-mode",
        choices=("auto", "heuristic", "llm-first", "llm-only"),
        default="auto",
    )
    parser.add_argument(
        "--llm-review-in",
        default=None,
        help="Optional JSONL file with precomputed LLM decisions keyed by tree_key.",
    )
    parser.add_argument(
        "--llm-review-out",
        default=None,
        help="Optional JSONL file to export review prompts for top candidates.",
    )
    return parser.parse_args()


def build_condition_options() -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for feature, comparator, thresholds in FEATURE_SPECS:
        for threshold in thresholds:
            out.append(ConditionSpec(feature=feature, comparator=comparator, threshold=float(threshold), invert=False))
            out.append(ConditionSpec(feature=feature, comparator=comparator, threshold=float(threshold), invert=True))
    return out


def build_market_features(df: pd.DataFrame, pairs: tuple[str, ...]) -> dict[str, pd.Series]:
    close = pd.concat([df[f"{asset}_close"].rename(asset) for asset in pairs], axis=1).sort_index()
    daily_close = close.resample("1D").last().dropna()
    btc_regime = 0.60 * daily_close["BTCUSDT"].pct_change(3) + 0.40 * daily_close["BTCUSDT"].pct_change(14)
    bnb_regime = 0.60 * daily_close["BNBUSDT"].pct_change(3) + 0.40 * daily_close["BNBUSDT"].pct_change(14)
    breadth = (daily_close.pct_change(3) > 0.0).mean(axis=1)
    btc_momentum_3d = daily_close["BTCUSDT"].pct_change(3)
    bnb_momentum_3d = daily_close["BNBUSDT"].pct_change(3)
    rel_strength = bnb_momentum_3d - btc_momentum_3d

    btc_vol_ann = close["BTCUSDT"].pct_change().rolling(12 * 24 * 3).std() * BAR_FACTOR
    bnb_vol_ann = close["BNBUSDT"].pct_change().rolling(12 * 24 * 3).std() * BAR_FACTOR
    btc_vol_rel = btc_vol_ann / btc_vol_ann.rolling(12 * 24 * 7).median()
    bnb_vol_rel = bnb_vol_ann / bnb_vol_ann.rolling(12 * 24 * 7).median()

    return {
        "btc_regime": btc_regime,
        "bnb_regime": bnb_regime,
        "breadth": breadth,
        "btc_momentum_3d": btc_momentum_3d,
        "bnb_momentum_3d": bnb_momentum_3d,
        "rel_strength_bnb_btc_3d": rel_strength,
        "btc_vol_rel": btc_vol_rel,
        "bnb_vol_rel": bnb_vol_rel,
    }


def materialize_feature_arrays(features: dict[str, pd.Series], index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    day_index = index.normalize()
    out: dict[str, np.ndarray] = {}
    for name, series in features.items():
        if name in {"btc_vol_rel", "bnb_vol_rel"}:
            values = series.reindex(index).ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(1.0)
        else:
            values = series.reindex(day_index, method="ffill").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[name] = values.to_numpy(dtype="float64")
    return out


def build_expert_pool(summary_paths: list[str], pool_size: int) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for raw_path in summary_paths:
        obj = json.loads(Path(raw_path).read_text())
        candidates: list[dict[str, Any]] = []
        if obj.get("selected_candidate") and obj["selected_candidate"].get("pair_configs"):
            candidates.append(obj["selected_candidate"])
        candidates.extend(obj.get("realistic_top_candidates", []))
        for candidate in candidates:
            if not candidate.get("pair_configs"):
                continue
            key = json.dumps(candidate["pair_configs"], sort_keys=True)
            windows = candidate.get("windows")
            score = score_realistic_candidate({"windows": windows}) if windows else float(candidate.get("score", 0.0))
            existing = by_key.get(key)
            if existing is None or score > existing["score"]:
                by_key[key] = {
                    "pair_configs": candidate["pair_configs"],
                    "windows": windows,
                    "score": float(score),
                }
    experts = list(by_key.values())
    experts.sort(key=lambda item: item["score"], reverse=True)
    return experts[:pool_size]


def expert_arrays_for_pair(
    pair: str,
    expert_pool: list[dict[str, Any]],
    route_thresholds: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    threshold_idx = []
    mapping = []
    for expert in expert_pool:
        cfg = expert["pair_configs"][pair]
        threshold_idx.append(route_thresholds.index(float(cfg["route_breadth_threshold"])))
        mapping.append([int(v) for v in cfg["mapping_indices"]])
    return np.asarray(threshold_idx, dtype="int16"), np.asarray(mapping, dtype="int64")


def _fractal_fast_kernel_impl(
    close: np.ndarray,
    bucket_codes_matrix: np.ndarray,
    regime: np.ndarray,
    breadth: np.ndarray,
    vol_ann: np.ndarray,
    smooth_signal_matrix: np.ndarray,
    library_signal_pos: np.ndarray,
    library_rebalance_bars: np.ndarray,
    library_regime_threshold: np.ndarray,
    library_breadth_threshold: np.ndarray,
    library_target_vol_ann: np.ndarray,
    library_gross_cap: np.ndarray,
    library_kill_switch_pct: np.ndarray,
    library_cooldown_days: np.ndarray,
    expert_threshold_idx: np.ndarray,
    expert_mapping: np.ndarray,
    expert_codes: np.ndarray,
    initial_cash: float,
    commission_pct: float,
    no_trade_band_pct: float,
    bars_per_day: int,
    daily_target: float,
    bar_factor: float,
) -> tuple[float, int, float, float, float, float, float, float, float, float]:
    equity = initial_cash
    peak_equity = initial_cash
    current_weight = 0.0
    cooldown_bars_left = 0
    n_trades = 0
    mean_bar = 0.0
    m2_bar = 0.0
    bar_count = 0
    day_accum = 1.0
    day_len = 0
    day_count = 0
    day_sum = 0.0
    day_wins = 0
    day_hits = 0
    worst_day = 0.0
    best_day = 0.0
    max_drawdown = 0.0

    for i in range(close.shape[0] - 1):
        expert = expert_codes[i]
        threshold_idx = expert_threshold_idx[expert]
        bucket = bucket_codes_matrix[threshold_idx, i]
        active_idx = expert_mapping[expert, bucket]

        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = smooth_signal_matrix[library_signal_pos[active_idx], i]
        signal_pct = min(500.0, max(-500.0, signal_pct))
        requested_weight = signal_pct / 100.0

        regime_score = regime[i]
        breadth_score = breadth[i]
        long_ok = regime_score >= library_regime_threshold[active_idx] and breadth_score >= library_breadth_threshold[active_idx]
        short_ok = regime_score <= -library_regime_threshold[active_idx] and breadth_score <= (1.0 - library_breadth_threshold[active_idx])
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = vol_ann[i]
        if bar_vol_ann == bar_vol_ann and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = library_target_vol_ann[active_idx] / bar_vol_ann
            gross_scale = library_gross_cap[active_idx] / max(abs(requested_weight), 1e-8)
            if gross_scale < vol_scale:
                vol_scale = gross_scale
            requested_weight *= vol_scale

        if requested_weight > library_gross_cap[active_idx]:
            requested_weight = library_gross_cap[active_idx]
        elif requested_weight < -library_gross_cap[active_idx]:
            requested_weight = -library_gross_cap[active_idx]

        drawdown = equity / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -library_kill_switch_pct[active_idx] and cooldown_bars_left == 0:
            cooldown_bars_left = library_cooldown_days[active_idx] * bars_per_day

        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif i % library_rebalance_bars[active_idx] == 0:
            target_weight = requested_weight

        if abs(target_weight - current_weight) < no_trade_band_pct / 100.0:
            target_weight = current_weight

        turnover = abs(target_weight - current_weight)
        if turnover > 0.001:
            n_trades += 1

        price_ret = close[i + 1] / close[i] - 1.0
        bar_net = target_weight * price_ret - turnover * commission_pct * 2.0
        equity *= 1.0 + bar_net
        if equity > peak_equity:
            peak_equity = equity
        current_weight = target_weight
        dd = equity / peak_equity - 1.0
        if dd < max_drawdown:
            max_drawdown = dd

        bar_count += 1
        delta = bar_net - mean_bar
        mean_bar += delta / bar_count
        m2_bar += delta * (bar_net - mean_bar)

        day_accum *= 1.0 + bar_net
        day_len += 1
        if day_len == bars_per_day or i == close.shape[0] - 2:
            day_ret = day_accum - 1.0
            day_sum += day_ret
            day_count += 1
            if day_ret > 0.0:
                day_wins += 1
            if day_ret >= daily_target:
                day_hits += 1
            if day_count == 1 or day_ret < worst_day:
                worst_day = day_ret
            if day_count == 1 or day_ret > best_day:
                best_day = day_ret
            day_accum = 1.0
            day_len = 0

    sharpe = 0.0
    if bar_count > 1:
        variance = m2_bar / bar_count
        if variance > 1e-12:
            sharpe = mean_bar / np.sqrt(variance) * bar_factor

    avg_daily = 0.0 if day_count == 0 else day_sum / day_count
    daily_target_hit_rate = 0.0 if day_count == 0 else day_hits / day_count
    daily_win_rate = 0.0 if day_count == 0 else day_wins / day_count
    return (
        equity / initial_cash - 1.0,
        n_trades,
        sharpe,
        max_drawdown,
        equity,
        avg_daily,
        daily_target_hit_rate,
        daily_win_rate,
        worst_day,
        best_day,
    )


if NUMBA_AVAILABLE:
    _fractal_fast_kernel = njit(cache=True)(_fractal_fast_kernel_impl)
else:  # pragma: no cover
    _fractal_fast_kernel = _fractal_fast_kernel_impl


def fast_fractal_replay_from_context(
    context: dict[str, Any],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
    expert_threshold_idx: np.ndarray,
    expert_mapping: np.ndarray,
    expert_codes: np.ndarray,
) -> dict[str, Any]:
    bucket_codes_matrix = np.vstack([context["bucket_codes"][float(th)] for th in route_thresholds]).astype("int8")
    result = _fractal_fast_kernel(
        context["close"],
        bucket_codes_matrix,
        context["regime"],
        context["breadth"],
        context["vol_ann"],
        context["smooth_signal_matrix"],
        library_lookup["signal_pos"],
        library_lookup["rebalance_bars"],
        library_lookup["regime_threshold"],
        library_lookup["breadth_threshold"],
        library_lookup["target_vol_ann"],
        library_lookup["gross_cap"],
        library_lookup["kill_switch_pct"],
        library_lookup["cooldown_days"],
        expert_threshold_idx,
        expert_mapping,
        expert_codes.astype("int16"),
        float(gp.INITIAL_CASH),
        float(gp.COMMISSION_PCT),
        float(gp.NO_TRADE_BAND),
        int(BARS_PER_DAY),
        float(gp.DAILY_TARGET_PCT),
        float(BAR_FACTOR),
    )
    return summarize_single_result(
        {
            "total_return": float(result[0]),
            "n_trades": int(result[1]),
            "sharpe": float(result[2]),
            "max_drawdown": float(result[3]),
            "final_equity": float(result[4]),
            "daily_metrics": {
                "avg_daily_return": float(result[5]),
                "daily_target_hit_rate": float(result[6]),
                "daily_win_rate": float(result[7]),
                "worst_day": float(result[8]),
                "best_day": float(result[9]),
            },
        }
    )


def fractal_fast_scalar_score(windows: dict[str, Any], filter_decision: FilterDecision, node: TreeNode) -> float:
    agg_2m = windows["recent_2m"]["aggregate"]
    agg_6m = windows["recent_6m"]["aggregate"]
    agg_4y = windows["full_4y"]["aggregate"]
    score = 0.0
    score += float(agg_2m["worst_pair_avg_daily_return"]) * 420000.0
    score += float(agg_6m["worst_pair_avg_daily_return"]) * 320000.0
    score += float(agg_4y["worst_pair_avg_daily_return"]) * 260000.0
    score += float(agg_4y["mean_avg_daily_return"]) * 220000.0
    score += float(agg_2m["mean_avg_daily_return"]) * 70000.0
    score += float(agg_6m["mean_avg_daily_return"]) * 50000.0
    score -= abs(float(agg_2m["worst_max_drawdown"])) * 20000.0
    score -= abs(float(agg_6m["worst_max_drawdown"])) * 15000.0
    score -= abs(float(agg_4y["worst_max_drawdown"])) * 11000.0
    score -= float(agg_2m["pair_return_dispersion"]) * 120000.0
    score -= float(agg_6m["pair_return_dispersion"]) * 90000.0
    score -= float(agg_4y["pair_return_dispersion"]) * 70000.0
    score -= tree_size(node) * 200.0
    score -= max(0, tree_depth(node) - 2) * 400.0
    if not filter_decision.accepted:
        score -= 10_000_000.0
    return float(score)


def build_seed_trees(expert_pool: list[dict[str, Any]], condition_options: list[ConditionSpec]) -> list[TreeNode]:
    seeds: list[TreeNode] = []
    top_count = min(4, len(expert_pool))
    for idx in range(top_count):
        seeds.append(LeafNode(idx))
    if len(expert_pool) >= 2:
        seeds.append(
            ConditionNode(
                spec=copy.deepcopy(condition_options[0]),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            )
        )
    if len(expert_pool) >= 4:
        seeds.append(
            ConditionNode(
                spec=copy.deepcopy(condition_options[10]),
                if_true=ConditionNode(
                    spec=copy.deepcopy(condition_options[4]),
                    if_true=LeafNode(0),
                    if_false=LeafNode(2),
                ),
                if_false=ConditionNode(
                    spec=copy.deepcopy(condition_options[18]),
                    if_true=LeafNode(1),
                    if_false=LeafNode(3),
                ),
            )
        )
    return seeds


def tournament_select(population: list[dict[str, Any]], rng: random.Random, k: int = 3) -> TreeNode:
    sample = rng.sample(population, k=min(k, len(population)))
    sample.sort(key=lambda item: item["fitness"], reverse=True)
    return copy.deepcopy(sample[0]["tree"])


def export_llm_review_queue(path: str | None, candidates: list[dict[str, Any]], expert_pool: list[dict[str, Any]]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in candidates:
        tree = item["tree"]
        lines.append(
            json.dumps(
                {
                    "tree_key": tree_key(tree),
                    "tree": serialize_tree(tree),
                    "llm_prompt": item["filter"].llm_prompt or build_llm_prompt(tree, expert_pool),
                    "accepted": item["filter"].accepted,
                    "reason": item["filter"].reason,
                    "source": item["filter"].source,
                },
                ensure_ascii=False,
            )
        )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    args = parse_args()
    started = perf_counter()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    pairs = parse_csv_tuple(args.pairs, str)
    route_thresholds = parse_csv_tuple(args.route_thresholds, float)
    _ = resolve_fast_engine(args.fast_engine)

    expert_summary_paths = [part.strip() for part in args.expert_summaries.split(",") if part.strip()]
    expert_pool = build_expert_pool(expert_summary_paths, args.expert_pool_size)
    if len(expert_pool) < 2:
        raise RuntimeError("Need at least 2 experts to build recursive fractal trees.")

    baseline_summary = json.loads(Path(args.baseline_summary).read_text())
    baseline_windows = baseline_summary["selected_candidate"]["windows"]
    llm_reviews = load_llm_review_map(args.llm_review_in)

    library = list(iter_params())
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
    expert_arrays = {pair: expert_arrays_for_pair(pair, expert_pool, route_thresholds) for pair in pairs}
    condition_options = build_condition_options()
    window_cache: dict[str, dict[str, Any]] = {}
    for label, start, end in DEFAULT_WINDOWS:
        df = df_all.loc[start:end].copy()
        feature_arrays = materialize_feature_arrays(build_market_features(df, pairs), pd.DatetimeIndex(df.index))
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
        window_cache[label] = {"features": feature_arrays, "pair_cache": pair_cache, "bars": int(len(df))}
    prepare_seconds = perf_counter() - prepare_started

    seed_trees = build_seed_trees(expert_pool, condition_options)
    population: list[TreeNode] = seed_trees[:]
    while len(population) < args.population:
        population.append(random_tree(rng, condition_options, len(expert_pool), args.max_depth))

    fast_cache: dict[str, dict[str, Any]] = {}

    def evaluate_tree(tree: TreeNode) -> dict[str, Any]:
        key = tree_key(tree)
        cached = fast_cache.get(key)
        if cached is not None:
            return cached
        filter_decision = semantic_filter(tree, expert_pool, args.max_depth, args.filter_mode, llm_reviews=llm_reviews)
        windows: dict[str, Any] = {}
        for label, _, _ in DEFAULT_WINDOWS:
            window_state = window_cache[label]
            expert_codes = evaluate_tree_codes(tree, window_state["features"])
            per_pair = {}
            for pair in pairs:
                threshold_idx, mappings = expert_arrays[pair]
                per_pair[pair] = fast_fractal_replay_from_context(
                    window_state["pair_cache"][pair]["fast_context"],
                    library_lookup,
                    route_thresholds,
                    threshold_idx,
                    mappings,
                    expert_codes,
                )
            windows[label] = {
                "start": next(start for name, start, _ in DEFAULT_WINDOWS if name == label),
                "end": next(end for name, _, end in DEFAULT_WINDOWS if name == label),
                "bars": window_state["bars"],
                "aggregate": aggregate_metrics(per_pair),
            }
        validation = build_validation_bundle(windows, baseline_windows)
        cached = {
            "tree": copy.deepcopy(tree),
            "filter": filter_decision,
            "windows": windows,
            "validation": validation,
            "fitness": fractal_fast_scalar_score(windows, filter_decision, tree),
        }
        fast_cache[key] = cached
        return cached

    search_started = perf_counter()
    for _ in range(args.generations):
        evaluated = [evaluate_tree(tree) for tree in population]
        evaluated.sort(key=lambda item: item["fitness"], reverse=True)
        elites = [copy.deepcopy(item["tree"]) for item in evaluated[: args.elite_count]]
        next_population = elites[:]
        while len(next_population) < args.population:
            parent_a = tournament_select(evaluated, rng)
            if rng.random() < 0.65:
                parent_b = tournament_select(evaluated, rng)
                child_a, child_b = crossover_tree(parent_a, parent_b, rng)
                candidate = child_a if rng.random() < 0.5 else child_b
            else:
                candidate = parent_a
            if rng.random() < 0.70:
                candidate = mutate_tree(candidate, rng, condition_options, len(expert_pool), args.max_depth)
            next_population.append(candidate)
        population = next_population[: args.population]
    search_seconds = perf_counter() - search_started

    evaluated = [evaluate_tree(tree) for tree in population]
    ranked = sorted({tree_key(item["tree"]): item for item in evaluated}.values(), key=lambda item: item["fitness"], reverse=True)
    top_candidates = ranked[: args.top_k]
    progressive_candidates = [item for item in top_candidates if item["validation"]["profiles"]["progressive_improvement"]["passed"]]
    target_candidates = [item for item in top_candidates if item["validation"]["profiles"]["target_060"]["passed"]]
    fallback_best = max(top_candidates, key=lambda item: score_realistic_candidate({"windows": item["windows"]})) if top_candidates else None
    selected = max(target_candidates, key=lambda item: score_realistic_candidate({"windows": item["windows"]})) if target_candidates else None
    selection_reason = "target_060_pass"
    if selected is None and progressive_candidates:
        selected = max(progressive_candidates, key=lambda item: score_realistic_candidate({"windows": item["windows"]}))
        selection_reason = "progressive_pass"
    if selected is None:
        selection_reason = "no_gate_pass"

    export_llm_review_queue(args.llm_review_out, top_candidates, expert_pool)

    report = {
        "search": {
            "algorithm": "fractal_genome_fast_stage",
            "population": args.population,
            "generations": args.generations,
            "elite_count": args.elite_count,
            "max_depth": args.max_depth,
            "seed": args.seed,
            "expert_pool_size": len(expert_pool),
            "filter_mode": args.filter_mode,
            "llm_review_in": args.llm_review_in,
            "llm_review_out": args.llm_review_out,
        },
        "pairs": list(pairs),
        "baseline_summary_path": str(args.baseline_summary),
        "baseline_candidate": baseline_summary["selected_candidate"],
        "expert_pool": expert_pool,
        "top_candidates": [
            {
                "tree": serialize_tree(item["tree"]),
                "fitness": float(item["fitness"]),
                "tree_depth": int(tree_depth(item["tree"])),
                "tree_size": int(tree_size(item["tree"])),
                "condition_count": int(len(collect_specs(item["tree"]))),
                "filter": {
                    "accepted": item["filter"].accepted,
                    "source": item["filter"].source,
                    "reason": item["filter"].reason,
                },
                "windows": item["windows"],
                "validation": item["validation"],
            }
            for item in top_candidates
        ],
        "promotion_candidates": {
            "target_060": [{"tree": serialize_tree(item["tree"])} for item in target_candidates],
            "progressive_improvement": [{"tree": serialize_tree(item["tree"])} for item in progressive_candidates],
        },
        "selection": {
            "reason": selection_reason,
            "target_060_pass_count": len(target_candidates),
            "progressive_pass_count": len(progressive_candidates),
        },
        "fallback_best_candidate": None if fallback_best is None else {
            "tree": serialize_tree(fallback_best["tree"]),
            "fitness": float(fallback_best["fitness"]),
            "windows": fallback_best["windows"],
            "validation": fallback_best["validation"],
            "filter": {
                "accepted": fallback_best["filter"].accepted,
                "source": fallback_best["filter"].source,
                "reason": fallback_best["filter"].reason,
            },
        },
        "selected_candidate": None if selected is None else {
            "tree": serialize_tree(selected["tree"]),
            "fitness": float(selected["fitness"]),
            "windows": selected["windows"],
            "validation": selected["validation"],
            "filter": {
                "accepted": selected["filter"].accepted,
                "source": selected["filter"].source,
                "reason": selected["filter"].reason,
            },
        },
        "runtime": {
            "prepare_context_seconds": prepare_seconds,
            "search_seconds": search_seconds,
            "total_seconds": perf_counter() - started,
            "evaluated_unique_candidates": len(fast_cache),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report["selection"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
