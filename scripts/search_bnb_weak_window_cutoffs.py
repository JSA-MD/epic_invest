#!/usr/bin/env python3
"""Search BNB-only weak-window regime/risk cutoffs on top of main pairwise replay."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from execution_gene_utils import derive_execution_profile, normalize_execution_gene
from pairwise_regime_mixture_shadow_live import load_strategy_bundle
from search_main_execution_beam import json_safe, write_json
from search_pair_subset_fractal_genome import load_funding_from_cache_or_empty
from search_pair_subset_pairwise_moo_router import EXECUTION_GENE_OPTIONS, candidate_id
from search_pair_subset_regime_mixture import aggregate_metrics, build_fast_context, build_library_lookup, build_overlay_inputs
from strategy_replay_dispatch import replay_candidate_from_context


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"
DEFAULT_BASE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
DEFAULT_MODEL = MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"
DEFAULT_CANDIDATE_SOURCE = Path("/tmp/main_execution_beam_tail_guarded_pass2_20260412.json")
DEFAULT_OUT = Path("/tmp/bnb_weak_window_cutoff_search_20260412.json")

BNB_PAIR = "BNBUSDT"
STATIC_PAIR = "BTCUSDT"
TARGET_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("weak_2024_02", "2024-02-06", "2024-08-06"),
    ("weak_2024_03", "2024-03-06", "2024-09-06"),
    ("weak_2025_05", "2025-05-06", "2025-11-06"),
)
GUARD_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("recent_6m", "2025-10-06", "2026-04-06"),
    ("recent_1y", "2025-04-07", "2026-04-06"),
    ("full_4y", "2022-04-06", "2026-04-06"),
)
SEARCH_WINDOWS: tuple[tuple[str, str, str], ...] = TARGET_WINDOWS + GUARD_WINDOWS
TARGET_WINDOW_LABELS: tuple[str, ...] = tuple(label for label, _, _ in TARGET_WINDOWS)
GUARD_WINDOW_LABELS: tuple[str, ...] = tuple(label for label, _, _ in GUARD_WINDOWS)
TARGET_EXECUTION_KEYS: tuple[str, ...] = (
    "signal_gate_pct",
    "regime_buffer_mult",
    "confirm_bars",
    "abstain_edge_pct",
    "specialist_isolation_mult",
    "flow_alignment_threshold",
    "dc_alignment_threshold",
    "min_alignment_votes",
    "range_signal_gate_mult",
    "panic_signal_gate_mult",
    "carry_signal_gate_mult",
    "range_regime_buffer_mult",
    "panic_regime_buffer_mult",
    "carry_regime_buffer_mult",
)
BUNDLE_STEPS: tuple[dict[str, int], ...] = (
    {
        "signal_gate_pct": 1,
        "abstain_edge_pct": 1,
        "range_signal_gate_mult": 1,
        "range_regime_buffer_mult": 1,
    },
    {
        "signal_gate_pct": 1,
        "abstain_edge_pct": 1,
        "panic_signal_gate_mult": 1,
        "panic_regime_buffer_mult": 1,
        "specialist_isolation_mult": 1,
    },
    {
        "signal_gate_pct": 1,
        "carry_signal_gate_mult": 1,
        "carry_regime_buffer_mult": 1,
        "confirm_bars": 1,
    },
    {
        "flow_alignment_threshold": 1,
        "dc_alignment_threshold": 1,
        "min_alignment_votes": 1,
    },
    {
        "signal_gate_pct": -1,
        "abstain_edge_pct": -1,
        "range_signal_gate_mult": -1,
        "panic_signal_gate_mult": -1,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search BNB-only weak-window regime/risk cutoffs.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--base-summary", default=str(DEFAULT_BASE_SUMMARY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--candidate-source", default=str(DEFAULT_CANDIDATE_SOURCE))
    parser.add_argument("--candidate-group", default="top_guard_passed")
    parser.add_argument("--seed-indices", default="0,1,2,3,4")
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--top-results", type=int, default=10)
    parser.add_argument("--summary-out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def clone_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(candidate))


def parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())


def load_seed_candidates(path: Path, group: str, indices: tuple[int, ...]) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get(group) or payload.get("top") or []
    seeds: list[dict[str, Any]] = []
    for index in indices:
        if not (0 <= int(index) < len(rows)):
            continue
        row = rows[int(index)]
        candidate = row.get("candidate")
        if isinstance(candidate, dict) and isinstance(candidate.get("pair_configs"), dict):
            candidate_copy = clone_candidate(candidate)
            candidate_copy["candidate_kind"] = str(candidate_copy.get("candidate_kind") or "pairwise_candidate")
            seeds.append(candidate_copy)
    return seeds


def option_step(value: Any, options: tuple[Any, ...], direction: int) -> Any | None:
    option_idx = min(range(len(options)), key=lambda idx: abs(float(options[idx]) - float(value)))
    new_idx = option_idx + int(direction)
    if not (0 <= new_idx < len(options)):
        return None
    return options[new_idx]


def append_bundle_variant(candidate: dict[str, Any], *, pair: str, base_gene: dict[str, Any], updates: dict[str, int]) -> list[dict[str, Any]]:
    variant = copy.deepcopy(candidate)
    execution_gene = variant["pair_configs"][pair].setdefault("execution_gene", {})
    changed = False
    for key, direction in updates.items():
        options = EXECUTION_GENE_OPTIONS[key]
        value = base_gene.get(key)
        stepped = option_step(value, options, direction)
        if stepped is None:
            continue
        execution_gene[key] = stepped
        changed = True
    return [variant] if changed else []


def build_bnb_neighbor_variants(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    base_gene = normalize_execution_gene((candidate.get("pair_configs") or {}).get(BNB_PAIR, {}).get("execution_gene"))
    for key in TARGET_EXECUTION_KEYS:
        options = EXECUTION_GENE_OPTIONS[key]
        value = base_gene.get(key)
        for direction in (-1, 1):
            stepped = option_step(value, options, direction)
            if stepped is None:
                continue
            variant = copy.deepcopy(candidate)
            variant["pair_configs"][BNB_PAIR].setdefault("execution_gene", {})
            variant["pair_configs"][BNB_PAIR]["execution_gene"][key] = stepped
            variants.append(variant)
    for updates in BUNDLE_STEPS:
        variants.extend(append_bundle_variant(candidate, pair=BNB_PAIR, base_gene=base_gene, updates=updates))
    return variants


def evaluate_pair_for_window(
    candidate: dict[str, Any],
    *,
    pair: str,
    pair_context: dict[str, Any],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    return replay_candidate_from_context(
        candidate=candidate,
        pair=pair,
        context=pair_context["fast_context"],
        library_lookup=library_lookup,
        route_thresholds=route_thresholds,
        leaf_runtime_array=None,
        leaf_codes=None,
    )


def evaluate_candidate_windows(
    candidate: dict[str, Any],
    *,
    static_pair_reports: dict[str, dict[str, Any]],
    window_cache: dict[str, Any],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for label, start, end in SEARCH_WINDOWS:
        per_pair = {
            STATIC_PAIR: static_pair_reports[label][STATIC_PAIR],
            BNB_PAIR: evaluate_pair_for_window(
                candidate,
                pair=BNB_PAIR,
                pair_context=window_cache[label]["pairs"][BNB_PAIR],
                library_lookup=library_lookup,
                route_thresholds=route_thresholds,
            ),
        }
        windows[label] = {
            "start": start,
            "end": end,
            "bars": int(window_cache[label]["bars"]),
            "per_pair": per_pair,
            "aggregate": aggregate_metrics(per_pair),
        }
    return windows


def compare_to_main(
    windows: dict[str, Any],
    baseline_windows: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate_compare: dict[str, Any] = {}
    weak_bnb_compare: dict[str, Any] = {}
    for label, *_ in SEARCH_WINDOWS:
        agg = windows[label]["aggregate"]
        base = baseline_windows[label]["aggregate"]
        aggregate_compare[label] = {
            "delta_mean_win_rate": float(agg.get("mean_daily_win_rate", 0.0)) - float(base.get("mean_daily_win_rate", 0.0)),
            "delta_worst_win_rate": float(agg.get("worst_pair_daily_win_rate", 0.0)) - float(base.get("worst_pair_daily_win_rate", 0.0)),
            "delta_mean_total_return": float(agg.get("mean_total_return", 0.0)) - float(base.get("mean_total_return", 0.0)),
            "delta_worst_total_return": float(agg.get("worst_pair_total_return", 0.0)) - float(base.get("worst_pair_total_return", 0.0)),
            "delta_worst_max_drawdown": float(agg.get("worst_max_drawdown", 0.0)) - float(base.get("worst_max_drawdown", 0.0)),
        }
        if label in TARGET_WINDOW_LABELS:
            bnb = windows[label]["per_pair"][BNB_PAIR]
            base_bnb = baseline_windows[label]["per_pair"][BNB_PAIR]
            weak_bnb_compare[label] = {
                "candidate_total_return": float(bnb.get("total_return", 0.0)),
                "main_total_return": float(base_bnb.get("total_return", 0.0)),
                "delta_total_return": float(bnb.get("total_return", 0.0)) - float(base_bnb.get("total_return", 0.0)),
                "candidate_daily_win_rate": float(bnb.get("daily_win_rate", 0.0)),
                "main_daily_win_rate": float(base_bnb.get("daily_win_rate", 0.0)),
                "delta_daily_win_rate": float(bnb.get("daily_win_rate", 0.0)) - float(base_bnb.get("daily_win_rate", 0.0)),
                "candidate_max_drawdown": float(bnb.get("max_drawdown", 0.0)),
                "main_max_drawdown": float(base_bnb.get("max_drawdown", 0.0)),
                "delta_max_drawdown": float(bnb.get("max_drawdown", 0.0)) - float(base_bnb.get("max_drawdown", 0.0)),
                "candidate_n_trades": int(bnb.get("n_trades", 0)),
                "main_n_trades": int(base_bnb.get("n_trades", 0)),
                "delta_n_trades": int(bnb.get("n_trades", 0)) - int(base_bnb.get("n_trades", 0)),
            }
    return aggregate_compare, weak_bnb_compare


def evaluate_guard(compare: dict[str, Any], weak_compare: dict[str, Any]) -> dict[str, Any]:
    weak_return_floor = min(float(weak_compare[label]["delta_total_return"]) for label in TARGET_WINDOW_LABELS)
    weak_mdd_floor = min(float(weak_compare[label]["delta_max_drawdown"]) for label in TARGET_WINDOW_LABELS)
    weak_win_floor = min(float(weak_compare[label]["delta_daily_win_rate"]) for label in TARGET_WINDOW_LABELS)
    weak_improvement_count = sum(1 for label in TARGET_WINDOW_LABELS if float(weak_compare[label]["delta_total_return"]) > 0.0)
    weak_mdd_improvement_count = sum(1 for label in TARGET_WINDOW_LABELS if float(weak_compare[label]["delta_max_drawdown"]) >= 0.0)
    durable_return_floor = min(float(compare[label]["delta_worst_total_return"]) for label in GUARD_WINDOW_LABELS)
    durable_mdd_floor = min(float(compare[label]["delta_worst_max_drawdown"]) for label in GUARD_WINDOW_LABELS)
    durable_win_floor = min(float(compare[label]["delta_worst_win_rate"]) for label in GUARD_WINDOW_LABELS)
    guard_pass = durable_return_floor >= -1e-12 and durable_mdd_floor >= -1e-12
    weak_pass = weak_improvement_count == len(TARGET_WINDOW_LABELS) and weak_mdd_improvement_count >= 2
    traffic_light = "green" if (guard_pass and weak_pass) else ("yellow" if weak_improvement_count >= 2 else "red")
    return {
        "guard_pass": bool(guard_pass),
        "weak_pass": bool(weak_pass),
        "traffic_light": traffic_light,
        "weak_improvement_count": int(weak_improvement_count),
        "weak_mdd_improvement_count": int(weak_mdd_improvement_count),
        "weak_return_floor": float(weak_return_floor),
        "weak_mdd_floor": float(weak_mdd_floor),
        "weak_win_floor": float(weak_win_floor),
        "durable_return_floor": float(durable_return_floor),
        "durable_mdd_floor": float(durable_mdd_floor),
        "durable_win_floor": float(durable_win_floor),
    }


def candidate_score(compare: dict[str, Any], weak_compare: dict[str, Any], guard: dict[str, Any]) -> float:
    score = 0.0
    for label in TARGET_WINDOW_LABELS:
        item = weak_compare[label]
        score += float(item["delta_total_return"]) * 420.0
        score += float(item["delta_daily_win_rate"]) * 22000.0
        score += float(item["delta_max_drawdown"]) * 180000.0
        score -= max(0.0, -float(item["candidate_total_return"])) * 220.0
        score -= max(0.0, -float(item["delta_total_return"])) * 300.0
    for label in GUARD_WINDOW_LABELS:
        item = compare[label]
        score += float(item["delta_worst_total_return"]) * 120.0
        score += float(item["delta_worst_win_rate"]) * 8000.0
        score -= max(0.0, -float(item["delta_worst_total_return"])) * 450.0
        score -= max(0.0, -float(item["delta_worst_max_drawdown"])) * 240000.0
    score += float(guard["weak_improvement_count"]) * 9000.0
    score += float(guard["weak_mdd_improvement_count"]) * 4500.0
    score += float(guard["durable_win_floor"]) * 2000.0
    if guard["guard_pass"]:
        score += 12000.0
    if guard["weak_pass"]:
        score += 18000.0
    return float(score)


def candidate_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    guard = row["guard"]
    return (
        1 if guard["guard_pass"] else 0,
        1 if guard["weak_pass"] else 0,
        int(guard["weak_improvement_count"]),
        float(guard["weak_return_floor"]),
        float(guard["durable_return_floor"]),
        float(row["score"]),
    )


def build_window_cache(
    *,
    pairs: tuple[str, ...],
    compiled: Any,
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
    baseline_candidate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.Series], dict[str, pd.DataFrame]]:
    start_all = SEARCH_WINDOWS[-1][1]
    end_all = SEARCH_WINDOWS[-1][2]
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=False)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {
        pair: load_funding_from_cache_or_empty(pair, start_all, end_all)
        for pair in pairs
    }
    window_cache: dict[str, Any] = {}
    for label, start, end in SEARCH_WINDOWS:
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
                "fast_context": build_fast_context(
                    df=df,
                    pair=pair,
                    raw_signal=signal_slice,
                    overlay_inputs=overlay_inputs,
                    route_thresholds=route_thresholds,
                    library_lookup=library_lookup,
                    funding_df=funding_slice,
                    route_state_mode=str((baseline_candidate.get("pair_configs") or {}).get(pair, {}).get("route_state_mode") or "equity_corr"),
                )
            }
        window_cache[label] = {"bars": int(len(df)), "pairs": pair_cache}
    return window_cache, raw_signal_all, funding_all


def build_static_pair_reports(
    *,
    baseline_candidate: dict[str, Any],
    window_cache: dict[str, Any],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    static_reports: dict[str, dict[str, Any]] = {}
    baseline_windows: dict[str, Any] = {}
    for label, start, end in SEARCH_WINDOWS:
        btc_report = evaluate_pair_for_window(
            baseline_candidate,
            pair=STATIC_PAIR,
            pair_context=window_cache[label]["pairs"][STATIC_PAIR],
            library_lookup=library_lookup,
            route_thresholds=route_thresholds,
        )
        bnb_report = evaluate_pair_for_window(
            baseline_candidate,
            pair=BNB_PAIR,
            pair_context=window_cache[label]["pairs"][BNB_PAIR],
            library_lookup=library_lookup,
            route_thresholds=route_thresholds,
        )
        static_reports[label] = {STATIC_PAIR: btc_report}
        baseline_windows[label] = {
            "start": start,
            "end": end,
            "bars": int(window_cache[label]["bars"]),
            "per_pair": {STATIC_PAIR: btc_report, BNB_PAIR: bnb_report},
            "aggregate": aggregate_metrics({STATIC_PAIR: btc_report, BNB_PAIR: bnb_report}),
        }
    return static_reports, baseline_windows


def main() -> None:
    args = parse_args()
    started = perf_counter()

    summary_path = Path(args.summary)
    summary = load_json(summary_path)
    baseline_candidate = clone_candidate(summary["selected_candidate"])
    baseline_candidate["candidate_kind"] = "pairwise_candidate"
    pairs = tuple(summary.get("pairs") or (STATIC_PAIR, BNB_PAIR))
    route_thresholds = tuple(float(v) for v in (summary.get("search", {}).get("route_thresholds") or (0.35, 0.50, 0.65, 0.80)))

    bundle = load_strategy_bundle(summary_path, Path(args.base_summary), Path(args.model), candidate_key="selected_candidate")
    library_lookup = build_library_lookup(bundle["library"])
    compiled = bundle["compiled_model"]

    window_cache, _, _ = build_window_cache(
        pairs=pairs,
        compiled=compiled,
        library_lookup=library_lookup,
        route_thresholds=route_thresholds,
        baseline_candidate=baseline_candidate,
    )
    static_pair_reports, baseline_windows = build_static_pair_reports(
        baseline_candidate=baseline_candidate,
        window_cache=window_cache,
        library_lookup=library_lookup,
        route_thresholds=route_thresholds,
    )

    seed_candidates = [baseline_candidate]
    seed_candidates.extend(
        load_seed_candidates(Path(args.candidate_source), str(args.candidate_group), parse_int_csv(str(args.seed_indices)))
    )
    unique_seed_candidates: list[dict[str, Any]] = []
    seen_seed_ids: set[str] = set()
    for candidate in seed_candidates:
        cid = candidate_id(candidate, pairs)
        if cid in seen_seed_ids:
            continue
        seen_seed_ids.add(cid)
        unique_seed_candidates.append(candidate)

    frontier: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for candidate in unique_seed_candidates:
        windows = evaluate_candidate_windows(
            candidate,
            static_pair_reports=static_pair_reports,
            window_cache=window_cache,
            library_lookup=library_lookup,
            route_thresholds=route_thresholds,
        )
        compare, weak_compare = compare_to_main(windows, baseline_windows)
        guard = evaluate_guard(compare, weak_compare)
        row = {
            "candidate_id": candidate_id(candidate, pairs),
            "candidate": candidate,
            "windows": windows,
            "compare_to_main": compare,
            "weak_bnb_compare": weak_compare,
            "guard": guard,
        }
        row["score"] = candidate_score(compare, weak_compare, guard)
        frontier.append(row)
        seen[row["candidate_id"]] = row

    for _ in range(max(int(args.rounds), 0)):
        frontier.sort(key=candidate_rank_key, reverse=True)
        beam = frontier[: max(int(args.beam_width), 1)]
        new_rows: list[dict[str, Any]] = []
        for row in beam:
            for variant in build_bnb_neighbor_variants(row["candidate"]):
                cid = candidate_id(variant, pairs)
                if cid in seen:
                    continue
                windows = evaluate_candidate_windows(
                    variant,
                    static_pair_reports=static_pair_reports,
                    window_cache=window_cache,
                    library_lookup=library_lookup,
                    route_thresholds=route_thresholds,
                )
                compare, weak_compare = compare_to_main(windows, baseline_windows)
                guard = evaluate_guard(compare, weak_compare)
                item = {
                    "candidate_id": cid,
                    "candidate": variant,
                    "windows": windows,
                    "compare_to_main": compare,
                    "weak_bnb_compare": weak_compare,
                    "guard": guard,
                }
                item["score"] = candidate_score(compare, weak_compare, guard)
                seen[cid] = item
                new_rows.append(item)
        if not new_rows:
            break
        frontier.extend(new_rows)

    ranked = sorted(frontier, key=candidate_rank_key, reverse=True)
    guard_ranked = [row for row in ranked if row["guard"]["guard_pass"]]
    weak_ranked = [row for row in ranked if row["guard"]["weak_pass"]]
    payload = {
        "search": {
            "summary": str(summary_path),
            "base_summary": str(args.base_summary),
            "model": str(args.model),
            "candidate_source": str(args.candidate_source),
            "candidate_group": str(args.candidate_group),
            "seed_indices": list(parse_int_csv(str(args.seed_indices))),
            "beam_width": int(args.beam_width),
            "rounds": int(args.rounds),
            "evaluated_candidates": int(len(frontier)),
            "elapsed_seconds": float(perf_counter() - started),
            "target_windows": list(TARGET_WINDOWS),
            "guard_windows": list(GUARD_WINDOWS),
        },
        "baseline_candidate": baseline_candidate,
        "baseline_windows": baseline_windows,
        "guard_pass_count": int(len(guard_ranked)),
        "weak_pass_count": int(len(weak_ranked)),
        "top": [
            {
                "candidate_id": row["candidate_id"],
                "candidate": row["candidate"],
                "score": row["score"],
                "guard": row["guard"],
                "weak_bnb_compare": row["weak_bnb_compare"],
                "compare_to_main": row["compare_to_main"],
                "windows": row["windows"],
            }
            for row in ranked[: max(int(args.top_results), 1)]
        ],
        "top_guard_passed": [
            {
                "candidate_id": row["candidate_id"],
                "candidate": row["candidate"],
                "score": row["score"],
                "guard": row["guard"],
                "weak_bnb_compare": row["weak_bnb_compare"],
                "compare_to_main": row["compare_to_main"],
                "windows": row["windows"],
            }
            for row in guard_ranked[: max(int(args.top_results), 1)]
        ],
        "top_weak_passed": [
            {
                "candidate_id": row["candidate_id"],
                "candidate": row["candidate"],
                "score": row["score"],
                "guard": row["guard"],
                "weak_bnb_compare": row["weak_bnb_compare"],
                "compare_to_main": row["compare_to_main"],
                "windows": row["windows"],
            }
            for row in weak_ranked[: max(int(args.top_results), 1)]
        ],
    }
    write_json(args.summary_out, payload)


if __name__ == "__main__":
    main()
