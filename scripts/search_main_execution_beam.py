#!/usr/bin/env python3
"""Beam search around the current main pairwise candidate using execution-gene neighbors."""

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
from execution_gene_utils import derive_execution_profile
from pairwise_regime_mixture_shadow_live import load_strategy_bundle
from search_pair_subset_fractal_genome import load_funding_from_cache_or_empty
from search_pair_subset_pairwise_moo_router import SEARCH_WINDOWS, build_neighbor_variants, candidate_id
from search_pair_subset_regime_mixture import aggregate_metrics, build_fast_context, build_library_lookup, build_overlay_inputs
from strategy_replay_dispatch import replay_candidate_from_context


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"
DEFAULT_BASE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
DEFAULT_MODEL = MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"
DEFAULT_PAIR_GATE = MODELS_DIR / "main_execution_pair_gate_grid_20260412.json"
DEFAULT_ROLE_GATE = MODELS_DIR / "main_execution_role_gate_grid_20260412.json"
DEFAULT_GLOBAL_GATE = MODELS_DIR / "main_execution_gate_grid_20260412.json"
DEFAULT_OUT = MODELS_DIR / "main_execution_beam_search_20260412.json"
GUARD_WINDOWS: tuple[str, ...] = ("recent_6m", "recent_1y", "full_4y")
STRICT_GUARD_WINDOWS: tuple[str, ...] = GUARD_WINDOWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local beam search around main using execution/gate variants.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--base-summary", default=str(DEFAULT_BASE_SUMMARY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--pair-gate-grid", default=str(DEFAULT_PAIR_GATE))
    parser.add_argument("--role-gate-grid", default=str(DEFAULT_ROLE_GATE))
    parser.add_argument("--global-gate-grid", default=str(DEFAULT_GLOBAL_GATE))
    parser.add_argument("--seed-top-k", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--objective",
        choices=("balanced", "winrate_guarded", "tail_guarded"),
        default="winrate_guarded",
    )
    parser.add_argument("--top-results", type=int, default=10)
    parser.add_argument("--return-retention-floor", type=float, default=0.995)
    parser.add_argument("--drawdown-ratio-cap", type=float, default=1.02)
    parser.add_argument("--extra-seed-results", default="")
    parser.add_argument("--summary-out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def clone_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(candidate))


def build_seed_candidates(
    baseline_candidate: dict[str, Any],
    *,
    pairs: tuple[str, ...],
    pair_gate_grid: Path,
    role_gate_grid: Path,
    global_gate_grid: Path,
    extra_seed_results: tuple[Path, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = [clone_candidate(baseline_candidate)]

    def merge_gene_payload(seed_payload: dict[str, Any], source: str) -> None:
        top = seed_payload.get("top") or []
        for row in top[: max(int(top_k), 0)]:
            genes = row.get("genes")
            gene = row.get("gene")
            candidate = clone_candidate(baseline_candidate)
            for pair in pairs:
                pair_cfg = candidate["pair_configs"][pair]
                pair_cfg["execution_gene"] = dict(derive_execution_profile(None)["gene"])
                if isinstance(genes, dict) and isinstance(genes.get(pair), dict):
                    pair_cfg["execution_gene"].update(genes[pair])
                elif isinstance(gene, dict):
                    pair_cfg["execution_gene"].update(gene)
            candidate["seed_source"] = source
            seeds.append(candidate)

    for source, path in (
        ("pair_gate", pair_gate_grid),
        ("role_gate", role_gate_grid),
        ("global_gate", global_gate_grid),
    ):
        if path.exists():
            merge_gene_payload(load_json(path), source)

    for path in extra_seed_results:
        if not path.exists():
            continue
        payload = load_json(path)
        for source_key in ("top_guard_passed", "top"):
            for row in (payload.get(source_key) or [])[: max(int(top_k), 0)]:
                candidate = row.get("candidate")
                if isinstance(candidate, dict) and isinstance(candidate.get("pair_configs"), dict):
                    candidate_copy = clone_candidate(candidate)
                    candidate_copy["seed_source"] = f"extra:{path.name}:{source_key}"
                    seeds.append(candidate_copy)

    unique: dict[str, dict[str, Any]] = {}
    for candidate in seeds:
        unique[candidate_id(candidate, pairs)] = candidate
    return list(unique.values())


def evaluate_candidate(
    candidate: dict[str, Any],
    *,
    pairs: tuple[str, ...],
    window_cache: dict[str, Any],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for label, start, end in SEARCH_WINDOWS:
        window_data = window_cache[label]
        per_pair: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            per_pair[pair] = replay_candidate_from_context(
                candidate=candidate,
                pair=pair,
                context=window_data["pairs"][pair]["fast_context"],
                library_lookup=library_lookup,
                route_thresholds=route_thresholds,
                leaf_runtime_array=None,
                leaf_codes=None,
            )
        windows[label] = {
            "start": start,
            "end": end,
            "bars": int(len(window_data["df"])),
            "per_pair": per_pair,
            "aggregate": aggregate_metrics(per_pair),
        }
    return windows


def compare_to_baseline(windows: dict[str, Any], baseline_windows: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for label, *_ in SEARCH_WINDOWS:
        agg = windows[label]["aggregate"]
        base = baseline_windows[label]["aggregate"]
        comparison[label] = {
            "delta_mean_win_rate": float(agg.get("mean_daily_win_rate", 0.0)) - float(base.get("mean_daily_win_rate", 0.0)),
            "delta_worst_win_rate": float(agg.get("worst_pair_daily_win_rate", 0.0)) - float(base.get("worst_pair_daily_win_rate", 0.0)),
            "delta_worst_avg_daily_return": float(agg["worst_pair_avg_daily_return"]) - float(base["worst_pair_avg_daily_return"]),
            "delta_worst_total_return": float(agg["worst_pair_total_return"]) - float(base["worst_pair_total_return"]),
            "delta_worst_max_drawdown": float(agg["worst_max_drawdown"]) - float(base["worst_max_drawdown"]),
        }
    return comparison


def evaluate_guard(
    windows: dict[str, Any],
    baseline_windows: dict[str, Any],
    compare: dict[str, Any],
    *,
    return_retention_floor: float,
    drawdown_ratio_cap: float,
) -> dict[str, Any]:
    worst_total_return_delta = min(float(compare[label]["delta_worst_total_return"]) for label in GUARD_WINDOWS)
    worst_drawdown_delta = min(float(compare[label]["delta_worst_max_drawdown"]) for label in GUARD_WINDOWS)
    durable_mean_win_delta = sum(float(compare[label]["delta_mean_win_rate"]) for label in GUARD_WINDOWS) / len(GUARD_WINDOWS)
    durable_worst_win_delta = sum(float(compare[label]["delta_worst_win_rate"]) for label in GUARD_WINDOWS) / len(GUARD_WINDOWS)
    recent_mean_win_delta = (
        float(compare["recent_2m"]["delta_mean_win_rate"]) + float(compare["recent_4m"]["delta_mean_win_rate"])
    ) / 2.0
    return_ratios = {
        label: float(windows[label]["aggregate"]["worst_pair_total_return"])
        / max(float(baseline_windows[label]["aggregate"]["worst_pair_total_return"]), 1e-12)
        for label in GUARD_WINDOWS
    }
    drawdown_ratios = {
        label: abs(float(windows[label]["aggregate"]["worst_max_drawdown"]))
        / max(abs(float(baseline_windows[label]["aggregate"]["worst_max_drawdown"])), 1e-12)
        for label in GUARD_WINDOWS
    }
    strict_nonworse = all(float(compare[label]["delta_worst_total_return"]) >= 0.0 for label in STRICT_GUARD_WINDOWS) and all(
        float(compare[label]["delta_worst_max_drawdown"]) >= -1e-12 for label in STRICT_GUARD_WINDOWS
    )
    retention_pass = min(return_ratios.values()) >= float(return_retention_floor)
    drawdown_pass = max(drawdown_ratios.values()) <= float(drawdown_ratio_cap)
    green = strict_nonworse and float(compare["recent_6m"]["delta_worst_win_rate"]) > 0.0
    yellow = strict_nonworse and not green
    return {
        "guard_pass": bool(retention_pass and drawdown_pass),
        "strict_nonworse_pass": bool(strict_nonworse),
        "traffic_light": "green" if green else ("yellow" if yellow else "red"),
        "worst_total_return_delta_floor": float(worst_total_return_delta),
        "worst_drawdown_delta_floor": float(worst_drawdown_delta),
        "durable_mean_win_delta": float(durable_mean_win_delta),
        "durable_worst_win_delta": float(durable_worst_win_delta),
        "recent_mean_win_delta": float(recent_mean_win_delta),
        "min_return_retention_ratio": float(min(return_ratios.values())),
        "max_drawdown_ratio": float(max(drawdown_ratios.values())),
    }


def candidate_score(
    windows: dict[str, Any],
    baseline_windows: dict[str, Any],
    *,
    objective: str,
    return_retention_floor: float = 0.995,
    drawdown_ratio_cap: float = 1.02,
) -> float:
    compare = compare_to_baseline(windows, baseline_windows)
    guard = evaluate_guard(
        windows,
        baseline_windows,
        compare,
        return_retention_floor=float(return_retention_floor),
        drawdown_ratio_cap=float(drawdown_ratio_cap),
    )
    if objective == "winrate_guarded":
        score = 0.0
        for label in GUARD_WINDOWS:
            agg = windows[label]["aggregate"]
            base = baseline_windows[label]["aggregate"]
            score += (float(agg.get("mean_daily_win_rate", 0.0)) - float(base.get("mean_daily_win_rate", 0.0))) * 24000.0
            score += (float(agg.get("worst_pair_daily_win_rate", 0.0)) - float(base.get("worst_pair_daily_win_rate", 0.0))) * 28000.0
            score += max(0.0, float(agg.get("mean_daily_win_rate", 0.0)) - 0.50) * 9000.0
            score += max(0.0, float(agg.get("worst_pair_daily_win_rate", 0.0)) - 0.50) * 11000.0
            score += (float(agg["worst_pair_total_return"]) - float(base["worst_pair_total_return"])) * 40.0
        for label in ("recent_2m", "recent_4m"):
            agg = windows[label]["aggregate"]
            base = baseline_windows[label]["aggregate"]
            score += (float(agg.get("mean_daily_win_rate", 0.0)) - float(base.get("mean_daily_win_rate", 0.0))) * 7000.0
            score += (float(agg.get("worst_pair_daily_win_rate", 0.0)) - float(base.get("worst_pair_daily_win_rate", 0.0))) * 8000.0
            score += max(0.0, float(agg.get("mean_daily_win_rate", 0.0)) - 0.50) * 2500.0
        for label in GUARD_WINDOWS:
            score -= max(0.0, -float(compare[label]["delta_worst_total_return"])) * 400.0
            score -= max(0.0, -float(compare[label]["delta_worst_max_drawdown"])) * 140000.0
        if not guard["guard_pass"]:
            score -= max(0.0, -float(guard["worst_total_return_delta_floor"])) * 1200.0
            score -= max(0.0, -float(guard["worst_drawdown_delta_floor"])) * 240000.0
        return float(score)
    if objective == "tail_guarded":
        score = 0.0
        for label in GUARD_WINDOWS:
            agg = windows[label]["aggregate"]
            base = baseline_windows[label]["aggregate"]
            score += (float(agg["worst_pair_total_return"]) - float(base["worst_pair_total_return"])) * 180.0
            score += (float(agg.get("worst_pair_daily_win_rate", 0.0)) - float(base.get("worst_pair_daily_win_rate", 0.0))) * 12000.0
            score -= max(0.0, abs(float(agg["worst_max_drawdown"])) - abs(float(base["worst_max_drawdown"]))) * 220000.0
        score += float(guard["durable_mean_win_delta"]) * 10000.0
        score += float(guard["recent_mean_win_delta"]) * 3500.0
        return float(score)
    score = 0.0
    for label, ret_w, win_w, dd_w in (
        ("recent_2m", 140000.0, 2200.0, 6000.0),
        ("recent_4m", 120000.0, 2200.0, 6000.0),
        ("recent_6m", 150000.0, 2600.0, 9000.0),
        ("recent_1y", 170000.0, 3200.0, 12000.0),
        ("full_4y", 110000.0, 3400.0, 9000.0),
    ):
        agg = windows[label]["aggregate"]
        base = baseline_windows[label]["aggregate"]
        score += (float(agg["worst_pair_avg_daily_return"]) - float(base["worst_pair_avg_daily_return"])) * ret_w
        score += (float(agg.get("mean_daily_win_rate", 0.0)) - float(base.get("mean_daily_win_rate", 0.0))) * win_w
        score += (float(agg.get("worst_pair_daily_win_rate", 0.0)) - float(base.get("worst_pair_daily_win_rate", 0.0))) * (win_w * 1.3)
        drawdown_deterioration = abs(float(agg["worst_max_drawdown"])) - abs(float(base["worst_max_drawdown"]))
        score -= max(0.0, drawdown_deterioration) * dd_w
        total_return_deterioration = float(base["worst_pair_total_return"]) - float(agg["worst_pair_total_return"])
        score -= max(0.0, total_return_deterioration) * (ret_w * 0.25)
    return float(score)


def candidate_rank_key(row: dict[str, Any], *, objective: str) -> tuple[Any, ...]:
    if objective == "winrate_guarded":
        compare = row["compare_to_main"]
        guard = row["guard"]
        worst_win_combo = (
            0.5 * float(compare["recent_6m"]["delta_worst_win_rate"])
            + 0.3 * float(compare["recent_1y"]["delta_worst_win_rate"])
            + 0.2 * float(compare["full_4y"]["delta_worst_win_rate"])
        )
        tail_floor = min(float(compare[label]["delta_worst_total_return"]) for label in GUARD_WINDOWS)
        durable_mean_combo = float(compare["recent_1y"]["delta_mean_win_rate"]) + float(compare["full_4y"]["delta_mean_win_rate"])
        return (
            1 if guard["guard_pass"] else 0,
            worst_win_combo,
            tail_floor,
            durable_mean_combo,
            float(row["score"]),
        )
    return (float(row["score"]),)


def main() -> None:
    args = parse_args()
    started = perf_counter()

    summary_path = Path(args.summary)
    summary = load_json(summary_path)
    baseline_candidate = clone_candidate(summary["selected_candidate"])
    baseline_candidate["candidate_kind"] = "pairwise_candidate"
    pairs = tuple(summary.get("pairs") or ("BTCUSDT", "BNBUSDT"))
    route_thresholds = tuple(float(v) for v in (summary.get("search", {}).get("route_thresholds") or (0.35, 0.50, 0.65, 0.80)))

    bundle = load_strategy_bundle(summary_path, Path(args.base_summary), Path(args.model), candidate_key="selected_candidate")
    library_lookup = build_library_lookup(bundle["library"])
    compiled = bundle["compiled_model"]

    start_all = SEARCH_WINDOWS[-1][1]
    end_all = SEARCH_WINDOWS[0][2]
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=False)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {pair: load_funding_from_cache_or_empty(pair, start_all, end_all) for pair in pairs}

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
        window_cache[label] = {"df": df, "pairs": pair_cache}

    baseline_windows = evaluate_candidate(
        baseline_candidate,
        pairs=pairs,
        window_cache=window_cache,
        library_lookup=library_lookup,
        route_thresholds=route_thresholds,
    )
    extra_seed_results = tuple(
        Path(value)
        for value in str(args.extra_seed_results).split(",")
        if value.strip()
    )

    seed_candidates = build_seed_candidates(
        baseline_candidate,
        pairs=pairs,
        pair_gate_grid=Path(args.pair_gate_grid),
        role_gate_grid=Path(args.role_gate_grid),
        global_gate_grid=Path(args.global_gate_grid),
        extra_seed_results=extra_seed_results,
        top_k=int(args.seed_top_k),
    )

    seen: dict[str, dict[str, Any]] = {}
    frontier: list[dict[str, Any]] = []
    for candidate in seed_candidates:
        cid = candidate_id(candidate, pairs)
        if cid in seen:
            continue
        windows = evaluate_candidate(
            candidate,
            pairs=pairs,
            window_cache=window_cache,
            library_lookup=library_lookup,
            route_thresholds=route_thresholds,
        )
        compare = compare_to_baseline(windows, baseline_windows)
        row = {
            "candidate_id": cid,
            "candidate": candidate,
            "windows": windows,
            "compare_to_main": compare,
        }
        row["guard"] = evaluate_guard(
            windows,
            baseline_windows,
            compare,
            return_retention_floor=float(args.return_retention_floor),
            drawdown_ratio_cap=float(args.drawdown_ratio_cap),
        )
        row["score"] = candidate_score(
            windows,
            baseline_windows,
            objective=str(args.objective),
            return_retention_floor=float(args.return_retention_floor),
            drawdown_ratio_cap=float(args.drawdown_ratio_cap),
        )
        seen[cid] = row
        frontier.append(row)

    for _ in range(max(int(args.rounds), 0)):
        frontier.sort(key=lambda item: candidate_rank_key(item, objective=str(args.objective)), reverse=True)
        beam = frontier[: max(int(args.beam_width), 1)]
        new_rows: list[dict[str, Any]] = []
        for row in beam:
            for variant in build_neighbor_variants(row["candidate"], pairs=pairs, route_thresholds=route_thresholds):
                cid = candidate_id(variant, pairs)
                if cid in seen:
                    continue
                windows = evaluate_candidate(
                    variant,
                    pairs=pairs,
                    window_cache=window_cache,
                    library_lookup=library_lookup,
                    route_thresholds=route_thresholds,
                )
                compare = compare_to_baseline(windows, baseline_windows)
                item = {
                    "candidate_id": cid,
                    "candidate": variant,
                    "windows": windows,
                    "compare_to_main": compare,
                }
                item["guard"] = evaluate_guard(
                    windows,
                    baseline_windows,
                    compare,
                    return_retention_floor=float(args.return_retention_floor),
                    drawdown_ratio_cap=float(args.drawdown_ratio_cap),
                )
                item["score"] = candidate_score(
                    windows,
                    baseline_windows,
                    objective=str(args.objective),
                    return_retention_floor=float(args.return_retention_floor),
                    drawdown_ratio_cap=float(args.drawdown_ratio_cap),
                )
                seen[cid] = item
                new_rows.append(item)
        if not new_rows:
            break
        frontier.extend(new_rows)

    ranked = sorted(frontier, key=lambda item: candidate_rank_key(item, objective=str(args.objective)), reverse=True)
    guard_ranked = [row for row in ranked if row["guard"]["guard_pass"]]
    payload = {
        "search": {
            "summary": str(summary_path),
            "base_summary": str(args.base_summary),
            "model": str(args.model),
            "seed_top_k": int(args.seed_top_k),
            "beam_width": int(args.beam_width),
            "rounds": int(args.rounds),
            "objective": str(args.objective),
            "return_retention_floor": float(args.return_retention_floor),
            "drawdown_ratio_cap": float(args.drawdown_ratio_cap),
            "extra_seed_results": [str(path) for path in extra_seed_results],
            "evaluated_candidates": int(len(frontier)),
            "elapsed_seconds": float(perf_counter() - started),
        },
        "baseline_candidate": baseline_candidate,
        "baseline_windows": baseline_windows,
        "guard_pass_count": int(len(guard_ranked)),
        "top": [
            {
                "candidate_id": row["candidate_id"],
                "candidate": row["candidate"],
                "score": row["score"],
                "guard": row["guard"],
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
                "compare_to_main": row["compare_to_main"],
                "windows": row["windows"],
            }
            for row in guard_ranked[: max(int(args.top_results), 1)]
        ],
    }
    write_json(args.summary_out, payload)
    print(args.summary_out)


if __name__ == "__main__":
    main()
