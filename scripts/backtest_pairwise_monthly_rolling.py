#!/usr/bin/env python3
"""Run monthly rolling fixed-horizon pairwise replays for baseline and challenger candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from pairwise_regime_mixture_shadow_live import load_strategy_bundle
from search_pair_subset_fractal_genome import load_funding_from_cache_or_empty
from search_pair_subset_regime_mixture import (
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
)
from strategy_replay_dispatch import replay_candidate_from_context


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"
DEFAULT_BASE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
DEFAULT_MODEL = MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"
DEFAULT_CANDIDATE_SOURCE = Path("/tmp/main_execution_beam_tail_guarded_pass2_20260412.json")
DEFAULT_JSON_OUT = Path("/tmp/pairwise_monthly_rolling_6m_20260412.json")
DEFAULT_CSV_OUT = Path("/tmp/pairwise_monthly_rolling_6m_20260412.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly rolling fixed-horizon pairwise replay.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--base-summary", default=str(DEFAULT_BASE_SUMMARY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--candidate-source", default=str(DEFAULT_CANDIDATE_SOURCE))
    parser.add_argument("--candidate-group", default="top_guard_passed")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--start", default="2022-04-06")
    parser.add_argument("--end", default="2025-10-06")
    parser.add_argument("--window-months", type=int, default=6)
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--csv-out", default=str(DEFAULT_CSV_OUT))
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


def clone_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(candidate))


def month_starts(start: str, end: str) -> list[pd.Timestamp]:
    current = pd.Timestamp(start, tz="UTC")
    limit = pd.Timestamp(end, tz="UTC")
    starts: list[pd.Timestamp] = []
    while current <= limit:
        starts.append(current)
        current = current + pd.DateOffset(months=1)
    return starts


def load_candidate_from_source(path: Path, group: str, index: int) -> dict[str, Any]:
    payload = load_json(path)
    rows = payload.get(group) or payload.get("top") or []
    if not rows:
        raise ValueError(f"No candidate rows found in {path}.")
    if not (0 <= int(index) < len(rows)):
        raise IndexError(f"Candidate index {index} out of range for group {group}.")
    row = rows[int(index)]
    candidate = row.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"Missing candidate payload in {path}.")
    candidate = clone_candidate(candidate)
    candidate["candidate_kind"] = str(candidate.get("candidate_kind") or "pairwise_candidate")
    return candidate


def evaluate_candidate_for_window(
    candidate: dict[str, Any],
    *,
    pairs: tuple[str, ...],
    df: pd.DataFrame,
    raw_signal_all: dict[str, pd.Series],
    funding_all: dict[str, pd.DataFrame],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    per_pair: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
        signal_slice = raw_signal_all[pair].loc[df.index[0] : df.index[-1]].copy()
        funding_slice = funding_all[pair]
        if not funding_slice.empty:
            funding_slice = funding_slice[
                (funding_slice["fundingTime"] >= start)
                & (funding_slice["fundingTime"] <= end + pd.Timedelta(days=1))
            ].copy()
        fast_context = build_fast_context(
            df=df,
            pair=pair,
            raw_signal=signal_slice,
            overlay_inputs=overlay_inputs,
            route_thresholds=route_thresholds,
            library_lookup=library_lookup,
            funding_df=funding_slice,
            route_state_mode=str((candidate.get("pair_configs") or {}).get(pair, {}).get("route_state_mode") or "equity_corr"),
        )
        per_pair[pair] = replay_candidate_from_context(
            candidate=candidate,
            pair=pair,
            context=fast_context,
            library_lookup=library_lookup,
            route_thresholds=route_thresholds,
            leaf_runtime_array=None,
            leaf_codes=None,
        )
    return {
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "bars": int(len(df)),
        "per_pair": per_pair,
        "aggregate": aggregate_metrics(per_pair),
    }


def aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "window_count": int(len(rows)),
        "mean_mean_win_rate": float(np.mean([row["aggregate"]["mean_daily_win_rate"] for row in rows])),
        "mean_worst_win_rate": float(np.mean([row["aggregate"]["worst_pair_daily_win_rate"] for row in rows])),
        "mean_mean_total_return": float(np.mean([row["aggregate"]["mean_total_return"] for row in rows])),
        "mean_worst_total_return": float(np.mean([row["aggregate"]["worst_pair_total_return"] for row in rows])),
        "worst_window_total_return": float(min(row["aggregate"]["worst_pair_total_return"] for row in rows)),
        "best_window_total_return": float(max(row["aggregate"]["worst_pair_total_return"] for row in rows)),
        "worst_window_mdd": float(min(row["aggregate"]["worst_max_drawdown"] for row in rows)),
        "best_window_mdd": float(max(row["aggregate"]["worst_max_drawdown"] for row in rows)),
    }


def compare_rows(challenger_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    compare: list[dict[str, Any]] = []
    challenger_by_start = {row["start"]: row for row in challenger_rows}
    baseline_by_start = {row["start"]: row for row in baseline_rows}
    for start in sorted(set(challenger_by_start) & set(baseline_by_start)):
        challenger = challenger_by_start[start]["aggregate"]
        baseline = baseline_by_start[start]["aggregate"]
        compare.append(
            {
                "start": start,
                "delta_mean_win_rate": float(challenger["mean_daily_win_rate"]) - float(baseline["mean_daily_win_rate"]),
                "delta_worst_win_rate": float(challenger["worst_pair_daily_win_rate"]) - float(baseline["worst_pair_daily_win_rate"]),
                "delta_mean_total_return": float(challenger["mean_total_return"]) - float(baseline["mean_total_return"]),
                "delta_worst_total_return": float(challenger["worst_pair_total_return"]) - float(baseline["worst_pair_total_return"]),
                "delta_worst_max_drawdown": float(challenger["worst_max_drawdown"]) - float(baseline["worst_max_drawdown"]),
            }
        )
    if not compare:
        return {"windows": []}
    return {
        "windows": compare,
        "summary": {
            "candidate_beats_main_mean_win_rate_count": int(sum(1 for row in compare if row["delta_mean_win_rate"] > 0.0)),
            "candidate_beats_main_worst_total_return_count": int(sum(1 for row in compare if row["delta_worst_total_return"] > 0.0)),
            "candidate_improves_or_matches_mdd_count": int(sum(1 for row in compare if row["delta_worst_max_drawdown"] >= -1e-12)),
            "mean_delta_mean_win_rate": float(np.mean([row["delta_mean_win_rate"] for row in compare])),
            "mean_delta_worst_total_return": float(np.mean([row["delta_worst_total_return"] for row in compare])),
            "mean_delta_worst_max_drawdown": float(np.mean([row["delta_worst_max_drawdown"] for row in compare])),
        },
    }


def write_csv(
    path: Path,
    *,
    baseline_rows: list[dict[str, Any]],
    challenger_rows: list[dict[str, Any]],
) -> None:
    baseline_by_start = {row["start"]: row for row in baseline_rows}
    challenger_by_start = {row["start"]: row for row in challenger_rows}
    fieldnames = [
        "start",
        "end",
        "baseline_mean_win_rate",
        "baseline_worst_win_rate",
        "baseline_mean_total_return",
        "baseline_worst_total_return",
        "baseline_worst_max_drawdown",
        "candidate_mean_win_rate",
        "candidate_worst_win_rate",
        "candidate_mean_total_return",
        "candidate_worst_total_return",
        "candidate_worst_max_drawdown",
        "delta_mean_win_rate",
        "delta_worst_win_rate",
        "delta_mean_total_return",
        "delta_worst_total_return",
        "delta_worst_max_drawdown",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for start in sorted(set(baseline_by_start) & set(challenger_by_start)):
            baseline = baseline_by_start[start]
            challenger = challenger_by_start[start]
            base_agg = baseline["aggregate"]
            cand_agg = challenger["aggregate"]
            writer.writerow(
                {
                    "start": start,
                    "end": baseline["end"],
                    "baseline_mean_win_rate": base_agg["mean_daily_win_rate"],
                    "baseline_worst_win_rate": base_agg["worst_pair_daily_win_rate"],
                    "baseline_mean_total_return": base_agg["mean_total_return"],
                    "baseline_worst_total_return": base_agg["worst_pair_total_return"],
                    "baseline_worst_max_drawdown": base_agg["worst_max_drawdown"],
                    "candidate_mean_win_rate": cand_agg["mean_daily_win_rate"],
                    "candidate_worst_win_rate": cand_agg["worst_pair_daily_win_rate"],
                    "candidate_mean_total_return": cand_agg["mean_total_return"],
                    "candidate_worst_total_return": cand_agg["worst_pair_total_return"],
                    "candidate_worst_max_drawdown": cand_agg["worst_max_drawdown"],
                    "delta_mean_win_rate": cand_agg["mean_daily_win_rate"] - base_agg["mean_daily_win_rate"],
                    "delta_worst_win_rate": cand_agg["worst_pair_daily_win_rate"] - base_agg["worst_pair_daily_win_rate"],
                    "delta_mean_total_return": cand_agg["mean_total_return"] - base_agg["mean_total_return"],
                    "delta_worst_total_return": cand_agg["worst_pair_total_return"] - base_agg["worst_pair_total_return"],
                    "delta_worst_max_drawdown": cand_agg["worst_max_drawdown"] - base_agg["worst_max_drawdown"],
                }
            )


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    summary = load_json(summary_path)
    baseline_candidate = clone_candidate(summary["selected_candidate"])
    baseline_candidate["candidate_kind"] = "pairwise_candidate"
    challenger_candidate = load_candidate_from_source(
        Path(args.candidate_source),
        group=str(args.candidate_group),
        index=int(args.candidate_index),
    )
    pairs = tuple(summary.get("pairs") or ("BTCUSDT", "BNBUSDT"))
    route_thresholds = tuple(float(v) for v in (summary.get("search", {}).get("route_thresholds") or (0.35, 0.50, 0.65, 0.80)))

    bundle = load_strategy_bundle(summary_path, Path(args.base_summary), Path(args.model), candidate_key="selected_candidate")
    library_lookup = build_library_lookup(bundle["library"])
    compiled = bundle["compiled_model"]

    start_all = pd.Timestamp(args.start, tz="UTC")
    end_all = pd.Timestamp(args.end, tz="UTC") + pd.DateOffset(months=int(args.window_months))
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all.date().isoformat(), end=end_all.date().isoformat(), refresh_cache=False)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {pair: load_funding_from_cache_or_empty(pair, start_all.date().isoformat(), end_all.date().isoformat()) for pair in pairs}

    baseline_rows: list[dict[str, Any]] = []
    challenger_rows: list[dict[str, Any]] = []
    for start in month_starts(args.start, args.end):
        end = start + pd.DateOffset(months=int(args.window_months))
        df = df_all.loc[start:end].copy()
        baseline_rows.append(
            evaluate_candidate_for_window(
                baseline_candidate,
                pairs=pairs,
                df=df,
                raw_signal_all=raw_signal_all,
                funding_all=funding_all,
                library_lookup=library_lookup,
                route_thresholds=route_thresholds,
                start=start,
                end=end,
            )
        )
        challenger_rows.append(
            evaluate_candidate_for_window(
                challenger_candidate,
                pairs=pairs,
                df=df,
                raw_signal_all=raw_signal_all,
                funding_all=funding_all,
                library_lookup=library_lookup,
                route_thresholds=route_thresholds,
                start=start,
                end=end,
            )
        )

    payload = {
        "search": {
            "summary": str(summary_path),
            "base_summary": str(args.base_summary),
            "model": str(args.model),
            "candidate_source": str(args.candidate_source),
            "candidate_group": str(args.candidate_group),
            "candidate_index": int(args.candidate_index),
            "start": str(args.start),
            "end": str(args.end),
            "window_months": int(args.window_months),
        },
        "baseline_candidate": baseline_candidate,
        "challenger_candidate": challenger_candidate,
        "baseline_rows": baseline_rows,
        "challenger_rows": challenger_rows,
        "baseline_summary": aggregate_summary(baseline_rows),
        "challenger_summary": aggregate_summary(challenger_rows),
        "comparison": compare_rows(challenger_rows, baseline_rows),
    }

    json_out = Path(args.json_out)
    json_out.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")
    csv_out = Path(args.csv_out)
    write_csv(csv_out, baseline_rows=baseline_rows, challenger_rows=challenger_rows)
    print(json_out)
    print(csv_out)


if __name__ == "__main__":
    main()
