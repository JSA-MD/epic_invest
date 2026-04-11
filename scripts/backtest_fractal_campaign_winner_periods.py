#!/usr/bin/env python3
"""Backtest a fractal campaign winner across executive horizon windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from backtest_core_stop_loss_compare import resolve_period_windows
from fractal_genome_core import deserialize_tree, evaluate_tree_leaf_codes
from pairwise_regime_mixture_shadow_live import load_strategy_bundle
from search_pair_subset_fractal_genome import (
    apply_label_horizon_to_feature_arrays,
    build_leaf_runtime_arrays_for_pair,
    build_market_features,
    fast_fractal_replay_from_context,
    load_funding_from_cache_or_empty,
    materialize_feature_arrays,
    project_feature_arrays_by_observation_mode,
)
from search_pair_subset_regime_mixture import aggregate_metrics, build_fast_context, build_library_lookup, build_overlay_inputs


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FULL_DAY_BARS_5M = gp.periods_per_day(gp.TIMEFRAME)
DEFAULT_BASE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
DEFAULT_MODEL_PATH = MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the best fractal campaign winner across 2m/4m/6m/1y/4y windows.",
    )
    parser.add_argument("--campaign-report", required=True)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--pipeline-report", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--base-summary", default=str(DEFAULT_BASE_SUMMARY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "fractal_campaign_winner_periods.json"),
    )
    return parser.parse_args()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def select_candidate_entry(campaign_report: dict[str, Any], seed: int | None) -> dict[str, Any]:
    if seed is None:
        best = campaign_report.get("best_candidate")
        if isinstance(best, dict):
            return best
        raise ValueError("Campaign report does not contain best_candidate.")

    for section in ("jobs", "top_candidates"):
        for item in campaign_report.get(section) or []:
            if int(item.get("seed", -1)) == int(seed):
                return item
    raise ValueError(f"Seed {seed} not found in campaign report.")


def infer_last_complete_day(index: pd.DatetimeIndex) -> pd.Timestamp:
    day_counts = pd.Series(1, index=index).groupby(index.normalize()).sum()
    if day_counts.empty:
        raise RuntimeError("Unable to infer last complete day.")
    last_day = pd.Timestamp(day_counts.index[-1]).normalize()
    if int(day_counts.iloc[-1]) < FULL_DAY_BARS_5M:
        return last_day - pd.Timedelta(days=1)
    return last_day


def evaluate_summary_periods(
    *,
    summary_path: Path,
    pipeline_path: Path | None,
    base_summary: Path,
    model_path: Path,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    pipeline = load_json(pipeline_path) if pipeline_path is not None and pipeline_path.exists() else {}
    selected = summary.get("selected_candidate") or {}
    pairs = tuple(summary.get("pairs") or ("BTCUSDT", "BNBUSDT"))
    bundle = load_strategy_bundle(summary_path, base_summary, model_path)
    expert_pool = bundle.get("expert_pool") or []
    if not expert_pool:
        raise RuntimeError("Fractal winner is missing expert pool metadata.")

    df_all = gp.load_all_pairs(pairs=list(pairs), start=None, end=None, refresh_cache=False)
    end_day = infer_last_complete_day(pd.DatetimeIndex(df_all.index))
    df_all = df_all.loc[: str(end_day.date())].copy()
    if df_all.empty:
        raise RuntimeError("Filtered dataset is empty.")

    tree_payload = selected.get("tree")
    if not isinstance(tree_payload, dict):
        raise RuntimeError("Selected candidate is missing tree payload.")
    tree = deserialize_tree(tree_payload)
    observation_mode = str(selected.get("observation_mode") or "time")
    label_horizon = str(selected.get("label_horizon") or "5m")
    route_thresholds = tuple(float(v) for v in (summary.get("search", {}).get("route_thresholds") or (0.35, 0.50, 0.65, 0.80)))
    library = bundle["library"]
    library_lookup = build_library_lookup(library)
    compiled = bundle["compiled_model"]

    first_day = pd.Timestamp(pd.DatetimeIndex(df_all.index).min()).normalize()
    period_reports: list[dict[str, Any]] = []
    for label, window_start in resolve_period_windows(end_day):
        actual_start = max(pd.Timestamp(window_start).normalize(), first_day)
        df = df_all.loc[str(actual_start.date()) : str(end_day.date())].copy()
        if df.empty:
            continue
        raw_signal_all = {
            pair: pd.Series(
                compiled(*gp.get_feature_arrays(df, pair)),
                index=df.index,
                dtype="float64",
            ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            for pair in pairs
        }
        feature_series = build_market_features(df, pairs)
        feature_arrays = materialize_feature_arrays(feature_series, pd.DatetimeIndex(df.index))
        projected_arrays = project_feature_arrays_by_observation_mode(feature_arrays, observation_mode)
        horizon_arrays = apply_label_horizon_to_feature_arrays(projected_arrays, label_horizon)
        leaf_codes, leaf_catalog = evaluate_tree_leaf_codes(tree, horizon_arrays)

        per_pair: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
            funding_df = load_funding_from_cache_or_empty(pair, str(actual_start.date()), str(end_day.date()))
            context = build_fast_context(
                df=df,
                pair=pair,
                raw_signal=raw_signal_all[pair],
                overlay_inputs=overlay_inputs,
                route_thresholds=route_thresholds,
                library_lookup=library_lookup,
                funding_df=funding_df,
            )
            leaf_runtime = build_leaf_runtime_arrays_for_pair(
                pair,
                leaf_catalog,
                expert_pool,
                route_thresholds,
                len(library),
            )
            per_pair[pair] = fast_fractal_replay_from_context(
                context,
                library_lookup,
                route_thresholds,
                leaf_runtime,
                leaf_codes,
            )

        period_reports.append(
            {
                "label": label,
                "start": str(actual_start.date()),
                "end": str(end_day.date()),
                "bars": int(len(df)),
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
        )

    return {
        "strategy_class": "fractal_campaign_winner_period_backtest",
        "campaign_report": str(summary_path),
        "selected_candidate": {
            "tree_key": selected.get("tree_key"),
            "observation_mode": observation_mode,
            "label_horizon": label_horizon,
            "tree_depth": selected.get("tree_depth"),
            "logic_depth": selected.get("logic_depth"),
        },
        "promotion_decision": pipeline.get("decision"),
        "periods": period_reports,
    }


def main() -> None:
    args = parse_args()
    campaign_report = load_json(args.campaign_report)
    entry = select_candidate_entry(campaign_report, args.seed)
    artifacts = entry.get("artifacts") or {}
    summary_path = Path(args.summary_path or artifacts.get("search_summary"))
    pipeline_path_str = args.pipeline_report or artifacts.get("pipeline_report")
    pipeline_path = Path(pipeline_path_str) if pipeline_path_str else None
    report = evaluate_summary_periods(
        summary_path=summary_path,
        pipeline_path=pipeline_path,
        base_summary=Path(args.base_summary),
        model_path=Path(args.model),
    )
    output_path = Path(args.summary_out)
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
