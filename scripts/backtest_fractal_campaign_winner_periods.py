#!/usr/bin/env python3
"""Backtest a fractal campaign winner across executive horizon windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from derivative_market_data import load_derivative_bundle, slice_derivative_bundle
import gp_crypto_evolution as gp
from backtest_core_stop_loss_compare import resolve_period_windows
from fractal_genome_core import deserialize_tree, evaluate_tree_leaf_codes
from pairwise_regime_mixture_shadow_live import (
    detect_candidate_kind,
    extract_strategy_artifact_reference,
    load_strategy_bundle,
    resolve_strategy_artifact_path,
)
from search_pair_subset_fractal_genome import (
    apply_label_horizon_to_feature_arrays,
    build_leaf_runtime_arrays_for_pair,
    build_market_features,
    load_funding_from_cache_or_empty,
    materialize_feature_arrays,
    project_feature_arrays_by_observation_mode,
)
from search_pair_subset_regime_mixture import aggregate_metrics, build_fast_context, build_library_lookup, build_overlay_inputs
from strategy_replay_dispatch import (
    SUMMARY_WINDOW_LABELS,
    audit_replay_against_candidate_windows,
    replay_candidate_from_context,
    resolve_pairwise_route_state_mode,
)


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FULL_DAY_BARS_5M = gp.periods_per_day(gp.TIMEFRAME)
DEFAULT_BASE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
DEFAULT_MODEL_PATH = MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"
WINDOW_SOURCE_ARTIFACT_LOCKED = "artifact_locked"
WINDOW_SOURCE_CURRENT_MARKET = "current_market"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the best fractal campaign winner across 2m/4m/6m/1y/4y windows.",
    )
    parser.add_argument("--campaign-report", required=True)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--pipeline-report", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--candidate-role", choices=["selected", "baseline"], default="selected")
    parser.add_argument("--base-summary", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--derivative-lookback-days", type=int, default=30)
    parser.add_argument(
        "--window-source",
        choices=(WINDOW_SOURCE_ARTIFACT_LOCKED, WINDOW_SOURCE_CURRENT_MARKET),
        default=WINDOW_SOURCE_ARTIFACT_LOCKED,
        help="Anchor replay windows to the artifact end-date or the latest local market data.",
    )
    parser.add_argument(
        "--allow-truncated-data",
        action="store_true",
        help="Allow replay windows to silently shrink when local data does not fully cover the requested period.",
    )
    parser.add_argument(
        "--no-strict-summary-audit",
        action="store_true",
        help="Do not fail when replayed 2m/6m/4y windows disagree with summary windows.",
    )
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


def resolve_default_bundle_path(raw_path: str | None, anchor_file: Path, default_path: Path) -> Path:
    if raw_path:
        return resolve_strategy_artifact_path(raw_path, anchor_file)
    rel_default = default_path.relative_to(ROOT)
    return resolve_strategy_artifact_path(rel_default, anchor_file)


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


def build_pair_data_coverage(pairs: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    coverage: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        df = gp.load_pair(pair, start=None, end=None, refresh_cache=False)
        if df.empty:
            coverage[pair] = {
                "rows": 0,
                "start": None,
                "end": None,
            }
            continue
        index = pd.DatetimeIndex(df.index)
        start = pd.Timestamp(index.min())
        end = pd.Timestamp(index.max())
        coverage[pair] = {
            "rows": int(len(df)),
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
    return coverage


def infer_artifact_anchor_end_day(selected: Mapping[str, Any]) -> pd.Timestamp | None:
    windows = selected.get("windows") or {}
    ends: list[pd.Timestamp] = []
    for window in windows.values():
        if not isinstance(window, dict):
            continue
        end_raw = window.get("end")
        if not end_raw:
            continue
        ends.append(pd.Timestamp(end_raw).normalize())
    if not ends:
        return None
    return max(ends)


def build_period_anchor_end_day(
    *,
    selected: Mapping[str, Any],
    dataset_end_day: pd.Timestamp,
    window_source: str,
) -> pd.Timestamp:
    if window_source == WINDOW_SOURCE_CURRENT_MARKET:
        return dataset_end_day
    anchor = infer_artifact_anchor_end_day(selected)
    if anchor is None:
        raise RuntimeError(
            "Artifact does not embed replay window end dates. Use --window-source current_market only for ad-hoc analysis."
        )
    return normalize_window_day(anchor, dataset_end_day)


def normalize_window_day(value: Any, reference: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if reference.tzinfo is not None:
        if ts.tzinfo is None:
            return ts.tz_localize(reference.tzinfo)
        return ts.tz_convert(reference.tzinfo)
    if ts.tzinfo is not None:
        return ts.tz_localize(None)
    return ts


def evaluate_summary_periods(
    *,
    summary_path: Path,
    pipeline_path: Path | None,
    base_summary: Path,
    model_path: Path,
    candidate_role: str,
    derivative_lookback_days: int,
    strict_summary_audit: bool,
    window_source: str = WINDOW_SOURCE_ARTIFACT_LOCKED,
    allow_truncated_data: bool = False,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    pipeline = load_json(pipeline_path) if pipeline_path is not None and pipeline_path.exists() else {}
    artifact_provenance = {
        "embedded_model_path_present": bool(summary.get("model_path") or (summary.get("search") or {}).get("model_path")),
        "embedded_base_summary_path_present": bool(
            summary.get("baseline_summary_path")
            or summary.get("base_summary_path")
            or (summary.get("search") or {}).get("base_summary_path")
        ),
    }
    candidate_key = f"{candidate_role}_candidate"
    selected = summary.get(candidate_key) or {}
    if not isinstance(selected, dict) or not selected:
        raise RuntimeError(f"Summary is missing {candidate_key}.")
    pairs = tuple(summary.get("pairs") or ("BTCUSDT", "BNBUSDT"))
    pair_data_coverage = build_pair_data_coverage(pairs)
    bundle = load_strategy_bundle(summary_path, base_summary, model_path, candidate_key=candidate_key)
    candidate_kind = detect_candidate_kind(selected)
    expert_pool = bundle.get("expert_pool") or []
    if candidate_kind == "fractal_tree" and not expert_pool:
        raise RuntimeError("Fractal winner is missing expert pool metadata.")

    df_all = gp.load_all_pairs(pairs=list(pairs), start=None, end=None, refresh_cache=False)
    dataset_end_day = infer_last_complete_day(pd.DatetimeIndex(df_all.index))
    period_anchor_end_day = build_period_anchor_end_day(
        selected=selected,
        dataset_end_day=dataset_end_day,
        window_source=str(window_source),
    )
    if period_anchor_end_day > dataset_end_day and not allow_truncated_data:
        raise RuntimeError(
            "Local data does not fully cover the artifact replay horizon. "
            f"artifact_end_day={period_anchor_end_day.date()} dataset_end_day={dataset_end_day.date()} "
            f"coverage={json.dumps(pair_data_coverage, ensure_ascii=False)}"
        )
    df_all = df_all.loc[: str(dataset_end_day.date())].copy()
    if df_all.empty:
        raise RuntimeError("Filtered dataset is empty.")

    route_thresholds = tuple(float(v) for v in (summary.get("search", {}).get("route_thresholds") or (0.35, 0.50, 0.65, 0.80)))
    library = bundle["library"]
    library_lookup = build_library_lookup(library)
    compiled = bundle["compiled_model"]
    tree = None
    if candidate_kind == "fractal_tree":
        observation_mode = str(selected.get("observation_mode") or "time")
        label_horizon = str(selected.get("label_horizon") or "5m")
        tree_payload = selected.get("tree")
        if not isinstance(tree_payload, dict):
            raise RuntimeError(f"{candidate_key} is missing tree payload.")
        tree = deserialize_tree(tree_payload)
    elif candidate_kind == "pairwise_candidate":
        observation_mode = None
        label_horizon = None
    else:
        raise RuntimeError(f"Unsupported candidate kind for replay: {candidate_kind}")

    first_day = pd.Timestamp(pd.DatetimeIndex(df_all.index).min()).normalize()
    period_reports: list[dict[str, Any]] = []

    def replay_period(label: str, start_day: pd.Timestamp, end_window_day: pd.Timestamp) -> dict[str, Any] | None:
        requested_start = pd.Timestamp(start_day).normalize()
        requested_end = pd.Timestamp(end_window_day).normalize()
        actual_start = max(requested_start, first_day)
        actual_end = min(requested_end, dataset_end_day)
        if (actual_start != requested_start or actual_end != requested_end) and not allow_truncated_data:
            raise RuntimeError(
                "Replay window is not fully covered by local data. "
                f"label={label} requested=({requested_start.date()}..{requested_end.date()}) "
                f"actual=({actual_start.date()}..{actual_end.date()}) "
                f"dataset=({first_day.date()}..{dataset_end_day.date()}) "
                f"coverage={json.dumps(pair_data_coverage, ensure_ascii=False)}"
            )
        if actual_start > actual_end:
            return None
        df = df_all.loc[str(actual_start.date()) : str(actual_end.date())].copy()
        if df.empty:
            return None
        derivatives_by_pair = {
            pair: slice_derivative_bundle(
                load_derivative_bundle(
                    pair,
                    start_dt=actual_start.to_pydatetime(),
                    end_dt=actual_end.to_pydatetime(),
                    fetch=False,
                    lookback_days=int(derivative_lookback_days),
                ),
                start_dt=actual_start.to_pydatetime(),
                end_dt=actual_end.to_pydatetime(),
                history=pd.Timedelta(days=max(8, int(derivative_lookback_days))).to_pytimedelta(),
            )
            for pair in pairs
        }
        raw_signal_all = {
            pair: pd.Series(
                compiled(*gp.get_feature_arrays(df, pair)),
                index=df.index,
                dtype="float64",
            ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            for pair in pairs
        }
        feature_series = build_market_features(df, pairs, derivatives_by_pair)
        if candidate_kind == "fractal_tree":
            feature_arrays = materialize_feature_arrays(feature_series, pd.DatetimeIndex(df.index))
            projected_arrays = project_feature_arrays_by_observation_mode(feature_arrays, observation_mode)
            horizon_arrays = apply_label_horizon_to_feature_arrays(projected_arrays, label_horizon)
            leaf_codes, leaf_catalog = evaluate_tree_leaf_codes(tree, horizon_arrays)
            leaf_runtime_arrays = {
                pair: build_leaf_runtime_arrays_for_pair(
                    pair,
                    leaf_catalog,
                    expert_pool,
                    route_thresholds,
                    len(library),
                )
                for pair in pairs
            }
        else:
            leaf_codes = np.zeros(len(df), dtype="int16")
            leaf_runtime_arrays = {}

        per_pair: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
            funding_df = load_funding_from_cache_or_empty(pair, str(actual_start.date()), str(actual_end.date()))
            route_state_mode = (
                resolve_pairwise_route_state_mode(selected, pair)
                if candidate_kind == "pairwise_candidate"
                else "base"
            )
            context = build_fast_context(
                df=df,
                pair=pair,
                raw_signal=raw_signal_all[pair],
                overlay_inputs=overlay_inputs,
                route_thresholds=route_thresholds,
                library_lookup=library_lookup,
                funding_df=funding_df,
                route_state_mode=route_state_mode,
            )
            per_pair[pair] = replay_candidate_from_context(
                candidate=selected,
                pair=pair,
                context=context,
                library_lookup=library_lookup,
                route_thresholds=route_thresholds,
                leaf_runtime_array=leaf_runtime_arrays.get(pair),
                leaf_codes=leaf_codes,
            )

        return {
            "label": label,
            "requested_start": str(requested_start.date()),
            "requested_end": str(requested_end.date()),
            "start": str(actual_start.date()),
            "end": str(actual_end.date()),
            "bars": int(len(df)),
            "per_pair": per_pair,
            "aggregate": aggregate_metrics(per_pair),
        }

    for label, window_start in resolve_period_windows(period_anchor_end_day):
        report = replay_period(label, normalize_window_day(window_start, period_anchor_end_day), period_anchor_end_day)
        if report is not None:
            period_reports.append(report)

    replay_contract_periods: list[dict[str, Any]] = []
    for label, summary_label in SUMMARY_WINDOW_LABELS.items():
        window = (selected.get("windows") or {}).get(summary_label)
        if not isinstance(window, dict):
            continue
        start_raw = window.get("start")
        end_raw = window.get("end")
        if not start_raw or not end_raw:
            continue
        report = replay_period(
            label,
            normalize_window_day(start_raw, dataset_end_day),
            normalize_window_day(end_raw, dataset_end_day),
        )
        if report is not None:
            replay_contract_periods.append(report)

    replay_audit = audit_replay_against_candidate_windows(replay_contract_periods, selected)
    if strict_summary_audit and replay_audit["status"] == "mismatch":
        provenance_hint = ""
        if not artifact_provenance["embedded_model_path_present"]:
            provenance_hint = " Summary does not embed model_path; replay may be using a moved default model."
        raise RuntimeError(
            "Replay summary audit failed: "
            + json.dumps(replay_audit["mismatches"], ensure_ascii=False)
            + provenance_hint
        )

    return {
        "strategy_class": "fractal_campaign_winner_period_backtest",
        "campaign_report": str(summary_path),
        "candidate_role": candidate_role,
        "selected_candidate": {
            "candidate_kind": candidate_kind,
            "tree_key": selected.get("tree_key"),
            "observation_mode": observation_mode,
            "label_horizon": label_horizon,
            "tree_depth": selected.get("tree_depth"),
            "logic_depth": selected.get("logic_depth"),
        },
        "promotion_decision": pipeline.get("decision"),
        "artifact_provenance": artifact_provenance,
        "input_contract": {
            "window_source": str(window_source),
            "allow_truncated_data": bool(allow_truncated_data),
            "period_anchor_end_day": str(period_anchor_end_day.date()),
            "dataset_first_day": str(first_day.date()),
            "dataset_last_complete_day": str(dataset_end_day.date()),
            "pair_data_coverage": pair_data_coverage,
        },
        "replay_audit": replay_audit,
        "replay_contract_periods": replay_contract_periods,
        "periods": period_reports,
    }


def main() -> None:
    args = parse_args()
    campaign_report_path = Path(args.campaign_report)
    campaign_report = load_json(campaign_report_path)
    entry = select_candidate_entry(campaign_report, args.seed)
    artifacts = entry.get("artifacts") or {}
    summary_ref = args.summary_path or artifacts.get("search_summary")
    if not summary_ref:
        raise RuntimeError("Unable to resolve summary path from campaign report.")
    summary_path = resolve_strategy_artifact_path(summary_ref, campaign_report_path)
    summary_payload = load_json(summary_path)
    pipeline_path_str = args.pipeline_report or artifacts.get("pipeline_report")
    pipeline_path = resolve_strategy_artifact_path(pipeline_path_str, campaign_report_path) if pipeline_path_str else None
    embedded_base_summary = extract_strategy_artifact_reference(
        summary_payload,
        "baseline_summary_path",
        "base_summary_path",
    )
    embedded_model = extract_strategy_artifact_reference(summary_payload, "model_path")
    base_summary_ref = args.base_summary or embedded_base_summary
    if not base_summary_ref:
        raise RuntimeError(
            "Summary does not embed baseline_summary_path/base_summary_path. Pass --base-summary explicitly."
        )
    model_ref = args.model or embedded_model
    if not model_ref:
        raise RuntimeError(
            "Summary does not embed model_path. Pass --model explicitly to reproduce the artifact deterministically."
        )
    report = evaluate_summary_periods(
        summary_path=summary_path,
        pipeline_path=pipeline_path,
        base_summary=resolve_strategy_artifact_path(base_summary_ref, summary_path),
        model_path=resolve_strategy_artifact_path(model_ref, summary_path),
        candidate_role=str(args.candidate_role),
        derivative_lookback_days=int(args.derivative_lookback_days),
        strict_summary_audit=not bool(args.no_strict_summary_audit),
        window_source=str(args.window_source),
        allow_truncated_data=bool(args.allow_truncated_data),
    )
    output_path = Path(args.summary_out)
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
