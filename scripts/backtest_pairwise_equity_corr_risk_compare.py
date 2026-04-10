#!/usr/bin/env python3
"""Compare baseline pairwise replay against BTC-equity correlation risk adjustment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import gp_crypto_evolution as gp
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    build_overlay_inputs,
    realistic_overlay_replay,
)


UTC = timezone.utc
FUNDING_RANGE_START = "2022-04-06"
FUNDING_RANGE_END = "2026-04-06"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest pairwise BTC-equity correlation risk adjustment.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=gp.MODELS_DIR / "pairwise_equity_corr_risk_compare.json",
    )
    return parser.parse_args()


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def load_funding_cache(pair: str) -> pd.DataFrame:
    path = gp.DATA_DIR / f"{pair}_funding_{FUNDING_RANGE_START}_{FUNDING_RANGE_END}.csv"
    df = pd.read_csv(path)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, format="mixed")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df.dropna(subset=["fundingTime", "fundingRate"]).sort_values("fundingTime").reset_index(drop=True)


def filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def filter_funding_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1)
    return df.loc[(df["fundingTime"] >= start_ts) & (df["fundingTime"] < end_ts)].copy()


def aggregate_pair_reports(pair_reports: dict[str, dict[str, Any]], section: str) -> dict[str, Any]:
    rows = [report[section] for report in pair_reports.values()]
    return {
        "mean_total_return": float(sum(float(item["total_return"]) for item in rows) / len(rows)),
        "worst_total_return": float(min(float(item["total_return"]) for item in rows)),
        "mean_avg_daily_return": float(sum(float(item["avg_daily_return"]) for item in rows) / len(rows)),
        "worst_avg_daily_return": float(min(float(item["avg_daily_return"]) for item in rows)),
        "mean_sharpe": float(sum(float(item["sharpe"]) for item in rows) / len(rows)),
        "worst_max_drawdown": float(min(float(item["max_drawdown"]) for item in rows)),
        "mean_max_drawdown": float(sum(float(item["max_drawdown"]) for item in rows) / len(rows)),
        "total_trades": int(sum(int(item["n_trades"]) for item in rows)),
    }


def build_delta_report(pair_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_return_improved = 0
    max_drawdown_improved = 0
    sharpe_improved = 0
    for item in pair_reports.values():
        base = item["baseline"]
        adj = item["corr_risk"]
        if float(adj["total_return"]) > float(base["total_return"]):
            total_return_improved += 1
        if float(adj["max_drawdown"]) > float(base["max_drawdown"]):
            max_drawdown_improved += 1
        if float(adj["sharpe"]) > float(base["sharpe"]):
            sharpe_improved += 1
    return {
        "pair_count": int(len(pair_reports)),
        "total_return_improved_pairs": int(total_return_improved),
        "max_drawdown_improved_pairs": int(max_drawdown_improved),
        "sharpe_improved_pairs": int(sharpe_improved),
    }


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_path.read_text())
    config = summary["selected_candidate"]["pair_configs"]
    library = list(iter_params())
    model_tree, _ = load_signal_model(args.model_path)
    compiled = gp.toolbox.compile(expr=model_tree)

    start_all = DEFAULT_WINDOWS[-1][1]
    end_all = DEFAULT_WINDOWS[-1][2]
    df_all = gp.load_all_pairs(pairs=list(PAIRS), start=start_all, end=end_all, refresh_cache=False)
    funding_cache = {pair: load_funding_cache(pair) for pair in PAIRS}

    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "strategy_class": "pairwise_equity_corr_risk_compare",
        "summary_path": str(args.summary_path),
        "model_path": str(args.model_path),
        "windows": {},
    }

    for label, start, end in DEFAULT_WINDOWS:
        df_window = filter_window(df_all, start, end)
        pair_reports: dict[str, Any] = {}
        for pair in PAIRS:
            raw_signal = pd.Series(
                compiled(*gp.get_feature_arrays(df_window, pair)),
                index=df_window.index,
                dtype="float64",
            )
            overlay_inputs = build_overlay_inputs(df_window, PAIRS, regime_pair=pair)
            funding_df = filter_funding_window(funding_cache[pair], start, end)
            baseline = realistic_overlay_replay(
                df_window,
                pair,
                raw_signal,
                overlay_inputs,
                funding_df,
                library,
                tuple(int(v) for v in config[pair]["mapping_indices"]),
                float(config[pair]["route_breadth_threshold"]),
                use_equity_corr_risk=False,
            )
            corr_risk = realistic_overlay_replay(
                df_window,
                pair,
                raw_signal,
                overlay_inputs,
                funding_df,
                library,
                tuple(int(v) for v in config[pair]["mapping_indices"]),
                float(config[pair]["route_breadth_threshold"]),
                use_equity_corr_risk=True,
            )
            pair_reports[pair] = {
                "equity_corr_context": overlay_inputs.get("equity_corr_context"),
                "equity_corr_source_mode": overlay_inputs.get("equity_corr_source_mode"),
                "baseline": baseline,
                "corr_risk": corr_risk,
                "delta": {
                    "total_return": float(corr_risk["total_return"]) - float(baseline["total_return"]),
                    "avg_daily_return": float(corr_risk["avg_daily_return"]) - float(baseline["avg_daily_return"]),
                    "sharpe": float(corr_risk["sharpe"]) - float(baseline["sharpe"]),
                    "max_drawdown": float(corr_risk["max_drawdown"]) - float(baseline["max_drawdown"]),
                    "n_trades": int(corr_risk["n_trades"]) - int(baseline["n_trades"]),
                },
            }

        report["windows"][label] = {
            "start": start,
            "end": end,
            "pairs": pair_reports,
            "baseline_aggregate": aggregate_pair_reports(pair_reports, "baseline"),
            "corr_risk_aggregate": aggregate_pair_reports(pair_reports, "corr_risk"),
            "delta_summary": build_delta_report(pair_reports),
        }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(json_safe(report), indent=2))
    print(json.dumps(json_safe(report), indent=2))


if __name__ == "__main__":
    main()
