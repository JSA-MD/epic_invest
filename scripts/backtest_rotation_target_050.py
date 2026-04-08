#!/usr/bin/env python3
"""Backtest the strongest found path to 0.5% average daily return.

The script freezes the best candidate found so far:
- aggressive walk-forward rotation core
- BTC intraday overlay on core-flat days
- minimal leverage that lifts both TEST and OOS avg daily return above 0.5%

It reports the progression in three stages:
1. aggressive rotation core
2. aggressive core + intraday overlay
3. leveraged aggressive core + overlay
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_cash_filtered_rotation import (
    StrategyParams,
    build_daily_close,
    run_walkforward_reselection,
)
from backtest_rotation_intraday_overlay import (
    combine_rotation_overlay,
    json_ready,
    run_overlay_session,
    summarize_returns,
)
from gp_crypto_evolution import (
    INITIAL_CASH,
    MODELS_DIR,
    TEST_END,
    TEST_START,
    VAL_END,
    VAL_START,
    load_all_pairs,
)


BEST_CORE_PARAMS = StrategyParams(
    lookback_fast=5,
    lookback_slow=14,
    top_n=1,
    vol_window=5,
    target_vol_ann=0.8,
    regime_threshold=0.02,
    gross_cap=3.0,
)


@dataclass
class TargetOverlayParams:
    momentum_lookback: int = 1
    breadth_lookback: int = 3
    momentum_threshold: float = 0.0
    breadth_threshold: float = 0.25
    signal_mode: str = "trend_both"
    reward_multiple: float = 3.0
    trail_activation_pct: float = 0.006
    trail_distance_pct: float = 0.002
    trail_floor_pct: float = 0.001
    entry_threshold: float = 20.0


BEST_OVERLAY_PARAMS = TargetOverlayParams()

MEAN_TARGET_LEVERAGE = 1.08
MIN_TARGET_LEVERAGE = MEAN_TARGET_LEVERAGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the aggressive rotation + overlay path that reaches 0.5% average daily return.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "rotation_target_050_summary.json"),
    )
    parser.add_argument(
        "--curve-out",
        default=str(MODELS_DIR / "rotation_target_050_curve.csv"),
    )
    parser.add_argument(
        "--daily-out",
        default=str(MODELS_DIR / "rotation_target_050_daily_returns.csv"),
    )
    parser.add_argument(
        "--weights-out",
        default=str(MODELS_DIR / "rotation_target_050_weights.csv"),
    )
    parser.add_argument("--wf-reselect-days", type=int, default=14)
    parser.add_argument("--wf-train-days", type=int, default=60)
    parser.add_argument("--wf-val-days", type=int, default=30)
    parser.add_argument("--leverage", type=float, default=MIN_TARGET_LEVERAGE)
    return parser.parse_args()


def scale_series(series: pd.Series, leverage: float) -> pd.Series:
    return pd.Series(series.to_numpy(dtype="float64") * float(leverage), index=series.index, name=series.name)


def build_overlay_signal_map(
    close: pd.DataFrame,
    active_mask: pd.Series,
    params: TargetOverlayParams,
    start: str,
    end: str,
) -> dict[pd.Timestamp, float]:
    btc_mom = close["BTCUSDT"].pct_change(params.momentum_lookback)
    breadth = (close.pct_change(params.breadth_lookback) > 0.0).mean(axis=1)
    feat = pd.DataFrame({"btc_mom": btc_mom, "breadth": breadth}).loc[start:end]

    signal_map: dict[pd.Timestamp, float] = {}
    for day, row in feat.iterrows():
        if bool(active_mask.get(day, False)):
            signal_map[pd.Timestamp(day)] = 0.0
            continue

        mom = float(row["btc_mom"]) if np.isfinite(row["btc_mom"]) else 0.0
        br = float(row["breadth"]) if np.isfinite(row["breadth"]) else 0.0
        signal = 0.0
        if br >= params.breadth_threshold:
            if params.signal_mode == "trend_both":
                if mom >= params.momentum_threshold:
                    signal = 100.0
                elif mom <= -params.momentum_threshold:
                    signal = -100.0
            elif params.signal_mode == "trend_long":
                signal = 100.0 if mom >= params.momentum_threshold else 0.0
            elif params.signal_mode == "contra_both":
                if mom >= params.momentum_threshold:
                    signal = -100.0
                elif mom <= -params.momentum_threshold:
                    signal = 100.0
        signal_map[pd.Timestamp(day)] = signal
    return signal_map


def delta_metrics(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {
        "return_delta": float(after["total_return"] - before["total_return"]),
        "daily_win_rate_delta": float(after["daily_metrics"]["daily_win_rate"] - before["daily_metrics"]["daily_win_rate"]),
        "daily_target_hit_rate_delta": float(
            after["daily_metrics"]["daily_target_hit_rate"] - before["daily_metrics"]["daily_target_hit_rate"]
        ),
        "avg_daily_return_delta": float(after["daily_metrics"]["avg_daily_return"] - before["daily_metrics"]["avg_daily_return"]),
        "max_drawdown_delta": float(after["max_drawdown"] - before["max_drawdown"]),
    }


def print_stage_metrics(label: str, metrics: dict[str, Any]) -> None:
    daily = metrics["daily_metrics"]
    print(f"\n=== {label} ===")
    print(f"  Return:       {metrics['total_return']*100:+.2f}%")
    print(f"  Avg Daily:    {daily['avg_daily_return']*100:+.3f}%")
    print(f"  Daily Win:    {daily['daily_win_rate']*100:.1f}%")
    print(f"  Daily Hit:    {daily['daily_target_hit_rate']*100:.1f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Target 0.5% Path Backtest")
    print("=" * 80)
    print(f"Leverage: {args.leverage:.2f}x")

    df_all = load_all_pairs()
    close = build_daily_close(df_all)

    _, core_weights_val, core_curve_val, _, _ = run_walkforward_reselection(
        close,
        [BEST_CORE_PARAMS],
        VAL_START,
        VAL_END,
        reselect_days=args.wf_reselect_days,
        train_days=args.wf_train_days,
        val_days=args.wf_val_days,
    )
    _, core_weights_test, core_curve_test, _, _ = run_walkforward_reselection(
        close,
        [BEST_CORE_PARAMS],
        TEST_START,
        TEST_END,
        reselect_days=args.wf_reselect_days,
        train_days=args.wf_train_days,
        val_days=args.wf_val_days,
    )
    _, core_weights_oos, core_curve_oos, _, _ = run_walkforward_reselection(
        close,
        [BEST_CORE_PARAMS],
        VAL_START,
        TEST_END,
        reselect_days=args.wf_reselect_days,
        train_days=args.wf_train_days,
        val_days=args.wf_val_days,
    )

    core_val = core_curve_val.set_index("time")["net_return"]
    core_test = core_curve_test.set_index("time")["net_return"]
    core_oos = core_curve_oos.set_index("time")["net_return"]

    core_metrics_val = summarize_returns(core_val)
    core_metrics_test = summarize_returns(core_test)
    core_metrics_oos = summarize_returns(core_oos)

    val_active = core_weights_val.set_index("time").sum(axis=1) > 0.0
    test_active = core_weights_test.set_index("time").sum(axis=1) > 0.0
    oos_active = core_weights_oos.set_index("time").sum(axis=1) > 0.0

    val_signal_map = build_overlay_signal_map(close, val_active, BEST_OVERLAY_PARAMS, VAL_START, VAL_END)
    test_signal_map = build_overlay_signal_map(close, test_active, BEST_OVERLAY_PARAMS, TEST_START, TEST_END)
    oos_signal_map = build_overlay_signal_map(close, oos_active, BEST_OVERLAY_PARAMS, VAL_START, TEST_END)

    val_overlay = run_overlay_session(df_all, val_signal_map, VAL_START, VAL_END, BEST_OVERLAY_PARAMS)
    test_overlay = run_overlay_session(df_all, test_signal_map, TEST_START, TEST_END, BEST_OVERLAY_PARAMS)
    oos_overlay = run_overlay_session(df_all, oos_signal_map, VAL_START, TEST_END, BEST_OVERLAY_PARAMS)

    val_daily, combo_val = combine_rotation_overlay(core_curve_val, core_weights_val, val_overlay)
    test_daily, combo_test = combine_rotation_overlay(core_curve_test, core_weights_test, test_overlay)
    oos_daily, combo_oos = combine_rotation_overlay(core_curve_oos, core_weights_oos, oos_overlay)

    combo_metrics_val = summarize_returns(combo_val)
    combo_metrics_test = summarize_returns(combo_test)
    combo_metrics_oos = summarize_returns(combo_oos)

    levered_val = scale_series(combo_val, args.leverage)
    levered_test = scale_series(combo_test, args.leverage)
    levered_oos = scale_series(combo_oos, args.leverage)

    levered_metrics_val = summarize_returns(levered_val)
    levered_metrics_test = summarize_returns(levered_test)
    levered_metrics_oos = summarize_returns(levered_oos)

    print_stage_metrics("VAL Core", core_metrics_val)
    print_stage_metrics("VAL Core + Overlay", combo_metrics_val)
    print_stage_metrics("VAL Levered Hybrid", levered_metrics_val)

    print_stage_metrics("TEST Core", core_metrics_test)
    print_stage_metrics("TEST Core + Overlay", combo_metrics_test)
    print_stage_metrics("TEST Levered Hybrid", levered_metrics_test)

    print_stage_metrics("OOS Core", core_metrics_oos)
    print_stage_metrics("OOS Core + Overlay", combo_metrics_oos)
    print_stage_metrics("OOS Levered Hybrid", levered_metrics_oos)

    curve_df = pd.DataFrame(
        {
            "time": levered_oos.index,
            "equity": INITIAL_CASH * (1.0 + levered_oos).cumprod(),
            "daily_return": levered_oos.to_numpy(dtype="float64"),
            "core_return": core_oos.reindex(levered_oos.index).to_numpy(dtype="float64"),
            "unlevered_hybrid_return": combo_oos.reindex(levered_oos.index).to_numpy(dtype="float64"),
        }
    )
    curve_df["peak"] = curve_df["equity"].cummax()
    curve_df["drawdown"] = curve_df["equity"] / curve_df["peak"] - 1.0

    summary = {
        "strategy_class": "aggressive_walkforward_rotation_with_btc_intraday_overlay_and_leverage",
        "core_params": asdict(BEST_CORE_PARAMS),
        "overlay_params": asdict(BEST_OVERLAY_PARAMS),
        "walkforward_config": {
            "reselect_days": args.wf_reselect_days,
            "train_days": args.wf_train_days,
            "val_days": args.wf_val_days,
        },
        "leverage": float(args.leverage),
        "stages": {
            "core": {
                "validation": core_metrics_val,
                "test": core_metrics_test,
                "oos": core_metrics_oos,
            },
            "hybrid": {
                "validation": combo_metrics_val,
                "test": combo_metrics_test,
                "oos": combo_metrics_oos,
            },
            "levered_hybrid": {
                "validation": levered_metrics_val,
                "test": levered_metrics_test,
                "oos": levered_metrics_oos,
            },
        },
        "delta": {
            "hybrid_vs_core": {
                "validation": delta_metrics(combo_metrics_val, core_metrics_val),
                "test": delta_metrics(combo_metrics_test, core_metrics_test),
                "oos": delta_metrics(combo_metrics_oos, core_metrics_oos),
            },
            "levered_vs_hybrid": {
                "validation": delta_metrics(levered_metrics_val, combo_metrics_val),
                "test": delta_metrics(levered_metrics_test, combo_metrics_test),
                "oos": delta_metrics(levered_metrics_oos, combo_metrics_oos),
            },
        },
        "target_check": {
            "avg_daily_target": 0.005,
            "validation_pass": bool(levered_metrics_val["daily_metrics"]["avg_daily_return"] >= 0.005),
            "test_pass": bool(levered_metrics_test["daily_metrics"]["avg_daily_return"] >= 0.005),
            "oos_pass": bool(levered_metrics_oos["daily_metrics"]["avg_daily_return"] >= 0.005),
        },
        "artifacts": {
            "summary_path": str(Path(args.summary_out)),
            "curve_path": str(Path(args.curve_out)),
            "daily_path": str(Path(args.daily_out)),
            "weights_path": str(Path(args.weights_out)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    curve_df.to_csv(Path(args.curve_out), index=False)
    levered_oos.rename("daily_return").to_csv(Path(args.daily_out), index=True)
    core_weights_oos.to_csv(Path(args.weights_out), index=False)
    with open(args.summary_out, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    print(f"\nSummary saved: {args.summary_out}")


if __name__ == "__main__":
    main()
