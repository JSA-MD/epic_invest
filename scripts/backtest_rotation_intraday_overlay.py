#!/usr/bin/env python3
"""Walk-forward rotation with intraday BTC overlay on cash days.

The strategy keeps the existing walk-forward cash rotation as the core engine.
When the rotation stays flat on a given day, a second layer can use the idle
capital for a BTC intraday session if the day opens in a broad bullish regime.

The overlay rule is intentionally simple and validation-selected:
- only on rotation-flat days
- go long BTC for the session if 3-day BTC momentum >= 0
- and 5-day cross-asset breadth >= 0.50
- manage the day using the existing daily-session execution engine
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
    build_daily_close,
    candidate_params,
    run_walkforward_reselection,
)
from gp_crypto_evolution import (
    INITIAL_CASH,
    MODELS_DIR,
    PRIMARY_PAIR,
    TEST_END,
    TEST_START,
    TRAIN_START,
    VAL_END,
    VAL_START,
    daily_session_backtest,
    load_all_pairs,
    summarize_monthly_returns,
    summarize_period_returns,
)


@dataclass
class OverlayParams:
    momentum_lookback: int = 3
    breadth_lookback: int = 5
    momentum_threshold: float = 0.0
    breadth_threshold: float = 0.50
    reward_multiple: float = 3.0
    trail_activation_pct: float = 0.006
    trail_distance_pct: float = 0.002
    trail_floor_pct: float = 0.001
    entry_threshold: float = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the walk-forward rotation plus BTC intraday overlay strategy.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "rotation_intraday_overlay_summary.json"),
    )
    parser.add_argument(
        "--curve-out",
        default=str(MODELS_DIR / "rotation_intraday_overlay_curve.csv"),
    )
    parser.add_argument(
        "--daily-out",
        default=str(MODELS_DIR / "rotation_intraday_overlay_daily_returns.csv"),
    )
    parser.add_argument(
        "--weights-out",
        default=str(MODELS_DIR / "rotation_intraday_overlay_weights.csv"),
    )
    parser.add_argument("--wf-reselect-days", type=int, default=14)
    parser.add_argument("--wf-train-days", type=int, default=60)
    parser.add_argument("--wf-val-days", type=int, default=30)
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def summarize_returns(series: pd.Series, initial_cash: float = INITIAL_CASH) -> dict[str, Any]:
    if series.empty:
        daily_metrics = summarize_period_returns(np.asarray([], dtype="float64"))
        monthly_metrics = summarize_monthly_returns(np.asarray([], dtype="float64"), pd.DatetimeIndex([]))
        return {
            "total_return": 0.0,
            "final_equity": initial_cash,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "daily_metrics": daily_metrics,
            "monthly_metrics": monthly_metrics,
            "active_ratio": 0.0,
            "n_trades": 0,
        }

    arr = series.to_numpy(dtype="float64")
    eq = float(initial_cash) * np.cumprod(1.0 + arr)
    curve = np.concatenate([[initial_cash], eq])
    peak = np.maximum.accumulate(curve)
    max_drawdown = float(np.min(curve / peak - 1.0))
    if len(arr) > 1 and np.std(arr) > 1e-12:
        sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(365.25))
    else:
        sharpe = 0.0
    daily_metrics = summarize_period_returns(arr)
    monthly_metrics = summarize_monthly_returns(arr, pd.DatetimeIndex(series.index))
    return {
        "total_return": float(eq[-1] / initial_cash - 1.0),
        "final_equity": float(eq[-1]),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
        "active_ratio": float(np.mean(arr != 0.0)),
        "n_trades": int(np.sum(arr != 0.0)),
    }


def build_overlay_signal_map(
    close: pd.DataFrame,
    active_mask: pd.Series,
    params: OverlayParams,
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
        if mom >= params.momentum_threshold and br >= params.breadth_threshold:
            signal_map[pd.Timestamp(day)] = 100.0
        else:
            signal_map[pd.Timestamp(day)] = 0.0
    return signal_map


def run_overlay_session(
    df_all: pd.DataFrame,
    signal_map: dict[pd.Timestamp, float],
    start: str,
    end: str,
    params: OverlayParams,
) -> pd.Series:
    sl = df_all.loc[start:end].copy()
    idx_days = pd.DatetimeIndex(sl.index.normalize())
    desired = idx_days.map(signal_map).to_numpy(dtype="float64")
    result = daily_session_backtest(
        sl,
        desired,
        pair=PRIMARY_PAIR,
        reward_multiple=params.reward_multiple,
        trail_activation_pct=params.trail_activation_pct,
        trail_distance_pct=params.trail_distance_pct,
        trail_floor_pct=params.trail_floor_pct,
        entry_threshold=params.entry_threshold,
    )
    days = pd.DatetimeIndex(pd.DatetimeIndex(sl.index.normalize().unique()))
    return pd.Series(result["net_ret"], index=days, name="overlay_return")


def combine_rotation_overlay(
    rotation_curve: pd.DataFrame,
    rotation_weights: pd.DataFrame,
    overlay_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    rot_ret = rotation_curve.set_index("time")["net_return"].copy()
    weights = rotation_weights.set_index("time").copy()
    active = weights.sum(axis=1) > 0.0
    combo = rot_ret.copy()
    flat_days = ~active.reindex(combo.index).fillna(False)
    combo.loc[flat_days] = combo.loc[flat_days] + overlay_returns.reindex(combo.index).fillna(0.0).loc[flat_days]

    details = pd.DataFrame(
        {
            "time": combo.index,
            "rotation_return": rot_ret.reindex(combo.index).to_numpy(dtype="float64"),
            "rotation_active": active.reindex(combo.index).fillna(False).to_numpy(dtype=bool),
            "overlay_return": overlay_returns.reindex(combo.index).fillna(0.0).to_numpy(dtype="float64"),
            "combined_return": combo.to_numpy(dtype="float64"),
        }
    )
    return details, combo


def print_metrics(label: str, metrics: dict[str, Any]) -> None:
    daily = metrics["daily_metrics"]
    monthly = metrics["monthly_metrics"]
    print(f"\n=== {label} ===")
    print(f"  Return:       {metrics['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Avg Daily:    {daily['avg_daily_return']*100:+.2f}%")
    print(f"  Daily Win:    {daily['daily_win_rate']*100:.1f}%")
    print(f"  Daily Hit:    {daily['daily_target_hit_rate']*100:.1f}%")
    print(f"  Monthly Hit:  {monthly['monthly_target_hit_rate']*100:.1f}%")


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    overlay_params = OverlayParams()

    print("=" * 80)
    print("  Rotation + Intraday Overlay Backtest")
    print("=" * 80)

    print("\n[Phase 1] Load Data")
    df_all = load_all_pairs()
    close = build_daily_close(df_all)
    params_list = candidate_params()
    print(f"  Daily bars: {len(close)}")

    print("\n[Phase 2] Walk-Forward Rotation Baseline")
    _, wf_weights_val, wf_curve_val, wf_metrics_val, _ = run_walkforward_reselection(
        close,
        params_list,
        VAL_START,
        VAL_END,
        reselect_days=args.wf_reselect_days,
        train_days=args.wf_train_days,
        val_days=args.wf_val_days,
    )
    _, wf_weights_test, wf_curve_test, wf_metrics_test, _ = run_walkforward_reselection(
        close,
        params_list,
        TEST_START,
        TEST_END,
        reselect_days=args.wf_reselect_days,
        train_days=args.wf_train_days,
        val_days=args.wf_val_days,
    )
    _, wf_weights_oos, wf_curve_oos, wf_metrics_oos, _ = run_walkforward_reselection(
        close,
        params_list,
        VAL_START,
        TEST_END,
        reselect_days=args.wf_reselect_days,
        train_days=args.wf_train_days,
        val_days=args.wf_val_days,
    )

    print_metrics("VAL Baseline", wf_metrics_val)
    print_metrics("TEST Baseline", wf_metrics_test)
    print_metrics("OOS Baseline", wf_metrics_oos)

    print("\n[Phase 3] Overlay Signals")
    val_active = wf_weights_val.set_index("time").sum(axis=1) > 0.0
    test_active = wf_weights_test.set_index("time").sum(axis=1) > 0.0
    oos_active = wf_weights_oos.set_index("time").sum(axis=1) > 0.0

    val_signal_map = build_overlay_signal_map(close, val_active, overlay_params, VAL_START, VAL_END)
    test_signal_map = build_overlay_signal_map(close, test_active, overlay_params, TEST_START, TEST_END)
    oos_signal_map = build_overlay_signal_map(close, oos_active, overlay_params, VAL_START, TEST_END)

    val_overlay = run_overlay_session(df_all, val_signal_map, VAL_START, VAL_END, overlay_params)
    test_overlay = run_overlay_session(df_all, test_signal_map, TEST_START, TEST_END, overlay_params)
    oos_overlay = run_overlay_session(df_all, oos_signal_map, VAL_START, TEST_END, overlay_params)

    print("\n[Phase 4] Combine")
    val_daily, val_combo = combine_rotation_overlay(wf_curve_val, wf_weights_val, val_overlay)
    test_daily, test_combo = combine_rotation_overlay(wf_curve_test, wf_weights_test, test_overlay)
    oos_daily, oos_combo = combine_rotation_overlay(wf_curve_oos, wf_weights_oos, oos_overlay)

    val_metrics = summarize_returns(val_combo)
    test_metrics = summarize_returns(test_combo)
    oos_metrics = summarize_returns(oos_combo)

    print_metrics("VAL Hybrid", val_metrics)
    print_metrics("TEST Hybrid", test_metrics)
    print_metrics("OOS Hybrid", oos_metrics)

    curve_df = pd.DataFrame(
        {
            "time": oos_combo.index,
            "equity": INITIAL_CASH * (1.0 + oos_combo).cumprod(),
            "daily_return": oos_combo.to_numpy(dtype="float64"),
        }
    )
    curve_df["peak"] = curve_df["equity"].cummax()
    curve_df["drawdown"] = curve_df["equity"] / curve_df["peak"] - 1.0

    baseline_val = {
        "total_return": wf_metrics_val["total_return"],
        "daily_win_rate": wf_metrics_val["daily_metrics"]["daily_win_rate"],
        "daily_target_hit_rate": wf_metrics_val["daily_metrics"]["daily_target_hit_rate"],
        "avg_daily_return": wf_metrics_val["daily_metrics"]["avg_daily_return"],
        "max_drawdown": wf_metrics_val["max_drawdown"],
    }
    baseline_test = {
        "total_return": wf_metrics_test["total_return"],
        "daily_win_rate": wf_metrics_test["daily_metrics"]["daily_win_rate"],
        "daily_target_hit_rate": wf_metrics_test["daily_metrics"]["daily_target_hit_rate"],
        "avg_daily_return": wf_metrics_test["daily_metrics"]["avg_daily_return"],
        "max_drawdown": wf_metrics_test["max_drawdown"],
    }
    baseline_oos = {
        "total_return": wf_metrics_oos["total_return"],
        "daily_win_rate": wf_metrics_oos["daily_metrics"]["daily_win_rate"],
        "daily_target_hit_rate": wf_metrics_oos["daily_metrics"]["daily_target_hit_rate"],
        "avg_daily_return": wf_metrics_oos["daily_metrics"]["avg_daily_return"],
        "max_drawdown": wf_metrics_oos["max_drawdown"],
    }

    summary = {
        "strategy_class": "walkforward_rotation_with_btc_intraday_overlay",
        "overlay_params": asdict(overlay_params),
        "walkforward_config": {
            "reselect_days": args.wf_reselect_days,
            "train_days": args.wf_train_days,
            "val_days": args.wf_val_days,
        },
        "baseline": {
            "validation": baseline_val,
            "test": baseline_test,
            "oos": baseline_oos,
        },
        "hybrid": {
            "validation": val_metrics,
            "test": test_metrics,
            "oos": oos_metrics,
        },
        "delta_vs_baseline": {
            "validation": {
                "return_delta": float(val_metrics["total_return"] - baseline_val["total_return"]),
                "daily_win_rate_delta": float(val_metrics["daily_metrics"]["daily_win_rate"] - baseline_val["daily_win_rate"]),
                "daily_target_hit_rate_delta": float(val_metrics["daily_metrics"]["daily_target_hit_rate"] - baseline_val["daily_target_hit_rate"]),
                "avg_daily_return_delta": float(val_metrics["daily_metrics"]["avg_daily_return"] - baseline_val["avg_daily_return"]),
                "max_drawdown_delta": float(val_metrics["max_drawdown"] - baseline_val["max_drawdown"]),
            },
            "test": {
                "return_delta": float(test_metrics["total_return"] - baseline_test["total_return"]),
                "daily_win_rate_delta": float(test_metrics["daily_metrics"]["daily_win_rate"] - baseline_test["daily_win_rate"]),
                "daily_target_hit_rate_delta": float(test_metrics["daily_metrics"]["daily_target_hit_rate"] - baseline_test["daily_target_hit_rate"]),
                "avg_daily_return_delta": float(test_metrics["daily_metrics"]["avg_daily_return"] - baseline_test["avg_daily_return"]),
                "max_drawdown_delta": float(test_metrics["max_drawdown"] - baseline_test["max_drawdown"]),
            },
            "oos": {
                "return_delta": float(oos_metrics["total_return"] - baseline_oos["total_return"]),
                "daily_win_rate_delta": float(oos_metrics["daily_metrics"]["daily_win_rate"] - baseline_oos["daily_win_rate"]),
                "daily_target_hit_rate_delta": float(oos_metrics["daily_metrics"]["daily_target_hit_rate"] - baseline_oos["daily_target_hit_rate"]),
                "avg_daily_return_delta": float(oos_metrics["daily_metrics"]["avg_daily_return"] - baseline_oos["avg_daily_return"]),
                "max_drawdown_delta": float(oos_metrics["max_drawdown"] - baseline_oos["max_drawdown"]),
            },
        },
        "artifacts": {
            "curve_path": str(Path(args.curve_out)),
            "daily_path": str(Path(args.daily_out)),
            "weights_path": str(Path(args.weights_out)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    curve_df.to_csv(Path(args.curve_out), index=False)
    oos_daily.to_csv(Path(args.daily_out), index=False)
    wf_weights_oos.to_csv(Path(args.weights_out), index=False)
    with open(args.summary_out, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    print(f"\nSummary saved: {args.summary_out}")


if __name__ == "__main__":
    main()
