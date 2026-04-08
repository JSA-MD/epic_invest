#!/usr/bin/env python3
"""Backtest a volatility-scaled multi-asset momentum rotation strategy."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gp_crypto_evolution import (
    COMMISSION_PCT,
    INITIAL_CASH,
    MODELS_DIR,
    PAIRS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    load_all_pairs,
    summarize_monthly_returns,
    summarize_period_returns,
)

HOUR_FACTOR = np.sqrt(365.25 * 24.0)


@dataclass
class StrategyParams:
    timeframe: str = "1h"
    lookback_fast: int = 24
    lookback_mid: int = 72
    lookback_slow: int = 168
    vol_window: int = 72
    volume_window: int = 24
    score_threshold: float = 0.35
    regime_threshold: float = 0.10
    top_n: int = 2
    target_vol_ann: float = 1.20
    gross_cap: float = 2.00
    max_asset_weight: float = 1.00
    volume_power: float = 0.50
    fee_rate: float = COMMISSION_PCT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a volatility-scaled multi-asset momentum rotation strategy.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "vol_target_momentum_strategy_summary.json"),
    )
    parser.add_argument(
        "--curve-out",
        default=str(MODELS_DIR / "vol_target_momentum_equity_curve.csv"),
    )
    parser.add_argument(
        "--weights-out",
        default=str(MODELS_DIR / "vol_target_momentum_weights.csv"),
    )
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


def extract_hourly_feature(df_all: pd.DataFrame, pair: str, feature: str) -> pd.Series:
    col = f"{pair}_{feature}"
    if col not in df_all.columns:
        raise KeyError(f"Missing column: {col}")
    series = df_all[col].copy()
    if feature == "close":
        return series.resample("1h").last().rename(pair)
    if feature == "open":
        return series.resample("1h").first().rename(pair)
    if feature == "high":
        return series.resample("1h").max().rename(pair)
    if feature == "low":
        return series.resample("1h").min().rename(pair)
    return series.resample("1h").mean().rename(pair)


def build_hourly_panel(df_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    close_parts = []
    activity_parts = []
    for pair in PAIRS:
        close = extract_hourly_feature(df_all, pair, "close")
        high = extract_hourly_feature(df_all, pair, "high")
        low = extract_hourly_feature(df_all, pair, "low")
        close_parts.append(close)

        vol_proxy = extract_hourly_feature(df_all, pair, "vol_sma")
        if float(vol_proxy.abs().sum()) <= 1e-12:
            vol_proxy = ((high - low) / close.replace(0.0, np.nan)).rename(pair)
        activity_parts.append(vol_proxy)

    close_df = pd.concat(close_parts, axis=1).dropna().sort_index()
    activity_df = pd.concat(activity_parts, axis=1).reindex(close_df.index).ffill().bfill()
    activity_df = activity_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return close_df, activity_df


def compute_signal_panel(
    close: pd.DataFrame,
    activity_proxy: pd.DataFrame,
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret_1h = close.pct_change()
    realized_vol = ret_1h.rolling(params.vol_window).std() * HOUR_FACTOR

    def standardized_momentum(lookback: int) -> pd.DataFrame:
        lb_ret = close / close.shift(lookback) - 1.0
        lb_scale = ret_1h.rolling(lookback).std() * np.sqrt(float(lookback))
        return lb_ret / lb_scale.replace(0.0, np.nan)

    score = (
        0.50 * standardized_momentum(params.lookback_fast)
        + 0.30 * standardized_momentum(params.lookback_mid)
        + 0.20 * standardized_momentum(params.lookback_slow)
    )
    rel_activity = activity_proxy / activity_proxy.rolling(params.volume_window).mean().replace(0.0, np.nan)
    rel_activity = rel_activity.clip(lower=0.5, upper=1.5).fillna(1.0)
    score = score * np.power(rel_activity, params.volume_power)
    score = score.replace([np.inf, -np.inf], np.nan)
    realized_vol = realized_vol.replace([np.inf, -np.inf], np.nan)
    return score, realized_vol


def build_target_weights(
    score: pd.DataFrame,
    realized_vol: pd.DataFrame,
    params: StrategyParams,
) -> pd.DataFrame:
    score_values = score.to_numpy(dtype="float64")
    vol_values = realized_vol.to_numpy(dtype="float64")
    out = np.zeros_like(score_values)

    for i in range(len(score.index)):
        row_score = score_values[i]
        row_vol = vol_values[i]
        valid = np.isfinite(row_score) & np.isfinite(row_vol) & (row_vol > 1e-8)
        if not np.any(valid):
            continue

        valid_idx = np.where(valid)[0]
        s = row_score[valid]
        v = row_vol[valid]
        regime_score = float(np.mean(s))
        if regime_score >= params.regime_threshold:
            select_mask = s >= params.score_threshold
            order = np.argsort(-s[select_mask])
        elif regime_score <= -params.regime_threshold:
            select_mask = s <= -params.score_threshold
            order = np.argsort(s[select_mask])
        else:
            continue

        selected_idx = valid_idx[select_mask]
        if len(selected_idx) == 0:
            continue

        selected_idx = selected_idx[order][: params.top_n]
        sel_score = row_score[selected_idx]
        sel_vol = row_vol[selected_idx]
        raw = np.abs(sel_score) / sel_vol
        raw_sum = float(raw.sum())
        if raw_sum <= 0.0:
            continue

        w = np.sign(sel_score) * raw / raw_sum
        diag_port_vol = float(np.sqrt(np.sum(np.square(w * sel_vol))))
        if np.isfinite(diag_port_vol) and diag_port_vol > 1e-8:
            gross_pre = max(float(np.abs(w).sum()), 1e-8)
            w = w * min(params.target_vol_ann / diag_port_vol, params.gross_cap / gross_pre)

        w = np.clip(w, -params.max_asset_weight, params.max_asset_weight)
        gross = float(np.abs(w).sum())
        if gross > params.gross_cap and gross > 1e-8:
            w = w * (params.gross_cap / gross)

        out[i, selected_idx] = w

    return pd.DataFrame(out, index=score.index, columns=score.columns)


def summarize_backtest(
    net_ret: pd.Series,
    weights: pd.DataFrame,
    initial_cash: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if net_ret.empty:
        curve = pd.DataFrame(columns=["time", "equity", "net_return", "gross_leverage", "turnover"])
        daily_metrics = summarize_period_returns(np.asarray([], dtype="float64"))
        monthly_metrics = summarize_monthly_returns(np.asarray([], dtype="float64"), pd.DatetimeIndex([]))
        return curve, {
            "total_return": 0.0,
            "final_equity": initial_cash,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_gross_leverage": 0.0,
            "avg_turnover": 0.0,
            "active_ratio": 0.0,
            "rebalances": 0,
            "daily_metrics": daily_metrics,
            "monthly_metrics": monthly_metrics,
        }

    equity = float(initial_cash) * (1.0 + net_ret).cumprod()
    curve = pd.DataFrame(
        {
            "time": net_ret.index,
            "equity": equity.to_numpy(dtype="float64"),
            "net_return": net_ret.to_numpy(dtype="float64"),
            "gross_leverage": weights.abs().sum(axis=1).to_numpy(dtype="float64"),
            "turnover": weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1)).to_numpy(dtype="float64"),
        }
    )
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0

    if len(net_ret) > 1 and net_ret.std() > 1e-12:
        sharpe = float(net_ret.mean() / net_ret.std() * HOUR_FACTOR)
    else:
        sharpe = 0.0

    daily_eq = curve.set_index("time")["equity"].resample("1D").last().dropna()
    prev = daily_eq.shift(1)
    daily_ret = daily_eq / prev - 1.0
    if len(daily_ret):
        daily_ret.iloc[0] = daily_eq.iloc[0] / initial_cash - 1.0
    daily_ret = daily_ret.dropna()
    daily_metrics = summarize_period_returns(daily_ret.to_numpy(dtype="float64"))
    monthly_metrics = summarize_monthly_returns(
        daily_ret.to_numpy(dtype="float64"),
        pd.DatetimeIndex(daily_ret.index),
    )

    return curve, {
        "total_return": float(equity.iloc[-1] / initial_cash - 1.0),
        "final_equity": float(equity.iloc[-1]),
        "max_drawdown": float(curve["drawdown"].min()),
        "sharpe": sharpe,
        "avg_gross_leverage": float(weights.abs().sum(axis=1).mean()),
        "avg_turnover": float(curve["turnover"].mean()),
        "active_ratio": float((weights.abs().sum(axis=1) > 0.0).mean()),
        "rebalances": int((curve["turnover"] > 0.0).sum()),
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
    }


def run_backtest(
    close: pd.DataFrame,
    volume_proxy: pd.DataFrame,
    params: StrategyParams,
    start: str,
    end: str,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score, realized_vol = compute_signal_panel(close, volume_proxy, params)
    target_weights = build_target_weights(score, realized_vol, params)

    work_close = close.loc[start:end].copy()
    work_target = target_weights.loc[start:end].copy()
    if work_close.empty:
        empty_curve, metrics = summarize_backtest(pd.Series(dtype="float64"), pd.DataFrame(), initial_cash)
        return pd.DataFrame(), empty_curve, metrics

    asset_ret = work_close.pct_change().fillna(0.0)
    weights = work_target.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    net_ret = (weights * asset_ret).sum(axis=1) - turnover * params.fee_rate

    curve, metrics = summarize_backtest(net_ret, weights, initial_cash)
    weights_out = weights.copy()
    weights_out.insert(0, "time", weights_out.index)
    return weights_out.reset_index(drop=True), curve, metrics


def strategy_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["total_return"] * 100.0
        + metrics["sharpe"] * 6.0
        + metrics["daily_metrics"]["daily_target_hit_rate"] * 16.0
        - abs(metrics["max_drawdown"]) * 70.0
        - metrics["monthly_metrics"]["monthly_shortfall_sum"] * 8.0
    )


def candidate_params() -> list[StrategyParams]:
    grid = []
    horizon_sets = [
        (12, 48, 120),
        (24, 72, 168),
    ]
    for (fast, mid, slow), top_n, score_threshold, regime_threshold, target_vol_ann, volume_power in itertools.product(
        horizon_sets,
        [1, 2],
        [0.35, 0.60],
        [0.10, 0.25],
        [0.80, 1.20, 1.60],
        [0.0, 0.5],
    ):
        grid.append(
            StrategyParams(
                lookback_fast=fast,
                lookback_mid=mid,
                lookback_slow=slow,
                top_n=top_n,
                score_threshold=score_threshold,
                regime_threshold=regime_threshold,
                target_vol_ann=target_vol_ann,
                volume_power=volume_power,
            )
        )
    return grid


def select_best_params(
    close: pd.DataFrame,
    volume_proxy: pd.DataFrame,
) -> tuple[StrategyParams, pd.DataFrame]:
    rows = []
    for params in candidate_params():
        _, _, train_metrics = run_backtest(close, volume_proxy, params, TRAIN_START, TRAIN_END)
        rows.append(
            {
                **asdict(params),
                "train_total_return": train_metrics["total_return"],
                "train_sharpe": train_metrics["sharpe"],
                "train_max_drawdown": train_metrics["max_drawdown"],
                "train_daily_target_hit_rate": train_metrics["daily_metrics"]["daily_target_hit_rate"],
                "train_monthly_shortfall": train_metrics["monthly_metrics"]["monthly_shortfall_sum"],
                "train_score": strategy_score(train_metrics),
            }
        )
    grid = pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True)
    shortlist = grid.head(16).copy()

    best_params = None
    best_score = -1e18
    for _, row in shortlist.iterrows():
        params = StrategyParams(
            lookback_fast=int(row["lookback_fast"]),
            lookback_mid=int(row["lookback_mid"]),
            lookback_slow=int(row["lookback_slow"]),
            vol_window=int(row["vol_window"]),
            top_n=int(row["top_n"]),
            score_threshold=float(row["score_threshold"]),
            regime_threshold=float(row["regime_threshold"]),
            target_vol_ann=float(row["target_vol_ann"]),
            volume_power=float(row["volume_power"]),
        )
        _, _, val_metrics = run_backtest(close, volume_proxy, params, VAL_START, VAL_END)
        score = strategy_score(val_metrics)
        if val_metrics["active_ratio"] < 0.20:
            score -= 100.0
        if score > best_score:
            best_score = score
            best_params = params

    assert best_params is not None
    return best_params, grid


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Volatility-Scaled Momentum Rotation Backtest")
    print("=" * 72)

    print("\n[Phase 1] Load Data")
    df_all = load_all_pairs()
    close, volume_proxy = build_hourly_panel(df_all)
    print(f"  Hourly bars: {len(close)}")

    print("\n[Phase 2] Parameter Search")
    best_params, grid = select_best_params(close, volume_proxy)
    print(f"  Best params: {asdict(best_params)}")

    print("\n[Phase 3] Backtests")
    _, curve_train, metrics_train = run_backtest(close, volume_proxy, best_params, TRAIN_START, TRAIN_END)
    _, curve_val, metrics_val = run_backtest(close, volume_proxy, best_params, VAL_START, VAL_END)
    _, curve_test, metrics_test = run_backtest(close, volume_proxy, best_params, TEST_START, TEST_END)
    weights_full, curve_full, metrics_full = run_backtest(close, volume_proxy, best_params, TRAIN_START, TEST_END)

    for label, metrics in (
        ("TRAIN", metrics_train),
        ("VAL", metrics_val),
        ("TEST", metrics_test),
        ("FULL", metrics_full),
    ):
        print(f"\n=== {label} ===")
        print(f"  Return:       {metrics['total_return']*100:+.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Active Ratio: {metrics['active_ratio']*100:.1f}%")
        print(f"  Gross Lev:    {metrics['avg_gross_leverage']:.2f}")
        print(f"  Avg Month:    {metrics['monthly_metrics']['avg_monthly_return']*100:+.2f}%")
        print(f"  Monthly Hit:  {metrics['monthly_metrics']['monthly_target_hit_rate']*100:.1f}%")

    grid_out = MODELS_DIR / "vol_target_momentum_grid.csv"
    grid.to_csv(grid_out, index=False)
    curve_full.to_csv(Path(args.curve_out), index=False)
    weights_full.to_csv(Path(args.weights_out), index=False)

    summary = {
        "strategy_class": "volatility_scaled_multi_asset_momentum_rotation",
        "timeframe": "1h",
        "pairs": PAIRS,
        "best_params": asdict(best_params),
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "test_metrics": metrics_test,
        "full_metrics": metrics_full,
        "grid_rows": int(len(grid)),
        "grid_path": str(grid_out),
        "curve_path": str(Path(args.curve_out)),
        "weights_path": str(Path(args.weights_out)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.summary_out, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    print(f"\nSummary saved: {args.summary_out}")


if __name__ == "__main__":
    main()
