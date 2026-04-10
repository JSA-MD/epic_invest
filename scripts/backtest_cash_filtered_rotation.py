#!/usr/bin/env python3
"""Backtest a cash-filtered long-only relative-strength rotation strategy."""

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

DAY_FACTOR = np.sqrt(365.25)


@dataclass
class StrategyParams:
    timeframe: str = "1d"
    lookback_fast: int = 7
    lookback_slow: int = 14
    top_n: int = 1
    vol_window: int = 5
    target_vol_ann: float = 0.6
    regime_threshold: float = 0.02
    breadth_threshold: float = 0.50
    gross_cap: float = 1.5
    fee_rate: float = COMMISSION_PCT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a cash-filtered long-only relative-strength rotation strategy.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "cash_filtered_rotation_summary.json"),
    )
    parser.add_argument(
        "--curve-out",
        default=str(MODELS_DIR / "cash_filtered_rotation_equity_curve.csv"),
    )
    parser.add_argument(
        "--weights-out",
        default=str(MODELS_DIR / "cash_filtered_rotation_weights.csv"),
    )
    parser.add_argument(
        "--wf-curve-out",
        default=str(MODELS_DIR / "cash_filtered_rotation_walkforward_curve.csv"),
    )
    parser.add_argument(
        "--wf-weights-out",
        default=str(MODELS_DIR / "cash_filtered_rotation_walkforward_weights.csv"),
    )
    parser.add_argument(
        "--wf-selection-out",
        default=str(MODELS_DIR / "cash_filtered_rotation_walkforward_selection.csv"),
    )
    parser.add_argument("--wf-reselect-days", type=int, default=14)
    parser.add_argument("--wf-train-days", type=int, default=60)
    parser.add_argument("--wf-val-days", type=int, default=30)
    return parser.parse_args()


def json_ready(obj: Any) -> Any:
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        value = obj.item()
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def build_daily_close(df_all: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for pair in PAIRS:
        col = f"{pair}_close"
        parts.append(df_all[col].resample("1D").last().rename(pair))
    return pd.concat(parts, axis=1).dropna().sort_index()


def build_target_weights(close: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    ret = close.pct_change()
    momentum = (
        0.60 * close.pct_change(params.lookback_fast)
        + 0.40 * close.pct_change(params.lookback_slow)
    )
    realized_vol = ret.rolling(params.vol_window).std() * DAY_FACTOR
    btc_regime = (
        0.50 * close["BTCUSDT"].pct_change(params.lookback_fast)
        + 0.50 * close["BTCUSDT"].pct_change(params.lookback_slow)
    )
    breadth = (momentum > 0.0).mean(axis=1)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for ts in close.index:
        if (
            not np.isfinite(btc_regime.loc[ts])
            or float(btc_regime.loc[ts]) <= params.regime_threshold
            or float(breadth.loc[ts]) < params.breadth_threshold
        ):
            continue

        ranked = momentum.loc[ts].dropna()
        ranked = ranked[ranked > 0.0].sort_values(ascending=False).head(params.top_n)
        if ranked.empty:
            continue

        vol = realized_vol.loc[ts, ranked.index].replace(0.0, np.nan).dropna()
        ranked = ranked.loc[vol.index]
        if ranked.empty:
            continue

        raw = (ranked / vol).abs()
        raw_sum = float(raw.sum())
        if raw_sum <= 0.0:
            continue

        w = raw / raw_sum
        port_vol = float(np.sqrt(np.sum(np.square(w.to_numpy() * vol.to_numpy()))))
        if np.isfinite(port_vol) and port_vol > 1e-8:
            w = w * min(params.target_vol_ann / port_vol, params.gross_cap / max(float(w.sum()), 1e-8))

        gross = float(w.sum())
        if gross > params.gross_cap and gross > 1e-8:
            w = w * (params.gross_cap / gross)

        weights.loc[ts, w.index] = w

    return weights


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
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    curve = pd.DataFrame(
        {
            "time": net_ret.index,
            "equity": equity.to_numpy(dtype="float64"),
            "net_return": net_ret.to_numpy(dtype="float64"),
            "gross_leverage": weights.sum(axis=1).to_numpy(dtype="float64"),
            "turnover": turnover.to_numpy(dtype="float64"),
        }
    )
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0

    if len(net_ret) > 1 and net_ret.std() > 1e-12:
        sharpe = float(net_ret.mean() / net_ret.std() * DAY_FACTOR)
    else:
        sharpe = 0.0

    daily_ret = net_ret.copy()
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
        "avg_gross_leverage": float(weights.sum(axis=1).mean()),
        "avg_turnover": float(turnover.mean()),
        "active_ratio": float((weights.sum(axis=1) > 0.0).mean()),
        "rebalances": int((turnover > 0.0).sum()),
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
    }


def evaluate_target_weights(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    start: str,
    end: str,
    initial_cash: float = INITIAL_CASH,
    fee_rate: float = COMMISSION_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work_close = close.loc[start:end].copy()
    work_target = target_weights.loc[start:end].copy()
    if work_close.empty:
        empty_curve, metrics = summarize_backtest(
            pd.Series(dtype="float64"),
            pd.DataFrame(),
            initial_cash,
        )
        return pd.DataFrame(), empty_curve, metrics

    daily_ret = work_close.pct_change().fillna(0.0)
    weights = work_target.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.sum(axis=1))
    net_ret = (weights * daily_ret).sum(axis=1) - turnover * fee_rate

    curve, metrics = summarize_backtest(net_ret, weights, initial_cash)
    weights_out = weights.copy()
    weights_out.insert(0, "time", weights_out.index)
    return weights_out.reset_index(drop=True), curve, metrics


def run_backtest(
    close: pd.DataFrame,
    params: StrategyParams,
    start: str,
    end: str,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target_weights = build_target_weights(close, params)
    return evaluate_target_weights(
        close,
        target_weights,
        start,
        end,
        initial_cash=initial_cash,
        fee_rate=params.fee_rate,
    )


def strategy_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["total_return"] * 100.0
        + metrics["sharpe"] * 5.0
        + metrics["monthly_metrics"]["monthly_target_hit_rate"] * 20.0
        - abs(metrics["max_drawdown"]) * 45.0
        + metrics["active_ratio"] * 4.0
    )


def walkforward_selection_score(
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
) -> float:
    return (
        val_metrics["total_return"] * 100.0
        + val_metrics["sharpe"] * 4.0
        + val_metrics["daily_metrics"]["daily_target_hit_rate"] * 12.0
        - abs(val_metrics["max_drawdown"]) * 35.0
        - val_metrics["monthly_metrics"]["monthly_shortfall_sum"] * 4.0
        + val_metrics["active_ratio"] * 2.0
        + train_metrics["total_return"] * 20.0
        - abs(train_metrics["max_drawdown"]) * 10.0
    )


def select_walkforward_params_for_day(
    close: pd.DataFrame,
    target_cache: list[tuple[StrategyParams, pd.DataFrame]],
    selection_day,
    train_days: int = 60,
    val_days: int = 30,
    initial_cash: float = INITIAL_CASH,
) -> dict[str, Any]:
    history = close.index[close.index < selection_day]
    if len(history) < train_days + val_days:
        return {
            "status": "insufficient_history",
            "selection_day": pd.Timestamp(selection_day),
        }

    train_start = history[-(train_days + val_days)]
    train_end = history[-(val_days + 1)]
    val_start = history[-val_days]
    val_end = history[-1]

    best_params = None
    best_target = None
    best_train = None
    best_val = None
    best_score = -1e18

    for params, target_weights in target_cache:
        _, _, train_metrics = evaluate_target_weights(
            close,
            target_weights,
            str(train_start.date()),
            str(train_end.date()),
            initial_cash=initial_cash,
            fee_rate=params.fee_rate,
        )
        _, _, val_metrics = evaluate_target_weights(
            close,
            target_weights,
            str(val_start.date()),
            str(val_end.date()),
            initial_cash=initial_cash,
            fee_rate=params.fee_rate,
        )
        score = walkforward_selection_score(train_metrics, val_metrics)
        if score > best_score:
            best_score = score
            best_params = params
            best_target = target_weights
            best_train = train_metrics
            best_val = val_metrics

    assert best_params is not None and best_target is not None
    return {
        "status": "ok",
        "selection_day": pd.Timestamp(selection_day),
        "best_params": best_params,
        "best_target_weights": best_target,
        "best_score": float(best_score),
        "train_metrics": best_train,
        "val_metrics": best_val,
    }


def candidate_params() -> list[StrategyParams]:
    grid = []
    for lookback_fast, lookback_slow, top_n, vol_window, target_vol_ann, regime_threshold in itertools.product(
        [3, 5, 7],
        [10, 14],
        [1, 2],
        [5, 10],
        [0.6, 0.8, 1.0],
        [0.0, 0.02, 0.05],
    ):
        if lookback_fast >= lookback_slow:
            continue
        grid.append(
            StrategyParams(
                lookback_fast=lookback_fast,
                lookback_slow=lookback_slow,
                top_n=top_n,
                vol_window=vol_window,
                target_vol_ann=target_vol_ann,
                regime_threshold=regime_threshold,
            )
        )
    return grid


def select_best_params(close: pd.DataFrame) -> tuple[StrategyParams, pd.DataFrame]:
    rows = []
    for params in candidate_params():
        _, _, train_metrics = run_backtest(close, params, TRAIN_START, TRAIN_END)
        rows.append(
            {
                **asdict(params),
                "train_total_return": train_metrics["total_return"],
                "train_sharpe": train_metrics["sharpe"],
                "train_max_drawdown": train_metrics["max_drawdown"],
                "train_active_ratio": train_metrics["active_ratio"],
                "train_monthly_hit": train_metrics["monthly_metrics"]["monthly_target_hit_rate"],
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
            lookback_slow=int(row["lookback_slow"]),
            top_n=int(row["top_n"]),
            vol_window=int(row["vol_window"]),
            target_vol_ann=float(row["target_vol_ann"]),
            regime_threshold=float(row["regime_threshold"]),
        )
        _, _, val_metrics = run_backtest(close, params, VAL_START, VAL_END)
        score = strategy_score(val_metrics)
        if val_metrics["active_ratio"] < 0.05:
            score -= 50.0
        if score > best_score:
            best_score = score
            best_params = params

    assert best_params is not None
    return best_params, grid


def run_walkforward_reselection(
    close: pd.DataFrame,
    params_list: list[StrategyParams],
    start: str,
    end: str,
    reselect_days: int = 14,
    train_days: int = 60,
    val_days: int = 30,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int]]:
    target_cache = [(params, build_target_weights(close, params)) for params in params_list]
    eval_days = close.loc[start:end].index

    scheduled_rows: list[pd.Series] = []
    selection_rows: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = {}
    i = 0

    while i < len(eval_days):
        block_days = eval_days[i:i + reselect_days]
        block_start = block_days[0]
        selection = select_walkforward_params_for_day(
            close,
            target_cache,
            block_start,
            train_days=train_days,
            val_days=val_days,
            initial_cash=initial_cash,
        )

        if selection["status"] != "ok":
            for day in block_days:
                scheduled_rows.append(pd.Series(0.0, index=close.columns, name=day))
            selection_rows.append(
                {
                    "selection_date": str(block_start.date()),
                    "apply_until": str(block_days[-1].date()),
                    "status": "insufficient_history",
                }
            )
            i += reselect_days
            continue

        best_params = selection["best_params"]
        best_target = selection["best_target_weights"]
        best_train = selection["train_metrics"]
        best_val = selection["val_metrics"]
        best_score = selection["best_score"]
        key = (
            f"f{best_params.lookback_fast}_s{best_params.lookback_slow}_"
            f"n{best_params.top_n}_v{best_params.vol_window}_"
            f"tv{best_params.target_vol_ann:.1f}_rt{best_params.regime_threshold:.2f}"
        )
        selection_counts[key] = selection_counts.get(key, 0) + 1

        for day in block_days:
            scheduled_rows.append(best_target.loc[day].rename(day))

        selection_rows.append(
            {
                "selection_date": str(block_start.date()),
                "apply_until": str(block_days[-1].date()),
                "score": float(best_score),
                "lookback_fast": best_params.lookback_fast,
                "lookback_slow": best_params.lookback_slow,
                "top_n": best_params.top_n,
                "vol_window": best_params.vol_window,
                "target_vol_ann": best_params.target_vol_ann,
                "regime_threshold": best_params.regime_threshold,
                "train_return": float(best_train["total_return"]),
                "train_max_drawdown": float(best_train["max_drawdown"]),
                "val_return": float(best_val["total_return"]),
                "val_max_drawdown": float(best_val["max_drawdown"]),
                "val_active_ratio": float(best_val["active_ratio"]),
            }
        )
        i += reselect_days

    scheduled_weights = pd.DataFrame(scheduled_rows).fillna(0.0)
    scheduled_weights.index = eval_days
    weights_out, curve, metrics = evaluate_target_weights(
        close,
        scheduled_weights,
        start,
        end,
        initial_cash=initial_cash,
        fee_rate=COMMISSION_PCT,
    )
    selection_df = pd.DataFrame(selection_rows)
    return selection_df, weights_out, curve, metrics, selection_counts


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Cash-Filtered Relative-Strength Rotation Backtest")
    print("=" * 72)

    print("\n[Phase 1] Load Data")
    df_all = load_all_pairs()
    close = build_daily_close(df_all)
    print(f"  Daily bars: {len(close)}")

    print("\n[Phase 2] Parameter Search")
    best_params, grid = select_best_params(close)
    print(f"  Best params: {asdict(best_params)}")

    print("\n[Phase 3] Backtests")
    _, curve_train, metrics_train = run_backtest(close, best_params, TRAIN_START, TRAIN_END)
    _, curve_val, metrics_val = run_backtest(close, best_params, VAL_START, VAL_END)
    _, curve_test, metrics_test = run_backtest(close, best_params, TEST_START, TEST_END)
    weights_full, curve_full, metrics_full = run_backtest(close, best_params, TRAIN_START, TEST_END)

    print("\n[Phase 4] Walk-Forward Reselection")
    selection_df, wf_weights_test, wf_curve_test, wf_metrics_test, wf_counts = run_walkforward_reselection(
        close,
        candidate_params(),
        TEST_START,
        TEST_END,
        reselect_days=args.wf_reselect_days,
        train_days=args.wf_train_days,
        val_days=args.wf_val_days,
    )
    _, _, wf_curve_oos, wf_metrics_oos, _ = run_walkforward_reselection(
        close,
        candidate_params(),
        VAL_START,
        TEST_END,
        reselect_days=args.wf_reselect_days,
        train_days=args.wf_train_days,
        val_days=args.wf_val_days,
    )

    for label, metrics in (
        ("TRAIN", metrics_train),
        ("VAL", metrics_val),
        ("TEST", metrics_test),
        ("FULL", metrics_full),
        ("WF TEST", wf_metrics_test),
        ("WF OOS", wf_metrics_oos),
    ):
        print(f"\n=== {label} ===")
        print(f"  Return:       {metrics['total_return']*100:+.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Active Ratio: {metrics['active_ratio']*100:.1f}%")
        print(f"  Avg Month:    {metrics['monthly_metrics']['avg_monthly_return']*100:+.2f}%")
        print(f"  Monthly Hit:  {metrics['monthly_metrics']['monthly_target_hit_rate']*100:.1f}%")

    grid_out = MODELS_DIR / "cash_filtered_rotation_grid.csv"
    grid.to_csv(grid_out, index=False)
    curve_full.to_csv(Path(args.curve_out), index=False)
    weights_full.to_csv(Path(args.weights_out), index=False)
    wf_curve_test.to_csv(Path(args.wf_curve_out), index=False)
    wf_weights_test.to_csv(Path(args.wf_weights_out), index=False)
    selection_df.to_csv(Path(args.wf_selection_out), index=False)

    summary = {
        "strategy_class": "cash_filtered_long_only_relative_strength_rotation",
        "timeframe": "1d",
        "pairs": PAIRS,
        "best_params": asdict(best_params),
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "test_metrics": metrics_test,
        "full_metrics": metrics_full,
        "walkforward_config": {
            "reselect_days": args.wf_reselect_days,
            "train_days": args.wf_train_days,
            "val_days": args.wf_val_days,
        },
        "walkforward_test_metrics": wf_metrics_test,
        "walkforward_oos_metrics": wf_metrics_oos,
        "walkforward_selection_counts": wf_counts,
        "grid_rows": int(len(grid)),
        "grid_path": str(grid_out),
        "curve_path": str(Path(args.curve_out)),
        "weights_path": str(Path(args.weights_out)),
        "walkforward_curve_path": str(Path(args.wf_curve_out)),
        "walkforward_weights_path": str(Path(args.wf_weights_out)),
        "walkforward_selection_path": str(Path(args.wf_selection_out)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.summary_out, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    print(f"\nSummary saved: {args.summary_out}")


if __name__ == "__main__":
    main()
