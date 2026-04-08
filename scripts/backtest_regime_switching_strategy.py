#!/usr/bin/env python3
"""Backtest a regime-switching breakout / mean-reversion strategy on local Binance futures data."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
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
    PRIMARY_PAIR,
    TRAIN_END,
    TRAIN_START,
    TEST_END,
    TEST_START,
    VAL_END,
    VAL_START,
    load_all_pairs,
    summarize_monthly_returns,
    summarize_period_returns,
)


@dataclass
class StrategyParams:
    timeframe: str = "1h"
    ema_fast: int = 24
    ema_slow: int = 72
    donchian: int = 20
    bb_window: int = 20
    bb_mult: float = 1.75
    rsi_window: int = 14
    atr_window: int = 14
    trend_gap: float = 0.008
    flat_gap: float = 0.003
    vol_expand: float = 1.10
    vol_contract: float = 0.92
    stop_atr: float = 1.2
    trend_target_r: float = 2.5
    trend_trail_activate_r: float = 1.0
    trend_trail_atr: float = 1.0
    revert_target_r: float = 1.2
    max_hold_bars_trend: int = 48
    max_hold_bars_revert: int = 18
    breadth_long: float = 0.60
    breadth_short: float = 0.40
    fee_rate: float = COMMISSION_PCT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest regime-switching breakout / mean-reversion strategy.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "regime_switching_strategy_summary.json"),
    )
    parser.add_argument(
        "--curve-out",
        default=str(MODELS_DIR / "regime_switching_equity_curve.csv"),
    )
    parser.add_argument(
        "--trades-out",
        default=str(MODELS_DIR / "regime_switching_trades.csv"),
    )
    return parser.parse_args()


def compute_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def extract_hourly_pair(df_all: pd.DataFrame, pair: str) -> pd.DataFrame:
    cols = [f"{pair}_open", f"{pair}_high", f"{pair}_low", f"{pair}_close"]
    rename = {
        f"{pair}_open": "open",
        f"{pair}_high": "high",
        f"{pair}_low": "low",
        f"{pair}_close": "close",
    }
    raw = df_all[cols].rename(columns=rename).copy()
    hourly = raw.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    return hourly


def build_hourly_dataset(df_all: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    btc = extract_hourly_pair(df_all, PRIMARY_PAIR)
    btc["ema_fast"] = btc["close"].ewm(span=params.ema_fast, adjust=False).mean()
    btc["ema_slow"] = btc["close"].ewm(span=params.ema_slow, adjust=False).mean()
    btc["ema_gap"] = btc["ema_fast"] / btc["ema_slow"] - 1.0
    btc["atr"] = compute_atr(btc, params.atr_window)
    btc["rsi"] = compute_rsi(btc["close"], params.rsi_window)
    btc["bb_mid"] = btc["close"].rolling(params.bb_window).mean()
    btc["bb_std"] = btc["close"].rolling(params.bb_window).std()
    btc["bb_upper"] = btc["bb_mid"] + params.bb_mult * btc["bb_std"]
    btc["bb_lower"] = btc["bb_mid"] - params.bb_mult * btc["bb_std"]
    btc["donchian_high"] = btc["high"].rolling(params.donchian).max().shift(1)
    btc["donchian_low"] = btc["low"].rolling(params.donchian).min().shift(1)

    ret_1h = btc["close"].pct_change()
    btc["vol_short"] = ret_1h.rolling(24).std()
    btc["vol_long"] = ret_1h.rolling(120).std()
    btc["vol_ratio"] = btc["vol_short"] / btc["vol_long"].replace(0.0, np.nan)

    breadth_parts = []
    for pair in [p for p in PAIRS if p != PRIMARY_PAIR]:
        hourly = extract_hourly_pair(df_all, pair)
        breadth_parts.append((hourly["close"].pct_change(12) > 0.0).astype(float).rename(pair))
    breadth = pd.concat(breadth_parts, axis=1).reindex(btc.index).ffill().fillna(0.0)
    btc["breadth_up"] = breadth.mean(axis=1)

    btc["trend_regime"] = (
        (btc["vol_ratio"] >= params.vol_expand)
        & (btc["ema_gap"].abs() >= params.trend_gap)
    )
    btc["revert_regime"] = (
        (btc["vol_ratio"] <= params.vol_contract)
        & (btc["ema_gap"].abs() <= params.flat_gap)
    )

    return btc.dropna().copy()


def net_trade_return(raw_return: float, fee_rate: float) -> float:
    return (1.0 + raw_return) * ((1.0 - fee_rate) ** 2) - 1.0


def trade_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["total_return"] * 100.0
        + metrics["profit_factor"] * 8.0
        - abs(metrics["max_drawdown"]) * 60.0
        - metrics["monthly_metrics"]["monthly_shortfall_sum"] * 8.0
    )


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


def summarize_curve(
    curve_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_cash: float,
) -> dict[str, Any]:
    if curve_df.empty:
        daily_metrics = summarize_period_returns(np.asarray([], dtype="float64"))
        monthly_metrics = summarize_monthly_returns(np.asarray([], dtype="float64"), pd.DatetimeIndex([]))
        return {
            "total_return": 0.0,
            "final_equity": initial_cash,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "daily_metrics": daily_metrics,
            "monthly_metrics": monthly_metrics,
        }

    curve = curve_df.copy()
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    max_dd = float(curve["drawdown"].min())
    total_return = float(curve["equity"].iloc[-1] / initial_cash - 1.0)

    hourly_ret = curve["equity"].pct_change().dropna()
    if len(hourly_ret) > 1 and hourly_ret.std() > 1e-12:
        sharpe = float(hourly_ret.mean() / hourly_ret.std() * np.sqrt(365.25 * 24))
    else:
        sharpe = 0.0

    eq = curve.set_index("time")["equity"].resample("1D").last().dropna()
    prev = eq.shift(1)
    daily_ret = eq / prev - 1.0
    if len(daily_ret):
        daily_ret.iloc[0] = eq.iloc[0] / initial_cash - 1.0
    daily_ret = daily_ret.dropna()
    daily_metrics = summarize_period_returns(daily_ret.to_numpy(dtype="float64"))
    monthly_metrics = summarize_monthly_returns(
        daily_ret.to_numpy(dtype="float64"),
        pd.DatetimeIndex(daily_ret.index),
    )

    if trades_df.empty:
        win_rate = 0.0
        profit_factor = 0.0
        expectancy = 0.0
    else:
        wins = trades_df[trades_df["net_pnl"] > 0.0]
        losses = trades_df[trades_df["net_pnl"] <= 0.0]
        gross_profit = float(wins["net_pnl"].sum())
        gross_loss = float(abs(losses["net_pnl"].sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        win_rate = float((trades_df["net_pnl"] > 0.0).mean())
        expectancy = float(trades_df["net_pnl"].mean())

    return {
        "total_return": total_return,
        "final_equity": float(curve["equity"].iloc[-1]),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": int(len(trades_df)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
    }


def run_backtest(
    hourly: pd.DataFrame,
    params: StrategyParams,
    start: str,
    end: str,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = hourly.loc[start:end].copy()
    if work.empty:
        metrics = summarize_curve(pd.DataFrame(), pd.DataFrame(), initial_cash)
        return pd.DataFrame(), pd.DataFrame(), metrics

    capital = float(initial_cash)
    pending: dict[str, Any] | None = None
    position: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []

    idx = work.index
    open_p = work["open"].to_numpy(dtype="float64")
    high_p = work["high"].to_numpy(dtype="float64")
    low_p = work["low"].to_numpy(dtype="float64")
    close_p = work["close"].to_numpy(dtype="float64")

    for i in range(1, len(work)):
        row = work.iloc[i]
        now = idx[i]

        if pending is not None and position is None:
            entry_price = open_p[i]
            atr = float(pending["atr"])
            direction = int(pending["direction"])
            kind = str(pending["kind"])
            risk = atr * params.stop_atr
            if kind == "trend":
                target = entry_price + direction * risk * params.trend_target_r
            else:
                target = entry_price + direction * risk * params.revert_target_r
            stop = entry_price - direction * risk
            position = {
                "entry_time": now,
                "entry_idx": i,
                "entry_price": entry_price,
                "direction": direction,
                "kind": kind,
                "atr": atr,
                "capital_start": capital,
                "stop_price": stop,
                "target_price": target,
                "trail_active": False,
                "best_price": entry_price,
            }
            pending = None

        if position is not None:
            direction = int(position["direction"])
            kind = str(position["kind"])
            risk = position["atr"] * params.stop_atr
            stop_price = float(position["stop_price"])
            target_price = float(position["target_price"])
            best_price = float(position["best_price"])
            trail_active = bool(position["trail_active"])

            if direction > 0:
                if high_p[i] > best_price:
                    best_price = high_p[i]
                if (not trail_active) and high_p[i] >= position["entry_price"] + risk * params.trend_trail_activate_r:
                    trail_active = True
                    best_price = max(best_price, high_p[i])
                dynamic_stop = stop_price
                if kind == "trend" and trail_active:
                    dynamic_stop = max(stop_price, best_price - position["atr"] * params.trend_trail_atr)
                exit_reason = None
                exit_price = None
                if low_p[i] <= dynamic_stop:
                    exit_reason = "stop"
                    exit_price = dynamic_stop
                elif high_p[i] >= target_price:
                    exit_reason = "target"
                    exit_price = target_price
            else:
                if low_p[i] < best_price:
                    best_price = low_p[i]
                if (not trail_active) and low_p[i] <= position["entry_price"] - risk * params.trend_trail_activate_r:
                    trail_active = True
                    best_price = min(best_price, low_p[i])
                dynamic_stop = stop_price
                if kind == "trend" and trail_active:
                    dynamic_stop = min(stop_price, best_price + position["atr"] * params.trend_trail_atr)
                exit_reason = None
                exit_price = None
                if high_p[i] >= dynamic_stop:
                    exit_reason = "stop"
                    exit_price = dynamic_stop
                elif low_p[i] <= target_price:
                    exit_reason = "target"
                    exit_price = target_price

            if exit_reason is None:
                bars_held = i - int(position["entry_idx"])
                if kind == "trend" and bars_held >= params.max_hold_bars_trend:
                    exit_reason = "time"
                    exit_price = close_p[i]
                elif kind == "revert" and bars_held >= params.max_hold_bars_revert:
                    exit_reason = "time"
                    exit_price = close_p[i]
                elif kind == "trend" and direction > 0 and row["ema_fast"] < row["ema_slow"]:
                    exit_reason = "trend_flip"
                    exit_price = close_p[i]
                elif kind == "trend" and direction < 0 and row["ema_fast"] > row["ema_slow"]:
                    exit_reason = "trend_flip"
                    exit_price = close_p[i]
                elif kind == "revert" and direction > 0 and close_p[i] >= row["bb_mid"]:
                    exit_reason = "mean_revert_exit"
                    exit_price = close_p[i]
                elif kind == "revert" and direction < 0 and close_p[i] <= row["bb_mid"]:
                    exit_reason = "mean_revert_exit"
                    exit_price = close_p[i]

            position["best_price"] = best_price
            position["trail_active"] = trail_active

            unrealized = direction * (close_p[i] / position["entry_price"] - 1.0)
            mark_equity = position["capital_start"] * (1.0 - params.fee_rate) * (1.0 + unrealized)

            if exit_reason is not None and exit_price is not None:
                raw_return = direction * (exit_price / position["entry_price"] - 1.0)
                net_return = net_trade_return(raw_return, params.fee_rate)
                capital = position["capital_start"] * (1.0 + net_return)
                net_pnl = capital - position["capital_start"]
                trades.append(
                    {
                        "entry_time": position["entry_time"].isoformat(),
                        "exit_time": now.isoformat(),
                        "kind": kind,
                        "direction": "LONG" if direction > 0 else "SHORT",
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "raw_return": raw_return,
                        "net_return": net_return,
                        "net_pnl": net_pnl,
                        "bars_held": i - int(position["entry_idx"]),
                        "reason": exit_reason,
                    }
                )
                position = None
                curve.append({"time": now, "equity": capital})
            else:
                curve.append({"time": now, "equity": mark_equity})
        else:
            curve.append({"time": now, "equity": capital})

        if position is None and pending is None and i < len(work) - 1:
            prev = work.iloc[i]
            trend_long = (
                bool(prev["trend_regime"])
                and prev["close"] > prev["donchian_high"]
                and prev["ema_fast"] > prev["ema_slow"]
                and prev["breadth_up"] >= params.breadth_long
            )
            trend_short = (
                bool(prev["trend_regime"])
                and prev["close"] < prev["donchian_low"]
                and prev["ema_fast"] < prev["ema_slow"]
                and prev["breadth_up"] <= params.breadth_short
            )
            revert_long = (
                bool(prev["revert_regime"])
                and prev["close"] < prev["bb_lower"]
                and prev["rsi"] <= 35.0
            )
            revert_short = (
                bool(prev["revert_regime"])
                and prev["close"] > prev["bb_upper"]
                and prev["rsi"] >= 65.0
            )

            if trend_long:
                pending = {"direction": 1, "kind": "trend", "atr": prev["atr"]}
            elif trend_short:
                pending = {"direction": -1, "kind": "trend", "atr": prev["atr"]}
            elif revert_long:
                pending = {"direction": 1, "kind": "revert", "atr": prev["atr"]}
            elif revert_short:
                pending = {"direction": -1, "kind": "revert", "atr": prev["atr"]}

    curve_df = pd.DataFrame(curve)
    trades_df = pd.DataFrame(trades)
    metrics = summarize_curve(curve_df, trades_df, initial_cash)
    return trades_df, curve_df, metrics


def candidate_params() -> list[StrategyParams]:
    grid = []
    for donchian, bb_mult, stop_atr, trend_target_r, vol_expand, vol_contract in itertools.product(
        [18, 24],
        [1.5, 1.75, 2.0],
        [1.0, 1.2, 1.5],
        [2.0, 2.5, 3.0],
        [1.05, 1.10, 1.20],
        [0.85, 0.92],
    ):
        grid.append(
            StrategyParams(
                donchian=donchian,
                bb_mult=bb_mult,
                stop_atr=stop_atr,
                trend_target_r=trend_target_r,
                vol_expand=vol_expand,
                vol_contract=vol_contract,
            )
        )
    return grid


def select_best_params(hourly: pd.DataFrame) -> tuple[StrategyParams, pd.DataFrame]:
    rows = []
    for params in candidate_params():
        _, _, train_metrics = run_backtest(hourly, params, TRAIN_START, TRAIN_END)
        rows.append(
            {
                **asdict(params),
                "train_total_return": train_metrics["total_return"],
                "train_max_drawdown": train_metrics["max_drawdown"],
                "train_profit_factor": train_metrics["profit_factor"],
                "train_monthly_shortfall": train_metrics["monthly_metrics"]["monthly_shortfall_sum"],
                "train_score": trade_score(train_metrics),
            }
        )
    grid = pd.DataFrame(rows).sort_values("train_score", ascending=False).reset_index(drop=True)
    shortlist = grid.head(12).copy()

    best_params = None
    best_score = -1e18
    for _, row in shortlist.iterrows():
        params = StrategyParams(
            donchian=int(row["donchian"]),
            bb_mult=float(row["bb_mult"]),
            stop_atr=float(row["stop_atr"]),
            trend_target_r=float(row["trend_target_r"]),
            vol_expand=float(row["vol_expand"]),
            vol_contract=float(row["vol_contract"]),
        )
        _, _, val_metrics = run_backtest(hourly, params, VAL_START, VAL_END)
        score = trade_score(val_metrics)
        if val_metrics["trades"] < 5:
            score -= 1000.0
        if score > best_score:
            best_score = score
            best_params = params

    assert best_params is not None
    return best_params, grid


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  Regime-Switching Strategy Backtest")
    print("=" * 72)

    print("\n[Phase 1] Load Data")
    df_all = load_all_pairs()
    base_params = StrategyParams()
    hourly = build_hourly_dataset(df_all, base_params)
    print(f"  Hourly bars: {len(hourly)}")

    print("\n[Phase 2] Parameter Search")
    best_params, grid = select_best_params(hourly)
    print(f"  Best params: {asdict(best_params)}")

    print("\n[Phase 3] Backtests")
    trades_train, curve_train, metrics_train = run_backtest(hourly, best_params, TRAIN_START, TRAIN_END)
    trades_val, curve_val, metrics_val = run_backtest(hourly, best_params, VAL_START, VAL_END)
    trades_test, curve_test, metrics_test = run_backtest(hourly, best_params, TEST_START, TEST_END)
    trades_full, curve_full, metrics_full = run_backtest(hourly, best_params, TRAIN_START, TEST_END)

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
        print(f"  Trades:       {metrics['trades']}")
        print(f"  Win Rate:     {metrics['win_rate']*100:.1f}%")
        print(f"  ProfitFactor: {metrics['profit_factor']:.2f}")
        print(f"  Avg Month:    {metrics['monthly_metrics']['avg_monthly_return']*100:+.2f}%")
        print(f"  Monthly Hit:  {metrics['monthly_metrics']['monthly_target_hit_rate']*100:.1f}%")

    grid_out = MODELS_DIR / "regime_switching_grid.csv"
    grid.to_csv(grid_out, index=False)

    curve_full.to_csv(Path(args.curve_out), index=False)
    trades_full.to_csv(Path(args.trades_out), index=False)

    summary = {
        "strategy_class": "regime_switching_breakout_mean_reversion",
        "selected_pair": PRIMARY_PAIR,
        "timeframe": "1h",
        "best_params": asdict(best_params),
        "train_metrics": metrics_train,
        "validation_metrics": metrics_val,
        "test_metrics": metrics_test,
        "full_metrics": metrics_full,
        "grid_rows": int(len(grid)),
        "grid_path": str(grid_out),
        "curve_path": str(Path(args.curve_out)),
        "trades_path": str(Path(args.trades_out)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.summary_out, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    print(f"\nSummary saved: {args.summary_out}")


if __name__ == "__main__":
    main()
