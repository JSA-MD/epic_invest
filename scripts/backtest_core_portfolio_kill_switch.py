#!/usr/bin/env python3
"""Evaluate a portfolio-level core emergency kill switch for the live strategy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_cash_filtered_rotation import build_daily_close, run_walkforward_reselection
from backtest_rotation_intraday_overlay import (
    build_overlay_signal_map,
    combine_rotation_overlay,
    run_overlay_session,
    summarize_returns,
)
from backtest_rotation_target_050 import BEST_CORE_PARAMS, BEST_OVERLAY_PARAMS, MIN_TARGET_LEVERAGE
from gp_crypto_evolution import COMMISSION_PCT, MODELS_DIR, PAIRS, TEST_END, TEST_START, VAL_END, VAL_START, load_all_pairs

DAY_FACTOR = np.sqrt(365.25)
EPSILON = 1e-12
CORE_KILL_SWITCH_SIGMA_MULTIPLE = 1.5
CORE_KILL_SWITCH_MIN_PCT = 0.06
CORE_KILL_SWITCH_MAX_PCT = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the portfolio-level emergency kill switch.")
    parser.add_argument("--leverage", type=float, default=MIN_TARGET_LEVERAGE)
    parser.add_argument("--kill-switch-pct", type=float, default=None)
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "core_portfolio_kill_switch_summary.json"),
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


def compute_default_kill_switch_pct() -> float:
    target_daily_vol = BEST_CORE_PARAMS.target_vol_ann / DAY_FACTOR
    return float(np.clip(
        CORE_KILL_SWITCH_SIGMA_MULTIPLE * target_daily_vol,
        CORE_KILL_SWITCH_MIN_PCT,
        CORE_KILL_SWITCH_MAX_PCT,
    ))


def build_intraday_close(df_all: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [df_all[f"{pair}_close"].rename(pair) for pair in PAIRS],
        axis=1,
    ).sort_index()


def simulate_core_series(
    intraday_close: pd.DataFrame,
    weights_df: pd.DataFrame,
    start: str,
    end: str,
    leverage: float,
    kill_switch_pct: float | None,
) -> tuple[pd.Series, pd.DataFrame]:
    weights = weights_df.set_index("time")[PAIRS].fillna(0.0).astype("float64") * float(leverage)
    eval_days = pd.DatetimeIndex(weights.loc[start:end].index)
    actual_weights = pd.Series(0.0, index=PAIRS, dtype="float64")
    daily_returns: list[tuple[pd.Timestamp, float]] = []
    trigger_rows: list[dict[str, Any]] = []

    for day in eval_days:
        target_weights = weights.loc[day].reindex(PAIRS).fillna(0.0).astype("float64")
        open_turnover = float((target_weights - actual_weights).abs().sum())
        open_fee = open_turnover * COMMISSION_PCT
        day_start = pd.Timestamp(day)
        prev_prices = intraday_close.loc[intraday_close.index < day_start]
        day_prices = intraday_close.loc[intraday_close.index.normalize() == day_start.normalize(), PAIRS]

        if prev_prices.empty or day_prices.empty:
            daily_returns.append((day_start, -open_fee))
            actual_weights = target_weights.copy()
            continue

        prev_close = prev_prices.iloc[-1].reindex(PAIRS).astype("float64")
        gross = float(target_weights.abs().sum())
        if gross <= EPSILON:
            daily_returns.append((day_start, -open_fee))
            actual_weights = target_weights.copy()
            continue

        rel_path = day_prices.divide(prev_close) - 1.0
        port_path = rel_path.mul(target_weights, axis=1).sum(axis=1)

        if kill_switch_pct is not None:
            breached = port_path[port_path <= -float(kill_switch_pct)]
        else:
            breached = pd.Series(dtype="float64")

        if not breached.empty:
            hit_time = pd.Timestamp(breached.index[0])
            hit_return = float(breached.iloc[0])
            close_fee = gross * COMMISSION_PCT
            day_return = hit_return - open_fee - close_fee
            actual_weights = pd.Series(0.0, index=PAIRS, dtype="float64")
            trigger_rows.append(
                {
                    "day": str(day_start.date()),
                    "hit_time": hit_time.isoformat(),
                    "gross_return_at_hit": hit_return,
                    "open_fee": open_fee,
                    "close_fee": close_fee,
                    "net_day_return": day_return,
                    "gross_leverage": gross,
                }
            )
        else:
            close_rel = day_prices.iloc[-1].divide(prev_close) - 1.0
            day_return = float((close_rel * target_weights).sum()) - open_fee
            actual_weights = target_weights.copy()

        daily_returns.append((day_start, day_return))

    series = pd.Series(
        [ret for _, ret in daily_returns],
        index=pd.DatetimeIndex([day for day, _ in daily_returns]),
        name="core_return",
    )
    trigger_df = pd.DataFrame(trigger_rows)
    return series, trigger_df


def build_live_equivalent_hybrid_series(
    core_series: pd.Series,
    core_weights: pd.DataFrame,
    overlay_series: pd.Series,
    leverage: float,
) -> pd.Series:
    active = core_weights.set_index("time")[PAIRS].sum(axis=1) > 0.0
    aligned_index = core_series.index
    combo = core_series.copy()
    flat_days = ~active.reindex(aligned_index).fillna(False)
    combo.loc[flat_days] = overlay_series.reindex(aligned_index).fillna(0.0).loc[flat_days] * float(leverage)
    return combo


def compare_metrics(after: dict[str, Any], before: dict[str, Any]) -> dict[str, float]:
    return {
        "return_delta": float(after["total_return"] - before["total_return"]),
        "max_drawdown_delta": float(after["max_drawdown"] - before["max_drawdown"]),
        "daily_win_rate_delta": float(after["daily_metrics"]["daily_win_rate"] - before["daily_metrics"]["daily_win_rate"]),
        "avg_daily_return_delta": float(after["daily_metrics"]["avg_daily_return"] - before["daily_metrics"]["avg_daily_return"]),
    }


def evaluate_period(
    df_all: pd.DataFrame,
    close: pd.DataFrame,
    weights_df: pd.DataFrame,
    start: str,
    end: str,
    leverage: float,
    kill_switch_pct: float,
) -> dict[str, Any]:
    intraday_close = build_intraday_close(df_all)
    core_baseline, _ = simulate_core_series(intraday_close, weights_df, start, end, leverage, kill_switch_pct=None)
    core_kill, trigger_df = simulate_core_series(intraday_close, weights_df, start, end, leverage, kill_switch_pct=kill_switch_pct)

    active = weights_df.set_index("time")[PAIRS].sum(axis=1) > 0.0
    overlay_signal_map = build_overlay_signal_map(close, active, BEST_OVERLAY_PARAMS, start, end)
    overlay_series = run_overlay_session(df_all, overlay_signal_map, start, end, BEST_OVERLAY_PARAMS)

    hybrid_baseline = build_live_equivalent_hybrid_series(core_baseline, weights_df, overlay_series, leverage)
    hybrid_kill = build_live_equivalent_hybrid_series(core_kill, weights_df, overlay_series, leverage)

    baseline_metrics = summarize_returns(hybrid_baseline)
    kill_metrics = summarize_returns(hybrid_kill)
    return {
        "baseline": baseline_metrics,
        "kill_switch": kill_metrics,
        "delta": compare_metrics(kill_metrics, baseline_metrics),
        "trigger_count": int(len(trigger_df)),
        "trigger_days": trigger_df.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    kill_switch_pct = compute_default_kill_switch_pct() if args.kill_switch_pct is None else float(args.kill_switch_pct)

    df_all = load_all_pairs()
    close = build_daily_close(df_all)

    _, core_weights_val, _, _, _ = run_walkforward_reselection(
        close,
        [BEST_CORE_PARAMS],
        VAL_START,
        VAL_END,
    )
    _, core_weights_test, _, _, _ = run_walkforward_reselection(
        close,
        [BEST_CORE_PARAMS],
        TEST_START,
        TEST_END,
    )
    _, core_weights_oos, _, _, _ = run_walkforward_reselection(
        close,
        [BEST_CORE_PARAMS],
        VAL_START,
        TEST_END,
    )

    summary = {
        "strategy_class": "core_portfolio_kill_switch",
        "leverage": float(args.leverage),
        "kill_switch_pct": float(kill_switch_pct),
        "default_formula_pct": compute_default_kill_switch_pct(),
        "periods": {
            "validation": evaluate_period(df_all, close, core_weights_val, VAL_START, VAL_END, args.leverage, kill_switch_pct),
            "test": evaluate_period(df_all, close, core_weights_test, TEST_START, TEST_END, args.leverage, kill_switch_pct),
            "oos": evaluate_period(df_all, close, core_weights_oos, VAL_START, TEST_END, args.leverage, kill_switch_pct),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(args.summary_out)
    out_path.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(f"\nSaved summary: {out_path}")


if __name__ == "__main__":
    main()
