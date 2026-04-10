#!/usr/bin/env python3
"""Backtest the current live system across executive horizon windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_core_stop_loss_compare import resolve_period_windows
from backtest_rotation_intraday_overlay import run_overlay_session, summarize_returns
from backtest_rotation_target_050 import BEST_OVERLAY_PARAMS, scale_series
from core_strategy_registry import DEFAULT_CORE_CHAMPION_PATH, build_core_target_weights, load_core_artifact
from gp_crypto_evolution import INITIAL_CASH, MODELS_DIR, PAIRS, PRIMARY_PAIR, load_pair
from rotation_target_050_live import compute_overlay_signal, resolve_default_leverage
from search_core_champion import compute_portfolio_frame, summarize_portfolio_frame

EPSILON = 1e-12
DEFAULT_SLIPPAGE_PCT = 0.0002
FULL_DAY_BARS_5M = 288
DEFAULT_DATA_START = "2022-04-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the current live system across 2m/4m/6m/1y/4y windows.",
    )
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_CORE_CHAMPION_PATH),
        help="Current live core champion artifact.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "live_system_period_backtest.json"),
    )
    parser.add_argument(
        "--data-start",
        default=DEFAULT_DATA_START,
        help="Warmup start date used to load local caches.",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=resolve_default_leverage(),
        help="Live-equivalent leverage applied to both core and overlay returns.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh caches from Binance before evaluation.",
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
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out.index = out.index.tz_convert(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_daily_close_frame(start: str, refresh_cache: bool) -> pd.DataFrame:
    series: list[pd.Series] = []
    for pair in PAIRS:
        frame = load_pair(pair, interval="1d", start=start, end=None, refresh_cache=refresh_cache)
        frame = normalize_index(frame)
        series.append(frame[f"{pair}_close"].rename(pair))
    close = pd.concat(series, axis=1).dropna().sort_index()
    if close.empty:
        raise RuntimeError("No daily close history available")
    return close


def load_intraday_primary_frame(start: str, refresh_cache: bool) -> pd.DataFrame:
    frame = load_pair(PRIMARY_PAIR, interval="5m", start=start, end=None, refresh_cache=refresh_cache)
    frame = normalize_index(frame)
    if frame.empty:
        raise RuntimeError(f"No intraday history available for {PRIMARY_PAIR}")
    return frame


def infer_last_complete_day(intraday_frame: pd.DataFrame) -> pd.Timestamp:
    day_counts = intraday_frame.groupby(intraday_frame.index.normalize()).size()
    if day_counts.empty:
        raise RuntimeError("Unable to infer last complete day from intraday history")
    last_day = pd.Timestamp(day_counts.index[-1]).normalize()
    if int(day_counts.iloc[-1]) < FULL_DAY_BARS_5M:
        return last_day - pd.Timedelta(days=1)
    return last_day


def previous_market_day(index: pd.DatetimeIndex, day: pd.Timestamp) -> pd.Timestamp | None:
    loc = index.get_indexer([day])[0]
    if loc <= 0:
        return None
    return pd.Timestamp(index[loc - 1])


def build_live_overlay_signal_map(
    close: pd.DataFrame,
    actual_core_weights: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[pd.Timestamp, float]:
    eval_days = pd.DatetimeIndex(close.loc[start:end].index)
    active = actual_core_weights.abs().sum(axis=1) > EPSILON
    signal_map: dict[pd.Timestamp, float] = {}
    for trade_day in eval_days:
        trade_day = pd.Timestamp(trade_day)
        if bool(active.get(trade_day, False)):
            signal_map[trade_day] = 0.0
            continue
        market_day = previous_market_day(close.index, trade_day)
        if market_day is None:
            signal_map[trade_day] = 0.0
            continue
        overlay = compute_overlay_signal(close, market_day)
        signal_map[trade_day] = float(overlay["signal_pct"])
    return signal_map


def combine_live_system_returns(
    core_curve: pd.DataFrame,
    actual_core_weights: pd.DataFrame,
    overlay_returns: pd.Series,
    leverage: float,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    core_series = core_curve.set_index("time")["net_return"].astype("float64")
    active = actual_core_weights.abs().sum(axis=1) > EPSILON
    scaled_overlay = scale_series(
        overlay_returns.reindex(core_series.index).fillna(0.0),
        leverage,
    )
    combo = core_series.copy()
    flat_days = ~active.reindex(core_series.index).fillna(False)
    combo.loc[flat_days] = combo.loc[flat_days] + scaled_overlay.loc[flat_days]

    detail = pd.DataFrame(
        {
            "time": core_series.index,
            "core_return": core_series.to_numpy(dtype="float64"),
            "overlay_return": scaled_overlay.to_numpy(dtype="float64"),
            "combined_return": combo.to_numpy(dtype="float64"),
            "core_active": active.reindex(core_series.index).fillna(False).to_numpy(dtype=bool),
        }
    )
    return detail, scaled_overlay, combo, active.reindex(core_series.index).fillna(False)


def evaluate_core_period(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    actual_start: pd.Timestamp,
    end_day: pd.Timestamp,
    *,
    fee_rate: float,
    slippage_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    model_start = previous_market_day(close.index, actual_start) or actual_start
    portfolio_frame, actual_weights = compute_portfolio_frame(
        close,
        target_weights,
        str(model_start.date()),
        str(end_day.date()),
        fee_rate=fee_rate,
        slippage_rate=slippage_pct,
    )
    portfolio_frame = portfolio_frame.loc[actual_start:end_day].copy()
    actual_weights = actual_weights.loc[actual_start:end_day].copy()
    core_curve, core_metrics = summarize_portfolio_frame(portfolio_frame, initial_cash=INITIAL_CASH)
    weights_out = actual_weights.copy()
    weights_out.insert(0, "time", weights_out.index)
    return portfolio_frame, weights_out.reset_index(drop=True), core_curve, core_metrics


def evaluate_period(
    label: str,
    actual_start: pd.Timestamp,
    end_day: pd.Timestamp,
    close: pd.DataFrame,
    intraday_primary: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    fee_rate: float,
    slippage_pct: float,
    leverage: float,
) -> dict[str, Any]:
    _, core_weights_out, core_curve, core_metrics = evaluate_core_period(
        close,
        target_weights * float(leverage),
        actual_start,
        end_day,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
    )
    actual_core_weights = core_weights_out.set_index("time")[PAIRS].fillna(0.0).astype("float64")

    signal_map = build_live_overlay_signal_map(close, actual_core_weights, actual_start, end_day)
    overlay_returns = run_overlay_session(
        intraday_primary,
        signal_map,
        str(actual_start.date()),
        str(end_day.date()),
        BEST_OVERLAY_PARAMS,
    )
    combined_daily, scaled_overlay, combined_series, core_active = combine_live_system_returns(
        core_curve,
        actual_core_weights,
        overlay_returns,
        leverage,
    )
    combined_metrics = summarize_returns(combined_series)
    overlay_metrics = summarize_returns(scaled_overlay)
    signal_series = pd.Series(signal_map, dtype="float64").reindex(combined_series.index).fillna(0.0)

    return {
        "label": label,
        "start": str(actual_start.date()),
        "end": str(end_day.date()),
        "days": int(len(combined_series)),
        "core_only": core_metrics,
        "overlay_only": overlay_metrics,
        "combined": combined_metrics,
        "activity": {
            "core_active_days": int(core_active.sum()),
            "core_flat_days": int((~core_active).sum()),
            "overlay_trade_days": int((scaled_overlay != 0.0).sum()),
            "overlay_long_days": int((signal_series > 0.0).sum()),
            "overlay_short_days": int((signal_series < 0.0).sum()),
        },
        "artifacts": {
            "core_curve_rows": int(len(core_curve)),
            "combined_daily_rows": int(len(combined_daily)),
        },
    }


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    strategy = load_core_artifact(args.artifact)
    fee_rate = float(strategy.params.fee_rate)
    slippage_pct = float(strategy.metadata.get("search_config", {}).get("base_slippage", DEFAULT_SLIPPAGE_PCT))

    close = load_daily_close_frame(args.data_start, args.refresh_cache)
    intraday_primary = load_intraday_primary_frame(args.data_start, args.refresh_cache)

    end_day = min(infer_last_complete_day(intraday_primary), pd.Timestamp(close.index.max()).normalize())
    close = close.loc[:str(end_day.date())].copy()
    intraday_primary = intraday_primary.loc[:str(end_day.date())].copy()
    if close.empty or intraday_primary.empty:
        raise RuntimeError("Filtered backtest dataset is empty")

    target_weights = build_core_target_weights(close, strategy)

    results: list[dict[str, Any]] = []
    for label, start_ts in resolve_period_windows(end_day):
        actual_start = max(pd.Timestamp(start_ts).normalize(), pd.Timestamp(close.index[0]).normalize())
        results.append(
            evaluate_period(
                label,
                actual_start,
                end_day,
                close,
                intraday_primary,
                target_weights,
                fee_rate=fee_rate,
                slippage_pct=slippage_pct,
                leverage=float(args.leverage),
            )
        )

    summary = {
        "strategy_class": "live_system_period_backtest",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "artifact": str(Path(args.artifact).resolve()),
        "selected_strategy": {
            "family": strategy.family,
            "key": strategy.key,
            "source": strategy.source,
        },
        "backtest_config": {
            "leverage": float(args.leverage),
            "fee_rate": fee_rate,
            "slippage_pct": slippage_pct,
            "overlay_params": {
                "momentum_lookback": int(BEST_OVERLAY_PARAMS.momentum_lookback),
                "breadth_lookback": int(BEST_OVERLAY_PARAMS.breadth_lookback),
                "momentum_threshold": float(BEST_OVERLAY_PARAMS.momentum_threshold),
                "breadth_threshold": float(BEST_OVERLAY_PARAMS.breadth_threshold),
                "signal_mode": str(BEST_OVERLAY_PARAMS.signal_mode),
                "reward_multiple": float(BEST_OVERLAY_PARAMS.reward_multiple),
                "trail_activation_pct": float(BEST_OVERLAY_PARAMS.trail_activation_pct),
                "trail_distance_pct": float(BEST_OVERLAY_PARAMS.trail_distance_pct),
                "trail_floor_pct": float(BEST_OVERLAY_PARAMS.trail_floor_pct),
                "entry_threshold": float(BEST_OVERLAY_PARAMS.entry_threshold),
            },
            "overlay_signal_timing": "market_day_t decides effective_day_t_plus_1",
            "refresh_cache": bool(args.refresh_cache),
            "initial_cash": float(INITIAL_CASH),
        },
        "dataset": {
            "daily_start": str(close.index[0].date()),
            "daily_end": str(close.index[-1].date()),
            "intraday_start": str(intraday_primary.index[0]),
            "intraday_end": str(intraday_primary.index[-1]),
            "last_complete_day": str(end_day.date()),
            "pairs": list(PAIRS),
            "intraday_pair": PRIMARY_PAIR,
            "daily_paths": {pair: str(Path(f"data/binance_futures/{pair}_1d.csv").resolve()) for pair in PAIRS},
            "intraday_path": str(Path(f"data/binance_futures/{PRIMARY_PAIR}_5m.csv").resolve()),
        },
        "periods": results,
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(json_ready(summary), f, ensure_ascii=False, indent=2)

    print("=" * 92)
    print("Current Live System Period Backtest")
    print("=" * 92)
    print(f"Strategy : {strategy.key}")
    print(f"Data End : {end_day.date()} | leverage={float(args.leverage):.2f}x | fee={fee_rate:.4%} | slippage={slippage_pct:.4%}")
    print("-" * 92)
    for row in results:
        combined = row["combined"]
        print(
            f"{row['label']:>3} | "
            f"ret={combined['total_return']*100:+8.2f}% "
            f"avg={combined['daily_metrics']['avg_daily_return']*100:+6.3f}% "
            f"mdd={combined['max_drawdown']*100:+7.2f}% "
            f"sharpe={combined['sharpe']:+6.2f} "
            f"overlay_days={row['activity']['overlay_trade_days']:>3d}"
        )
    print("-" * 92)
    print(f"Summary saved: {out_path}")


if __name__ == "__main__":
    main()
