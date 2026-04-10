#!/usr/bin/env python3
"""Compare current core strategy baseline vs entry-anchored 2% stop-loss across multiple horizons."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core_strategy_registry import (
    DEFAULT_CORE_CHAMPION_PATH,
    LONG_ONLY_FAMILY,
    build_core_target_weights,
    load_core_artifact,
)
from gp_crypto_evolution import (
    INITIAL_CASH,
    MODELS_DIR,
    PAIRS,
    load_pair,
    summarize_monthly_returns,
    summarize_period_returns,
)

DAY_FACTOR = np.sqrt(365.25)
DEFAULT_STOP_LOSS_PCT = 0.02
DEFAULT_SLIPPAGE_PCT = 0.0002
UTC = "UTC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare current core strategy baseline vs stop-loss across rolling horizons.",
    )
    parser.add_argument(
        "--artifact",
        default=str(DEFAULT_CORE_CHAMPION_PATH),
        help="Current core champion artifact.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "core_stop_loss_2pct_compare.json"),
    )
    parser.add_argument(
        "--stop-loss-pct",
        type=float,
        default=DEFAULT_STOP_LOSS_PCT,
    )
    parser.add_argument(
        "--slippage-pct",
        type=float,
        default=None,
        help="Override slippage. Defaults to champion search config base_slippage or 0.0002.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh Binance daily caches.",
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


def resolve_period_windows(end_day: pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
    return [
        ("2m", (end_day - pd.DateOffset(months=2) + pd.Timedelta(days=1)).normalize()),
        ("4m", (end_day - pd.DateOffset(months=4) + pd.Timedelta(days=1)).normalize()),
        ("6m", (end_day - pd.DateOffset(months=6) + pd.Timedelta(days=1)).normalize()),
        ("1y", (end_day - pd.DateOffset(years=1) + pd.Timedelta(days=1)).normalize()),
        ("4y", (end_day - pd.DateOffset(years=4) + pd.Timedelta(days=1)).normalize()),
    ]


def load_daily_ohlc(
    pairs: list[str],
    *,
    start: str,
    end: str,
    refresh_cache: bool,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    frames: dict[str, pd.DataFrame] = {}
    common_index: pd.DatetimeIndex | None = None
    for pair in pairs:
        df = load_pair(
            pair,
            interval="1d",
            start=start,
            end=end,
            refresh_cache=refresh_cache,
        )
        cols = {
            "open": f"{pair}_open",
            "high": f"{pair}_high",
            "low": f"{pair}_low",
            "close": f"{pair}_close",
        }
        frame = df[[cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
        frame.columns = ["open", "high", "low", "close"]
        frame.index = pd.DatetimeIndex(frame.index).tz_convert(UTC).normalize()
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frames[pair] = frame
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)

    if common_index is None:
        return frames, pd.DatetimeIndex([])

    for pair in pairs:
        frames[pair] = frames[pair].loc[common_index].copy()
    return frames, common_index


def build_close_frame(ohlc_by_pair: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.concat([ohlc_by_pair[pair].loc[index, "close"].rename(pair) for pair in PAIRS], axis=1).dropna()


def summarize_returns(
    net_ret: pd.Series,
    gross_leverage: pd.Series,
    turnover: pd.Series,
    stop_events: pd.Series,
    *,
    initial_cash: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if net_ret.empty:
        curve = pd.DataFrame(
            columns=["time", "equity", "net_return", "gross_leverage", "turnover", "stop_events"]
        )
        daily_metrics = summarize_period_returns(np.asarray([], dtype="float64"))
        monthly_metrics = summarize_monthly_returns(np.asarray([], dtype="float64"), pd.DatetimeIndex([]))
        return curve, {
            "total_return": 0.0,
            "final_equity": float(initial_cash),
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_daily_return": 0.0,
            "daily_win_rate": 0.0,
            "cvar": 0.0,
            "avg_gross_leverage": 0.0,
            "avg_turnover": 0.0,
            "active_ratio": 0.0,
            "stop_days": 0,
            "stop_event_count": 0,
            "daily_metrics": daily_metrics,
            "monthly_metrics": monthly_metrics,
        }

    equity = float(initial_cash) * (1.0 + net_ret).cumprod()
    curve = pd.DataFrame(
        {
            "time": net_ret.index,
            "equity": equity.to_numpy(dtype="float64"),
            "net_return": net_ret.to_numpy(dtype="float64"),
            "gross_leverage": gross_leverage.to_numpy(dtype="float64"),
            "turnover": turnover.to_numpy(dtype="float64"),
            "stop_events": stop_events.to_numpy(dtype="int64"),
        }
    )
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0

    if len(net_ret) > 1 and net_ret.std() > 1e-12:
        sharpe = float(net_ret.mean() / net_ret.std() * DAY_FACTOR)
    else:
        sharpe = 0.0

    daily_metrics = summarize_period_returns(net_ret.to_numpy(dtype="float64"))
    monthly_metrics = summarize_monthly_returns(net_ret.to_numpy(dtype="float64"), pd.DatetimeIndex(net_ret.index))
    return curve, {
        "total_return": float(equity.iloc[-1] / initial_cash - 1.0),
        "final_equity": float(equity.iloc[-1]),
        "max_drawdown": float(curve["drawdown"].min()),
        "sharpe": sharpe,
        "avg_daily_return": float(daily_metrics["avg_daily_return"]),
        "daily_win_rate": float(daily_metrics["daily_win_rate"]),
        "cvar": float(daily_metrics["cvar"]),
        "avg_gross_leverage": float(gross_leverage.mean()),
        "avg_turnover": float(turnover.mean()),
        "active_ratio": float((gross_leverage > 1e-12).mean()),
        "stop_days": int((stop_events > 0).sum()),
        "stop_event_count": int(stop_events.sum()),
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
    }


def simulate_core_strategy(
    close: pd.DataFrame,
    ohlc_by_pair: dict[str, pd.DataFrame],
    target_weights: pd.DataFrame,
    *,
    start: str,
    end: str,
    fee_rate: float,
    slippage_pct: float,
    stop_loss_pct: float | None,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    idx = pd.DatetimeIndex(close.loc[start:end].index)
    if idx.empty:
        return summarize_returns(
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="float64"),
            pd.Series(dtype="int64"),
            initial_cash=initial_cash,
        )

    desired_weights = target_weights.shift(1).reindex(idx).fillna(0.0)
    prev_close = close.shift(1).reindex(idx)
    open_df = pd.concat([ohlc_by_pair[pair].loc[idx, "open"].rename(pair) for pair in close.columns], axis=1)
    low_df = pd.concat([ohlc_by_pair[pair].loc[idx, "low"].rename(pair) for pair in close.columns], axis=1)
    close_df = close.reindex(idx)

    net_ret_rows: list[float] = []
    gross_rows: list[float] = []
    turnover_rows: list[float] = []
    stop_rows: list[int] = []

    actual_end_weights = pd.Series(0.0, index=close.columns, dtype="float64")
    entry_basis = pd.Series(np.nan, index=close.columns, dtype="float64")
    per_trade_cost = float(fee_rate) + float(slippage_pct)
    eps = 1e-12

    for day in idx:
        desired = desired_weights.loc[day].astype("float64").fillna(0.0)
        prev_close_day = prev_close.loc[day].astype("float64")
        open_day = open_df.loc[day].astype("float64")
        low_day = low_df.loc[day].astype("float64")
        close_day = close_df.loc[day].astype("float64")

        if prev_close_day.isna().all():
            net_ret_rows.append(0.0)
            gross_rows.append(0.0)
            turnover_rows.append(0.0)
            stop_rows.append(0)
            actual_end_weights = pd.Series(0.0, index=close.columns, dtype="float64")
            entry_basis = pd.Series(np.nan, index=close.columns, dtype="float64")
            continue

        open_turnover = float((desired - actual_end_weights).abs().sum())
        open_cost = open_turnover * per_trade_cost

        active_entry_basis = pd.Series(np.nan, index=close.columns, dtype="float64")
        for pair in close.columns:
            desired_weight = float(desired[pair])
            if abs(desired_weight) <= eps:
                continue
            rebalance_price = float(prev_close_day[pair]) if np.isfinite(prev_close_day[pair]) else np.nan
            if not np.isfinite(rebalance_price) or rebalance_price <= 0.0:
                continue
            prior_weight = float(actual_end_weights[pair])
            prior_entry = float(entry_basis[pair]) if np.isfinite(entry_basis[pair]) else np.nan
            if abs(prior_weight) <= eps or not np.isfinite(prior_entry) or prior_entry <= 0.0:
                active_entry_basis[pair] = rebalance_price
                continue
            if desired_weight > prior_weight + eps:
                add_weight = desired_weight - prior_weight
                active_entry_basis[pair] = (
                    (prior_weight * prior_entry) + (add_weight * rebalance_price)
                ) / desired_weight
            else:
                active_entry_basis[pair] = prior_entry

        asset_returns = pd.Series(0.0, index=close.columns, dtype="float64")
        stop_mask = pd.Series(False, index=close.columns)
        for pair in close.columns:
            weight = float(desired[pair])
            rebalance_price = float(prev_close_day[pair]) if np.isfinite(prev_close_day[pair]) else np.nan
            stop_anchor_price = float(active_entry_basis[pair]) if np.isfinite(active_entry_basis[pair]) else np.nan
            if not np.isfinite(rebalance_price) or rebalance_price <= 0.0 or abs(weight) <= eps:
                asset_returns[pair] = 0.0
                continue

            close_return = float(close_day[pair] / rebalance_price - 1.0)
            if (
                stop_loss_pct is None
                or stop_loss_pct <= 0.0
                or not np.isfinite(stop_anchor_price)
                or stop_anchor_price <= 0.0
            ):
                asset_returns[pair] = close_return
                continue

            stop_price = stop_anchor_price * (1.0 - float(stop_loss_pct))
            if np.isfinite(open_day[pair]) and float(open_day[pair]) <= stop_price:
                asset_returns[pair] = float(open_day[pair] / rebalance_price - 1.0)
                stop_mask[pair] = True
            elif np.isfinite(low_day[pair]) and float(low_day[pair]) <= stop_price:
                asset_returns[pair] = float(stop_price / rebalance_price - 1.0)
                stop_mask[pair] = True
            else:
                asset_returns[pair] = close_return

        gross_port_ret = float((desired * asset_returns).sum())
        stop_exit_turnover = float(desired.where(stop_mask, 0.0).abs().sum())
        stop_exit_cost = stop_exit_turnover * per_trade_cost
        net_port_ret = gross_port_ret - open_cost - stop_exit_cost

        if 1.0 + net_port_ret <= 1e-12:
            end_weights = pd.Series(0.0, index=close.columns, dtype="float64")
        else:
            end_values = desired.where(~stop_mask, 0.0) * (1.0 + asset_returns.where(~stop_mask, 0.0))
            end_weights = end_values / (1.0 + net_port_ret)
            end_weights = end_weights.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        actual_end_weights = end_weights.astype("float64")
        entry_basis = active_entry_basis.where((~stop_mask) & (actual_end_weights.abs() > eps), np.nan)
        net_ret_rows.append(net_port_ret)
        gross_rows.append(float(desired.abs().sum()))
        turnover_rows.append(open_turnover + stop_exit_turnover)
        stop_rows.append(int(stop_mask.sum()))

    net_ret = pd.Series(net_ret_rows, index=idx, dtype="float64")
    gross = pd.Series(gross_rows, index=idx, dtype="float64")
    turnover = pd.Series(turnover_rows, index=idx, dtype="float64")
    stop_events = pd.Series(stop_rows, index=idx, dtype="int64")
    return summarize_returns(
        net_ret,
        gross,
        turnover,
        stop_events,
        initial_cash=initial_cash,
    )


def compare_metrics(baseline: dict[str, Any], stop2: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return_delta": float(stop2["total_return"] - baseline["total_return"]),
        "sharpe_delta": float(stop2["sharpe"] - baseline["sharpe"]),
        "max_drawdown_delta": float(stop2["max_drawdown"] - baseline["max_drawdown"]),
        "avg_daily_return_delta": float(stop2["avg_daily_return"] - baseline["avg_daily_return"]),
        "daily_win_rate_delta": float(stop2["daily_win_rate"] - baseline["daily_win_rate"]),
        "cvar_delta": float(stop2["cvar"] - baseline["cvar"]),
        "avg_turnover_delta": float(stop2["avg_turnover"] - baseline["avg_turnover"]),
    }


def main() -> None:
    args = parse_args()
    strategy = load_core_artifact(args.artifact)
    if strategy.family != LONG_ONLY_FAMILY:
        raise ValueError(f"Current comparison script supports long_only core only, got {strategy.family!r}")

    end_day = (pd.Timestamp.now(tz=UTC).normalize() - pd.Timedelta(days=1)).normalize()
    start_day = (end_day - pd.DateOffset(years=4) + pd.Timedelta(days=1)).normalize()
    ohlc_by_pair, common_index = load_daily_ohlc(
        PAIRS,
        start=str(start_day.date()),
        end=str(end_day.date()),
        refresh_cache=args.refresh_cache,
    )
    close = build_close_frame(ohlc_by_pair, common_index)
    if close.empty:
        raise RuntimeError("No daily OHLC data loaded for comparison")

    target_weights = build_core_target_weights(close, strategy)
    fee_rate = float(strategy.params.fee_rate)
    if args.slippage_pct is not None:
        slippage_pct = float(args.slippage_pct)
    else:
        slippage_pct = float(strategy.metadata.get("search_config", {}).get("base_slippage", DEFAULT_SLIPPAGE_PCT))

    windows = resolve_period_windows(close.index[-1])
    results: list[dict[str, Any]] = []
    for label, start_ts in windows:
        actual_start = max(start_ts, close.index[0])
        baseline_curve, baseline_metrics = simulate_core_strategy(
            close,
            ohlc_by_pair,
            target_weights,
            start=str(actual_start.date()),
            end=str(close.index[-1].date()),
            fee_rate=fee_rate,
            slippage_pct=slippage_pct,
            stop_loss_pct=None,
        )
        stop_curve, stop_metrics = simulate_core_strategy(
            close,
            ohlc_by_pair,
            target_weights,
            start=str(actual_start.date()),
            end=str(close.index[-1].date()),
            fee_rate=fee_rate,
            slippage_pct=slippage_pct,
            stop_loss_pct=float(args.stop_loss_pct),
        )
        results.append(
            {
                "label": label,
                "start": str(actual_start.date()),
                "end": str(close.index[-1].date()),
                "days": int(len(close.loc[str(actual_start.date()): str(close.index[-1].date())])),
                "baseline": baseline_metrics,
                "stop_loss_2pct": stop_metrics,
                "delta": compare_metrics(baseline_metrics, stop_metrics),
                "artifacts": {
                    "baseline_curve_rows": int(len(baseline_curve)),
                    "stop_curve_rows": int(len(stop_curve)),
                },
            }
        )

    summary = {
        "strategy_class": "core_stop_loss_compare",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "artifact": str(Path(args.artifact).resolve()),
        "selected_strategy": {
            "family": strategy.family,
            "key": strategy.key,
            "source": strategy.source,
            "params": asdict(strategy.params),
        },
        "backtest_config": {
            "stop_loss_pct": float(args.stop_loss_pct),
            "stop_anchor_mode": "position_entry",
            "fee_rate": fee_rate,
            "slippage_pct": slippage_pct,
            "refresh_cache": bool(args.refresh_cache),
            "initial_cash": float(INITIAL_CASH),
        },
        "dataset": {
            "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()),
            "n_days": int(len(close)),
            "pairs": list(close.columns),
            "paths": {pair: str(Path(f"data/binance_futures/{pair}_1d.csv").resolve()) for pair in close.columns},
        },
        "periods": results,
    }

    output_path = Path(args.summary_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    print("=" * 88)
    print("Core Strategy Baseline vs Entry-Anchored 2% Stop-Loss")
    print("=" * 88)
    print(f"Strategy: {strategy.key}")
    print(f"Data    : {close.index[0].date()} -> {close.index[-1].date()} | days={len(close)}")
    print(f"Costs   : fee={fee_rate:.4%} slippage={slippage_pct:.4%} stop={float(args.stop_loss_pct):.2%}")
    print("-" * 88)
    for row in results:
        base = row["baseline"]
        stop2 = row["stop_loss_2pct"]
        print(
            f"{row['label']:>3} | "
            f"baseline ret={base['total_return']*100:+7.2f}% mdd={base['max_drawdown']*100:+6.2f}% sharpe={base['sharpe']:+5.2f} | "
            f"stop2 ret={stop2['total_return']*100:+7.2f}% mdd={stop2['max_drawdown']*100:+6.2f}% sharpe={stop2['sharpe']:+5.2f} "
            f"stops={stop2['stop_event_count']}"
        )
    print("-" * 88)
    print(f"Summary saved: {output_path}")


if __name__ == "__main__":
    main()
