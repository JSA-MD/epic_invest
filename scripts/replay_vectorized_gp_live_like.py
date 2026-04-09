#!/usr/bin/env python3
"""Replay a vectorized GP strategy bar-by-bar using live-like semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd

import gp_crypto_evolution as gp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a vectorized GP strategy one bar at a time.",
    )
    parser.add_argument("--model", required=True, help="Path to a GP dill payload or raw tree")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--pair", default=gp.PRIMARY_PAIR)
    parser.add_argument(
        "--report-out",
        default=None,
        help="Optional JSON report path",
    )
    parser.add_argument(
        "--bars-out",
        default=None,
        help="Optional CSV path for per-bar replay logs",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__module__") and str(value.__module__).startswith("deap."):
        return str(value)
    if hasattr(value, "__class__") and value.__class__.__name__ in {"PrimitiveTree", "Primitive", "Terminal"}:
        return str(value)
    return value


def load_model(path: Path):
    with open(path, "rb") as f:
        payload = dill.load(f)
    if isinstance(payload, dict) and "tree" in payload:
        return payload["tree"], payload
    return payload, None


def replay_bar_by_bar(
    df: pd.DataFrame,
    model,
    pair: str = gp.PRIMARY_PAIR,
    initial_cash: float = gp.INITIAL_CASH,
    commission: float = gp.COMMISSION_PCT,
    dead_band: float = gp.NO_TRADE_BAND,
) -> tuple[dict[str, Any], pd.DataFrame]:
    compiled = gp.toolbox.compile(expr=model)
    close = df[f"{pair}_close"].to_numpy(dtype="float64")

    equity = float(initial_cash)
    equity_curve = [float(initial_cash)]
    current_weight = 0.0
    net_ret = []
    logs = []
    n_trades = 0

    for i in range(len(df) - 1):
        ts = pd.Timestamp(df.index[i])
        row = df.iloc[i]
        inputs = gp.get_feature_values(row, pair)
        signal_raw = float(compiled(*inputs))
        signal_pct = float(np.clip(np.where(np.isfinite(signal_raw), signal_raw, 0.0), -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        target_weight = requested_weight
        if abs(target_weight - current_weight) < dead_band / 100.0:
            target_weight = current_weight

        turnover = abs(target_weight - current_weight)
        if turnover > 0.001:
            n_trades += 1

        next_ret = float(close[i + 1] / close[i] - 1.0)
        bar_net = target_weight * next_ret - turnover * commission * 2
        equity *= (1.0 + bar_net)
        net_ret.append(bar_net)
        equity_curve.append(float(equity))

        logs.append(
            {
                "timestamp": ts,
                "close": float(close[i]),
                "next_close": float(close[i + 1]),
                "signal_pct": signal_pct,
                "requested_weight": float(requested_weight),
                "target_weight": float(target_weight),
                "prev_weight": float(current_weight),
                "turnover": float(turnover),
                "next_bar_return": float(next_ret),
                "net_return": float(bar_net),
                "equity": float(equity),
            }
        )
        current_weight = target_weight

    net_ret_arr = np.asarray(net_ret, dtype="float64")
    equity_curve_arr = np.asarray(equity_curve, dtype="float64")
    final_equity = float(equity_curve_arr[-1]) if len(equity_curve_arr) else float(initial_cash)
    total_return = final_equity / initial_cash - 1.0

    if len(net_ret_arr) > 1 and np.std(net_ret_arr) > 1e-12:
        sharpe = float(np.mean(net_ret_arr) / np.std(net_ret_arr) * np.sqrt(365.25 * 24))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve_arr)
    max_drawdown = float(np.min(equity_curve_arr / peak - 1.0))
    result = {
        "total_return": total_return,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_equity": final_equity,
        "equity_curve": equity_curve_arr,
        "net_ret": net_ret_arr,
        "daily_metrics": gp.compute_daily_metrics(net_ret_arr),
    }
    return result, pd.DataFrame(logs)


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    daily = result["daily_metrics"]
    return {
        "avg_daily_return": float(daily["avg_daily_return"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]),
        "daily_target_hit_rate": float(daily["daily_target_hit_rate"]),
        "daily_win_rate": float(daily["daily_win_rate"]),
        "worst_day": float(daily["worst_day"]),
        "best_day": float(daily["best_day"]),
        "n_trades": int(result["n_trades"]),
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    model, payload = load_model(model_path)

    df = gp.load_all_pairs(start=args.start, end=args.end, refresh_cache=False)
    if df.empty:
        raise RuntimeError("no data available for replay window")

    compiled = gp.toolbox.compile(expr=model)
    desired = compiled(*gp.get_feature_arrays(df, args.pair))
    vector_result = gp.vectorized_backtest(
        df[f"{args.pair}_close"].to_numpy(dtype="float64"),
        desired,
    )
    sequential_result, bar_logs = replay_bar_by_bar(df, model, pair=args.pair)

    vector_summary = summarize(vector_result)
    sequential_summary = summarize(sequential_result)
    diff = {
        key: sequential_summary[key] - vector_summary[key]
        for key in ("avg_daily_return", "total_return", "max_drawdown", "sharpe")
    }
    diff["n_trades"] = sequential_summary["n_trades"] - vector_summary["n_trades"]

    payload_meta = None
    if isinstance(payload, dict):
        payload_meta = {
            "algorithm": payload.get("algorithm"),
            "window_start": payload.get("window_start"),
            "window_end": payload.get("window_end"),
            "target_avg_daily_return": payload.get("target_avg_daily_return"),
            "created_at": payload.get("created_at"),
        }

    report = {
        "model_path": model_path,
        "payload_meta": payload_meta,
        "pair": args.pair,
        "start": args.start,
        "end": args.end,
        "bars": len(df),
        "vectorized": vector_summary,
        "sequential_replay": sequential_summary,
        "diff": diff,
    }

    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))

    if args.report_out:
        with open(args.report_out, "w") as f:
            json.dump(json_safe(report), f, ensure_ascii=False, indent=2)
    if args.bars_out:
        out = Path(args.bars_out)
        bar_logs.to_csv(out, index=False)


if __name__ == "__main__":
    main()
