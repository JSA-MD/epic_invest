#!/usr/bin/env python3
"""Inference CLI for the walk-forward cash-filtered rotation strategy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_cash_filtered_rotation import (
    candidate_params,
    build_daily_close,
    build_target_weights,
    select_walkforward_params_for_day,
)
from gp_crypto_evolution import INITIAL_CASH, MODELS_DIR, TEST_START, load_all_pairs


def normalize_day(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference tools for the walk-forward cash-filtered rotation strategy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    signal = subparsers.add_parser(
        "signal",
        help="Generate the next-session target weights from the walk-forward rotation strategy.",
    )
    signal.add_argument(
        "--summary",
        default=str(MODELS_DIR / "cash_filtered_rotation_summary.json"),
        help="Path to the strategy summary JSON.",
    )
    signal.add_argument(
        "--effective-date",
        default=None,
        help="Trading day to generate weights for (YYYY-MM-DD). Defaults to next day after the latest daily close.",
    )
    signal.add_argument(
        "--equity",
        type=float,
        default=INITIAL_CASH,
        help="Account equity for notional sizing output.",
    )
    signal.add_argument(
        "--json",
        action="store_true",
        help="Print the full payload as JSON.",
    )

    return parser.parse_args()


def load_summary(summary_path: str | Path) -> dict[str, Any]:
    path = Path(summary_path)
    with open(path, "r") as f:
        summary = json.load(f)
    config = summary.get("walkforward_config")
    if not isinstance(config, dict):
        raise ValueError(f"walkforward_config missing in {path}")
    for key in ("reselect_days", "train_days", "val_days"):
        if key not in config:
            raise ValueError(f"walkforward_config.{key} missing in {path}")
    summary["_path"] = str(path)
    return summary


def load_selection_anchor(summary: dict[str, Any]) -> pd.Timestamp:
    selection_path = summary.get("walkforward_selection_path")
    if selection_path:
        selection_file = Path(selection_path)
        if selection_file.exists():
            selection_df = pd.read_csv(selection_file)
            if not selection_df.empty and "selection_date" in selection_df.columns:
                return normalize_day(selection_df["selection_date"].iloc[0])
    return normalize_day(TEST_START)


def resolve_effective_day(close: pd.DataFrame, effective_date: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    latest_close_day = normalize_day(close.index.max())
    if effective_date:
        effective_day = normalize_day(effective_date)
    else:
        effective_day = latest_close_day + pd.Timedelta(days=1)
    return latest_close_day, effective_day


def resolve_selection_day(
    effective_day: pd.Timestamp,
    anchor_day: pd.Timestamp,
    reselect_days: int,
) -> pd.Timestamp:
    if effective_day <= anchor_day:
        return anchor_day
    days_since_anchor = int((effective_day - anchor_day).days)
    block_index = days_since_anchor // int(reselect_days)
    return anchor_day + pd.Timedelta(days=block_index * int(reselect_days))


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    daily = metrics.get("daily_metrics", {})
    monthly = metrics.get("monthly_metrics", {})
    return {
        "total_return": float(metrics.get("total_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "active_ratio": float(metrics.get("active_ratio", 0.0)),
        "daily_target_hit_rate": float(daily.get("daily_target_hit_rate", 0.0)),
        "monthly_target_hit_rate": float(monthly.get("monthly_target_hit_rate", 0.0)),
    }


def build_signal_payload(
    summary_path: str | Path,
    effective_date: str | None,
    equity: float,
) -> dict[str, Any]:
    summary = load_summary(summary_path)
    config = summary["walkforward_config"]

    close = build_daily_close(load_all_pairs())
    if close.empty:
        raise ValueError("No daily close data available")
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)

    latest_close_day, effective_day = resolve_effective_day(close, effective_date)
    if effective_day > latest_close_day + pd.Timedelta(days=1):
        raise ValueError(
            "effective_date cannot be more than one day after the latest available close",
        )
    history = close.index[close.index < effective_day]
    if history.empty:
        raise ValueError(f"No history available before effective date {effective_day.date()}")

    market_day = normalize_day(history[-1])
    anchor_day = load_selection_anchor(summary)
    selection_day = resolve_selection_day(
        effective_day=effective_day,
        anchor_day=anchor_day,
        reselect_days=int(config["reselect_days"]),
    )

    target_cache = [
        (params, build_target_weights(close, params))
        for params in candidate_params()
    ]
    selection = select_walkforward_params_for_day(
        close,
        target_cache,
        selection_day,
        train_days=int(config["train_days"]),
        val_days=int(config["val_days"]),
        initial_cash=INITIAL_CASH,
    )
    if selection["status"] != "ok":
        raise ValueError(
            f"Walk-forward selection failed for {selection_day.date()}: {selection['status']}",
        )

    best_params = selection["best_params"]
    best_target_weights = selection["best_target_weights"]
    if market_day not in best_target_weights.index:
        raise ValueError(f"Target weights missing for market day {market_day.date()}")

    weights = best_target_weights.loc[market_day].fillna(0.0)
    weights = weights.astype("float64")
    notionals = (weights * float(equity)).astype("float64")
    gross_leverage = float(weights.abs().sum())
    active_weights = {
        pair: float(weight)
        for pair, weight in weights.items()
        if abs(float(weight)) > 1e-12
    }
    active_notionals = {
        pair: float(notional)
        for pair, notional in notionals.items()
        if abs(float(notional)) > 1e-8
    }

    return {
        "strategy_class": summary.get("strategy_class"),
        "summary_path": summary["_path"],
        "effective_date": effective_day.date().isoformat(),
        "market_day": market_day.date().isoformat(),
        "latest_available_close_day": latest_close_day.date().isoformat(),
        "selection_anchor_day": anchor_day.date().isoformat(),
        "selection_day": selection_day.date().isoformat(),
        "apply_until": (
            selection_day + pd.Timedelta(days=int(config["reselect_days"]) - 1)
        ).date().isoformat(),
        "selection_score": float(selection["best_score"]),
        "selected_params": asdict(best_params),
        "train_snapshot": compact_metrics(selection["train_metrics"]),
        "validation_snapshot": compact_metrics(selection["val_metrics"]),
        "gross_leverage": gross_leverage,
        "equity": float(equity),
        "weights": {pair: float(weights[pair]) for pair in weights.index},
        "active_weights": active_weights,
        "notionals": {pair: float(notionals[pair]) for pair in notionals.index},
        "active_notionals": active_notionals,
    }


def print_signal(payload: dict[str, Any]) -> None:
    print("=" * 72)
    print("  Walk-Forward Rotation Signal")
    print("=" * 72)
    print(f"Summary Path    : {payload['summary_path']}")
    print(f"Latest Close Day: {payload['latest_available_close_day']}")
    print(f"Market Day      : {payload['market_day']}")
    print(f"Effective Day   : {payload['effective_date']}")
    print(f"Selection Day   : {payload['selection_day']} -> {payload['apply_until']}")
    print(f"Selection Score : {payload['selection_score']:.4f}")
    print(f"Gross Leverage  : {payload['gross_leverage']:.3f}x")
    print(f"Equity Basis    : ${payload['equity']:,.2f}")

    params = payload["selected_params"]
    print("\nSelected Params")
    print(
        "  fast={lookback_fast}, slow={lookback_slow}, top_n={top_n}, "
        "vol_window={vol_window}, target_vol={target_vol_ann:.2f}, regime={regime_threshold:.2f}".format(
            **params,
        ),
    )

    print("\nTarget Weights")
    for pair, weight in payload["weights"].items():
        notional = payload["notionals"][pair]
        print(f"  {pair:8s}  {weight:+7.3%}  ${notional:>11,.2f}")

    if not payload["active_weights"]:
        print("  No active positions for this session.")


def main() -> None:
    args = parse_args()

    if args.command == "signal":
        payload = build_signal_payload(
            summary_path=args.summary,
            effective_date=args.effective_date,
            equity=args.equity,
        )
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print_signal(payload)


if __name__ == "__main__":
    main()
