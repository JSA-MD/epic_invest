#!/usr/bin/env python3
"""Regime-switched archive-of-experts GP.

This experiment keeps the current multi-tree + domain-primitive GP as the
specialist learner, but changes the search structure:

- use the current global champion as the safe fallback
- split the market into a small set of regime buckets
- train specialist GPs only on the days for each bucket
- promote a specialist only if it beats the fallback on bucket-local
  validation return and win rate
- route each day to the promoted specialist for that regime, otherwise stay on
  the global fallback
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd

import gp_crypto_evolution_domain_gp as domain_gp
import gp_crypto_evolution_multitree_domain as multi_domain
from gp_crypto_evolution import (
    DAILY_MAX_LOSS_PCT,
    DAILY_TARGET_PCT,
    INITIAL_CASH,
    MODELS_DIR,
    PRIMARY_PAIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    daily_session_backtest,
    load_all_pairs,
    summarize_monthly_returns,
    summarize_period_returns,
)


RNG_SEED = 42
LAYOUTS = ("tb", "tbd", "tbv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regime-switched archive-of-experts GP over the current multi-tree domain GP.",
    )
    parser.add_argument("--pop-size", type=int, default=12)
    parser.add_argument("--n-gen", type=int, default=1)
    parser.add_argument("--fold-days", type=int, default=20)
    parser.add_argument("--step-days", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=10)
    parser.add_argument("--min-val-days", type=int, default=5)
    parser.add_argument("--committee-size", type=int, default=3)
    parser.add_argument(
        "--layouts",
        default="tb,tbd,tbv",
        help="Comma-separated router layouts to evaluate. Supported: tb,tbd,tbv",
    )
    parser.add_argument(
        "--global-model",
        default=str(MODELS_DIR / "best_crypto_gp_multitree_domain.dill"),
    )
    parser.add_argument(
        "--baseline-meta",
        default=str(MODELS_DIR / "best_crypto_gp_multitree_domain_after_fix.json"),
    )
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_archive_experts.dill"),
    )
    parser.add_argument(
        "--meta-out",
        default=str(MODELS_DIR / "best_crypto_gp_archive_experts_meta.json"),
    )
    return parser.parse_args()


def parse_layouts(raw: str) -> list[str]:
    layouts = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in LAYOUTS:
            raise ValueError(f"Unsupported layout: {name}")
        layouts.append(name)
    if not layouts:
        raise ValueError("No router layouts specified")
    return layouts


def summarize_for_report(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "total_return": float(metrics["total_return"]),
        "win_rate": float(metrics["win_rate"]),
        "daily_target_hit_rate": float(metrics["daily_metrics"]["daily_target_hit_rate"]),
        "avg_daily_return": float(metrics["daily_metrics"]["avg_daily_return"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "monthly_target_hit_rate": float(metrics["monthly_metrics"]["monthly_target_hit_rate"]),
    }


def load_baseline_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_global_expert(path: Path):
    with open(path, "rb") as f:
        return dill.load(f)


def build_daily_regime_features(df_slice: pd.DataFrame) -> pd.DataFrame:
    per_day = df_slice.groupby(df_slice.index.normalize()).first().copy()
    per_day["trend_signal"] = per_day[f"{PRIMARY_PAIR}_dc_trend_05"].astype("float64")
    per_day["breadth"] = per_day["cross_breadth"].astype("float64")
    per_day["dispersion"] = per_day["cross_dispersion"].astype("float64")
    close = per_day[f"{PRIMARY_PAIR}_close"].astype("float64").replace(0.0, np.nan)
    per_day["vol_ratio"] = (
        per_day[f"{PRIMARY_PAIR}_atr_14"].astype("float64") / close
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return per_day[["trend_signal", "breadth", "dispersion", "vol_ratio"]]


def fit_regime_thresholds(train_daily: pd.DataFrame) -> dict[str, float]:
    return {
        "breadth_threshold": 0.5,
        "dispersion_threshold": float(train_daily["dispersion"].median()),
        "vol_threshold": float(train_daily["vol_ratio"].median()),
    }


def assign_layout_keys(
    daily_features: pd.DataFrame,
    layout: str,
    thresholds: dict[str, float],
) -> pd.Series:
    trend = (daily_features["trend_signal"] > 0.0).astype(int).astype(str)
    breadth = (daily_features["breadth"] >= thresholds["breadth_threshold"]).astype(int).astype(str)
    dispersion = (daily_features["dispersion"] >= thresholds["dispersion_threshold"]).astype(int).astype(str)
    vol = (daily_features["vol_ratio"] >= thresholds["vol_threshold"]).astype(int).astype(str)

    if layout == "tb":
        labels = "T" + trend + "B" + breadth
    elif layout == "tbd":
        labels = "T" + trend + "B" + breadth + "D" + dispersion
    elif layout == "tbv":
        labels = "T" + trend + "B" + breadth + "V" + vol
    else:
        raise ValueError(layout)
    labels.index = daily_features.index
    return labels


def subset_by_days(df_slice: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    if len(days) == 0:
        return df_slice.iloc[0:0].copy()
    mask = df_slice.index.normalize().isin(days)
    return df_slice.loc[mask].copy()


def local_fold_config(n_days: int, args: argparse.Namespace) -> tuple[int, int]:
    fold_days = min(args.fold_days, max(8, n_days))
    if n_days > 12:
        fold_days = min(fold_days, max(10, n_days // 2))
    step_days = min(args.step_days, max(4, fold_days // 2))
    return int(fold_days), int(step_days)


def risk_params_from_result(result: dict[str, Any]) -> dict[str, float]:
    risk = np.asarray(result["risk"], dtype="float64")
    risk_summary = float(np.median(risk)) if len(risk) else 0.5
    return {
        "reward_multiple": float(result["reward_multiple"]),
        "entry_threshold": float(result["entry_threshold"]),
        "trail_activation_pct": float(np.clip(0.0035 + 0.0035 * risk_summary, 0.003, 0.009)),
        "trail_distance_pct": float(np.clip(0.0020 + 0.0030 * (1.0 - risk_summary), 0.002, 0.008)),
        "trail_floor_pct": float(np.clip(0.0015 + 0.0020 * (1.0 - risk_summary), 0.001, 0.005)),
    }


def execute_expert_day(
    expert,
    role_psets: dict[str, Any],
    day_df: pd.DataFrame,
    equity: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(expert, tuple):
        models = list(expert)
    elif isinstance(expert, list) and expert and hasattr(expert[0], "fitness"):
        models = list(expert)
    else:
        models = [expert]
    eval_results = [multi_domain.evaluate_trees(model, role_psets, day_df) for model in models]
    signal_stack = np.vstack([res["signal"] for res in eval_results])
    merged_signal = np.median(signal_stack, axis=0)
    risk_params_list = [risk_params_from_result(res) for res in eval_results]
    risk_params = {
        "reward_multiple": float(np.median([item["reward_multiple"] for item in risk_params_list])),
        "entry_threshold": float(np.median([item["entry_threshold"] for item in risk_params_list])),
        "trail_activation_pct": float(np.median([item["trail_activation_pct"] for item in risk_params_list])),
        "trail_distance_pct": float(np.median([item["trail_distance_pct"] for item in risk_params_list])),
        "trail_floor_pct": float(np.median([item["trail_floor_pct"] for item in risk_params_list])),
    }
    session = daily_session_backtest(
        day_df,
        merged_signal,
        pair=PRIMARY_PAIR,
        initial_cash=equity,
        daily_target_pct=DAILY_TARGET_PCT,
        daily_stop_pct=DAILY_MAX_LOSS_PCT,
        reward_multiple=risk_params["reward_multiple"],
        trail_activation_pct=risk_params["trail_activation_pct"],
        trail_distance_pct=risk_params["trail_distance_pct"],
        trail_floor_pct=risk_params["trail_floor_pct"],
        entry_threshold=risk_params["entry_threshold"],
    )
    if session["trade_log"]:
        trade = dict(session["trade_log"][0])
    else:
        trade = {
            "date": str(pd.Timestamp(day_df.index[0]).date()),
            "direction": "FLAT",
            "gross_return": 0.0,
            "net_return": 0.0,
            "exit_reason": "no_trade_log",
        }
    return session, trade


def aggregate_day_sessions(
    daily_returns: list[float],
    daily_index: list[pd.Timestamp],
    equity_curve: list[float],
    trade_log: list[dict[str, Any]],
) -> dict[str, Any]:
    period_returns = np.asarray(daily_returns, dtype="float64")
    curve = np.asarray(equity_curve, dtype="float64")
    dates = pd.DatetimeIndex(daily_index)
    final_equity = float(curve[-1]) if len(curve) else INITIAL_CASH
    total_return = float(final_equity / INITIAL_CASH - 1.0)
    if len(period_returns) > 1 and np.std(period_returns) > 1e-12:
        sharpe = float(np.mean(period_returns) / np.std(period_returns) * np.sqrt(365.25))
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(curve) if len(curve) else np.asarray([INITIAL_CASH], dtype="float64")
    max_drawdown = float(np.min(curve / peak - 1.0)) if len(curve) else 0.0
    daily_metrics = summarize_period_returns(period_returns)
    monthly_metrics = summarize_monthly_returns(period_returns, dates)
    n_trades = int(sum(1 for trade in trade_log if trade.get("direction") != "FLAT"))
    win_rate = float(np.mean(period_returns > 0.0)) if len(period_returns) else 0.0
    return {
        "total_return": total_return,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "final_equity": final_equity,
        "equity_curve": curve,
        "net_ret": period_returns,
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
        "win_rate": win_rate,
        "target_hit_rate": float(daily_metrics["daily_target_hit_rate"]),
        "trade_log": trade_log,
        "active_ratio": float(n_trades / len(period_returns)) if len(period_returns) else 0.0,
    }


def route_backtest(
    df_slice: pd.DataFrame,
    role_psets: dict[str, Any],
    global_expert,
    day_layout_keys: pd.Series,
    archive: dict[str, Any],
) -> dict[str, Any]:
    equity = float(INITIAL_CASH)
    daily_returns: list[float] = []
    daily_index: list[pd.Timestamp] = []
    equity_curve = [float(INITIAL_CASH)]
    trade_log: list[dict[str, Any]] = []
    route_counts: dict[str, int] = {}

    for day, day_df in df_slice.groupby(df_slice.index.normalize()):
        key = str(day_layout_keys.get(day, "GLOBAL"))
        source = key if key in archive else "GLOBAL"
        expert = archive.get(key, global_expert)
        if expert is None:
            daily_returns.append(0.0)
            daily_index.append(pd.Timestamp(day))
            equity_curve.append(equity)
            trade_log.append(
                {
                    "date": str(pd.Timestamp(day).date()),
                    "direction": "FLAT",
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "exit_reason": "no_expert",
                    "route_key": key,
                    "route_source": source,
                }
            )
            route_counts[source] = route_counts.get(source, 0) + 1
            continue

        try:
            session, trade = execute_expert_day(expert, role_psets, day_df, equity)
        except Exception:
            if source != "GLOBAL" and global_expert is not None:
                source = "GLOBAL_FALLBACK"
                session, trade = execute_expert_day(global_expert, role_psets, day_df, equity)
            else:
                daily_returns.append(0.0)
                daily_index.append(pd.Timestamp(day))
                equity_curve.append(equity)
                trade_log.append(
                    {
                        "date": str(pd.Timestamp(day).date()),
                        "direction": "FLAT",
                        "gross_return": 0.0,
                        "net_return": 0.0,
                        "exit_reason": "expert_error",
                        "route_key": key,
                        "route_source": source,
                    }
                )
                route_counts[source] = route_counts.get(source, 0) + 1
                continue
        equity = float(session["final_equity"])
        daily_ret = float(session["net_ret"][0]) if len(session["net_ret"]) else 0.0
        daily_returns.append(daily_ret)
        daily_index.append(pd.Timestamp(day))
        equity_curve.append(equity)
        trade["route_key"] = key
        trade["route_source"] = source
        trade_log.append(trade)
        route_counts[source] = route_counts.get(source, 0) + 1

    result = aggregate_day_sessions(daily_returns, daily_index, equity_curve, trade_log)
    result["route_counts"] = route_counts
    return result


def evaluate_fixed_expert(df_slice: pd.DataFrame, role_psets: dict[str, Any], expert) -> dict[str, Any]:
    day_keys = pd.Series(["GLOBAL"] * len(pd.Index(df_slice.index.normalize().unique())), index=pd.Index(df_slice.index.normalize().unique()))
    return route_backtest(df_slice, role_psets, expert, day_keys, {})


def train_specialist(
    train_bucket_df: pd.DataFrame,
    val_bucket_df: pd.DataFrame,
    role_psets: dict[str, Any],
    toolbox,
    args: argparse.Namespace,
):
    n_days = len(pd.Index(train_bucket_df.index.normalize().unique()))
    fold_days, step_days = local_fold_config(n_days, args)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        pareto = multi_domain.run_evolution(
            train_df=train_bucket_df,
            toolbox=toolbox,
            role_psets=role_psets,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            fold_days=fold_days,
            step_days=step_days,
        )
    specialist, val_pick = multi_domain.select_best_on_validation(pareto, role_psets, val_bucket_df)
    return pareto, specialist, val_pick, len(pareto), {"fold_days": fold_days, "step_days": step_days}


def build_archive_for_layout(
    layout: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_keys: pd.Series,
    val_keys: pd.Series,
    role_psets: dict[str, Any],
    toolbox,
    global_expert,
    global_val_bucket_cache: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive: dict[str, Any] = {}
    bucket_records: list[dict[str, Any]] = []

    for bucket_key, train_days in train_keys.groupby(train_keys):
        val_days = val_keys[val_keys == bucket_key].index
        train_day_count = int(len(train_days))
        val_day_count = int(len(val_days))
        record = {
            "bucket": str(bucket_key),
            "train_days": train_day_count,
            "val_days": val_day_count,
            "adopted": False,
            "reason": "",
        }
        if train_day_count < args.min_train_days:
            record["reason"] = "insufficient_train_days"
            bucket_records.append(record)
            continue
        if val_day_count < args.min_val_days:
            record["reason"] = "insufficient_val_days"
            bucket_records.append(record)
            continue

        train_bucket_df = subset_by_days(train_df, pd.Index(train_days.index))
        val_bucket_df = subset_by_days(val_df, val_days)

        try:
            pareto, specialist, val_pick, pareto_size, fold_cfg = train_specialist(
                train_bucket_df=train_bucket_df,
                val_bucket_df=val_bucket_df,
                role_psets=role_psets,
                toolbox=toolbox,
                args=args,
            )
        except Exception as exc:
            record["reason"] = f"train_error:{exc}"
            bucket_records.append(record)
            continue

        specialist_val = evaluate_fixed_expert(val_bucket_df, role_psets, specialist)
        if bucket_key not in global_val_bucket_cache:
            global_val_bucket_cache[bucket_key] = evaluate_fixed_expert(val_bucket_df, role_psets, global_expert)
        global_val = global_val_bucket_cache[bucket_key]

        eligible_models = []
        for model in pareto:
            try:
                model_val = evaluate_fixed_expert(val_bucket_df, role_psets, model)
            except Exception:
                continue
            if (
                model_val["total_return"] > global_val["total_return"] + 1e-9
                and model_val["win_rate"] >= global_val["win_rate"] - 1e-9
                and model_val["max_drawdown"] >= global_val["max_drawdown"] - 0.01
            ):
                eligible_models.append((strategy_score(model_val), model, model_val))

        eligible_models.sort(key=lambda item: item[0])
        committee = [item[1] for item in eligible_models[:max(1, args.committee_size)]]
        try:
            committee_val = (
                evaluate_fixed_expert(val_bucket_df, role_psets, committee)
                if committee
                else specialist_val
            )
        except Exception:
            committee_val = specialist_val
        improved_return = committee_val["total_return"] > global_val["total_return"] + 1e-9
        improved_win = committee_val["win_rate"] >= global_val["win_rate"] - 1e-9
        improved_dd = committee_val["max_drawdown"] >= global_val["max_drawdown"] - 0.01
        adopted = bool(committee and improved_return and improved_win and improved_dd)

        if adopted:
            archive[str(bucket_key)] = tuple(committee) if len(committee) > 1 else committee[0]
            record["reason"] = "beats_global_on_bucket_val"
        else:
            record["reason"] = "fails_local_promotion_rule"

        record.update(
            {
                "adopted": adopted,
                "pareto_size": pareto_size,
                "eligible_models": int(len(eligible_models)),
                "committee_size": int(len(committee)),
                "fold_days": fold_cfg["fold_days"],
                "step_days": fold_cfg["step_days"],
                "validation_pick": val_pick,
                "specialist_val": summarize_for_report(specialist_val),
                "committee_val": summarize_for_report(committee_val),
                "global_val": summarize_for_report(global_val),
            }
        )
        bucket_records.append(record)

    return archive, bucket_records


def strategy_score(metrics: dict[str, Any]) -> float:
    daily = metrics["daily_metrics"]
    monthly = metrics["monthly_metrics"]
    return (
        metrics["total_return"] * -150.0
        + abs(metrics["max_drawdown"]) * 70.0
        + (1.0 - daily["daily_target_hit_rate"]) * 50.0
        + daily["daily_shortfall_mean"] * 120.0
        + monthly["monthly_shortfall_mean"] * 80.0
        - metrics["win_rate"] * 10.0
    )


def choose_candidate(
    candidates: list[dict[str, Any]],
    baseline_val: dict[str, Any],
) -> dict[str, Any]:
    better = []
    for candidate in candidates:
        val = candidate["validation"]
        if (
            val["total_return"] > baseline_val["total_return"] + 1e-9
            and val["win_rate"] >= baseline_val["win_rate"] - 1e-9
        ):
            better.append(candidate)
    if better:
        return min(better, key=lambda item: strategy_score(item["validation"]))
    return min(candidates, key=lambda item: strategy_score(item["validation"]))


def print_summary(label: str, metrics: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print(f"  Return:       {metrics['total_return']*100:+.2f}%")
    print(f"  Win Rate:     {metrics['win_rate']*100:.1f}%")
    print(f"  Target Hit:   {metrics['daily_metrics']['daily_target_hit_rate']*100:.1f}%")
    print(f"  Avg Daily:    {metrics['daily_metrics']['avg_daily_return']*100:+.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Month Hit:    {metrics['monthly_metrics']['monthly_target_hit_rate']*100:.1f}%")
    print(f"  Trades:       {metrics['n_trades']}")


def main() -> None:
    args = parse_args()
    if args.pop_size % 4 != 0:
        raise ValueError("pop-size must be divisible by 4 for selTournamentDCD")

    layouts = parse_layouts(args.layouts)
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    multi_domain.ensure_creator_types()
    role_psets = multi_domain.build_role_psets()
    toolbox = multi_domain.build_toolbox(role_psets)

    print("=" * 92)
    print("  Archive-of-Experts GP")
    print("=" * 92)

    print("\n[Phase 1] Data Loading")
    df_all = domain_gp.build_domain_feature_frame(load_all_pairs())
    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:VAL_END].copy()
    test_df = df_all.loc[TEST_START:TEST_END].copy()
    print(f"  Train: {len(train_df)} bars | Val: {len(val_df)} bars | Test: {len(test_df)} bars")

    print("\n[Phase 2] Global Fallback")
    global_model_path = Path(args.global_model)
    if not global_model_path.exists():
        raise FileNotFoundError(f"Missing global model: {global_model_path}")
    global_expert = load_global_expert(global_model_path)
    baseline_meta = load_baseline_meta(Path(args.baseline_meta))
    routed_global_val = evaluate_fixed_expert(val_df, role_psets, global_expert)
    routed_global_test = evaluate_fixed_expert(test_df, role_psets, global_expert)
    routed_global_full = evaluate_fixed_expert(df_all, role_psets, global_expert)
    print_summary("VALIDATION (Global Routed Baseline)", routed_global_val)
    print_summary("TEST (Global Routed Baseline)", routed_global_test)

    print("\n[Phase 3] Regime Layout Search")
    train_daily = build_daily_regime_features(train_df)
    val_daily = build_daily_regime_features(val_df)
    test_daily = build_daily_regime_features(test_df)
    full_daily = build_daily_regime_features(df_all)
    thresholds = fit_regime_thresholds(train_daily)
    print(
        "  Thresholds:"
        f" breadth={thresholds['breadth_threshold']:.3f},"
        f" dispersion={thresholds['dispersion_threshold']:.6f},"
        f" vol={thresholds['vol_threshold']:.6f}"
    )

    candidates: list[dict[str, Any]] = []
    global_val_bucket_cache: dict[str, dict[str, Any]] = {}
    for layout in layouts:
        print(f"\n  Layout: {layout}")
        train_keys = assign_layout_keys(train_daily, layout, thresholds)
        val_keys = assign_layout_keys(val_daily, layout, thresholds)
        test_keys = assign_layout_keys(test_daily, layout, thresholds)
        full_keys = assign_layout_keys(full_daily, layout, thresholds)
        print(f"    Train buckets: {train_keys.value_counts().sort_index().to_dict()}")

        archive, bucket_records = build_archive_for_layout(
            layout=layout,
            train_df=train_df,
            val_df=val_df,
            train_keys=train_keys,
            val_keys=val_keys,
            role_psets=role_psets,
            toolbox=toolbox,
            global_expert=global_expert,
            global_val_bucket_cache=global_val_bucket_cache,
            args=args,
        )
        routed_val = route_backtest(val_df, role_psets, global_expert, val_keys, archive)
        routed_test = route_backtest(test_df, role_psets, global_expert, test_keys, archive)
        routed_full = route_backtest(df_all, role_psets, global_expert, full_keys, archive)
        print(
            f"    Adopted specialists: {sum(1 for r in bucket_records if r['adopted'])}/"
            f"{len(bucket_records)}"
        )
        print(
            f"    Val: return={routed_val['total_return']*100:+.2f}% "
            f"win={routed_val['win_rate']*100:.1f}% "
            f"hit={routed_val['daily_metrics']['daily_target_hit_rate']*100:.1f}%"
        )
        print(
            f"    Test: return={routed_test['total_return']*100:+.2f}% "
            f"win={routed_test['win_rate']*100:.1f}% "
            f"hit={routed_test['daily_metrics']['daily_target_hit_rate']*100:.1f}%"
        )

        candidates.append(
            {
                "layout": layout,
                "archive": archive,
                "bucket_records": bucket_records,
                "validation": routed_val,
                "test": routed_test,
                "full": routed_full,
                "train_bucket_counts": train_keys.value_counts().sort_index().to_dict(),
            }
        )

    if not candidates:
        raise RuntimeError("No archive candidates were evaluated")

    selected = choose_candidate(candidates, routed_global_val)
    print("\n[Phase 4] Candidate Selection")
    print(f"  Selected layout: {selected['layout']}")
    print_summary("VALIDATION (Selected Archive)", selected["validation"])
    print_summary("TEST (Selected Archive)", selected["test"])
    print_summary("FULL PERIOD (Selected Archive)", selected["full"])

    compare = {
        "official_baseline_after_fix": baseline_meta,
        "routed_global_baseline": {
            "validation": summarize_for_report(routed_global_val),
            "test": summarize_for_report(routed_global_test),
            "full": summarize_for_report(routed_global_full),
        },
        "selected_archive": {
            "layout": selected["layout"],
            "validation": summarize_for_report(selected["validation"]),
            "test": summarize_for_report(selected["test"]),
            "full": summarize_for_report(selected["full"]),
        },
        "test_delta_vs_routed_global": {
            "return_delta": float(selected["test"]["total_return"] - routed_global_test["total_return"]),
            "win_rate_delta": float(selected["test"]["win_rate"] - routed_global_test["win_rate"]),
            "target_hit_delta": float(
                selected["test"]["daily_metrics"]["daily_target_hit_rate"]
                - routed_global_test["daily_metrics"]["daily_target_hit_rate"]
            ),
            "avg_daily_return_delta": float(
                selected["test"]["daily_metrics"]["avg_daily_return"]
                - routed_global_test["daily_metrics"]["avg_daily_return"]
            ),
            "max_drawdown_delta": float(selected["test"]["max_drawdown"] - routed_global_test["max_drawdown"]),
        },
    }

    model_payload = {
        "algorithm": "archive_of_experts_gp",
        "global_model_path": str(global_model_path),
        "selected_layout": selected["layout"],
        "thresholds": thresholds,
        "archive": selected["archive"],
        "bucket_records": selected["bucket_records"],
        "train_bucket_counts": selected["train_bucket_counts"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)
    with open(model_path, "wb") as f:
        dill.dump(model_payload, f)

    meta = {
        "algorithm": "archive_of_experts_gp",
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period": f"{VAL_START} ~ {VAL_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "fold_days": args.fold_days,
        "step_days": args.step_days,
        "min_train_days": args.min_train_days,
        "min_val_days": args.min_val_days,
        "layouts_evaluated": layouts,
        "selected_layout": selected["layout"],
        "thresholds": thresholds,
        "compare": compare,
        "candidate_summaries": [
            {
                "layout": candidate["layout"],
                "validation": summarize_for_report(candidate["validation"]),
                "test": summarize_for_report(candidate["test"]),
                "full": summarize_for_report(candidate["full"]),
                "adopted_buckets": int(sum(1 for record in candidate["bucket_records"] if record["adopted"])),
                "bucket_records": candidate["bucket_records"],
                "train_bucket_counts": candidate["train_bucket_counts"],
            }
            for candidate in candidates
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(domain_gp.json_ready(meta), f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
