#!/usr/bin/env python3
"""Multi-tree domain GP with per-day adaptive stop width and locked +0.5% target."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
from deap import base, tools

import gp_crypto_evolution_domain_gp as domain_gp
import gp_crypto_evolution_multitree_domain as base_mt
from gp_crypto_evolution import (
    COMMISSION_PCT,
    DAILY_TARGET_PCT,
    INITIAL_CASH,
    MAX_LEN,
    MODELS_DIR,
    P_CX,
    P_MUT,
    PRIMARY_PAIR,
    TEST_START,
    TRAIN_START,
    VAL_START,
    load_all_pairs,
    summarize_monthly_returns,
    summarize_period_returns,
)

RNG_SEED = 42
DEFAULT_POP_SIZE = 24
DEFAULT_N_GEN = 3
DEFAULT_FOLD_DAYS = 30
DEFAULT_STEP_DAYS = 15
MIN_STOP_PCT = 0.0025
MAX_STOP_PCT = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-tree GP with adaptive daily stop width and a locked +0.5% target.",
    )
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN)
    parser.add_argument("--fold-days", type=int, default=DEFAULT_FOLD_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_multitree_locked_target.dill"),
    )
    parser.add_argument(
        "--meta-out",
        default=str(MODELS_DIR / "best_crypto_gp_multitree_locked_target_meta.json"),
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


def daily_session_backtest_locked_target(
    df_slice: pd.DataFrame,
    desired_pcts: np.ndarray,
    stop_pct_by_day: pd.Series,
    entry_threshold_by_day: pd.Series,
    trail_activation_by_day: pd.Series,
    trail_distance_by_day: pd.Series,
    trail_floor_by_day: pd.Series,
    pair: str = PRIMARY_PAIR,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION_PCT,
    daily_target_pct: float = DAILY_TARGET_PCT,
) -> dict[str, Any]:
    if df_slice.empty:
        return {
            "total_return": 0.0,
            "n_trades": 0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "final_equity": initial_cash,
            "equity_curve": np.asarray([initial_cash], dtype="float64"),
            "net_ret": np.asarray([], dtype="float64"),
            "daily_metrics": summarize_period_returns(np.asarray([], dtype="float64")),
            "monthly_metrics": summarize_monthly_returns(np.asarray([], dtype="float64"), pd.DatetimeIndex([])),
            "win_rate": 0.0,
            "target_hit_rate": 0.0,
            "avg_stop_pct": 0.0,
            "avg_reward_multiple": 0.0,
            "trade_log": [],
        }

    idx = pd.DatetimeIndex(df_slice.index)
    close = df_slice[f"{pair}_close"].to_numpy(dtype="float64")
    high = df_slice[f"{pair}_high"].to_numpy(dtype="float64")
    low = df_slice[f"{pair}_low"].to_numpy(dtype="float64")
    signal = np.where(np.isfinite(desired_pcts), desired_pcts, 0.0)
    signal = np.clip(signal, -100.0, 100.0)

    equity = float(initial_cash)
    equity_curve = [initial_cash]
    daily_returns = []
    trade_log = []
    used_stops = []
    used_rewards = []

    unique_days = pd.Index(idx.normalize().unique())
    for day in unique_days:
        pos = np.where(idx.normalize() == day)[0]
        if len(pos) < 2:
            equity_curve.append(equity)
            daily_returns.append(0.0)
            continue

        day_key = pd.Timestamp(day).normalize()
        stop_pct = float(stop_pct_by_day.get(day_key, MAX_STOP_PCT))
        stop_pct = float(np.clip(stop_pct, MIN_STOP_PCT, MAX_STOP_PCT))
        entry_threshold = float(entry_threshold_by_day.get(day_key, 20.0))
        trail_activation_pct = float(trail_activation_by_day.get(day_key, daily_target_pct))
        trail_distance_pct = float(trail_distance_by_day.get(day_key, stop_pct * 0.60))
        trail_floor_pct = float(trail_floor_by_day.get(day_key, stop_pct * 0.40))

        reward_multiple = float(daily_target_pct / max(stop_pct, 1e-12))
        gross_stop = -stop_pct + 2 * commission
        gross_target = daily_target_pct + 2 * commission
        gross_trail_activation = trail_activation_pct + 2 * commission
        gross_trail_floor = trail_floor_pct + 2 * commission

        start = pos[0]
        end = pos[-1]
        entry_signal = float(signal[start])
        if abs(entry_signal) <= entry_threshold:
            equity_curve.append(equity)
            daily_returns.append(0.0)
            trade_log.append(
                {
                    "date": str(day.date()),
                    "direction": "FLAT",
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "exit_reason": "no_signal",
                    "stop_pct": stop_pct,
                    "reward_multiple": reward_multiple,
                }
            )
            continue

        direction = 1.0 if entry_signal > 0 else -1.0
        entry_price = float(close[start])
        exit_price = float(close[end])
        exit_reason = "eod"
        gross_return = direction * (exit_price / entry_price - 1.0)
        best_favorable = 0.0
        trail_active = False

        for j in pos[1:]:
            bar_high = direction * (float(high[j]) / entry_price - 1.0)
            bar_low = direction * (float(low[j]) / entry_price - 1.0)
            favorable = max(bar_high, bar_low)
            adverse = min(bar_high, bar_low)
            best_favorable = max(best_favorable, favorable)

            dynamic_stop = gross_stop
            if best_favorable >= gross_trail_activation:
                trail_active = True
                dynamic_stop = max(
                    gross_stop,
                    gross_trail_floor,
                    best_favorable - trail_distance_pct,
                )

            stop_hit = adverse <= dynamic_stop
            target_hit = favorable >= gross_target

            if stop_hit and target_hit:
                gross_return = dynamic_stop
                exit_reason = "trail_stop_and_target_same_bar" if trail_active else "stop_and_target_same_bar"
                exit_price = entry_price * (1.0 + direction * gross_return)
                break
            if stop_hit:
                gross_return = dynamic_stop
                exit_reason = "trail_stop" if trail_active and dynamic_stop > gross_stop else "stop"
                exit_price = entry_price * (1.0 + direction * gross_return)
                break
            if target_hit:
                gross_return = gross_target
                exit_reason = "target"
                exit_price = entry_price * (1.0 + direction * gross_target)
                break

        net_return = gross_return - 2 * commission
        equity *= (1.0 + net_return)
        equity_curve.append(equity)
        daily_returns.append(net_return)
        used_stops.append(stop_pct)
        used_rewards.append(reward_multiple)
        trade_log.append(
            {
                "date": str(day.date()),
                "direction": "LONG" if direction > 0 else "SHORT",
                "signal": entry_signal,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "exit_reason": exit_reason,
                "stop_pct": stop_pct,
                "reward_multiple": reward_multiple,
                "trail_active": trail_active,
            }
        )

    daily_returns = np.asarray(daily_returns, dtype="float64")
    equity_curve = np.asarray(equity_curve, dtype="float64")
    total_return = float(equity / initial_cash - 1.0)
    n_trades = int(sum(1 for row in trade_log if row["direction"] != "FLAT"))

    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-12:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365.25))
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    max_dd = float(np.min(equity_curve / peak - 1.0))
    daily_metrics = summarize_period_returns(daily_returns)
    daily_index = pd.DatetimeIndex(unique_days)
    monthly_metrics = summarize_monthly_returns(daily_returns, daily_index)
    win_rate = float(np.mean(daily_returns > 0.0)) if len(daily_returns) else 0.0

    return {
        "total_return": total_return,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(equity),
        "equity_curve": equity_curve,
        "net_ret": daily_returns,
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
        "win_rate": win_rate,
        "target_hit_rate": daily_metrics["daily_target_hit_rate"],
        "avg_stop_pct": float(np.mean(used_stops)) if used_stops else 0.0,
        "avg_reward_multiple": float(np.mean(used_rewards)) if used_rewards else 0.0,
        "trade_log": trade_log,
    }


def evaluate_trees(
    individual,
    role_psets,
    df_slice: pd.DataFrame,
) -> dict[str, Any]:
    funcs = base_mt.compile_heads(individual, role_psets)
    inputs = domain_gp.build_domain_inputs(df_slice)

    part_raw = np.asarray(funcs["participation"](*inputs), dtype="float64")
    dir_raw = np.asarray(funcs["direction"](*inputs), dtype="float64")
    risk_raw = np.asarray(funcs["risk"](*inputs), dtype="float64")

    part_raw = np.where(np.isfinite(part_raw), part_raw, 0.0)
    dir_raw = np.where(np.isfinite(dir_raw), dir_raw, 0.0)
    risk_raw = np.where(np.isfinite(risk_raw), risk_raw, 0.0)

    participation = np.where(part_raw > 0.5, 1.0, 0.0)
    direction = np.clip(-np.tanh(dir_raw), -1.0, 1.0)
    risk = np.clip(0.5 * (np.tanh(risk_raw / 5.0) + 1.0), 0.0, 1.0)
    signal = np.where(participation > 0.5, direction * (0.20 + 0.80 * risk) * 100.0, 0.0)

    idx = pd.DatetimeIndex(df_slice.index)
    day_index = pd.DatetimeIndex(idx.normalize().unique())
    risk_series = pd.Series(risk, index=idx).groupby(idx.normalize()).first().reindex(day_index).fillna(0.0)

    stop_pct_by_day = MIN_STOP_PCT + risk_series * (MAX_STOP_PCT - MIN_STOP_PCT)
    entry_threshold_by_day = 4.0 + (1.0 - risk_series) * 16.0
    trail_activation_by_day = 0.003 + risk_series * 0.002
    trail_distance_by_day = 0.001 + risk_series * 0.002
    trail_floor_by_day = 0.0005 + risk_series * 0.0015

    primary = daily_session_backtest_locked_target(
        df_slice=df_slice,
        desired_pcts=signal,
        stop_pct_by_day=stop_pct_by_day,
        entry_threshold_by_day=entry_threshold_by_day,
        trail_activation_by_day=trail_activation_by_day,
        trail_distance_by_day=trail_distance_by_day,
        trail_floor_by_day=trail_floor_by_day,
        pair=PRIMARY_PAIR,
        initial_cash=INITIAL_CASH,
        daily_target_pct=DAILY_TARGET_PCT,
    )

    return {
        "primary": primary,
        "participation": participation,
        "direction": direction,
        "risk": risk,
        "signal": signal,
        "avg_stop_pct": float(stop_pct_by_day.mean()),
        "avg_reward_multiple": float((DAILY_TARGET_PCT / stop_pct_by_day.clip(lower=1e-12)).mean()),
        "avg_entry_threshold": float(entry_threshold_by_day.mean()),
    }


def evaluate_fold(individual, role_psets, df_slice: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
    try:
        result = evaluate_trees(individual, role_psets, df_slice)
        primary = result["primary"]
        daily = primary["daily_metrics"]
        monthly = primary["monthly_metrics"]
        complexity = float(sum(len(tree) for tree in individual) / (3.0 * MAX_LEN))

        if len(daily["daily_returns"]) < 20 or primary["n_trades"] < 3:
            return (1e6, 1e6, 1e6, 1e6, 1e6, 1e3)
        if primary["max_drawdown"] < -0.35 or daily["worst_day"] <= -0.05:
            return (1e6, 1e6, 1e6, 1e6, 1e6, 1e3)

        stop_efficiency = (
            float(monthly["monthly_shortfall_sum"])
            + float(result["avg_stop_pct"]) * 10.0
            + max(0.0, 0.80 - float(result["avg_reward_multiple"])) * 0.25
        )

        return (
            float(-primary["total_return"]),
            float(daily["daily_shortfall_mean"]),
            float(abs(primary["max_drawdown"])),
            float(1.0 - daily["daily_target_hit_rate"]),
            stop_efficiency,
            complexity,
        )
    except Exception:
        return (1e6, 1e6, 1e6, 1e6, 1e6, 1e3)


def evaluate_individual(individual, role_psets, folds: list[pd.DataFrame]) -> tuple[float, float, float, float, float, float]:
    arr = np.asarray([evaluate_fold(individual, role_psets, fold) for fold in folds], dtype="float64")
    return tuple(np.mean(arr, axis=0).tolist())


def validation_score(metrics: dict[str, Any]) -> float:
    daily = metrics["daily_metrics"]
    monthly = metrics["monthly_metrics"]
    return (
        metrics["total_return"] * -150.0
        + abs(metrics["max_drawdown"]) * 60.0
        + (1.0 - daily["daily_target_hit_rate"]) * 80.0
        + daily["daily_shortfall_mean"] * 140.0
        + monthly["monthly_shortfall_mean"] * 80.0
        - metrics["win_rate"] * 12.0
    )


def run_evolution(train_df, toolbox: base.Toolbox, role_psets, pop_size: int, n_gen: int, fold_days: int, step_days: int):
    if pop_size % 4 != 0:
        raise ValueError("pop-size must be divisible by 4 for selTournamentDCD")
    folds = base_mt.build_folds(train_df, fold_days=fold_days, step_days=step_days)
    if not folds:
        raise RuntimeError("No training folds available")

    def evaluate(ind):
        return evaluate_individual(ind, role_psets, folds)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = evaluate(ind)
    pop = tools.selNSGA2(pop, len(pop))
    hof = tools.ParetoFront(similar=lambda a, b: base_mt.stringify_individual(a) == base_mt.stringify_individual(b))
    hof.update(pop)

    start = time.time()
    print(
        f"\nEvolution(MultiTree+LockedTarget): pop={pop_size}, gen={n_gen}, cx={P_CX}, mut={P_MUT}, "
        f"fold_days={fold_days}, step_days={step_days}"
    )
    print(
        f"{'Gen':>4} | {'Front':>5} | {'Ret':>8} | {'DD':>7} | {'Hit':>6} | "
        f"{'Short':>8} | {'Month':>8} | {'Time':>7}"
    )
    print("-" * 86)

    for gen in range(1, n_gen + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for left, right in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                base_mt.mate_individual(left, right, role_psets)
                del left.fitness.values, right.fitness.values

        for mutant in offspring:
            if random.random() < P_MUT:
                base_mt.mutate_individual(mutant, role_psets)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = evaluate(ind)

        pop = tools.selNSGA2(pop + offspring, pop_size)
        hof.update(pop)
        fits = np.asarray([ind.fitness.values for ind in pop], dtype="float64")
        valid = fits[np.all(fits < 1e5, axis=1)]
        stats = valid if len(valid) else fits
        front_size = len(tools.sortNondominated(pop, len(pop), first_front_only=True)[0])
        elapsed = time.time() - start
        print(
            f"{gen:4d} | {front_size:5d} | {-stats[:,0].mean():8.2%} | "
            f"{stats[:,2].mean():6.2%} | {1.0 - stats[:,3].mean():6.2%} | "
            f"{stats[:,1].mean():8.4f} | {stats[:,4].mean():8.4f} | {elapsed:6.1f}s"
        )
    return list(hof)


def select_best_on_validation(pareto, role_psets, val_df: pd.DataFrame):
    best_ind = None
    best_payload = None
    best_score = math.inf

    for rank, ind in enumerate(pareto, start=1):
        try:
            result = evaluate_trees(ind, role_psets, val_df)
            primary = result["primary"]
            score = validation_score(primary)
            if score < best_score:
                best_score = score
                best_ind = ind
                best_payload = {
                    "rank": rank,
                    "validation_score": float(score),
                    "validation_return": float(primary["total_return"]),
                    "validation_max_drawdown": float(primary["max_drawdown"]),
                    "validation_daily_target_hit_rate": float(primary["daily_metrics"]["daily_target_hit_rate"]),
                    "validation_avg_stop_pct": float(result["avg_stop_pct"]),
                    "validation_avg_reward_multiple": float(result["avg_reward_multiple"]),
                    "validation_avg_entry_threshold": float(result["avg_entry_threshold"]),
                }
        except Exception:
            continue
    if best_ind is None or best_payload is None:
        raise RuntimeError("No valid individual survived validation selection")
    return best_ind, best_payload


def backtest_and_print(label: str, individual, role_psets, df_slice: pd.DataFrame) -> dict[str, Any]:
    payload = evaluate_trees(individual, role_psets, df_slice)
    result = payload["primary"]
    print(f"\n=== {label} ===")
    print(f"  Return:       {result['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Trades:       {result['n_trades']}")
    print(f"  Win Rate:     {result['win_rate']*100:.1f}%")
    print(f"  Target Hit:   {result['daily_metrics']['daily_target_hit_rate']*100:.1f}%")
    print(f"  Avg Daily:    {result['daily_metrics']['avg_daily_return']*100:+.2f}%")
    print(f"  Avg Stop:     {payload['avg_stop_pct']*100:.2f}%")
    print(f"  Avg R:        1:{payload['avg_reward_multiple']:.2f}")
    return {**result, **payload}


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    base_mt.ensure_creator_types()
    role_psets = base_mt.build_role_psets()
    toolbox = base_mt.build_toolbox(role_psets)

    print("=" * 90)
    print("  Multi-Tree GP With Adaptive Stop Width And Locked +0.5% Target")
    print("=" * 90)

    print("\n[Phase 1] Data Loading")
    df_all = domain_gp.build_domain_feature_frame(load_all_pairs())
    train_df = df_all.loc[TRAIN_START:domain_gp.TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:domain_gp.VAL_END].copy()
    test_df = df_all.loc[TEST_START:domain_gp.TEST_END].copy()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} bars")

    print("\n[Phase 2] Evolution")
    pareto = run_evolution(
        train_df=train_df,
        toolbox=toolbox,
        role_psets=role_psets,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        fold_days=args.fold_days,
        step_days=args.step_days,
    )
    print(f"  Pareto front size: {len(pareto)}")

    print("\n[Phase 3] Validation Selection")
    best, val_pick = select_best_on_validation(pareto, role_psets, val_df)
    print(
        f"  Selected validation score: {val_pick['validation_score']:.6f} "
        f"(rank #{val_pick['rank']})"
    )

    print("\n[Phase 4] Testing")
    test_result = backtest_and_print("TEST (LockedTarget)", best, role_psets, test_df)
    full_result = backtest_and_print("FULL PERIOD (LockedTarget)", best, role_psets, df_all)

    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)
    with open(model_path, "wb") as f:
        dill.dump(best, f)

    meta = {
        "algorithm": "multi_tree_locked_target_gp",
        "pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "primary_pair": PRIMARY_PAIR,
        "train_period": f"{TRAIN_START} ~ {domain_gp.TRAIN_END}",
        "val_period": f"{VAL_START} ~ {domain_gp.VAL_END}",
        "test_period": f"{TEST_START} ~ {domain_gp.TEST_END}",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "fold_days": args.fold_days,
        "step_days": args.step_days,
        "pareto_front_size": len(pareto),
        "tree_sizes": [len(tree) for tree in best],
        "expressions": base_mt.stringify_individual(best),
        "feature_count": len(domain_gp.domain_feature_names()),
        "locked_daily_target_pct": DAILY_TARGET_PCT,
        "adaptive_stop_range": [MIN_STOP_PCT, MAX_STOP_PCT],
        "validation_pick": val_pick,
        "test_return": float(test_result["total_return"]),
        "test_max_dd": float(test_result["max_drawdown"]),
        "test_win_rate": float(test_result["win_rate"]),
        "test_daily_target_hit_rate": float(test_result["daily_metrics"]["daily_target_hit_rate"]),
        "test_avg_daily_return": float(test_result["daily_metrics"]["avg_daily_return"]),
        "test_avg_stop_pct": float(test_result["avg_stop_pct"]),
        "test_avg_reward_multiple": float(test_result["avg_reward_multiple"]),
        "full_return": float(full_result["total_return"]),
        "full_max_dd": float(full_result["max_drawdown"]),
        "full_win_rate": float(full_result["win_rate"]),
        "full_daily_target_hit_rate": float(full_result["daily_metrics"]["daily_target_hit_rate"]),
        "full_avg_daily_return": float(full_result["daily_metrics"]["avg_daily_return"]),
        "full_avg_stop_pct": float(full_result["avg_stop_pct"]),
        "full_avg_reward_multiple": float(full_result["avg_reward_multiple"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(json_ready(meta), f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
