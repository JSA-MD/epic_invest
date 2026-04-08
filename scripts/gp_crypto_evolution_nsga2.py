#!/usr/bin/env python3
"""Multi-objective GP evolution using walk-forward folds and NSGA-II."""

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
from deap import base, creator, gp, tools

from gp_crypto_evolution import (
    DAILY_TARGET_PCT,
    MAX_DEPTH,
    MAX_LEN,
    MODELS_DIR,
    N_GEN,
    P_CX,
    P_MUT,
    POP_SIZE,
    PRIMARY_PAIR,
    TEST_START,
    TRAIN_START,
    VAL_START,
    backtest_on_slice,
    get_feature_arrays,
    load_all_pairs,
    pset,
    robust_rr_backtest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-objective GP evolution with walk-forward folds and NSGA-II.",
    )
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--n-gen", type=int, default=N_GEN)
    parser.add_argument("--fold-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=15)
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_nsga2.dill"),
    )
    parser.add_argument(
        "--meta-out",
        default=str(MODELS_DIR / "best_crypto_gp_nsga2_meta.json"),
    )
    return parser.parse_args()


def ensure_creator_types() -> None:
    if not hasattr(creator, "FitnessNSGA2"):
        creator.create(
            "FitnessNSGA2",
            base.Fitness,
            weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -0.05),
        )
    if not hasattr(creator, "IndividualNSGA2"):
        creator.create(
            "IndividualNSGA2",
            gp.PrimitiveTree,
            fitness=creator.FitnessNSGA2,
        )


def build_toolbox() -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.IndividualNSGA2, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_LEN))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_LEN))
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def build_walkforward_folds(
    df_slice: pd.DataFrame,
    fold_days: int = 30,
    step_days: int = 15,
) -> list[pd.DataFrame]:
    days = pd.DatetimeIndex(df_slice.index.normalize().unique())
    if len(days) == 0:
        return []
    if len(days) <= fold_days:
        return [df_slice.copy()]

    folds: list[pd.DataFrame] = []
    for start_idx in range(0, len(days) - fold_days + 1, step_days):
        seg = days[start_idx:start_idx + fold_days]
        start_day = str(seg[0].date())
        end_day = str(seg[-1].date())
        fold = df_slice.loc[start_day:end_day].copy()
        if not fold.empty:
            folds.append(fold)
    return folds


def make_multiobjective_tuple(
    fold_metrics: list[dict[str, Any]],
    tree_size: int,
) -> tuple[float, float, float, float, float, float]:
    daily_shortfall = float(np.mean([
        m["daily_metrics"]["daily_shortfall_mean"] for m in fold_metrics
    ]))
    target_miss = float(1.0 - np.mean([
        m["daily_metrics"]["daily_target_hit_rate"] for m in fold_metrics
    ]))
    tail_risk = float(np.mean([
        max(0.0, -m["daily_metrics"]["cvar"]) + max(0.0, -m["daily_metrics"]["worst_day"])
        for m in fold_metrics
    ]))
    drawdown_risk = float(np.mean([
        abs(m["max_drawdown"]) for m in fold_metrics
    ]))
    avg_daily_returns = np.asarray(
        [m["daily_metrics"]["avg_daily_return"] for m in fold_metrics],
        dtype="float64",
    )
    instability = float(
        np.std(avg_daily_returns)
        + max(0.0, DAILY_TARGET_PCT - float(np.min(avg_daily_returns)))
    )
    complexity = float(tree_size / max(MAX_LEN, 1))
    return (
        daily_shortfall,
        target_miss,
        tail_risk,
        drawdown_risk,
        instability,
        complexity,
    )


def evaluate_fold_result(
    result: dict[str, Any],
) -> bool:
    daily = result["daily_metrics"]
    return (
        len(daily["daily_returns"]) >= 20
        and result["n_trades"] >= 3
        and result["max_drawdown"] > -0.35
        and daily["worst_day"] > -0.05
    )


def evaluate_individual_multi(
    ind: gp.PrimitiveTree,
    toolbox: base.Toolbox,
    folds: list[pd.DataFrame],
) -> tuple[float, float, float, float, float, float]:
    try:
        func = toolbox.compile(expr=ind)
        fold_metrics = []
        for fold in folds:
            cols = get_feature_arrays(fold, PRIMARY_PAIR)
            desired_pcts = func(*cols)
            result = robust_rr_backtest(fold, desired_pcts, PRIMARY_PAIR)["primary"]
            if not evaluate_fold_result(result):
                return (1e6, 1e6, 1e6, 1e6, 1e6, 1e3)
            fold_metrics.append(result)
        return make_multiobjective_tuple(fold_metrics, len(ind))
    except Exception:
        return (1e6, 1e6, 1e6, 1e6, 1e6, 1e3)


def _evaluate_with_folds(ind: gp.PrimitiveTree) -> tuple[float, float, float, float, float, float]:
    return evaluate_individual_multi(ind, _evaluate_with_folds.toolbox, _evaluate_with_folds.folds)


def objective_summary(population: list[gp.PrimitiveTree]) -> dict[str, float]:
    fits = np.asarray([ind.fitness.values for ind in population], dtype="float64")
    fronts = tools.sortNondominated(population, len(population), first_front_only=True)
    front_size = len(fronts[0]) if fronts else 0
    return {
        "front_size": float(front_size),
        "shortfall_min": float(np.min(fits[:, 0])),
        "target_miss_min": float(np.min(fits[:, 1])),
        "tail_risk_min": float(np.min(fits[:, 2])),
        "drawdown_min": float(np.min(fits[:, 3])),
        "instability_min": float(np.min(fits[:, 4])),
    }


def weighted_validation_score(
    objectives: tuple[float, float, float, float, float, float],
    total_return: float,
) -> float:
    shortfall, target_miss, tail_risk, drawdown_risk, instability, complexity = objectives
    return (
        shortfall * 180.0
        + target_miss * 14.0
        + tail_risk * 40.0
        + drawdown_risk * 30.0
        + instability * 120.0
        + complexity * 0.25
        - total_return * 6.0
    )


def select_best_on_validation_nsga2(
    pareto_front: tools.ParetoFront,
    toolbox: base.Toolbox,
    val_df: pd.DataFrame,
) -> tuple[gp.PrimitiveTree, dict[str, Any]]:
    folds = build_walkforward_folds(val_df, fold_days=30, step_days=15)
    if not folds:
        folds = [val_df]

    best_ind = None
    best_payload = None
    best_score = math.inf

    for idx, ind in enumerate(pareto_front, start=1):
        objectives = evaluate_individual_multi(ind, toolbox, folds)
        if objectives[0] >= 1e6:
            continue
        result = robust_rr_backtest(
            val_df,
            toolbox.compile(expr=ind)(*get_feature_arrays(val_df, PRIMARY_PAIR)),
            PRIMARY_PAIR,
        )["primary"]
        score = weighted_validation_score(objectives, result["total_return"])
        if score < best_score:
            best_score = score
            best_ind = ind
            best_payload = {
                "rank": idx,
                "validation_objectives": objectives,
                "validation_score": float(score),
                "validation_return": float(result["total_return"]),
                "validation_max_drawdown": float(result["max_drawdown"]),
                "validation_daily_target_hit_rate": float(result["daily_metrics"]["daily_target_hit_rate"]),
            }

    if best_ind is None or best_payload is None:
        raise RuntimeError("No valid individual survived validation selection")
    return best_ind, best_payload


def run_evolution_nsga2(
    train_df: pd.DataFrame,
    toolbox: base.Toolbox,
    pop_size: int,
    n_gen: int,
    fold_days: int,
    step_days: int,
) -> tools.ParetoFront:
    folds = build_walkforward_folds(train_df, fold_days=fold_days, step_days=step_days)
    if not folds:
        raise RuntimeError("No training folds available for NSGA-II evolution")

    _evaluate_with_folds.toolbox = toolbox
    _evaluate_with_folds.folds = folds
    toolbox.register("evaluate", _evaluate_with_folds)

    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront(similar=lambda a, b: a == b)

    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, len(pop))
    hof.update(pop)

    print(f"\nEvolution(NSGA-II): pop={pop_size}, gen={n_gen}, cx={P_CX}, mut={P_MUT}")
    print(
        f"{'Gen':>4} | {'Front':>5} | {'Shortfall':>10} | {'HitMiss':>8} | "
        f"{'Tail':>8} | {'DD':>8} | {'Instab':>8} | {'Time':>8}"
    )
    print("-" * 86)

    start_ts = time.time()
    for gen in range(1, n_gen + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUT:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, pop_size)
        hof.update(pop)
        summary = objective_summary(pop)
        elapsed = time.time() - start_ts
        print(
            f"{gen:4d} | {summary['front_size']:5.0f} | {summary['shortfall_min']:10.6f} | "
            f"{summary['target_miss_min']:8.4f} | {summary['tail_risk_min']:8.4f} | "
            f"{summary['drawdown_min']:8.4f} | {summary['instability_min']:8.4f} | "
            f"{elapsed:7.1f}s"
        )

    return hof


def main() -> None:
    args = parse_args()
    if args.pop_size % 4 != 0:
        raise ValueError("pop-size must be divisible by 4 for selTournamentDCD")
    ensure_creator_types()
    toolbox = build_toolbox()

    print("=" * 78)
    print("  Multi-Objective GP Crypto Strategy (NSGA-II)")
    print("=" * 78)

    print("\n[Phase 1] Data Loading")
    df_all = load_all_pairs()
    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:VAL_END].copy()
    test_df = df_all.loc[TEST_START:TEST_END].copy()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} bars")

    print("\n[Phase 2] NSGA-II Evolution")
    pareto_front = run_evolution_nsga2(
        train_df=train_df,
        toolbox=toolbox,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        fold_days=args.fold_days,
        step_days=args.step_days,
    )
    print(f"  Pareto front size: {len(pareto_front)}")

    print("\n[Phase 3] Validation Selection")
    best, val_pick = select_best_on_validation_nsga2(pareto_front, toolbox, val_df)
    print(
        f"  Selected validation score: {val_pick['validation_score']:.6f} "
        f"(rank #{val_pick['rank']} on front)"
    )

    print("\n[Phase 4] Out-of-Sample Test")
    test_result = backtest_on_slice(best, test_df, "TEST (NSGA-II)")

    print("\n[Phase 5] Full-Period Backtest")
    full_result = backtest_on_slice(best, df_all, "FULL PERIOD (NSGA-II)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)

    with open(model_path, "wb") as f:
        dill.dump(gp.PrimitiveTree(best), f)

    meta = {
        "algorithm": "nsga2_walkforward_gp",
        "pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "primary_pair": PRIMARY_PAIR,
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period": f"{VAL_START} ~ {VAL_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "fold_days": args.fold_days,
        "step_days": args.step_days,
        "pareto_front_size": len(pareto_front),
        "tree_size": len(best),
        "fitness_vector": list(best.fitness.values),
        "validation_pick": val_pick,
        "test_return": float(test_result["total_return"]),
        "test_sharpe": float(test_result["sharpe"]),
        "test_max_dd": float(test_result["max_drawdown"]),
        "test_daily_win_rate": float(test_result["daily_metrics"]["daily_win_rate"]),
        "test_daily_target_hit_rate": float(test_result["daily_metrics"]["daily_target_hit_rate"]),
        "full_return": float(full_result["total_return"]),
        "full_max_dd": float(full_result["max_drawdown"]),
        "expression": str(best),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
