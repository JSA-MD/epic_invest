#!/usr/bin/env python3
"""Multi-tree GP coevolution for crypto strategy discovery.

This experimental runner evolves three coupled trees per individual:
- participation tree: decides whether to participate
- direction tree: decides long vs short bias
- risk tree: controls reward/risk and signal scaling

The implementation reuses the existing data loaders and backtests from
``gp_crypto_evolution`` but changes the representation and selection loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import functools
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from gp_crypto_evolution import (
    ARG_NAMES,
    DEFAULT_REWARD_MULTIPLE,
    DAILY_TARGET_PCT,
    DAILY_MAX_LOSS_PCT,
    INITIAL_CASH,
    MAX_DEPTH,
    MAX_LEN,
    MODELS_DIR,
    P_CX,
    P_MUT,
    PRIMARY_PAIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    daily_session_backtest,
    get_feature_arrays,
    load_all_pairs,
    robust_rr_backtest,
)


RNG_SEED = 42
DEFAULT_POP_SIZE = 120
DEFAULT_N_GEN = 4
DEFAULT_FOLD_DAYS = 30
DEFAULT_STEP_DAYS = 15

TREE_NAMES = ("participation", "direction", "risk")
PSET: gp.PrimitiveSet | None = None


def pdiv(a, b):
    return np.divide(a, b, out=np.copy(a).astype(float), where=np.abs(b) > 1e-8)


def gt_signal(a, b):
    return np.where(a > b, 100.0, -100.0)


def normalize_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def squash_01(values: np.ndarray, scale: float = 5.0) -> np.ndarray:
    values = normalize_array(values)
    return 0.5 * (np.tanh(values / scale) + 1.0)


def squash_signed(values: np.ndarray, scale: float = 5.0) -> np.ndarray:
    values = normalize_array(values)
    return np.tanh(values / scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-tree GP coevolution for crypto strategy discovery.",
    )
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN)
    parser.add_argument("--fold-days", type=int, default=DEFAULT_FOLD_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_multitree.dill"),
    )
    parser.add_argument(
        "--meta-out",
        default=str(MODELS_DIR / "best_crypto_gp_multitree_meta.json"),
    )
    return parser.parse_args()


def ensure_creator_types() -> None:
    if not hasattr(creator, "FitnessMultiTree"):
        creator.create(
            "FitnessMultiTree",
            base.Fitness,
            weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -0.05),
        )
    if not hasattr(creator, "MultiTreeIndividual"):
        creator.create("MultiTreeIndividual", list, fitness=creator.FitnessMultiTree)


def build_pset() -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("CRYPTO_MULTI", len(ARG_NAMES), prefix="inp")
    for op in (np.add, np.subtract, np.multiply):
        pset.addPrimitive(op, 2)
    pset.addPrimitive(pdiv, 2, name="pdiv")
    pset.addPrimitive(np.sin, 1, name="sin")
    pset.addPrimitive(np.cos, 1, name="cos")
    pset.addPrimitive(np.tanh, 1, name="tanh")
    pset.addPrimitive(gt_signal, 2, name="gt")
    pset.addPrimitive(np.maximum, 2, name="max")
    pset.addPrimitive(np.minimum, 2, name="min")
    pset.addPrimitive(np.abs, 1, name="abs")
    pset.addPrimitive(lambda a: -a, 1, name="neg")
    pset.addEphemeralConstant("rand", functools.partial(random.uniform, -1.0, 1.0))
    for i, name in enumerate(ARG_NAMES):
        pset.renameArguments(**{f"inp{i}": name})
    return pset


def make_head_expr(part_t: gp.PrimitiveSet, role: str):
    if role == "participation":
        return gp.genHalfAndHalf(pset=part_t, min_=1, max_=3)
    if role == "direction":
        return gp.genHalfAndHalf(pset=part_t, min_=1, max_=5)
    if role == "risk":
        return gp.genHalfAndHalf(pset=part_t, min_=1, max_=4)
    raise ValueError(role)


def create_individual(pset: gp.PrimitiveSet) -> creator.MultiTreeIndividual:
    trees = [
        gp.PrimitiveTree(make_head_expr(pset, role))
        for role in TREE_NAMES
    ]
    return creator.MultiTreeIndividual(trees)


def build_toolbox(pset: gp.PrimitiveSet) -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("individual", create_individual, pset)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("clone", copy.deepcopy)
    return toolbox


def build_folds(df_slice: pd.DataFrame, fold_days: int, step_days: int) -> list[pd.DataFrame]:
    if df_slice.empty:
        return []

    days = pd.DatetimeIndex(df_slice.index.normalize().unique())
    if len(days) <= fold_days:
        return [df_slice.copy()]

    folds: list[pd.DataFrame] = []
    for start_idx in range(0, len(days) - fold_days + 1, step_days):
        seg = days[start_idx:start_idx + fold_days]
        fold = df_slice.loc[str(seg[0].date()):str(seg[-1].date())].copy()
        if not fold.empty:
            folds.append(fold)
    return folds


def evaluate_trees(
    individual: creator.MultiTreeIndividual,
    toolbox: base.Toolbox,
    df_slice: pd.DataFrame,
) -> dict[str, Any]:
    part_func = toolbox.compile(expr=individual[0])
    dir_func = toolbox.compile(expr=individual[1])
    risk_func = toolbox.compile(expr=individual[2])

    cols = get_feature_arrays(df_slice, PRIMARY_PAIR)

    part_raw = normalize_array(part_func(*cols))
    dir_raw = normalize_array(dir_func(*cols))
    risk_raw = normalize_array(risk_func(*cols))

    participation = squash_01(part_raw, scale=4.0)
    direction = squash_signed(dir_raw, scale=4.0)
    risk = squash_01(risk_raw, scale=4.0)

    # Participation tree controls whether we are willing to act.
    active = participation >= 0.55
    signal = np.where(active, direction * (0.25 + 0.75 * risk) * 100.0, 0.0)

    risk_summary = float(np.median(risk))
    reward_multiple = float(np.clip(2.0 + 2.0 * risk_summary, 2.0, 4.0))
    entry_threshold = float(np.clip(5.0 + 18.0 * (1.0 - risk_summary), 2.0, 25.0))
    trail_activation_pct = float(np.clip(0.0035 + 0.0045 * risk_summary, 0.003, 0.010))
    trail_distance_pct = float(np.clip(0.0025 + 0.0030 * (1.0 - risk_summary), 0.002, 0.008))
    trail_floor_pct = float(np.clip(0.0015 + 0.0020 * (1.0 - risk_summary), 0.001, 0.006))

    primary = daily_session_backtest(
        df_slice,
        signal,
        pair=PRIMARY_PAIR,
        initial_cash=INITIAL_CASH,
        daily_target_pct=DAILY_TARGET_PCT,
        daily_stop_pct=DAILY_MAX_LOSS_PCT,
        reward_multiple=reward_multiple,
        trail_activation_pct=trail_activation_pct,
        trail_distance_pct=trail_distance_pct,
        trail_floor_pct=trail_floor_pct,
        entry_threshold=entry_threshold,
    )
    robust = robust_rr_backtest(df_slice, signal, PRIMARY_PAIR)

    return {
        "signal": signal,
        "participation": participation,
        "direction": direction,
        "risk": risk,
        "primary": primary,
        "robust": robust,
        "reward_multiple": reward_multiple,
        "entry_threshold": entry_threshold,
        "trail_activation_pct": trail_activation_pct,
        "trail_distance_pct": trail_distance_pct,
        "trail_floor_pct": trail_floor_pct,
    }


def evaluate_fold(
    individual: creator.MultiTreeIndividual,
    toolbox: base.Toolbox,
    df_slice: pd.DataFrame,
) -> tuple[float, float, float, float, float, float]:
    try:
        result = evaluate_trees(individual, toolbox, df_slice)
        primary = result["primary"]
        robust = result["robust"]
        daily = primary["daily_metrics"]
        monthly = primary["monthly_metrics"]
        complexity = float(sum(len(tree) for tree in individual) / (3.0 * MAX_LEN))

        if len(daily["daily_returns"]) < 20 or primary["n_trades"] < 3:
            return (1e6, 1e6, 1e6, 1e6, 1e6, 1e3)
        if primary["max_drawdown"] < -0.35 or daily["worst_day"] <= -0.05:
            return (1e6, 1e6, 1e6, 1e6, 1e6, 1e3)

        return (
            float(-primary["total_return"]),
            float(daily["daily_shortfall_mean"]),
            float(abs(primary["max_drawdown"])),
            float(1.0 - daily["daily_target_hit_rate"]),
            float(robust["aggregate"]["monthly_shortfall_sum"]),
            complexity,
        )
    except Exception:
        return (1e6, 1e6, 1e6, 1e6, 1e6, 1e3)


def aggregate_objectives(objectives: list[tuple[float, float, float, float, float, float]]) -> tuple[float, ...]:
    arr = np.asarray(objectives, dtype="float64")
    return tuple(np.mean(arr, axis=0).tolist())


def evaluate_individual(
    individual: creator.MultiTreeIndividual,
    toolbox: base.Toolbox,
    folds: list[pd.DataFrame],
) -> tuple[float, float, float, float, float, float]:
    fold_objectives = [evaluate_fold(individual, toolbox, fold) for fold in folds]
    return aggregate_objectives(fold_objectives)


def validation_score(metrics: dict[str, Any]) -> float:
    daily = metrics["daily_metrics"]
    monthly = metrics["monthly_metrics"]
    return (
        metrics["total_return"] * -120.0
        + abs(metrics["max_drawdown"]) * 60.0
        + (1.0 - daily["daily_target_hit_rate"]) * 40.0
        + daily["daily_shortfall_mean"] * 100.0
        + monthly["monthly_shortfall_mean"] * 70.0
        - metrics["win_rate"] * 8.0
    )


def select_best_on_validation(
    population: list[creator.MultiTreeIndividual],
    toolbox: base.Toolbox,
    val_df: pd.DataFrame,
) -> tuple[creator.MultiTreeIndividual, dict[str, Any]]:
    best_ind = None
    best_payload = None
    best_score = math.inf

    for rank, ind in enumerate(population, start=1):
        try:
            result = evaluate_trees(ind, toolbox, val_df)
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
                    "validation_reward_multiple": float(result["reward_multiple"]),
                    "validation_entry_threshold": float(result["entry_threshold"]),
                }
        except Exception:
            continue

    if best_ind is None or best_payload is None:
        raise RuntimeError("No valid individual survived validation selection")
    return best_ind, best_payload


def stringify_individual(individual: creator.MultiTreeIndividual) -> list[str]:
    return [str(tree) for tree in individual]


def run_evolution(
    train_df: pd.DataFrame,
    toolbox: base.Toolbox,
    pop_size: int,
    n_gen: int,
    fold_days: int,
    step_days: int,
) -> list[creator.MultiTreeIndividual]:
    folds = build_folds(train_df, fold_days=fold_days, step_days=step_days)
    if not folds:
        raise RuntimeError("No training folds available")

    def eval_for_toolbox(ind):
        return evaluate_individual(ind, toolbox, folds)

    toolbox.register("evaluate", eval_for_toolbox)
    toolbox.register("mate", mate_individual)
    toolbox.register("mutate", mutate_individual)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, len(pop))

    hof = tools.ParetoFront(similar=lambda a, b: stringify_individual(a) == stringify_individual(b))
    hof.update(pop)

    start = time.time()
    print(
        f"\nEvolution(MultiTree): pop={pop_size}, gen={n_gen}, cx={P_CX}, mut={P_MUT}, "
        f"fold_days={fold_days}, step_days={step_days}"
    )
    print(
        f"{'Gen':>4} | {'Front':>5} | {'Ret':>8} | {'DD':>7} | {'Hit':>6} | "
        f"{'Short':>8} | {'Month':>8} | {'Time':>7}"
    )
    print("-" * 78)

    for gen in range(1, n_gen + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for left, right in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                mate_individual(left, right)
                del left.fitness.values, right.fitness.values

        for mutant in offspring:
            if random.random() < P_MUT:
                mutate_individual(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, pop_size)
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


def mate_individual(left: creator.MultiTreeIndividual, right: creator.MultiTreeIndividual) -> tuple[creator.MultiTreeIndividual, creator.MultiTreeIndividual]:
    if PSET is None:
        raise RuntimeError("PSET not initialized")
    for idx in range(3):
        if random.random() < 0.75:
            gp.cxOnePoint(left[idx], right[idx])
            if len(left[idx]) > MAX_LEN:
                left[idx] = gp.PrimitiveTree(make_head_expr(PSET, TREE_NAMES[idx]))
            if len(right[idx]) > MAX_LEN:
                right[idx] = gp.PrimitiveTree(make_head_expr(PSET, TREE_NAMES[idx]))
    return left, right


def mutate_individual(individual: creator.MultiTreeIndividual) -> tuple[creator.MultiTreeIndividual]:
    if PSET is None:
        raise RuntimeError("PSET not initialized")
    mutated = False
    for idx in range(3):
        if random.random() < 0.35:
            new_tree = gp.PrimitiveTree(gp.genFull(pset=PSET, min_=0, max_=2))
            individual[idx] = new_tree
            mutated = True
    if not mutated:
        idx = random.randrange(3)
        individual[idx] = gp.PrimitiveTree(gp.genGrow(pset=PSET, min_=0, max_=3))
    return (individual,)


def backtest_and_print(label: str, individual: creator.MultiTreeIndividual, toolbox: base.Toolbox, df_slice: pd.DataFrame) -> dict[str, Any]:
    result = evaluate_trees(individual, toolbox, df_slice)["primary"]
    print(f"\n=== {label} ===")
    print(f"  Return:       {result['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Trades:       {result['n_trades']}")
    print(f"  Win Rate:     {result['win_rate']*100:.1f}%")
    print(f"  Target Hit:   {result['daily_metrics']['daily_target_hit_rate']*100:.1f}%")
    print(f"  Avg Month:    {result['monthly_metrics']['avg_monthly_return']*100:+.2f}%")
    print(f"  Worst Month:  {result['monthly_metrics']['worst_month']*100:.2f}%")
    return result


def main() -> None:
    args = parse_args()
    if args.pop_size % 4 != 0:
        raise ValueError("pop-size must be divisible by 4 for selTournamentDCD")

    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    ensure_creator_types()
    global PSET
    PSET = build_pset()
    toolbox = build_toolbox(PSET)

    print("=" * 78)
    print("  Multi-Tree GP Coevolution")
    print("=" * 78)

    print("\n[Phase 1] Data Loading")
    df_all = load_all_pairs()
    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:VAL_END].copy()
    test_df = df_all.loc[TEST_START:TEST_END].copy()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} bars")

    print("\n[Phase 2] Evolution")
    pareto = run_evolution(
        train_df=train_df,
        toolbox=toolbox,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        fold_days=args.fold_days,
        step_days=args.step_days,
    )
    print(f"  Pareto front size: {len(pareto)}")

    print("\n[Phase 3] Validation Selection")
    best, val_pick = select_best_on_validation(pareto, toolbox, val_df)
    print(
        f"  Selected validation score: {val_pick['validation_score']:.6f} "
        f"(rank #{val_pick['rank']})"
    )

    print("\n[Phase 4] Testing")
    test_result = backtest_and_print("TEST (MultiTree)", best, toolbox, test_df)
    full_result = backtest_and_print("FULL PERIOD (MultiTree)", best, toolbox, df_all)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)

    with open(model_path, "wb") as f:
        dill.dump(best, f)

    meta = {
        "algorithm": "multi_tree_coevolution_gp",
        "pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "primary_pair": PRIMARY_PAIR,
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period": f"{VAL_START} ~ {VAL_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "fold_days": args.fold_days,
        "step_days": args.step_days,
        "pareto_front_size": len(pareto),
        "tree_sizes": [len(tree) for tree in best],
        "expressions": stringify_individual(best),
        "validation_pick": val_pick,
        "test_return": float(test_result["total_return"]),
        "test_max_dd": float(test_result["max_drawdown"]),
        "test_daily_target_hit_rate": float(test_result["daily_metrics"]["daily_target_hit_rate"]),
        "full_return": float(full_result["total_return"]),
        "full_max_dd": float(full_result["max_drawdown"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
