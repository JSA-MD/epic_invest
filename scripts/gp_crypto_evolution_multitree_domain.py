#!/usr/bin/env python3
"""Multi-tree coevolution with domain-specific typed primitives."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

import gp_crypto_evolution_domain_gp as domain_gp
from gp_crypto_evolution import (
    DAILY_MAX_LOSS_PCT,
    DAILY_TARGET_PCT,
    INITIAL_CASH,
    MAX_DEPTH,
    MAX_LEN,
    MODELS_DIR,
    P_CX,
    P_MUT,
    PRIMARY_PAIR,
    TEST_START,
    TRAIN_START,
    VAL_START,
    daily_session_backtest,
    robust_rr_backtest,
    load_all_pairs,
)


RNG_SEED = 42
DEFAULT_POP_SIZE = 24
DEFAULT_N_GEN = 3
DEFAULT_FOLD_DAYS = 30
DEFAULT_STEP_DAYS = 15
TREE_ROLES = ("participation", "direction", "risk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-tree GP with domain-specific typed primitives.",
    )
    parser.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    parser.add_argument("--n-gen", type=int, default=DEFAULT_N_GEN)
    parser.add_argument("--fold-days", type=int, default=DEFAULT_FOLD_DAYS)
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_multitree_domain.dill"),
    )
    parser.add_argument(
        "--meta-out",
        default=str(MODELS_DIR / "best_crypto_gp_multitree_domain_meta.json"),
    )
    return parser.parse_args()


def ensure_creator_types() -> None:
    if not hasattr(creator, "FitnessMultiTreeDomain"):
        creator.create(
            "FitnessMultiTreeDomain",
            base.Fitness,
            weights=(-1.0, -1.0, -1.0, -1.0, -1.0, -0.05),
        )
    if not hasattr(creator, "MultiTreeDomainIndividual"):
        creator.create(
            "MultiTreeDomainIndividual",
            list,
            fitness=creator.FitnessMultiTreeDomain,
        )


def _rename_inputs(pset: gp.PrimitiveSetTyped) -> gp.PrimitiveSetTyped:
    for i, name in enumerate(domain_gp.domain_feature_names()):
        pset.renameArguments(**{f"ARG{i}": name})
    return pset


def setup_signal_pset() -> gp.PrimitiveSetTyped:
    inputs = [domain_gp.Signal] * len(domain_gp.domain_feature_names())
    pset = gp.PrimitiveSetTyped("MTD_SIGNAL", inputs, domain_gp.Signal)

    for op in (domain_gp.sig_add, domain_gp.sig_sub, domain_gp.sig_mul, domain_gp.sig_safe_div):
        pset.addPrimitive(op, [domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_neg, [domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_abs, [domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_tanh, [domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_spread, [domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_vol_scale, [domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_rank2, [domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_blend, [domain_gp.Gate, domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_from_gate, [domain_gp.Gate, float], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_scale, [domain_gp.Signal, float], domain_gp.Signal)

    pset.addPrimitive(domain_gp.gate_gt, [domain_gp.Signal, domain_gp.Signal], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_lt, [domain_gp.Signal, domain_gp.Signal], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_between, [domain_gp.Signal, float, float], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_and, [domain_gp.Gate, domain_gp.Gate], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_or, [domain_gp.Gate, domain_gp.Gate], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_not, [domain_gp.Gate], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_regime, [domain_gp.Signal, domain_gp.Signal, float], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_strength, [domain_gp.Signal, domain_gp.Signal, float], domain_gp.Gate)

    for op in (domain_gp.f_add, domain_gp.f_sub, domain_gp.f_mul, domain_gp.f_div):
        pset.addPrimitive(op, [float, float], float)
    pset.addPrimitive(domain_gp.f_neg, [float], float)
    pset.addPrimitive(domain_gp.f_abs, [float], float)
    pset.addPrimitive(domain_gp.f_tanh, [float], float)

    pset.addTerminal(0.0, domain_gp.Gate)
    pset.addTerminal(1.0, domain_gp.Gate)
    pset.addEphemeralConstant("rand_scale", functools.partial(random.uniform, 2.0, 40.0), float)
    pset.addEphemeralConstant("rand_threshold", functools.partial(random.uniform, 0.0, 1.5), float)
    return _rename_inputs(pset)


def setup_gate_pset() -> gp.PrimitiveSetTyped:
    inputs = [domain_gp.Signal] * len(domain_gp.domain_feature_names())
    pset = gp.PrimitiveSetTyped("MTD_GATE", inputs, domain_gp.Gate)

    pset.addPrimitive(domain_gp.gate_gt, [domain_gp.Signal, domain_gp.Signal], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_lt, [domain_gp.Signal, domain_gp.Signal], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_between, [domain_gp.Signal, float, float], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_and, [domain_gp.Gate, domain_gp.Gate], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_or, [domain_gp.Gate, domain_gp.Gate], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_not, [domain_gp.Gate], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_regime, [domain_gp.Signal, domain_gp.Signal, float], domain_gp.Gate)
    pset.addPrimitive(domain_gp.gate_strength, [domain_gp.Signal, domain_gp.Signal, float], domain_gp.Gate)

    for op in (domain_gp.sig_add, domain_gp.sig_sub, domain_gp.sig_mul, domain_gp.sig_safe_div):
        pset.addPrimitive(op, [domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_abs, [domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_tanh, [domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_spread, [domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_vol_scale, [domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)
    pset.addPrimitive(domain_gp.sig_rank2, [domain_gp.Signal, domain_gp.Signal], domain_gp.Signal)

    for op in (domain_gp.f_add, domain_gp.f_sub, domain_gp.f_mul, domain_gp.f_div):
        pset.addPrimitive(op, [float, float], float)
    pset.addPrimitive(domain_gp.f_neg, [float], float)
    pset.addPrimitive(domain_gp.f_abs, [float], float)
    pset.addPrimitive(domain_gp.f_tanh, [float], float)
    pset.addTerminal(0.0, domain_gp.Gate)
    pset.addTerminal(1.0, domain_gp.Gate)
    pset.addEphemeralConstant("rand_threshold", functools.partial(random.uniform, 0.0, 1.5), float)
    return _rename_inputs(pset)


def build_role_psets() -> dict[str, gp.PrimitiveSetTyped]:
    return {
        "participation": setup_gate_pset(),
        "direction": setup_signal_pset(),
        "risk": setup_signal_pset(),
    }


def create_individual(role_psets: dict[str, gp.PrimitiveSetTyped]) -> creator.MultiTreeDomainIndividual:
    specs = {
        "participation": (1, 3),
        "direction": (1, 5),
        "risk": (1, 4),
    }
    trees = []
    for role in TREE_ROLES:
        min_depth, max_depth = specs[role]
        expr = gp.genHalfAndHalf(pset=role_psets[role], min_=min_depth, max_=max_depth)
        trees.append(gp.PrimitiveTree(expr))
    return creator.MultiTreeDomainIndividual(trees)


def build_toolbox(role_psets: dict[str, gp.PrimitiveSetTyped]) -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, role_psets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("clone", copy.deepcopy)
    return toolbox


def build_folds(df_slice: pd.DataFrame, fold_days: int, step_days: int) -> list[pd.DataFrame]:
    days = pd.DatetimeIndex(df_slice.index.normalize().unique())
    if len(days) == 0:
        return []
    if len(days) <= fold_days:
        return [df_slice.copy()]
    folds = []
    for start_idx in range(0, len(days) - fold_days + 1, step_days):
        seg = days[start_idx:start_idx + fold_days]
        fold = df_slice.loc[str(seg[0].date()):str(seg[-1].date())].copy()
        if not fold.empty:
            folds.append(fold)
    return folds


def compile_heads(
    individual: creator.MultiTreeDomainIndividual,
    role_psets: dict[str, gp.PrimitiveSetTyped],
) -> dict[str, Any]:
    return {
        role: gp.compile(expr=tree, pset=role_psets[role])
        for role, tree in zip(TREE_ROLES, individual)
    }


def evaluate_trees(
    individual: creator.MultiTreeDomainIndividual,
    role_psets: dict[str, gp.PrimitiveSetTyped],
    df_slice: pd.DataFrame,
) -> dict[str, Any]:
    funcs = compile_heads(individual, role_psets)
    inputs = domain_gp.build_domain_inputs(df_slice)

    part_raw = np.asarray(funcs["participation"](*inputs), dtype="float64")
    dir_raw = np.asarray(funcs["direction"](*inputs), dtype="float64")
    risk_raw = np.asarray(funcs["risk"](*inputs), dtype="float64")

    part_raw = np.where(np.isfinite(part_raw), part_raw, 0.0)
    dir_raw = np.where(np.isfinite(dir_raw), dir_raw, 0.0)
    risk_raw = np.where(np.isfinite(risk_raw), risk_raw, 0.0)

    participation = np.where(part_raw > 0.5, 1.0, 0.0)
    # The direction head's typed primitives tend to express "bullishness deficit".
    # Invert the signed output so positive signal magnitude maps to long bias.
    direction = np.clip(-np.tanh(dir_raw), -1.0, 1.0)
    risk = 0.5 * (np.tanh(risk_raw / 5.0) + 1.0)
    risk = np.clip(risk, 0.0, 1.0)

    signal = np.where(participation > 0.5, direction * (0.20 + 0.80 * risk) * 100.0, 0.0)
    risk_summary = float(np.median(risk))
    reward_multiple = float(np.clip(2.0 + 2.0 * risk_summary, 2.0, 4.0))
    entry_threshold = float(np.clip(4.0 + 16.0 * (1.0 - risk_summary), 2.0, 20.0))
    trail_activation_pct = float(np.clip(0.0035 + 0.0035 * risk_summary, 0.003, 0.009))
    trail_distance_pct = float(np.clip(0.0020 + 0.0030 * (1.0 - risk_summary), 0.002, 0.008))
    trail_floor_pct = float(np.clip(0.0015 + 0.0020 * (1.0 - risk_summary), 0.001, 0.005))

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
        "primary": primary,
        "robust": robust,
        "participation": participation,
        "direction": direction,
        "risk": risk,
        "signal": signal,
        "reward_multiple": reward_multiple,
        "entry_threshold": entry_threshold,
    }


def evaluate_fold(
    individual: creator.MultiTreeDomainIndividual,
    role_psets: dict[str, gp.PrimitiveSetTyped],
    df_slice: pd.DataFrame,
) -> tuple[float, float, float, float, float, float]:
    try:
        result = evaluate_trees(individual, role_psets, df_slice)
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


def evaluate_individual(
    individual: creator.MultiTreeDomainIndividual,
    role_psets: dict[str, gp.PrimitiveSetTyped],
    folds: list[pd.DataFrame],
) -> tuple[float, float, float, float, float, float]:
    arr = np.asarray([evaluate_fold(individual, role_psets, fold) for fold in folds], dtype="float64")
    return tuple(np.mean(arr, axis=0).tolist())


def validation_score(metrics: dict[str, Any]) -> float:
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


def stringify_individual(individual: creator.MultiTreeDomainIndividual) -> list[str]:
    return [str(tree) for tree in individual]


def mate_individual(
    left: creator.MultiTreeDomainIndividual,
    right: creator.MultiTreeDomainIndividual,
    role_psets: dict[str, gp.PrimitiveSetTyped],
) -> tuple[creator.MultiTreeDomainIndividual, creator.MultiTreeDomainIndividual]:
    for idx, role in enumerate(TREE_ROLES):
        if random.random() < 0.75:
            gp.cxOnePoint(left[idx], right[idx])
            if len(left[idx]) > MAX_LEN:
                left[idx] = gp.PrimitiveTree(gp.genHalfAndHalf(pset=role_psets[role], min_=1, max_=3))
            if len(right[idx]) > MAX_LEN:
                right[idx] = gp.PrimitiveTree(gp.genHalfAndHalf(pset=role_psets[role], min_=1, max_=3))
    return left, right


def mutate_individual(
    individual: creator.MultiTreeDomainIndividual,
    role_psets: dict[str, gp.PrimitiveSetTyped],
) -> tuple[creator.MultiTreeDomainIndividual]:
    mutated = False
    for idx, role in enumerate(TREE_ROLES):
        if random.random() < 0.35:
            individual[idx] = gp.PrimitiveTree(gp.genGrow(pset=role_psets[role], min_=0, max_=3))
            mutated = True
    if not mutated:
        idx = random.randrange(3)
        role = TREE_ROLES[idx]
        individual[idx] = gp.PrimitiveTree(gp.genFull(pset=role_psets[role], min_=0, max_=2))
    return (individual,)


def run_evolution(
    train_df: pd.DataFrame,
    toolbox: base.Toolbox,
    role_psets: dict[str, gp.PrimitiveSetTyped],
    pop_size: int,
    n_gen: int,
    fold_days: int,
    step_days: int,
) -> list[creator.MultiTreeDomainIndividual]:
    if pop_size % 4 != 0:
        raise ValueError("pop-size must be divisible by 4 for selTournamentDCD")
    folds = build_folds(train_df, fold_days=fold_days, step_days=step_days)
    if not folds:
        raise RuntimeError("No training folds available")

    def evaluate(ind):
        return evaluate_individual(ind, role_psets, folds)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = evaluate(ind)
    pop = tools.selNSGA2(pop, len(pop))
    hof = tools.ParetoFront(similar=lambda a, b: stringify_individual(a) == stringify_individual(b))
    hof.update(pop)

    start = time.time()
    print(
        f"\nEvolution(MultiTree+Domain): pop={pop_size}, gen={n_gen}, cx={P_CX}, mut={P_MUT}, "
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
                mate_individual(left, right, role_psets)
                del left.fitness.values, right.fitness.values

        for mutant in offspring:
            if random.random() < P_MUT:
                mutate_individual(mutant, role_psets)
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


def select_best_on_validation(
    pareto: list[creator.MultiTreeDomainIndividual],
    role_psets: dict[str, gp.PrimitiveSetTyped],
    val_df: pd.DataFrame,
) -> tuple[creator.MultiTreeDomainIndividual, dict[str, Any]]:
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
                    "validation_reward_multiple": float(result["reward_multiple"]),
                    "validation_entry_threshold": float(result["entry_threshold"]),
                }
        except Exception:
            continue
    if best_ind is None or best_payload is None:
        raise RuntimeError("No valid individual survived validation selection")
    return best_ind, best_payload


def backtest_and_print(
    label: str,
    individual: creator.MultiTreeDomainIndividual,
    role_psets: dict[str, gp.PrimitiveSetTyped],
    df_slice: pd.DataFrame,
) -> dict[str, Any]:
    result = evaluate_trees(individual, role_psets, df_slice)["primary"]
    print(f"\n=== {label} ===")
    print(f"  Return:       {result['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Trades:       {result['n_trades']}")
    print(f"  Win Rate:     {result['win_rate']*100:.1f}%")
    print(f"  Target Hit:   {result['daily_metrics']['daily_target_hit_rate']*100:.1f}%")
    print(f"  Avg Daily:    {result['daily_metrics']['avg_daily_return']*100:+.2f}%")
    print(f"  Avg Month:    {result['monthly_metrics']['avg_monthly_return']*100:+.2f}%")
    return result


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    ensure_creator_types()
    role_psets = build_role_psets()
    toolbox = build_toolbox(role_psets)

    print("=" * 86)
    print("  Multi-Tree + Domain-Primitive GP")
    print("=" * 86)

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
    test_result = backtest_and_print("TEST (MultiTree+Domain)", best, role_psets, test_df)
    full_result = backtest_and_print("FULL PERIOD (MultiTree+Domain)", best, role_psets, df_all)

    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)
    with open(model_path, "wb") as f:
        dill.dump(best, f)

    meta = {
        "algorithm": "multi_tree_domain_gp",
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
        "expressions": stringify_individual(best),
        "feature_count": len(domain_gp.domain_feature_names()),
        "validation_pick": val_pick,
        "test_return": float(test_result["total_return"]),
        "test_max_dd": float(test_result["max_drawdown"]),
        "test_win_rate": float(test_result["win_rate"]),
        "test_daily_target_hit_rate": float(test_result["daily_metrics"]["daily_target_hit_rate"]),
        "test_avg_daily_return": float(test_result["daily_metrics"]["avg_daily_return"]),
        "test_monthly_target_hit_rate": float(test_result["monthly_metrics"]["monthly_target_hit_rate"]),
        "full_return": float(full_result["total_return"]),
        "full_max_dd": float(full_result["max_drawdown"]),
        "full_win_rate": float(full_result["win_rate"]),
        "full_daily_target_hit_rate": float(full_result["daily_metrics"]["daily_target_hit_rate"]),
        "full_avg_daily_return": float(full_result["daily_metrics"]["avg_daily_return"]),
        "full_monthly_target_hit_rate": float(full_result["monthly_metrics"]["monthly_target_hit_rate"]),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(domain_gp.json_ready(meta), f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
