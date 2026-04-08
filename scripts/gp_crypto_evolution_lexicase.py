#!/usr/bin/env python3
"""Lexicase / case-wise GP evolution for crypto signal discovery."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools

from gp_crypto_evolution import (
    ARG_NAMES,
    DAILY_TARGET_PCT,
    DEFAULT_REWARD_MULTIPLE,
    INITIAL_CASH,
    MAX_DEPTH,
    MAX_LEN,
    MODELS_DIR,
    N_GEN,
    P_CX,
    P_MUT,
    POP_SIZE,
    PRIMARY_PAIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    backtest_on_slice,
    get_feature_arrays,
    load_all_pairs,
    pset,
    robust_rr_backtest,
)


RNG_SEED = 42
CASE_WINDOWS = (1, 5, 10, 20)

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


@dataclass(frozen=True)
class CaseSpec:
    kind: str
    start: int
    end: int
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lexicase / case-wise GP evolution for crypto trading.",
    )
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--n-gen", type=int, default=N_GEN)
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_lexicase.dill"),
    )
    parser.add_argument(
        "--meta-out",
        default=str(MODELS_DIR / "best_crypto_gp_lexicase_meta.json"),
    )
    return parser.parse_args()


def ensure_creator_types() -> None:
    if not hasattr(creator, "FitnessLexicase"):
        creator.create("FitnessLexicase", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualLexicase"):
        creator.create(
            "IndividualLexicase",
            gp.PrimitiveTree,
            fitness=creator.FitnessLexicase,
        )


def build_toolbox() -> base.Toolbox:
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.IndividualLexicase, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_LEN))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_LEN))
    toolbox.register("select", lexicase_select)
    toolbox.register("clone", copy.deepcopy)
    return toolbox


def normalize_signal_output(raw: Any, size: int) -> np.ndarray:
    arr = np.asarray(raw, dtype="float64")
    if arr.shape == ():
        arr = np.full(size, float(arr), dtype="float64")
    if arr.shape[0] != size:
        if arr.size == 1:
            arr = np.full(size, float(arr.reshape(())), dtype="float64")
        else:
            raise ValueError(f"Signal output size mismatch: expected {size}, got {arr.shape[0]}")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -100.0, 100.0)


def build_case_specs(index: pd.DatetimeIndex, windows: tuple[int, ...] = CASE_WINDOWS) -> list[CaseSpec]:
    days = pd.DatetimeIndex(index.normalize().unique())
    if len(days) == 0:
        return []

    specs: list[CaseSpec] = []
    for i, day in enumerate(days):
        specs.append(CaseSpec("day", i, i + 1, f"day:{day.date()}"))

    for window in windows:
        if window <= 1:
            continue
        for start in range(0, len(days), window):
            end = min(start + window, len(days))
            if end - start < 2:
                continue
            specs.append(CaseSpec("block", start, end, f"block{window}:{days[start].date()}_{days[end-1].date()}"))
    return specs


def block_average_daily_return(daily_returns: np.ndarray) -> float:
    if len(daily_returns) == 0:
        return -1.0
    clipped = np.clip(daily_returns.astype("float64"), -0.99, 10.0)
    compounded = float(np.prod(1.0 + clipped))
    return float(compounded ** (1.0 / len(clipped)) - 1.0)


def build_case_scores(daily_returns: np.ndarray, case_specs: list[CaseSpec]) -> np.ndarray:
    scores: list[float] = []
    for spec in case_specs:
        if spec.end <= spec.start:
            scores.append(-1.0)
            continue
        if spec.kind == "day":
            idx = spec.start
            if idx >= len(daily_returns):
                scores.append(-1.0)
            else:
                scores.append(float(daily_returns[idx]))
            continue
        segment = daily_returns[spec.start:spec.end]
        scores.append(block_average_daily_return(segment))
    return np.asarray(scores, dtype="float64")


def summary_score(metrics: dict[str, Any], aggregate: dict[str, Any], tree_size: int) -> float:
    daily = metrics["daily_metrics"]
    monthly = metrics["monthly_metrics"]
    score = 0.0
    score += max(0.0, DAILY_TARGET_PCT - daily["avg_daily_return"]) * 12000.0
    score += daily["daily_shortfall_sum"] * 3000.0
    score += monthly["monthly_shortfall_sum"] * 7000.0
    score += max(0.0, 0.0 - metrics["total_return"]) * 800.0
    score += abs(metrics["max_drawdown"]) * 350.0
    score += max(0.0, 0.05 - daily["daily_win_rate"]) * 300.0
    score += max(0.0, 0.02 - daily["daily_target_hit_rate"]) * 1000.0
    score += max(0.0, -aggregate["avg_cvar"]) * 800.0
    score += max(0.0, -daily["worst_day"]) * 500.0
    score += tree_size * 0.02
    return float(score)


def evaluate_individual(
    ind: gp.PrimitiveTree,
    toolbox: base.Toolbox,
    df_slice: pd.DataFrame,
    case_specs: list[CaseSpec],
) -> tuple[float]:
    try:
        func = toolbox.compile(expr=ind)
        cols = get_feature_arrays(df_slice, PRIMARY_PAIR)
        raw_signal = func(*cols)
        desired_pcts = normalize_signal_output(raw_signal, len(df_slice))

        robust = robust_rr_backtest(df_slice, desired_pcts, PRIMARY_PAIR)
        primary = robust["primary"]
        aggregate = robust["aggregate"]
        daily = primary["daily_metrics"]
        monthly = primary["monthly_metrics"]

        case_scores = build_case_scores(primary["net_ret"], case_specs)
        ind.case_scores = case_scores
        ind.case_labels = [spec.label for spec in case_specs]
        ind.case_summary = {
            "case_count": int(len(case_scores)),
            "case_mean": float(np.mean(case_scores)) if len(case_scores) else 0.0,
            "case_median": float(np.median(case_scores)) if len(case_scores) else 0.0,
            "case_best": float(np.max(case_scores)) if len(case_scores) else 0.0,
            "case_worst": float(np.min(case_scores)) if len(case_scores) else 0.0,
        }

        if len(daily["daily_returns"]) < 20 or primary["n_trades"] < 3:
            return (1e6,)
        if primary["max_drawdown"] < -0.30:
            return (1e6,)
        if daily["worst_day"] <= -0.04:
            return (1e6,)

        return (
            summary_score(primary, aggregate, len(ind)),
        )
    except Exception:
        ind.case_scores = np.full(len(case_specs), -1.0, dtype="float64")
        ind.case_labels = [spec.label for spec in case_specs]
        ind.case_summary = {
            "case_count": int(len(case_specs)),
            "case_mean": -1.0,
            "case_median": -1.0,
            "case_best": -1.0,
            "case_worst": -1.0,
        }
        return (1e6,)


def compute_case_epsilons(case_matrix: np.ndarray) -> np.ndarray:
    eps = []
    for col in case_matrix.T:
        median = float(np.median(col))
        mad = float(np.median(np.abs(col - median)))
        eps.append(max(1e-6, mad))
    return np.asarray(eps, dtype="float64")


def lexicase_select(population: list[gp.PrimitiveTree], k: int) -> list[gp.PrimitiveTree]:
    case_matrix = getattr(lexicase_select, "case_matrix", None)
    eps = getattr(lexicase_select, "case_eps", None)
    if case_matrix is None or eps is None or len(population) == 0:
        return random.choices(population, k=k)

    case_matrix = np.asarray(case_matrix, dtype="float64")
    eps = np.asarray(eps, dtype="float64")
    n_cases = case_matrix.shape[1]
    selected: list[gp.PrimitiveTree] = []

    for _ in range(k):
        candidates = list(range(len(population)))
        for case_idx in np.random.permutation(n_cases):
            vals = case_matrix[candidates, case_idx]
            best = float(np.max(vals))
            threshold = best - float(eps[case_idx])
            candidates = [idx for idx in candidates if case_matrix[idx, case_idx] >= threshold]
            if len(candidates) <= 1:
                break

        if not candidates:
            selected.append(random.choice(population))
        elif len(candidates) == 1:
            selected.append(population[candidates[0]])
        else:
            selected.append(population[random.choice(candidates)])

    return selected


def update_selection_cache(population: list[gp.PrimitiveTree], toolbox: base.Toolbox) -> None:
    case_matrix = np.vstack([ind.case_scores for ind in population])
    toolbox.case_matrix = case_matrix
    toolbox.case_eps = compute_case_epsilons(case_matrix)
    lexicase_select.case_matrix = case_matrix
    lexicase_select.case_eps = toolbox.case_eps


def run_evolution(
    train_df: pd.DataFrame,
    toolbox: base.Toolbox,
    case_specs: list[CaseSpec],
    pop_size: int,
    n_gen: int,
) -> tools.HallOfFame:
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10, similar=lambda a, b: a == b)
    eval_count = 0
    start_time = time.time()

    print(f"\nEvolution(Lexicase): pop={pop_size}, gen={n_gen}, cx={P_CX}, mut={P_MUT}")
    print(f"{'Gen':>4} | {'Best':>12} | {'Avg':>12} | {'Cases':>6} | {'Time':>8}")
    print("-" * 52)

    def evaluate_population(population: list[gp.PrimitiveTree]) -> None:
        nonlocal eval_count
        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = evaluate_individual(ind, toolbox, train_df, case_specs)
            eval_count += 1

    evaluate_population(pop)
    update_selection_cache(pop, toolbox)
    hof.update(pop)

    for gen in range(1, n_gen + 1):
        parents = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in parents]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUT:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        evaluate_population(offspring)
        pop[:] = offspring
        update_selection_cache(pop, toolbox)
        hof.update(pop)

        fitness_vals = np.asarray([ind.fitness.values[0] for ind in pop], dtype="float64")
        elapsed = time.time() - start_time
        print(
            f"{gen:4d} | {np.min(fitness_vals):12.6f} | {np.mean(fitness_vals):12.6f} | "
            f"{toolbox.case_matrix.shape[1]:6d} | {elapsed:7.1f}s"
        )

    print(f"\nEvolution complete: evals={eval_count:,}, best={hof[0].fitness.values[0]:.6f}")
    return hof


def validate_best(hof: tools.HallOfFame, val_df: pd.DataFrame) -> tuple[gp.PrimitiveTree, dict[str, Any]]:
    best_ind = None
    best_payload = None
    best_score = math.inf

    for idx, ind in enumerate(hof, start=1):
        score = evaluate_individual(ind, build_toolbox(), val_df, build_case_specs(val_df.index))[0]
        payload = {
            "rank": idx,
            "validation_score": float(score),
            "tree_size": len(ind),
        }
        if score < best_score:
            best_score = score
            best_ind = ind
            best_payload = payload

    if best_ind is None or best_payload is None:
        raise RuntimeError("Validation selection failed")
    return best_ind, best_payload


def main() -> None:
    args = parse_args()
    ensure_creator_types()
    toolbox = build_toolbox()

    print("=" * 72)
    print("  Lexicase / Case-Wise GP Crypto Strategy")
    print("=" * 72)

    print("\n[Phase 1] Load Data")
    df_all = load_all_pairs()
    train_df = df_all.loc[TRAIN_START:TRAIN_END].copy()
    val_df = df_all.loc[VAL_START:VAL_END].copy()
    test_df = df_all.loc[TEST_START:TEST_END].copy()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} bars")

    case_specs = build_case_specs(train_df.index)
    if not case_specs:
        raise RuntimeError("No cases built from training data")
    print(f"  Cases: {len(case_specs)} (windows={CASE_WINDOWS})")

    print("\n[Phase 2] Evolution")
    hof = run_evolution(
        train_df=train_df,
        toolbox=toolbox,
        case_specs=case_specs,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
    )

    print("\n[Phase 3] Validation")
    best, val_pick = validate_best(hof, val_df)
    print(f"  Selected rank #{val_pick['rank']} with score {val_pick['validation_score']:.6f}")

    print("\n[Phase 4] Out-of-Sample Test")
    test_result = backtest_on_slice(best, test_df, "TEST (Lexicase)")

    print("\n[Phase 5] Full-Period Backtest")
    full_result = backtest_on_slice(best, df_all, "FULL PERIOD (Lexicase)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)
    with open(model_path, "wb") as f:
        dill.dump(gp.PrimitiveTree(best), f)

    meta = {
        "algorithm": "lexicase_casewise_gp",
        "pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        "primary_pair": PRIMARY_PAIR,
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period": f"{VAL_START} ~ {VAL_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "case_windows": list(CASE_WINDOWS),
        "case_count": len(case_specs),
        "tree_size": len(best),
        "fitness": float(best.fitness.values[0]),
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
