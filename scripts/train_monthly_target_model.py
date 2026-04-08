"""
Train a monthly-target GP model using NSGA-II and daily-session execution.
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
from deap import base, creator, gp, tools

from gp_crypto_evolution import (
    DAILY_TARGET_PCT,
    DEFAULT_REWARD_MULTIPLE,
    MAX_DEPTH,
    MAX_LEN,
    MODELS_DIR,
    N_GEN,
    P_CX,
    P_MUT,
    POP_SIZE,
    TEST_END,
    TEST_START,
    TIMEFRAME,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    PAIRS,
    PRIMARY_PAIR,
    backtest_on_slice,
    daily_session_backtest,
    get_feature_arrays,
    load_all_pairs,
    pset,
    robust_rr_backtest,
    ROBUST_REWARD_MULTIPLES,
    select_best_on_validation,
    split_dataset,
)
from gp_crypto_infer import cmd_walkforward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a monthly-target GP model with NSGA-II.",
    )
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--n-gen", type=int, default=N_GEN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_monthly_target.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "best_crypto_gp_monthly_target_summary.json"),
    )
    parser.add_argument("--walkforward-window", type=int, default=3)
    parser.add_argument("--walkforward-step", type=int, default=1)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def ensure_monthly_creators() -> None:
    if not hasattr(creator, "FitnessMonthlyNSGA"):
        creator.create(
            "FitnessMonthlyNSGA",
            base.Fitness,
            weights=(-1.0, -1.0, -1.0),
        )
    if not hasattr(creator, "MonthlyNSGAIndividual"):
        creator.create(
            "MonthlyNSGAIndividual",
            gp.PrimitiveTree,
            fitness=creator.FitnessMonthlyNSGA,
        )


def build_monthly_toolbox() -> base.Toolbox:
    ensure_monthly_creators()

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=MAX_DEPTH)
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.MonthlyNSGAIndividual,
        toolbox.expr,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_LEN))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_LEN))
    return toolbox


def run_backtest(individual, df_slice, toolbox: base.Toolbox) -> dict:
    func = toolbox.compile(expr=individual)
    cols = get_feature_arrays(df_slice, PRIMARY_PAIR)
    desired_pcts = func(*cols)
    return daily_session_backtest(
        df_slice,
        desired_pcts,
        PRIMARY_PAIR,
        reward_multiple=DEFAULT_REWARD_MULTIPLE,
    )


def monthly_validation_score(result: dict, tree_size: int) -> float:
    daily = result["daily_metrics"]
    monthly = result["monthly_metrics"]

    score = 0.0
    score += monthly["monthly_shortfall_sum"] * 2500.0
    score += (1.0 - monthly["monthly_target_hit_rate"]) * 600.0
    score += (1.0 - monthly["monthly_avg_daily_target_hit_rate"]) * 300.0
    score += abs(result["max_drawdown"]) * 250.0
    score += max(0.0, -daily["cvar"]) * 400.0
    score += max(0.0, -monthly["avg_monthly_return"]) * 250.0
    score += max(0.0, tree_size - 40) * 0.5
    score -= result["total_return"] * 50.0
    return score


def evaluate_monthly_objectives(
    individual,
    df_slice,
    toolbox: base.Toolbox,
) -> tuple[float, float, float]:
    try:
        func = toolbox.compile(expr=individual)
        cols = get_feature_arrays(df_slice, PRIMARY_PAIR)
        desired_pcts = func(*cols)
        robust = robust_rr_backtest(
            df_slice,
            desired_pcts,
            PRIMARY_PAIR,
            reward_multiples=ROBUST_REWARD_MULTIPLES,
        )
        result = robust["primary"]
        aggregate = robust["aggregate"]
        daily = result["daily_metrics"]
        monthly = result["monthly_metrics"]

        if monthly["n_months"] < 2 or result["n_trades"] < 10:
            return (1e6, 1.0, 1e3)
        if result["max_drawdown"] < -0.25:
            return (1e6, 1.0, 1e3)
        if daily["worst_day"] <= -0.03:
            return (1e6, 1.0, 1e3)

        obj_shortfall = aggregate["monthly_shortfall_sum"]
        obj_miss_rate = 1.0 - aggregate["monthly_target_hit_rate"]
        obj_risk = (
            aggregate["avg_abs_max_drawdown"]
            + max(0.0, -aggregate["avg_cvar"])
            + monthly["monthly_shortfall_mean"]
        )
        return (obj_shortfall, obj_miss_rate, obj_risk)
    except Exception:
        return (1e6, 1.0, 1e3)


def print_nsga_result(label: str, result: dict) -> None:
    daily = result["daily_metrics"]
    monthly = result["monthly_metrics"]
    print(f"\n=== {label} ===")
    print(f"  Return:       {result['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Trades:       {result['n_trades']}")
    print(f"  Win Rate:     {result['win_rate']*100:.1f}%")
    print(f"  Final Equity: ${result['final_equity']:,.2f}")
    print(f"  Avg Daily:    {daily['avg_daily_return']*100:+.2f}%")
    print(f"  Daily Win:    {daily['daily_win_rate']*100:.1f}%")
    print(f"  Target Hit:   {daily['daily_target_hit_rate']*100:.1f}% "
          f"(>= {DAILY_TARGET_PCT*100:.2f}%/day)")
    print(f"  Monthly Hit:  {monthly['monthly_target_hit_rate']*100:.1f}%")
    print(f"  Avg Month:    {monthly['avg_monthly_return']*100:+.2f}%")
    print(f"  Worst Month:  {monthly['worst_month']*100:.2f}%")


def run_nsga2_evolution(
    train_df,
    toolbox: base.Toolbox,
    pop_size: int,
    n_gen: int,
) -> tools.ParetoFront:
    pop = toolbox.population(n=pop_size)
    pareto = tools.ParetoFront(similar=lambda a, b: a == b)

    print(f"\nNSGA-II Evolution: pop={pop_size}, gen={n_gen}, cx={P_CX}, mut={P_MUT}")
    print(f"{'Gen':>4} | {'Shortfall':>19} | {'MissRate':>19} | {'Risk':>19} | {'Front':>6} | {'Time':>8}")
    print("-" * 92)

    start_time = time.time()

    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind in invalid:
        ind.fitness.values = evaluate_monthly_objectives(ind, train_df, toolbox)

    pop = toolbox.select(pop, len(pop))
    pareto.update(pop)

    for gen in range(1, n_gen + 1):
        k = len(pop)
        if k % 4 != 0:
            k -= k % 4
        offspring = tools.selTournamentDCD(pop, k)
        if len(offspring) < len(pop):
            offspring.extend(random.choice(pop) for _ in range(len(pop) - len(offspring)))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                toolbox.mate(ind1, ind2)
                del ind1.fitness.values, ind2.fitness.values

        for ind in offspring:
            if random.random() < P_MUT:
                toolbox.mutate(ind)
                del ind.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = evaluate_monthly_objectives(ind, train_df, toolbox)

        pop = toolbox.select(pop + offspring, pop_size)
        pareto.update(pop)

        fits = np.asarray([ind.fitness.values for ind in pop], dtype="float64")
        mins = fits.min(axis=0)
        avgs = fits.mean(axis=0)
        elapsed = time.time() - start_time
        print(
            f"{gen:4d} | "
            f"{mins[0]:8.4f}/{avgs[0]:8.4f} | "
            f"{mins[1]:8.4f}/{avgs[1]:8.4f} | "
            f"{mins[2]:8.4f}/{avgs[2]:8.4f} | "
            f"{len(pareto):6d} | "
            f"{elapsed:7.1f}s"
        )

    return pareto


def select_best_from_pareto(pareto, val_df, toolbox: base.Toolbox):
    print("\nEvaluating Pareto front on validation set...")
    best = None
    best_score = float("inf")

    for i, ind in enumerate(pareto):
        result = run_backtest(ind, val_df, toolbox)
        score = monthly_validation_score(result, len(ind))
        monthly = result["monthly_metrics"]
        print(
            f"  #{i+1}: score={score:.4f}, "
            f"month_hit={monthly['monthly_target_hit_rate']*100:.1f}%, "
            f"return={result['total_return']*100:+.2f}%, size={len(ind)}"
        )
        if score < best_score:
            best = ind
            best_score = score

    print(f"Best validation score: {best_score:.4f}")
    return best


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 72)
    print("  Monthly-Target GP Training (NSGA-II)")
    print("=" * 72)

    print("\n[Phase 1] Data Loading")
    df_all = load_all_pairs()
    train_df, val_df, test_df = split_dataset(df_all)

    print("\n[Phase 2] NSGA-II Evolution")
    toolbox = build_monthly_toolbox()
    pareto = run_nsga2_evolution(train_df, toolbox, args.pop_size, args.n_gen)

    print("\n[Phase 3] Validation")
    best = select_best_from_pareto(pareto, val_df, toolbox)

    print("\n[Phase 4] Out-of-Sample Test")
    test_result = run_backtest(best, test_df, toolbox)
    print_nsga_result("TEST (Monthly Target)", test_result)

    print("\n[Phase 5] Full-Period Backtest")
    full_result = run_backtest(best, df_all, toolbox)
    print_nsga_result("FULL PERIOD (Monthly Target)", full_result)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_out)
    with open(model_path, "wb") as f:
        dill.dump(best, f)
    print(f"\nModel saved: {model_path}")

    print("\n[Phase 6] Walk-Forward")
    walkforward_df = cmd_walkforward(
        model_path=str(model_path),
        window_months=args.walkforward_window,
        step_months=args.walkforward_step,
    )

    summary = {
        "pairs": PAIRS,
        "primary_pair": PRIMARY_PAIR,
        "timeframe": TIMEFRAME,
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period": f"{VAL_START} ~ {VAL_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "seed": args.seed,
        "tree_size": len(best),
        "fitness": list(best.fitness.values),
        "daily_target_pct": DAILY_TARGET_PCT,
        "reward_multiple": DEFAULT_REWARD_MULTIPLE,
        "robust_reward_multiples": list(ROBUST_REWARD_MULTIPLES),
        "test": test_result,
        "full_period": full_result,
        "walkforward": {
            "n_windows": int(len(walkforward_df)),
            "avg_return": (
                float(walkforward_df["total_return"].mean())
                if not walkforward_df.empty
                else 0.0
            ),
            "avg_sharpe": (
                float(walkforward_df["sharpe"].mean())
                if not walkforward_df.empty
                else 0.0
            ),
            "win_rate": (
                float((walkforward_df["total_return"] > 0).mean())
                if not walkforward_df.empty
                else 0.0
            ),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = Path(args.summary_out)
    with open(summary_path, "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    print(f"Summary saved: {summary_path}")

    print(f"\nBest tree:\n  {best}")


if __name__ == "__main__":
    main()
