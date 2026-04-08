#!/usr/bin/env python3
"""Search a recent-window vectorized GP strategy for high avg daily return."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd
from deap import base as deap_base
from deap import tools

import gp_crypto_evolution as gp


@dataclass
class SearchConfig:
    months: int
    pop_size: int
    n_gen: int
    restarts: int
    seed: int
    target_avg_daily_return: float
    max_drawdown_limit: float
    min_trades: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a recent-window vectorized GP strategy.",
    )
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--pop-size", type=int, default=240)
    parser.add_argument("--n-gen", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--target-avg-daily-return", type=float, default=0.005)
    parser.add_argument("--max-drawdown-limit", type=float, default=-0.60)
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument(
        "--model-out",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_summary.json"),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        months=args.months,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        restarts=args.restarts,
        seed=args.seed,
        target_avg_daily_return=args.target_avg_daily_return,
        max_drawdown_limit=args.max_drawdown_limit,
        min_trades=args.min_trades,
    )


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
    return value


def build_recent_window(months: int) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    df_all = gp.load_all_pairs(refresh_cache=False)
    end = pd.Timestamp(df_all.index.max())
    start = end - pd.DateOffset(months=months)
    recent = df_all.loc[start:end].copy()
    if recent.empty:
        raise RuntimeError("recent window is empty")
    return recent, pd.Timestamp(recent.index.min()), pd.Timestamp(recent.index.max())


def build_toolbox() -> deap_base.Toolbox:
    toolbox = deap_base.Toolbox()
    toolbox.register("expr", gp.gp.genHalfAndHalf, pset=gp.pset, min_=1, max_=gp.MAX_DEPTH)
    def bounded_individual():
        while True:
            individual = gp.creator.Individual(toolbox.expr())
            if len(individual) <= gp.MAX_LEN:
                return individual
    toolbox.register("individual", bounded_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.gp.compile, pset=gp.pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.gp.cxOnePoint)
    toolbox.register("expr_mut", gp.gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.gp.mutUniform, expr=toolbox.expr_mut, pset=gp.pset)
    toolbox.decorate("mate", gp.gp.staticLimit(key=len, max_value=gp.MAX_LEN))
    toolbox.decorate("mutate", gp.gp.staticLimit(key=len, max_value=gp.MAX_LEN))
    return toolbox


def candidate_score(result: dict[str, Any], cfg: SearchConfig, tree_size: int) -> float:
    daily = result["daily_metrics"]
    avg_daily_return = float(daily["avg_daily_return"])
    total_return = float(result["total_return"])
    max_drawdown = float(result["max_drawdown"])
    hit_rate = float(daily["daily_target_hit_rate"])
    win_rate = float(daily["daily_win_rate"])
    worst_day = float(daily["worst_day"])
    monthly_target_hit_rate = float(result["daily_metrics"]["daily_target_hit_rate"])

    score = 0.0
    score += max(0.0, cfg.target_avg_daily_return - avg_daily_return) * 500000.0
    score += abs(max_drawdown) * 12000.0
    score += max(0.0, -worst_day) * 25000.0
    score -= avg_daily_return * 160000.0
    score -= total_return * 9000.0
    score -= hit_rate * 2500.0
    score -= win_rate * 1000.0
    score -= monthly_target_hit_rate * 500.0
    if tree_size > 40:
        score += (tree_size - 40) * 0.1
    return score


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    result = candidate["result"]
    daily = result["daily_metrics"]
    return {
        "tree": str(candidate["tree"]),
        "tree_size": int(candidate["tree_size"]),
        "fitness_score": float(candidate["fitness_score"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]),
        "n_trades": int(result["n_trades"]),
        "avg_daily_return": float(daily["avg_daily_return"]),
        "daily_target_hit_rate": float(daily["daily_target_hit_rate"]),
        "daily_win_rate": float(daily["daily_win_rate"]),
        "worst_day": float(daily["worst_day"]),
        "best_day": float(daily["best_day"]),
    }


def evaluate_individual(
    individual,
    toolbox: deap_base.Toolbox,
    df_slice: pd.DataFrame,
    close: np.ndarray,
    cfg: SearchConfig,
) -> tuple[float]:
    try:
        if len(individual) > gp.MAX_LEN:
            return (1e9,)
        func = toolbox.compile(expr=individual)
        desired_pcts = func(*gp.get_feature_arrays(df_slice, gp.PRIMARY_PAIR))
        result = gp.vectorized_backtest(close, desired_pcts)
        if result["n_trades"] < cfg.min_trades:
            return (1e9,)
        if result["max_drawdown"] < cfg.max_drawdown_limit:
            return (1e9,)
        score = candidate_score(result, cfg, len(individual))
        return (score,)
    except Exception:
        return (1e9,)


def evaluate_candidate(
    individual,
    toolbox: deap_base.Toolbox,
    df_slice: pd.DataFrame,
    close: np.ndarray,
    cfg: SearchConfig,
) -> dict[str, Any] | None:
    try:
        if len(individual) > gp.MAX_LEN:
            return None
        func = toolbox.compile(expr=individual)
        desired_pcts = func(*gp.get_feature_arrays(df_slice, gp.PRIMARY_PAIR))
        result = gp.vectorized_backtest(close, desired_pcts)
        if result["n_trades"] < cfg.min_trades:
            return None
        if result["max_drawdown"] < cfg.max_drawdown_limit:
            return None
        return {
            "tree": individual,
            "tree_size": len(individual),
            "fitness_score": candidate_score(result, cfg, len(individual)),
            "result": result,
        }
    except Exception:
        return None


def run_restart(
    restart_idx: int,
    recent_df: pd.DataFrame,
    close: np.ndarray,
    cfg: SearchConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = cfg.seed + restart_idx
    random.seed(seed)
    np.random.seed(seed)
    toolbox = build_toolbox()

    def eval_func(individual):
        return evaluate_individual(individual, toolbox, recent_df, close, cfg)

    toolbox.register("evaluate", eval_func)
    population = toolbox.population(n=cfg.pop_size)
    hall = tools.HallOfFame(10, similar=lambda left, right: left == right)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    start_ts = time.time()

    print(
        f"\n[Restart {restart_idx + 1}/{cfg.restarts}] seed={seed} "
        f"pop={cfg.pop_size} gen={cfg.n_gen}"
    )
    print(f"{'Gen':>4} | {'Min Fitness':>12} | {'Avg Fitness':>12} | {'Elapsed':>8}")
    print("-" * 50)

    for gen in range(1, cfg.n_gen + 1):
        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        hall.update(population)

        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
        for left, right in zip(offspring[::2], offspring[1::2]):
            if random.random() < gp.P_CX:
                toolbox.mate(left, right)
                del left.fitness.values, right.fitness.values

        for mutant in offspring:
            if random.random() < gp.P_MUT:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring
        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        record = stats.compile(population)
        print(
            f"{gen:4d} | {record['min']:12.3f} | {record['avg']:12.3f} | "
            f"{time.time() - start_ts:7.1f}s"
        )

    candidates = []
    for individual in hall:
        payload = evaluate_candidate(individual, toolbox, recent_df, close, cfg)
        if payload is not None:
            candidates.append(payload)

    if not candidates:
        raise RuntimeError("no valid candidate survived Hall of Fame evaluation")

    candidates.sort(
        key=lambda item: (
            item["result"]["daily_metrics"]["avg_daily_return"],
            item["result"]["total_return"],
            item["result"]["sharpe"],
        ),
        reverse=True,
    )
    best = candidates[0]
    summary = summarize_candidate(best)
    print(
        "  Best restart candidate:"
        f" avg_daily={summary['avg_daily_return']*100:+.3f}%"
        f" total={summary['total_return']*100:+.2f}%"
        f" dd={summary['max_drawdown']*100:.2f}%"
        f" trades={summary['n_trades']}"
        f" size={summary['tree_size']}"
    )
    return best, [summarize_candidate(candidate) for candidate in candidates[:5]]


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    gp.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    recent_df, actual_start, actual_end = build_recent_window(cfg.months)
    close = recent_df[f"{gp.PRIMARY_PAIR}_close"].to_numpy(dtype="float64")
    print("=" * 78)
    print("  Recent-Window Vectorized GP Search")
    print("=" * 78)
    print(
        f"Window: {actual_start.isoformat()} -> {actual_end.isoformat()} "
        f"({len(recent_df)} bars)"
    )

    best_overall = None
    restart_summaries = []
    for restart_idx in range(cfg.restarts):
        candidate, top5 = run_restart(restart_idx, recent_df, close, cfg)
        summary = summarize_candidate(candidate)
        restart_summaries.append(
            {
                "restart": restart_idx + 1,
                "seed": cfg.seed + restart_idx,
                "best": summary,
                "top5": top5,
            }
        )
        if best_overall is None or (
            summary["avg_daily_return"],
            summary["total_return"],
            summary["sharpe"],
        ) > (
            summarize_candidate(best_overall)["avg_daily_return"],
            summarize_candidate(best_overall)["total_return"],
            summarize_candidate(best_overall)["sharpe"],
        ):
            best_overall = candidate

        if summary["avg_daily_return"] >= cfg.target_avg_daily_return:
            print(
                f"Target reached at restart {restart_idx + 1}: "
                f"{summary['avg_daily_return']*100:+.3f}%"
            )
            break

    if best_overall is None:
        raise RuntimeError("search produced no candidate")

    best_summary = summarize_candidate(best_overall)
    model_payload = {
        "algorithm": "recent_window_vectorized_gp_search",
        "tree": best_overall["tree"],
        "window_start": actual_start,
        "window_end": actual_end,
        "target_avg_daily_return": cfg.target_avg_daily_return,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_payload = {
        "algorithm": "recent_window_vectorized_gp_search",
        "window_start": actual_start,
        "window_end": actual_end,
        "bars": len(recent_df),
        "config": asdict(cfg),
        "best": best_summary,
        "target_reached": bool(best_summary["avg_daily_return"] >= cfg.target_avg_daily_return),
        "restart_summaries": restart_summaries,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "This search optimizes directly on the recent window and should be treated as in-sample until revalidated.",
    }

    with open(Path(args.model_out), "wb") as f:
        dill.dump(model_payload, f)
    with open(Path(args.summary_out), "w") as f:
        json.dump(json_safe(summary_payload), f, indent=2)

    print("\n[Final Best]")
    print(f"  Avg Daily:    {best_summary['avg_daily_return']*100:+.3f}%")
    print(f"  Total Return: {best_summary['total_return']*100:+.2f}%")
    print(f"  Max DD:       {best_summary['max_drawdown']*100:.2f}%")
    print(f"  Sharpe:       {best_summary['sharpe']:.3f}")
    print(f"  Trades:       {best_summary['n_trades']}")
    print(f"  Tree Size:    {best_summary['tree_size']}")
    print(f"  Target Met:   {summary_payload['target_reached']}")
    print(f"Model saved: {args.model_out}")
    print(f"Summary saved: {args.summary_out}")


if __name__ == "__main__":
    main()
