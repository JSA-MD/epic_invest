#!/usr/bin/env python3
"""Search a recent-window GP strategy for high average daily return."""

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


DEFAULT_TRAIN_REWARD_MULTIPLES = (3.0, 5.0, 8.0, 10.0)
DEFAULT_TRAIN_ENTRY_THRESHOLDS = (0.0, 10.0, 20.0, 30.0)
DEFAULT_FINAL_REWARD_MULTIPLES = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
DEFAULT_FINAL_ENTRY_THRESHOLDS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)


@dataclass
class SearchConfig:
    pop_size: int
    n_gen: int
    restarts: int
    months: int
    target_avg_daily_return: float
    min_trades: int
    max_drawdown_limit: float
    worst_day_limit: float
    train_reward_multiples: tuple[float, ...]
    train_entry_thresholds: tuple[float, ...]
    final_reward_multiples: tuple[float, ...]
    final_entry_thresholds: tuple[float, ...]
    seed: int


def parse_csv_floats(raw: str) -> tuple[float, ...]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("at least one numeric value is required")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a recent-window GP strategy for avg_daily_return >= target.",
    )
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument("--pop-size", type=int, default=160)
    parser.add_argument("--n-gen", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--target-avg-daily-return", type=float, default=0.005)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--max-drawdown-limit", type=float, default=-0.35)
    parser.add_argument("--worst-day-limit", type=float, default=-0.05)
    parser.add_argument(
        "--train-reward-multiples",
        default="3,5,8,10",
    )
    parser.add_argument(
        "--train-entry-thresholds",
        default="0,10,20,30",
    )
    parser.add_argument(
        "--final-reward-multiples",
        default="2,3,4,5,6,8,10,12",
    )
    parser.add_argument(
        "--final-entry-thresholds",
        default="0,5,10,15,20,25,30,40",
    )
    parser.add_argument(
        "--model-out",
        default=str(gp.MODELS_DIR / "recent_6m_gp_target_search.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "recent_6m_gp_target_search_summary.json"),
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        restarts=args.restarts,
        months=args.months,
        target_avg_daily_return=args.target_avg_daily_return,
        min_trades=args.min_trades,
        max_drawdown_limit=args.max_drawdown_limit,
        worst_day_limit=args.worst_day_limit,
        train_reward_multiples=parse_csv_floats(args.train_reward_multiples),
        train_entry_thresholds=parse_csv_floats(args.train_entry_thresholds),
        final_reward_multiples=parse_csv_floats(args.final_reward_multiples),
        final_entry_thresholds=parse_csv_floats(args.final_entry_thresholds),
        seed=args.seed,
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


def candidate_score(result: dict[str, Any], target: float) -> float:
    daily = result["daily_metrics"]
    avg_daily_return = float(daily["avg_daily_return"])
    total_return = float(result["total_return"])
    max_drawdown = float(result["max_drawdown"])
    worst_day = float(daily["worst_day"])
    hit_rate = float(daily["daily_target_hit_rate"])
    win_rate = float(daily["daily_win_rate"])
    monthly_shortfall = float(result["monthly_metrics"]["monthly_shortfall_sum"])

    score = 0.0
    score += max(0.0, target - avg_daily_return) * 600000.0
    score += monthly_shortfall * 200000.0
    score += abs(max_drawdown) * 15000.0
    score += max(0.0, -worst_day) * 25000.0
    score -= avg_daily_return * 150000.0
    score -= total_return * 7000.0
    score -= hit_rate * 2500.0
    score -= win_rate * 800.0
    return score


def passes_constraints(result: dict[str, Any], cfg: SearchConfig) -> bool:
    daily = result["daily_metrics"]
    return (
        result["n_trades"] >= cfg.min_trades
        and result["max_drawdown"] > cfg.max_drawdown_limit
        and daily["worst_day"] > cfg.worst_day_limit
    )


def evaluate_grid(
    desired_pcts: np.ndarray,
    df_slice: pd.DataFrame,
    cfg: SearchConfig,
    reward_multiples: tuple[float, ...],
    entry_thresholds: tuple[float, ...],
) -> tuple[float, dict[str, Any] | None]:
    best_payload = None
    best_score = float("inf")

    for reward_multiple in reward_multiples:
        for entry_threshold in entry_thresholds:
            result = gp.daily_session_backtest(
                df_slice,
                desired_pcts,
                pair=gp.PRIMARY_PAIR,
                reward_multiple=float(reward_multiple),
                entry_threshold=float(entry_threshold),
            )
            if not passes_constraints(result, cfg):
                continue
            score = candidate_score(result, cfg.target_avg_daily_return)
            if best_payload is None or score < best_score:
                best_score = score
                best_payload = {
                    "result": result,
                    "reward_multiple": float(reward_multiple),
                    "entry_threshold": float(entry_threshold),
                    "score": float(score),
                }

    return best_score, best_payload


def evaluate_individual(
    individual,
    toolbox: deap_base.Toolbox,
    df_slice: pd.DataFrame,
    cfg: SearchConfig,
) -> tuple[float]:
    try:
        if len(individual) > gp.MAX_LEN:
            return (1e9,)
        func = toolbox.compile(expr=individual)
        desired_pcts = func(*gp.get_feature_arrays(df_slice, gp.PRIMARY_PAIR))
        score, payload = evaluate_grid(
            desired_pcts=desired_pcts,
            df_slice=df_slice,
            cfg=cfg,
            reward_multiples=cfg.train_reward_multiples,
            entry_thresholds=cfg.train_entry_thresholds,
        )
        if payload is None:
            return (1e9,)

        tree_size = len(individual)
        if tree_size > 40:
            score += (tree_size - 40) * 0.05
        return (score,)
    except Exception:
        return (1e9,)


def evaluate_hof_candidate(
    individual,
    toolbox: deap_base.Toolbox,
    df_slice: pd.DataFrame,
    cfg: SearchConfig,
) -> dict[str, Any] | None:
    try:
        if len(individual) > gp.MAX_LEN:
            return None
        func = toolbox.compile(expr=individual)
        desired_pcts = func(*gp.get_feature_arrays(df_slice, gp.PRIMARY_PAIR))
        _, payload = evaluate_grid(
            desired_pcts=desired_pcts,
            df_slice=df_slice,
            cfg=cfg,
            reward_multiples=cfg.final_reward_multiples,
            entry_thresholds=cfg.final_entry_thresholds,
        )
        if payload is None:
            return None
        return {
            "tree": individual,
            "tree_size": len(individual),
            "reward_multiple": payload["reward_multiple"],
            "entry_threshold": payload["entry_threshold"],
            "fitness_score": payload["score"],
            "result": payload["result"],
        }
    except Exception:
        return None


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    result = candidate["result"]
    daily = result["daily_metrics"]
    monthly = result["monthly_metrics"]
    return {
        "tree": str(candidate["tree"]),
        "tree_size": candidate["tree_size"],
        "reward_multiple": candidate["reward_multiple"],
        "entry_threshold": candidate["entry_threshold"],
        "fitness_score": candidate["fitness_score"],
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]),
        "n_trades": int(result["n_trades"]),
        "avg_daily_return": float(daily["avg_daily_return"]),
        "daily_target_hit_rate": float(daily["daily_target_hit_rate"]),
        "daily_win_rate": float(daily["daily_win_rate"]),
        "worst_day": float(daily["worst_day"]),
        "avg_monthly_return": float(monthly["avg_monthly_return"]),
        "monthly_target_hit_rate": float(monthly["monthly_target_hit_rate"]),
    }


def run_restart(
    restart_idx: int,
    recent_df: pd.DataFrame,
    cfg: SearchConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = cfg.seed + restart_idx
    random.seed(seed)
    np.random.seed(seed)
    toolbox = build_toolbox()

    def eval_func(individual):
        return evaluate_individual(individual, toolbox, recent_df, cfg)

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
        payload = evaluate_hof_candidate(individual, toolbox, recent_df, cfg)
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
        f" r={summary['reward_multiple']:.1f}"
        f" th={summary['entry_threshold']:.1f}"
        f" trades={summary['n_trades']}"
    )
    return best, [summarize_candidate(candidate) for candidate in candidates[:5]]


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    gp.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    recent_df, actual_start, actual_end = build_recent_window(cfg.months)
    print("=" * 78)
    print("  Recent-Window GP Target Search")
    print("=" * 78)
    print(
        f"Window: {actual_start.isoformat()} -> {actual_end.isoformat()} "
        f"({len(recent_df)} bars)"
    )
    print(
        "Search grid:"
        f" train_reward={cfg.train_reward_multiples},"
        f" train_threshold={cfg.train_entry_thresholds},"
        f" final_reward={cfg.final_reward_multiples},"
        f" final_threshold={cfg.final_entry_thresholds}"
    )

    best_overall = None
    restart_summaries = []
    for restart_idx in range(cfg.restarts):
        candidate, top5 = run_restart(restart_idx, recent_df, cfg)
        candidate_summary = summarize_candidate(candidate)
        restart_summaries.append(
            {
                "restart": restart_idx + 1,
                "seed": cfg.seed + restart_idx,
                "best": candidate_summary,
                "top5": top5,
            }
        )
        if best_overall is None or (
            candidate_summary["avg_daily_return"],
            candidate_summary["total_return"],
            candidate_summary["sharpe"],
        ) > (
            summarize_candidate(best_overall)["avg_daily_return"],
            summarize_candidate(best_overall)["total_return"],
            summarize_candidate(best_overall)["sharpe"],
        ):
            best_overall = candidate

        if candidate_summary["avg_daily_return"] >= cfg.target_avg_daily_return:
            print(
                f"Target reached at restart {restart_idx + 1}: "
                f"{candidate_summary['avg_daily_return']*100:+.3f}%"
            )
            break

    if best_overall is None:
        raise RuntimeError("search produced no candidate")

    best_summary = summarize_candidate(best_overall)
    model_payload = {
        "algorithm": "recent_window_gp_target_search",
        "tree": best_overall["tree"],
        "reward_multiple": best_overall["reward_multiple"],
        "entry_threshold": best_overall["entry_threshold"],
        "window_start": actual_start,
        "window_end": actual_end,
        "target_avg_daily_return": cfg.target_avg_daily_return,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_payload = {
        "algorithm": "recent_window_gp_target_search",
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
    print(f"  Reward R:     1:{best_summary['reward_multiple']:.1f}")
    print(f"  Entry Thr:    {best_summary['entry_threshold']:.1f}")
    print(f"  Tree Size:    {best_summary['tree_size']}")
    print(f"  Target Met:   {summary_payload['target_reached']}")
    print(f"Model saved: {args.model_out}")
    print(f"Summary saved: {args.summary_out}")


if __name__ == "__main__":
    main()
