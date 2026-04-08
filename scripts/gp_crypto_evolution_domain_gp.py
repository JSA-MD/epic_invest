#!/usr/bin/env python3
"""Domain-primitive / grammar-guided GP for crypto rotation signals.

This experiment keeps the existing data loading and backtesting utilities but
replaces the generic primitive set with a strongly-typed, domain-specific one:

- signal transforms: spread, volatility scaling, gated blending, safe division
- regime gates: momentum / breadth / threshold / rank comparisons
- cross-asset inputs: BTC-relative strength and cross-sectional ranks

The intent is to reduce meaningless algebraic trees and bias evolution toward
interpretable momentum / regime / relative-strength expressions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import functools
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
    DAILY_TARGET_PCT,
    DEFAULT_REWARD_MULTIPLE,
    INITIAL_CASH,
    MAX_DEPTH,
    MAX_LEN,
    MODELS_DIR,
    N_GEN,
    P_CX,
    P_MUT,
    PAIRS,
    PRIMARY_PAIR,
    ROBUST_REWARD_MULTIPLES,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    load_all_pairs,
    robust_rr_backtest,
    daily_session_backtest,
)


RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

Signal = type("Signal", (), {})
Gate = type("Gate", (), {})

CORE_FEATURES = (
    "close",
    "rsi_14",
    "atr_14",
    "macd_h",
    "bb_p",
    "cci_14",
    "mfi_14",
    "dc_trend_05",
    "dc_event_05",
    "dc_overshoot_05",
    "dc_run_05",
    "vol_sma",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Domain-primitive / grammar-guided GP crypto experiment.",
    )
    parser.add_argument("--pop-size", type=int, default=250)
    parser.add_argument("--n-gen", type=int, default=6)
    parser.add_argument("--fold-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=15)
    parser.add_argument(
        "--entry-threshold",
        type=float,
        default=5.0,
        help="Minimum signal magnitude required to open a daily session.",
    )
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_domain.dill"),
    )
    parser.add_argument(
        "--meta-out",
        default=str(MODELS_DIR / "best_crypto_gp_domain_meta.json"),
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


def _finite(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype="float64", copy=True)
    if out.ndim == 0:
        if not np.isfinite(out):
            out[...] = 0.0
        return out
    out[~np.isfinite(out)] = 0.0
    return out


def sig_add(a, b):
    return np.clip(_finite(a) + _finite(b), -100.0, 100.0)


def sig_sub(a, b):
    return np.clip(_finite(a) - _finite(b), -100.0, 100.0)


def sig_mul(a, b):
    return np.clip(_finite(a) * _finite(b), -100.0, 100.0)


def sig_safe_div(a, b):
    a = _finite(a)
    b = _finite(b)
    out = np.zeros_like(a, dtype="float64")
    np.divide(a, b, out=out, where=np.abs(b) > 1e-8)
    return np.clip(out, -100.0, 100.0)


def sig_neg(a):
    return np.clip(-_finite(a), -100.0, 100.0)


def sig_abs(a):
    return np.abs(_finite(a))


def sig_tanh(a):
    return np.tanh(_finite(a))


def sig_spread(a, b):
    return np.tanh(_finite(a) - _finite(b))


def sig_vol_scale(signal, vol):
    signal = _finite(signal)
    vol = np.abs(_finite(vol))
    return np.clip(signal / (1.0 + vol), -100.0, 100.0)


def sig_rank2(a, b):
    return np.where(_finite(a) >= _finite(b), 1.0, -1.0)


def sig_blend(gate, a, b):
    gate_arr = np.where(_finite(gate) > 0.5, 1.0, 0.0)
    return np.clip(gate_arr * _finite(a) + (1.0 - gate_arr) * _finite(b), -100.0, 100.0)


def sig_from_gate(gate, scale):
    scale_arr = float(scale)
    if not np.isfinite(scale_arr):
        scale_arr = 0.0
    gate_arr = np.where(_finite(gate) > 0.5, 1.0, -1.0)
    return np.clip(gate_arr * scale_arr, -100.0, 100.0)


def sig_scale(a, scale):
    return np.clip(_finite(a) * float(scale), -100.0, 100.0)


def gate_gt(a, b):
    return np.where(_finite(a) > _finite(b), 1.0, 0.0)


def gate_lt(a, b):
    return np.where(_finite(a) < _finite(b), 1.0, 0.0)


def gate_between(a, low, high):
    lo = float(low)
    hi = float(high)
    if lo > hi:
        lo, hi = hi, lo
    arr = _finite(a)
    return np.where((arr >= lo) & (arr <= hi), 1.0, 0.0)


def gate_and(a, b):
    return np.minimum(np.where(_finite(a) > 0.5, 1.0, 0.0), np.where(_finite(b) > 0.5, 1.0, 0.0))


def gate_or(a, b):
    return np.maximum(np.where(_finite(a) > 0.5, 1.0, 0.0), np.where(_finite(b) > 0.5, 1.0, 0.0))


def gate_not(a):
    return 1.0 - np.where(_finite(a) > 0.5, 1.0, 0.0)


def gate_regime(momentum, vol, threshold):
    thr = float(threshold)
    momentum = _finite(momentum)
    vol = np.abs(_finite(vol))
    return np.where((momentum > thr) & (vol < max(1e-8, thr * 6.0)), 1.0, 0.0)


def gate_strength(a, b, margin):
    margin = abs(float(margin))
    diff = _finite(a) - _finite(b)
    return np.where(np.abs(diff) > margin, 1.0, 0.0)


def f_add(a, b):
    return float(a) + float(b)


def f_sub(a, b):
    return float(a) - float(b)


def f_mul(a, b):
    return float(a) * float(b)


def f_div(a, b):
    b = float(b)
    if abs(b) <= 1e-8:
        return float(a)
    return float(a) / b


def f_neg(a):
    return -float(a)


def f_abs(a):
    return abs(float(a))


def f_tanh(a):
    return math.tanh(float(a))


def build_domain_feature_frame(df_all: pd.DataFrame) -> pd.DataFrame:
    frame = df_all.copy()
    btc_close = frame[f"{PRIMARY_PAIR}_close"].astype("float64")
    btc_ret_12 = btc_close.pct_change(12).replace([np.inf, -np.inf], np.nan)
    btc_atr = frame[f"{PRIMARY_PAIR}_atr_14"].astype("float64").replace(0.0, np.nan)
    btc_dc = frame[f"{PRIMARY_PAIR}_dc_trend_05"].astype("float64")

    for pair in PAIRS:
        close = frame[f"{pair}_close"].astype("float64")
        ret_12 = close.pct_change(12).replace([np.inf, -np.inf], np.nan)
        atr = frame[f"{pair}_atr_14"].astype("float64").replace(0.0, np.nan)
        dc = frame[f"{pair}_dc_trend_05"].astype("float64")

        if pair != PRIMARY_PAIR:
            frame[f"{pair}_close_rel_btc"] = (close / btc_close - 1.0).replace([np.inf, -np.inf], np.nan)
            frame[f"{pair}_mom_rel_btc"] = (ret_12 - btc_ret_12).replace([np.inf, -np.inf], np.nan)
            frame[f"{pair}_atr_rel_btc"] = (atr / btc_atr - 1.0).replace([np.inf, -np.inf], np.nan)
            frame[f"{pair}_dc_rel_btc"] = dc - btc_dc

    momentum_cols = {}
    for pair in PAIRS:
        momentum_cols[pair] = frame[f"{pair}_close"].astype("float64").pct_change(12).replace([np.inf, -np.inf], np.nan)
    momentum_df = pd.DataFrame(momentum_cols, index=frame.index)
    frame["cross_breadth"] = momentum_df.gt(0.0).mean(axis=1)
    frame["cross_dispersion"] = momentum_df.std(axis=1)
    rank_df = momentum_df.rank(axis=1, pct=True)
    for pair in PAIRS:
        frame[f"{pair}_mom_rank"] = rank_df[pair]

    feature_cols = domain_feature_names()
    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = 0.0

    frame[feature_cols] = frame[feature_cols].bfill().ffill().fillna(0.0)
    frame.fillna(0.0, inplace=True)
    return frame


def domain_feature_names() -> list[str]:
    names: list[str] = []
    for pair in PAIRS:
        for feat in CORE_FEATURES:
            names.append(f"{pair}_{feat}")
    for pair in PAIRS:
        names.append(f"{pair}_mom_rank")
    for pair in PAIRS:
        if pair == PRIMARY_PAIR:
            continue
        names.extend(
            [
                f"{pair}_close_rel_btc",
                f"{pair}_mom_rel_btc",
                f"{pair}_atr_rel_btc",
                f"{pair}_dc_rel_btc",
            ]
        )
    names.extend(["cross_breadth", "cross_dispersion"])
    return names


def build_domain_inputs(df_slice: pd.DataFrame) -> list[np.ndarray]:
    return [df_slice[col].to_numpy(dtype="float64") for col in domain_feature_names()]


def build_folds(df_slice: pd.DataFrame, fold_days: int, step_days: int) -> list[pd.DataFrame]:
    days = pd.DatetimeIndex(df_slice.index.normalize().unique())
    if len(days) == 0:
        return []
    if len(days) <= fold_days:
        return [df_slice.copy()]
    folds: list[pd.DataFrame] = []
    for start_idx in range(0, len(days) - fold_days + 1, step_days):
        seg = days[start_idx:start_idx + fold_days]
        fold = df_slice.loc[str(seg[0].date()):str(seg[-1].date())].copy()
        if not fold.empty:
            folds.append(fold)
    return folds


def setup_pset() -> gp.PrimitiveSetTyped:
    inputs = [Signal] * len(domain_feature_names())
    pset = gp.PrimitiveSetTyped("DGPC", inputs, Signal)

    for op in (sig_add, sig_sub, sig_mul, sig_safe_div):
        pset.addPrimitive(op, [Signal, Signal], Signal)
    pset.addPrimitive(sig_neg, [Signal], Signal)
    pset.addPrimitive(sig_abs, [Signal], Signal)
    pset.addPrimitive(sig_tanh, [Signal], Signal)
    pset.addPrimitive(sig_spread, [Signal, Signal], Signal)
    pset.addPrimitive(sig_vol_scale, [Signal, Signal], Signal)
    pset.addPrimitive(sig_rank2, [Signal, Signal], Signal)
    pset.addPrimitive(sig_blend, [Gate, Signal, Signal], Signal)
    pset.addPrimitive(sig_from_gate, [Gate, float], Signal)
    pset.addPrimitive(sig_scale, [Signal, float], Signal)

    pset.addPrimitive(gate_gt, [Signal, Signal], Gate)
    pset.addPrimitive(gate_lt, [Signal, Signal], Gate)
    pset.addPrimitive(gate_between, [Signal, float, float], Gate)
    pset.addPrimitive(gate_and, [Gate, Gate], Gate)
    pset.addPrimitive(gate_or, [Gate, Gate], Gate)
    pset.addPrimitive(gate_not, [Gate], Gate)
    pset.addPrimitive(gate_regime, [Signal, Signal, float], Gate)
    pset.addPrimitive(gate_strength, [Signal, Signal, float], Gate)

    pset.addPrimitive(f_add, [float, float], float)
    pset.addPrimitive(f_sub, [float, float], float)
    pset.addPrimitive(f_mul, [float, float], float)
    pset.addPrimitive(f_div, [float, float], float)
    pset.addPrimitive(f_neg, [float], float)
    pset.addPrimitive(f_abs, [float], float)
    pset.addPrimitive(f_tanh, [float], float)

    pset.addTerminal(0.0, Gate)
    pset.addTerminal(1.0, Gate)
    pset.addEphemeralConstant("rand_scale", functools.partial(random.uniform, 2.0, 40.0), float)
    pset.addEphemeralConstant("rand_threshold", functools.partial(random.uniform, 0.0, 1.5), float)

    for i, name in enumerate(domain_feature_names()):
        pset.renameArguments(**{f"ARG{i}": name})
    return pset


def evaluate_fold_metrics(result: dict[str, Any]) -> tuple[float, float, float, float, float]:
    daily = result["daily_metrics"]
    monthly = result["monthly_metrics"]
    return (
        float(daily["daily_shortfall_sum"]),
        float(monthly["monthly_shortfall_sum"]),
        float(daily["daily_target_hit_rate"]),
        float(result["max_drawdown"]),
        float(result["total_return"]),
    )


def evaluate_individual(
    ind,
    toolbox: base.Toolbox,
    folds: list[pd.DataFrame],
    entry_threshold: float,
) -> tuple[float]:
    try:
        func = toolbox.compile(expr=ind)
        scores = []
        trade_count = 0
        for fold in folds:
            desired = func(*build_domain_inputs(fold))
            desired = np.where(np.isfinite(desired), desired, 0.0)
            desired = np.clip(desired, -100.0, 100.0)
            desired = np.where(np.abs(desired) >= entry_threshold, desired, 0.0)
            result = robust_rr_backtest(
                fold,
                desired,
                PRIMARY_PAIR,
            )["primary"]
            daily = result["daily_metrics"]
            monthly = result["monthly_metrics"]

            if len(daily["daily_returns"]) < 20:
                return (1e6,)
            if result["max_drawdown"] < -0.40 or daily["worst_day"] <= -0.05:
                return (1e6,)

            fold_score = 0.0
            fold_score += daily["daily_shortfall_sum"] * 110.0
            fold_score += monthly["monthly_shortfall_sum"] * 55.0
            fold_score += max(0.0, DAILY_TARGET_PCT - daily["avg_daily_return"]) * 180.0
            fold_score += max(0.0, -result["total_return"]) * 90.0
            fold_score += abs(result["max_drawdown"]) * 40.0
            fold_score += max(0.0, -result["daily_metrics"]["cvar"]) * 30.0
            fold_score += max(0.0, 0.10 - result["daily_metrics"]["daily_win_rate"]) * 20.0
            fold_score += max(0.0, 0.15 - result["daily_metrics"]["daily_target_hit_rate"]) * 60.0
            fold_score += max(0.0, entry_threshold - 1.0) * 0.5
            scores.append(fold_score)
            trade_count += int(result["n_trades"])

        score = float(np.mean(scores))
        score += max(0.0, 8.0 - (trade_count / max(len(folds), 1))) * 5.0
        score += len(ind) * 0.015
        return (score,)
    except Exception:
        return (1e6,)


def backtest_individual_on_slice(
    ind,
    toolbox: base.Toolbox,
    df_slice: pd.DataFrame,
    label: str,
    entry_threshold: float,
) -> dict[str, Any]:
    func = toolbox.compile(expr=ind)
    desired = func(*build_domain_inputs(df_slice))
    desired = np.where(np.isfinite(desired), desired, 0.0)
    desired = np.clip(desired, -100.0, 100.0)
    desired = np.where(np.abs(desired) >= entry_threshold, desired, 0.0)
    result = robust_rr_backtest(df_slice, desired, PRIMARY_PAIR)["primary"]

    print(f"\n=== {label} ===")
    print(f"  Return:       {result['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Trades:       {result['n_trades']}")
    print(f"  Win Rate:     {result.get('win_rate', 0.0)*100:.1f}%")
    print(f"  Reward R:     1:{result.get('reward_multiple', DEFAULT_REWARD_MULTIPLE):.1f}")
    print(f"  Final Equity: ${result['final_equity']:,.2f}")
    daily = result["daily_metrics"]
    monthly = result["monthly_metrics"]
    print(f"  Avg Daily:    {daily['avg_daily_return']*100:+.2f}%")
    print(f"  Daily Hit:    {daily['daily_target_hit_rate']*100:.1f}%")
    print(f"  Worst Day:    {daily['worst_day']*100:.2f}%")
    if monthly["n_months"] > 0:
        print(f"  Monthly Hit:  {monthly['monthly_target_hit_rate']*100:.1f}%")
        print(f"  Avg Month:    {monthly['avg_monthly_return']*100:+.2f}%")
    return result


def run_evolution(
    train_df: pd.DataFrame,
    toolbox: base.Toolbox,
    pop_size: int,
    n_gen: int,
    fold_days: int,
    step_days: int,
    entry_threshold: float,
) -> tools.HallOfFame:
    folds = build_folds(train_df, fold_days=fold_days, step_days=step_days)
    if not folds:
        raise RuntimeError("No training folds available")

    def evaluate(ind):
        return evaluate_individual(ind, toolbox, folds, entry_threshold)

    toolbox.register("evaluate", evaluate)
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10, similar=lambda a, b: a == b)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    start_ts = time.time()
    print(f"\nEvolution: pop={pop_size}, gen={n_gen}, cx={P_CX}, mut={P_MUT}")
    print(f"{'Gen':>4} | {'Min Fitness':>12} | {'Avg Fitness':>12} | {'Time':>8}")
    print("-" * 54)

    for gen in range(1, n_gen + 1):
        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        hof.update(population)

        offspring = list(map(toolbox.clone, toolbox.select(population, len(population))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mutant in offspring:
            if random.random() < P_MUT:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        population[:] = offspring
        invalid = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        record = stats.compile(population)
        elapsed = time.time() - start_ts
        print(f"{gen:4d} | {record['min']:12.6f} | {record['avg']:12.6f} | {elapsed:7.1f}s")

    print(f"\nEvolution complete: {time.time() - start_ts:.1f}s")
    print(f"Evaluations: {sum(1 for ind in population if ind.fitness.valid):,}")
    print(f"Best fitness: {hof[0].fitness.values[0]:.6f} (tree: {len(hof[0])} nodes)")
    return hof


def select_best_on_validation(
    hof: tools.HallOfFame,
    toolbox: base.Toolbox,
    val_df: pd.DataFrame,
    fold_days: int,
    step_days: int,
    entry_threshold: float,
) -> tuple[Any, dict[str, Any]]:
    folds = build_folds(val_df, fold_days=fold_days, step_days=step_days)
    if not folds:
        folds = [val_df]

    best = None
    best_score = math.inf
    best_payload: dict[str, Any] | None = None
    for i, ind in enumerate(hof, start=1):
        score = evaluate_individual(ind, toolbox, folds, entry_threshold)[0]
        print(f"  #{i}: fitness={score:.6f}, size={len(ind)} nodes")
        if score < best_score:
            best_score = score
            best = ind
            best_payload = {
                "rank": i,
                "validation_score": float(score),
            }
    if best is None or best_payload is None:
        raise RuntimeError("No valid individual survived validation")
    print(f"Best: #{best_payload['rank']} (val fitness: {best_payload['validation_score']:.6f})")
    return best, best_payload


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not hasattr(creator, "FitnessDomainGP"):
        creator.create("FitnessDomainGP", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "IndividualDomainGP"):
        creator.create("IndividualDomainGP", gp.PrimitiveTree, fitness=creator.FitnessDomainGP)

    pset = setup_pset()
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=MAX_DEPTH)
    toolbox.register("individual", tools.initIterate, creator.IndividualDomainGP, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_LEN))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_LEN))

    print("=" * 78)
    print("  Domain-Primitive / Grammar-Guided GP Crypto Strategy")
    print("=" * 78)

    print("\n[Phase 1] Data Loading")
    df_all = load_all_pairs()
    domain_df = build_domain_feature_frame(df_all)
    train_df = domain_df.loc[TRAIN_START:TRAIN_END].copy()
    val_df = domain_df.loc[VAL_START:VAL_END].copy()
    test_df = domain_df.loc[TEST_START:TEST_END].copy()
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} bars")

    print("\n[Phase 2] GP Evolution")
    hof = run_evolution(
        train_df=train_df,
        toolbox=toolbox,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        fold_days=args.fold_days,
        step_days=args.step_days,
        entry_threshold=args.entry_threshold,
    )

    print("\n[Phase 3] Validation")
    best, val_pick = select_best_on_validation(
        hof,
        toolbox,
        val_df,
        fold_days=args.fold_days,
        step_days=args.step_days,
        entry_threshold=args.entry_threshold,
    )

    print("\n[Phase 4] Out-of-Sample Test")
    test_result = backtest_individual_on_slice(best, toolbox, test_df, "TEST (Domain GP)", args.entry_threshold)

    print("\n[Phase 5] Full-Period Backtest")
    full_result = backtest_individual_on_slice(
        best,
        toolbox,
        domain_df,
        "FULL PERIOD (Domain GP)",
        args.entry_threshold,
    )

    model_path = Path(args.model_out)
    meta_path = Path(args.meta_out)
    with open(model_path, "wb") as f:
        dill.dump(best, f)

    meta = {
        "algorithm": "domain_primitive_grammar_guided_gp",
        "pairs": PAIRS,
        "primary_pair": PRIMARY_PAIR,
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period": f"{VAL_START} ~ {VAL_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "fold_days": args.fold_days,
        "step_days": args.step_days,
        "entry_threshold": args.entry_threshold,
        "feature_count": len(domain_feature_names()),
        "feature_names": domain_feature_names(),
        "tree_size": len(best),
        "fitness": float(best.fitness.values[0]),
        "validation_pick": val_pick,
        "test_return": float(test_result["total_return"]),
        "test_sharpe": float(test_result["sharpe"]),
        "test_max_dd": float(test_result["max_drawdown"]),
        "test_daily_hit_rate": float(test_result["daily_metrics"]["daily_target_hit_rate"]),
        "full_return": float(full_result["total_return"]),
        "full_max_dd": float(full_result["max_drawdown"]),
        "expression": str(best),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(json_ready(meta), f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"\nGP tree expression:\n  {best}")


if __name__ == "__main__":
    main()
