#!/usr/bin/env python3
"""Genetic-algorithm search for a long/short daily rotation strategy.

The strategy keeps the current daily-rotation structure but adds explicit
bear-market short exposure:

- Uptrend: go long the strongest assets.
- Downtrend: short the weakest assets.
- Sideways: stay flat.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from deap import base, creator, tools

from backtest_cash_filtered_rotation import build_daily_close, json_ready
from gp_crypto_evolution import (
    COMMISSION_PCT,
    DATA_DIR,
    INITIAL_CASH,
    MODELS_DIR,
    PAIRS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    load_all_pairs,
    summarize_monthly_returns,
    summarize_period_returns,
)

DAY_FACTOR = np.sqrt(365.25)
RNG_SEED = 42
EPSILON = 1e-12

FAST_OPTIONS = [2, 3, 5, 7, 10]
SLOW_OPTIONS = [8, 10, 14, 21, 28]
TOP_N_OPTIONS = [1, 2]
VOL_WINDOW_OPTIONS = [3, 5, 8, 10]
TARGET_VOL_OPTIONS = [0.4, 0.6, 0.8, 1.0, 1.2]
LONG_REGIME_OPTIONS = [0.0, 0.01, 0.02, 0.04, 0.06]
SHORT_REGIME_OPTIONS = [0.0, 0.01, 0.02, 0.04, 0.06]
LONG_BREADTH_OPTIONS = [0.35, 0.50, 0.65, 0.80]
SHORT_BREADTH_OPTIONS = [0.65, 0.50, 0.35, 0.20]
GROSS_CAP_OPTIONS = [1.0, 1.5, 2.0, 2.5, 3.0]
SHORT_VOL_MULT_OPTIONS = [0.75, 1.0, 1.25]
GENE_OPTION_COUNTS = [
    len(FAST_OPTIONS),
    len(SLOW_OPTIONS),
    len(TOP_N_OPTIONS),
    len(VOL_WINDOW_OPTIONS),
    len(TARGET_VOL_OPTIONS),
    len(LONG_REGIME_OPTIONS),
    len(SHORT_REGIME_OPTIONS),
    len(LONG_BREADTH_OPTIONS),
    len(SHORT_BREADTH_OPTIONS),
    len(GROSS_CAP_OPTIONS),
    len(SHORT_VOL_MULT_OPTIONS),
]


@dataclass(frozen=True)
class LongShortParams:
    lookback_fast: int
    lookback_slow: int
    top_n: int
    vol_window: int
    target_vol_ann: float
    long_regime_threshold: float
    short_regime_threshold: float
    long_breadth_threshold: float
    short_breadth_threshold: float
    gross_cap: float
    short_vol_mult: float
    fee_rate: float = COMMISSION_PCT

    def key(self) -> str:
        return (
            f"f{self.lookback_fast}_s{self.lookback_slow}_n{self.top_n}_"
            f"vw{self.vol_window}_tv{self.target_vol_ann:.2f}_"
            f"lrt{self.long_regime_threshold:.2f}_srt{self.short_regime_threshold:.2f}_"
            f"lb{self.long_breadth_threshold:.2f}_sb{self.short_breadth_threshold:.2f}_"
            f"gc{self.gross_cap:.2f}_sv{self.short_vol_mult:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a long/short market-regime rotation strategy with a genetic algorithm.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "ga_long_short_rotation_summary.json"),
    )
    parser.add_argument(
        "--curve-out",
        default=str(MODELS_DIR / "ga_long_short_rotation_curve.csv"),
    )
    parser.add_argument(
        "--weights-out",
        default=str(MODELS_DIR / "ga_long_short_rotation_weights.csv"),
    )
    parser.add_argument(
        "--selection-out",
        default=str(MODELS_DIR / "ga_long_short_rotation_selection.csv"),
    )
    parser.add_argument(
        "--daily-out",
        default=str(MODELS_DIR / "ga_long_short_rotation_daily_returns.csv"),
    )
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=18)
    parser.add_argument("--hof-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--reselect-days", type=int, default=14)
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh Binance cache before backtesting.",
    )
    return parser.parse_args()


def decode_individual(individual: list[int]) -> LongShortParams:
    fast = FAST_OPTIONS[individual[0] % len(FAST_OPTIONS)]
    valid_slow = [value for value in SLOW_OPTIONS if value > fast]
    slow = valid_slow[individual[1] % len(valid_slow)]
    return LongShortParams(
        lookback_fast=fast,
        lookback_slow=slow,
        top_n=TOP_N_OPTIONS[individual[2] % len(TOP_N_OPTIONS)],
        vol_window=VOL_WINDOW_OPTIONS[individual[3] % len(VOL_WINDOW_OPTIONS)],
        target_vol_ann=TARGET_VOL_OPTIONS[individual[4] % len(TARGET_VOL_OPTIONS)],
        long_regime_threshold=LONG_REGIME_OPTIONS[individual[5] % len(LONG_REGIME_OPTIONS)],
        short_regime_threshold=SHORT_REGIME_OPTIONS[individual[6] % len(SHORT_REGIME_OPTIONS)],
        long_breadth_threshold=LONG_BREADTH_OPTIONS[individual[7] % len(LONG_BREADTH_OPTIONS)],
        short_breadth_threshold=SHORT_BREADTH_OPTIONS[individual[8] % len(SHORT_BREADTH_OPTIONS)],
        gross_cap=GROSS_CAP_OPTIONS[individual[9] % len(GROSS_CAP_OPTIONS)],
        short_vol_mult=SHORT_VOL_MULT_OPTIONS[individual[10] % len(SHORT_VOL_MULT_OPTIONS)],
    )


def build_candidate_individual() -> list[int]:
    return [random.randrange(count) for count in GENE_OPTION_COUNTS]


def load_daily_close(refresh_cache: bool = False) -> tuple[pd.DataFrame, list[str]]:
    if refresh_cache:
        df_all = load_all_pairs(refresh_cache=True)
        close = build_daily_close(df_all).loc[TRAIN_START:TEST_END].copy()
        return close, ["refreshed cache via load_all_pairs()"]

    parts: list[pd.Series] = []
    sources: list[str] = []
    for pair in PAIRS:
        snapshot_path = DATA_DIR / f"{pair}_5m_{TRAIN_START}_{TEST_END}.csv"
        canonical_path = DATA_DIR / f"{pair}_5m.csv"
        source_path = snapshot_path if snapshot_path.exists() else canonical_path

        df = pd.read_csv(source_path, usecols=["open_time", "close"])
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna().set_index("open_time").sort_index()
        df = df.loc[TRAIN_START:TEST_END]
        parts.append(df["close"].resample("1D").last().rename(pair))
        sources.append(str(source_path.name))

    close = pd.concat(parts, axis=1).dropna().sort_index()
    return close, sources


def mutate_individual(individual: list[int], indpb: float = 0.25) -> tuple[list[int]]:
    for i, count in enumerate(GENE_OPTION_COUNTS):
        if random.random() < indpb:
            current = individual[i]
            choices = [value for value in range(count) if value != current]
            if choices:
                individual[i] = random.choice(choices)
    return (individual,)


def build_long_short_target_weights(close: pd.DataFrame, params: LongShortParams) -> pd.DataFrame:
    ret = close.pct_change()
    momentum = 0.60 * close.pct_change(params.lookback_fast) + 0.40 * close.pct_change(params.lookback_slow)
    realized_vol = ret.rolling(params.vol_window).std() * DAY_FACTOR
    btc_regime = 0.50 * close["BTCUSDT"].pct_change(params.lookback_fast) + 0.50 * close["BTCUSDT"].pct_change(
        params.lookback_slow
    )
    breadth = (momentum > 0.0).mean(axis=1)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for ts in close.index:
        regime_value = btc_regime.loc[ts]
        breadth_value = breadth.loc[ts]
        if not np.isfinite(regime_value) or not np.isfinite(breadth_value):
            continue

        side = "flat"
        ranked = momentum.loc[ts].dropna()
        target_vol = params.target_vol_ann
        if float(regime_value) >= params.long_regime_threshold and float(breadth_value) >= params.long_breadth_threshold:
            ranked = ranked[ranked > 0.0].sort_values(ascending=False).head(params.top_n)
            side = "long"
        elif float(regime_value) <= -params.short_regime_threshold and float(breadth_value) <= params.short_breadth_threshold:
            ranked = ranked[ranked < 0.0].sort_values(ascending=True).head(params.top_n)
            side = "short"
            target_vol = params.target_vol_ann * params.short_vol_mult
        else:
            continue

        if ranked.empty:
            continue

        vol = realized_vol.loc[ts, ranked.index].replace(0.0, np.nan).dropna()
        ranked = ranked.loc[vol.index]
        if ranked.empty:
            continue

        raw = (ranked.abs() / vol).replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            continue
        raw_sum = float(raw.sum())
        if raw_sum <= 0.0:
            continue

        w = raw / raw_sum
        port_vol = float(np.sqrt(np.sum(np.square(w.to_numpy() * vol.to_numpy()))))
        if np.isfinite(port_vol) and port_vol > 1e-8:
            w = w * min(target_vol / port_vol, params.gross_cap / max(float(w.sum()), 1e-8))

        gross = float(w.sum())
        if gross > params.gross_cap and gross > 1e-8:
            w = w * (params.gross_cap / gross)

        signed = -w if side == "short" else w
        weights.loc[ts, signed.index] = signed

    return weights


def summarize_long_short_backtest(
    net_ret: pd.Series,
    weights: pd.DataFrame,
    initial_cash: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if net_ret.empty:
        curve = pd.DataFrame(
            columns=["time", "equity", "net_return", "gross_leverage", "net_exposure", "long_leverage", "short_leverage", "turnover"]
        )
        daily_metrics = summarize_period_returns(np.asarray([], dtype="float64"))
        monthly_metrics = summarize_monthly_returns(np.asarray([], dtype="float64"), pd.DatetimeIndex([]))
        return curve, {
            "total_return": 0.0,
            "final_equity": initial_cash,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "avg_gross_leverage": 0.0,
            "avg_net_exposure": 0.0,
            "avg_turnover": 0.0,
            "active_ratio": 0.0,
            "long_active_ratio": 0.0,
            "short_active_ratio": 0.0,
            "rebalances": 0,
            "daily_metrics": daily_metrics,
            "monthly_metrics": monthly_metrics,
        }

    gross_leverage = weights.abs().sum(axis=1)
    long_leverage = weights.clip(lower=0.0).sum(axis=1)
    short_leverage = weights.clip(upper=0.0).abs().sum(axis=1)
    net_exposure = weights.sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))

    equity = float(initial_cash) * (1.0 + net_ret).cumprod()
    curve = pd.DataFrame(
        {
            "time": net_ret.index,
            "equity": equity.to_numpy(dtype="float64"),
            "net_return": net_ret.to_numpy(dtype="float64"),
            "gross_leverage": gross_leverage.to_numpy(dtype="float64"),
            "net_exposure": net_exposure.to_numpy(dtype="float64"),
            "long_leverage": long_leverage.to_numpy(dtype="float64"),
            "short_leverage": short_leverage.to_numpy(dtype="float64"),
            "turnover": turnover.to_numpy(dtype="float64"),
        }
    )
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0

    if len(net_ret) > 1 and net_ret.std() > 1e-12:
        sharpe = float(net_ret.mean() / net_ret.std() * DAY_FACTOR)
    else:
        sharpe = 0.0

    daily_ret = net_ret.to_numpy(dtype="float64")
    daily_metrics = summarize_period_returns(daily_ret)
    monthly_metrics = summarize_monthly_returns(daily_ret, pd.DatetimeIndex(net_ret.index))

    return curve, {
        "total_return": float(equity.iloc[-1] / initial_cash - 1.0),
        "final_equity": float(equity.iloc[-1]),
        "max_drawdown": float(curve["drawdown"].min()),
        "sharpe": sharpe,
        "avg_gross_leverage": float(gross_leverage.mean()),
        "avg_net_exposure": float(net_exposure.mean()),
        "avg_turnover": float(turnover.mean()),
        "active_ratio": float((gross_leverage > EPSILON).mean()),
        "long_active_ratio": float((long_leverage > EPSILON).mean()),
        "short_active_ratio": float((short_leverage > EPSILON).mean()),
        "rebalances": int((turnover > 0.0).sum()),
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
    }


def evaluate_target_weights_long_short(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    start: str,
    end: str,
    initial_cash: float = INITIAL_CASH,
    fee_rate: float = COMMISSION_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work_close = close.loc[start:end].copy()
    work_target = target_weights.loc[start:end].copy()
    if work_close.empty:
        curve, metrics = summarize_long_short_backtest(pd.Series(dtype="float64"), pd.DataFrame(), initial_cash)
        return pd.DataFrame(), curve, metrics

    daily_ret = work_close.pct_change().fillna(0.0)
    weights = work_target.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    net_ret = (weights * daily_ret).sum(axis=1) - turnover * fee_rate
    curve, metrics = summarize_long_short_backtest(net_ret, weights, initial_cash)
    weights_out = weights.copy()
    weights_out.insert(0, "time", weights_out.index)
    return weights_out.reset_index(drop=True), curve, metrics


def run_backtest_long_short(
    close: pd.DataFrame,
    params: LongShortParams,
    start: str,
    end: str,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    target_weights = build_long_short_target_weights(close, params)
    return evaluate_target_weights_long_short(
        close,
        target_weights,
        start,
        end,
        initial_cash=initial_cash,
        fee_rate=params.fee_rate,
    )


def train_score(metrics: dict[str, Any]) -> float:
    balance_penalty = abs(metrics["short_active_ratio"] - metrics["long_active_ratio"]) * 12.0
    score = (
        metrics["total_return"] * 120.0
        + metrics["sharpe"] * 10.0
        + metrics["daily_metrics"]["daily_target_hit_rate"] * 10.0
        + metrics["monthly_metrics"]["monthly_target_hit_rate"] * 8.0
        - abs(metrics["max_drawdown"]) * 140.0
        - metrics["daily_metrics"]["daily_shortfall_mean"] * 120.0
        - metrics["avg_gross_leverage"] * 10.0
        - metrics["avg_turnover"] * 8.0
        + metrics["active_ratio"] * 6.0
        + metrics["short_active_ratio"] * 4.0
        + metrics["long_active_ratio"] * 4.0
        - balance_penalty
    )
    if metrics["total_return"] <= 0.0:
        score -= 40.0
    if metrics["max_drawdown"] < -0.30:
        score -= 30.0
    if metrics["avg_gross_leverage"] > 1.5:
        score -= (metrics["avg_gross_leverage"] - 1.5) * 25.0
    if metrics["short_active_ratio"] < 0.05:
        score -= 25.0
    if metrics["long_active_ratio"] < 0.05:
        score -= 25.0
    return float(score)


def selection_score(train_metrics: dict[str, Any], val_metrics: dict[str, Any]) -> float:
    balance_penalty = abs(val_metrics["short_active_ratio"] - val_metrics["long_active_ratio"]) * 14.0
    score = (
        val_metrics["total_return"] * 140.0
        + val_metrics["sharpe"] * 12.0
        + val_metrics["daily_metrics"]["daily_target_hit_rate"] * 12.0
        + val_metrics["monthly_metrics"]["monthly_target_hit_rate"] * 8.0
        - abs(val_metrics["max_drawdown"]) * 170.0
        - val_metrics["daily_metrics"]["daily_shortfall_mean"] * 120.0
        - val_metrics["avg_gross_leverage"] * 14.0
        - val_metrics["avg_turnover"] * 10.0
        + val_metrics["active_ratio"] * 4.0
        + val_metrics["short_active_ratio"] * 3.0
        + val_metrics["long_active_ratio"] * 3.0
        - balance_penalty
        + train_metrics["total_return"] * 20.0
        + train_metrics["sharpe"] * 3.0
        - abs(train_metrics["max_drawdown"]) * 20.0
    )
    if val_metrics["total_return"] <= 0.0:
        score -= 60.0
    if val_metrics["max_drawdown"] < -0.25:
        score -= 40.0
    if val_metrics["avg_gross_leverage"] > 1.5:
        score -= (val_metrics["avg_gross_leverage"] - 1.5) * 30.0
    if val_metrics["short_active_ratio"] < 0.05:
        score -= 25.0
    if val_metrics["long_active_ratio"] < 0.05:
        score -= 25.0
    return float(score)


def select_walkforward_params_for_day(
    close: pd.DataFrame,
    target_cache: list[tuple[LongShortParams, pd.DataFrame]],
    selection_day: pd.Timestamp,
    train_days: int = 60,
    val_days: int = 30,
    initial_cash: float = INITIAL_CASH,
) -> dict[str, Any]:
    history = close.index[close.index < selection_day]
    if len(history) < train_days + val_days:
        return {
            "status": "insufficient_history",
            "selection_day": pd.Timestamp(selection_day),
        }

    train_start = history[-(train_days + val_days)]
    train_end = history[-(val_days + 1)]
    val_start = history[-val_days]
    val_end = history[-1]

    best_params = None
    best_target = None
    best_train = None
    best_val = None
    best_score = -1e18

    for params, target_weights in target_cache:
        _, _, train_metrics = evaluate_target_weights_long_short(
            close,
            target_weights,
            str(train_start.date()),
            str(train_end.date()),
            initial_cash=initial_cash,
            fee_rate=params.fee_rate,
        )
        _, _, val_metrics = evaluate_target_weights_long_short(
            close,
            target_weights,
            str(val_start.date()),
            str(val_end.date()),
            initial_cash=initial_cash,
            fee_rate=params.fee_rate,
        )
        score = selection_score(train_metrics, val_metrics)
        if score > best_score:
            best_score = score
            best_params = params
            best_target = target_weights
            best_train = train_metrics
            best_val = val_metrics

    assert best_params is not None and best_target is not None
    return {
        "status": "ok",
        "selection_day": pd.Timestamp(selection_day),
        "best_params": best_params,
        "best_target_weights": best_target,
        "best_score": float(best_score),
        "train_metrics": best_train,
        "val_metrics": best_val,
    }


def run_walkforward_pool_selection(
    close: pd.DataFrame,
    params_list: list[LongShortParams],
    start: str,
    end: str,
    reselect_days: int = 14,
    train_days: int = 60,
    val_days: int = 30,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int]]:
    target_cache = [(params, build_long_short_target_weights(close, params)) for params in params_list]
    eval_days = close.loc[start:end].index

    scheduled_rows: list[pd.Series] = []
    selection_rows: list[dict[str, Any]] = []
    selection_counts: dict[str, int] = {}
    i = 0

    while i < len(eval_days):
        block_days = eval_days[i:i + reselect_days]
        block_start = block_days[0]
        selection = select_walkforward_params_for_day(
            close,
            target_cache,
            block_start,
            train_days=train_days,
            val_days=val_days,
            initial_cash=initial_cash,
        )
        if selection["status"] != "ok":
            for day in block_days:
                scheduled_rows.append(pd.Series(0.0, index=close.columns, name=day))
            selection_rows.append(
                {
                    "selection_date": str(block_start.date()),
                    "apply_until": str(block_days[-1].date()),
                    "status": "insufficient_history",
                }
            )
            i += reselect_days
            continue

        best_params = selection["best_params"]
        best_target = selection["best_target_weights"]
        best_train = selection["train_metrics"]
        best_val = selection["val_metrics"]
        best_score = selection["best_score"]
        selection_counts[best_params.key()] = selection_counts.get(best_params.key(), 0) + 1

        for day in block_days:
            scheduled_rows.append(best_target.loc[day].rename(day))

        selection_rows.append(
            {
                "selection_date": str(block_start.date()),
                "apply_until": str(block_days[-1].date()),
                "status": "ok",
                "score": float(best_score),
                **asdict(best_params),
                "train_return": float(best_train["total_return"]),
                "train_max_drawdown": float(best_train["max_drawdown"]),
                "train_short_active_ratio": float(best_train["short_active_ratio"]),
                "val_return": float(best_val["total_return"]),
                "val_max_drawdown": float(best_val["max_drawdown"]),
                "val_short_active_ratio": float(best_val["short_active_ratio"]),
                "val_daily_hit": float(best_val["daily_metrics"]["daily_target_hit_rate"]),
            }
        )
        i += reselect_days

    scheduled_weights = pd.DataFrame(scheduled_rows).fillna(0.0)
    scheduled_weights.index = eval_days
    weights_out, curve, metrics = evaluate_target_weights_long_short(
        close,
        scheduled_weights,
        start,
        end,
        initial_cash=initial_cash,
        fee_rate=COMMISSION_PCT,
    )
    selection_df = pd.DataFrame(selection_rows)
    return selection_df, weights_out, curve, metrics, selection_counts


def register_creator_types() -> None:
    if not hasattr(creator, "LongShortFitnessMax"):
        creator.create("LongShortFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "LongShortIndividual"):
        creator.create("LongShortIndividual", list, fitness=creator.LongShortFitnessMax)


def pick_candidate_pool(
    population: list[Any],
    hof: tools.HallOfFame,
    limit: int,
) -> list[LongShortParams]:
    ranked = list(hof) + sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
    pool: list[LongShortParams] = []
    seen: set[str] = set()
    for individual in ranked:
        params = decode_individual(list(individual))
        key = params.key()
        if key in seen:
            continue
        seen.add(key)
        pool.append(params)
        if len(pool) >= limit:
            break
    return pool


def extract_stage_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "total_return": float(metrics["total_return"]),
        "final_equity": float(metrics["final_equity"]),
        "sharpe": float(metrics["sharpe"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "avg_gross_leverage": float(metrics["avg_gross_leverage"]),
        "avg_net_exposure": float(metrics["avg_net_exposure"]),
        "avg_turnover": float(metrics["avg_turnover"]),
        "active_ratio": float(metrics["active_ratio"]),
        "long_active_ratio": float(metrics["long_active_ratio"]),
        "short_active_ratio": float(metrics["short_active_ratio"]),
        "rebalances": int(metrics["rebalances"]),
        "daily_metrics": metrics["daily_metrics"],
        "monthly_metrics": metrics["monthly_metrics"],
    }


def load_current_strategy_reference() -> dict[str, Any] | None:
    path = MODELS_DIR / "rotation_target_050_summary.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    try:
        stage = data["stages"]["levered_hybrid"]
    except KeyError:
        return None
    return {
        "strategy_class": data.get("strategy_class"),
        "leverage": data.get("leverage"),
        "validation": {
            "total_return": stage["validation"]["total_return"],
            "sharpe": stage["validation"]["sharpe"],
            "max_drawdown": stage["validation"]["max_drawdown"],
            "avg_daily_return": stage["validation"]["daily_metrics"]["avg_daily_return"],
            "daily_target_hit_rate": stage["validation"]["daily_metrics"]["daily_target_hit_rate"],
            "n_trades": stage["validation"]["n_trades"],
        },
        "test": {
            "total_return": stage["test"]["total_return"],
            "sharpe": stage["test"]["sharpe"],
            "max_drawdown": stage["test"]["max_drawdown"],
            "avg_daily_return": stage["test"]["daily_metrics"]["avg_daily_return"],
            "daily_target_hit_rate": stage["test"]["daily_metrics"]["daily_target_hit_rate"],
            "n_trades": stage["test"]["n_trades"],
        },
        "oos": {
            "total_return": stage["oos"]["total_return"],
            "sharpe": stage["oos"]["sharpe"],
            "max_drawdown": stage["oos"]["max_drawdown"],
            "avg_daily_return": stage["oos"]["daily_metrics"]["avg_daily_return"],
            "daily_target_hit_rate": stage["oos"]["daily_metrics"]["daily_target_hit_rate"],
            "n_trades": stage["oos"]["n_trades"],
        },
    }


def build_comparison(new_summary: dict[str, Any], current_reference: dict[str, Any] | None) -> dict[str, Any] | None:
    if current_reference is None:
        return None
    comparison: dict[str, Any] = {}
    for stage_key, current_stage in (
        ("validation", current_reference["validation"]),
        ("test", current_reference["test"]),
        ("oos", current_reference["oos"]),
    ):
        new_stage = new_summary["stages"][stage_key]
        comparison[stage_key] = {
            "new_total_return": new_stage["total_return"],
            "current_total_return": current_stage["total_return"],
            "return_delta": float(new_stage["total_return"] - current_stage["total_return"]),
            "new_sharpe": new_stage["sharpe"],
            "current_sharpe": current_stage["sharpe"],
            "sharpe_delta": float(new_stage["sharpe"] - current_stage["sharpe"]),
            "new_max_drawdown": new_stage["max_drawdown"],
            "current_max_drawdown": current_stage["max_drawdown"],
            "drawdown_delta": float(new_stage["max_drawdown"] - current_stage["max_drawdown"]),
            "new_avg_daily_return": new_stage["daily_metrics"]["avg_daily_return"],
            "current_avg_daily_return": current_stage["avg_daily_return"],
            "avg_daily_return_delta": float(new_stage["daily_metrics"]["avg_daily_return"] - current_stage["avg_daily_return"]),
            "new_daily_target_hit_rate": new_stage["daily_metrics"]["daily_target_hit_rate"],
            "current_daily_target_hit_rate": current_stage["daily_target_hit_rate"],
            "daily_target_hit_rate_delta": float(
                new_stage["daily_metrics"]["daily_target_hit_rate"] - current_stage["daily_target_hit_rate"]
            ),
        }
    return comparison


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  GA Long/Short Rotation Search")
    print("=" * 80)
    print(f"Population: {args.population} | Generations: {args.generations} | Seed: {args.seed}")

    print("\n[Phase 1] Load Data")
    close, data_sources = load_daily_close(refresh_cache=args.refresh_cache)
    print(f"  Daily bars: {len(close)}")
    if not close.empty:
        print(f"  Date range: {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"  Sources: {', '.join(data_sources)}")

    train_metrics_cache: dict[str, dict[str, Any]] = {}

    def evaluate_individual(individual: list[int]) -> tuple[float]:
        params = decode_individual(individual)
        key = params.key()
        if key not in train_metrics_cache:
            _, _, metrics = run_backtest_long_short(close, params, TRAIN_START, TRAIN_END)
            train_metrics_cache[key] = metrics
        return (train_score(train_metrics_cache[key]),)

    print("\n[Phase 2] GA Search")
    register_creator_types()
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.LongShortIndividual, build_candidate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual, indpb=0.25)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=args.population)
    hof = tools.HallOfFame(args.hof_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    for generation in range(args.generations):
        invalid = [individual for individual in population if not individual.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness

        hof.update(population)
        record = stats.compile(population)
        print(
            f"  Gen {generation + 1:02d}/{args.generations}: "
            f"avg={record['avg']:.2f} max={record['max']:.2f} min={record['min']:.2f}"
        )

        if generation == args.generations - 1:
            break

        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.25:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        population = offspring

    candidate_pool = pick_candidate_pool(population, hof, limit=max(args.hof_size * 2, 32))
    print(f"  Candidate pool: {len(candidate_pool)}")

    print("\n[Phase 3] Walk-Forward Validation/Test")
    val_selection, _, _, val_metrics, val_counts = run_walkforward_pool_selection(
        close,
        candidate_pool,
        VAL_START,
        VAL_END,
        reselect_days=args.reselect_days,
        train_days=args.train_days,
        val_days=args.val_days,
    )
    test_selection, _, _, test_metrics, test_counts = run_walkforward_pool_selection(
        close,
        candidate_pool,
        TEST_START,
        TEST_END,
        reselect_days=args.reselect_days,
        train_days=args.train_days,
        val_days=args.val_days,
    )
    oos_selection, oos_weights, oos_curve, oos_metrics, oos_counts = run_walkforward_pool_selection(
        close,
        candidate_pool,
        VAL_START,
        TEST_END,
        reselect_days=args.reselect_days,
        train_days=args.train_days,
        val_days=args.val_days,
    )

    best_train_candidate = decode_individual(list(hof[0]))
    _, _, train_metrics = run_backtest_long_short(close, best_train_candidate, TRAIN_START, TRAIN_END)

    print("\n[Phase 4] Save Artifacts")
    selection_df = pd.concat(
        [
            val_selection.assign(stage="validation"),
            test_selection.assign(stage="test"),
            oos_selection.assign(stage="oos"),
        ],
        ignore_index=True,
    )
    selection_df.to_csv(Path(args.selection_out), index=False)
    oos_weights.to_csv(Path(args.weights_out), index=False)
    oos_curve.to_csv(Path(args.curve_out), index=False)
    if not oos_curve.empty:
        oos_curve[["time", "net_return"]].to_csv(Path(args.daily_out), index=False)

    current_reference = load_current_strategy_reference()
    summary = {
        "strategy_class": "ga_market_regime_long_short_rotation",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "search_config": {
            "population": args.population,
            "generations": args.generations,
            "hof_size": args.hof_size,
            "seed": args.seed,
            "refresh_cache": bool(args.refresh_cache),
        },
        "walkforward_config": {
            "reselect_days": args.reselect_days,
            "train_days": args.train_days,
            "val_days": args.val_days,
        },
        "best_train_candidate": asdict(best_train_candidate),
        "candidate_pool": [asdict(params) for params in candidate_pool],
        "stages": {
            "train_static": extract_stage_metrics(train_metrics),
            "validation": extract_stage_metrics(val_metrics),
            "test": extract_stage_metrics(test_metrics),
            "oos": extract_stage_metrics(oos_metrics),
        },
        "selection_counts": {
            "validation": val_counts,
            "test": test_counts,
            "oos": oos_counts,
        },
        "artifacts": {
            "selection_path": str(Path(args.selection_out)),
            "weights_path": str(Path(args.weights_out)),
            "curve_path": str(Path(args.curve_out)),
            "daily_path": str(Path(args.daily_out)),
        },
        "current_strategy_reference": current_reference,
    }
    summary["comparison_vs_current"] = build_comparison(summary, current_reference)

    with open(Path(args.summary_out), "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    for label, metrics in (
        ("TRAIN(static)", train_metrics),
        ("VALIDATION", val_metrics),
        ("TEST", test_metrics),
        ("OOS", oos_metrics),
    ):
        daily = metrics["daily_metrics"]
        print(
            f"  {label:<12} "
            f"return={metrics['total_return']*100:+6.2f}% "
            f"sharpe={metrics['sharpe']:+.2f} "
            f"mdd={metrics['max_drawdown']*100:+6.2f}% "
            f"avg_day={daily['avg_daily_return']*100:+.3f}% "
            f"short_days={metrics['short_active_ratio']*100:.1f}%"
        )

    print(f"\nSummary saved: {args.summary_out}")


if __name__ == "__main__":
    main()
