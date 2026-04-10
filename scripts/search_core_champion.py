#!/usr/bin/env python3
"""Unified long-only vs long-short core competition search."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from deap import base, creator, tools

from backtest_cash_filtered_rotation import (
    StrategyParams,
    candidate_params,
    json_ready,
)
from backtest_rotation_target_050 import BEST_CORE_PARAMS
from core_strategy_registry import (
    DEFAULT_CORE_CHAMPION_PATH,
    LONG_ONLY_FAMILY,
    LONG_SHORT_FAMILY,
    ResolvedCoreStrategy,
    build_core_target_weights,
    resolve_core_strategy,
    save_core_artifact,
)
from ga_long_short_rotation import (
    LongShortParams,
    build_long_short_target_weights,
    build_candidate_individual,
    decode_individual,
    load_daily_close,
    mutate_individual,
)
from market_context import load_market_context_dataset
from gp_crypto_evolution import (
    INITIAL_CASH,
    MODELS_DIR,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    summarize_monthly_returns,
    summarize_period_returns,
)

DAY_FACTOR = np.sqrt(365.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a unified core champion across long-only and long-short families.")
    parser.add_argument("--summary-out", default=str(MODELS_DIR / "core_competition_summary.json"))
    parser.add_argument("--artifact-out", default=str(DEFAULT_CORE_CHAMPION_PATH))
    parser.add_argument("--population", type=int, default=60, help="Long-short GA population.")
    parser.add_argument("--generations", type=int, default=14, help="Long-short GA generations.")
    parser.add_argument("--hof-size", type=int, default=18)
    parser.add_argument("--long-short-pool", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--finalists", type=int, default=12)
    parser.add_argument("--fold-days", type=int, default=30)
    parser.add_argument("--fold-step", type=int, default=15)
    parser.add_argument("--base-slippage", type=float, default=0.0002)
    parser.add_argument("--stress-commission-multipliers", default="1.0,1.5,2.0")
    parser.add_argument("--stress-slippage-multipliers", default="1.0,1.5,2.0")
    parser.add_argument("--cpcv-blocks", type=int, default=6)
    parser.add_argument("--cpcv-test-blocks", type=int, default=2)
    parser.add_argument("--cpcv-embargo-days", type=int, default=2)
    parser.add_argument("--corr-window", type=int, default=20)
    parser.add_argument("--corr-state-min-days", type=int, default=10)
    parser.add_argument("--neighbor-count", type=int, default=4)
    parser.add_argument("--refresh-market-context", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def parse_csv_floats(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def empty_metrics(initial_cash: float = INITIAL_CASH) -> tuple[pd.DataFrame, dict[str, Any]]:
    curve = pd.DataFrame(
        columns=[
            "time",
            "equity",
            "net_return",
            "gross_leverage",
            "net_exposure",
            "long_leverage",
            "short_leverage",
            "turnover",
        ]
    )
    daily_metrics = summarize_period_returns(np.asarray([], dtype="float64"))
    monthly_metrics = summarize_monthly_returns(np.asarray([], dtype="float64"), pd.DatetimeIndex([]))
    return curve, {
        "total_return": 0.0,
        "final_equity": float(initial_cash),
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


def compute_portfolio_frame(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    start: str,
    end: str,
    *,
    fee_rate: float,
    slippage_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work_close = close.loc[start:end].copy()
    work_target = target_weights.loc[start:end].copy()
    if work_close.empty:
        return pd.DataFrame(), pd.DataFrame()

    daily_ret = work_close.pct_change().fillna(0.0)
    weights = work_target.shift(1).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    net_ret = (weights * daily_ret).sum(axis=1) - turnover * (float(fee_rate) + float(slippage_rate))

    frame = pd.DataFrame(
        {
            "net_return": net_ret.to_numpy(dtype="float64"),
            "gross_leverage": weights.abs().sum(axis=1).to_numpy(dtype="float64"),
            "net_exposure": weights.sum(axis=1).to_numpy(dtype="float64"),
            "long_leverage": weights.clip(lower=0.0).sum(axis=1).to_numpy(dtype="float64"),
            "short_leverage": weights.clip(upper=0.0).abs().sum(axis=1).to_numpy(dtype="float64"),
            "turnover": turnover.to_numpy(dtype="float64"),
        },
        index=weights.index,
    )
    return frame, weights


def summarize_portfolio_frame(
    portfolio_frame: pd.DataFrame,
    *,
    initial_cash: float = INITIAL_CASH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if portfolio_frame.empty:
        return empty_metrics(initial_cash)

    net_ret = portfolio_frame["net_return"].astype("float64")
    equity = float(initial_cash) * (1.0 + net_ret).cumprod()
    curve = pd.DataFrame(
        {
            "time": net_ret.index,
            "equity": equity.to_numpy(dtype="float64"),
            "net_return": net_ret.to_numpy(dtype="float64"),
            "gross_leverage": portfolio_frame["gross_leverage"].to_numpy(dtype="float64"),
            "net_exposure": portfolio_frame["net_exposure"].to_numpy(dtype="float64"),
            "long_leverage": portfolio_frame["long_leverage"].to_numpy(dtype="float64"),
            "short_leverage": portfolio_frame["short_leverage"].to_numpy(dtype="float64"),
            "turnover": portfolio_frame["turnover"].to_numpy(dtype="float64"),
        }
    )
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0

    if len(net_ret) > 1 and net_ret.std() > 1e-12:
        sharpe = float(net_ret.mean() / net_ret.std() * DAY_FACTOR)
    else:
        sharpe = 0.0

    daily_metrics = summarize_period_returns(net_ret.to_numpy(dtype="float64"))
    monthly_metrics = summarize_monthly_returns(net_ret.to_numpy(dtype="float64"), pd.DatetimeIndex(net_ret.index))
    gross_leverage = portfolio_frame["gross_leverage"]
    long_leverage = portfolio_frame["long_leverage"]
    short_leverage = portfolio_frame["short_leverage"]
    turnover = portfolio_frame["turnover"]
    net_exposure = portfolio_frame["net_exposure"]

    return curve, {
        "total_return": float(equity.iloc[-1] / initial_cash - 1.0),
        "final_equity": float(equity.iloc[-1]),
        "max_drawdown": float(curve["drawdown"].min()),
        "sharpe": sharpe,
        "avg_gross_leverage": float(gross_leverage.mean()),
        "avg_net_exposure": float(net_exposure.mean()),
        "avg_turnover": float(turnover.mean()),
        "active_ratio": float((gross_leverage > 1e-12).mean()),
        "long_active_ratio": float((long_leverage > 1e-12).mean()),
        "short_active_ratio": float((short_leverage > 1e-12).mean()),
        "rebalances": int((turnover > 0.0).sum()),
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
    }


def evaluate_target_weights_generic(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    start: str,
    end: str,
    *,
    initial_cash: float = INITIAL_CASH,
    fee_rate: float,
    slippage_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    portfolio_frame, weights = compute_portfolio_frame(
        close,
        target_weights,
        start,
        end,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    curve, metrics = summarize_portfolio_frame(portfolio_frame, initial_cash=initial_cash)
    weights_out = weights.copy()
    weights_out.insert(0, "time", weights_out.index)
    return weights_out.reset_index(drop=True), curve, metrics


def train_score_generic(metrics: dict[str, Any]) -> float:
    return float(
        metrics["total_return"] * 120.0
        + metrics["sharpe"] * 10.0
        + metrics["daily_metrics"]["daily_target_hit_rate"] * 12.0
        + metrics["monthly_metrics"]["monthly_target_hit_rate"] * 8.0
        - abs(metrics["max_drawdown"]) * 140.0
        - metrics["daily_metrics"]["daily_shortfall_mean"] * 120.0
        - metrics["avg_turnover"] * 10.0
        - max(0.0, metrics["avg_gross_leverage"] - 1.5) * 25.0
        + metrics["active_ratio"] * 4.0
    )


def validation_score_generic(train_metrics: dict[str, Any], val_metrics: dict[str, Any]) -> float:
    score = (
        val_metrics["total_return"] * 140.0
        + val_metrics["sharpe"] * 12.0
        + val_metrics["daily_metrics"]["daily_target_hit_rate"] * 12.0
        + val_metrics["monthly_metrics"]["monthly_target_hit_rate"] * 8.0
        - abs(val_metrics["max_drawdown"]) * 170.0
        - val_metrics["daily_metrics"]["daily_shortfall_mean"] * 120.0
        - val_metrics["avg_turnover"] * 12.0
        - max(0.0, val_metrics["avg_gross_leverage"] - 1.5) * 30.0
        + val_metrics["active_ratio"] * 4.0
        + train_metrics["total_return"] * 18.0
        + train_metrics["sharpe"] * 3.0
        - abs(train_metrics["max_drawdown"]) * 20.0
    )
    if val_metrics["total_return"] <= 0.0:
        score -= 40.0
    if val_metrics["active_ratio"] < 0.05:
        score -= 25.0
    if val_metrics["max_drawdown"] < -0.25:
        score -= 35.0
    return float(score)


def safe_calmar(total_return: float, max_drawdown: float) -> float:
    drawdown = abs(float(max_drawdown))
    if drawdown <= 1e-8:
        return 0.0 if total_return <= 0.0 else 10.0
    return float(np.clip(float(total_return) / drawdown, -10.0, 10.0))


def expected_max_sharpe(trial_count: int) -> float:
    trials = max(1, int(trial_count))
    if trials <= 1:
        return 0.0
    dist = NormalDist()
    euler_gamma = 0.5772156649015329
    a = dist.inv_cdf(1.0 - 1.0 / trials)
    b = dist.inv_cdf(1.0 - 1.0 / (trials * math.e))
    return float((1.0 - euler_gamma) * a + euler_gamma * b)


def compute_dsr_proxy(period_returns: np.ndarray, *, trial_count: int) -> float:
    returns = np.asarray(period_returns, dtype="float64")
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0

    mean = float(returns.mean())
    sharpe = mean / std * DAY_FACTOR
    standardized = (returns - mean) / std
    skew = float(np.mean(np.power(standardized, 3)))
    kurtosis = float(np.mean(np.power(standardized, 4)))
    variance_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * (sharpe**2)
    sr_std = math.sqrt(max(variance_term, 1e-8) / max(len(returns) - 1, 1))
    z_score = (sharpe - expected_max_sharpe(trial_count)) / max(sr_std, 1e-8)
    return float(np.clip(NormalDist().cdf(z_score), 0.0, 1.0))


def build_objective_metrics(candidate: dict[str, Any], *, trial_count: int) -> dict[str, Any]:
    oos = candidate["oos"]
    train = candidate["train"]
    validation = candidate["validation"]
    test = candidate["test"]
    oos_daily = np.asarray(oos["daily_metrics"]["daily_returns"], dtype="float64")
    return {
        "oos_calmar": safe_calmar(oos["total_return"], oos["max_drawdown"]),
        "oos_dsr_proxy": compute_dsr_proxy(oos_daily, trial_count=trial_count),
        "oos_cvar": float(oos["daily_metrics"]["cvar"]),
        "generalization_gap": float(
            max(0.0, train["total_return"] - oos["total_return"])
            + max(0.0, validation["total_return"] - test["total_return"])
        ),
        "return_stability_gap": float(abs(test["total_return"] - oos["total_return"])),
    }


def summarize_regime_stats(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    included = [row for row in states.values() if row.get("days", 0) > 0]
    if not included:
        return {
            "included_states": 0,
            "positive_rate": 0.0,
            "avg_total_return": 0.0,
            "min_total_return": 0.0,
        }

    returns = np.asarray([row["total_return"] for row in included], dtype="float64")
    return {
        "included_states": int(len(included)),
        "positive_rate": float(np.mean(returns > 0.0)),
        "avg_total_return": float(np.mean(returns)),
        "min_total_return": float(np.min(returns)),
    }


LONG_ONLY_PARAM_FIELDS = (
    "lookback_fast",
    "lookback_slow",
    "top_n",
    "vol_window",
    "target_vol_ann",
    "regime_threshold",
    "breadth_threshold",
    "gross_cap",
)
LONG_SHORT_PARAM_FIELDS = (
    "lookback_fast",
    "lookback_slow",
    "top_n",
    "vol_window",
    "target_vol_ann",
    "long_regime_threshold",
    "short_regime_threshold",
    "long_breadth_threshold",
    "short_breadth_threshold",
    "gross_cap",
    "short_vol_mult",
)


def parameter_fields_for_family(family: str) -> tuple[str, ...]:
    if family == LONG_ONLY_FAMILY:
        return LONG_ONLY_PARAM_FIELDS
    return LONG_SHORT_PARAM_FIELDS


def build_parameter_stability(rows: list[dict[str, Any]], *, neighbor_count: int) -> dict[str, dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    stability: dict[str, dict[str, Any]] = {}
    for family, family_rows in by_family.items():
        fields = parameter_fields_for_family(family)
        mins = {field: min(float(item["params"][field]) for item in family_rows) for field in fields}
        maxs = {field: max(float(item["params"][field]) for item in family_rows) for field in fields}

        def distance(left: dict[str, Any], right: dict[str, Any]) -> float:
            diffs: list[float] = []
            for field in fields:
                scale = max(maxs[field] - mins[field], 1e-8)
                diffs.append(abs(float(left["params"][field]) - float(right["params"][field])) / scale)
            return float(np.mean(diffs)) if diffs else 0.0

        for row in family_rows:
            peers = [peer for peer in family_rows if peer["key"] != row["key"]]
            if not peers:
                stability[row["key"]] = {
                    "neighbor_count": 0,
                    "neighbor_positive_rate": 0.0,
                    "neighbor_avg_oos_return": 0.0,
                    "neighbor_min_oos_return": 0.0,
                    "neighbor_oos_std": 0.0,
                    "nearest_distance": 0.0,
                }
                continue

            ranked_peers = sorted(((distance(row, peer), peer) for peer in peers), key=lambda item: item[0])
            chosen = ranked_peers[: max(1, int(neighbor_count))]
            oos_returns = np.asarray([float(peer["oos"]["total_return"]) for _, peer in chosen], dtype="float64")
            stability[row["key"]] = {
                "neighbor_count": int(len(chosen)),
                "neighbor_positive_rate": float(np.mean(oos_returns > 0.0)),
                "neighbor_avg_oos_return": float(np.mean(oos_returns)),
                "neighbor_min_oos_return": float(np.min(oos_returns)),
                "neighbor_oos_std": float(np.std(oos_returns)),
                "nearest_distance": float(chosen[0][0]),
            }
    return stability


PARETO_OBJECTIVES: tuple[tuple[str, bool], ...] = (
    ("oos_total_return", True),
    ("oos_dsr_proxy", True),
    ("cpcv_pass_rate", True),
    ("stress_survival_rate", True),
    ("corr_positive_rate", True),
    ("oos_max_drawdown_abs", False),
    ("oos_cvar_abs", False),
)


def build_pareto_vector(candidate: dict[str, Any]) -> dict[str, float]:
    return {
        "oos_total_return": float(candidate["oos"]["total_return"]),
        "oos_dsr_proxy": float(candidate["objective_metrics"]["oos_dsr_proxy"]),
        "cpcv_pass_rate": float(candidate["cpcv"]["pass_rate"]),
        "stress_survival_rate": float(candidate["stress"]["stress_survival_rate"]),
        "corr_positive_rate": float(candidate["corr_state_robustness"]["summary"]["positive_rate"]),
        "oos_max_drawdown_abs": float(abs(candidate["oos"]["max_drawdown"])),
        "oos_cvar_abs": float(abs(candidate["objective_metrics"]["oos_cvar"])),
    }


def dominates(left: dict[str, Any], right: dict[str, Any], *, eps: float = 1e-12) -> bool:
    better_or_equal = True
    strictly_better = False
    left_vec = left["pareto_vector"]
    right_vec = right["pareto_vector"]
    for name, maximize in PARETO_OBJECTIVES:
        lhs = float(left_vec[name])
        rhs = float(right_vec[name])
        if maximize:
            if lhs + eps < rhs:
                better_or_equal = False
                break
            if lhs > rhs + eps:
                strictly_better = True
        else:
            if lhs > rhs + eps:
                better_or_equal = False
                break
            if lhs + eps < rhs:
                strictly_better = True
    return better_or_equal and strictly_better


def assign_pareto_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    remaining = list(rows)
    fronts: list[list[dict[str, Any]]] = []
    while remaining:
        front: list[dict[str, Any]] = []
        for row in remaining:
            if not any(dominates(other, row) for other in remaining if other["key"] != row["key"]):
                front.append(row)
        fronts.append(front)
        front_keys = {row["key"] for row in front}
        remaining = [row for row in remaining if row["key"] not in front_keys]

    metadata: dict[str, dict[str, Any]] = {}
    for rank, front in enumerate(fronts, start=1):
        distances = {row["key"]: 0.0 for row in front}
        if len(front) <= 2:
            for row in front:
                distances[row["key"]] = math.inf
        else:
            for name, _ in PARETO_OBJECTIVES:
                ordered = sorted(front, key=lambda row: float(row["pareto_vector"][name]))
                min_value = float(ordered[0]["pareto_vector"][name])
                max_value = float(ordered[-1]["pareto_vector"][name])
                distances[ordered[0]["key"]] = math.inf
                distances[ordered[-1]["key"]] = math.inf
                scale = max(max_value - min_value, 1e-8)
                for idx in range(1, len(ordered) - 1):
                    if math.isinf(distances[ordered[idx]["key"]]):
                        continue
                    prev_value = float(ordered[idx - 1]["pareto_vector"][name])
                    next_value = float(ordered[idx + 1]["pareto_vector"][name])
                    distances[ordered[idx]["key"]] += (next_value - prev_value) / scale

        for row in front:
            crowding_distance = float(distances[row["key"]])
            metadata[row["key"]] = {
                "rank": int(rank),
                "is_nondominated": bool(rank == 1),
                "crowding_distance": crowding_distance,
                "crowding_sort_value": float(1e9 if math.isinf(crowding_distance) else crowding_distance),
            }
    return metadata


def promotion_score(candidate: dict[str, Any]) -> float:
    test = candidate["test"]
    oos = candidate["oos"]
    folds = candidate["fold_robustness"]
    cpcv = candidate["cpcv"]
    stress = candidate["stress"]
    objective_metrics = candidate["objective_metrics"]
    regime_summary = candidate["regime_breakdown"]["summary"]
    corr_summary = candidate["corr_state_robustness"]["summary"]
    stability = candidate["parameter_stability"]
    pareto = candidate["pareto"]
    return float(
        candidate["selection_score"]
        + test["total_return"] * 50.0
        + oos["total_return"] * 80.0
        + oos["sharpe"] * 8.0
        - abs(oos["max_drawdown"]) * 60.0
        + objective_metrics["oos_calmar"] * 10.0
        + objective_metrics["oos_dsr_proxy"] * 35.0
        - abs(objective_metrics["oos_cvar"]) * 120.0
        - objective_metrics["generalization_gap"] * 40.0
        - objective_metrics["return_stability_gap"] * 18.0
        + folds["fold_positive_rate"] * 10.0
        + folds["fold_avg_return"] * 80.0
        - abs(folds["fold_min_return"]) * 25.0
        + cpcv["test_positive_rate"] * 14.0
        + cpcv["pass_rate"] * 18.0
        + cpcv["avg_test_return"] * 80.0
        - abs(cpcv["min_test_return"]) * 18.0
        + stress["stress_survival_rate"] * 20.0
        + regime_summary["positive_rate"] * 12.0
        + regime_summary["avg_total_return"] * 40.0
        - abs(regime_summary["min_total_return"]) * 12.0
        + corr_summary["positive_rate"] * 14.0
        + corr_summary["avg_total_return"] * 50.0
        - abs(corr_summary["min_total_return"]) * 14.0
        + stability["neighbor_positive_rate"] * 12.0
        + stability["neighbor_avg_oos_return"] * 60.0
        - abs(stability["neighbor_min_oos_return"]) * 12.0
        - stability["neighbor_oos_std"] * 20.0
        + max(0.0, 5.0 - float(pareto["rank"])) * 6.0
        + min(float(pareto["crowding_sort_value"]), 10.0) * 0.75
    )


def build_day_folds(index: pd.DatetimeIndex, fold_days: int, fold_step: int) -> list[tuple[str, str]]:
    days = pd.DatetimeIndex(index.normalize().unique())
    if len(days) == 0:
        return []
    if len(days) <= fold_days:
        return [(str(days[0].date()), str(days[-1].date()))]
    windows: list[tuple[str, str]] = []
    for start_idx in range(0, len(days) - fold_days + 1, fold_step):
        seg = days[start_idx:start_idx + fold_days]
        windows.append((str(seg[0].date()), str(seg[-1].date())))
    return windows


def summarize_fold_robustness(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    fee_rate: float,
    *,
    fold_days: int,
    fold_step: int,
    slippage_rate: float = 0.0,
) -> dict[str, Any]:
    portfolio_frame, _ = compute_portfolio_frame(
        close,
        target_weights,
        VAL_START,
        TEST_END,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    return summarize_fold_robustness_from_portfolio_frame(
        portfolio_frame,
        fold_days=fold_days,
        fold_step=fold_step,
    )


def evaluate_stress_profiles(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    base_fee_rate: float,
    *,
    base_slippage: float,
    commission_multipliers: list[float],
    slippage_multipliers: list[float],
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for commission_multiplier in commission_multipliers:
        for slippage_multiplier in slippage_multipliers:
            combo = (float(commission_multiplier), float(slippage_multiplier))
            if combo in seen:
                continue
            seen.add(combo)
            effective_fee = float(base_fee_rate) * float(commission_multiplier)
            effective_slippage = float(base_slippage) * float(slippage_multiplier)
            _, _, metrics = evaluate_target_weights_generic(
                close,
                target_weights,
                VAL_START,
                TEST_END,
                fee_rate=effective_fee,
                slippage_rate=effective_slippage,
            )
            passed = bool(metrics["total_return"] > 0.0 and metrics["max_drawdown"] >= -0.30)
            runs.append(
                {
                    "name": f"fee_{commission_multiplier:.2f}x_slip_{slippage_multiplier:.2f}x",
                    "commission_multiplier": float(commission_multiplier),
                    "slippage_multiplier": float(slippage_multiplier),
                    "effective_fee_rate": effective_fee,
                    "effective_slippage_rate": effective_slippage,
                    "passed": passed,
                    "total_return": float(metrics["total_return"]),
                    "sharpe": float(metrics["sharpe"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "avg_daily_return": float(metrics["daily_metrics"]["avg_daily_return"]),
                }
            )
    return {
        "profiles": runs,
        "stress_survival_rate": float(np.mean([1.0 if row["passed"] else 0.0 for row in runs])) if runs else 0.0,
    }


def split_cpcv_blocks(index: pd.DatetimeIndex, n_blocks: int) -> list[pd.DatetimeIndex]:
    days = pd.DatetimeIndex(index.normalize().unique())
    if len(days) == 0:
        return []
    n_blocks = max(1, min(int(n_blocks), len(days)))
    base_size = len(days) // n_blocks
    remainder = len(days) % n_blocks
    blocks: list[pd.DatetimeIndex] = []
    cursor = 0
    for idx in range(n_blocks):
        size = base_size + (1 if idx < remainder else 0)
        seg = days[cursor:cursor + size]
        if len(seg) > 0:
            blocks.append(seg)
        cursor += size
    return blocks


def summarize_cpcv_lite(
    portfolio_frame: pd.DataFrame,
    *,
    n_blocks: int,
    test_blocks: int,
    embargo_days: int,
) -> dict[str, Any]:
    period_frame = portfolio_frame.loc[VAL_START:TEST_END].copy()
    if period_frame.empty:
        return {
            "n_splits": 0,
            "pass_rate": 0.0,
            "test_positive_rate": 0.0,
            "avg_test_return": 0.0,
            "min_test_return": 0.0,
            "avg_test_mdd": 0.0,
            "splits": [],
        }

    blocks = split_cpcv_blocks(period_frame.index, n_blocks)
    split_rows: list[dict[str, Any]] = []
    for combo in itertools.combinations(range(len(blocks)), min(test_blocks, len(blocks))):
        test_days = pd.DatetimeIndex(sorted({day for idx in combo for day in blocks[idx]}))
        test_mask = period_frame.index.normalize().isin(test_days)
        embargo_set: set[pd.Timestamp] = set()
        if embargo_days > 0:
            for idx in combo:
                block = blocks[idx]
                start_day = block[0]
                end_day = block[-1]
                for offset in range(1, embargo_days + 1):
                    embargo_set.add(start_day - pd.Timedelta(days=offset))
                    embargo_set.add(end_day + pd.Timedelta(days=offset))
        train_mask = ~test_mask
        if embargo_set:
            train_mask &= ~period_frame.index.normalize().isin(pd.DatetimeIndex(sorted(embargo_set)))

        test_frame = period_frame.loc[test_mask]
        train_frame = period_frame.loc[train_mask]
        _, test_metrics = summarize_portfolio_frame(test_frame)
        _, train_metrics = summarize_portfolio_frame(train_frame)
        score = validation_score_generic(train_metrics, test_metrics)
        passed = bool(test_metrics["total_return"] > 0.0 and test_metrics["max_drawdown"] >= -0.30)
        split_rows.append(
            {
                "test_blocks": list(combo),
                "train_days": int(len(train_frame)),
                "test_days": int(len(test_frame)),
                "score": float(score),
                "passed": passed,
                "test_total_return": float(test_metrics["total_return"]),
                "test_sharpe": float(test_metrics["sharpe"]),
                "test_max_drawdown": float(test_metrics["max_drawdown"]),
                "train_total_return": float(train_metrics["total_return"]),
            }
        )

    if not split_rows:
        return {
            "n_splits": 0,
            "pass_rate": 0.0,
            "test_positive_rate": 0.0,
            "avg_test_return": 0.0,
            "min_test_return": 0.0,
            "avg_test_mdd": 0.0,
            "splits": [],
        }

    test_returns = np.asarray([row["test_total_return"] for row in split_rows], dtype="float64")
    test_mdds = np.asarray([row["test_max_drawdown"] for row in split_rows], dtype="float64")
    return {
        "n_splits": int(len(split_rows)),
        "pass_rate": float(np.mean([1.0 if row["passed"] else 0.0 for row in split_rows])),
        "test_positive_rate": float(np.mean(test_returns > 0.0)),
        "avg_test_return": float(np.mean(test_returns)),
        "min_test_return": float(np.min(test_returns)),
        "avg_test_mdd": float(np.mean(test_mdds)),
        "splits": split_rows,
    }


def build_btc_regime_labels(
    close_slice: pd.DataFrame,
    *,
    fast_lookback: int = 5,
    slow_lookback: int = 14,
    threshold: float = 0.02,
) -> pd.Series:
    btc_fast = close_slice["BTCUSDT"].pct_change(fast_lookback)
    btc_slow = close_slice["BTCUSDT"].pct_change(slow_lookback)
    regime = 0.5 * btc_fast + 0.5 * btc_slow
    labels = pd.Series("sideways", index=close_slice.index)
    labels.loc[regime >= float(threshold)] = "bull"
    labels.loc[regime <= -float(threshold)] = "bear"
    return labels


def build_regime_breakdown_from_portfolio_frame(
    close_slice: pd.DataFrame,
    portfolio_frame: pd.DataFrame,
    *,
    fast_lookback: int = 5,
    slow_lookback: int = 14,
    threshold: float = 0.02,
) -> dict[str, Any]:
    if close_slice.empty or portfolio_frame.empty:
        return {
            "summary": {
                "included_states": 0,
                "positive_rate": 0.0,
                "avg_total_return": 0.0,
                "min_total_return": 0.0,
            },
            "states": {},
        }

    labels = build_btc_regime_labels(
        close_slice,
        fast_lookback=fast_lookback,
        slow_lookback=slow_lookback,
        threshold=threshold,
    )
    net_ret = portfolio_frame["net_return"]
    states: dict[str, Any] = {}
    for name in ("bull", "bear", "sideways"):
        seg = net_ret.loc[labels == name]
        if seg.empty:
            states[name] = {"days": 0, "total_return": 0.0, "avg_daily_return": 0.0, "win_rate": 0.0}
            continue
        states[name] = {
            "days": int(len(seg)),
            "total_return": float(np.prod(1.0 + seg.to_numpy(dtype="float64")) - 1.0),
            "avg_daily_return": float(seg.mean()),
            "win_rate": float((seg > 0.0).mean()),
        }
    return {
        "summary": summarize_regime_stats(states),
        "states": states,
    }


def summarize_fold_robustness_from_portfolio_frame(
    portfolio_frame: pd.DataFrame,
    *,
    fold_days: int,
    fold_step: int,
) -> dict[str, Any]:
    windows = build_day_folds(pd.DatetimeIndex(portfolio_frame.index), fold_days=fold_days, fold_step=fold_step)
    fold_rows: list[dict[str, Any]] = []
    for start, end in windows:
        fold_frame = portfolio_frame.loc[start:end]
        _, metrics = summarize_portfolio_frame(fold_frame)
        fold_rows.append(
            {
                "start": start,
                "end": end,
                "total_return": float(metrics["total_return"]),
                "sharpe": float(metrics["sharpe"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "avg_daily_return": float(metrics["daily_metrics"]["avg_daily_return"]),
            }
        )

    if not fold_rows:
        return {
            "n_folds": 0,
            "fold_positive_rate": 0.0,
            "fold_avg_return": 0.0,
            "fold_min_return": 0.0,
            "fold_avg_mdd": 0.0,
            "folds": [],
        }

    returns = np.asarray([row["total_return"] for row in fold_rows], dtype="float64")
    mdds = np.asarray([row["max_drawdown"] for row in fold_rows], dtype="float64")
    return {
        "n_folds": int(len(fold_rows)),
        "fold_positive_rate": float(np.mean(returns > 0.0)),
        "fold_avg_return": float(np.mean(returns)),
        "fold_min_return": float(np.min(returns)),
        "fold_avg_mdd": float(np.mean(mdds)),
        "folds": fold_rows,
    }


def build_regime_breakdown(
    close: pd.DataFrame,
    target_weights: pd.DataFrame,
    fee_rate: float,
    *,
    slippage_rate: float,
) -> dict[str, Any]:
    work_close = close.loc[VAL_START:TEST_END].copy()
    portfolio_frame, _ = compute_portfolio_frame(
        close,
        target_weights,
        VAL_START,
        TEST_END,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    if work_close.empty or portfolio_frame.empty:
        return {"summary": summarize_regime_stats({}), "states": {}}
    return build_regime_breakdown_from_portfolio_frame(work_close, portfolio_frame)


def build_corr_state_breakdown(
    close: pd.DataFrame,
    market_context_close: pd.DataFrame,
    portfolio_frame: pd.DataFrame,
    *,
    corr_window: int,
    min_state_days: int,
) -> dict[str, Any]:
    work_close = close.loc[VAL_START:TEST_END].copy()
    if work_close.empty or portfolio_frame.empty:
        return {
            "summary": {
                "included_states": 0,
                "positive_rate": 0.0,
                "avg_total_return": 0.0,
                "min_total_return": 0.0,
                "corr_window": int(corr_window),
            },
            "states": {},
        }

    btc_ret = work_close["BTCUSDT"].pct_change()
    net_ret = portfolio_frame["net_return"]
    context_frame = market_context_close.loc[VAL_START:TEST_END].copy() if not market_context_close.empty else pd.DataFrame()
    context_names = list(context_frame.columns) if not context_frame.empty else []
    use_internal_fallback = not context_names

    if use_internal_fallback:
        alt_cols = [col for col in work_close.columns if col != "BTCUSDT"]
        if alt_cols:
            context_frame = pd.DataFrame({"internal_alt_basket": work_close[alt_cols].mean(axis=1)})
            context_names = ["internal_alt_basket"]

    if not context_names:
        return {
            "summary": {
                "included_states": 0,
                "positive_rate": 0.0,
                "avg_total_return": 0.0,
                "min_total_return": 0.0,
                "corr_window": int(corr_window),
                "contexts_used": [],
                "source_mode": "missing",
            },
            "states": {},
        }

    states_by_context: dict[str, Any] = {}
    flat_states: dict[str, Any] = {}
    for context_name in context_names:
        context_ret = context_frame[context_name].pct_change()
        rolling_corr = btc_ret.rolling(int(corr_window)).corr(context_ret).replace([np.inf, -np.inf], np.nan)
        valid_corr = rolling_corr.dropna()
        if valid_corr.empty:
            states_by_context[context_name] = {
                "summary": {
                    "included_states": 0,
                    "positive_rate": 0.0,
                    "avg_total_return": 0.0,
                    "min_total_return": 0.0,
                    "corr_window": int(corr_window),
                    "low_cut": 0.0,
                    "high_cut": 0.0,
                },
                "states": {},
            }
            continue

        low_cut = float(valid_corr.quantile(0.33))
        high_cut = float(valid_corr.quantile(0.67))
        labels = pd.Series("mid_corr", index=rolling_corr.index)
        labels.loc[rolling_corr <= low_cut] = "low_corr"
        labels.loc[rolling_corr >= high_cut] = "high_corr"
        labels = labels.reindex(portfolio_frame.index)

        context_states: dict[str, Any] = {}
        for name in ("low_corr", "mid_corr", "high_corr"):
            seg = net_ret.loc[labels == name]
            include = bool(len(seg) >= int(min_state_days))
            if seg.empty:
                context_states[name] = {
                    "days": 0,
                    "included": False,
                    "corr_mean": 0.0,
                    "total_return": 0.0,
                    "avg_daily_return": 0.0,
                    "win_rate": 0.0,
                }
                continue
            state_corr = rolling_corr.reindex(seg.index).dropna()
            context_states[name] = {
                "days": int(len(seg)),
                "included": include,
                "corr_mean": float(state_corr.mean()) if not state_corr.empty else 0.0,
                "total_return": float(np.prod(1.0 + seg.to_numpy(dtype="float64")) - 1.0),
                "avg_daily_return": float(seg.mean()),
                "win_rate": float((seg > 0.0).mean()),
            }
            if include:
                flat_states[f"{context_name}:{name}"] = context_states[name]

        states_by_context[context_name] = {
            "summary": {
                **summarize_regime_stats({key: value for key, value in context_states.items() if value.get("included")}),
                "corr_window": int(corr_window),
                "low_cut": low_cut,
                "high_cut": high_cut,
            },
            "states": context_states,
        }

    summary = summarize_regime_stats(flat_states)
    summary["corr_window"] = int(corr_window)
    summary["contexts_used"] = list(states_by_context.keys())
    summary["source_mode"] = "internal_fallback" if use_internal_fallback else "market_context"
    return {
        "summary": summary,
        "states": states_by_context,
    }


def build_long_only_candidates() -> list[ResolvedCoreStrategy]:
    strategies = [resolve_core_strategy(LONG_ONLY_FAMILY, params, source="long_only_grid") for params in candidate_params()]
    incumbent = resolve_core_strategy(LONG_ONLY_FAMILY, BEST_CORE_PARAMS, source="target_050_incumbent")
    if all(strategy.key != incumbent.key for strategy in strategies):
        strategies.append(incumbent)
    deduped: dict[str, ResolvedCoreStrategy] = {}
    for strategy in strategies:
        deduped[strategy.key] = strategy
    return list(deduped.values())


def ensure_long_short_nsga_types() -> None:
    if not hasattr(creator, "SearchCoreLongShortFitnessNSGA"):
        creator.create(
            "SearchCoreLongShortFitnessNSGA",
            base.Fitness,
            weights=(1.0, 1.0, 1.0, -1.0, -1.0, -1.0),
        )
    if not hasattr(creator, "SearchCoreLongShortIndividualNSGA"):
        creator.create(
            "SearchCoreLongShortIndividualNSGA",
            list,
            fitness=creator.SearchCoreLongShortFitnessNSGA,
        )


def long_short_nsga_objectives(
    close: pd.DataFrame,
    portfolio_frame: pd.DataFrame,
    metrics: dict[str, Any],
) -> tuple[float, float, float, float, float, float]:
    folds = summarize_fold_robustness_from_portfolio_frame(
        portfolio_frame,
        fold_days=30,
        fold_step=15,
    )
    breakdown = build_regime_breakdown_from_portfolio_frame(
        close.loc[TRAIN_START:TRAIN_END].copy(),
        portfolio_frame,
    )
    bear_return = float(breakdown["states"].get("bear", {}).get("total_return", 0.0))
    return (
        float(metrics["total_return"]),
        float(folds["fold_positive_rate"]),
        bear_return,
        float(abs(metrics["max_drawdown"])),
        float(abs(metrics["daily_metrics"]["cvar"])),
        float(metrics["avg_turnover"]),
    )


def long_short_candidate_scalar(metrics: dict[str, Any], portfolio_frame: pd.DataFrame, close: pd.DataFrame) -> float:
    objectives = long_short_nsga_objectives(close, portfolio_frame, metrics)
    return float(
        objectives[0] * 120.0
        + objectives[1] * 18.0
        + objectives[2] * 60.0
        - objectives[3] * 140.0
        - objectives[4] * 120.0
        - objectives[5] * 10.0
    )


def pick_long_short_candidate_pool_nsga(
    population: list[Any],
    pareto_front: tools.ParetoFront,
    *,
    limit: int,
    metric_cache: dict[str, dict[str, Any]],
    portfolio_cache: dict[str, pd.DataFrame],
    close: pd.DataFrame,
) -> list[ResolvedCoreStrategy]:
    pareto_keys = {decode_individual(list(ind)).key() for ind in pareto_front}
    ranked = list(pareto_front) + sorted(
        population,
        key=lambda ind: long_short_candidate_scalar(
            metric_cache[decode_individual(list(ind)).key()],
            portfolio_cache[decode_individual(list(ind)).key()],
            close,
        ),
        reverse=True,
    )

    pool: list[ResolvedCoreStrategy] = []
    seen: set[str] = set()
    for individual in ranked:
        params = decode_individual(list(individual))
        key = params.key()
        if key in seen:
            continue
        seen.add(key)
        source = "ga_long_short_nsga_pareto" if key in pareto_keys else "ga_long_short_nsga_population"
        pool.append(resolve_core_strategy(LONG_SHORT_FAMILY, params, source=source))
        if len(pool) >= limit:
            break
    return pool


def search_long_short_candidate_pool(
    close: pd.DataFrame,
    *,
    population: int,
    generations: int,
    hof_size: int,
    seed: int,
    pool_limit: int,
    base_slippage: float,
) -> list[ResolvedCoreStrategy]:
    random.seed(seed)
    np.random.seed(seed)
    ensure_long_short_nsga_types()

    metric_cache: dict[str, dict[str, Any]] = {}
    portfolio_cache: dict[str, pd.DataFrame] = {}
    weight_cache: dict[str, pd.DataFrame] = {}

    def evaluate_individual(individual: list[int]) -> tuple[float, float, float, float, float, float]:
        params = decode_individual(individual)
        key = params.key()
        if key not in metric_cache:
            if key not in weight_cache:
                weight_cache[key] = build_long_short_target_weights(close, params)
            portfolio_frame, _ = compute_portfolio_frame(
                close,
                weight_cache[key],
                TRAIN_START,
                TRAIN_END,
                fee_rate=params.fee_rate,
                slippage_rate=base_slippage,
            )
            portfolio_cache[key] = portfolio_frame
            _, metrics = summarize_portfolio_frame(portfolio_frame)
            metric_cache[key] = metrics
        return long_short_nsga_objectives(close, portfolio_cache[key], metric_cache[key])

    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.SearchCoreLongShortIndividualNSGA,
        build_candidate_individual,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate_individual, indpb=0.25)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("clone", copy.deepcopy)

    pop_size = int(population)
    if pop_size % 4 != 0:
        pop_size += 4 - (pop_size % 4)

    pop = toolbox.population(n=pop_size)
    invalid = [item for item in pop if not item.fitness.valid]
    for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
        individual.fitness.values = fitness
    pop = toolbox.select(pop, len(pop))
    pareto = tools.ParetoFront(similar=lambda a, b: decode_individual(list(a)).key() == decode_individual(list(b)).key())
    pareto.update(pop)

    for generation in range(1, generations + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.6:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < 0.25:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid = [item for item in offspring if not item.fitness.valid]
        for individual, fitness in zip(invalid, map(toolbox.evaluate, invalid)):
            individual.fitness.values = fitness
        pop = toolbox.select(pop + offspring, pop_size)
        pareto.update(pop)

    return pick_long_short_candidate_pool_nsga(
        pop,
        pareto,
        limit=pool_limit,
        metric_cache=metric_cache,
        portfolio_cache=portfolio_cache,
        close=close,
    )


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Unified Core Champion Search")
    print("=" * 80)

    close, data_sources = load_daily_close(refresh_cache=args.refresh_cache)
    print(f"Daily bars: {len(close)} | Range: {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"Sources: {', '.join(data_sources)}")
    market_context_close, market_context_status = load_market_context_dataset(
        refresh=args.refresh_market_context,
        allow_fetch_on_miss=bool(args.refresh_market_context),
        target_index=close.index,
    )
    context_labels = ", ".join(market_context_status.get("usable_columns", [])) or "none"
    print(f"Market context: {market_context_status.get('status')} | usable={context_labels}")

    long_only_candidates = build_long_only_candidates()
    long_short_candidates = search_long_short_candidate_pool(
        close,
        population=args.population,
        generations=args.generations,
        hof_size=args.hof_size,
        seed=args.seed,
        pool_limit=args.long_short_pool,
        base_slippage=args.base_slippage,
    )
    candidates = long_only_candidates + long_short_candidates
    print(
        f"Candidate counts: long_only={len(long_only_candidates)} "
        f"long_short={len(long_short_candidates)} total={len(candidates)}"
    )

    weight_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    trial_count = max(1, len(candidates))
    commission_stress = parse_csv_floats(args.stress_commission_multipliers)
    slippage_stress = parse_csv_floats(args.stress_slippage_multipliers)

    for strategy in candidates:
        target_weights = build_core_target_weights(close, strategy)
        weight_cache[strategy.key] = target_weights
        fee_rate = float(strategy.params.fee_rate)
        _, _, train_metrics = evaluate_target_weights_generic(
            close,
            target_weights,
            TRAIN_START,
            TRAIN_END,
            fee_rate=fee_rate,
            slippage_rate=args.base_slippage,
        )
        _, _, val_metrics = evaluate_target_weights_generic(
            close,
            target_weights,
            VAL_START,
            VAL_END,
            fee_rate=fee_rate,
            slippage_rate=args.base_slippage,
        )
        _, _, test_metrics = evaluate_target_weights_generic(
            close,
            target_weights,
            TEST_START,
            TEST_END,
            fee_rate=fee_rate,
            slippage_rate=args.base_slippage,
        )
        _, _, oos_metrics = evaluate_target_weights_generic(
            close,
            target_weights,
            VAL_START,
            TEST_END,
            fee_rate=fee_rate,
            slippage_rate=args.base_slippage,
        )
        selection_score = validation_score_generic(train_metrics, val_metrics)
        rows.append(
            {
                "family": strategy.family,
                "key": strategy.key,
                "source": strategy.source,
                "params": asdict(strategy.params),
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics,
                "oos": oos_metrics,
                "train_score": train_score_generic(train_metrics),
                "selection_score": selection_score,
            }
        )

    for row in rows:
        strategy = resolve_core_strategy(row["family"], row["params"], key=row["key"], source=row["source"])
        target_weights = weight_cache[strategy.key]
        fee_rate = float(strategy.params.fee_rate)
        row["objective_metrics"] = build_objective_metrics(row, trial_count=trial_count)
        portfolio_frame, _ = compute_portfolio_frame(
            close,
            target_weights,
            VAL_START,
            TEST_END,
            fee_rate=fee_rate,
            slippage_rate=args.base_slippage,
        )
        row["fold_robustness"] = summarize_fold_robustness(
            close,
            target_weights,
            fee_rate,
            fold_days=args.fold_days,
            fold_step=args.fold_step,
            slippage_rate=args.base_slippage,
        )
        row["cpcv"] = summarize_cpcv_lite(
            portfolio_frame,
            n_blocks=args.cpcv_blocks,
            test_blocks=args.cpcv_test_blocks,
            embargo_days=args.cpcv_embargo_days,
        )
        row["stress"] = evaluate_stress_profiles(
            close,
            target_weights,
            fee_rate,
            base_slippage=args.base_slippage,
            commission_multipliers=commission_stress,
            slippage_multipliers=slippage_stress,
        )
        row["regime_breakdown"] = build_regime_breakdown(
            close,
            target_weights,
            fee_rate,
            slippage_rate=args.base_slippage,
        )
        row["corr_state_robustness"] = build_corr_state_breakdown(
            close,
            market_context_close,
            portfolio_frame,
            corr_window=args.corr_window,
            min_state_days=args.corr_state_min_days,
        )

    parameter_stability = build_parameter_stability(rows, neighbor_count=args.neighbor_count)
    for row in rows:
        row["parameter_stability"] = parameter_stability[row["key"]]
        row["pareto_vector"] = build_pareto_vector(row)

    pareto_metadata = assign_pareto_metadata(rows)
    for row in rows:
        row["pareto"] = pareto_metadata[row["key"]]
        row["promotion_gate"] = {
            "passes_test_positive": bool(row["test"]["total_return"] > 0.0),
            "passes_oos_positive": bool(row["oos"]["total_return"] > 0.0),
            "passes_oos_mdd": bool(row["oos"]["max_drawdown"] >= -0.30),
            "passes_stress_survival": bool(row["stress"]["stress_survival_rate"] >= (2.0 / 3.0)),
            "passes_fold_positive_rate": bool(row["fold_robustness"]["fold_positive_rate"] >= 0.50),
            "passes_cpcv_positive_rate": bool(row["cpcv"]["test_positive_rate"] >= 0.50),
            "passes_cpcv_pass_rate": bool(row["cpcv"]["pass_rate"] >= 0.50),
        }
        row["promotion_gate"]["passed"] = bool(all(row["promotion_gate"].values()))
        row["promotion_score"] = promotion_score(row)

    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row["promotion_gate"]["passed"] else 1,
            row["pareto"]["rank"],
            -row["promotion_score"],
            -row["pareto"]["crowding_sort_value"],
            -row["selection_score"],
        ),
    )
    finalists_sorted = ranked[: max(1, args.finalists)]
    eligible = [row for row in ranked if row["promotion_gate"]["passed"]]
    selected_row = eligible[0] if eligible else ranked[0]
    print(f"Finalists: {len(finalists_sorted)}")
    selected_strategy = resolve_core_strategy(
        selected_row["family"],
        selected_row["params"],
        key=selected_row["key"],
        source=selected_row["source"],
        metadata={
            "promotion_gate": selected_row["promotion_gate"],
            "selection_score": selected_row["selection_score"],
            "promotion_score": selected_row["promotion_score"],
            "objective_metrics": selected_row["objective_metrics"],
            "pareto": selected_row["pareto"],
        },
    )

    summary = {
        "strategy_class": "unified_core_competition_search",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "dataset": {
            "n_days": int(len(close)),
            "start": str(close.index[0].date()),
            "end": str(close.index[-1].date()),
            "sources": data_sources,
            "market_context": market_context_status,
        },
        "search_config": {
            "population": args.population,
            "generations": args.generations,
            "hof_size": args.hof_size,
            "long_short_pool": args.long_short_pool,
            "long_short_engine": "nsga2",
            "seed": args.seed,
            "finalists": args.finalists,
            "fold_days": args.fold_days,
            "fold_step": args.fold_step,
            "base_slippage": args.base_slippage,
            "stress_commission_multipliers": commission_stress,
            "stress_slippage_multipliers": slippage_stress,
            "cpcv_blocks": args.cpcv_blocks,
            "cpcv_test_blocks": args.cpcv_test_blocks,
            "cpcv_embargo_days": args.cpcv_embargo_days,
            "corr_window": args.corr_window,
            "corr_state_min_days": args.corr_state_min_days,
            "neighbor_count": args.neighbor_count,
            "refresh_market_context": bool(args.refresh_market_context),
            "refresh_cache": bool(args.refresh_cache),
        },
        "candidate_counts": {
            "long_only": int(len(long_only_candidates)),
            "long_short": int(len(long_short_candidates)),
            "total": int(len(candidates)),
        },
        "selected": {
            "family": selected_strategy.family,
            "key": selected_strategy.key,
            "source": selected_strategy.source,
            "params": asdict(selected_strategy.params),
            "selection_score": float(selected_row["selection_score"]),
            "promotion_score": float(selected_row["promotion_score"]),
            "promotion_gate": selected_row["promotion_gate"],
            "objective_metrics": selected_row["objective_metrics"],
            "pareto": selected_row["pareto"],
        },
        "finalists": finalists_sorted,
        "artifact_path": str(Path(args.artifact_out)),
    }
    summary_path = Path(args.summary_out)
    with open(summary_path, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    save_core_artifact(
        args.artifact_out,
        selected_strategy,
        selected_score=float(selected_row["selection_score"]),
        summary_path=summary_path,
        extra={
            "created_at": summary["created_at"],
            "search_config": summary["search_config"],
        },
    )

    print(
        f"Selected champion: {selected_strategy.family} | {selected_strategy.key} "
        f"| val={selected_row['validation']['total_return']*100:+.2f}% "
        f"| test={selected_row['test']['total_return']*100:+.2f}% "
        f"| oos={selected_row['oos']['total_return']*100:+.2f}% "
        f"| stress={selected_row['stress']['stress_survival_rate']:.2f} "
        f"| dsr={selected_row['objective_metrics']['oos_dsr_proxy']:.2f} "
        f"| pareto={selected_row['pareto']['rank']}"
    )
    print(f"Summary saved: {summary_path}")
    print(f"Artifact saved: {args.artifact_out}")


if __name__ == "__main__":
    main()
