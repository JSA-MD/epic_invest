#!/usr/bin/env python3
"""Validation-first scoring helpers for pairwise candidates."""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp


DAY_FACTOR = math.sqrt(365.25)
VALIDATION_THRESHOLDS = {
    "dsr_proxy_min": 0.55,
    "cpcv_positive_rate_min": 0.55,
    "cpcv_pass_rate_min": 0.55,
    "cpcv_min_test_return_min": -0.08,
    "cpcv_overfit_rate_max": 0.45,
    "pbo_selected_below_median_rate_max": 0.45,
    "pbo_avg_selected_test_percentile_min": 0.45,
    "pbo_selection_share_min": 0.04,
    "false_positive_risk_max": 0.45,
    "validation_quality_score_min": 0.55,
    "market_os_fitness_min": 0.50,
    "daily_win_rate_oos_min": 0.50,
    "corr_state_robustness_min": 0.40,
    "regime_coverage_min": 0.35,
    "special_regime_robustness_min": 0.35,
    "parameter_instability_max": 0.85,
    "final_oos_total_return_min": 0.0,
    "final_oos_max_drawdown_cap": 0.18,
}
RECENT_VALIDATION_DAYS = 182
FINAL_OOS_DAYS = 61
OPERATING_SYSTEM_FITNESS_WEIGHTS = {
    "dsr_oos": 0.23,
    "calmar_oos": 0.16,
    "median_fold_expectancy": 0.14,
    "daily_win_rate_oos": 0.12,
    "regime_coverage": 0.08,
    "corr_state_robustness": 0.08,
    "special_regime_robustness": 0.07,
    "max_drawdown": -0.10,
    "cvar_95": -0.05,
    "turnover_cost": -0.03,
    "parameter_instability": -0.02,
}
VALIDATION_ROBUSTNESS_WEIGHTS = {
    "false_positive_risk": 0.20,
    "validation_quality_shortfall": 0.10,
    "cpcv_overfit_rate": 0.15,
    "pbo_selected_below_median_rate": 0.10,
    "pbo_avg_selected_test_percentile": 0.07,
    "pbo_selection_share": 0.03,
    "market_os_fitness": 0.12,
    "daily_win_rate_oos": 0.05,
    "corr_state_robustness": 0.08,
    "special_regime_robustness": 0.07,
    "regime_coverage": 0.03,
    "parameter_instability": 0.10,
}


def threshold_shortfall(actual: float, minimum: float) -> float:
    scale = max(abs(float(minimum)), 1e-8)
    return float(np.clip(max(0.0, float(minimum) - float(actual)) / scale, 0.0, 1.0))


def threshold_excess(actual: float, maximum: float) -> float:
    scale = max(abs(float(maximum)), 1e-8)
    return float(np.clip(max(0.0, float(actual) - float(maximum)) / scale, 0.0, 1.0))


def clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def normalize_positive_metric(value: float, target: float) -> float:
    scale = max(abs(float(target)), 1e-8)
    return clip01(float(value) / scale)


def normalize_negative_metric(value: float, cap: float) -> float:
    scale = max(abs(float(cap)), 1e-8)
    return clip01(abs(float(value)) / scale)


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


def build_return_frame(daily_returns: np.ndarray, daily_index: pd.DatetimeIndex | list[Any]) -> pd.DataFrame:
    returns = np.asarray(daily_returns, dtype="float64")
    if len(returns) == 0:
        return pd.DataFrame(columns=["net_return"])
    index = pd.DatetimeIndex(daily_index)
    if len(index) < len(returns):
        if len(index) == 0:
            index = pd.date_range("2022-01-01", periods=len(returns), freq="D", tz="UTC")
        else:
            start = index[0]
            index = pd.date_range(start, periods=len(returns), freq="D", tz=start.tzinfo or "UTC")
    elif len(index) > len(returns):
        index = index[: len(returns)]
    return pd.DataFrame({"net_return": returns}, index=index)


def summarize_return_frame(frame: pd.DataFrame, *, initial_cash: float = gp.INITIAL_CASH) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        empty_daily = gp.summarize_period_returns(np.asarray([], dtype="float64"))
        empty_monthly = gp.summarize_monthly_returns(np.asarray([], dtype="float64"), pd.DatetimeIndex([]))
        return pd.DataFrame(columns=["time", "equity", "net_return", "drawdown"]), {
            "total_return": 0.0,
            "final_equity": float(initial_cash),
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "daily_metrics": empty_daily,
            "monthly_metrics": empty_monthly,
        }
    net_ret = frame["net_return"].astype("float64")
    equity = float(initial_cash) * (1.0 + net_ret).cumprod()
    curve = pd.DataFrame(
        {
            "time": net_ret.index,
            "equity": equity.to_numpy(dtype="float64"),
            "net_return": net_ret.to_numpy(dtype="float64"),
        }
    )
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    if len(net_ret) > 1 and net_ret.std() > 1e-12:
        sharpe = float(net_ret.mean() / net_ret.std() * DAY_FACTOR)
    else:
        sharpe = 0.0
    daily_values = net_ret.to_numpy(dtype="float64")
    return curve, {
        "total_return": float(equity.iloc[-1] / initial_cash - 1.0),
        "final_equity": float(equity.iloc[-1]),
        "max_drawdown": float(curve["drawdown"].min()),
        "sharpe": sharpe,
        "daily_metrics": gp.summarize_period_returns(daily_values),
        "monthly_metrics": gp.summarize_monthly_returns(daily_values, pd.DatetimeIndex(net_ret.index)),
    }


def compute_annualized_return(total_return: float, periods: int) -> float:
    period_years = max(float(periods) / 365.25, 1.0 / 365.25)
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (1.0 / period_years) - 1.0)


def compute_calmar_ratio(total_return: float, max_drawdown: float, periods: int) -> float:
    drawdown = abs(float(max_drawdown))
    if drawdown <= 1e-8:
        return 0.0
    return float(compute_annualized_return(total_return, periods) / drawdown)


def compute_cvar_95(period_returns: np.ndarray) -> float:
    returns = np.asarray(period_returns, dtype="float64")
    if len(returns) == 0:
        return 0.0
    tail_count = max(1, int(math.ceil(len(returns) * 0.05)))
    tail = np.sort(returns)[:tail_count]
    return float(np.mean(tail))


def split_market_os_frames(
    return_frame: pd.DataFrame,
    *,
    recent_validation_days: int = RECENT_VALIDATION_DAYS,
    final_oos_days: int = FINAL_OOS_DAYS,
) -> dict[str, pd.DataFrame]:
    if return_frame.empty:
        empty = return_frame.iloc[0:0].copy()
        return {
            "structure_train": empty,
            "recent_adaptation": empty,
            "final_oos": empty,
        }
    total_len = len(return_frame)
    final_len = max(1, min(int(final_oos_days), total_len))
    adaptation_len = max(0, min(max(int(recent_validation_days) - final_len, 0), total_len - final_len))
    structure_end = max(total_len - final_len - adaptation_len, 0)
    adaptation_end = total_len - final_len
    return {
        "structure_train": return_frame.iloc[:structure_end].copy(),
        "recent_adaptation": return_frame.iloc[structure_end:adaptation_end].copy(),
        "final_oos": return_frame.iloc[adaptation_end:].copy(),
    }


def summarize_stage_frame(
    frame: pd.DataFrame,
    *,
    trial_count: int,
    initial_cash: float = gp.INITIAL_CASH,
) -> dict[str, Any]:
    returns = frame["net_return"].to_numpy(dtype="float64") if not frame.empty else np.asarray([], dtype="float64")
    _, metrics = summarize_return_frame(frame, initial_cash=initial_cash)
    return {
        "days": int(len(frame)),
        "total_return": float(metrics["total_return"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "sharpe": float(metrics["sharpe"]),
        "avg_daily_return": float(metrics["daily_metrics"]["avg_daily_return"]),
        "daily_win_rate": float(metrics["daily_metrics"]["daily_win_rate"]),
        "daily_target_hit_rate": float(metrics["daily_metrics"]["daily_target_hit_rate"]),
        "cvar_95": float(compute_cvar_95(returns)),
        "calmar": float(compute_calmar_ratio(metrics["total_return"], metrics["max_drawdown"], len(frame))),
        "dsr_proxy": float(compute_dsr_proxy(returns, trial_count=trial_count)),
    }


def period_selection_score(metrics: dict[str, Any]) -> float:
    return float(
        metrics["total_return"] * 140.0
        + metrics["sharpe"] * 12.0
        + metrics["daily_metrics"]["daily_win_rate"] * 18.0
        + metrics["daily_metrics"]["daily_target_hit_rate"] * 12.0
        + metrics["monthly_metrics"]["monthly_target_hit_rate"] * 8.0
        - abs(metrics["max_drawdown"]) * 170.0
        - metrics["daily_metrics"]["daily_shortfall_mean"] * 120.0
    )


def validation_score_generic(train_metrics: dict[str, Any], test_metrics: dict[str, Any]) -> float:
    score = (
        test_metrics["total_return"] * 140.0
        + test_metrics["sharpe"] * 12.0
        + test_metrics["daily_metrics"]["daily_win_rate"] * 18.0
        + test_metrics["daily_metrics"]["daily_target_hit_rate"] * 12.0
        + test_metrics["monthly_metrics"]["monthly_target_hit_rate"] * 8.0
        - abs(test_metrics["max_drawdown"]) * 170.0
        - test_metrics["daily_metrics"]["daily_shortfall_mean"] * 120.0
        + train_metrics["total_return"] * 18.0
        + train_metrics["sharpe"] * 3.0
        + train_metrics["daily_metrics"]["daily_win_rate"] * 3.0
        - abs(train_metrics["max_drawdown"]) * 20.0
    )
    if test_metrics["total_return"] <= 0.0:
        score -= 40.0
    if test_metrics["max_drawdown"] < -0.25:
        score -= 35.0
    return float(score)


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
    return_frame: pd.DataFrame,
    *,
    n_blocks: int,
    test_blocks: int,
    embargo_days: int,
) -> dict[str, Any]:
    if return_frame.empty:
        return {
            "n_splits": 0,
            "pass_rate": 0.0,
            "test_positive_rate": 0.0,
            "avg_test_return": 0.0,
            "min_test_return": 0.0,
            "avg_test_mdd": 0.0,
            "splits": [],
        }
    blocks = split_cpcv_blocks(return_frame.index, n_blocks)
    split_rows: list[dict[str, Any]] = []
    for combo in itertools.combinations(range(len(blocks)), min(test_blocks, len(blocks))):
        test_days = pd.DatetimeIndex(sorted({day for idx in combo for day in blocks[idx]}))
        test_mask = return_frame.index.normalize().isin(test_days)
        embargo_set: set[pd.Timestamp] = set()
        if embargo_days > 0:
            for idx in combo:
                block = blocks[idx]
                for offset in range(1, embargo_days + 1):
                    embargo_set.add(block[0] - pd.Timedelta(days=offset))
                    embargo_set.add(block[-1] + pd.Timedelta(days=offset))
        train_mask = ~test_mask
        if embargo_set:
            train_mask &= ~return_frame.index.normalize().isin(pd.DatetimeIndex(sorted(embargo_set)))
        train_frame = return_frame.loc[train_mask]
        test_frame = return_frame.loc[test_mask]
        _, train_metrics = summarize_return_frame(train_frame)
        _, test_metrics = summarize_return_frame(test_frame)
        passed = bool(test_metrics["total_return"] > 0.0 and test_metrics["max_drawdown"] >= -0.30)
        split_rows.append(
            {
                "test_blocks": list(combo),
                "train_days": int(len(train_frame)),
                "test_days": int(len(test_frame)),
                "score": float(validation_score_generic(train_metrics, test_metrics)),
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


def empty_candidate_pbo_profile() -> dict[str, Any]:
    return {
        "selected_count": 0,
        "selection_share": 0.0,
        "selected_below_median_rate": 1.0,
        "avg_selected_test_percentile": 0.0,
        "worst_selected_test_percentile": 0.0,
        "avg_selected_test_score": 0.0,
        "avg_selected_test_return": 0.0,
    }


def rank_index_to_percentile(rank_index: int, total_count: int) -> float:
    count = max(1, int(total_count))
    if count <= 1:
        return 1.0
    return float(np.clip(1.0 - (float(rank_index) / float(count - 1)), 0.0, 1.0))


def summarize_candidate_selection_pbo(split_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not split_candidates:
        return {
            "n_splits": 0,
            "pbo": 1.0,
            "avg_selected_test_percentile": 0.0,
            "worst_selected_test_percentile": 0.0,
            "profiles": {},
            "splits": [],
        }
    all_keys = sorted({candidate["key"] for split in split_candidates for candidate in split.get("candidates", [])})
    profile_state: dict[str, dict[str, Any]] = {
        key: {
            "selected_count": 0,
            "selected_test_percentiles": [],
            "selected_test_scores": [],
            "selected_test_returns": [],
        }
        for key in all_keys
    }
    split_rows: list[dict[str, Any]] = []
    for split in split_candidates:
        candidates = list(split.get("candidates") or [])
        if not candidates:
            continue
        train_sorted = sorted(
            candidates,
            key=lambda row: (float(row["train_score"]), float(row["train_total_return"]), float(row["test_score"])),
            reverse=True,
        )
        test_sorted = sorted(
            candidates,
            key=lambda row: (float(row["test_score"]), float(row["test_total_return"]), float(row["train_score"])),
            reverse=True,
        )
        chosen = train_sorted[0]
        chosen_rank = next(idx for idx, row in enumerate(test_sorted) if row["key"] == chosen["key"])
        chosen_percentile = rank_index_to_percentile(chosen_rank, len(test_sorted))
        chosen_below_median = bool(chosen_percentile < 0.5)
        state = profile_state[chosen["key"]]
        state["selected_count"] += 1
        state["selected_test_percentiles"].append(chosen_percentile)
        state["selected_test_scores"].append(float(chosen["test_score"]))
        state["selected_test_returns"].append(float(chosen["test_total_return"]))
        split_rows.append(
            {
                "test_blocks": split["test_blocks"],
                "selected_key": chosen["key"],
                "selected_train_score": float(chosen["train_score"]),
                "selected_test_score": float(chosen["test_score"]),
                "selected_test_return": float(chosen["test_total_return"]),
                "selected_test_percentile": chosen_percentile,
                "selected_below_median": chosen_below_median,
            }
        )
    if not split_rows:
        return {
            "n_splits": 0,
            "pbo": 1.0,
            "avg_selected_test_percentile": 0.0,
            "worst_selected_test_percentile": 0.0,
            "profiles": {key: empty_candidate_pbo_profile() for key in all_keys},
            "splits": [],
        }
    n_splits = len(split_rows)
    profiles: dict[str, Any] = {}
    for key, state in profile_state.items():
        selected_count = int(state["selected_count"])
        if selected_count > 0:
            percentiles = np.asarray(state["selected_test_percentiles"], dtype="float64")
            scores = np.asarray(state["selected_test_scores"], dtype="float64")
            returns = np.asarray(state["selected_test_returns"], dtype="float64")
            profiles[key] = {
                "selected_count": selected_count,
                "selection_share": float(selected_count / n_splits),
                "selected_below_median_rate": float(np.mean(percentiles < 0.5)),
                "avg_selected_test_percentile": float(np.mean(percentiles)),
                "worst_selected_test_percentile": float(np.min(percentiles)),
                "avg_selected_test_score": float(np.mean(scores)),
                "avg_selected_test_return": float(np.mean(returns)),
            }
        else:
            profiles[key] = empty_candidate_pbo_profile()
    selected_percentiles = np.asarray([row["selected_test_percentile"] for row in split_rows], dtype="float64")
    return {
        "n_splits": int(n_splits),
        "pbo": float(np.mean(selected_percentiles < 0.5)),
        "avg_selected_test_percentile": float(np.mean(selected_percentiles)),
        "worst_selected_test_percentile": float(np.min(selected_percentiles)),
        "profiles": profiles,
        "splits": split_rows,
    }


def summarize_cpcv_candidate_selection_pbo(
    frames_by_key: dict[str, pd.DataFrame],
    *,
    n_blocks: int,
    test_blocks: int,
    embargo_days: int,
) -> dict[str, Any]:
    first_frame = next((frame for frame in frames_by_key.values() if not frame.empty), pd.DataFrame())
    if first_frame.empty:
        return {
            "n_splits": 0,
            "pbo": 1.0,
            "avg_selected_test_percentile": 0.0,
            "worst_selected_test_percentile": 0.0,
            "profiles": {key: empty_candidate_pbo_profile() for key in frames_by_key},
            "splits": [],
        }
    blocks = split_cpcv_blocks(first_frame.index, n_blocks)
    split_candidates: list[dict[str, Any]] = []
    for combo in itertools.combinations(range(len(blocks)), min(test_blocks, len(blocks))):
        test_days = pd.DatetimeIndex(sorted({day for idx in combo for day in blocks[idx]}))
        embargo_set: set[pd.Timestamp] = set()
        if embargo_days > 0:
            for idx in combo:
                block = blocks[idx]
                for offset in range(1, embargo_days + 1):
                    embargo_set.add(block[0] - pd.Timedelta(days=offset))
                    embargo_set.add(block[-1] + pd.Timedelta(days=offset))
        candidates: list[dict[str, Any]] = []
        for key, frame in frames_by_key.items():
            if frame.empty:
                continue
            test_mask = frame.index.normalize().isin(test_days)
            train_mask = ~test_mask
            if embargo_set:
                train_mask &= ~frame.index.normalize().isin(pd.DatetimeIndex(sorted(embargo_set)))
            train_frame = frame.loc[train_mask]
            test_frame = frame.loc[test_mask]
            _, train_metrics = summarize_return_frame(train_frame)
            _, test_metrics = summarize_return_frame(test_frame)
            candidates.append(
                {
                    "key": key,
                    "train_score": period_selection_score(train_metrics),
                    "test_score": period_selection_score(test_metrics),
                    "train_total_return": float(train_metrics["total_return"]),
                    "test_total_return": float(test_metrics["total_return"]),
                }
            )
        split_candidates.append({"test_blocks": list(combo), "candidates": candidates})
    result = summarize_candidate_selection_pbo(split_candidates)
    for key in frames_by_key:
        result["profiles"].setdefault(key, empty_candidate_pbo_profile())
    return result


def compute_cpcv_overfit_rate(cpcv: dict[str, Any]) -> float:
    splits = list(cpcv.get("splits") or [])
    if not splits:
        return 1.0
    signals: list[float] = []
    min_test_return_min = float(VALIDATION_THRESHOLDS["cpcv_min_test_return_min"])
    for row in splits:
        train_total_return = float(row.get("train_total_return", 0.0))
        test_total_return = float(row.get("test_total_return", 0.0))
        signal = 0.0
        if train_total_return > 0.0 and test_total_return <= 0.0:
            signal = 1.0
        elif train_total_return > 0.0 and test_total_return < train_total_return * 0.25:
            signal = 0.5
        if test_total_return < min_test_return_min:
            signal = max(signal, 1.0)
        signals.append(signal)
    return float(np.mean(signals))


def compute_median_fold_expectancy(cpcv: dict[str, Any]) -> float:
    splits = list(cpcv.get("splits") or [])
    if not splits:
        return 0.0
    expectancies = [
        float(row.get("test_total_return", 0.0)) / max(int(row.get("test_days", 0)), 1)
        for row in splits
    ]
    return float(np.median(np.asarray(expectancies, dtype="float64")))


def summarize_state_payload(state_payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = state_payload or {}
    route_state_returns = {
        str(name): np.asarray(values, dtype="float64")
        for name, values in (payload.get("route_state_returns") or {}).items()
        if values
    }
    corr_bucket_returns = {
        str(name): np.asarray(values, dtype="float64")
        for name, values in (payload.get("corr_bucket_returns") or {}).items()
        if values
    }
    special_regime_returns = {
        str(name): np.asarray(values, dtype="float64")
        for name, values in (payload.get("special_regime_returns") or {}).items()
        if values
    }
    corr_input_samples = {
        str(name): np.asarray(values, dtype="float64")
        for name, values in (payload.get("corr_input_samples") or {}).items()
        if values
    }
    total_states = max(int(payload.get("total_route_states", 0)), 1)
    observed_states = len(route_state_returns)
    state_mean_returns = {
        name: float(values.mean())
        for name, values in route_state_returns.items()
    }
    positive_states = int(sum(1 for value in state_mean_returns.values() if value >= 0.0))
    coverage_share = float(observed_states / total_states)
    positive_share = float(positive_states / max(observed_states, 1))
    regime_coverage = clip01(0.55 * coverage_share + 0.45 * positive_share)

    bucket_mean_returns = {
        name: float(values.mean())
        for name, values in corr_bucket_returns.items()
    }
    if bucket_mean_returns:
        bucket_values = np.asarray(list(bucket_mean_returns.values()), dtype="float64")
        corr_bucket_count = int(payload.get("total_corr_buckets", max(len(bucket_mean_returns), 1)))
        corr_coverage = float(len(bucket_mean_returns) / max(corr_bucket_count, 1))
        corr_positive_share = float(np.mean(bucket_values >= 0.0))
        corr_worst_bucket = float(np.min(bucket_values))
        corr_dispersion = float(np.std(bucket_values))
        corr_state_robustness = clip01(
            0.35 * corr_coverage
            + 0.30 * corr_positive_share
            + 0.20 * normalize_positive_metric(corr_worst_bucket + 0.002, 0.004)
            + 0.15 * (1.0 - normalize_negative_metric(corr_dispersion, 0.004))
        )
    else:
        corr_state_robustness = 0.0
        corr_worst_bucket = 0.0
        corr_dispersion = 0.0

    special_regime_mean_returns = {
        name: float(values.mean())
        for name, values in special_regime_returns.items()
    }
    observed_special_regimes = len(special_regime_mean_returns)
    positive_special_regimes = int(sum(1 for value in special_regime_mean_returns.values() if value >= 0.0))
    special_regime_coverage = float(observed_special_regimes / 4.0)
    if special_regime_mean_returns:
        special_values = np.asarray(list(special_regime_mean_returns.values()), dtype="float64")
        special_positive_share = float(np.mean(special_values >= 0.0))
        special_worst = float(np.min(special_values))
        special_dispersion = float(np.std(special_values))
        special_regime_robustness = clip01(
            0.35 * special_regime_coverage
            + 0.30 * special_positive_share
            + 0.20 * normalize_positive_metric(special_worst + 0.0015, 0.0030)
            + 0.15 * (1.0 - normalize_negative_metric(special_dispersion, 0.0040))
        )
    else:
        special_regime_robustness = 0.0
        special_worst = 0.0
        special_dispersion = 0.0
    corr_input_means = {
        name: float(values.mean())
        for name, values in corr_input_samples.items()
    }

    return {
        "route_state_mean_returns": state_mean_returns,
        "corr_bucket_mean_returns": bucket_mean_returns,
        "special_regime_mean_returns": special_regime_mean_returns,
        "corr_input_means": corr_input_means,
        "observed_state_count": int(observed_states),
        "positive_state_count": int(positive_states),
        "state_coverage_share": float(coverage_share),
        "regime_coverage": float(regime_coverage),
        "corr_state_robustness": float(corr_state_robustness),
        "special_regime_coverage": float(special_regime_coverage),
        "special_regime_robustness": float(special_regime_robustness),
        "positive_special_regime_count": int(positive_special_regimes),
        "special_regime_worst_return": float(special_worst),
        "special_regime_dispersion": float(special_dispersion),
        "corr_worst_bucket_return": float(corr_worst_bucket),
        "corr_bucket_dispersion": float(corr_dispersion),
    }


def summarize_cost_reference(cost_reference: dict[str, Any] | None) -> dict[str, Any]:
    reference = cost_reference or {}
    mean_cost_ratio = float(reference.get("mean_cost_ratio", 0.0))
    max_cost_ratio = float(reference.get("max_cost_ratio", mean_cost_ratio))
    mean_n_trades = float(reference.get("mean_n_trades", 0.0))
    return {
        "mean_cost_ratio": mean_cost_ratio,
        "max_cost_ratio": max_cost_ratio,
        "mean_n_trades": mean_n_trades,
    }


def compute_parameter_instability(cpcv: dict[str, Any], pbo_profile: dict[str, Any]) -> float:
    splits = list(cpcv.get("splits") or [])
    if not splits:
        return 1.0
    test_scores = np.asarray([float(row.get("score", 0.0)) for row in splits], dtype="float64")
    expectancies = np.asarray(
        [
            float(row.get("test_total_return", 0.0)) / max(int(row.get("test_days", 0)), 1)
            for row in splits
        ],
        dtype="float64",
    )
    score_penalty = normalize_negative_metric(float(np.std(test_scores)), 40.0)
    expectancy_penalty = normalize_negative_metric(float(np.std(expectancies)), 0.0025)
    selection_penalty = clip01(1.0 - float(pbo_profile.get("selection_share", 0.0)))
    return clip01(0.40 * selection_penalty + 0.35 * expectancy_penalty + 0.25 * score_penalty)


def build_market_operating_system(
    return_frame: pd.DataFrame,
    *,
    trial_count: int,
    cpcv: dict[str, Any],
    pbo_profile: dict[str, Any],
    state_payload: dict[str, Any] | None,
    cost_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    staged_frames = split_market_os_frames(return_frame)
    stages = {
        name: summarize_stage_frame(frame, trial_count=trial_count)
        for name, frame in staged_frames.items()
    }
    state_summary = summarize_state_payload(state_payload)
    cost_summary = summarize_cost_reference(cost_reference)
    median_fold_expectancy = compute_median_fold_expectancy(cpcv)
    parameter_instability = compute_parameter_instability(cpcv, pbo_profile)
    recent_adaptation = stages["recent_adaptation"]
    final_oos = stages["final_oos"]
    normalized = {
        "dsr_oos": float(recent_adaptation["dsr_proxy"]),
        "calmar_oos": normalize_positive_metric(recent_adaptation["calmar"], 3.0),
        "median_fold_expectancy": normalize_positive_metric(median_fold_expectancy, 0.002),
        "daily_win_rate_oos": normalize_positive_metric(recent_adaptation["daily_win_rate"], 0.55),
        "regime_coverage": float(state_summary["regime_coverage"]),
        "corr_state_robustness": float(state_summary["corr_state_robustness"]),
        "special_regime_robustness": float(state_summary.get("special_regime_robustness", 0.0)),
        "max_drawdown": normalize_negative_metric(
            min(float(stages["structure_train"]["max_drawdown"]), float(recent_adaptation["max_drawdown"])),
            0.25,
        ),
        "cvar_95": normalize_negative_metric(min(float(recent_adaptation["cvar_95"]), 0.0), 0.03),
        "turnover_cost": normalize_negative_metric(cost_summary["mean_cost_ratio"], 0.20),
        "parameter_instability": float(parameter_instability),
    }
    score = 0.0
    for name, weight in OPERATING_SYSTEM_FITNESS_WEIGHTS.items():
        score += float(normalized[name]) * float(weight)
    fitness = {
        "score": float(score),
        "weights": OPERATING_SYSTEM_FITNESS_WEIGHTS,
        "normalized": normalized,
        "raw": {
            "dsr_oos": float(recent_adaptation["dsr_proxy"]),
            "calmar_oos": float(recent_adaptation["calmar"]),
            "median_fold_expectancy": float(median_fold_expectancy),
            "daily_win_rate_oos": float(recent_adaptation["daily_win_rate"]),
            "final_oos_daily_win_rate": float(final_oos["daily_win_rate"]),
            "regime_coverage": float(state_summary["regime_coverage"]),
            "corr_state_robustness": float(state_summary["corr_state_robustness"]),
            "special_regime_coverage": float(state_summary.get("special_regime_coverage", 0.0)),
            "special_regime_robustness": float(state_summary.get("special_regime_robustness", 0.0)),
            "max_drawdown": float(min(stages["structure_train"]["max_drawdown"], recent_adaptation["max_drawdown"])),
            "cvar_95": float(recent_adaptation["cvar_95"]),
            "turnover_cost": float(cost_summary["mean_cost_ratio"]),
            "parameter_instability": float(parameter_instability),
        },
    }
    gate = {
        "passes_market_os_fitness": bool(fitness["score"] >= VALIDATION_THRESHOLDS["market_os_fitness_min"]),
        "passes_daily_win_rate_oos": bool(
            recent_adaptation["daily_win_rate"] >= VALIDATION_THRESHOLDS["daily_win_rate_oos_min"]
        ),
        "passes_corr_state_robustness": bool(
            state_summary["corr_state_robustness"] >= VALIDATION_THRESHOLDS["corr_state_robustness_min"]
        ),
        "passes_regime_coverage": bool(
            state_summary["regime_coverage"] >= VALIDATION_THRESHOLDS["regime_coverage_min"]
        ),
        "passes_parameter_instability": bool(
            parameter_instability <= VALIDATION_THRESHOLDS["parameter_instability_max"]
        ),
    }
    gate["failed_checks"] = [name for name, passed in gate.items() if name != "failed_checks" and not passed]
    gate["passed"] = bool(not gate["failed_checks"])
    audit = {
        "passes_final_oos_total_return": bool(
            final_oos["total_return"] >= VALIDATION_THRESHOLDS["final_oos_total_return_min"]
        ),
        "passes_final_oos_max_drawdown": bool(
            abs(final_oos["max_drawdown"]) <= VALIDATION_THRESHOLDS["final_oos_max_drawdown_cap"]
        ),
    }
    audit["failed_checks"] = [name for name, passed in audit.items() if name != "failed_checks" and not passed]
    audit["passed"] = bool(not audit["failed_checks"])
    return {
        "stages": stages,
        "state_summary": state_summary,
        "cost_summary": cost_summary,
        "fitness": fitness,
        "gate": gate,
        "audit": audit,
    }


def build_validation_profile(dsr_proxy: float, cpcv: dict[str, Any], pbo_profile: dict[str, Any]) -> dict[str, Any]:
    risk_components = {
        "dsr_shortfall": threshold_shortfall(dsr_proxy, VALIDATION_THRESHOLDS["dsr_proxy_min"]),
        "cpcv_pass_shortfall": threshold_shortfall(cpcv["pass_rate"], VALIDATION_THRESHOLDS["cpcv_pass_rate_min"]),
        "cpcv_positive_shortfall": threshold_shortfall(cpcv["test_positive_rate"], VALIDATION_THRESHOLDS["cpcv_positive_rate_min"]),
        "cpcv_tail_loss_shortfall": threshold_shortfall(cpcv["min_test_return"], VALIDATION_THRESHOLDS["cpcv_min_test_return_min"]),
        "cpcv_overfit_rate": compute_cpcv_overfit_rate(cpcv),
        "pbo_selected_below_median_rate": threshold_excess(
            pbo_profile["selected_below_median_rate"],
            VALIDATION_THRESHOLDS["pbo_selected_below_median_rate_max"],
        ),
        "pbo_avg_selected_test_percentile": threshold_shortfall(
            pbo_profile["avg_selected_test_percentile"],
            VALIDATION_THRESHOLDS["pbo_avg_selected_test_percentile_min"],
        ),
        "pbo_selection_share": threshold_shortfall(
            pbo_profile["selection_share"],
            VALIDATION_THRESHOLDS["pbo_selection_share_min"],
        ),
    }
    weights = {
        "dsr_shortfall": 0.25,
        "cpcv_pass_shortfall": 0.14,
        "cpcv_positive_shortfall": 0.12,
        "cpcv_tail_loss_shortfall": 0.10,
        "cpcv_overfit_rate": 0.18,
        "pbo_selected_below_median_rate": 0.10,
        "pbo_avg_selected_test_percentile": 0.07,
        "pbo_selection_share": 0.04,
    }
    false_positive_risk = float(
        np.clip(sum(float(risk_components[name]) * weight for name, weight in weights.items()), 0.0, 1.0)
    )
    return {
        "false_positive_risk": false_positive_risk,
        "validation_quality_score": float(np.clip(1.0 - false_positive_risk, 0.0, 1.0)),
        "cpcv_overfit_rate": float(risk_components["cpcv_overfit_rate"]),
        "pbo_selected_below_median_rate": float(pbo_profile["selected_below_median_rate"]),
        "pbo_avg_selected_test_percentile": float(pbo_profile["avg_selected_test_percentile"]),
        "pbo_selection_share": float(pbo_profile["selection_share"]),
        "risk_components": risk_components,
    }


def build_validation_gate(
    dsr_proxy: float,
    cpcv: dict[str, Any],
    profile: dict[str, Any],
    market_operating_system: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checks = {
        "passes_dsr_proxy": bool(dsr_proxy >= VALIDATION_THRESHOLDS["dsr_proxy_min"]),
        "passes_cpcv_positive_rate": bool(cpcv["test_positive_rate"] >= VALIDATION_THRESHOLDS["cpcv_positive_rate_min"]),
        "passes_cpcv_pass_rate": bool(cpcv["pass_rate"] >= VALIDATION_THRESHOLDS["cpcv_pass_rate_min"]),
        "passes_cpcv_tail_loss": bool(cpcv["min_test_return"] >= VALIDATION_THRESHOLDS["cpcv_min_test_return_min"]),
        "passes_cpcv_overfit_rate": bool(profile["cpcv_overfit_rate"] <= VALIDATION_THRESHOLDS["cpcv_overfit_rate_max"]),
        "passes_pbo_selected_below_median_rate": bool(
            profile["pbo_selected_below_median_rate"] <= VALIDATION_THRESHOLDS["pbo_selected_below_median_rate_max"]
        ),
        "passes_pbo_avg_selected_test_percentile": bool(
            profile["pbo_avg_selected_test_percentile"] >= VALIDATION_THRESHOLDS["pbo_avg_selected_test_percentile_min"]
        ),
        "passes_pbo_selection_share": bool(
            profile["pbo_selection_share"] >= VALIDATION_THRESHOLDS["pbo_selection_share_min"]
        ),
        "passes_false_positive_risk": bool(
            profile["false_positive_risk"] <= VALIDATION_THRESHOLDS["false_positive_risk_max"]
        ),
        "passes_validation_quality": bool(
            profile["validation_quality_score"] >= VALIDATION_THRESHOLDS["validation_quality_score_min"]
        ),
    }
    market_gate = (market_operating_system or {}).get("gate") or {}
    for name, passed in market_gate.items():
        if name in {"failed_checks", "passed"}:
            continue
        checks[name] = bool(passed)
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_checks": failed_checks,
        "passed": bool(not failed_checks),
    }


def build_validation_robustness_profile(validation_engine: dict[str, Any] | None) -> dict[str, Any]:
    validation_engine = validation_engine or {}
    profile = validation_engine.get("profile") or {}
    market_os = validation_engine.get("market_operating_system") or {}
    market_os_fitness = market_os.get("fitness") or {}
    market_os_raw = market_os_fitness.get("raw") or {}
    state_summary = market_os.get("state_summary") or {}
    gate = validation_engine.get("gate") or {}
    gate_checks = [
        bool(passed)
        for name, passed in gate.items()
        if name not in {"failed_checks", "passed"}
    ]
    gate_pass_ratio = float(sum(gate_checks) / len(gate_checks)) if gate_checks else 0.0
    components = {
        "false_positive_risk": clip01(profile.get("false_positive_risk", 1.0)),
        "validation_quality_shortfall": threshold_shortfall(
            profile.get("validation_quality_score", 0.0),
            VALIDATION_THRESHOLDS["validation_quality_score_min"],
        ),
        "cpcv_overfit_rate": threshold_excess(
            profile.get("cpcv_overfit_rate", 1.0),
            VALIDATION_THRESHOLDS["cpcv_overfit_rate_max"],
        ),
        "pbo_selected_below_median_rate": threshold_excess(
            profile.get("pbo_selected_below_median_rate", 1.0),
            VALIDATION_THRESHOLDS["pbo_selected_below_median_rate_max"],
        ),
        "pbo_avg_selected_test_percentile": threshold_shortfall(
            profile.get("pbo_avg_selected_test_percentile", 0.0),
            VALIDATION_THRESHOLDS["pbo_avg_selected_test_percentile_min"],
        ),
        "pbo_selection_share": threshold_shortfall(
            profile.get("pbo_selection_share", 0.0),
            VALIDATION_THRESHOLDS["pbo_selection_share_min"],
        ),
        "market_os_fitness": threshold_shortfall(
            market_os_fitness.get("score", 0.0),
            VALIDATION_THRESHOLDS["market_os_fitness_min"],
        ),
        "daily_win_rate_oos": threshold_shortfall(
            market_os_raw.get("daily_win_rate_oos", 0.0),
            VALIDATION_THRESHOLDS["daily_win_rate_oos_min"],
        ),
        "corr_state_robustness": threshold_shortfall(
            state_summary.get("corr_state_robustness", market_os_raw.get("corr_state_robustness", 0.0)),
            VALIDATION_THRESHOLDS["corr_state_robustness_min"],
        ),
        "special_regime_robustness": threshold_shortfall(
            state_summary.get("special_regime_robustness", market_os_raw.get("special_regime_robustness", 0.0)),
            VALIDATION_THRESHOLDS["special_regime_robustness_min"],
        ),
        "regime_coverage": threshold_shortfall(
            state_summary.get("regime_coverage", market_os_raw.get("regime_coverage", 0.0)),
            VALIDATION_THRESHOLDS["regime_coverage_min"],
        ),
        "parameter_instability": threshold_excess(
            market_os_raw.get("parameter_instability", 1.0),
            VALIDATION_THRESHOLDS["parameter_instability_max"],
        ),
    }
    penalty = float(
        np.clip(
            sum(
                float(components[name]) * float(weight)
                for name, weight in VALIDATION_ROBUSTNESS_WEIGHTS.items()
            ),
            0.0,
            1.0,
        )
    )
    score = float(np.clip(1.0 - penalty, 0.0, 1.0))
    return {
        "score": score,
        "penalty": penalty,
        "reserve": float(score - VALIDATION_THRESHOLDS["validation_quality_score_min"]),
        "gate_pass_ratio": gate_pass_ratio,
        "gate_passed": bool(gate.get("passed", False)),
        "components": components,
        "weights": VALIDATION_ROBUSTNESS_WEIGHTS,
    }


def validation_robustness_score(validation_engine: dict[str, Any] | None) -> float:
    return float(build_validation_robustness_profile(validation_engine).get("score", 0.0))


def build_candidate_validation_bundle(
    candidate_key: str,
    daily_returns: np.ndarray,
    daily_index: pd.DatetimeIndex | list[Any],
    *,
    trial_count: int,
    peer_frames_by_key: dict[str, pd.DataFrame],
    state_payload: dict[str, Any] | None = None,
    cost_reference: dict[str, Any] | None = None,
    cpcv_blocks: int = 6,
    cpcv_test_blocks: int = 2,
    cpcv_embargo_days: int = 2,
) -> dict[str, Any]:
    frame = build_return_frame(daily_returns, daily_index)
    _, metrics = summarize_return_frame(frame)
    dsr_proxy = compute_dsr_proxy(np.asarray(daily_returns, dtype="float64"), trial_count=trial_count)
    cpcv = summarize_cpcv_lite(
        frame,
        n_blocks=cpcv_blocks,
        test_blocks=cpcv_test_blocks,
        embargo_days=cpcv_embargo_days,
    )
    candidate_selection_pbo = summarize_cpcv_candidate_selection_pbo(
        peer_frames_by_key,
        n_blocks=cpcv_blocks,
        test_blocks=cpcv_test_blocks,
        embargo_days=cpcv_embargo_days,
    )
    pbo_profile = candidate_selection_pbo["profiles"].get(candidate_key, empty_candidate_pbo_profile())
    profile = build_validation_profile(dsr_proxy, cpcv, pbo_profile)
    market_operating_system = build_market_operating_system(
        frame,
        trial_count=trial_count,
        cpcv=cpcv,
        pbo_profile=pbo_profile,
        state_payload=state_payload,
        cost_reference=cost_reference,
    )
    gate = build_validation_gate(dsr_proxy, cpcv, profile, market_operating_system)
    robustness = build_validation_robustness_profile(
        {
            "profile": profile,
            "market_operating_system": market_operating_system,
            "gate": gate,
        }
    )
    daily_metrics = metrics["daily_metrics"]
    return {
        "metrics": {
            "total_return": float(metrics["total_return"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "sharpe": float(metrics["sharpe"]),
            "avg_daily_return": float(daily_metrics["avg_daily_return"]),
            "daily_win_rate": float(daily_metrics["daily_win_rate"]),
            "daily_target_hit_rate": float(daily_metrics["daily_target_hit_rate"]),
            "worst_day": float(daily_metrics["worst_day"]),
            "best_day": float(daily_metrics["best_day"]),
        },
        "dsr_proxy": float(dsr_proxy),
        "cpcv": cpcv,
        "candidate_selection_pbo": {
            "n_splits": int(candidate_selection_pbo["n_splits"]),
            "pbo": float(candidate_selection_pbo["pbo"]),
            "avg_selected_test_percentile": float(candidate_selection_pbo["avg_selected_test_percentile"]),
            "worst_selected_test_percentile": float(candidate_selection_pbo["worst_selected_test_percentile"]),
        },
        "pbo_profile": pbo_profile,
        "profile": profile,
        "market_operating_system": market_operating_system,
        "gate": gate,
        "robustness": robustness,
    }
