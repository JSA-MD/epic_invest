#!/usr/bin/env python3
"""Search regime-mixture overlays for a subset of trade pairs."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from replay_regime_mixture_realistic import (
    fetch_funding_rates,
    load_model,
    quantize_amount,
    resolve_candidate,
    summarize,
)
from search_gp_drawdown_overlay import OverlayParams
from validate_pair_subset_summary import build_validation_bundle


UTC = timezone.utc
DEFAULT_WINDOWS = (
    ("recent_2m", "2026-02-06", "2026-04-06"),
    ("recent_6m", "2025-10-06", "2026-04-06"),
    ("full_4y", "2022-04-06", "2026-04-06"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search a regime-mixture overlay using a subset of trade pairs.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_search_summary.json"),
    )
    parser.add_argument("--top-k-realistic", type=int, default=5)
    parser.add_argument(
        "--subset-indices",
        default="0,1,2,5,7",
        help="Library indices allowed in searched mappings.",
    )
    parser.add_argument(
        "--route-thresholds",
        default="0.50,0.65",
        help="Comma-separated route breadth thresholds to test.",
    )
    return parser.parse_args()


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


def build_overlay_inputs(df: pd.DataFrame, pairs: tuple[str, ...], regime_pair: str) -> dict[str, pd.Series]:
    close = pd.concat([df[f"{asset}_close"].rename(asset) for asset in pairs], axis=1).sort_index()
    daily_close = close.resample("1D").last().dropna()
    regime = 0.60 * daily_close[regime_pair].pct_change(3) + 0.40 * daily_close[regime_pair].pct_change(14)
    breadth = (daily_close.pct_change(3) > 0.0).mean(axis=1)
    bar_ret = close[regime_pair].pct_change()
    vol_ann = bar_ret.rolling(12 * 24 * 3).std() * np.sqrt(365.25 * 24.0 * 60.0 / 5.0)
    return {
        "btc_regime_daily": regime,
        "breadth_daily": breadth,
        "vol_ann_bar": vol_ann,
    }


def build_route_bucket_codes(
    index: pd.DatetimeIndex,
    overlay_inputs: dict[str, pd.Series],
    breadth_threshold: float,
) -> np.ndarray:
    day_index = index.normalize()
    regime_daily = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0)
    breadth_daily = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0)
    is_up = (regime_daily >= 0.0).astype(np.int8)
    is_broad = (breadth_daily >= breadth_threshold).astype(np.int8)
    return (is_up * 2 + is_broad).to_numpy(dtype="int8")


def load_or_fetch_funding(symbol: str, start: str, end: str) -> pd.DataFrame:
    path = gp.DATA_DIR / f"{symbol}_funding_{start}_{end}.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, format="mixed")
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df.dropna(subset=["fundingTime", "fundingRate"]).sort_values("fundingTime").reset_index(drop=True)
    df = fetch_funding_rates(
        symbol,
        datetime.fromisoformat(start).replace(tzinfo=UTC),
        datetime.fromisoformat(end).replace(tzinfo=UTC) + pd.Timedelta(days=1),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def summarize_single_result(result: dict[str, Any]) -> dict[str, Any]:
    daily = result["daily_metrics"]
    return {
        "avg_daily_return": float(daily["avg_daily_return"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]),
        "n_trades": int(result["n_trades"]),
        "daily_target_hit_rate": float(daily["daily_target_hit_rate"]),
        "daily_win_rate": float(daily["daily_win_rate"]),
        "worst_day": float(daily["worst_day"]),
        "best_day": float(daily["best_day"]),
    }


def aggregate_metrics(per_pair: dict[str, dict[str, Any]]) -> dict[str, Any]:
    avg_daily = np.asarray([m["avg_daily_return"] for m in per_pair.values()], dtype="float64")
    total = np.asarray([m["total_return"] for m in per_pair.values()], dtype="float64")
    max_dd = np.asarray([m["max_drawdown"] for m in per_pair.values()], dtype="float64")
    return {
        "mean_avg_daily_return": float(np.mean(avg_daily)),
        "worst_pair_avg_daily_return": float(np.min(avg_daily)),
        "best_pair_avg_daily_return": float(np.max(avg_daily)),
        "positive_pair_count": int(np.sum(avg_daily > 0.0)),
        "mean_total_return": float(np.mean(total)),
        "worst_pair_total_return": float(np.min(total)),
        "worst_max_drawdown": float(np.min(max_dd)),
        "pair_return_dispersion": float(np.std(avg_daily)),
    }


def fast_overlay_replay(
    df: pd.DataFrame,
    pair: str,
    raw_signal: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    library: list[OverlayParams],
    mapping: tuple[int, int, int, int],
    route_breadth_threshold: float,
) -> dict[str, Any]:
    close = df[f"{pair}_close"].to_numpy(dtype="float64")
    idx = pd.DatetimeIndex(df.index)
    day_index = idx.normalize()
    bucket_codes = build_route_bucket_codes(idx, overlay_inputs, route_breadth_threshold)
    regime = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    breadth = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    vol_ann = overlay_inputs["vol_ann_bar"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype="float64")
    spans = sorted({params.signal_span for params in library})
    smooth_signals = {
        span: raw_signal.ewm(span=span, adjust=False).mean().to_numpy(dtype="float64")
        for span in spans
    }

    equity = float(gp.INITIAL_CASH)
    peak_equity = float(gp.INITIAL_CASH)
    current_weight = 0.0
    cooldown_bars_left = 0
    net_ret: list[float] = []
    equity_curve = [float(gp.INITIAL_CASH)]
    n_trades = 0

    for i in range(len(df) - 1):
        active_idx = int(mapping[int(bucket_codes[i])])
        params = library[active_idx]

        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = float(np.clip(smooth_signals[params.signal_span][i], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(regime[i])
        breadth_score = float(breadth[i])
        long_ok = regime_score >= params.regime_threshold and breadth_score >= params.breadth_threshold
        short_ok = regime_score <= -params.regime_threshold and breadth_score <= (1.0 - params.breadth_threshold)
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = float(vol_ann[i])
        if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = min(
                params.target_vol_ann / bar_vol_ann,
                params.gross_cap / max(abs(requested_weight), 1e-8),
            )
            requested_weight *= float(vol_scale)
        requested_weight = float(np.clip(requested_weight, -params.gross_cap, params.gross_cap))

        drawdown = equity / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -params.kill_switch_pct and cooldown_bars_left == 0:
            cooldown_bars_left = params.cooldown_days * gp.periods_per_day(gp.TIMEFRAME)

        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif i % params.rebalance_bars == 0:
            target_weight = requested_weight

        if abs(target_weight - current_weight) < gp.NO_TRADE_BAND / 100.0:
            target_weight = current_weight

        turnover = abs(target_weight - current_weight)
        if turnover > 0.001:
            n_trades += 1

        price_ret = float(close[i + 1] / close[i] - 1.0)
        bar_net = target_weight * price_ret - turnover * gp.COMMISSION_PCT * 2
        equity *= (1.0 + bar_net)
        peak_equity = max(peak_equity, equity)
        current_weight = target_weight
        net_ret.append(bar_net)
        equity_curve.append(float(equity))

    return {
        "total_return": float(equity / gp.INITIAL_CASH - 1.0),
        "n_trades": int(n_trades),
        "sharpe": float(np.mean(net_ret) / np.std(net_ret) * np.sqrt(365.25 * 24 * 60.0 / 5.0)) if len(net_ret) > 1 and np.std(net_ret) > 1e-12 else 0.0,
        "max_drawdown": float(np.min(np.asarray(equity_curve) / np.maximum.accumulate(np.asarray(equity_curve)) - 1.0)),
        "final_equity": float(equity),
        "daily_metrics": gp.compute_daily_metrics(np.asarray(net_ret, dtype="float64")),
    }


def realistic_overlay_replay(
    df: pd.DataFrame,
    trade_pair: str,
    raw_signal: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    funding_df: pd.DataFrame,
    library: list[OverlayParams],
    mapping: tuple[int, int, int, int],
    route_breadth_threshold: float,
) -> dict[str, Any]:
    idx = pd.DatetimeIndex(df.index)
    open_p = df[f"{trade_pair}_open"].to_numpy(dtype="float64")
    close_p = df[f"{trade_pair}_close"].to_numpy(dtype="float64")
    vol_ann = overlay_inputs["vol_ann_bar"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype="float64")
    day_index = idx.normalize()
    regime = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    breadth = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    bucket_codes = build_route_bucket_codes(idx, overlay_inputs, route_breadth_threshold)
    spans = sorted({params.signal_span for params in library})
    smooth_signals = {
        span: raw_signal.ewm(span=span, adjust=False).mean().to_numpy(dtype="float64")
        for span in spans
    }

    funding_map = {}
    if not funding_df.empty:
        for _, row in funding_df.iterrows():
            funding_map[pd.Timestamp(row["fundingTime"]).tz_convert("UTC")] = float(row["fundingRate"])

    cash = float(gp.INITIAL_CASH)
    qty = 0.0
    n_trades = 0
    fee_paid = 0.0
    slippage_paid = 0.0
    funding_paid = 0.0
    funding_events = 0
    net_ret: list[float] = []
    equity_curve = [float(gp.INITIAL_CASH)]
    peak_equity = float(gp.INITIAL_CASH)
    cooldown_bars_left = 0

    for exec_idx in range(1, len(df) - 1):
        signal_idx = exec_idx - 1
        ts_open = pd.Timestamp(idx[exec_idx])
        px_open = float(open_p[exec_idx])
        next_open = float(open_p[exec_idx + 1])
        prev_close = float(close_p[signal_idx])

        if qty != 0.0 and ts_open in funding_map:
            funding_rate = funding_map[ts_open]
            funding_cashflow = -qty * px_open * funding_rate
            cash += funding_cashflow
            funding_paid += funding_cashflow
            funding_events += 1

        equity_before = cash + qty * px_open
        if equity_before <= 1e-9:
            equity_before = 1e-9
        peak_equity = max(peak_equity, equity_before)

        active_idx = int(mapping[int(bucket_codes[signal_idx])])
        params = library[active_idx]
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = float(np.clip(smooth_signals[params.signal_span][signal_idx], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(regime[signal_idx])
        breadth_score = float(breadth[signal_idx])
        long_ok = regime_score >= params.regime_threshold and breadth_score >= params.breadth_threshold
        short_ok = regime_score <= -params.regime_threshold and breadth_score <= (1.0 - params.breadth_threshold)
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = float(vol_ann[signal_idx])
        if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = min(
                params.target_vol_ann / bar_vol_ann,
                params.gross_cap / max(abs(requested_weight), 1e-8),
            )
            requested_weight *= float(vol_scale)
        requested_weight = float(np.clip(requested_weight, -params.gross_cap, params.gross_cap))

        drawdown = equity_before / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -params.kill_switch_pct and cooldown_bars_left == 0:
            cooldown_bars_left = params.cooldown_days * gp.periods_per_day(gp.TIMEFRAME)

        current_weight = qty * px_open / equity_before if abs(equity_before) > 1e-9 else 0.0
        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif signal_idx % params.rebalance_bars == 0:
            target_weight = requested_weight

        if abs(target_weight - current_weight) < gp.NO_TRADE_BAND / 100.0:
            target_weight = current_weight

        target_notional = equity_before * target_weight
        target_qty = quantize_amount(target_notional / prev_close if abs(prev_close) > 1e-12 else 0.0, 0.001, 0.001)
        diff_qty = quantize_amount(target_qty - qty, 0.001, 0.001)

        if abs(diff_qty) > 0.0:
            side = 1.0 if diff_qty > 0.0 else -1.0
            exec_price = px_open * (1.0 + 0.0002 * side)
            trade_notional = diff_qty * exec_price
            fee = abs(diff_qty) * exec_price * 0.0004
            cash -= trade_notional
            cash -= fee
            qty += diff_qty
            n_trades += 1
            fee_paid += fee
            slippage_paid += abs(diff_qty) * px_open * 0.0002

        equity_after = cash + qty * next_open
        net_ret.append(float(equity_after / equity_before - 1.0))
        equity_curve.append(float(equity_after))

    return summarize(
        np.asarray(net_ret, dtype="float64"),
        np.asarray(equity_curve, dtype="float64"),
        n_trades=n_trades,
        fee_paid=fee_paid,
        slippage_paid=slippage_paid,
        funding_paid=funding_paid,
        funding_events=funding_events,
    )


def score_candidate(agg_2m: dict[str, Any], agg_6m: dict[str, Any]) -> float:
    score = 0.0
    score -= agg_6m["mean_avg_daily_return"] * 250000.0
    score -= agg_2m["mean_avg_daily_return"] * 180000.0
    score -= agg_6m["worst_pair_avg_daily_return"] * 220000.0
    score -= agg_2m["worst_pair_avg_daily_return"] * 160000.0
    score -= agg_6m["mean_total_return"] * 12000.0
    score += abs(agg_6m["worst_max_drawdown"]) * 18000.0
    score += abs(agg_2m["worst_max_drawdown"]) * 15000.0
    score += agg_6m["pair_return_dispersion"] * 120000.0
    score += agg_2m["pair_return_dispersion"] * 100000.0
    return float(score)


def score_realistic_candidate(report: dict[str, Any]) -> float:
    recent_2m = report["windows"]["recent_2m"]["aggregate"]
    recent_6m = report["windows"]["recent_6m"]["aggregate"]
    full_4y = report["windows"]["full_4y"]["aggregate"]

    score = 0.0
    score += float(recent_2m["worst_pair_avg_daily_return"]) * 420000.0
    score += float(recent_6m["worst_pair_avg_daily_return"]) * 320000.0
    score += float(full_4y["worst_pair_avg_daily_return"]) * 240000.0
    score += float(full_4y["mean_avg_daily_return"]) * 180000.0
    score += float(recent_2m["mean_avg_daily_return"]) * 60000.0
    score += float(recent_6m["mean_avg_daily_return"]) * 40000.0
    score -= abs(float(recent_2m["worst_max_drawdown"])) * 18000.0
    score -= abs(float(recent_6m["worst_max_drawdown"])) * 14000.0
    score -= abs(float(full_4y["worst_max_drawdown"])) * 9000.0
    score -= float(recent_2m["pair_return_dispersion"]) * 120000.0
    score -= float(recent_6m["pair_return_dispersion"]) * 90000.0
    score -= float(full_4y["pair_return_dispersion"]) * 60000.0
    return float(score)


def parse_csv_tuple(raw: str, cast) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def main() -> None:
    args = parse_args()
    pairs = parse_csv_tuple(args.pairs, str)
    subset_indices = parse_csv_tuple(args.subset_indices, int)
    route_thresholds = parse_csv_tuple(args.route_thresholds, float)

    baseline_candidate, library, _ = resolve_candidate(Path(args.summary), None, None)
    model, _ = load_model(Path(args.model))
    compiled = gp.toolbox.compile(expr=model)

    start_all = DEFAULT_WINDOWS[-1][1]
    end_all = DEFAULT_WINDOWS[-1][2]
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=False)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {pair: load_or_fetch_funding(pair, start_all, end_all) for pair in pairs}

    baseline_fast = {}
    baseline_realistic = {}
    for label, start, end in DEFAULT_WINDOWS:
        df = df_all.loc[start:end].copy()
        per_pair_fast = {}
        per_pair_realistic = {}
        for pair in pairs:
            overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
            signal_slice = raw_signal_all[pair].loc[start:end].copy()
            per_pair_fast[pair] = summarize_single_result(
                fast_overlay_replay(
                    df,
                    pair,
                    signal_slice,
                    overlay_inputs,
                    library,
                    baseline_candidate.mapping_indices,
                    baseline_candidate.route_breadth_threshold,
                )
            )
            funding_slice = funding_all[pair]
            if not funding_slice.empty:
                funding_slice = funding_slice[
                    (funding_slice["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                    & (funding_slice["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
                ].copy()
            per_pair_realistic[pair] = realistic_overlay_replay(
                df,
                pair,
                signal_slice,
                overlay_inputs,
                funding_slice,
                library,
                baseline_candidate.mapping_indices,
                baseline_candidate.route_breadth_threshold,
            )
        baseline_fast[label] = aggregate_metrics(per_pair_fast)
        baseline_realistic[label] = {
            "start": start,
            "end": end,
            "bars": int(len(df)),
            "per_pair": per_pair_realistic,
            "aggregate": aggregate_metrics(per_pair_realistic),
        }

    candidate_pool = [
        (route_threshold, tuple(int(v) for v in mapping))
        for route_threshold in route_thresholds
        for mapping in itertools.product(subset_indices, repeat=4)
    ]
    if (baseline_candidate.route_breadth_threshold, baseline_candidate.mapping_indices) not in candidate_pool:
        candidate_pool.append((baseline_candidate.route_breadth_threshold, baseline_candidate.mapping_indices))

    scored = []
    for route_threshold, mapping in candidate_pool:
        windows = {}
        for label, start, end in DEFAULT_WINDOWS[:2]:
            df = df_all.loc[start:end].copy()
            per_pair = {}
            for pair in pairs:
                overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
                per_pair[pair] = summarize_single_result(
                    fast_overlay_replay(
                        df,
                        pair,
                        raw_signal_all[pair].loc[start:end].copy(),
                        overlay_inputs,
                        library,
                        mapping,
                        route_threshold,
                    )
                )
            windows[label] = aggregate_metrics(per_pair)

        recent_2m = windows["recent_2m"]
        recent_6m = windows["recent_6m"]
        if recent_2m["positive_pair_count"] < len(pairs) or recent_6m["positive_pair_count"] < len(pairs):
            continue

        scored.append(
            {
                "route_breadth_threshold": route_threshold,
                "mapping_indices": list(mapping),
                "recent_2m": recent_2m,
                "recent_6m": recent_6m,
                "score": score_candidate(recent_2m, recent_6m),
            }
        )

    scored.sort(key=lambda item: item["score"])
    top_fast = scored[: args.top_k_realistic]

    realistic_top = []
    for item in top_fast:
        route_threshold = float(item["route_breadth_threshold"])
        mapping = tuple(int(v) for v in item["mapping_indices"])
        windows = {}
        for label, start, end in DEFAULT_WINDOWS:
            df = df_all.loc[start:end].copy()
            per_pair = {}
            for pair in pairs:
                overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
                funding_slice = funding_all[pair]
                if not funding_slice.empty:
                    funding_slice = funding_slice[
                        (funding_slice["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                        & (funding_slice["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
                    ].copy()
                per_pair[pair] = realistic_overlay_replay(
                    df,
                    pair,
                    raw_signal_all[pair].loc[start:end].copy(),
                    overlay_inputs,
                    funding_slice,
                    library,
                    mapping,
                    route_threshold,
                )
            windows[label] = {
                "start": start,
                "end": end,
                "bars": int(len(df)),
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
        realistic_top.append(
            {
                "route_breadth_threshold": route_threshold,
                "mapping_indices": list(mapping),
                "score": item["score"],
                "windows": windows,
                "validation": build_validation_bundle(windows, baseline_realistic),
            }
        )

    progressive_candidates = [
        item
        for item in realistic_top
        if item["validation"]["profiles"]["progressive_improvement"]["passed"]
    ]
    target_060_candidates = [
        item
        for item in realistic_top
        if item["validation"]["profiles"]["target_060"]["passed"]
    ]
    fallback_best = max(realistic_top, key=score_realistic_candidate) if realistic_top else None
    selected = max(target_060_candidates, key=score_realistic_candidate) if target_060_candidates else None
    selection_reason = "target_060_pass"
    if selected is None and progressive_candidates:
        selected = max(progressive_candidates, key=score_realistic_candidate)
        selection_reason = "progressive_pass"
    if selected is None:
        selection_reason = "no_gate_pass"

    report = {
        "pairs": list(pairs),
        "baseline_candidate": {
            "route_breadth_threshold": baseline_candidate.route_breadth_threshold,
            "mapping_indices": list(baseline_candidate.mapping_indices),
        },
        "baseline_fast": baseline_fast,
        "baseline_realistic": baseline_realistic,
        "top_fast_candidates": top_fast,
        "realistic_top_candidates": realistic_top,
        "promotion_candidates": {
            "target_060": [
                {
                    "route_breadth_threshold": item["route_breadth_threshold"],
                    "mapping_indices": item["mapping_indices"],
                }
                for item in target_060_candidates
            ],
            "progressive_improvement": [
                {
                    "route_breadth_threshold": item["route_breadth_threshold"],
                    "mapping_indices": item["mapping_indices"],
                }
                for item in progressive_candidates
            ],
        },
        "selection": {
            "reason": selection_reason,
            "target_060_pass_count": len(target_060_candidates),
            "progressive_pass_count": len(progressive_candidates),
            "realistic_top_count": len(realistic_top),
        },
        "selected_candidate": selected,
        "fallback_best_candidate": fallback_best,
        "created_at": datetime.now(UTC).isoformat(),
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
