#!/usr/bin/env python3
"""Search regime-mixture overlays for a subset of trade pairs."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    njit = None
    NUMBA_AVAILABLE = False

import gp_crypto_evolution as gp
from equity_corr_regime import build_btc_equity_corr_overlay
from replay_regime_mixture_realistic import (
    fetch_funding_rates,
    load_model,
    resolve_candidate,
)
from search_gp_drawdown_overlay import OverlayParams
from validate_pair_subset_summary import build_validation_bundle


UTC = timezone.utc
BARS_PER_DAY = gp.periods_per_day(gp.TIMEFRAME)
BAR_FACTOR = np.sqrt(365.25 * 24.0 * 60.0 / 5.0)
DEFAULT_WINDOWS = (
    ("recent_2m", "2026-02-06", "2026-04-06"),
    ("recent_6m", "2025-10-06", "2026-04-06"),
    ("full_4y", "2022-04-06", "2026-04-06"),
)
ROUTE_STATE_MODE_BASE = "base"
ROUTE_STATE_MODE_EQUITY_CORR = "equity_corr"
BASE_ROUTE_STATE_NAMES = (
    "bear_narrow",
    "bear_broad",
    "bull_narrow",
    "bull_broad",
)
EQUITY_CORR_ROUTE_BUCKETS = (
    "equity_inverse",
    "equity_mixed",
    "equity_aligned",
)
EQUITY_CORR_BUCKET_CODES = {
    "equity_inverse": 0,
    "equity_mixed": 1,
    "equity_aligned": 2,
    "equity_unknown": 1,
}
EQUITY_CORR_ROUTE_STATE_NAMES = tuple(
    f"{corr_bucket}:{base_state}"
    for corr_bucket in EQUITY_CORR_ROUTE_BUCKETS
    for base_state in BASE_ROUTE_STATE_NAMES
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
    parser.add_argument(
        "--fast-engine",
        choices=("auto", "python", "numba"),
        default="auto",
        help="Fast replay engine for the candidate pre-search stage.",
    )
    parser.add_argument(
        "--route-state-mode",
        choices=(ROUTE_STATE_MODE_BASE, ROUTE_STATE_MODE_EQUITY_CORR),
        default=ROUTE_STATE_MODE_BASE,
        help="Route state space: base=4-state, equity_corr=12-state (BTC regime x breadth x BTC-equity corr).",
    )
    return parser.parse_args()


def resolve_fast_engine(requested: str) -> str:
    if requested == "python":
        return "python"
    if requested == "numba":
        if not NUMBA_AVAILABLE:
            raise RuntimeError("Numba fast engine requested but numba is not installed.")
        return "numba"
    return "numba" if NUMBA_AVAILABLE else "python"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_route_state_mode(route_state_mode: str | None) -> str:
    mode = str(route_state_mode or ROUTE_STATE_MODE_BASE).strip().lower()
    if mode not in {ROUTE_STATE_MODE_BASE, ROUTE_STATE_MODE_EQUITY_CORR}:
        raise ValueError(f"Unsupported route_state_mode: {route_state_mode}")
    return mode


def route_state_names(route_state_mode: str) -> tuple[str, ...]:
    mode = normalize_route_state_mode(route_state_mode)
    if mode == ROUTE_STATE_MODE_BASE:
        return BASE_ROUTE_STATE_NAMES
    return EQUITY_CORR_ROUTE_STATE_NAMES


def route_state_count(route_state_mode: str) -> int:
    return len(route_state_names(route_state_mode))


def normalize_mapping_indices(mapping: tuple[int, ...] | list[int], route_state_mode: str) -> tuple[int, ...]:
    mode = normalize_route_state_mode(route_state_mode)
    values = tuple(int(v) for v in mapping)
    if mode == ROUTE_STATE_MODE_BASE:
        if len(values) == len(BASE_ROUTE_STATE_NAMES):
            return values
        if (
            len(values) == len(EQUITY_CORR_ROUTE_STATE_NAMES)
            and values[:4] == values[4:8]
            and values[:4] == values[8:12]
        ):
            return values[:4]
        raise ValueError("Base route_state_mode requires four indices or a compressible repeated 12-state mapping")
    if len(values) == len(EQUITY_CORR_ROUTE_STATE_NAMES):
        return values
    if len(values) == len(BASE_ROUTE_STATE_NAMES):
        return values * len(EQUITY_CORR_ROUTE_BUCKETS)
    raise ValueError("Equity correlation route_state_mode requires four or twelve indices")


def build_library_lookup(library: list[OverlayParams]) -> dict[str, Any]:
    spans = sorted({params.signal_span for params in library})
    span_to_pos = {span: idx for idx, span in enumerate(spans)}
    return {
        "spans": tuple(spans),
        "signal_pos": np.asarray([span_to_pos[params.signal_span] for params in library], dtype="int64"),
        "rebalance_bars": np.asarray([params.rebalance_bars for params in library], dtype="int64"),
        "regime_threshold": np.asarray([params.regime_threshold for params in library], dtype="float64"),
        "breadth_threshold": np.asarray([params.breadth_threshold for params in library], dtype="float64"),
        "target_vol_ann": np.asarray([params.target_vol_ann for params in library], dtype="float64"),
        "gross_cap": np.asarray([params.gross_cap for params in library], dtype="float64"),
        "kill_switch_pct": np.asarray([params.kill_switch_pct for params in library], dtype="float64"),
        "cooldown_days": np.asarray([params.cooldown_days for params in library], dtype="int64"),
    }


def build_fast_context(
    df: pd.DataFrame,
    pair: str,
    raw_signal: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    route_thresholds: tuple[float, ...],
    library_lookup: dict[str, Any],
    funding_df: pd.DataFrame | None = None,
    route_state_mode: str = ROUTE_STATE_MODE_BASE,
) -> dict[str, Any]:
    route_state_mode = normalize_route_state_mode(route_state_mode)
    idx = pd.DatetimeIndex(df.index)
    day_index = idx.normalize()
    spans = library_lookup["spans"]
    smooth_signal_matrix = np.vstack(
        [
            raw_signal.ewm(span=span, adjust=False).mean().to_numpy(dtype="float64")
            for span in spans
        ]
    )
    bucket_codes = {
        float(threshold): build_route_bucket_codes(
            idx,
            overlay_inputs,
            float(threshold),
            route_state_mode=route_state_mode,
        ).astype("int64")
        for threshold in route_thresholds
    }
    funding_rates = np.zeros(len(idx), dtype="float64")
    validation_daily_index = pd.DatetimeIndex(day_index.unique())
    if funding_df is not None and not funding_df.empty:
        funding_series = (
            funding_df[["fundingTime", "fundingRate"]]
            .dropna(subset=["fundingTime", "fundingRate"])
            .assign(fundingTime=lambda frame: pd.to_datetime(frame["fundingTime"], utc=True))
            .drop_duplicates(subset=["fundingTime"], keep="last")
            .set_index("fundingTime")["fundingRate"]
        )
        funding_rates = (
            funding_series.reindex(idx, fill_value=0.0)
            .to_numpy(dtype="float64")
        )
    equity_corr_daily = overlay_inputs.get(
        "equity_corr_daily",
        pd.Series(0.0, index=validation_daily_index, dtype="float64"),
    )
    equity_corr_gross_scale_daily = overlay_inputs.get(
        "equity_corr_gross_scale_daily",
        pd.Series(1.0, index=validation_daily_index, dtype="float64"),
    )
    equity_corr_regime_mult_daily = overlay_inputs.get(
        "equity_corr_regime_threshold_mult_daily",
        pd.Series(1.0, index=validation_daily_index, dtype="float64"),
    )
    return {
        "open": df[f"{pair}_open"].to_numpy(dtype="float64"),
        "close": df[f"{pair}_close"].to_numpy(dtype="float64"),
        "bucket_codes": bucket_codes,
        "bar_day_index": pd.DatetimeIndex(day_index),
        "regime": overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64"),
        "breadth": overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64"),
        "vol_ann": overlay_inputs["vol_ann_bar"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype="float64"),
        "equity_corr": equity_corr_daily.reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64"),
        "equity_corr_gross_scale": equity_corr_gross_scale_daily.reindex(day_index, method="ffill").fillna(1.0).to_numpy(dtype="float64"),
        "equity_corr_regime_mult": equity_corr_regime_mult_daily.reindex(day_index, method="ffill").fillna(1.0).to_numpy(dtype="float64"),
        "smooth_signal_matrix": smooth_signal_matrix,
        "funding_rates": funding_rates,
        "equity_corr_context": overlay_inputs.get("equity_corr_context"),
        "equity_corr_source_mode": overlay_inputs.get("equity_corr_source_mode"),
        "route_state_mode": route_state_mode,
        "route_state_names": route_state_names(route_state_mode),
        "validation_daily_index": validation_daily_index,
    }


def _fast_overlay_replay_kernel_impl(
    close: np.ndarray,
    bucket_codes: np.ndarray,
    regime: np.ndarray,
    breadth: np.ndarray,
    vol_ann: np.ndarray,
    equity_corr_gross_scale: np.ndarray,
    equity_corr_regime_mult: np.ndarray,
    smooth_signal_matrix: np.ndarray,
    library_signal_pos: np.ndarray,
    library_rebalance_bars: np.ndarray,
    library_regime_threshold: np.ndarray,
    library_breadth_threshold: np.ndarray,
    library_target_vol_ann: np.ndarray,
    library_gross_cap: np.ndarray,
    library_kill_switch_pct: np.ndarray,
    library_cooldown_days: np.ndarray,
    mapping: np.ndarray,
    initial_cash: float,
    commission_pct: float,
    no_trade_band_pct: float,
    bars_per_day: int,
    daily_target: float,
    bar_factor: float,
) -> tuple[float, int, float, float, float, float, float, float, float]:
    equity = initial_cash
    peak_equity = initial_cash
    current_weight = 0.0
    cooldown_bars_left = 0
    n_trades = 0
    max_drawdown = 0.0

    mean_bar = 0.0
    m2_bar = 0.0
    bar_count = 0

    day_accum = 1.0
    day_len = 0
    day_count = 0
    day_sum = 0.0
    day_wins = 0
    day_hits = 0
    worst_day = 0.0
    best_day = 0.0

    for i in range(close.shape[0] - 1):
        active_idx = mapping[bucket_codes[i]]
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = smooth_signal_matrix[library_signal_pos[active_idx], i]
        if signal_pct > 500.0:
            signal_pct = 500.0
        elif signal_pct < -500.0:
            signal_pct = -500.0

        requested_weight = signal_pct / 100.0
        regime_score = regime[i]
        breadth_score = breadth[i]
        effective_regime_threshold = library_regime_threshold[active_idx] * equity_corr_regime_mult[i]
        effective_gross_cap = library_gross_cap[active_idx] * equity_corr_gross_scale[i]
        long_ok = regime_score >= effective_regime_threshold and breadth_score >= library_breadth_threshold[active_idx]
        short_ok = regime_score <= -effective_regime_threshold and breadth_score <= (1.0 - library_breadth_threshold[active_idx])
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = vol_ann[i]
        if bar_vol_ann == bar_vol_ann and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = library_target_vol_ann[active_idx] / bar_vol_ann
            gross_scale = effective_gross_cap / max(abs(requested_weight), 1e-8)
            if gross_scale < vol_scale:
                vol_scale = gross_scale
            requested_weight *= vol_scale

        gross_cap = effective_gross_cap
        if requested_weight > gross_cap:
            requested_weight = gross_cap
        elif requested_weight < -gross_cap:
            requested_weight = -gross_cap

        drawdown = equity / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -library_kill_switch_pct[active_idx] and cooldown_bars_left == 0:
            cooldown_bars_left = library_cooldown_days[active_idx] * bars_per_day

        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif i % library_rebalance_bars[active_idx] == 0:
            target_weight = requested_weight

        if abs(target_weight - current_weight) < no_trade_band_pct / 100.0:
            target_weight = current_weight

        turnover = abs(target_weight - current_weight)
        if turnover > 0.001:
            n_trades += 1

        price_ret = close[i + 1] / close[i] - 1.0
        bar_net = target_weight * price_ret - turnover * commission_pct * 2.0

        equity *= (1.0 + bar_net)
        if equity > peak_equity:
            peak_equity = equity
        current_weight = target_weight

        dd = equity / peak_equity - 1.0
        if dd < max_drawdown:
            max_drawdown = dd

        bar_count += 1
        delta = bar_net - mean_bar
        mean_bar += delta / bar_count
        m2_bar += delta * (bar_net - mean_bar)

        day_accum *= (1.0 + bar_net)
        day_len += 1
        if day_len == bars_per_day or i == close.shape[0] - 2:
            day_ret = day_accum - 1.0
            day_sum += day_ret
            day_count += 1
            if day_ret > 0.0:
                day_wins += 1
            if day_ret >= daily_target:
                day_hits += 1
            if day_count == 1 or day_ret < worst_day:
                worst_day = day_ret
            if day_count == 1 or day_ret > best_day:
                best_day = day_ret
            day_accum = 1.0
            day_len = 0

    total_return = equity / initial_cash - 1.0
    sharpe = 0.0
    if bar_count > 1:
        variance = m2_bar / bar_count
        if variance > 1e-12:
            sharpe = mean_bar / np.sqrt(variance) * bar_factor

    avg_daily = 0.0 if day_count == 0 else day_sum / day_count
    daily_target_hit_rate = 0.0 if day_count == 0 else day_hits / day_count
    daily_win_rate = 0.0 if day_count == 0 else day_wins / day_count

    return (
        total_return,
        n_trades,
        sharpe,
        max_drawdown,
        equity,
        avg_daily,
        daily_target_hit_rate,
        daily_win_rate,
        worst_day,
        best_day,
    )


if NUMBA_AVAILABLE:
    _fast_overlay_replay_kernel = njit(cache=True)(_fast_overlay_replay_kernel_impl)
else:  # pragma: no cover - fallback for environments without numba
    _fast_overlay_replay_kernel = _fast_overlay_replay_kernel_impl


def _quantize_amount_kernel(value: float, step: float, min_qty: float) -> float:
    sign = 1.0 if value >= 0.0 else -1.0
    raw = abs(value)
    if raw < min_qty:
        return 0.0
    precise = np.floor(raw / step + 1e-12) * step
    if precise < min_qty:
        return 0.0
    return sign * precise


if NUMBA_AVAILABLE:
    _quantize_amount_nb = njit(cache=True)(_quantize_amount_kernel)
else:  # pragma: no cover - fallback for environments without numba
    _quantize_amount_nb = _quantize_amount_kernel


def _realistic_overlay_replay_kernel_impl(
    open_p: np.ndarray,
    close_p: np.ndarray,
    funding_rates: np.ndarray,
    bucket_codes: np.ndarray,
    regime: np.ndarray,
    breadth: np.ndarray,
    vol_ann: np.ndarray,
    equity_corr_gross_scale: np.ndarray,
    equity_corr_regime_mult: np.ndarray,
    smooth_signal_matrix: np.ndarray,
    library_signal_pos: np.ndarray,
    library_rebalance_bars: np.ndarray,
    library_regime_threshold: np.ndarray,
    library_breadth_threshold: np.ndarray,
    library_target_vol_ann: np.ndarray,
    library_gross_cap: np.ndarray,
    library_kill_switch_pct: np.ndarray,
    library_cooldown_days: np.ndarray,
    mapping: np.ndarray,
    initial_cash: float,
    fee_rate: float,
    slippage: float,
    amount_step: float,
    min_qty: float,
    no_trade_band_pct: float,
    bars_per_day: int,
    daily_target: float,
    bar_factor: float,
) -> tuple[float, int, float, float, float, float, float, float, float, float, float, float, float, int]:
    cash = initial_cash
    qty = 0.0
    n_trades = 0
    fee_paid = 0.0
    slippage_paid = 0.0
    funding_paid = 0.0
    funding_events = 0
    peak_equity = initial_cash
    max_drawdown = 0.0
    cooldown_bars_left = 0

    mean_bar = 0.0
    m2_bar = 0.0
    bar_count = 0

    day_accum = 1.0
    day_len = 0
    day_count = 0
    day_sum = 0.0
    day_wins = 0
    day_hits = 0
    worst_day = 0.0
    best_day = 0.0

    for exec_idx in range(1, open_p.shape[0] - 1):
        signal_idx = exec_idx - 1
        px_open = open_p[exec_idx]
        next_open = open_p[exec_idx + 1]
        prev_close = close_p[signal_idx]

        funding_rate = funding_rates[exec_idx]
        if qty != 0.0 and funding_rate != 0.0:
            funding_cashflow = -qty * px_open * funding_rate
            cash += funding_cashflow
            funding_paid += funding_cashflow
            funding_events += 1

        equity_before = cash + qty * px_open
        if equity_before <= 1e-9:
            equity_before = 1e-9
        if equity_before > peak_equity:
            peak_equity = equity_before

        active_idx = mapping[bucket_codes[signal_idx]]
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = smooth_signal_matrix[library_signal_pos[active_idx], signal_idx]
        if signal_pct > 500.0:
            signal_pct = 500.0
        elif signal_pct < -500.0:
            signal_pct = -500.0

        requested_weight = signal_pct / 100.0
        regime_score = regime[signal_idx]
        breadth_score = breadth[signal_idx]
        effective_regime_threshold = library_regime_threshold[active_idx] * equity_corr_regime_mult[signal_idx]
        effective_gross_cap = library_gross_cap[active_idx] * equity_corr_gross_scale[signal_idx]
        long_ok = regime_score >= effective_regime_threshold and breadth_score >= library_breadth_threshold[active_idx]
        short_ok = regime_score <= -effective_regime_threshold and breadth_score <= (1.0 - library_breadth_threshold[active_idx])
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = vol_ann[signal_idx]
        if bar_vol_ann == bar_vol_ann and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = library_target_vol_ann[active_idx] / bar_vol_ann
            gross_scale = effective_gross_cap / max(abs(requested_weight), 1e-8)
            if gross_scale < vol_scale:
                vol_scale = gross_scale
            requested_weight *= vol_scale

        gross_cap = effective_gross_cap
        if requested_weight > gross_cap:
            requested_weight = gross_cap
        elif requested_weight < -gross_cap:
            requested_weight = -gross_cap

        drawdown = equity_before / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -library_kill_switch_pct[active_idx] and cooldown_bars_left == 0:
            cooldown_bars_left = library_cooldown_days[active_idx] * bars_per_day

        current_weight = 0.0
        if abs(equity_before) > 1e-9:
            current_weight = qty * px_open / equity_before
        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif signal_idx % library_rebalance_bars[active_idx] == 0:
            target_weight = requested_weight

        if abs(target_weight - current_weight) < no_trade_band_pct / 100.0:
            target_weight = current_weight

        target_notional = equity_before * target_weight
        target_qty = 0.0
        if abs(prev_close) > 1e-12:
            target_qty = _quantize_amount_nb(target_notional / prev_close, amount_step, min_qty)
        diff_qty = _quantize_amount_nb(target_qty - qty, amount_step, min_qty)

        if abs(diff_qty) > 0.0:
            side = 1.0 if diff_qty > 0.0 else -1.0
            exec_price = px_open * (1.0 + slippage * side)
            trade_notional = diff_qty * exec_price
            fee = abs(diff_qty) * exec_price * fee_rate
            cash -= trade_notional
            cash -= fee
            qty += diff_qty
            n_trades += 1
            fee_paid += fee
            slippage_paid += abs(diff_qty) * px_open * slippage

        equity_after = cash + qty * next_open
        if equity_after > peak_equity:
            peak_equity = equity_after
        dd = equity_after / peak_equity - 1.0
        if dd < max_drawdown:
            max_drawdown = dd

        bar_net = equity_after / equity_before - 1.0
        bar_count += 1
        delta = bar_net - mean_bar
        mean_bar += delta / bar_count
        m2_bar += delta * (bar_net - mean_bar)

        day_accum *= (1.0 + bar_net)
        day_len += 1
        if day_len == bars_per_day or exec_idx == open_p.shape[0] - 2:
            day_ret = day_accum - 1.0
            day_sum += day_ret
            day_count += 1
            if day_ret > 0.0:
                day_wins += 1
            if day_ret >= daily_target:
                day_hits += 1
            if day_count == 1 or day_ret < worst_day:
                worst_day = day_ret
            if day_count == 1 or day_ret > best_day:
                best_day = day_ret
            day_accum = 1.0
            day_len = 0

    total_return = cash + qty * open_p[-1]
    total_return = total_return / initial_cash - 1.0
    sharpe = 0.0
    if bar_count > 1:
        variance = m2_bar / bar_count
        if variance > 1e-12:
            sharpe = mean_bar / np.sqrt(variance) * bar_factor

    avg_daily = 0.0 if day_count == 0 else day_sum / day_count
    daily_target_hit_rate = 0.0 if day_count == 0 else day_hits / day_count
    daily_win_rate = 0.0 if day_count == 0 else day_wins / day_count
    final_equity = cash + qty * open_p[-1]

    return (
        total_return,
        n_trades,
        sharpe,
        max_drawdown,
        final_equity,
        avg_daily,
        daily_target_hit_rate,
        daily_win_rate,
        worst_day,
        best_day,
        fee_paid,
        slippage_paid,
        funding_paid,
        funding_events,
    )


if NUMBA_AVAILABLE:
    _realistic_overlay_replay_kernel = njit(cache=True)(_realistic_overlay_replay_kernel_impl)
else:  # pragma: no cover - fallback for environments without numba
    _realistic_overlay_replay_kernel = _realistic_overlay_replay_kernel_impl


def build_overlay_inputs(df: pd.DataFrame, pairs: tuple[str, ...], regime_pair: str) -> dict[str, pd.Series]:
    close = pd.concat([df[f"{asset}_close"].rename(asset) for asset in pairs], axis=1).sort_index()
    daily_close = close.resample("1D").last().dropna()
    regime = 0.60 * daily_close[regime_pair].pct_change(3) + 0.40 * daily_close[regime_pair].pct_change(14)
    breadth = (daily_close.pct_change(3) > 0.0).mean(axis=1)
    bar_ret = close[regime_pair].pct_change()
    vol_ann = bar_ret.rolling(12 * 24 * 3).std() * np.sqrt(365.25 * 24.0 * 60.0 / 5.0)
    overlay = {
        "btc_regime_daily": regime,
        "breadth_daily": breadth,
        "vol_ann_bar": vol_ann,
    }
    overlay.update(build_btc_equity_corr_overlay(close))
    return overlay


def build_route_bucket_codes(
    index: pd.DatetimeIndex,
    overlay_inputs: dict[str, pd.Series],
    breadth_threshold: float,
    route_state_mode: str = ROUTE_STATE_MODE_BASE,
) -> np.ndarray:
    route_state_mode = normalize_route_state_mode(route_state_mode)
    day_index = index.normalize()
    regime_daily = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0)
    breadth_daily = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0)
    is_up = (regime_daily >= 0.0).astype(np.int8)
    is_broad = (breadth_daily >= breadth_threshold).astype(np.int8)
    base_codes = (is_up * 2 + is_broad).to_numpy(dtype="int8")
    if route_state_mode == ROUTE_STATE_MODE_BASE:
        return base_codes
    corr_bucket = (
        overlay_inputs["equity_corr_bucket_daily"]
        .reindex(day_index, method="ffill")
        .fillna("equity_unknown")
    )
    corr_codes = corr_bucket.map(EQUITY_CORR_BUCKET_CODES).fillna(EQUITY_CORR_BUCKET_CODES["equity_unknown"]).astype("int8")
    return corr_codes.to_numpy(dtype="int8") * len(BASE_ROUTE_STATE_NAMES) + base_codes


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


def fast_overlay_replay_from_context(
    context: dict[str, Any],
    library: list[OverlayParams],
    library_lookup: dict[str, Any],
    mapping: tuple[int, ...],
    route_breadth_threshold: float,
    fast_engine: str,
    *,
    use_equity_corr_risk: bool = False,
) -> dict[str, Any]:
    route_state_mode = normalize_route_state_mode(context.get("route_state_mode"))
    mapping = normalize_mapping_indices(mapping, route_state_mode)
    corr_gross_scale = context["equity_corr_gross_scale"] if use_equity_corr_risk else np.ones_like(context["regime"])
    corr_regime_mult = context["equity_corr_regime_mult"] if use_equity_corr_risk else np.ones_like(context["regime"])
    if fast_engine == "numba":
        result = _fast_overlay_replay_kernel(
            context["close"],
            context["bucket_codes"][float(route_breadth_threshold)],
            context["regime"],
            context["breadth"],
            context["vol_ann"],
            corr_gross_scale,
            corr_regime_mult,
            context["smooth_signal_matrix"],
            library_lookup["signal_pos"],
            library_lookup["rebalance_bars"],
            library_lookup["regime_threshold"],
            library_lookup["breadth_threshold"],
            library_lookup["target_vol_ann"],
            library_lookup["gross_cap"],
            library_lookup["kill_switch_pct"],
            library_lookup["cooldown_days"],
            np.asarray(mapping, dtype="int64"),
            float(gp.INITIAL_CASH),
            float(gp.COMMISSION_PCT),
            float(gp.NO_TRADE_BAND),
            int(BARS_PER_DAY),
            float(gp.DAILY_TARGET_PCT),
            float(BAR_FACTOR),
        )
        return {
            "total_return": float(result[0]),
            "n_trades": int(result[1]),
            "sharpe": float(result[2]),
            "max_drawdown": float(result[3]),
            "final_equity": float(result[4]),
            "daily_metrics": {
                "avg_daily_return": float(result[5]),
                "daily_target_hit_rate": float(result[6]),
                "daily_win_rate": float(result[7]),
                "worst_day": float(result[8]),
                "best_day": float(result[9]),
            },
        }

    close = context["close"]
    bucket_codes = context["bucket_codes"][float(route_breadth_threshold)]
    regime = context["regime"]
    breadth = context["breadth"]
    vol_ann = context["vol_ann"]
    equity_corr_gross_scale = corr_gross_scale
    equity_corr_regime_mult = corr_regime_mult
    smooth_signal_matrix = context["smooth_signal_matrix"]
    signal_positions = library_lookup["signal_pos"]

    equity = float(gp.INITIAL_CASH)
    peak_equity = float(gp.INITIAL_CASH)
    current_weight = 0.0
    cooldown_bars_left = 0
    net_ret: list[float] = []
    equity_curve = [float(gp.INITIAL_CASH)]
    n_trades = 0

    for i in range(len(close) - 1):
        active_idx = int(mapping[int(bucket_codes[i])])
        params = library[active_idx]

        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = float(np.clip(smooth_signal_matrix[signal_positions[active_idx], i], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(regime[i])
        breadth_score = float(breadth[i])
        effective_regime_threshold = float(params.regime_threshold) * float(equity_corr_regime_mult[i])
        effective_gross_cap = float(params.gross_cap) * float(equity_corr_gross_scale[i])
        long_ok = regime_score >= effective_regime_threshold and breadth_score >= params.breadth_threshold
        short_ok = regime_score <= -effective_regime_threshold and breadth_score <= (1.0 - params.breadth_threshold)
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = float(vol_ann[i])
        if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = min(
                params.target_vol_ann / bar_vol_ann,
                effective_gross_cap / max(abs(requested_weight), 1e-8),
            )
            requested_weight *= float(vol_scale)
        requested_weight = float(np.clip(requested_weight, -effective_gross_cap, effective_gross_cap))

        drawdown = equity / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -params.kill_switch_pct and cooldown_bars_left == 0:
            cooldown_bars_left = params.cooldown_days * BARS_PER_DAY

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
        "sharpe": float(np.mean(net_ret) / np.std(net_ret) * BAR_FACTOR) if len(net_ret) > 1 and np.std(net_ret) > 1e-12 else 0.0,
        "max_drawdown": float(np.min(np.asarray(equity_curve) / np.maximum.accumulate(np.asarray(equity_curve)) - 1.0)),
        "final_equity": float(equity),
        "daily_metrics": gp.compute_daily_metrics(np.asarray(net_ret, dtype="float64")),
    }


def realistic_overlay_replay_from_context(
    context: dict[str, Any],
    library_lookup: dict[str, Any],
    mapping: tuple[int, ...],
    route_breadth_threshold: float,
    fee_rate: float = 0.0004,
    slippage: float = 0.0002,
    amount_step: float = 0.001,
    min_qty: float = 0.001,
    *,
    use_equity_corr_risk: bool = False,
) -> dict[str, Any]:
    route_state_mode = normalize_route_state_mode(context.get("route_state_mode"))
    mapping = normalize_mapping_indices(mapping, route_state_mode)
    corr_gross_scale = context["equity_corr_gross_scale"] if use_equity_corr_risk else np.ones_like(context["regime"])
    corr_regime_mult = context["equity_corr_regime_mult"] if use_equity_corr_risk else np.ones_like(context["regime"])
    result = _realistic_overlay_replay_kernel(
        context["open"],
        context["close"],
        context["funding_rates"],
        context["bucket_codes"][float(route_breadth_threshold)],
        context["regime"],
        context["breadth"],
        context["vol_ann"],
        corr_gross_scale,
        corr_regime_mult,
        context["smooth_signal_matrix"],
        library_lookup["signal_pos"],
        library_lookup["rebalance_bars"],
        library_lookup["regime_threshold"],
        library_lookup["breadth_threshold"],
        library_lookup["target_vol_ann"],
        library_lookup["gross_cap"],
        library_lookup["kill_switch_pct"],
        library_lookup["cooldown_days"],
        np.asarray(mapping, dtype="int64"),
        float(gp.INITIAL_CASH),
        float(fee_rate),
        float(slippage),
        float(amount_step),
        float(min_qty),
        float(gp.NO_TRADE_BAND),
        int(BARS_PER_DAY),
        float(gp.DAILY_TARGET_PCT),
        float(BAR_FACTOR),
    )
    return {
        "avg_daily_return": float(result[5]),
        "total_return": float(result[0]),
        "max_drawdown": float(result[3]),
        "sharpe": float(result[2]),
        "daily_target_hit_rate": float(result[6]),
        "daily_win_rate": float(result[7]),
        "worst_day": float(result[8]),
        "best_day": float(result[9]),
        "n_trades": int(result[1]),
        "fee_paid": float(result[10]),
        "slippage_paid": float(result[11]),
        "funding_paid": float(result[12]),
        "funding_events": int(result[13]),
        "final_equity": float(result[4]),
    }


def realistic_overlay_replay(
    df: pd.DataFrame,
    trade_pair: str,
    raw_signal: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    funding_df: pd.DataFrame,
    library: list[OverlayParams],
    mapping: tuple[int, ...],
    route_breadth_threshold: float,
    *,
    use_equity_corr_risk: bool = False,
    route_state_mode: str = ROUTE_STATE_MODE_BASE,
) -> dict[str, Any]:
    library_lookup = build_library_lookup(library)
    context = build_fast_context(
        df=df,
        pair=trade_pair,
        raw_signal=raw_signal,
        overlay_inputs=overlay_inputs,
        route_thresholds=(float(route_breadth_threshold),),
        library_lookup=library_lookup,
        funding_df=funding_df,
        route_state_mode=route_state_mode,
    )
    return realistic_overlay_replay_from_context(
        context,
        library_lookup,
        mapping,
        route_breadth_threshold,
        use_equity_corr_risk=use_equity_corr_risk,
    )


def score_candidate(agg_6m: dict[str, Any], agg_4y: dict[str, Any]) -> float:
    score = 0.0
    score -= agg_6m["mean_avg_daily_return"] * 220000.0
    score -= agg_4y["mean_avg_daily_return"] * 180000.0
    score -= agg_6m["worst_pair_avg_daily_return"] * 240000.0
    score -= agg_4y["worst_pair_avg_daily_return"] * 180000.0
    score -= agg_6m["mean_total_return"] * 10000.0
    score += abs(agg_6m["worst_max_drawdown"]) * 18000.0
    score += abs(agg_4y["worst_max_drawdown"]) * 15000.0
    score += agg_6m["pair_return_dispersion"] * 120000.0
    score += agg_4y["pair_return_dispersion"] * 100000.0
    return float(score)


def score_realistic_candidate(report: dict[str, Any]) -> float:
    recent_6m = report["windows"]["recent_6m"]["aggregate"]
    full_4y = report["windows"]["full_4y"]["aggregate"]

    score = 0.0
    score += float(recent_6m["worst_pair_avg_daily_return"]) * 380000.0
    score += float(full_4y["worst_pair_avg_daily_return"]) * 280000.0
    score += float(full_4y["mean_avg_daily_return"]) * 180000.0
    score += float(recent_6m["mean_avg_daily_return"]) * 50000.0
    score -= abs(float(recent_6m["worst_max_drawdown"])) * 18000.0
    score -= abs(float(full_4y["worst_max_drawdown"])) * 9000.0
    score -= float(recent_6m["pair_return_dispersion"]) * 120000.0
    score -= float(full_4y["pair_return_dispersion"]) * 60000.0
    return float(score)


def parse_csv_tuple(raw: str, cast) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def build_search_candidate_pool(
    route_thresholds: tuple[float, ...],
    subset_indices: tuple[int, ...],
    baseline_mapping: tuple[int, ...],
    baseline_route_threshold: float,
    route_state_mode: str,
) -> list[tuple[float, tuple[int, ...]]]:
    mode = normalize_route_state_mode(route_state_mode)
    baseline_mapping = normalize_mapping_indices(baseline_mapping, mode)
    ordered: list[tuple[float, tuple[int, ...]]] = []
    seen: set[tuple[float, tuple[int, ...]]] = set()

    def add(route_threshold: float, mapping: tuple[int, ...] | list[int]) -> None:
        key = (float(route_threshold), normalize_mapping_indices(mapping, mode))
        if key in seen:
            return
        seen.add(key)
        ordered.append(key)

    if mode == ROUTE_STATE_MODE_BASE:
        for route_threshold in route_thresholds:
            for mapping in itertools.product(subset_indices, repeat=len(BASE_ROUTE_STATE_NAMES)):
                add(route_threshold, mapping)
        add(baseline_route_threshold, baseline_mapping)
        return ordered

    base_candidates = list(itertools.product(subset_indices, repeat=len(BASE_ROUTE_STATE_NAMES)))
    for route_threshold in route_thresholds:
        add(route_threshold, baseline_mapping)
        for mapping in base_candidates:
            add(route_threshold, mapping)
            for corr_block in range(len(EQUITY_CORR_ROUTE_BUCKETS)):
                mutated = list(baseline_mapping)
                start = corr_block * len(BASE_ROUTE_STATE_NAMES)
                mutated[start:start + len(BASE_ROUTE_STATE_NAMES)] = mapping
                add(route_threshold, mutated)
        for bucket in range(route_state_count(mode)):
            for value in subset_indices:
                mutated = list(baseline_mapping)
                mutated[bucket] = int(value)
                add(route_threshold, mutated)
    return ordered


def main() -> None:
    args = parse_args()
    pairs = parse_csv_tuple(args.pairs, str)
    subset_indices = parse_csv_tuple(args.subset_indices, int)
    route_thresholds = parse_csv_tuple(args.route_thresholds, float)
    fast_engine = resolve_fast_engine(args.fast_engine)
    route_state_mode = normalize_route_state_mode(args.route_state_mode)

    total_started = perf_counter()
    baseline_candidate, library, _ = resolve_candidate(Path(args.summary), None, None)
    baseline_mapping = normalize_mapping_indices(baseline_candidate.mapping_indices, route_state_mode)
    if baseline_candidate.route_breadth_threshold not in route_thresholds:
        route_thresholds = tuple(sorted(set(route_thresholds + (baseline_candidate.route_breadth_threshold,))))
    library_lookup = build_library_lookup(library)
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

    prepare_started = perf_counter()
    window_cache = {}
    for label, start, end in DEFAULT_WINDOWS:
        df = df_all.loc[start:end].copy()
        pair_cache = {}
        for pair in pairs:
            overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
            signal_slice = raw_signal_all[pair].loc[start:end].copy()
            funding_slice = funding_all[pair]
            if not funding_slice.empty:
                funding_slice = funding_slice[
                    (funding_slice["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                    & (funding_slice["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
                ].copy()
            pair_cache[pair] = {
                "overlay_inputs": overlay_inputs,
                "signal": signal_slice,
                "funding": funding_slice,
                "fast_context": build_fast_context(
                    df=df,
                    pair=pair,
                    raw_signal=signal_slice,
                    overlay_inputs=overlay_inputs,
                    route_thresholds=route_thresholds,
                    library_lookup=library_lookup,
                    funding_df=funding_slice,
                    route_state_mode=route_state_mode,
                ),
            }
        window_cache[label] = {
            "start": start,
            "end": end,
            "df": df,
            "pairs": pair_cache,
        }
    prepare_seconds = perf_counter() - prepare_started

    baseline_fast = {}
    baseline_realistic = {}
    baseline_started = perf_counter()
    for label, start, end in DEFAULT_WINDOWS:
        window_data = window_cache[label]
        df = window_data["df"]
        per_pair_fast = {}
        per_pair_realistic = {}
        for pair in pairs:
            pair_data = window_data["pairs"][pair]
            per_pair_fast[pair] = summarize_single_result(
                fast_overlay_replay_from_context(
                    pair_data["fast_context"],
                    library,
                    library_lookup,
                    baseline_mapping,
                    baseline_candidate.route_breadth_threshold,
                    fast_engine,
                )
            )
            per_pair_realistic[pair] = realistic_overlay_replay_from_context(
                pair_data["fast_context"],
                library_lookup,
                baseline_mapping,
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
    baseline_seconds = perf_counter() - baseline_started

    candidate_pool = build_search_candidate_pool(
        route_thresholds=route_thresholds,
        subset_indices=subset_indices,
        baseline_mapping=baseline_mapping,
        baseline_route_threshold=baseline_candidate.route_breadth_threshold,
        route_state_mode=route_state_mode,
    )

    scored = []
    fast_search_started = perf_counter()
    for route_threshold, mapping in candidate_pool:
        windows = {}
        for label, start, end in DEFAULT_WINDOWS[1:]:
            window_data = window_cache[label]
            per_pair = {}
            for pair in pairs:
                pair_data = window_data["pairs"][pair]
                per_pair[pair] = summarize_single_result(
                    fast_overlay_replay_from_context(
                        pair_data["fast_context"],
                        library,
                        library_lookup,
                        mapping,
                        route_threshold,
                        fast_engine,
                    )
                )
            windows[label] = aggregate_metrics(per_pair)

        recent_6m = windows["recent_6m"]
        full_4y = windows["full_4y"]
        if recent_6m["positive_pair_count"] < len(pairs) or full_4y["positive_pair_count"] < len(pairs):
            continue

        scored.append(
            {
                "route_breadth_threshold": route_threshold,
                "mapping_indices": list(mapping),
                "route_state_mode": route_state_mode,
                "route_state_names": list(route_state_names(route_state_mode)),
                "recent_6m": recent_6m,
                "full_4y": full_4y,
                "score": score_candidate(recent_6m, full_4y),
            }
        )
    fast_search_seconds = perf_counter() - fast_search_started

    scored.sort(key=lambda item: item["score"])
    top_fast = scored[: args.top_k_realistic]

    realistic_top = []
    realistic_started = perf_counter()
    for item in top_fast:
        route_threshold = float(item["route_breadth_threshold"])
        mapping = tuple(int(v) for v in item["mapping_indices"])
        windows = {}
        for label, start, end in DEFAULT_WINDOWS:
            window_data = window_cache[label]
            df = window_data["df"]
            per_pair = {}
            for pair in pairs:
                pair_data = window_data["pairs"][pair]
                per_pair[pair] = realistic_overlay_replay_from_context(
                    pair_data["fast_context"],
                    library_lookup,
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
                "route_state_mode": route_state_mode,
                "route_state_names": list(route_state_names(route_state_mode)),
                "score": item["score"],
                "windows": windows,
                "validation": build_validation_bundle(windows, baseline_realistic),
            }
        )
    realistic_seconds = perf_counter() - realistic_started

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
    final_oos_audit_pass_count = sum(
        1 for item in realistic_top if item["validation"]["profiles"]["final_oos"]["passed"]
    )
    fallback_best = max(realistic_top, key=score_realistic_candidate) if realistic_top else None
    selected = max(target_060_candidates, key=score_realistic_candidate) if target_060_candidates else None
    selection_reason = "target_060"
    if selected is None and progressive_candidates:
        selected = max(progressive_candidates, key=score_realistic_candidate)
        selection_reason = "progressive_improvement"
    if selected is None:
        selection_reason = "no_gate_pass"

    report = {
        "pairs": list(pairs),
        "baseline_candidate": {
            "route_breadth_threshold": baseline_candidate.route_breadth_threshold,
            "mapping_indices": list(baseline_mapping),
            "route_state_mode": route_state_mode,
            "route_state_names": list(route_state_names(route_state_mode)),
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
            "final_oos_audit_pass_count": final_oos_audit_pass_count,
            "progressive_pass_count": len(progressive_candidates),
            "realistic_top_count": len(realistic_top),
            "selected_final_oos_passed": bool(
                selected and selected["validation"]["profiles"]["final_oos"]["passed"]
            ),
        },
        "route_state": {
            "mode": route_state_mode,
            "state_names": list(route_state_names(route_state_mode)),
            "state_count": route_state_count(route_state_mode),
        },
        "runtime": {
            "fast_engine": fast_engine,
            "numba_available": NUMBA_AVAILABLE,
            "prepare_context_seconds": prepare_seconds,
            "baseline_seconds": baseline_seconds,
            "fast_search_seconds": fast_search_seconds,
            "realistic_seconds": realistic_seconds,
            "total_seconds": perf_counter() - total_started,
            "candidate_pool_size": len(candidate_pool),
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
