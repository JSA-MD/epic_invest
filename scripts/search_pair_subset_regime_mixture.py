#!/usr/bin/env python3
"""Search regime-mixture overlays for a subset of trade pairs."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from dataclasses import asdict, is_dataclass

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from regime_gate_helper import gate_overrides as _gate_overrides  # noqa: E402
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
from execution_gene_utils import (
    SPECIALIST_ROLE_NAMES,
    blended_entry_quality_score,
    blended_microstructure_score,
    candle_microstructure_proxy_score,
    dc_alignment_score,
    derive_execution_profile,
    derivative_positioning_score,
    legacy_execution_profile,
    microstructure_alignment_score,
    should_abstain_for_alignment,
    should_allow_countertrend_entry,
    should_abstain_for_liquidity,
    should_abstain_for_weak_tape,
)
from derivative_market_data import load_derivative_metric_cache
from replay_regime_mixture_realistic import (
    fetch_funding_rates,
    load_model,
    resolve_candidate,
)
from search_gp_drawdown_overlay import OverlayParams
from validate_pair_subset_summary import build_validation_bundle


UTC = timezone.utc
BARS_PER_DAY = gp.periods_per_day(gp.TIMEFRAME)

# Opt-in noise tolerance for the breadth binary classifier.
# Default 0.0 = legacy behaviour (no change).  Set e.g. 0.005 to treat a
# 3-day pct_change > -0.5% as "positive" so noise-level flat markets do not
# collapse breadth to 0.
PAIRWISE_BREADTH_NOISE_EPSILON = float(os.getenv("PAIRWISE_BREADTH_NOISE_EPSILON", "0.0"))
BAR_FACTOR = np.sqrt(365.25 * 24.0 * 60.0 / 5.0)
MAX_REGIME_BUCKETS = 16
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


def default_state_specialists_for_router(route_names: tuple[str, ...]) -> tuple[int, ...]:
    role_to_idx = {name: idx for idx, name in enumerate(SPECIALIST_ROLE_NAMES)}
    assignments: list[int] = []
    for route_name in route_names:
        if "bear_broad" in route_name:
            role = "panic"
        elif "bull_broad" in route_name:
            role = "trend"
        elif "equity_inverse" in route_name or "bear_narrow" in route_name:
            role = "carry"
        else:
            role = "range"
        assignments.append(int(role_to_idx[role]))
    return tuple(assignments)


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


def _load_derivative_bundle(symbol: str) -> dict[str, pd.DataFrame]:
    metrics = (
        "open_interest",
        "basis_perpetual",
        "top_trader_position_ratio",
        "taker_buy_sell_ratio",
    )
    return {metric: load_derivative_metric_cache(symbol, metric) for metric in metrics}


def _derivative_metric_history_series(frame: pd.DataFrame | None, value_column: str) -> pd.Series:
    if frame is None or frame.empty or "timestamp" not in frame.columns or value_column not in frame.columns:
        return pd.Series(dtype="float64")
    return (
        frame[["timestamp", value_column]]
        .copy()
        .assign(
            timestamp=lambda df_: pd.to_datetime(df_["timestamp"], utc=True, errors="coerce", format="mixed"),
            value=lambda df_: pd.to_numeric(df_[value_column], errors="coerce"),
        )
        .dropna(subset=["timestamp", "value"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .set_index("timestamp")["value"]
        .sort_index()
    )


def _derivative_metric_series(frame: pd.DataFrame | None, value_column: str, index: pd.DatetimeIndex) -> pd.Series:
    series = _derivative_metric_history_series(frame, value_column)
    if series.empty:
        return pd.Series(np.nan, index=index, dtype="float64")
    return series.reindex(index).ffill().replace([np.inf, -np.inf], np.nan).astype("float64")


def _log_ratio_feature(series: pd.Series) -> pd.Series:
    positive = pd.to_numeric(series, errors="coerce").where(lambda values: values > 0.0)
    return np.log(positive).replace([np.inf, -np.inf], np.nan).clip(-3.0, 3.0).astype("float64")


def _open_interest_relative_metric(frame: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series:
    history_series = _derivative_metric_history_series(frame, "open_interest")
    if history_series.empty:
        return pd.Series(np.nan, index=index, dtype="float64")
    combined_index = history_series.index.union(pd.DatetimeIndex(index)).sort_values()
    aligned = history_series.reindex(combined_index).ffill().replace([np.inf, -np.inf], np.nan).astype("float64")
    baseline = aligned.rolling(12 * 24 * 7, min_periods=12 * 24).median()
    relative = aligned / baseline.replace(0.0, np.nan)
    return relative.reindex(index).ffill().replace([np.inf, -np.inf], np.nan).clip(0.0, 5.0).astype("float64")


def build_fast_context(
    df: pd.DataFrame,
    pair: str,
    raw_signal: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    route_thresholds: tuple[float, ...],
    library_lookup: dict[str, Any],
    funding_df: pd.DataFrame | None = None,
    derivative_bundle: dict[str, pd.DataFrame] | None = None,
    route_state_mode: str = ROUTE_STATE_MODE_BASE,
    strict_external_asof: bool = False,
) -> dict[str, Any]:
    route_state_mode = normalize_route_state_mode(route_state_mode)
    idx = pd.DatetimeIndex(df.index)
    day_index = idx.normalize()
    completed_day_index = day_index - pd.Timedelta(days=1)
    effective_day_index = completed_day_index if strict_external_asof else day_index
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
            strict_external_asof=strict_external_asof,
        ).astype("int64")
        for threshold in route_thresholds
    }
    funding_rates = np.zeros(len(idx), dtype="float64")
    funding_unmatched = 0
    validation_daily_index = pd.DatetimeIndex(effective_day_index.unique())
    if funding_df is not None and not funding_df.empty:
        funding_series = (
            funding_df[["fundingTime", "fundingRate"]]
            .dropna(subset=["fundingTime", "fundingRate"])
            .assign(fundingTime=lambda frame: pd.to_datetime(frame["fundingTime"], utc=True))
            .drop_duplicates(subset=["fundingTime"], keep="last")
            .set_index("fundingTime")["fundingRate"]
            .sort_index()
        )
        _idx_frame = pd.DataFrame({"ts": idx}, index=range(len(idx)))
        _funding_frame = funding_series.reset_index().rename(columns={"fundingTime": "ts", "fundingRate": "rate"})
        _merged = pd.merge_asof(
            _idx_frame,
            _funding_frame.sort_values("ts"),
            on="ts",
            direction="nearest",
            tolerance=pd.Timedelta("1min"),
        )
        funding_rates = _merged["rate"].fillna(0.0).to_numpy(dtype="float64")
        _total_funding_events = len(funding_series)
        _matched = int((_merged["rate"].notna()).sum())
        funding_unmatched = max(0, _total_funding_events - _matched)
    def pair_feature_series(suffix: str, *, fill_value: float = 0.0) -> pd.Series:
        column = f"{pair}_{suffix}"
        if column not in df.columns:
            return pd.Series(fill_value, index=idx, dtype="float64")
        return (
            pd.to_numeric(df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fill_value)
            .astype("float64")
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
    btc_qqq_corr_5d_daily = overlay_inputs.get(
        "btc_qqq_corr_5d_daily",
        pd.Series(np.nan, index=validation_daily_index, dtype="float64"),
    )
    btc_qqq_corr_20d_daily = overlay_inputs.get(
        "btc_qqq_corr_20d_daily",
        pd.Series(np.nan, index=validation_daily_index, dtype="float64"),
    )
    btc_spy_beta_20d_daily = overlay_inputs.get(
        "btc_spy_beta_20d_daily",
        pd.Series(np.nan, index=validation_daily_index, dtype="float64"),
    )
    btc_dxy_corr_20d_daily = overlay_inputs.get(
        "btc_dxy_corr_20d_daily",
        pd.Series(np.nan, index=validation_daily_index, dtype="float64"),
    )
    btc_gold_corr_20d_daily = overlay_inputs.get(
        "btc_gold_corr_20d_daily",
        pd.Series(np.nan, index=validation_daily_index, dtype="float64"),
    )
    oi_rel = _open_interest_relative_metric((derivative_bundle or {}).get("open_interest"), idx)
    basis_rate = _derivative_metric_series((derivative_bundle or {}).get("basis_perpetual"), "basis_rate", idx).clip(-0.01, 0.01)
    top_pos_log_ratio = _log_ratio_feature(
        _derivative_metric_series((derivative_bundle or {}).get("top_trader_position_ratio"), "long_short_ratio", idx)
    )
    taker_buy_sell_log_ratio = _log_ratio_feature(
        _derivative_metric_series((derivative_bundle or {}).get("taker_buy_sell_ratio"), "buy_sell_ratio", idx)
    )
    return {
        "open": df[f"{pair}_open"].to_numpy(dtype="float64"),
        "close": df[f"{pair}_close"].to_numpy(dtype="float64"),
        "bucket_codes": bucket_codes,
        "bar_day_index": pd.DatetimeIndex(day_index),
        "regime": overlay_inputs["btc_regime_daily"].reindex(effective_day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64"),
        "breadth": overlay_inputs["breadth_daily"].reindex(effective_day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64"),
        "vol_ann": (
            overlay_inputs["vol_ann_bar"].reindex(idx).ffill().fillna(0.0)
            if strict_external_asof
            else overlay_inputs["vol_ann_bar"].reindex(idx).ffill().bfill().fillna(0.0)
        ).to_numpy(dtype="float64"),
        "equity_corr": equity_corr_daily.reindex(effective_day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64"),
        "equity_corr_gross_scale": equity_corr_gross_scale_daily.reindex(effective_day_index, method="ffill").fillna(1.0).to_numpy(dtype="float64"),
        "equity_corr_regime_mult": equity_corr_regime_mult_daily.reindex(effective_day_index, method="ffill").fillna(1.0).to_numpy(dtype="float64"),
        "btc_qqq_corr_5d": btc_qqq_corr_5d_daily.reindex(effective_day_index, method="ffill").to_numpy(dtype="float64"),
        "btc_qqq_corr_20d": btc_qqq_corr_20d_daily.reindex(effective_day_index, method="ffill").to_numpy(dtype="float64"),
        "btc_spy_beta_20d": btc_spy_beta_20d_daily.reindex(effective_day_index, method="ffill").to_numpy(dtype="float64"),
        "btc_dxy_corr_20d": btc_dxy_corr_20d_daily.reindex(effective_day_index, method="ffill").to_numpy(dtype="float64"),
        "btc_gold_corr_20d": btc_gold_corr_20d_daily.reindex(effective_day_index, method="ffill").to_numpy(dtype="float64"),
        "buy_volume_share": pair_feature_series("buy_volume_share", fill_value=0.5).to_numpy(dtype="float64"),
        "order_imbalance": pair_feature_series("order_imbalance", fill_value=0.0).to_numpy(dtype="float64"),
        "close_location_value": pair_feature_series("close_location_value", fill_value=0.0).to_numpy(dtype="float64"),
        "body_to_range": pair_feature_series("body_to_range", fill_value=0.0).to_numpy(dtype="float64"),
        "wick_skew": pair_feature_series("wick_skew", fill_value=0.0).to_numpy(dtype="float64"),
        "candle_micro_score": pair_feature_series("candle_micro_score", fill_value=0.0).to_numpy(dtype="float64"),
        "oi_rel": oi_rel.fillna(1.0).to_numpy(dtype="float64"),
        "basis_rate": basis_rate.fillna(0.0).to_numpy(dtype="float64"),
        "top_pos_log_ratio": top_pos_log_ratio.fillna(0.0).to_numpy(dtype="float64"),
        "taker_buy_sell_log_ratio": taker_buy_sell_log_ratio.fillna(0.0).to_numpy(dtype="float64"),
        "range_bps": (
            (
                (
                    pair_feature_series("high", fill_value=0.0)
                    - pair_feature_series("low", fill_value=0.0)
                )
                / pair_feature_series("close", fill_value=np.nan).replace(0.0, np.nan)
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            * 10_000.0
        ).to_numpy(dtype="float64"),
        "volume_ratio": (
            (
                pair_feature_series("volume", fill_value=0.0)
                / pair_feature_series("vol_sma", fill_value=np.nan).replace(0.0, np.nan)
            )
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
        ).to_numpy(dtype="float64"),
        "dc_trend_05": pair_feature_series("dc_trend_05", fill_value=0.0).to_numpy(dtype="float64"),
        "dc_event_05": pair_feature_series("dc_event_05", fill_value=0.0).to_numpy(dtype="float64"),
        "dc_overshoot_05": pair_feature_series("dc_overshoot_05", fill_value=0.0).to_numpy(dtype="float64"),
        "dc_run_05": pair_feature_series("dc_run_05", fill_value=0.0).to_numpy(dtype="float64"),
        "smooth_signal_matrix": smooth_signal_matrix,
        "funding_rates": funding_rates,
        "funding_unmatched": funding_unmatched,
        "equity_corr_context": overlay_inputs.get("equity_corr_context"),
        "equity_corr_source_mode": overlay_inputs.get("equity_corr_source_mode"),
        "route_state_mode": route_state_mode,
        "route_state_names": route_state_names(route_state_mode),
        "route_state_specialists": np.asarray(default_state_specialists_for_router(route_state_names(route_state_mode)), dtype="int64"),
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
    order_imbalance: np.ndarray,
    buy_volume_share: np.ndarray,
    close_location_value: np.ndarray,
    body_to_range: np.ndarray,
    wick_skew: np.ndarray,
    candle_micro_score: np.ndarray,
    oi_rel: np.ndarray,
    basis_rate: np.ndarray,
    top_pos_log_ratio: np.ndarray,
    taker_buy_sell_log_ratio: np.ndarray,
    range_bps: np.ndarray,
    volume_ratio: np.ndarray,
    dc_trend_05: np.ndarray,
    dc_run_05: np.ndarray,
    mapping: np.ndarray,
    initial_cash: float,
    commission_pct: float,
    no_trade_band_pct: float,
    signal_gate_pct: float,
    regime_buffer_mult: float,
    confirm_bars: int,
    state_specialists: np.ndarray,
    role_signal_gate_mults: np.ndarray,
    role_regime_buffer_mults: np.ndarray,
    abstain_edge_pct: float,
    specialist_isolation_mult: float,
    liquidity_range_gate_bp: float,
    liquidity_volume_ratio_floor: float,
    short_horizon_abstain_mult: float,
    microstructure_align_gate_pct: float,
    dc_align_gate_pct: float,
    min_alignment_votes: int,
    bars_per_day: int,
    daily_target: float,
    bar_factor: float,
    gate_threshold_scale: float,
    gate_disabled: bool,
    *,
    initial_cooldown_bars: int = 0,
    final_decision_cooldown_override: int | None = None,
) -> tuple[float, int, float, float, float, float, float, float, float]:
    equity = initial_cash
    peak_equity = initial_cash
    current_weight = 0.0
    cooldown_bars_left = int(initial_cooldown_bars)
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
    confirm_side = 0
    confirm_count = 0
    last_role_idx = -1
    # Gate override args are received from caller (env-derived) so this
    # body remains njit-compilable. Don't call os.environ here.
    _gate_threshold_scale = gate_threshold_scale
    _gate_disabled = gate_disabled

    for i in range(close.shape[0] - 1):
        active_idx = mapping[bucket_codes[i]]
        role_idx = state_specialists[bucket_codes[i]]
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1
        # Final-decision override: at the live-decision bar force the gate
        # cooldown to the exact live value, then below — *after* the gate
        # check that uses cooldown_bars_left — replace it with override-1 so
        # the trace/persisted value carries one bar of natural decrement.
        # Without the post-gate adjustment, shadow.cooldown_bars_left gets
        # the raw override value back every cycle and never expires.
        _is_final_bar_override = (
            final_decision_cooldown_override is not None
            and i == close.shape[0] - 2
        )
        if _is_final_bar_override:
            cooldown_bars_left = int(final_decision_cooldown_override)

        role_changed = role_idx != last_role_idx
        if role_changed:
            confirm_side = 0
            confirm_count = 0
            last_role_idx = role_idx

        signal_pct = smooth_signal_matrix[library_signal_pos[active_idx], i]
        if signal_pct > 500.0:
            signal_pct = 500.0
        elif signal_pct < -500.0:
            signal_pct = -500.0

        requested_weight = signal_pct / 100.0
        regime_score = regime[i]
        breadth_score = breadth[i]
        role_signal_gate_pct = signal_gate_pct * role_signal_gate_mults[role_idx]
        role_regime_buffer_mult = regime_buffer_mult * role_regime_buffer_mults[role_idx]
        effective_regime_threshold = library_regime_threshold[active_idx] * equity_corr_regime_mult[i] * (1.0 + role_regime_buffer_mult) * _gate_threshold_scale
        effective_gross_cap = library_gross_cap[active_idx] * equity_corr_gross_scale[i]
        if _gate_disabled:
            long_ok = True
            short_ok = True
        else:
            long_ok = regime_score >= effective_regime_threshold and breadth_score >= library_breadth_threshold[active_idx]
            short_ok = regime_score <= -effective_regime_threshold and breadth_score <= (1.0 - library_breadth_threshold[active_idx])
        if abs(signal_pct) < (role_signal_gate_pct + abstain_edge_pct):
            requested_weight = 0.0
        elif requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0
        else:
            requested_side = 0
            if requested_weight > 1e-12:
                requested_side = 1
            elif requested_weight < -1e-12:
                requested_side = -1
            if requested_side != 0:
                candle_score = candle_micro_score[i]
                if candle_score != candle_score:
                    directional_body = (1.0 if close_location_value[i] >= 0.0 else -1.0) * body_to_range[i]
                    candle_score = 0.50 * close_location_value[i] + 0.30 * directional_body + 0.20 * wick_skew[i]
                flow_score = 0.65 * (
                    0.70 * order_imbalance[i] + 0.30 * (2.0 * buy_volume_share[i] - 1.0)
                ) + 0.35 * candle_score
                oi_component = (oi_rel[i] - 1.0) / 0.10
                if oi_component > 1.0:
                    oi_component = 1.0
                elif oi_component < -1.0:
                    oi_component = -1.0
                basis_component = basis_rate[i] / 0.0015
                if basis_component > 1.0:
                    basis_component = 1.0
                elif basis_component < -1.0:
                    basis_component = -1.0
                positioning_score = (
                    0.35 * top_pos_log_ratio[i]
                    + 0.35 * taker_buy_sell_log_ratio[i]
                    + 0.20 * oi_component
                    + 0.10 * basis_component
                )
                if positioning_score > 1.0:
                    positioning_score = 1.0
                elif positioning_score < -1.0:
                    positioning_score = -1.0
                micro_score = flow_score
                if micro_score > 1.0:
                    micro_score = 1.0
                elif micro_score < -1.0:
                    micro_score = -1.0
                if current_weight * requested_side <= 0.0:
                    if requested_side * positioning_score < -0.35 and requested_side * micro_score < 0.10:
                        requested_weight = 0.0
                    alignment_votes = 0
                    if requested_weight != 0.0 and requested_side * micro_score >= microstructure_align_gate_pct:
                        alignment_votes += 1
                    if requested_weight != 0.0 and requested_side * (
                        0.75 * dc_trend_05[i] + 0.25 * dc_run_05[i]
                    ) >= dc_align_gate_pct:
                        alignment_votes += 1
                    if requested_weight != 0.0 and alignment_votes < min_alignment_votes:
                        requested_weight = 0.0
                if requested_weight != 0.0:
                    if current_weight * requested_side <= 0.0 and short_horizon_abstain_mult > 0.0:
                        weak_votes = 0
                        micro_side_score = requested_side * micro_score
                        dc_score = requested_side * (
                            0.75 * dc_trend_05[i] + 0.25 * dc_run_05[i]
                        )
                        if micro_side_score < 0.15 * short_horizon_abstain_mult:
                            weak_votes += 1
                        if dc_score < 0.10 * short_horizon_abstain_mult:
                            weak_votes += 1
                        if range_bps[i] <= 45.0 and volume_ratio[i] <= 1.0:
                            weak_votes += 1
                        if equity_corr_gross_scale[i] < max(0.75, 1.0 - 0.10 * short_horizon_abstain_mult):
                            weak_votes += 1
                        if weak_votes >= 2:
                            requested_weight = 0.0
                    range_gate_enabled = liquidity_range_gate_bp > 0.0
                    volume_gate_enabled = liquidity_volume_ratio_floor > 0.0
                    if current_weight * requested_side <= 0.0 and (range_gate_enabled or volume_gate_enabled):
                        range_bad = range_gate_enabled and range_bps[i] >= liquidity_range_gate_bp
                        volume_bad = volume_gate_enabled and volume_ratio[i] <= liquidity_volume_ratio_floor
                        if (range_gate_enabled and volume_gate_enabled and range_bad and volume_bad) or (
                            (not range_gate_enabled or not volume_gate_enabled) and (range_bad or volume_bad)
                        ):
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
            role_confirm_bars = confirm_bars
            if role_changed:
                role_confirm_bars = confirm_bars + int(np.rint(specialist_isolation_mult * 2.0))
            requested_side = 0
            if requested_weight > 1e-12:
                requested_side = 1
            elif requested_weight < -1e-12:
                requested_side = -1
            if requested_side == 0:
                confirm_side = 0
                confirm_count = 0
            elif requested_side == confirm_side:
                confirm_count += 1
            else:
                confirm_side = requested_side
                confirm_count = 1
            if requested_side != 0 and current_weight * requested_side <= 0.0 and confirm_count < role_confirm_bars:
                requested_weight = 0.0
            target_weight = requested_weight

        # Post-gate decrement for final-bar override (mirrors realistic kernel):
        # gate saw full override value; now drop by 1 for persistence consistency.
        if _is_final_bar_override:
            cooldown_bars_left = max(0, cooldown_bars_left - 1)

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


def _context_feature_array(
    context: dict[str, Any],
    key: str,
    *,
    fill_value: float = 0.0,
    reference_key: str = "regime",
) -> np.ndarray:
    values = context.get(key)
    if values is None:
        reference = np.asarray(context[reference_key], dtype="float64")
        return np.full(reference.shape, fill_value, dtype="float64")
    return np.asarray(values, dtype="float64")


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
    order_imbalance: np.ndarray,
    buy_volume_share: np.ndarray,
    close_location_value: np.ndarray,
    body_to_range: np.ndarray,
    wick_skew: np.ndarray,
    candle_micro_score: np.ndarray,
    oi_rel: np.ndarray,
    basis_rate: np.ndarray,
    top_pos_log_ratio: np.ndarray,
    taker_buy_sell_log_ratio: np.ndarray,
    range_bps: np.ndarray,
    volume_ratio: np.ndarray,
    dc_trend_05: np.ndarray,
    dc_run_05: np.ndarray,
    mapping: np.ndarray,
    initial_cash: float,
    fee_rate: float,
    slippage: float,
    amount_step: float,
    min_qty: float,
    no_trade_band_pct: float,
    signal_gate_pct: float,
    regime_buffer_mult: float,
    confirm_bars: int,
    state_specialists: np.ndarray,
    role_signal_gate_mults: np.ndarray,
    role_regime_buffer_mults: np.ndarray,
    abstain_edge_pct: float,
    specialist_isolation_mult: float,
    liquidity_range_gate_bp: float,
    liquidity_volume_ratio_floor: float,
    short_horizon_abstain_mult: float,
    countertrend_weight_scale: float,
    countertrend_rebalance_bypass: bool,
    countertrend_signal_floor_pct: float,
    countertrend_regime_score_cap: float,
    countertrend_breadth_slack: float,
    countertrend_microstructure_floor: float,
    countertrend_positioning_floor: float,
    countertrend_dc_floor: float,
    countertrend_range_bps_floor: float,
    countertrend_volume_ratio_floor: float,
    microstructure_align_gate_pct: float,
    dc_align_gate_pct: float,
    min_alignment_votes: int,
    bars_per_day: int,
    daily_target: float,
    bar_factor: float,
    entry_keep_flags: np.ndarray | None = None,
    return_trace: bool = False,
    min_notional_usd: float = 25.0,
    max_hold_bars: int = 288,
    initial_cooldown_bars: int = 0,
    runtime_gross_cap: float = -1.0,
    final_decision_cooldown_override: int | None = None,
) -> tuple[float, int, float, float, float, float, float, float, float, float, float, float, float, int, int, int, int, float, float, float, float, tuple, tuple] | dict[str, Any]:
    cash = initial_cash
    qty = 0.0
    n_trades = 0
    n_wins = 0
    n_losses = 0
    total_win_pnl = 0.0
    total_loss_pnl = 0.0
    entry_lots: list[list[float]] = []
    prev_side = 0
    regime_n_wins = np.zeros(MAX_REGIME_BUCKETS, dtype=np.int64)
    regime_n_losses = np.zeros(MAX_REGIME_BUCKETS, dtype=np.int64)
    fee_paid = 0.0
    slippage_paid = 0.0
    funding_paid = 0.0
    funding_events = 0
    peak_equity = initial_cash
    max_drawdown = 0.0
    cooldown_bars_left = int(initial_cooldown_bars)

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
    confirm_side = 0
    confirm_count = 0
    last_role_idx = -1
    hold_bars = 0
    force_close_next = False
    net_ret: list[float] = []
    target_weight_trace: list[float] = []
    requested_weight_trace: list[float] = []
    signal_pct_trace: list[float] = []
    role_idx_trace: list[int] = []
    cooldown_trace: list[int] = []
    confirm_side_trace: list[int] = []
    confirm_count_trace: list[int] = []
    _gate_threshold_scale, _gate_disabled = _gate_overrides()

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
        current_weight = 0.0
        if abs(equity_before) > 1e-9:
            current_weight = qty * px_open / equity_before

        active_idx = mapping[bucket_codes[signal_idx]]
        role_idx = state_specialists[bucket_codes[signal_idx]]
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1
        # Apply final-decision override AFTER the per-bar decrement so the
        # gate later in the loop body sees exactly the live value. Applying
        # it before the decrement burns one bar off (live=1 → replay=0 →
        # unblocks one bar early).
        if final_decision_cooldown_override is not None and exec_idx == open_p.shape[0] - 2:
            cooldown_bars_left = int(final_decision_cooldown_override)

        role_changed = role_idx != last_role_idx
        if role_changed:
            confirm_side = 0
            confirm_count = 0
            last_role_idx = role_idx

        signal_pct = smooth_signal_matrix[library_signal_pos[active_idx], signal_idx]
        if signal_pct > 500.0:
            signal_pct = 500.0
        elif signal_pct < -500.0:
            signal_pct = -500.0

        requested_weight = signal_pct / 100.0
        base_requested_weight = requested_weight
        regime_score = regime[signal_idx]
        breadth_score = breadth[signal_idx]
        role_signal_gate_pct = signal_gate_pct * role_signal_gate_mults[role_idx]
        role_regime_buffer_mult = regime_buffer_mult * role_regime_buffer_mults[role_idx]
        effective_regime_threshold = library_regime_threshold[active_idx] * equity_corr_regime_mult[signal_idx] * (1.0 + role_regime_buffer_mult) * _gate_threshold_scale
        effective_gross_cap = library_gross_cap[active_idx] * equity_corr_gross_scale[signal_idx]
        if _gate_disabled:
            long_ok = True
            short_ok = True
        else:
            long_ok = regime_score >= effective_regime_threshold and breadth_score >= library_breadth_threshold[active_idx]
            short_ok = regime_score <= -effective_regime_threshold and breadth_score <= (1.0 - library_breadth_threshold[active_idx])
        requested_side = 0
        if requested_weight > 1e-12:
            requested_side = 1
        elif requested_weight < -1e-12:
            requested_side = -1
        candle_score = candle_micro_score[signal_idx]
        if candle_score != candle_score:
            directional_body = (1.0 if close_location_value[signal_idx] >= 0.0 else -1.0) * body_to_range[signal_idx]
            candle_score = (
                0.50 * close_location_value[signal_idx]
                + 0.30 * directional_body
                + 0.20 * wick_skew[signal_idx]
            )
        flow_score = 0.65 * (
            0.70 * order_imbalance[signal_idx] + 0.30 * (2.0 * buy_volume_share[signal_idx] - 1.0)
        ) + 0.35 * candle_score
        oi_component = (oi_rel[signal_idx] - 1.0) / 0.10
        if oi_component > 1.0:
            oi_component = 1.0
        elif oi_component < -1.0:
            oi_component = -1.0
        basis_component = basis_rate[signal_idx] / 0.0015
        if basis_component > 1.0:
            basis_component = 1.0
        elif basis_component < -1.0:
            basis_component = -1.0
        positioning_score = (
            0.35 * top_pos_log_ratio[signal_idx]
            + 0.35 * taker_buy_sell_log_ratio[signal_idx]
            + 0.20 * oi_component
            + 0.10 * basis_component
        )
        if positioning_score > 1.0:
            positioning_score = 1.0
        elif positioning_score < -1.0:
            positioning_score = -1.0
        micro_score = flow_score
        if micro_score > 1.0:
            micro_score = 1.0
        elif micro_score < -1.0:
            micro_score = -1.0
        dc_alignment = 0.75 * dc_trend_05[signal_idx] + 0.25 * dc_run_05[signal_idx]
        countertrend_override = False
        if abs(signal_pct) < (role_signal_gate_pct + abstain_edge_pct):
            requested_weight = 0.0
        elif requested_weight > 0.0 and not long_ok:
            countertrend_override = should_allow_countertrend_entry(
                requested_side,
                signal_pct,
                regime_score,
                breadth_score,
                library_breadth_threshold[active_idx],
                micro_score,
                positioning_score,
                dc_alignment,
                range_bps[signal_idx],
                volume_ratio[signal_idx],
                countertrend_signal_floor_pct,
                countertrend_regime_score_cap,
                countertrend_breadth_slack,
                countertrend_microstructure_floor,
                countertrend_positioning_floor,
                countertrend_dc_floor,
                countertrend_range_bps_floor,
                countertrend_volume_ratio_floor,
            )
            if not countertrend_override:
                requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            countertrend_override = should_allow_countertrend_entry(
                requested_side,
                signal_pct,
                regime_score,
                breadth_score,
                library_breadth_threshold[active_idx],
                micro_score,
                positioning_score,
                dc_alignment,
                range_bps[signal_idx],
                volume_ratio[signal_idx],
                countertrend_signal_floor_pct,
                countertrend_regime_score_cap,
                countertrend_breadth_slack,
                countertrend_microstructure_floor,
                countertrend_positioning_floor,
                countertrend_dc_floor,
                countertrend_range_bps_floor,
                countertrend_volume_ratio_floor,
            )
            if not countertrend_override:
                requested_weight = 0.0
        else:
            if requested_side != 0:
                if current_weight * requested_side <= 0.0:
                    if requested_side * positioning_score < -0.35 and requested_side * micro_score < 0.10:
                        requested_weight = 0.0
                    alignment_votes = 0
                    if requested_weight != 0.0 and requested_side * micro_score >= microstructure_align_gate_pct:
                        alignment_votes += 1
                    if requested_weight != 0.0 and requested_side * (
                        0.75 * dc_trend_05[signal_idx] + 0.25 * dc_run_05[signal_idx]
                    ) >= dc_align_gate_pct:
                        alignment_votes += 1
                    if requested_weight != 0.0 and alignment_votes < min_alignment_votes:
                        requested_weight = 0.0
                if (
                    requested_weight != 0.0
                    and entry_keep_flags is not None
                    and current_weight * requested_side <= 0.0
                    and not bool(entry_keep_flags[signal_idx])
                ):
                    requested_weight = 0.0
                if requested_weight != 0.0:
                    if current_weight * requested_side <= 0.0 and short_horizon_abstain_mult > 0.0:
                        weak_votes = 0
                        micro_side_score = requested_side * micro_score
                        dc_score = requested_side * (
                            0.75 * dc_trend_05[signal_idx] + 0.25 * dc_run_05[signal_idx]
                        )
                        if micro_side_score < 0.15 * short_horizon_abstain_mult:
                            weak_votes += 1
                        if dc_score < 0.10 * short_horizon_abstain_mult:
                            weak_votes += 1
                        if range_bps[signal_idx] <= 45.0 and volume_ratio[signal_idx] <= 1.0:
                            weak_votes += 1
                        if equity_corr_gross_scale[signal_idx] < max(0.75, 1.0 - 0.10 * short_horizon_abstain_mult):
                            weak_votes += 1
                        if weak_votes >= 2:
                            requested_weight = 0.0
                    range_gate_enabled = liquidity_range_gate_bp > 0.0
                    volume_gate_enabled = liquidity_volume_ratio_floor > 0.0
                    if current_weight * requested_side <= 0.0 and (range_gate_enabled or volume_gate_enabled):
                        range_bad = range_gate_enabled and range_bps[signal_idx] >= liquidity_range_gate_bp
                        volume_bad = volume_gate_enabled and volume_ratio[signal_idx] <= liquidity_volume_ratio_floor
                        if (range_gate_enabled and volume_gate_enabled and range_bad and volume_bad) or (
                            (not range_gate_enabled or not volume_gate_enabled) and (range_bad or volume_bad)
                        ):
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
        if countertrend_override and abs(requested_weight) > 1e-12:
            requested_weight = np.sign(base_requested_weight) * min(abs(requested_weight), gross_cap) * countertrend_weight_scale

        drawdown = equity_before / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -library_kill_switch_pct[active_idx] and cooldown_bars_left == 0:
            cooldown_bars_left = library_cooldown_days[active_idx] * bars_per_day

        target_weight = current_weight
        countertrend_bypass_active = bool(
            countertrend_override
            and bool(countertrend_rebalance_bypass)
            and abs(requested_weight) > 1e-12
        )
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif countertrend_bypass_active or signal_idx % library_rebalance_bars[active_idx] == 0:
            role_confirm_bars = confirm_bars
            if role_changed:
                role_confirm_bars = confirm_bars + int(np.rint(specialist_isolation_mult * 2.0))
            requested_side = 0
            if requested_weight > 1e-12:
                requested_side = 1
            elif requested_weight < -1e-12:
                requested_side = -1
            if requested_side == 0:
                confirm_side = 0
                confirm_count = 0
            elif requested_side == confirm_side:
                confirm_count += 1
            else:
                confirm_side = requested_side
                confirm_count = 1
            if requested_side != 0 and current_weight * requested_side <= 0.0 and confirm_count < role_confirm_bars:
                requested_weight = 0.0
            target_weight = requested_weight

        # Post-gate decrement for final-bar override: gate already saw the full
        # override value and blocked correctly; now drop by 1 so the trace/
        # persisted cooldown carries the decremented value to live.  Without
        # this, shadow.cooldown_bars_left receives the raw override every cycle
        # and the cooldown never expires.
        if final_decision_cooldown_override is not None and exec_idx == open_p.shape[0] - 2:
            cooldown_bars_left = max(0, cooldown_bars_left - 1)

        if abs(target_weight - current_weight) < no_trade_band_pct / 100.0:
            target_weight = current_weight

        if runtime_gross_cap >= 0.0:
            target_weight = float(np.clip(target_weight, -runtime_gross_cap, runtime_gross_cap))

        target_notional = equity_before * target_weight
        target_qty = 0.0
        if abs(prev_close) > 1e-12:
            target_qty = _quantize_amount_nb(target_notional / prev_close, amount_step, min_qty)

        # P1-6: D1 max-hold enforcement (mirrors live safety_guards MAX_HOLD_BARS)
        if force_close_next:
            target_qty = 0.0
            force_close_next = False
            hold_bars = 0
        elif qty != 0.0 and max_hold_bars > 0:
            hold_bars += 1
            if hold_bars >= max_hold_bars:
                force_close_next = True
        else:
            hold_bars = 0

        diff_qty = _quantize_amount_nb(target_qty - qty, amount_step, min_qty)

        # P1-5: min notional filter (mirrors live diff_notional < 25.0 skip)
        if abs(diff_qty) * prev_close < min_notional_usd:
            diff_qty = 0.0

        if abs(diff_qty) > 0.0:
            side = 1.0 if diff_qty > 0.0 else -1.0
            exec_price = px_open * (1.0 + slippage * side)
            trade_notional = diff_qty * exec_price
            fee = abs(diff_qty) * exec_price * fee_rate
            prev_qty_signed = qty
            cash -= trade_notional
            cash -= fee
            qty += diff_qty
            n_trades += 1
            fee_paid += fee
            slippage_paid += abs(diff_qty) * px_open * slippage
            new_qty_signed = qty
            current_regime_idx = int(bucket_codes[signal_idx])
            if current_regime_idx >= MAX_REGIME_BUCKETS:
                raise RuntimeError(
                    f"current_regime_idx {current_regime_idx} >= MAX_REGIME_BUCKETS {MAX_REGIME_BUCKETS}"
                )
            new_side = 1 if new_qty_signed > 1e-12 else (-1 if new_qty_signed < -1e-12 else 0)

            if prev_side == 0 and new_side != 0:
                entry_lots.append([abs(diff_qty), exec_price, fee, current_regime_idx])
                prev_side = new_side
            elif prev_side != 0 and new_side == prev_side and (
                (prev_side > 0 and diff_qty > 0.0) or (prev_side < 0 and diff_qty < 0.0)
            ):
                entry_lots.append([abs(diff_qty), exec_price, fee, current_regime_idx])
            else:
                if prev_side != 0 and new_side == prev_side:
                    close_qty_total = abs(diff_qty)
                    fee_close_total = fee
                else:
                    if entry_lots:
                        close_qty_total = sum(lot[0] for lot in entry_lots)
                    else:
                        close_qty_total = 0.0
                    if abs(diff_qty) > 0.0:
                        close_share = close_qty_total / abs(diff_qty)
                    else:
                        close_share = 0.0
                    fee_close_total = fee * close_share
                fee_open_new = fee - fee_close_total
                if close_qty_total > 1e-12:
                    fee_per_unit_close = fee_close_total / close_qty_total
                else:
                    fee_per_unit_close = 0.0
                sign_prev = prev_side
                remaining_to_close = close_qty_total
                while remaining_to_close > 1e-12 and entry_lots:
                    lot_qty, lot_price, lot_entry_fee, lot_regime = entry_lots[0]
                    if lot_qty <= remaining_to_close + 1e-12:
                        gross = (exec_price - lot_price) * lot_qty * float(sign_prev)
                        slice_close_fee = fee_per_unit_close * lot_qty
                        net = gross - lot_entry_fee - slice_close_fee
                        if net > 0.0:
                            n_wins += 1
                            total_win_pnl += net
                            regime_n_wins[lot_regime] += 1
                        elif net < 0.0:
                            n_losses += 1
                            total_loss_pnl += abs(net)
                            regime_n_losses[lot_regime] += 1
                        remaining_to_close -= lot_qty
                        entry_lots.pop(0)
                    else:
                        portion = remaining_to_close
                        gross = (exec_price - lot_price) * portion * float(sign_prev)
                        partial_entry_fee = lot_entry_fee * (portion / lot_qty)
                        slice_close_fee = fee_per_unit_close * portion
                        net = gross - partial_entry_fee - slice_close_fee
                        if net > 0.0:
                            n_wins += 1
                            total_win_pnl += net
                            regime_n_wins[lot_regime] += 1
                        elif net < 0.0:
                            n_losses += 1
                            total_loss_pnl += abs(net)
                            regime_n_losses[lot_regime] += 1
                        entry_lots[0][0] = lot_qty - portion
                        entry_lots[0][2] = lot_entry_fee - partial_entry_fee
                        remaining_to_close = 0.0
                if not entry_lots:
                    prev_side = 0
                if new_side != 0 and new_side != prev_side:
                    new_open_qty = abs(new_qty_signed)
                    entry_lots.append([new_open_qty, exec_price, fee_open_new, current_regime_idx])
                    prev_side = new_side

        equity_after = cash + qty * next_open
        if equity_after > peak_equity:
            peak_equity = equity_after
        dd = equity_after / peak_equity - 1.0
        if dd < max_drawdown:
            max_drawdown = dd

        bar_net = equity_after / equity_before - 1.0
        net_ret.append(float(bar_net))
        if return_trace:
            target_weight_trace.append(float(target_weight))
            requested_weight_trace.append(float(requested_weight))
            signal_pct_trace.append(float(signal_pct))
            role_idx_trace.append(int(role_idx))
            cooldown_trace.append(int(cooldown_bars_left))
            confirm_side_trace.append(int(confirm_side))
            confirm_count_trace.append(int(confirm_count))
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

    open_position_at_end = bool(entry_lots)
    open_roundtrip_realized_pnl = 0.0
    open_roundtrip_fees_paid = float(sum(lot[2] for lot in entry_lots)) if entry_lots else 0.0

    decided = n_wins + n_losses
    avg_win_size = total_win_pnl / n_wins if n_wins > 0 else 0.0
    avg_loss_size = total_loss_pnl / n_losses if n_losses > 0 else 0.0
    payoff_ratio = avg_win_size / avg_loss_size if avg_loss_size > 0.0 else float("inf")
    if decided > 0 and avg_loss_size > 0.0:
        win_rate = n_wins / decided
        _payoff_denom = payoff_ratio if payoff_ratio > 0.0 else 1e-12
        kelly_fraction = max(-1.0, min(1.0, win_rate - (1.0 - win_rate) / _payoff_denom))
    else:
        kelly_fraction = 0.0
    payoff_ratio_safe = payoff_ratio if payoff_ratio != float("inf") else 0.0

    regime_n_wins_list = [int(x) for x in regime_n_wins]
    regime_n_losses_list = [int(x) for x in regime_n_losses]
    regime_win_rate_list = [
        float(regime_n_wins_list[i]) / float(regime_n_wins_list[i] + regime_n_losses_list[i])
        if (regime_n_wins_list[i] + regime_n_losses_list[i]) > 0 else 0.0
        for i in range(MAX_REGIME_BUCKETS)
    ]

    if return_trace:
        return {
            "total_return": float(total_return),
            "n_trades": int(n_trades),
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
            "open_position_at_end": int(open_position_at_end),
            "open_roundtrip_realized_pnl": float(open_roundtrip_realized_pnl),
            "open_roundtrip_fees_paid": float(open_roundtrip_fees_paid),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "final_equity": float(final_equity),
            "fee_paid": float(fee_paid),
            "slippage_paid": float(slippage_paid),
            "funding_paid": float(funding_paid),
            "funding_events": int(funding_events),
            "avg_win_size": float(avg_win_size),
            "avg_loss_size": float(avg_loss_size),
            "total_win_pnl": float(total_win_pnl),
            "total_loss_pnl": float(total_loss_pnl),
            "payoff_ratio": float(payoff_ratio_safe),
            "kelly_fraction": float(kelly_fraction),
            "regime_n_wins": regime_n_wins_list,
            "regime_n_losses": regime_n_losses_list,
            "regime_win_rate": regime_win_rate_list,
            "daily_metrics": {
                "avg_daily_return": float(avg_daily),
                "daily_target_hit_rate": float(daily_target_hit_rate),
                "daily_win_rate": float(daily_win_rate),
                "worst_day": float(worst_day),
                "best_day": float(best_day),
            },
            "trace": {
                "bar_net": np.asarray(net_ret, dtype="float64"),
                "target_weight": np.asarray(target_weight_trace, dtype="float64"),
                "requested_weight": np.asarray(requested_weight_trace, dtype="float64"),
                "signal_pct": np.asarray(signal_pct_trace, dtype="float64"),
                "role_idx": np.asarray(role_idx_trace, dtype="int64"),
                "cooldown_bars_left": np.asarray(cooldown_trace, dtype="int64"),
                "confirm_side": np.asarray(confirm_side_trace, dtype="int64"),
                "confirm_count": np.asarray(confirm_count_trace, dtype="int64"),
            },
        }
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
        n_wins,
        n_losses,
        int(open_position_at_end),
        float(open_roundtrip_realized_pnl),
        float(open_roundtrip_fees_paid),
        float(total_win_pnl),
        float(total_loss_pnl),
        tuple(int(x) for x in regime_n_wins),
        tuple(int(x) for x in regime_n_losses),
    )


def _realistic_overlay_replay_kernel_numba_impl(
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
    order_imbalance: np.ndarray,
    buy_volume_share: np.ndarray,
    close_location_value: np.ndarray,
    body_to_range: np.ndarray,
    wick_skew: np.ndarray,
    candle_micro_score: np.ndarray,
    oi_rel: np.ndarray,
    basis_rate: np.ndarray,
    top_pos_log_ratio: np.ndarray,
    taker_buy_sell_log_ratio: np.ndarray,
    range_bps: np.ndarray,
    volume_ratio: np.ndarray,
    dc_trend_05: np.ndarray,
    dc_run_05: np.ndarray,
    mapping: np.ndarray,
    initial_cash: float,
    fee_rate: float,
    slippage: float,
    amount_step: float,
    min_qty: float,
    no_trade_band_pct: float,
    signal_gate_pct: float,
    regime_buffer_mult: float,
    confirm_bars: int,
    state_specialists: np.ndarray,
    role_signal_gate_mults: np.ndarray,
    role_regime_buffer_mults: np.ndarray,
    abstain_edge_pct: float,
    specialist_isolation_mult: float,
    liquidity_range_gate_bp: float,
    liquidity_volume_ratio_floor: float,
    short_horizon_abstain_mult: float,
    microstructure_align_gate_pct: float,
    dc_align_gate_pct: float,
    min_alignment_votes: int,
    bars_per_day: int,
    daily_target: float,
    bar_factor: float,
    min_notional_usd: float = 25.0,
    max_hold_bars: int = 288,
    initial_cooldown_bars: int = 0,
    runtime_gross_cap: float = -1.0,
    final_decision_cooldown_override: int | None = None,
) -> tuple[float, int, float, float, float, float, float, float, float, float, float, float, float, int, int, int, int, float, float, float, float, tuple[int, ...], tuple[int, ...]]:
    return _realistic_overlay_replay_kernel_impl(
        open_p,
        close_p,
        funding_rates,
        bucket_codes,
        regime,
        breadth,
        vol_ann,
        equity_corr_gross_scale,
        equity_corr_regime_mult,
        smooth_signal_matrix,
        library_signal_pos,
        library_rebalance_bars,
        library_regime_threshold,
        library_breadth_threshold,
        library_target_vol_ann,
        library_gross_cap,
        library_kill_switch_pct,
        library_cooldown_days,
        order_imbalance,
        buy_volume_share,
        close_location_value,
        body_to_range,
        wick_skew,
        candle_micro_score,
        oi_rel,
        basis_rate,
        top_pos_log_ratio,
        taker_buy_sell_log_ratio,
        range_bps,
        volume_ratio,
        dc_trend_05,
        dc_run_05,
        mapping,
        initial_cash,
        fee_rate,
        slippage,
        amount_step,
        min_qty,
        no_trade_band_pct,
        signal_gate_pct,
        regime_buffer_mult,
        confirm_bars,
        state_specialists,
        role_signal_gate_mults,
        role_regime_buffer_mults,
        abstain_edge_pct,
        specialist_isolation_mult,
        liquidity_range_gate_bp,
        liquidity_volume_ratio_floor,
        short_horizon_abstain_mult,
        microstructure_align_gate_pct,
        dc_align_gate_pct,
        min_alignment_votes,
        bars_per_day,
        daily_target,
        bar_factor,
        None,
        False,
        min_notional_usd,
        max_hold_bars,
        initial_cooldown_bars,
        runtime_gross_cap,
        final_decision_cooldown_override,
    )


# Keep the realistic replay on the Python implementation so trace-capable behavior
# and execution-gene parity stay aligned across search, validation, and runtime.
_realistic_overlay_replay_kernel = _realistic_overlay_replay_kernel_impl


def build_overlay_inputs(df: pd.DataFrame, pairs: tuple[str, ...], regime_pair: str) -> dict[str, pd.Series]:
    close = pd.concat([df[f"{asset}_close"].rename(asset) for asset in pairs], axis=1).sort_index()
    daily_close = close.resample("1D").last().dropna()
    regime = 0.60 * daily_close[regime_pair].pct_change(3) + 0.40 * daily_close[regime_pair].pct_change(14)
    breadth = (daily_close.pct_change(3) > -PAIRWISE_BREADTH_NOISE_EPSILON).mean(axis=1)
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
    strict_external_asof: bool = False,
) -> np.ndarray:
    route_state_mode = normalize_route_state_mode(route_state_mode)
    day_index = index.normalize()
    completed_day_index = day_index - pd.Timedelta(days=1)
    effective_day_index = completed_day_index if strict_external_asof else day_index
    regime_daily = overlay_inputs["btc_regime_daily"].reindex(effective_day_index, method="ffill").fillna(0.0)
    breadth_daily = overlay_inputs["breadth_daily"].reindex(effective_day_index, method="ffill").fillna(0.0)
    is_up = (regime_daily >= 0.0).astype(np.int8)
    is_broad = (breadth_daily >= breadth_threshold).astype(np.int8)
    base_codes = (is_up * 2 + is_broad).to_numpy(dtype="int8")
    if route_state_mode == ROUTE_STATE_MODE_BASE:
        return base_codes
    corr_bucket = (
        overlay_inputs["equity_corr_bucket_daily"]
        .reindex(effective_day_index, method="ffill")
        .fillna("equity_unknown")
    )
    corr_codes = corr_bucket.map(EQUITY_CORR_BUCKET_CODES).fillna(EQUITY_CORR_BUCKET_CODES["equity_unknown"]).astype("int8")
    return corr_codes.to_numpy(dtype="int8") * len(BASE_ROUTE_STATE_NAMES) + base_codes


def load_or_fetch_funding(
    symbol: str,
    start: str,
    end: str,
    *,
    require_coverage: bool = True,
    coverage_tolerance_days: float = 1.0,
) -> pd.DataFrame:
    """Load funding rates merging DB + CSV cache.

    Validates that the union covers ``end - coverage_tolerance_days`` so a
    partially-populated DB cannot silently look authoritative. Falls back to
    Binance fetch only when neither DB nor CSV is present.
    """
    pg_frame = gp.load_funding_rates(symbol, start, end)
    path = gp.DATA_DIR / f"{symbol}_funding_{start}_{end}.csv"
    csv_frame = pd.DataFrame(columns=["fundingTime", "fundingRate"])
    if path.exists():
        csv_frame = pd.read_csv(path)
        csv_frame["fundingTime"] = pd.to_datetime(csv_frame["fundingTime"], utc=True, format="mixed")
        csv_frame["fundingRate"] = pd.to_numeric(csv_frame["fundingRate"], errors="coerce")
        csv_frame = csv_frame.dropna(subset=["fundingTime", "fundingRate"])

    if pg_frame.empty and csv_frame.empty:
        df = fetch_funding_rates(
            symbol,
            datetime.fromisoformat(start).replace(tzinfo=UTC),
            datetime.fromisoformat(end).replace(tzinfo=UTC) + pd.Timedelta(days=1),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    else:
        frames = [frame for frame in (pg_frame, csv_frame) if not frame.empty]
        df = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["fundingTime"], keep="first")
            .sort_values("fundingTime")
            .reset_index(drop=True)
        )

    if require_coverage:
        gp.validate_funding_coverage(
            df,
            symbol,
            start,
            end,
            tolerance_days=coverage_tolerance_days,
            raise_on_gap=True,
        )
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
    daily_win = np.asarray([m.get("daily_win_rate", 0.0) for m in per_pair.values()], dtype="float64")
    daily_target = np.asarray([m.get("daily_target_hit_rate", 0.0) for m in per_pair.values()], dtype="float64")
    n_trades = np.asarray([m.get("n_trades", 0.0) for m in per_pair.values()], dtype="float64")
    return {
        "mean_avg_daily_return": float(np.mean(avg_daily)),
        "worst_pair_avg_daily_return": float(np.min(avg_daily)),
        "best_pair_avg_daily_return": float(np.max(avg_daily)),
        "positive_pair_count": int(np.sum(avg_daily > 0.0)),
        "mean_daily_win_rate": float(np.mean(daily_win)),
        "worst_pair_daily_win_rate": float(np.min(daily_win)),
        "mean_daily_target_hit_rate": float(np.mean(daily_target)),
        "worst_pair_daily_target_hit_rate": float(np.min(daily_target)),
        "mean_n_trades": float(np.mean(n_trades)),
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
    commission_pct: float | None = None,
    no_trade_band_pct: float | None = None,
    signal_gate_pct: float | None = None,
    regime_buffer_mult: float | None = None,
    confirm_bars: int | None = None,
    *,
    use_equity_corr_risk: bool = False,
    execution_gene: dict[str, Any] | None = None,
    state_specialists: tuple[int, ...] | list[int] | None = None,
    return_trace: bool = False,
) -> dict[str, Any]:
    route_state_mode = normalize_route_state_mode(context.get("route_state_mode"))
    expected_state_count = len(route_state_names(route_state_mode))
    mapping = normalize_mapping_indices(mapping, route_state_mode)
    execution_profile = legacy_execution_profile() if execution_gene is None else derive_execution_profile(execution_gene)
    effective_commission_pct = float(gp.COMMISSION_PCT if commission_pct is None else commission_pct)
    effective_no_trade_band_pct = float(gp.NO_TRADE_BAND if no_trade_band_pct is None else no_trade_band_pct)
    effective_signal_gate_pct = 0.0 if signal_gate_pct is None else float(signal_gate_pct)
    effective_regime_buffer_mult = 0.0 if regime_buffer_mult is None else float(regime_buffer_mult)
    effective_confirm_bars = 1 if confirm_bars is None else int(confirm_bars)
    state_specialists_source = state_specialists if state_specialists is not None else context.get("route_state_specialists")
    if state_specialists_source is None or len(state_specialists_source) != expected_state_count:
        state_specialists_source = default_state_specialists_for_router(route_state_names(route_state_mode))
    effective_state_specialists = np.asarray(state_specialists_source, dtype="int64")
    role_signal_gate_mults = np.asarray(execution_profile["role_signal_gate_mults"], dtype="float64")
    role_regime_buffer_mults = np.asarray(execution_profile["role_regime_buffer_mults"], dtype="float64")
    abstain_edge_pct = float(execution_profile["abstain_edge_pct"])
    specialist_isolation_mult = float(execution_profile["specialist_isolation_mult"])
    liquidity_range_gate_bp = float(execution_profile["liquidity_range_gate_bp"])
    liquidity_volume_ratio_floor = float(execution_profile["liquidity_volume_ratio_floor"])
    short_horizon_abstain_mult = float(execution_profile["short_horizon_abstain_mult"])
    microstructure_align_gate_pct = float(execution_profile["microstructure_align_gate_pct"])
    dc_align_gate_pct = float(execution_profile["dc_align_gate_pct"])
    min_alignment_votes = int(execution_profile["min_alignment_votes"])
    corr_gross_scale = context["equity_corr_gross_scale"] if use_equity_corr_risk else np.ones_like(context["regime"])
    corr_regime_mult = context["equity_corr_regime_mult"] if use_equity_corr_risk else np.ones_like(context["regime"])
    close_location_value = _context_feature_array(context, "close_location_value", fill_value=0.0)
    body_to_range = _context_feature_array(context, "body_to_range", fill_value=0.0)
    wick_skew = _context_feature_array(context, "wick_skew", fill_value=0.0)
    candle_micro_score = _context_feature_array(context, "candle_micro_score", fill_value=0.0)
    oi_rel = _context_feature_array(context, "oi_rel", fill_value=1.0)
    basis_rate = _context_feature_array(context, "basis_rate", fill_value=0.0)
    top_pos_log_ratio = _context_feature_array(context, "top_pos_log_ratio", fill_value=0.0)
    taker_buy_sell_log_ratio = _context_feature_array(context, "taker_buy_sell_log_ratio", fill_value=0.0)
    range_bps = _context_feature_array(context, "range_bps", fill_value=0.0)
    volume_ratio = _context_feature_array(context, "volume_ratio", fill_value=0.0)
    # Read gate overrides once and forward to either kernel path so numba
    # JIT body never has to call os.environ at compile time.
    _gate_scale_for_kernel, _gate_disabled_for_kernel = _gate_overrides()
    if fast_engine == "numba" and not return_trace:
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
            context["order_imbalance"],
            context["buy_volume_share"],
            close_location_value,
            body_to_range,
            wick_skew,
            candle_micro_score,
            oi_rel,
            basis_rate,
            top_pos_log_ratio,
            taker_buy_sell_log_ratio,
            range_bps,
            volume_ratio,
            context["dc_trend_05"],
            context["dc_run_05"],
            np.asarray(mapping, dtype="int64"),
            float(gp.INITIAL_CASH),
            float(effective_commission_pct),
            float(effective_no_trade_band_pct),
            float(effective_signal_gate_pct),
            float(effective_regime_buffer_mult),
            int(effective_confirm_bars),
            effective_state_specialists,
            role_signal_gate_mults,
            role_regime_buffer_mults,
            float(abstain_edge_pct),
            float(specialist_isolation_mult),
            float(liquidity_range_gate_bp),
            float(liquidity_volume_ratio_floor),
            float(short_horizon_abstain_mult),
            float(microstructure_align_gate_pct),
            float(dc_align_gate_pct),
            int(min_alignment_votes),
            int(BARS_PER_DAY),
            float(gp.DAILY_TARGET_PCT),
            float(BAR_FACTOR),
            float(_gate_scale_for_kernel),
            bool(_gate_disabled_for_kernel),
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
    confirm_side = 0
    confirm_count = 0
    last_role_idx = -1
    target_weight_trace: list[float] = []
    requested_weight_trace: list[float] = []
    signal_pct_trace: list[float] = []
    role_idx_trace: list[int] = []
    _gate_threshold_scale, _gate_disabled = _gate_overrides()

    for i in range(len(close) - 1):
        active_idx = int(mapping[int(bucket_codes[i])])
        role_idx = int(effective_state_specialists[int(bucket_codes[i])])
        params = library[active_idx]

        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        role_changed = role_idx != last_role_idx
        if role_changed:
            confirm_side = 0
            confirm_count = 0
            last_role_idx = role_idx

        signal_pct = float(np.clip(smooth_signal_matrix[signal_positions[active_idx], i], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(regime[i])
        breadth_score = float(breadth[i])
        role_signal_gate_pct = float(effective_signal_gate_pct) * float(role_signal_gate_mults[role_idx])
        role_regime_buffer_mult = float(effective_regime_buffer_mult) * float(role_regime_buffer_mults[role_idx])
        effective_regime_threshold = float(params.regime_threshold) * float(equity_corr_regime_mult[i]) * (1.0 + float(role_regime_buffer_mult)) * _gate_threshold_scale
        effective_gross_cap = float(params.gross_cap) * float(equity_corr_gross_scale[i])
        if _gate_disabled:
            long_ok = True
            short_ok = True
        else:
            long_ok = regime_score >= effective_regime_threshold and breadth_score >= params.breadth_threshold
            short_ok = regime_score <= -effective_regime_threshold and breadth_score <= (1.0 - params.breadth_threshold)
        if abs(signal_pct) < float(role_signal_gate_pct + abstain_edge_pct):
            requested_weight = 0.0
        elif requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0
        else:
            requested_side = 0
            if requested_weight > 1e-12:
                requested_side = 1
            elif requested_weight < -1e-12:
                requested_side = -1
            micro_score = blended_microstructure_score(
                float(context["order_imbalance"][i]),
                float(context["buy_volume_share"][i]),
                float(close_location_value[i]),
                float(body_to_range[i]),
                float(wick_skew[i]),
            )
            positioning_score = derivative_positioning_score(
                float(oi_rel[i]),
                float(basis_rate[i]),
                float(top_pos_log_ratio[i]),
                float(taker_buy_sell_log_ratio[i]),
            )
            if current_weight * requested_side <= 0.0 and requested_side * positioning_score < -0.35 and requested_side * micro_score < 0.10:
                requested_weight = 0.0
            if requested_weight != 0.0 and current_weight * requested_side <= 0.0 and should_abstain_for_alignment(
                requested_side,
                micro_score,
                dc_alignment_score(
                    float(context["dc_trend_05"][i]),
                    float(context["dc_run_05"][i]),
                ),
                float(microstructure_align_gate_pct),
                float(dc_align_gate_pct),
                int(min_alignment_votes),
            ):
                requested_weight = 0.0
            elif should_abstain_for_liquidity(
                requested_side,
                current_weight,
                float(range_bps[i]),
                float(volume_ratio[i]),
                float(liquidity_range_gate_bp),
                float(liquidity_volume_ratio_floor),
            ):
                requested_weight = 0.0
            elif should_abstain_for_weak_tape(
                requested_side,
                current_weight,
                micro_score,
                dc_alignment_score(
                    float(context["dc_trend_05"][i]),
                    float(context["dc_run_05"][i]),
                ),
                float(range_bps[i]),
                float(volume_ratio[i]),
                float(equity_corr_gross_scale[i]),
                float(short_horizon_abstain_mult),
            ):
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
            role_confirm_bars = int(effective_confirm_bars)
            if role_changed:
                role_confirm_bars += int(np.rint(specialist_isolation_mult * 2.0))
            requested_side = 0
            if requested_weight > 1e-12:
                requested_side = 1
            elif requested_weight < -1e-12:
                requested_side = -1
            if requested_side == 0:
                confirm_side = 0
                confirm_count = 0
            elif requested_side == confirm_side:
                confirm_count += 1
            else:
                confirm_side = requested_side
                confirm_count = 1
            if requested_side != 0 and current_weight * requested_side <= 0.0 and confirm_count < role_confirm_bars:
                requested_weight = 0.0
            target_weight = requested_weight

        if abs(target_weight - current_weight) < effective_no_trade_band_pct / 100.0:
            target_weight = current_weight

        turnover = abs(target_weight - current_weight)
        if turnover > 0.001:
            n_trades += 1

        price_ret = float(close[i + 1] / close[i] - 1.0)
        bar_net = target_weight * price_ret - turnover * effective_commission_pct * 2
        equity *= (1.0 + bar_net)
        peak_equity = max(peak_equity, equity)
        current_weight = target_weight
        if return_trace:
            target_weight_trace.append(float(target_weight))
            requested_weight_trace.append(float(requested_weight))
            signal_pct_trace.append(float(signal_pct))
            role_idx_trace.append(int(role_idx))
        net_ret.append(bar_net)
        equity_curve.append(float(equity))

    result = {
        "total_return": float(equity / gp.INITIAL_CASH - 1.0),
        "n_trades": int(n_trades),
        "sharpe": float(np.mean(net_ret) / np.std(net_ret) * BAR_FACTOR) if len(net_ret) > 1 and np.std(net_ret) > 1e-12 else 0.0,
        "max_drawdown": float(np.min(np.asarray(equity_curve) / np.maximum.accumulate(np.asarray(equity_curve)) - 1.0)),
        "final_equity": float(equity),
        "daily_metrics": gp.compute_daily_metrics(np.asarray(net_ret, dtype="float64")),
    }
    if return_trace:
        result["trace"] = {
            "target_weight": np.asarray(target_weight_trace, dtype="float64"),
            "requested_weight": np.asarray(requested_weight_trace, dtype="float64"),
            "signal_pct": np.asarray(signal_pct_trace, dtype="float64"),
            "role_idx": np.asarray(role_idx_trace, dtype="int64"),
        }
    return result


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
    execution_gene: dict[str, Any] | None = None,
    state_specialists: tuple[int, ...] | list[int] | None = None,
    engine: str = "auto",
    entry_keep_flags: np.ndarray | None = None,
    return_trace: bool = False,
    initial_cooldown_bars: int = 0,
    min_notional_usd: float = 25.0,
    max_hold_bars: int = 288,
    runtime_gross_cap: float | None = None,
    final_decision_cooldown_override: int | None = None,
) -> dict[str, Any]:
    route_state_mode = normalize_route_state_mode(context.get("route_state_mode"))
    expected_state_count = len(route_state_names(route_state_mode))
    mapping = normalize_mapping_indices(mapping, route_state_mode)
    execution_profile = legacy_execution_profile() if execution_gene is None else derive_execution_profile(execution_gene)
    state_specialists_source = state_specialists if state_specialists is not None else context.get("route_state_specialists")
    if state_specialists_source is None or len(state_specialists_source) != expected_state_count:
        state_specialists_source = default_state_specialists_for_router(route_state_names(route_state_mode))
    effective_state_specialists = np.asarray(state_specialists_source, dtype="int64")
    corr_gross_scale = context["equity_corr_gross_scale"] if use_equity_corr_risk else np.ones_like(context["regime"])
    corr_regime_mult = context["equity_corr_regime_mult"] if use_equity_corr_risk else np.ones_like(context["regime"])
    close_location_value = _context_feature_array(context, "close_location_value", fill_value=0.0)
    body_to_range = _context_feature_array(context, "body_to_range", fill_value=0.0)
    wick_skew = _context_feature_array(context, "wick_skew", fill_value=0.0)
    candle_micro_score = _context_feature_array(context, "candle_micro_score", fill_value=0.0)
    oi_rel = _context_feature_array(context, "oi_rel", fill_value=1.0)
    basis_rate = _context_feature_array(context, "basis_rate", fill_value=0.0)
    top_pos_log_ratio = _context_feature_array(context, "top_pos_log_ratio", fill_value=0.0)
    taker_buy_sell_log_ratio = _context_feature_array(context, "taker_buy_sell_log_ratio", fill_value=0.0)
    range_bps = _context_feature_array(context, "range_bps", fill_value=0.0)
    volume_ratio = _context_feature_array(context, "volume_ratio", fill_value=0.0)
    if execution_gene is not None:
        fee_rate = float(execution_profile["fee_rate"])
        slippage = float(execution_profile["slippage"])
        amount_step = float(execution_profile["amount_step"])
        min_qty = float(execution_profile["min_qty"])
        no_trade_band_pct = float(execution_profile["no_trade_band_pct"])
        signal_gate_pct = float(execution_profile["signal_gate_pct"])
        regime_buffer_mult = float(execution_profile["regime_buffer_mult"])
        confirm_bars = int(execution_profile["confirm_bars"])
    else:
        no_trade_band_pct = float(gp.NO_TRADE_BAND)
        signal_gate_pct = 0.0
        regime_buffer_mult = 0.0
        confirm_bars = 1
        liquidity_range_gate_bp = 0.0
        liquidity_volume_ratio_floor = 0.0
        short_horizon_abstain_mult = 0.0
    min_alignment_votes = int(execution_profile["min_alignment_votes"])
    if execution_gene is not None:
        liquidity_range_gate_bp = float(execution_profile["liquidity_range_gate_bp"])
        liquidity_volume_ratio_floor = float(execution_profile["liquidity_volume_ratio_floor"])
        short_horizon_abstain_mult = float(execution_profile["short_horizon_abstain_mult"])
    kernel = _realistic_overlay_replay_kernel
    if str(engine) == "python" or entry_keep_flags is not None or return_trace:
        kernel = _realistic_overlay_replay_kernel_impl
    base_args = (
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
        context["order_imbalance"],
        context["buy_volume_share"],
        close_location_value,
        body_to_range,
        wick_skew,
        candle_micro_score,
        oi_rel,
        basis_rate,
        top_pos_log_ratio,
        taker_buy_sell_log_ratio,
        range_bps,
        volume_ratio,
        context["dc_trend_05"],
        context["dc_run_05"],
        np.asarray(mapping, dtype="int64"),
        float(gp.INITIAL_CASH),
        float(fee_rate),
        float(slippage),
        float(amount_step),
        float(min_qty),
        float(no_trade_band_pct),
        float(signal_gate_pct),
        float(regime_buffer_mult),
        int(confirm_bars),
        effective_state_specialists,
        np.asarray(execution_profile["role_signal_gate_mults"], dtype="float64"),
        np.asarray(execution_profile["role_regime_buffer_mults"], dtype="float64"),
        float(execution_profile["abstain_edge_pct"]),
        float(execution_profile["specialist_isolation_mult"]),
        float(liquidity_range_gate_bp),
        float(liquidity_volume_ratio_floor),
        float(short_horizon_abstain_mult),
        float(execution_profile["countertrend_weight_scale"]),
        bool(execution_profile["countertrend_rebalance_bypass"]),
        float(execution_profile["countertrend_signal_floor_pct"]),
        float(execution_profile["countertrend_regime_score_cap"]),
        float(execution_profile["countertrend_breadth_slack"]),
        float(execution_profile["countertrend_microstructure_floor"]),
        float(execution_profile["countertrend_positioning_floor"]),
        float(execution_profile["countertrend_dc_floor"]),
        float(execution_profile["countertrend_range_bps_floor"]),
        float(execution_profile["countertrend_volume_ratio_floor"]),
        float(execution_profile["microstructure_align_gate_pct"]),
        float(execution_profile["dc_align_gate_pct"]),
        int(min_alignment_votes),
        int(BARS_PER_DAY),
        float(gp.DAILY_TARGET_PCT),
        float(BAR_FACTOR),
    )
    funding_unmatched = int(context.get("funding_unmatched", 0))
    runtime_cap_arg = -1.0 if runtime_gross_cap is None else float(runtime_gross_cap)
    if kernel is _realistic_overlay_replay_kernel_impl:
        result = kernel(
            *base_args,
            entry_keep_flags,
            return_trace,
            float(min_notional_usd),
            int(max_hold_bars),
            int(initial_cooldown_bars),
            runtime_cap_arg,
            final_decision_cooldown_override,
        )
    else:
        result = kernel(
            *base_args,
            float(min_notional_usd),
            int(max_hold_bars),
            int(initial_cooldown_bars),
            runtime_cap_arg,
            final_decision_cooldown_override,
        )
    if kernel is _realistic_overlay_replay_kernel_impl:
        if isinstance(result, tuple):
            n_wins_v = int(result[14]) if len(result) > 14 else 0
            n_losses_v = int(result[15]) if len(result) > 15 else 0
            open_at_end_v = int(result[16]) if len(result) > 16 else 0
            open_realized_pnl_v = float(result[17]) if len(result) > 17 else 0.0
            open_fees_v = float(result[18]) if len(result) > 18 else 0.0
            total_win_pnl_v = float(result[19]) if len(result) > 19 else 0.0
            total_loss_pnl_v = float(result[20]) if len(result) > 20 else 0.0
            regime_n_wins_v = list(result[21]) if len(result) > 21 else [0] * MAX_REGIME_BUCKETS
            regime_n_losses_v = list(result[22]) if len(result) > 22 else [0] * MAX_REGIME_BUCKETS
            n_trades_v = int(result[1])
            decided = n_wins_v + n_losses_v
            roundtrip_winrate = (n_wins_v / decided) if decided > 0 else 0.0
            avg_win_size_v = total_win_pnl_v / n_wins_v if n_wins_v > 0 else 0.0
            avg_loss_size_v = total_loss_pnl_v / n_losses_v if n_losses_v > 0 else 0.0
            payoff_ratio_v = avg_win_size_v / avg_loss_size_v if avg_loss_size_v > 0.0 else 0.0
            if decided > 0 and avg_loss_size_v > 0.0:
                win_rate_v = n_wins_v / decided
                kelly_v = max(-1.0, min(1.0, win_rate_v - (1.0 - win_rate_v) / (payoff_ratio_v if payoff_ratio_v > 0.0 else 1e-12)))
            else:
                kelly_v = 0.0
            regime_win_rate_v = [
                float(regime_n_wins_v[i]) / float(regime_n_wins_v[i] + regime_n_losses_v[i])
                if (regime_n_wins_v[i] + regime_n_losses_v[i]) > 0 else 0.0
                for i in range(len(regime_n_wins_v))
            ]
            return {
                "avg_daily_return": float(result[5]),
                "total_return": float(result[0]),
                "max_drawdown": float(result[3]),
                "sharpe": float(result[2]),
                "daily_target_hit_rate": float(result[6]),
                "daily_win_rate": float(result[7]),
                "worst_day": float(result[8]),
                "best_day": float(result[9]),
                "n_trades": n_trades_v,
                "n_wins": n_wins_v,
                "n_losses": n_losses_v,
                "roundtrip_win_rate": float(roundtrip_winrate),
                "open_position_at_end": bool(open_at_end_v),
                "open_roundtrip_realized_pnl": float(open_realized_pnl_v),
                "open_roundtrip_fees_paid": float(open_fees_v),
                "fee_paid": float(result[10]),
                "slippage_paid": float(result[11]),
                "funding_paid": float(result[12]),
                "funding_events": int(result[13]),
                "funding_unmatched": funding_unmatched,
                "final_equity": float(result[4]),
                "avg_win_size": float(avg_win_size_v),
                "avg_loss_size": float(avg_loss_size_v),
                "total_win_pnl": float(total_win_pnl_v),
                "total_loss_pnl": float(total_loss_pnl_v),
                "payoff_ratio": float(payoff_ratio_v),
                "kelly_fraction": float(kelly_v),
                "regime_n_wins": regime_n_wins_v,
                "regime_n_losses": regime_n_losses_v,
                "regime_win_rate": regime_win_rate_v,
            }
        n_wins_v = int(result.get("n_wins", 0))
        n_losses_v = int(result.get("n_losses", 0))
        decided = n_wins_v + n_losses_v
        roundtrip_winrate = (n_wins_v / decided) if decided > 0 else 0.0
        total_win_pnl_v = float(result.get("total_win_pnl", 0.0))
        total_loss_pnl_v = float(result.get("total_loss_pnl", 0.0))
        avg_win_size_v = total_win_pnl_v / n_wins_v if n_wins_v > 0 else 0.0
        avg_loss_size_v = total_loss_pnl_v / n_losses_v if n_losses_v > 0 else 0.0
        payoff_ratio_v = avg_win_size_v / avg_loss_size_v if avg_loss_size_v > 0.0 else 0.0
        if decided > 0 and avg_loss_size_v > 0.0:
            win_rate_v = n_wins_v / decided
            kelly_v = max(-1.0, min(1.0, win_rate_v - (1.0 - win_rate_v) / (payoff_ratio_v if payoff_ratio_v > 0.0 else 1e-12)))
        else:
            kelly_v = 0.0
        regime_n_wins_v = list(result.get("regime_n_wins", [0] * MAX_REGIME_BUCKETS))
        regime_n_losses_v = list(result.get("regime_n_losses", [0] * MAX_REGIME_BUCKETS))
        regime_win_rate_v = [
            float(regime_n_wins_v[i]) / float(regime_n_wins_v[i] + regime_n_losses_v[i])
            if (regime_n_wins_v[i] + regime_n_losses_v[i]) > 0 else 0.0
            for i in range(len(regime_n_wins_v))
        ]
        metrics = {
            "avg_daily_return": float(result["daily_metrics"]["avg_daily_return"]),
            "total_return": float(result["total_return"]),
            "max_drawdown": float(result["max_drawdown"]),
            "sharpe": float(result["sharpe"]),
            "daily_target_hit_rate": float(result["daily_metrics"]["daily_target_hit_rate"]),
            "daily_win_rate": float(result["daily_metrics"]["daily_win_rate"]),
            "worst_day": float(result["daily_metrics"]["worst_day"]),
            "best_day": float(result["daily_metrics"]["best_day"]),
            "n_trades": int(result["n_trades"]),
            "n_wins": n_wins_v,
            "n_losses": n_losses_v,
            "roundtrip_win_rate": float(roundtrip_winrate),
            "open_position_at_end": bool(int(result.get("open_position_at_end", 0))),
            "open_roundtrip_realized_pnl": float(result.get("open_roundtrip_realized_pnl", 0.0)),
            "open_roundtrip_fees_paid": float(result.get("open_roundtrip_fees_paid", 0.0)),
            "fee_paid": float(result.get("fee_paid", 0.0)),
            "slippage_paid": float(result.get("slippage_paid", 0.0)),
            "funding_paid": float(result.get("funding_paid", 0.0)),
            "funding_events": int(result.get("funding_events", 0)),
            "funding_unmatched": funding_unmatched,
            "final_equity": float(result["final_equity"]),
            "avg_win_size": float(avg_win_size_v),
            "avg_loss_size": float(avg_loss_size_v),
            "total_win_pnl": float(total_win_pnl_v),
            "total_loss_pnl": float(total_loss_pnl_v),
            "payoff_ratio": float(payoff_ratio_v),
            "kelly_fraction": float(kelly_v),
            "regime_n_wins": regime_n_wins_v,
            "regime_n_losses": regime_n_losses_v,
            "regime_win_rate": regime_win_rate_v,
        }
        if return_trace:
            metrics["trace"] = result.get("trace") or {}
        return metrics
    n_wins_v = int(result[14]) if len(result) > 14 else 0
    n_losses_v = int(result[15]) if len(result) > 15 else 0
    open_at_end_v = int(result[16]) if len(result) > 16 else 0
    open_realized_pnl_v = float(result[17]) if len(result) > 17 else 0.0
    open_fees_v = float(result[18]) if len(result) > 18 else 0.0
    total_win_pnl_v = float(result[19]) if len(result) > 19 else 0.0
    total_loss_pnl_v = float(result[20]) if len(result) > 20 else 0.0
    regime_n_wins_v = list(result[21]) if len(result) > 21 else [0] * MAX_REGIME_BUCKETS
    regime_n_losses_v = list(result[22]) if len(result) > 22 else [0] * MAX_REGIME_BUCKETS
    decided = n_wins_v + n_losses_v
    roundtrip_winrate = (n_wins_v / decided) if decided > 0 else 0.0
    avg_win_size_v = total_win_pnl_v / n_wins_v if n_wins_v > 0 else 0.0
    avg_loss_size_v = total_loss_pnl_v / n_losses_v if n_losses_v > 0 else 0.0
    payoff_ratio_v = avg_win_size_v / avg_loss_size_v if avg_loss_size_v > 0.0 else 0.0
    if decided > 0 and avg_loss_size_v > 0.0:
        win_rate_v = n_wins_v / decided
        kelly_v = max(-1.0, min(1.0, win_rate_v - (1.0 - win_rate_v) / (payoff_ratio_v if payoff_ratio_v > 0.0 else 1e-12)))
    else:
        kelly_v = 0.0
    regime_win_rate_v = [
        float(regime_n_wins_v[i]) / float(regime_n_wins_v[i] + regime_n_losses_v[i])
        if (regime_n_wins_v[i] + regime_n_losses_v[i]) > 0 else 0.0
        for i in range(len(regime_n_wins_v))
    ]
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
        "n_wins": n_wins_v,
        "n_losses": n_losses_v,
        "roundtrip_win_rate": float(roundtrip_winrate),
        "open_position_at_end": bool(open_at_end_v),
        "open_roundtrip_realized_pnl": float(open_realized_pnl_v),
        "open_roundtrip_fees_paid": float(open_fees_v),
        "fee_paid": float(result[10]),
        "slippage_paid": float(result[11]),
        "funding_paid": float(result[12]),
        "funding_events": int(result[13]),
        "funding_unmatched": funding_unmatched,
        "final_equity": float(result[4]),
        "avg_win_size": float(avg_win_size_v),
        "avg_loss_size": float(avg_loss_size_v),
        "total_win_pnl": float(total_win_pnl_v),
        "total_loss_pnl": float(total_loss_pnl_v),
        "payoff_ratio": float(payoff_ratio_v),
        "kelly_fraction": float(kelly_v),
        "regime_n_wins": regime_n_wins_v,
        "regime_n_losses": regime_n_losses_v,
        "regime_win_rate": regime_win_rate_v,
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
    execution_gene: dict[str, Any] | None = None,
    state_specialists: tuple[int, ...] | list[int] | None = None,
    initial_cooldown_bars: int = 0,
    min_notional_usd: float = 25.0,
    max_hold_bars: int = 288,
    runtime_gross_cap: float | None = None,
    final_decision_cooldown_override: int | None = None,
    return_trace: bool = False,
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
        execution_gene=execution_gene,
        state_specialists=state_specialists,
        initial_cooldown_bars=initial_cooldown_bars,
        min_notional_usd=min_notional_usd,
        max_hold_bars=max_hold_bars,
        runtime_gross_cap=runtime_gross_cap,
        final_decision_cooldown_override=final_decision_cooldown_override,
        return_trace=return_trace,
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
    recent_2m = report["windows"]["recent_2m"]["aggregate"]
    recent_4m = report["windows"]["recent_4m"]["aggregate"]
    recent_6m = report["windows"]["recent_6m"]["aggregate"]
    full_4y = report["windows"]["full_4y"]["aggregate"]

    score = 0.0
    score += float(recent_2m["worst_pair_avg_daily_return"]) * 240000.0
    score += float(recent_2m["mean_avg_daily_return"]) * 60000.0
    score += float(recent_6m["worst_pair_avg_daily_return"]) * 380000.0
    score += float(full_4y["worst_pair_avg_daily_return"]) * 280000.0
    score += float(full_4y["mean_avg_daily_return"]) * 180000.0
    score += float(recent_6m["mean_avg_daily_return"]) * 50000.0
    score -= abs(float(recent_2m["worst_max_drawdown"])) * 15000.0
    score -= abs(float(recent_6m["worst_max_drawdown"])) * 18000.0
    score -= abs(float(full_4y["worst_max_drawdown"])) * 9000.0
    score -= float(recent_2m["pair_return_dispersion"]) * 100000.0
    score -= float(recent_6m["pair_return_dispersion"]) * 120000.0
    score -= float(full_4y["pair_return_dispersion"]) * 60000.0
    recent_2m_trades = float(recent_2m.get("mean_n_trades", 0.0))
    recent_4m_trades = float(recent_4m.get("mean_n_trades", recent_2m_trades))
    recent_6m_trades = float(recent_6m.get("mean_n_trades", 0.0))
    full_4y_trades = float(full_4y.get("mean_n_trades", 0.0))
    score += min(recent_2m_trades, 48.0) * 180.0
    score += min(recent_4m_trades, 96.0) * 90.0
    score += min(recent_6m_trades, 192.0) * 35.0
    score += min(full_4y_trades, 1440.0) * 1.5
    score -= max(0.0, 12.0 - recent_2m_trades) * 900.0
    score -= max(0.0, 24.0 - recent_4m_trades) * 420.0
    score -= max(0.0, 36.0 - recent_6m_trades) * 220.0
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
    derivatives_all = {pair: _load_derivative_bundle(pair) for pair in pairs}

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
                    derivative_bundle=derivatives_all[pair],
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
        "model_path": str(args.model),
        "baseline_summary_path": str(args.summary),
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
