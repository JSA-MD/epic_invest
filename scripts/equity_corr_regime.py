#!/usr/bin/env python3
"""Shared BTC-equity correlation regime helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core_market_profile import build_corr_state_profiles
from market_context import load_market_context_dataset


DEFAULT_CONTEXT_PREFERENCE: tuple[str, ...] = ("QQQ", "SPY")
EXTENDED_CONTEXT_NAMES: tuple[str, ...] = ("QQQ", "SPY", "GLD", "DXY")
DEFAULT_CORR_WINDOW = 20
FAST_CORR_WINDOW = 5
POSITIVE_CORR_THRESHOLD = 0.15
NEGATIVE_CORR_THRESHOLD = -0.15

ALIGNED_GROSS_SCALE = 1.00
MIXED_GROSS_SCALE = 1.00
INVERSE_GROSS_SCALE = 0.80

ALIGNED_REGIME_THRESHOLD_MULT = 1.00
MIXED_REGIME_THRESHOLD_MULT = 1.00
INVERSE_REGIME_THRESHOLD_MULT = 1.10


def resolve_corr_bucket(corr_value: float | None) -> str:
    if corr_value is None or not np.isfinite(corr_value):
        return "equity_unknown"
    if corr_value >= POSITIVE_CORR_THRESHOLD:
        return "equity_aligned"
    if corr_value <= NEGATIVE_CORR_THRESHOLD:
        return "equity_inverse"
    return "equity_mixed"


def resolve_corr_risk_scales(corr_value: float | None) -> tuple[float, float, str]:
    bucket = resolve_corr_bucket(corr_value)
    if bucket == "equity_aligned":
        return ALIGNED_GROSS_SCALE, ALIGNED_REGIME_THRESHOLD_MULT, bucket
    if bucket == "equity_inverse":
        return INVERSE_GROSS_SCALE, INVERSE_REGIME_THRESHOLD_MULT, bucket
    if bucket == "equity_mixed":
        return MIXED_GROSS_SCALE, MIXED_REGIME_THRESHOLD_MULT, bucket
    return 1.0, 1.0, bucket


def build_btc_equity_corr_overlay(
    close: pd.DataFrame,
    *,
    btc_asset: str = "BTCUSDT",
    context_preference: tuple[str, ...] = DEFAULT_CONTEXT_PREFERENCE,
    corr_window: int = DEFAULT_CORR_WINDOW,
    min_context_days: int = 20,
) -> dict[str, Any]:
    daily_close = close.resample("1D").last().dropna(how="all")
    if btc_asset not in daily_close.columns or daily_close.empty:
        empty_index = daily_close.index
        return {
            "equity_corr_daily": pd.Series(index=empty_index, dtype="float64"),
            "equity_corr_quantile_state_daily": pd.Series(index=empty_index, dtype="object"),
            "equity_corr_bucket_daily": pd.Series(index=empty_index, dtype="object"),
            "equity_corr_gross_scale_daily": pd.Series(index=empty_index, dtype="float64"),
            "equity_corr_regime_threshold_mult_daily": pd.Series(index=empty_index, dtype="float64"),
            "equity_corr_context": None,
            "equity_corr_source_mode": "missing",
            "equity_corr_market_context_status": {"status": "missing", "usable_columns": []},
            "equity_corr_low_cut": None,
            "equity_corr_high_cut": None,
        }

    requested_context_names = tuple(dict.fromkeys((*context_preference, *EXTENDED_CONTEXT_NAMES)))
    market_context_close, market_context_status = load_market_context_dataset(
        names=requested_context_names,
        refresh=False,
        allow_fetch_on_miss=False,
        target_index=daily_close.index,
        min_context_days=max(int(min_context_days), int(corr_window)),
    )
    corr_profiles = build_corr_state_profiles(
        daily_close,
        market_context_close,
        corr_window=int(corr_window),
    )
    profiles = corr_profiles.get("profiles") or {}
    chosen_context = next((name for name in context_preference if name in profiles), None)
    if chosen_context is None and profiles:
        chosen_context = next(iter(profiles.keys()))

    if chosen_context is None:
        index = daily_close.index
        unknown_bucket = pd.Series("equity_unknown", index=index, dtype="object")
        ones = pd.Series(1.0, index=index, dtype="float64")
        return {
            "equity_corr_daily": pd.Series(index=index, dtype="float64"),
            "equity_corr_quantile_state_daily": pd.Series("missing", index=index, dtype="object"),
            "equity_corr_bucket_daily": unknown_bucket,
            "equity_corr_gross_scale_daily": ones,
            "equity_corr_regime_threshold_mult_daily": ones,
            "equity_corr_context": None,
            "equity_corr_source_mode": str(corr_profiles.get("source_mode", "missing")),
            "equity_corr_market_context_status": market_context_status,
            "equity_corr_low_cut": None,
            "equity_corr_high_cut": None,
        }

    profile = profiles[chosen_context]
    rolling_corr = profile["rolling_corr"].reindex(daily_close.index)
    quantile_state = profile["labels"].reindex(daily_close.index).fillna("missing").astype("object")

    bucket_values: list[str] = []
    gross_values: list[float] = []
    regime_mult_values: list[float] = []
    for raw in rolling_corr.to_numpy(dtype="float64"):
        corr_value = float(raw) if np.isfinite(raw) else None
        gross_scale, regime_mult, bucket = resolve_corr_risk_scales(corr_value)
        bucket_values.append(bucket)
        gross_values.append(float(gross_scale))
        regime_mult_values.append(float(regime_mult))

    index = daily_close.index
    btc_returns = daily_close[btc_asset].pct_change()
    qqq_returns = market_context_close["QQQ"].pct_change() if "QQQ" in market_context_close.columns else pd.Series(index=index, dtype="float64")
    spy_returns = market_context_close["SPY"].pct_change() if "SPY" in market_context_close.columns else pd.Series(index=index, dtype="float64")
    gld_returns = market_context_close["GLD"].pct_change() if "GLD" in market_context_close.columns else pd.Series(index=index, dtype="float64")
    dxy_returns = market_context_close["DXY"].pct_change() if "DXY" in market_context_close.columns else pd.Series(index=index, dtype="float64")
    qqq_corr_5d = btc_returns.rolling(FAST_CORR_WINDOW).corr(qqq_returns).reindex(index).astype("float64")
    qqq_corr_20d = btc_returns.rolling(corr_window).corr(qqq_returns).reindex(index).astype("float64")
    dxy_corr_20d = btc_returns.rolling(corr_window).corr(dxy_returns).reindex(index).astype("float64")
    gold_corr_20d = btc_returns.rolling(corr_window).corr(gld_returns).reindex(index).astype("float64")
    spy_var_20d = spy_returns.rolling(corr_window).var()
    btc_spy_cov_20d = btc_returns.rolling(corr_window).cov(spy_returns)
    spy_beta_20d = (
        btc_spy_cov_20d / spy_var_20d.replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).reindex(index).astype("float64")
    return {
        "equity_corr_daily": rolling_corr.astype("float64"),
        "equity_corr_quantile_state_daily": quantile_state,
        "equity_corr_bucket_daily": pd.Series(bucket_values, index=index, dtype="object"),
        "equity_corr_gross_scale_daily": pd.Series(gross_values, index=index, dtype="float64"),
        "equity_corr_regime_threshold_mult_daily": pd.Series(regime_mult_values, index=index, dtype="float64"),
        "btc_qqq_corr_5d_daily": qqq_corr_5d,
        "btc_qqq_corr_20d_daily": qqq_corr_20d,
        "btc_spy_beta_20d_daily": spy_beta_20d,
        "btc_dxy_corr_20d_daily": dxy_corr_20d,
        "btc_gold_corr_20d_daily": gold_corr_20d,
        "equity_corr_context": str(chosen_context),
        "equity_corr_source_mode": str(corr_profiles.get("source_mode", "missing")),
        "equity_corr_market_context_status": market_context_status,
        "equity_corr_low_cut": float(profile["low_cut"]),
        "equity_corr_high_cut": float(profile["high_cut"]),
    }
