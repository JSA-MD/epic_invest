#!/usr/bin/env python3
"""Shared market-state representation for core search and live diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DAY_FACTOR = np.sqrt(365.25)
ROUTE_BUCKET_LABELS: dict[int, str] = {
    0: "bear_narrow",
    1: "bear_broad",
    2: "bull_narrow",
    3: "bull_broad",
}


def compute_cross_sectional_momentum(
    close: pd.DataFrame,
    *,
    fast_lookback: int,
    slow_lookback: int,
    fast_weight: float = 0.60,
) -> pd.DataFrame:
    fast = close.pct_change(int(fast_lookback))
    slow = close.pct_change(int(slow_lookback))
    slow_weight = 1.0 - float(fast_weight)
    return float(fast_weight) * fast + slow_weight * slow


def compute_regime_score(
    close: pd.DataFrame,
    *,
    asset: str = "BTCUSDT",
    fast_lookback: int,
    slow_lookback: int,
    fast_weight: float = 0.50,
) -> pd.Series:
    fast = close[asset].pct_change(int(fast_lookback))
    slow = close[asset].pct_change(int(slow_lookback))
    slow_weight = 1.0 - float(fast_weight)
    return float(fast_weight) * fast + slow_weight * slow


def compute_breadth(momentum: pd.DataFrame) -> pd.Series:
    return (momentum > 0.0).mean(axis=1)


def compute_realized_vol(close: pd.DataFrame, *, vol_window: int) -> pd.DataFrame:
    return close.pct_change().rolling(int(vol_window)).std() * DAY_FACTOR


def classify_regime_labels(regime_score: pd.Series, *, threshold: float) -> pd.Series:
    labels = pd.Series("sideways", index=regime_score.index, dtype="object")
    labels.loc[regime_score >= float(threshold)] = "bull"
    labels.loc[regime_score <= -float(threshold)] = "bear"
    return labels


def build_route_bucket_series(
    regime_score: pd.Series,
    breadth: pd.Series,
    *,
    breadth_threshold: float,
) -> pd.Series:
    regime_binary = (regime_score >= 0.0).astype("int8")
    breadth_binary = (breadth >= float(breadth_threshold)).astype("int8")
    codes = regime_binary * 2 + breadth_binary
    return pd.Series(
        [ROUTE_BUCKET_LABELS.get(int(code), "unknown") for code in codes.to_numpy(dtype="int8")],
        index=regime_score.index,
        dtype="object",
    )


def resolve_market_context_frame(
    close: pd.DataFrame,
    market_context_close: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    context_frame = market_context_close.copy() if not market_context_close.empty else pd.DataFrame()
    if not context_frame.empty:
        return context_frame, "market_context"

    alt_cols = [col for col in close.columns if col != "BTCUSDT"]
    if alt_cols:
        return pd.DataFrame({"internal_alt_basket": close[alt_cols].mean(axis=1)}), "internal_fallback"
    return pd.DataFrame(), "missing"


def build_corr_state_profiles(
    close: pd.DataFrame,
    market_context_close: pd.DataFrame,
    *,
    corr_window: int,
) -> dict[str, Any]:
    btc_ret = close["BTCUSDT"].pct_change()
    context_frame, source_mode = resolve_market_context_frame(close, market_context_close)
    profiles: dict[str, Any] = {}
    if context_frame.empty:
        return {"source_mode": source_mode, "profiles": profiles}

    for context_name in context_frame.columns:
        context_ret = context_frame[context_name].pct_change()
        rolling_corr = btc_ret.rolling(int(corr_window)).corr(context_ret).replace([np.inf, -np.inf], np.nan)
        valid_corr = rolling_corr.dropna()
        if valid_corr.empty:
            profiles[context_name] = {
                "rolling_corr": rolling_corr,
                "labels": pd.Series(dtype="object"),
                "low_cut": 0.0,
                "high_cut": 0.0,
            }
            continue

        low_cut = float(valid_corr.quantile(0.33))
        high_cut = float(valid_corr.quantile(0.67))
        labels = pd.Series("mid_corr", index=rolling_corr.index, dtype="object")
        labels.loc[rolling_corr <= low_cut] = "low_corr"
        labels.loc[rolling_corr >= high_cut] = "high_corr"
        profiles[context_name] = {
            "rolling_corr": rolling_corr,
            "labels": labels,
            "low_cut": low_cut,
            "high_cut": high_cut,
        }
    return {"source_mode": source_mode, "profiles": profiles}


def build_core_market_profile(
    close: pd.DataFrame,
    market_context_close: pd.DataFrame,
    *,
    fast_lookback: int,
    slow_lookback: int,
    vol_window: int,
    corr_window: int,
    regime_threshold: float,
    breadth_threshold: float,
) -> dict[str, Any]:
    momentum = compute_cross_sectional_momentum(
        close,
        fast_lookback=fast_lookback,
        slow_lookback=slow_lookback,
    )
    breadth = compute_breadth(momentum)
    realized_vol = compute_realized_vol(close, vol_window=vol_window)
    regime_score = compute_regime_score(
        close,
        fast_lookback=fast_lookback,
        slow_lookback=slow_lookback,
    )
    regime_labels = classify_regime_labels(regime_score, threshold=regime_threshold)
    route_buckets = build_route_bucket_series(
        regime_score,
        breadth,
        breadth_threshold=breadth_threshold,
    )
    corr_profiles = build_corr_state_profiles(
        close,
        market_context_close,
        corr_window=corr_window,
    )
    feature_frame = pd.DataFrame(
        {
            "btc_regime": regime_score,
            "breadth": breadth,
            "regime_label": regime_labels,
            "route_bucket": route_buckets,
        },
        index=close.index,
    )
    return {
        "feature_frame": feature_frame,
        "momentum": momentum,
        "breadth": breadth,
        "realized_vol": realized_vol,
        "regime_score": regime_score,
        "regime_labels": regime_labels,
        "route_buckets": route_buckets,
        "corr_state_profiles": corr_profiles,
    }


def build_route_state_snapshot(
    feature_frame: pd.DataFrame,
    market_day: pd.Timestamp,
) -> dict[str, Any]:
    if market_day not in feature_frame.index:
        raise KeyError(f"Missing market profile for {market_day}")
    row = feature_frame.loc[market_day]
    bucket = str(row["route_bucket"])
    bucket_code = next((code for code, label in ROUTE_BUCKET_LABELS.items() if label == bucket), None)
    return {
        "market_day": market_day.date().isoformat(),
        "btc_regime": float(row["btc_regime"]),
        "breadth": float(row["breadth"]),
        "regime_label": str(row["regime_label"]),
        "route_bucket": bucket,
        "route_bucket_code": int(bucket_code) if bucket_code is not None else None,
    }


def build_context_corr_snapshot(
    corr_state_profiles: dict[str, Any],
    market_day: pd.Timestamp,
) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for context_name, profile in (corr_state_profiles.get("profiles") or {}).items():
        rolling_corr = profile["rolling_corr"]
        labels = profile["labels"]
        corr_value = rolling_corr.get(market_day) if market_day in rolling_corr.index else np.nan
        label_value = labels.get(market_day) if market_day in labels.index else None
        snapshots[context_name] = {
            "corr": float(corr_value) if pd.notna(corr_value) else None,
            "state": str(label_value) if label_value is not None else None,
            "low_cut": float(profile["low_cut"]),
            "high_cut": float(profile["high_cut"]),
        }
    return {
        "source_mode": str(corr_state_profiles.get("source_mode", "missing")),
        "contexts": snapshots,
    }
