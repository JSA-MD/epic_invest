#!/usr/bin/env python3
"""First-stage fractal genome search over recursive If-Then-Else trees."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from fractal_genome_core import (
    AndCell,
    ConditionNode,
    ConditionSpec,
    FilterDecision,
    LeafGene,
    LeafNode,
    NotCell,
    OrCell,
    ThresholdCell,
    TreeNode,
    build_llm_prompt,
    collect_leaves,
    collect_leaf_keys,
    collect_specs,
    crossover_tree,
    evaluate_tree_codes,
    evaluate_tree_leaf_codes,
    load_llm_review_map,
    mutate_tree,
    random_tree,
    semantic_filter,
    serialize_tree,
    tree_depth,
    tree_key,
    tree_logic_depth,
    tree_logic_size,
    tree_size,
)
from replay_regime_mixture_realistic import load_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    BAR_FACTOR,
    BARS_PER_DAY,
    NUMBA_AVAILABLE,
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    json_safe,
    parse_csv_tuple,
    resolve_fast_engine,
    score_realistic_candidate,
    summarize_single_result,
)
from validate_pair_subset_summary import build_validation_bundle

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


UTC = timezone.utc

FEATURE_SET_NAME = "fractal_market_feature_v3"
FEATURE_SET_DESCRIPTION = (
    "Expanded single-asset and pairwise inputs covering returns, momentum, "
    "RSI, ATR, MACD, Bollinger, MFI, CCI, Donchian, drawdown, volatility, "
    "volume, session, and cross-asset spread features."
)
WINDOW_COMPAT_LABEL_FULL = "full_4y"

BASE_FEATURE_SPECS: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    ("btc_regime", ">=", (-0.05, 0.0, 0.05, 0.10)),
    ("breadth", ">=", (0.35, 0.50, 0.65, 0.80)),
    ("btc_vol_rel", "<=", (0.80, 1.00, 1.20)),
    ("btc_return_1d", ">=", (-0.03, -0.015, 0.0, 0.015, 0.03)),
    ("btc_return_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("btc_return_7d", ">=", (-0.10, -0.05, 0.0, 0.05, 0.10)),
    ("btc_momentum_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("btc_momentum_1d", ">=", (-0.03, -0.015, 0.0, 0.015, 0.03)),
    ("btc_momentum_accel_1d_3d", ">=", (-0.04, -0.02, 0.0, 0.02, 0.04)),
    ("btc_drawdown_7d", "<=", (-0.20, -0.15, -0.10, -0.05, 0.0)),
    ("btc_drawdown_21d", "<=", (-0.30, -0.20, -0.15, -0.10, -0.05, 0.0)),
    ("btc_rsi_14d", ">=", (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
    ("btc_atr_pct_14d", "<=", (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)),
    ("btc_volatility_7d", "<=", (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)),
    ("btc_volatility_21d", "<=", (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)),
    ("btc_volume_z_7d", ">=", (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)),
    ("btc_macd_line_12_26_9", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("btc_macd_hist_12_26_9", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("btc_bb_pct_b_20_2", ">=", (0.10, 0.25, 0.50, 0.75, 0.90)),
    ("btc_bb_width_20_2", "<=", (0.02, 0.04, 0.06, 0.08, 0.12)),
    ("btc_mfi_14d", ">=", (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
    ("btc_cci_20d", ">=", (-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0)),
    ("btc_dc_pos_20d", ">=", (0.10, 0.25, 0.50, 0.75, 0.90)),
    ("btc_intraday_return_1h", ">=", (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)),
    ("btc_intraday_return_6h", ">=", (-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04)),
    ("btc_intraday_drawdown_24h", "<=", (-0.12, -0.08, -0.05, -0.03, -0.01, 0.0)),
    ("btc_rsi_14_1h", ">=", (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
    ("btc_mfi_14_1h", ">=", (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
    ("btc_atr_pct_1h", "<=", (0.0005, 0.0010, 0.0020, 0.0030, 0.0040, 0.0060)),
    ("btc_macd_h_pct_1h", ">=", (-0.0010, -0.0005, -0.0002, 0.0, 0.0002, 0.0005, 0.0010)),
    ("btc_bb_p_1h", ">=", (0.20, 0.35, 0.50, 0.65, 0.80)),
    ("btc_volume_rel_1h", ">=", (0.40, 0.70, 1.00, 1.30, 1.80, 2.40)),
    ("btc_cci_scaled_1h", ">=", (-1.50, -0.75, -0.25, 0.0, 0.25, 0.75, 1.50)),
    ("btc_dc_trend_05_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_event_05_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_overshoot_05_1h", ">=", (-0.015, -0.005, 0.0, 0.005, 0.015)),
    ("btc_dc_run_05_1h", ">=", (0.0, 0.01, 0.02, 0.05)),
    ("session_utc_phase", ">=", (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)),
    ("session_asia_flag", ">=", (0.50,)),
    ("session_eu_flag", ">=", (0.50,)),
    ("session_us_flag", ">=", (0.50,)),
)

MULTI_PAIR_FEATURE_SPECS: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    ("bnb_regime", ">=", (-0.05, 0.0, 0.05, 0.10)),
    ("bnb_vol_rel", "<=", (0.80, 1.00, 1.20)),
    ("bnb_return_1d", ">=", (-0.03, -0.015, 0.0, 0.015, 0.03)),
    ("bnb_return_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("bnb_return_7d", ">=", (-0.10, -0.05, 0.0, 0.05, 0.10)),
    ("bnb_momentum_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("rel_strength_bnb_btc_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("bnb_momentum_1d", ">=", (-0.03, -0.015, 0.0, 0.015, 0.03)),
    ("bnb_momentum_accel_1d_3d", ">=", (-0.04, -0.02, 0.0, 0.02, 0.04)),
    ("regime_spread_btc_minus_bnb", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("breadth_change_1d", ">=", (-0.25, -0.10, 0.0, 0.10, 0.25)),
    ("bnb_drawdown_7d", "<=", (-0.20, -0.15, -0.10, -0.05, 0.0)),
    ("bnb_drawdown_21d", "<=", (-0.30, -0.20, -0.15, -0.10, -0.05, 0.0)),
    ("vol_rel_spread_btc_minus_bnb", ">=", (-0.25, 0.0, 0.25)),
    ("rel_strength_bnb_btc_1d", ">=", (-0.03, -0.015, 0.0, 0.015, 0.03)),
    ("bnb_rsi_14d", ">=", (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
    ("bnb_atr_pct_14d", "<=", (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)),
    ("bnb_volatility_7d", "<=", (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)),
    ("bnb_volatility_21d", "<=", (0.01, 0.015, 0.02, 0.03, 0.04, 0.05)),
    ("bnb_volume_z_7d", ">=", (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)),
    ("bnb_macd_line_12_26_9", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("bnb_macd_hist_12_26_9", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("bnb_bb_pct_b_20_2", ">=", (0.10, 0.25, 0.50, 0.75, 0.90)),
    ("bnb_bb_width_20_2", "<=", (0.02, 0.04, 0.06, 0.08, 0.12)),
    ("bnb_mfi_14d", ">=", (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
    ("bnb_cci_20d", ">=", (-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0)),
    ("bnb_dc_pos_20d", ">=", (0.10, 0.25, 0.50, 0.75, 0.90)),
    ("bnb_intraday_return_1h", ">=", (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)),
    ("bnb_intraday_return_6h", ">=", (-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04)),
    ("bnb_intraday_drawdown_24h", "<=", (-0.12, -0.08, -0.05, -0.03, -0.01, 0.0)),
    ("bnb_rsi_14_1h", ">=", (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
    ("bnb_mfi_14_1h", ">=", (20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)),
    ("bnb_atr_pct_1h", "<=", (0.0005, 0.0010, 0.0020, 0.0030, 0.0040, 0.0060)),
    ("bnb_macd_h_pct_1h", ">=", (-0.0010, -0.0005, -0.0002, 0.0, 0.0002, 0.0005, 0.0010)),
    ("bnb_bb_p_1h", ">=", (0.20, 0.35, 0.50, 0.65, 0.80)),
    ("bnb_volume_rel_1h", ">=", (0.40, 0.70, 1.00, 1.30, 1.80, 2.40)),
    ("bnb_cci_scaled_1h", ">=", (-1.50, -0.75, -0.25, 0.0, 0.25, 0.75, 1.50)),
    ("bnb_dc_trend_05_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_event_05_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_overshoot_05_1h", ">=", (-0.015, -0.005, 0.0, 0.005, 0.015)),
    ("bnb_dc_run_05_1h", ">=", (0.0, 0.01, 0.02, 0.05)),
    ("rsi_spread_btc_minus_bnb_14d", ">=", (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)),
    ("atr_spread_btc_minus_bnb_14d", ">=", (-0.02, -0.01, 0.0, 0.01, 0.02)),
    ("macd_hist_spread_btc_minus_bnb_12_26_9", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("bb_pct_b_spread_btc_minus_bnb_20_2", ">=", (-0.40, -0.20, 0.0, 0.20, 0.40)),
    ("mfi_spread_btc_minus_bnb_14d", ">=", (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)),
    ("cci_spread_btc_minus_bnb_20d", ">=", (-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0)),
    ("dc_pos_spread_btc_minus_bnb_20d", ">=", (-0.40, -0.20, 0.0, 0.20, 0.40)),
    ("volume_z_spread_btc_minus_bnb_7d", ">=", (-2.0, -1.0, 0.0, 1.0, 2.0)),
    ("return_spread_btc_minus_bnb_1d", ">=", (-0.03, -0.015, 0.0, 0.015, 0.03)),
    ("return_spread_btc_minus_bnb_3d", ">=", (-0.05, -0.02, 0.0, 0.02, 0.05)),
    ("return_spread_btc_minus_bnb_7d", ">=", (-0.10, -0.05, 0.0, 0.05, 0.10)),
    ("rsi_spread_btc_minus_bnb_1h", ">=", (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)),
    ("mfi_spread_btc_minus_bnb_1h", ">=", (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)),
    ("atr_pct_spread_btc_minus_bnb_1h", ">=", (-0.004, -0.002, -0.001, 0.0, 0.001, 0.002, 0.004)),
    ("macd_h_pct_spread_btc_minus_bnb_1h", ">=", (-0.0015, -0.0007, -0.0002, 0.0, 0.0002, 0.0007, 0.0015)),
    ("volume_rel_spread_btc_minus_bnb_1h", ">=", (-1.50, -0.50, 0.0, 0.50, 1.50)),
    ("cci_scaled_spread_btc_minus_bnb_1h", ">=", (-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0)),
    ("dc_trend_spread_btc_minus_bnb_1h", ">=", (-2.0, -1.0, 0.0, 1.0, 2.0)),
    ("dc_event_spread_btc_minus_bnb_1h", ">=", (-1.0, 0.0, 1.0)),
    ("dc_overshoot_spread_btc_minus_bnb_1h", ">=", (-0.020, -0.010, 0.0, 0.010, 0.020)),
    ("dc_run_spread_btc_minus_bnb_1h", ">=", (-0.10, -0.05, 0.0, 0.05, 0.10)),
    ("intraday_return_spread_btc_minus_bnb_1h", ">=", (-0.03, -0.015, 0.0, 0.015, 0.03)),
    ("intraday_return_spread_btc_minus_bnb_6h", ">=", (-0.06, -0.03, 0.0, 0.03, 0.06)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search recursive fractal-genome trees that route BTC/BNB expert overlays.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--expert-summaries",
        default=(
            "models/gp_regime_mixture_btc_bnb_pairwise_repair_summary.json,"
            "models/gp_regime_mixture_btc_bnb_pairwise_fullgrid_seed_pool.json,"
            "models/gp_regime_mixture_btc_bnb_pairwise_stress_report.json"
        ),
    )
    parser.add_argument(
        "--baseline-summary",
        default="models/gp_regime_mixture_btc_bnb_pairwise_repair_summary.json",
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default="models/gp_regime_mixture_btc_bnb_fractal_genome_summary.json",
    )
    parser.add_argument("--expert-pool-size", type=int, default=18)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--logic-max-depth", type=int, default=2)
    parser.add_argument("--curriculum-min-depth", type=int, default=1)
    parser.add_argument("--curriculum-min-logic-depth", type=int, default=1)
    parser.add_argument(
        "--survivor-diversity-weight",
        type=float,
        default=0.40,
        help="Weight for structural diversity when selecting generation survivors.",
    )
    parser.add_argument(
        "--survivor-depth-weight",
        type=float,
        default=0.55,
        help="Weight for depth/logic-depth coverage when selecting generation survivors.",
    )
    parser.add_argument(
        "--immigrant-rate",
        type=float,
        default=0.12,
        help="Fraction of each generation reserved for fresh immigrant trees.",
    )
    parser.add_argument(
        "--robustness-folds",
        type=int,
        default=3,
        help="Number of walk-forward folds evaluated during the main search.",
    )
    parser.add_argument(
        "--robustness-test-months",
        type=int,
        default=2,
        help="Months per walk-forward fold evaluated during the main search.",
    )
    parser.add_argument(
        "--commission-stress",
        default="1.0,1.5,2.0",
        help="Commission multipliers used to score stress robustness during search.",
    )
    parser.add_argument(
        "--stress-survival-threshold",
        type=float,
        default=0.67,
        help="Minimum average stress survival rate expected from robustness folds.",
    )
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument(
        "--route-thresholds",
        default="0.35,0.50,0.65,0.80",
    )
    parser.add_argument(
        "--fast-engine",
        choices=("auto", "python", "numba"),
        default="auto",
    )
    parser.add_argument(
        "--filter-mode",
        choices=("auto", "heuristic", "llm-first", "llm-only"),
        default="auto",
    )
    parser.add_argument(
        "--llm-review-in",
        default=None,
        help="Optional JSONL file with precomputed LLM decisions keyed by tree_key.",
    )
    parser.add_argument(
        "--llm-review-out",
        default=None,
        help="Optional JSONL file to export review prompts for top candidates.",
    )
    parser.add_argument(
        "--auto-llm-review-top-n",
        type=int,
        default=0,
        help="Automatically review the top-N candidates per generation when OPENAI_API_KEY is set.",
    )
    parser.add_argument(
        "--auto-llm-review-model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model used for optional automatic candidate review.",
    )
    parser.add_argument(
        "--auto-llm-review-timeout-seconds",
        type=float,
        default=20.0,
        help="Timeout for each optional automatic LLM review request.",
    )
    return parser.parse_args()


def build_feature_specs(pairs: tuple[str, ...]) -> tuple[tuple[str, str, tuple[float, ...]], ...]:
    if len(pairs) <= 1:
        return BASE_FEATURE_SPECS
    return BASE_FEATURE_SPECS + MULTI_PAIR_FEATURE_SPECS


def build_condition_options(feature_specs: tuple[tuple[str, str, tuple[float, ...]], ...]) -> list[ConditionSpec]:
    out: list[ConditionSpec] = []
    for feature, comparator, thresholds in feature_specs:
        for threshold in thresholds:
            out.append(ConditionSpec(feature=feature, comparator=comparator, threshold=float(threshold), invert=False))
            out.append(ConditionSpec(feature=feature, comparator=comparator, threshold=float(threshold), invert=True))
    return out


def describe_feature_set(feature_specs: tuple[tuple[str, str, tuple[float, ...]], ...]) -> dict[str, Any]:
    return {
        "name": FEATURE_SET_NAME,
        "description": FEATURE_SET_DESCRIPTION,
        "features": [
            {
                "feature": feature,
                "comparator": comparator,
                "thresholds": [float(threshold) for threshold in thresholds],
            }
            for feature, comparator, thresholds in feature_specs
        ],
    }


def _daily_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0)).abs()
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.replace([np.inf, -np.inf], np.nan).fillna(50.0)


def _daily_atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return (atr / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_zscore(series: pd.Series, period: int = 7) -> pd.Series:
    mean = series.rolling(period, min_periods=max(3, period // 2)).mean()
    std = series.rolling(period, min_periods=max(3, period // 2)).std()
    return ((series - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=max(2, span // 2)).mean()


def _macd_features(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return (
        macd_line.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        macd_signal.replace([np.inf, -np.inf], np.nan).fillna(0.0),
        macd_hist.replace([np.inf, -np.inf], np.nan).fillna(0.0),
    )


def _bollinger_features(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(period, min_periods=max(5, period // 2)).mean()
    std = close.rolling(period, min_periods=max(5, period // 2)).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return (
        mid.replace([np.inf, -np.inf], np.nan).ffill().fillna(close),
        upper.replace([np.inf, -np.inf], np.nan).fillna(close),
        lower.replace([np.inf, -np.inf], np.nan).fillna(close),
        pct_b.replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(-1.0, 2.0),
        width.replace([np.inf, -np.inf], np.nan).fillna(0.0),
    )


def _money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    typical = (high + low + close) / 3.0
    money_flow = typical * volume.fillna(0.0)
    direction = typical.diff()
    positive = money_flow.where(direction >= 0.0, 0.0)
    negative = money_flow.where(direction < 0.0, 0.0).abs()
    pos_sum = positive.rolling(period, min_periods=max(5, period // 2)).sum()
    neg_sum = negative.rolling(period, min_periods=max(5, period // 2)).sum()
    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    return mfi.replace([np.inf, -np.inf], np.nan).fillna(50.0)


def _commodity_channel_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    sma = typical.rolling(period, min_periods=max(5, period // 2)).mean()
    mean_abs_dev = (typical - sma).abs().rolling(period, min_periods=max(5, period // 2)).mean()
    cci = (typical - sma) / (0.015 * mean_abs_dev.replace(0.0, np.nan))
    return cci.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_volatility(close: pd.Series, period: int = 7) -> pd.Series:
    vol = close.pct_change().rolling(period, min_periods=max(3, period // 2)).std()
    return vol.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_drawdown(close: pd.Series, period: int = 7) -> pd.Series:
    peak = close.rolling(period, min_periods=1).max()
    return (close / peak - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _donchian_position(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    upper = high.rolling(period, min_periods=max(5, period // 2)).max()
    lower = low.rolling(period, min_periods=max(5, period // 2)).min()
    span = (upper - lower).replace(0.0, np.nan)
    position = (close - lower) / span
    return position.replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(-0.5, 1.5)


def _return_spread(left: pd.Series, right: pd.Series, period: int) -> pd.Series:
    return left.pct_change(period) - right.pct_change(period)


def _safe_numeric_feature(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    series = pd.to_numeric(df[column], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan).fillna(default).astype("float64")


def _session_phase(index: pd.DatetimeIndex) -> pd.Series:
    normalized = pd.DatetimeIndex(index)
    if normalized.tz is None:
        normalized = normalized.tz_localize("UTC")
    else:
        normalized = normalized.tz_convert("UTC")
    phase = (normalized.hour * 60.0 + normalized.minute) / (24.0 * 60.0)
    return pd.Series(phase.astype("float64"), index=index)


def _session_flag(index: pd.DatetimeIndex, start_hour: int, end_hour: int) -> pd.Series:
    normalized = pd.DatetimeIndex(index)
    if normalized.tz is None:
        normalized = normalized.tz_localize("UTC")
    else:
        normalized = normalized.tz_convert("UTC")
    hour = normalized.hour + normalized.minute / 60.0
    return pd.Series(((hour >= start_hour) & (hour < end_hour)).astype("float64"), index=index)


def build_market_features(df: pd.DataFrame, pairs: tuple[str, ...]) -> dict[str, pd.Series]:
    primary_pair = pairs[0]
    secondary_pair = pairs[1] if len(pairs) > 1 else primary_pair
    close = pd.concat([df[f"{asset}_close"].rename(asset) for asset in pairs], axis=1).sort_index()
    volume = pd.concat(
        [
            df[f"{asset}_volume"].rename(asset) if f"{asset}_volume" in df.columns else pd.Series(0.0, index=df.index, name=asset)
            for asset in pairs
        ],
        axis=1,
    ).sort_index()
    high = pd.concat(
        [
            df[f"{asset}_high"].rename(asset) if f"{asset}_high" in df.columns else df[f"{asset}_close"].rename(asset)
            for asset in pairs
        ],
        axis=1,
    ).sort_index()
    low = pd.concat(
        [
            df[f"{asset}_low"].rename(asset) if f"{asset}_low" in df.columns else df[f"{asset}_close"].rename(asset)
            for asset in pairs
        ],
        axis=1,
    ).sort_index()
    daily_close = close.resample("1D").last().dropna()
    daily_high = high.resample("1D").max().reindex(daily_close.index).ffill()
    daily_low = low.resample("1D").min().reindex(daily_close.index).ffill()
    daily_volume = volume.resample("1D").sum(min_count=1).reindex(daily_close.index).fillna(0.0)
    daily_ret = daily_close.pct_change()
    btc_return_1d = daily_ret[primary_pair]
    bnb_return_1d = daily_ret[secondary_pair]
    btc_return_3d = daily_close[primary_pair].pct_change(3)
    bnb_return_3d = daily_close[secondary_pair].pct_change(3)
    btc_return_7d = daily_close[primary_pair].pct_change(7)
    bnb_return_7d = daily_close[secondary_pair].pct_change(7)
    btc_momentum_1d = btc_return_1d
    bnb_momentum_1d = bnb_return_1d
    btc_momentum_3d = btc_return_3d
    bnb_momentum_3d = bnb_return_3d
    btc_regime = 0.60 * btc_momentum_3d + 0.40 * daily_close[primary_pair].pct_change(14)
    bnb_regime = 0.60 * bnb_momentum_3d + 0.40 * daily_close[secondary_pair].pct_change(14)
    regime_spread = btc_regime - bnb_regime
    breadth = (daily_close.pct_change(3) > 0.0).mean(axis=1)
    breadth_change_1d = breadth.diff()
    btc_momentum_accel_1d_3d = btc_momentum_1d - btc_momentum_3d
    bnb_momentum_accel_1d_3d = bnb_momentum_1d - bnb_momentum_3d
    rel_strength = bnb_momentum_3d - btc_momentum_3d

    btc_vol_ann = close[primary_pair].pct_change().rolling(12 * 24 * 3).std() * BAR_FACTOR
    bnb_vol_ann = close[secondary_pair].pct_change().rolling(12 * 24 * 3).std() * BAR_FACTOR
    btc_vol_rel = btc_vol_ann / btc_vol_ann.rolling(12 * 24 * 7).median()
    bnb_vol_rel = bnb_vol_ann / bnb_vol_ann.rolling(12 * 24 * 7).median()
    vol_rel_spread = btc_vol_rel - bnb_vol_rel
    btc_intraday_return_1h = close[primary_pair].pct_change()
    bnb_intraday_return_1h = close[secondary_pair].pct_change()
    btc_intraday_return_6h = close[primary_pair].pct_change(6)
    bnb_intraday_return_6h = close[secondary_pair].pct_change(6)
    btc_intraday_drawdown_24h = _rolling_drawdown(close[primary_pair], 24)
    bnb_intraday_drawdown_24h = _rolling_drawdown(close[secondary_pair], 24)
    btc_rsi_14_1h = _safe_numeric_feature(df, f"{primary_pair}_rsi_14", 50.0)
    bnb_rsi_14_1h = _safe_numeric_feature(df, f"{secondary_pair}_rsi_14", 50.0)
    btc_mfi_14_1h = _safe_numeric_feature(df, f"{primary_pair}_mfi_14", 50.0)
    bnb_mfi_14_1h = _safe_numeric_feature(df, f"{secondary_pair}_mfi_14", 50.0)
    btc_atr_pct_1h = _safe_numeric_feature(df, f"{primary_pair}_atr_14", 0.0) / close[primary_pair].replace(0.0, np.nan)
    bnb_atr_pct_1h = _safe_numeric_feature(df, f"{secondary_pair}_atr_14", 0.0) / close[secondary_pair].replace(0.0, np.nan)
    btc_macd_h_pct_1h = _safe_numeric_feature(df, f"{primary_pair}_macd_h", 0.0) / close[primary_pair].replace(0.0, np.nan)
    bnb_macd_h_pct_1h = _safe_numeric_feature(df, f"{secondary_pair}_macd_h", 0.0) / close[secondary_pair].replace(0.0, np.nan)
    btc_bb_p_1h = _safe_numeric_feature(df, f"{primary_pair}_bb_p", 0.5).clip(-1.0, 2.0)
    bnb_bb_p_1h = _safe_numeric_feature(df, f"{secondary_pair}_bb_p", 0.5).clip(-1.0, 2.0)
    btc_volume_rel_1h = volume[primary_pair] / _safe_numeric_feature(df, f"{primary_pair}_vol_sma", 0.0).replace(0.0, np.nan)
    bnb_volume_rel_1h = volume[secondary_pair] / _safe_numeric_feature(df, f"{secondary_pair}_vol_sma", 0.0).replace(0.0, np.nan)
    btc_cci_scaled_1h = (_safe_numeric_feature(df, f"{primary_pair}_cci_14", 0.0) / 100000.0).clip(-3.0, 3.0)
    bnb_cci_scaled_1h = (_safe_numeric_feature(df, f"{secondary_pair}_cci_14", 0.0) / 100000.0).clip(-3.0, 3.0)
    btc_dc_trend_05_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_trend_05", 0.0)
    bnb_dc_trend_05_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_trend_05", 0.0)
    btc_dc_event_05_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_event_05", 0.0)
    bnb_dc_event_05_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_event_05", 0.0)
    btc_dc_overshoot_05_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_overshoot_05", 0.0)
    bnb_dc_overshoot_05_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_overshoot_05", 0.0)
    btc_dc_run_05_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_run_05", 0.0)
    bnb_dc_run_05_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_run_05", 0.0)

    btc_drawdown_7d = daily_close[primary_pair] / daily_close[primary_pair].rolling(7, min_periods=1).max() - 1.0
    bnb_drawdown_7d = daily_close[secondary_pair] / daily_close[secondary_pair].rolling(7, min_periods=1).max() - 1.0
    btc_drawdown_21d = _rolling_drawdown(daily_close[primary_pair], 21)
    bnb_drawdown_21d = _rolling_drawdown(daily_close[secondary_pair], 21)
    rel_strength_1d = bnb_momentum_1d - btc_momentum_1d
    btc_rsi_14d = _daily_rsi(daily_close[primary_pair], 14)
    bnb_rsi_14d = _daily_rsi(daily_close[secondary_pair], 14)
    btc_atr_pct_14d = _daily_atr_pct(daily_high[primary_pair], daily_low[primary_pair], daily_close[primary_pair], 14)
    bnb_atr_pct_14d = _daily_atr_pct(daily_high[secondary_pair], daily_low[secondary_pair], daily_close[secondary_pair], 14)
    btc_volatility_7d = _rolling_volatility(daily_close[primary_pair], 7)
    btc_volatility_21d = _rolling_volatility(daily_close[primary_pair], 21)
    bnb_volatility_7d = _rolling_volatility(daily_close[secondary_pair], 7)
    bnb_volatility_21d = _rolling_volatility(daily_close[secondary_pair], 21)
    btc_volume_z_7d = _rolling_zscore(daily_volume[primary_pair], 7)
    bnb_volume_z_7d = _rolling_zscore(daily_volume[secondary_pair], 7)
    btc_macd_line_12_26_9, _, btc_macd_hist_12_26_9 = _macd_features(daily_close[primary_pair], 12, 26, 9)
    bnb_macd_line_12_26_9, _, bnb_macd_hist_12_26_9 = _macd_features(daily_close[secondary_pair], 12, 26, 9)
    _, _, _, btc_bb_pct_b_20_2, btc_bb_width_20_2 = _bollinger_features(daily_close[primary_pair], 20, 2.0)
    _, _, _, bnb_bb_pct_b_20_2, bnb_bb_width_20_2 = _bollinger_features(daily_close[secondary_pair], 20, 2.0)
    btc_mfi_14d = _money_flow_index(daily_high[primary_pair], daily_low[primary_pair], daily_close[primary_pair], daily_volume[primary_pair], 14)
    bnb_mfi_14d = _money_flow_index(daily_high[secondary_pair], daily_low[secondary_pair], daily_close[secondary_pair], daily_volume[secondary_pair], 14)
    btc_cci_20d = _commodity_channel_index(daily_high[primary_pair], daily_low[primary_pair], daily_close[primary_pair], 20)
    bnb_cci_20d = _commodity_channel_index(daily_high[secondary_pair], daily_low[secondary_pair], daily_close[secondary_pair], 20)
    btc_dc_pos_20d = _donchian_position(daily_high[primary_pair], daily_low[primary_pair], daily_close[primary_pair], 20)
    bnb_dc_pos_20d = _donchian_position(daily_high[secondary_pair], daily_low[secondary_pair], daily_close[secondary_pair], 20)
    rsi_spread = btc_rsi_14d - bnb_rsi_14d
    atr_spread = btc_atr_pct_14d - bnb_atr_pct_14d
    macd_hist_spread = btc_macd_hist_12_26_9 - bnb_macd_hist_12_26_9
    bb_pct_b_spread = btc_bb_pct_b_20_2 - bnb_bb_pct_b_20_2
    mfi_spread = btc_mfi_14d - bnb_mfi_14d
    cci_spread = btc_cci_20d - bnb_cci_20d
    dc_pos_spread = btc_dc_pos_20d - bnb_dc_pos_20d
    volume_z_spread = btc_volume_z_7d - bnb_volume_z_7d
    return_spread_1d = btc_return_1d - bnb_return_1d
    return_spread_3d = btc_return_3d - bnb_return_3d
    return_spread_7d = btc_return_7d - bnb_return_7d
    rsi_spread_1h = btc_rsi_14_1h - bnb_rsi_14_1h
    mfi_spread_1h = btc_mfi_14_1h - bnb_mfi_14_1h
    atr_pct_spread_1h = btc_atr_pct_1h - bnb_atr_pct_1h
    macd_h_pct_spread_1h = btc_macd_h_pct_1h - bnb_macd_h_pct_1h
    volume_rel_spread_1h = btc_volume_rel_1h - bnb_volume_rel_1h
    cci_scaled_spread_1h = btc_cci_scaled_1h - bnb_cci_scaled_1h
    dc_trend_spread_1h = btc_dc_trend_05_1h - bnb_dc_trend_05_1h
    dc_event_spread_1h = btc_dc_event_05_1h - bnb_dc_event_05_1h
    dc_overshoot_spread_1h = btc_dc_overshoot_05_1h - bnb_dc_overshoot_05_1h
    dc_run_spread_1h = btc_dc_run_05_1h - bnb_dc_run_05_1h
    intraday_return_spread_1h = btc_intraday_return_1h - bnb_intraday_return_1h
    intraday_return_spread_6h = btc_intraday_return_6h - bnb_intraday_return_6h
    session_phase = _session_phase(pd.DatetimeIndex(df.index))
    session_asia_flag = _session_flag(pd.DatetimeIndex(df.index), 0, 8)
    session_eu_flag = _session_flag(pd.DatetimeIndex(df.index), 7, 15)
    session_us_flag = _session_flag(pd.DatetimeIndex(df.index), 13, 21)

    return {
        "btc_regime": btc_regime,
        "bnb_regime": bnb_regime,
        "regime_spread_btc_minus_bnb": regime_spread,
        "breadth": breadth,
        "breadth_change_1d": breadth_change_1d,
        "btc_return_1d": btc_return_1d,
        "btc_return_3d": btc_return_3d,
        "btc_return_7d": btc_return_7d,
        "bnb_return_1d": bnb_return_1d,
        "bnb_return_3d": bnb_return_3d,
        "bnb_return_7d": bnb_return_7d,
        "btc_momentum_1d": btc_momentum_1d,
        "bnb_momentum_1d": bnb_momentum_1d,
        "btc_momentum_3d": btc_momentum_3d,
        "bnb_momentum_3d": bnb_momentum_3d,
        "btc_momentum_accel_1d_3d": btc_momentum_accel_1d_3d,
        "bnb_momentum_accel_1d_3d": bnb_momentum_accel_1d_3d,
        "rel_strength_bnb_btc_3d": rel_strength,
        "rel_strength_bnb_btc_1d": rel_strength_1d,
        "btc_vol_rel": btc_vol_rel,
        "bnb_vol_rel": bnb_vol_rel,
        "vol_rel_spread_btc_minus_bnb": vol_rel_spread,
        "btc_drawdown_7d": btc_drawdown_7d,
        "bnb_drawdown_7d": bnb_drawdown_7d,
        "btc_drawdown_21d": btc_drawdown_21d,
        "bnb_drawdown_21d": bnb_drawdown_21d,
        "btc_rsi_14d": btc_rsi_14d,
        "bnb_rsi_14d": bnb_rsi_14d,
        "btc_atr_pct_14d": btc_atr_pct_14d,
        "bnb_atr_pct_14d": bnb_atr_pct_14d,
        "btc_volatility_7d": btc_volatility_7d,
        "btc_volatility_21d": btc_volatility_21d,
        "bnb_volatility_7d": bnb_volatility_7d,
        "bnb_volatility_21d": bnb_volatility_21d,
        "btc_volume_z_7d": btc_volume_z_7d,
        "bnb_volume_z_7d": bnb_volume_z_7d,
        "btc_macd_line_12_26_9": btc_macd_line_12_26_9,
        "btc_macd_hist_12_26_9": btc_macd_hist_12_26_9,
        "bnb_macd_line_12_26_9": bnb_macd_line_12_26_9,
        "bnb_macd_hist_12_26_9": bnb_macd_hist_12_26_9,
        "btc_bb_pct_b_20_2": btc_bb_pct_b_20_2,
        "btc_bb_width_20_2": btc_bb_width_20_2,
        "bnb_bb_pct_b_20_2": bnb_bb_pct_b_20_2,
        "bnb_bb_width_20_2": bnb_bb_width_20_2,
        "btc_mfi_14d": btc_mfi_14d,
        "bnb_mfi_14d": bnb_mfi_14d,
        "btc_cci_20d": btc_cci_20d,
        "bnb_cci_20d": bnb_cci_20d,
        "btc_dc_pos_20d": btc_dc_pos_20d,
        "bnb_dc_pos_20d": bnb_dc_pos_20d,
        "btc_intraday_return_1h": btc_intraday_return_1h,
        "bnb_intraday_return_1h": bnb_intraday_return_1h,
        "btc_intraday_return_6h": btc_intraday_return_6h,
        "bnb_intraday_return_6h": bnb_intraday_return_6h,
        "btc_intraday_drawdown_24h": btc_intraday_drawdown_24h,
        "bnb_intraday_drawdown_24h": bnb_intraday_drawdown_24h,
        "btc_rsi_14_1h": btc_rsi_14_1h,
        "bnb_rsi_14_1h": bnb_rsi_14_1h,
        "btc_mfi_14_1h": btc_mfi_14_1h,
        "bnb_mfi_14_1h": bnb_mfi_14_1h,
        "btc_atr_pct_1h": btc_atr_pct_1h,
        "bnb_atr_pct_1h": bnb_atr_pct_1h,
        "btc_macd_h_pct_1h": btc_macd_h_pct_1h,
        "bnb_macd_h_pct_1h": bnb_macd_h_pct_1h,
        "btc_bb_p_1h": btc_bb_p_1h,
        "bnb_bb_p_1h": bnb_bb_p_1h,
        "btc_volume_rel_1h": btc_volume_rel_1h,
        "bnb_volume_rel_1h": bnb_volume_rel_1h,
        "btc_cci_scaled_1h": btc_cci_scaled_1h,
        "bnb_cci_scaled_1h": bnb_cci_scaled_1h,
        "btc_dc_trend_05_1h": btc_dc_trend_05_1h,
        "bnb_dc_trend_05_1h": bnb_dc_trend_05_1h,
        "btc_dc_event_05_1h": btc_dc_event_05_1h,
        "bnb_dc_event_05_1h": bnb_dc_event_05_1h,
        "btc_dc_overshoot_05_1h": btc_dc_overshoot_05_1h,
        "bnb_dc_overshoot_05_1h": bnb_dc_overshoot_05_1h,
        "btc_dc_run_05_1h": btc_dc_run_05_1h,
        "bnb_dc_run_05_1h": bnb_dc_run_05_1h,
        "rsi_spread_btc_minus_bnb_14d": rsi_spread,
        "atr_spread_btc_minus_bnb_14d": atr_spread,
        "macd_hist_spread_btc_minus_bnb_12_26_9": macd_hist_spread,
        "bb_pct_b_spread_btc_minus_bnb_20_2": bb_pct_b_spread,
        "mfi_spread_btc_minus_bnb_14d": mfi_spread,
        "cci_spread_btc_minus_bnb_20d": cci_spread,
        "dc_pos_spread_btc_minus_bnb_20d": dc_pos_spread,
        "volume_z_spread_btc_minus_bnb_7d": volume_z_spread,
        "return_spread_btc_minus_bnb_1d": return_spread_1d,
        "return_spread_btc_minus_bnb_3d": return_spread_3d,
        "return_spread_btc_minus_bnb_7d": return_spread_7d,
        "rsi_spread_btc_minus_bnb_1h": rsi_spread_1h,
        "mfi_spread_btc_minus_bnb_1h": mfi_spread_1h,
        "atr_pct_spread_btc_minus_bnb_1h": atr_pct_spread_1h,
        "macd_h_pct_spread_btc_minus_bnb_1h": macd_h_pct_spread_1h,
        "volume_rel_spread_btc_minus_bnb_1h": volume_rel_spread_1h,
        "cci_scaled_spread_btc_minus_bnb_1h": cci_scaled_spread_1h,
        "dc_trend_spread_btc_minus_bnb_1h": dc_trend_spread_1h,
        "dc_event_spread_btc_minus_bnb_1h": dc_event_spread_1h,
        "dc_overshoot_spread_btc_minus_bnb_1h": dc_overshoot_spread_1h,
        "dc_run_spread_btc_minus_bnb_1h": dc_run_spread_1h,
        "intraday_return_spread_btc_minus_bnb_1h": intraday_return_spread_1h,
        "intraday_return_spread_btc_minus_bnb_6h": intraday_return_spread_6h,
        "session_utc_phase": session_phase,
        "session_asia_flag": session_asia_flag,
        "session_eu_flag": session_eu_flag,
        "session_us_flag": session_us_flag,
    }


def materialize_feature_arrays(features: dict[str, pd.Series], index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    day_index = index.normalize()
    intraday_features = {
        "btc_vol_rel",
        "bnb_vol_rel",
        "vol_rel_spread_btc_minus_bnb",
        "session_utc_phase",
        "session_asia_flag",
        "session_eu_flag",
        "session_us_flag",
    }
    out: dict[str, np.ndarray] = {}
    for name, series in features.items():
        if name in intraday_features or name.endswith(("_1h", "_6h", "_24h")):
            values = series.reindex(index).ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(1.0)
        else:
            values = series.reindex(day_index, method="ffill").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out[name] = values.to_numpy(dtype="float64")
    return out


def project_pair_configs(pair_configs: dict[str, Any], required_pairs: tuple[str, ...]) -> dict[str, Any]:
    return {
        pair: copy.deepcopy(pair_configs[pair])
        for pair in required_pairs
        if pair in pair_configs
    }


def pair_config_signature(pair_configs: dict[str, Any], required_pairs: tuple[str, ...]) -> str:
    return json.dumps(project_pair_configs(pair_configs, required_pairs), sort_keys=True, ensure_ascii=False)


def _route_threshold_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    return abs(float(left["route_breadth_threshold"]) - float(right["route_breadth_threshold"]))


def _mapping_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_mapping = [int(v) for v in left["mapping_indices"]]
    right_mapping = [int(v) for v in right["mapping_indices"]]
    if not left_mapping and not right_mapping:
        return 0.0
    length = max(len(left_mapping), len(right_mapping), 1)
    padded_left = left_mapping + [left_mapping[-1] if left_mapping else 0] * (length - len(left_mapping))
    padded_right = right_mapping + [right_mapping[-1] if right_mapping else 0] * (length - len(right_mapping))
    return float(sum(abs(a - b) for a, b in zip(padded_left, padded_right)) / length)


def projected_pair_config_distance(
    left_pair_configs: dict[str, Any],
    right_pair_configs: dict[str, Any],
    required_pairs: tuple[str, ...],
) -> float:
    total = 0.0
    counted = 0
    for pair in required_pairs:
        left = left_pair_configs.get(pair)
        right = right_pair_configs.get(pair)
        if left is None or right is None:
            continue
        total += 0.55 * _route_threshold_distance(left, right) + 0.45 * _mapping_distance(left, right)
        counted += 1
    return 0.0 if counted == 0 else total / counted


def _candidate_projected_record(candidate: dict[str, Any], required_pairs: tuple[str, ...]) -> dict[str, Any] | None:
    pair_configs = candidate.get("pair_configs")
    if not pair_configs:
        return None
    projected = project_pair_configs(pair_configs, required_pairs)
    if not projected:
        return None
    windows = candidate.get("windows")
    score = score_realistic_candidate({"windows": windows}) if windows else float(candidate.get("score", 0.0))
    return {
        "pair_configs": candidate["pair_configs"],
        "projected_pair_configs": projected,
        "signature": pair_config_signature(candidate["pair_configs"], required_pairs),
        "score": float(score),
    }


def build_expert_pool(
    summary_paths: list[str],
    pool_size: int,
    required_pairs: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_candidate_count = 0
    projected_candidates: list[dict[str, Any]] = []
    for raw_path in summary_paths:
        obj = json.loads(Path(raw_path).read_text())
        candidates: list[dict[str, Any]] = []
        if obj.get("selected_candidate") and obj["selected_candidate"].get("pair_configs"):
            candidates.append(obj["selected_candidate"])
        candidates.extend(obj.get("realistic_top_candidates", []))
        for candidate in candidates:
            raw_candidate_count += 1
            record = _candidate_projected_record(candidate, required_pairs)
            if record is not None:
                projected_candidates.append(record)

    by_key: dict[str, dict[str, Any]] = {}
    for record in projected_candidates:
        existing = by_key.get(record["signature"])
        if existing is None or record["score"] > existing["score"]:
            by_key[record["signature"]] = record

    unique_candidates = list(by_key.values())
    score_values = [item["score"] for item in unique_candidates]
    score_min = min(score_values) if score_values else 0.0
    score_max = max(score_values) if score_values else 0.0
    score_span = max(score_max - score_min, 1e-12)

    def normalized_score(item: dict[str, Any]) -> float:
        return (float(item["score"]) - score_min) / score_span

    selected: list[dict[str, Any]] = []
    selection_trace: list[dict[str, Any]] = []
    coverage_state: dict[str, dict[str, set[Any]]] = {
        pair: {"route": set(), "mapping": set()} for pair in required_pairs
    }
    remaining = unique_candidates[:]
    while remaining and len(selected) < pool_size:
        best_item = None
        best_key: tuple[float, float, float, str] | None = None
        for item in remaining:
            min_distance = 1.0 if not selected else min(
                projected_pair_config_distance(item["projected_pair_configs"], other["projected_pair_configs"], required_pairs)
                for other in selected
            )
            coverage_bonus = 0.0
            for pair in required_pairs:
                projected_cfg = item["projected_pair_configs"][pair]
                route_key = float(projected_cfg["route_breadth_threshold"])
                mapping_key = tuple(int(v) for v in projected_cfg["mapping_indices"])
                if route_key not in coverage_state[pair]["route"]:
                    coverage_bonus += 0.35
                if mapping_key not in coverage_state[pair]["mapping"]:
                    coverage_bonus += 0.35
            utility = normalized_score(item) + 0.45 * min_distance + coverage_bonus
            tie_break = (utility, float(item["score"]), min_distance, item["signature"])
            if best_key is None or tie_break > best_key:
                best_key = tie_break
                best_item = item
        if best_item is None:
            break
        selected.append(best_item)
        selection_trace.append(
            {
                "rank": len(selected),
                "signature": best_item["signature"],
                "score": best_item["score"],
                "normalized_score": normalized_score(best_item),
                "min_distance_to_selected": 0.0 if len(selected) == 1 else min(
                    projected_pair_config_distance(best_item["projected_pair_configs"], other["projected_pair_configs"], required_pairs)
                    for other in selected[:-1]
                ),
            }
        )
        for pair in required_pairs:
            projected_cfg = best_item["projected_pair_configs"][pair]
            coverage_state[pair]["route"].add(float(projected_cfg["route_breadth_threshold"]))
            coverage_state[pair]["mapping"].add(tuple(int(v) for v in projected_cfg["mapping_indices"]))
        remaining = [item for item in remaining if item["signature"] != best_item["signature"]]

    pairwise_distances = [
        projected_pair_config_distance(a["projected_pair_configs"], b["projected_pair_configs"], required_pairs)
        for idx, a in enumerate(selected)
        for b in selected[idx + 1 :]
    ]
    diagnostics = {
        "required_pairs": list(required_pairs),
        "raw_candidate_count": raw_candidate_count,
        "projected_candidate_count": len(projected_candidates),
        "unique_projected_candidate_count": len(unique_candidates),
        "projection_collision_count": len(projected_candidates) - len(unique_candidates),
        "selected_count": len(selected),
        "score_range": {"min": score_min, "max": score_max, "span": score_span},
        "pairwise_distance": {
            "min": min(pairwise_distances) if pairwise_distances else 0.0,
            "mean": float(sum(pairwise_distances) / len(pairwise_distances)) if pairwise_distances else 0.0,
            "max": max(pairwise_distances) if pairwise_distances else 0.0,
        },
        "selection_trace": selection_trace,
        "coverage": {
            pair: {
                "route_threshold_count": len(coverage_state[pair]["route"]),
                "mapping_signature_count": len(coverage_state[pair]["mapping"]),
            }
            for pair in required_pairs
        },
    }
    return selected, diagnostics


def _tree_leaf_signature(node: TreeNode) -> tuple[str, ...]:
    return tuple(sorted(set(collect_leaf_keys(node))))


def _candidate_tree_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    leaf_left = set(left["leaf_signature"])
    leaf_right = set(right["leaf_signature"])
    if leaf_left or leaf_right:
        leaf_union = leaf_left | leaf_right
        leaf_intersection = leaf_left & leaf_right
        leaf_distance = 1.0 - (len(leaf_intersection) / len(leaf_union))
    else:
        leaf_distance = 0.0
    depth_distance = abs(left["tree_depth"] - right["tree_depth"]) / max(left["tree_depth"], right["tree_depth"], 1)
    logic_depth_distance = abs(left["logic_depth"] - right["logic_depth"]) / max(left["logic_depth"], right["logic_depth"], 1)
    size_distance = abs(left["tree_size"] - right["tree_size"]) / max(left["tree_size"], right["tree_size"], 1)
    logic_size_distance = abs(left["logic_size"] - right["logic_size"]) / max(left["logic_size"], right["logic_size"], 1)
    condition_distance = abs(left["condition_count"] - right["condition_count"]) / max(left["condition_count"], right["condition_count"], 1)
    return (
        0.30 * leaf_distance
        + 0.20 * depth_distance
        + 0.15 * logic_depth_distance
        + 0.15 * size_distance
        + 0.10 * logic_size_distance
        + 0.10 * condition_distance
    )


def select_generation_survivors(
    evaluated: list[dict[str, Any]],
    survivor_count: int,
    diversity_weight: float,
    depth_weight: float,
    target_tree_depth: int = 1,
    target_logic_depth: int = 0,
    persistent_tree_depth: int = 3,
    persistent_logic_depth: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if survivor_count <= 0:
        return [], {"selected_count": 0}
    ranked = sorted(evaluated, key=lambda item: item["search_fitness"], reverse=True)
    if not ranked:
        return [], {"selected_count": 0}

    candidate_count = len(ranked)
    candidates = ranked[:candidate_count]
    fitness_values = [float(item["search_fitness"]) for item in candidates]
    fitness_min = min(fitness_values)
    fitness_max = max(fitness_values)
    fitness_span = max(fitness_max - fitness_min, 1e-12)

    def normalized_fitness(item: dict[str, Any]) -> float:
        return (float(item["search_fitness"]) - fitness_min) / fitness_span

    selected: list[dict[str, Any]] = []
    selected_depths: set[int] = set()
    selected_logic_depths: set[int] = set()
    persistent_archive_candidate = None
    archived_target_candidate = None

    persistent_candidates = [
        item
        for item in candidates
        if int(item["tree_depth"]) >= int(persistent_tree_depth) or int(item["logic_depth"]) >= int(persistent_logic_depth)
    ]
    if persistent_candidates:
        persistent_archive_candidate = max(
            persistent_candidates,
            key=lambda item: (
                1 if candidate_stress_pass(item) else 0,
                1 if candidate_wf1_pass(item) else 0,
                int(item["tree_depth"]) >= int(persistent_tree_depth),
                int(item["logic_depth"]) >= int(persistent_logic_depth),
                float(item["search_fitness"]),
                float(item["structural_score"]),
                item["tree_key"],
            ),
        )
        selected.append(persistent_archive_candidate)
        selected_depths.add(int(persistent_archive_candidate["tree_depth"]))
        selected_logic_depths.add(int(persistent_archive_candidate["logic_depth"]))
        candidates = [item for item in candidates if item["tree_key"] != persistent_archive_candidate["tree_key"]]

    if target_tree_depth > 1 or target_logic_depth > 0:
        target_candidates = [
            item
            for item in candidates
            if int(item["tree_depth"]) >= int(target_tree_depth)
        ]
        if not target_candidates and target_logic_depth > 0:
            target_candidates = [
                item
                for item in candidates
                if int(item["logic_depth"]) >= int(target_logic_depth)
            ]
        if target_candidates:
            archived_target_candidate = max(
                target_candidates,
                key=lambda item: (
                    1 if candidate_cost_reserve_pass(item) else 0,
                    1 if candidate_stress_pass(item) else 0,
                    1 if candidate_wf1_pass(item) else 0,
                    min(int(item["tree_depth"]), int(target_tree_depth)),
                    min(int(item["logic_depth"]), int(target_logic_depth)),
                    float(item["search_fitness"]),
                    float(item["structural_score"]),
                    item["tree_key"],
                ),
            )
            if len(selected) < survivor_count:
                selected.append(archived_target_candidate)
                selected_depths.add(int(archived_target_candidate["tree_depth"]))
                selected_logic_depths.add(int(archived_target_candidate["logic_depth"]))
                candidates = [item for item in candidates if item["tree_key"] != archived_target_candidate["tree_key"]]

    while candidates and len(selected) < survivor_count:
        best_item = None
        best_key: tuple[float, float, float, str] | None = None
        for item in candidates:
            min_distance = 1.0 if not selected else min(_candidate_tree_distance(item, other) for other in selected)
            depth_bucket = int(item["tree_depth"])
            logic_bucket = int(item["logic_depth"])
            depth_coverage_bonus = 0.0
            if depth_bucket not in selected_depths:
                depth_coverage_bonus += 1.0
            if logic_bucket not in selected_logic_depths:
                depth_coverage_bonus += 0.5
            if depth_bucket >= int(persistent_tree_depth) or logic_bucket >= int(persistent_logic_depth):
                depth_coverage_bonus += 0.85
            target_alignment_bonus = 0.0
            if depth_bucket >= int(target_tree_depth):
                target_alignment_bonus += 1.0
            else:
                target_alignment_bonus -= 0.45 * float(int(target_tree_depth) - depth_bucket)
            if logic_bucket >= int(target_logic_depth):
                target_alignment_bonus += 0.5
            else:
                target_alignment_bonus -= 0.30 * float(int(target_logic_depth) - logic_bucket)
            robustness = item["robustness"]
            stress_floor = float(
                robustness.get("min_fold_stress_survival_rate", robustness.get("stress_survival_rate_min", 0.0))
            )
            stress_mean = float(robustness.get("stress_survival_rate_mean", 0.0))
            stress_threshold = float(robustness.get("stress_survival_threshold", 0.0))
            stress_reserve_score = float(robustness.get("latest_fold_stress_reserve_score", 0.0))
            non_nominal_stress_rate = float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0))
            non_nominal_stress_floor = float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0))
            non_nominal_stress_reserve = float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0))
            robustness_bonus = 0.0
            robustness_bonus += 0.90 if candidate_wf1_pass(item) else -0.90
            robustness_bonus += 2.20 if candidate_stress_pass(item) else -3.20
            robustness_bonus += 2.60 if candidate_cost_reserve_pass(item) else -3.40
            robustness_bonus += max(0.0, stress_mean - stress_threshold) * 3.80
            robustness_bonus += max(0.0, stress_floor - stress_threshold) * 4.60
            robustness_bonus += max(0.0, stress_reserve_score) / 6000.0
            robustness_bonus += max(0.0, non_nominal_stress_rate - stress_threshold) * 5.20
            robustness_bonus += max(0.0, non_nominal_stress_floor - stress_threshold) * 6.20
            robustness_bonus += max(0.0, non_nominal_stress_reserve) / 6000.0
            if stress_mean < stress_threshold:
                robustness_bonus -= (stress_threshold - stress_mean) * 4.20
            if stress_floor < stress_threshold:
                robustness_bonus -= (stress_threshold - stress_floor) * 5.40
            if stress_reserve_score < 0.0:
                robustness_bonus -= abs(stress_reserve_score) / 2500.0
            if non_nominal_stress_rate < stress_threshold:
                robustness_bonus -= (stress_threshold - non_nominal_stress_rate) * 6.20
            if non_nominal_stress_floor < stress_threshold:
                robustness_bonus -= (stress_threshold - non_nominal_stress_floor) * 7.20
            if non_nominal_stress_reserve < 0.0:
                robustness_bonus -= abs(non_nominal_stress_reserve) / 2200.0
            utility = normalized_fitness(item) + diversity_weight * min_distance + depth_weight * depth_coverage_bonus
            utility += depth_weight * 0.75 * target_alignment_bonus
            utility += depth_weight * 0.45 * robustness_bonus
            utility += 0.35 * float(item["structural_score"])
            tie_break = (
                utility,
                1.0 if candidate_cost_reserve_pass(item) else 0.0,
                1.0 if candidate_wf1_pass(item) else 0.0,
                1.0 if candidate_stress_pass(item) else 0.0,
                float(item["search_fitness"]),
                min_distance,
                item["tree_key"],
            )
            if best_key is None or tie_break > best_key:
                best_key = tie_break
                best_item = item
        if best_item is None:
            break
        selected.append(best_item)
        selected_depths.add(int(best_item["tree_depth"]))
        selected_logic_depths.add(int(best_item["logic_depth"]))
        candidates = [item for item in candidates if item["tree_key"] != best_item["tree_key"]]

    pairwise_distances = [
        _candidate_tree_distance(a, b)
        for idx, a in enumerate(selected)
        for b in selected[idx + 1 :]
    ]
    diagnostics = {
        "selected_count": len(selected),
        "candidate_count": len(ranked),
        "fitness_range": {"min": fitness_min, "max": fitness_max, "span": fitness_span},
        "structural_score_range": {
            "min": min(float(item["structural_score"]) for item in ranked) if ranked else 0.0,
            "max": max(float(item["structural_score"]) for item in ranked) if ranked else 0.0,
        },
        "selected_depths": [int(item["tree_depth"]) for item in selected],
        "selected_logic_depths": [int(item["logic_depth"]) for item in selected],
        "selected_leaf_cardinalities": [int(item["leaf_cardinality"]) for item in selected],
        "depth_coverage": {
            str(depth): sum(1 for item in selected if int(item["tree_depth"]) == depth)
            for depth in sorted(selected_depths)
        },
        "logic_depth_coverage": {
            str(depth): sum(1 for item in selected if int(item["logic_depth"]) == depth)
            for depth in sorted(selected_logic_depths)
        },
        "pairwise_distance": {
            "min": min(pairwise_distances) if pairwise_distances else 0.0,
            "mean": float(sum(pairwise_distances) / len(pairwise_distances)) if pairwise_distances else 0.0,
            "max": max(pairwise_distances) if pairwise_distances else 0.0,
        },
        "diversity_weight": diversity_weight,
        "depth_weight": depth_weight,
        "target_tree_depth": int(target_tree_depth),
        "target_logic_depth": int(target_logic_depth),
        "persistent_tree_depth": int(persistent_tree_depth),
        "persistent_logic_depth": int(persistent_logic_depth),
        "persistent_archive_candidate": None if persistent_archive_candidate is None else {
            "tree_key": persistent_archive_candidate["tree_key"],
            "tree_depth": int(persistent_archive_candidate["tree_depth"]),
            "logic_depth": int(persistent_archive_candidate["logic_depth"]),
        },
        "archived_target_candidate": None if archived_target_candidate is None else {
            "tree_key": archived_target_candidate["tree_key"],
            "tree_depth": int(archived_target_candidate["tree_depth"]),
            "logic_depth": int(archived_target_candidate["logic_depth"]),
        },
        "wf1_pass_count": sum(1 for item in selected if candidate_wf1_pass(item)),
        "stress_pass_count": sum(1 for item in selected if candidate_stress_pass(item)),
        "cost_reserve_pass_count": sum(1 for item in selected if candidate_cost_reserve_pass(item)),
        "top_fitness_depths": [int(item["tree_depth"]) for item in ranked[:survivor_count]],
        "top_fitness_logic_depths": [int(item["logic_depth"]) for item in ranked[:survivor_count]],
    }
    return selected, diagnostics


def select_near_frontier_structural_winner(
    candidates: list[dict[str, Any]],
    frontier_ratio: float = 0.080,
    frontier_floor: float = 250.0,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not candidates:
        return None, {"selected": False, "candidate_count": 0}
    best_performance = max(float(item["performance_score"]) for item in candidates)
    frontier_band = max(frontier_floor, abs(best_performance) * frontier_ratio)
    frontier = [
        item
        for item in candidates
        if best_performance - float(item["performance_score"]) <= frontier_band
    ]
    if not frontier:
        frontier = candidates[:]

    wf1_frontier = [item for item in frontier if candidate_wf1_pass(item)]
    stress_frontier = [item for item in wf1_frontier if candidate_stress_pass(item)]
    reserve_frontier = [item for item in wf1_frontier if candidate_cost_reserve_pass(item)]
    selection_frontier = stress_frontier or reserve_frontier or wf1_frontier or frontier

    def key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, str]:
        robustness = item.get("robustness", {})
        return (
            float(candidate_wf1_pass(item)),
            float(candidate_stress_pass(item)),
            float(candidate_cost_reserve_pass(item)),
            float(robustness.get("latest_fold_stress_reserve_score", 0.0)),
            float(item["structural_score"]),
            float(item["logic_depth"]),
            float(item["tree_depth"]),
            float(item["leaf_cardinality"]),
            float(item["performance_score"]),
            float(item["fitness"]),
            item["tree_key"],
        )

    winner = max(selection_frontier, key=key)
    diagnostics = {
        "selected": True,
        "candidate_count": len(candidates),
        "frontier_count": len(frontier),
        "wf1_frontier_count": len(wf1_frontier),
        "stress_frontier_count": len(stress_frontier),
        "reserve_frontier_count": len(reserve_frontier),
        "selection_frontier_count": len(selection_frontier),
        "best_performance_score": best_performance,
        "frontier_band": frontier_band,
        "winner_performance_score": float(winner["performance_score"]),
        "winner_structural_score": float(winner["structural_score"]),
        "winner_tree_depth": int(winner["tree_depth"]),
        "winner_logic_depth": int(winner["logic_depth"]),
        "winner_leaf_cardinality": int(winner["leaf_cardinality"]),
        "winner_stress_reserve_score": float(winner.get("robustness", {}).get("latest_fold_stress_reserve_score", 0.0)),
        "winner_wf1_pass": candidate_wf1_pass(winner),
        "winner_stress_pass": candidate_stress_pass(winner),
        "winner_cost_reserve_pass": candidate_cost_reserve_pass(winner),
    }
    return winner, diagnostics


def compute_dynamic_windows(index: pd.DatetimeIndex) -> list[dict[str, str]]:
    if len(index) == 0:
        raise RuntimeError("Cannot compute dynamic windows on an empty dataset.")
    normalized = pd.DatetimeIndex(index)
    if normalized.tz is None:
        normalized = normalized.tz_localize("UTC")
    else:
        normalized = normalized.tz_convert("UTC")
    first_ts = normalized[0]
    latest_ts = normalized[-1]
    end_day = latest_ts.normalize()
    first_day = first_ts.normalize()

    def clamp_start(months: int) -> pd.Timestamp:
        return max(first_day, end_day - pd.DateOffset(months=months))

    windows = [
        {
            "key": "recent_2m",
            "label": "recent_2m",
            "description": "Latest 2 months ending at the most recent bar",
            "start": clamp_start(2).date().isoformat(),
            "end": end_day.date().isoformat(),
        },
        {
            "key": "recent_6m",
            "label": "recent_6m",
            "description": "Latest 6 months ending at the most recent bar",
            "start": clamp_start(6).date().isoformat(),
            "end": end_day.date().isoformat(),
        },
        {
            "key": WINDOW_COMPAT_LABEL_FULL,
            "label": "full_history",
            "description": "From the first collected bar to the most recent bar",
            "start": first_day.date().isoformat(),
            "end": end_day.date().isoformat(),
        },
    ]
    return windows


def build_walk_forward_ranges(
    index: pd.DatetimeIndex,
    folds: int,
    test_months: int,
) -> list[dict[str, str]]:
    if folds <= 0 or test_months <= 0:
        return []
    normalized = pd.DatetimeIndex(index)
    if normalized.tz is None:
        normalized = normalized.tz_localize("UTC")
    else:
        normalized = normalized.tz_convert("UTC")
    first_day = normalized[0].normalize()
    latest_day = normalized[-1].normalize()
    ranges: list[dict[str, str]] = []
    current_end = latest_day
    for idx in range(folds):
        start = max(first_day, current_end - pd.DateOffset(months=test_months))
        if start >= current_end:
            break
        ranges.append(
            {
                "key": f"wf_{idx + 1}",
                "label": f"walk_forward_{idx + 1}",
                "description": f"Walk-forward fold {idx + 1} covering the previous {test_months} months.",
                "start": start.strftime("%Y-%m-%d"),
                "end": current_end.strftime("%Y-%m-%d"),
            }
        )
        current_end = start
        if current_end <= first_day:
            break
    ranges.reverse()
    return ranges


def score_window_labels(windows: dict[str, Any]) -> tuple[str, str, str]:
    full_key = WINDOW_COMPAT_LABEL_FULL if WINDOW_COMPAT_LABEL_FULL in windows else "full_history"
    return "recent_2m", "recent_6m", full_key


def build_baseline_windows_for_pairs(
    baseline_windows: dict[str, Any],
    pairs: tuple[str, ...],
    window_specs: list[dict[str, str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in window_specs:
        key = spec["key"]
        source_window = baseline_windows[key]
        per_pair_source = source_window.get("per_pair", {})
        filtered_per_pair = {pair: per_pair_source[pair] for pair in pairs if pair in per_pair_source}
        aggregate = source_window["aggregate"] if not filtered_per_pair else aggregate_metrics(filtered_per_pair)
        out[key] = {
            "start": spec["start"],
            "end": spec["end"],
            "bars": source_window.get("bars"),
            "per_pair": filtered_per_pair,
            "aggregate": aggregate,
        }
    return out


def curriculum_budget(generation_idx: int, total_generations: int, min_depth: int, max_depth: int) -> int:
    capped_min = max(1, min_depth)
    capped_max = max(capped_min, max_depth)
    if total_generations <= 1 or capped_min == capped_max:
        return capped_max
    progress = generation_idx / float(total_generations - 1)
    return int(round(capped_min + (capped_max - capped_min) * progress))


def load_funding_from_cache_or_empty(symbol: str, start: str, end: str) -> pd.DataFrame:
    exact_path = gp.DATA_DIR / f"{symbol}_funding_{start}_{end}.csv"
    candidate_paths = []
    if exact_path.exists():
        candidate_paths.append(exact_path)
    candidate_paths.extend(sorted(gp.DATA_DIR.glob(f"{symbol}_funding_*.csv")))
    for path in candidate_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "fundingTime" not in df.columns or "fundingRate" not in df.columns:
            continue
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, format="mixed", errors="coerce")
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        sliced = df.dropna(subset=["fundingTime", "fundingRate"]).sort_values("fundingTime")
        if sliced.empty:
            continue
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        sliced = sliced[(sliced["fundingTime"] >= start_ts) & (sliced["fundingTime"] <= end_ts)].reset_index(drop=True)
        if not sliced.empty:
            return sliced
    return pd.DataFrame(columns=["fundingTime", "fundingRate"])


def call_openai_tree_review(
    prompt: str,
    model: str,
    timeout_seconds: float,
) -> FilterDecision | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Review trading logic trees for semantic plausibility. "
                    "Return compact JSON with keys accepted (boolean) and reason (string)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib_request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None
    try:
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        accepted = bool(parsed["accepted"])
        reason = str(parsed.get("reason", "reviewed"))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError):
        return None
    return FilterDecision(
        accepted=accepted,
        source="auto_llm_review",
        reason=reason,
        llm_prompt=prompt,
    )


def auto_review_top_candidates(
    candidates: list[dict[str, Any]],
    expert_pool: list[dict[str, Any]],
    llm_reviews: dict[str, FilterDecision],
    top_n: int,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if top_n <= 0:
        return {"enabled": False, "attempted": 0, "added": 0, "skipped": "top_n_zero"}
    if not os.environ.get("OPENAI_API_KEY"):
        return {"enabled": False, "attempted": 0, "added": 0, "skipped": "missing_openai_api_key"}

    attempted = 0
    added = 0
    reviewed_keys: list[str] = []
    for item in candidates[:top_n]:
        key = tree_key(item["tree"])
        if key in llm_reviews:
            continue
        attempted += 1
        prompt = item["filter"].llm_prompt or build_llm_prompt(item["tree"], expert_pool)
        decision = call_openai_tree_review(prompt, model=model, timeout_seconds=timeout_seconds)
        if decision is None:
            continue
        llm_reviews[key] = decision
        reviewed_keys.append(key)
        added += 1
    return {
        "enabled": True,
        "attempted": attempted,
        "added": added,
        "reviewed_keys": reviewed_keys,
        "model": model,
    }


def expert_arrays_for_pair(
    pair: str,
    expert_pool: list[dict[str, Any]],
    route_thresholds: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    threshold_idx = []
    mapping = []
    for expert in expert_pool:
        cfg = expert["pair_configs"][pair]
        threshold_idx.append(route_thresholds.index(float(cfg["route_breadth_threshold"])))
        mapping.append([int(v) for v in cfg["mapping_indices"]])
    return np.asarray(threshold_idx, dtype="int16"), np.asarray(mapping, dtype="int64")


def build_leaf_runtime_arrays_for_pair(
    pair: str,
    leaf_catalog: list[LeafNode],
    expert_pool: list[dict[str, Any]],
    route_thresholds: tuple[float, ...],
    library_size: int,
) -> dict[str, np.ndarray]:
    return build_leaf_runtime_arrays_from_pair_configs(
        [
            {
                "pair_config": expert_pool[int(leaf.expert_idx)]["pair_configs"][pair],
                "gene": leaf.gene,
            }
            for leaf in leaf_catalog
        ],
        route_thresholds,
        library_size,
    )


def build_leaf_runtime_arrays_from_pair_configs(
    leaf_records: list[dict[str, Any]],
    route_thresholds: tuple[float, ...],
    library_size: int,
) -> dict[str, np.ndarray]:
    threshold_idx: list[int] = []
    mapping: list[list[int]] = []
    target_vol_scale: list[float] = []
    gross_cap_scale: list[float] = []
    kill_switch_scale: list[float] = []
    cooldown_scale: list[float] = []
    for record in leaf_records:
        cfg = record["pair_config"]
        gene = record.get("gene", LeafGene())
        base_threshold_idx = route_thresholds.index(float(cfg["route_breadth_threshold"]))
        biased_threshold_idx = min(max(base_threshold_idx + int(gene.route_threshold_bias), 0), len(route_thresholds) - 1)
        threshold_idx.append(biased_threshold_idx)
        shifted_mapping = [
            min(max(int(value) + int(gene.mapping_shift), 0), max(library_size - 1, 0))
            for value in cfg["mapping_indices"]
        ]
        mapping.append(shifted_mapping)
        target_vol_scale.append(float(gene.target_vol_scale))
        gross_cap_scale.append(float(gene.gross_cap_scale))
        kill_switch_scale.append(float(gene.kill_switch_scale))
        cooldown_scale.append(float(gene.cooldown_scale))
    return {
        "threshold_idx": np.asarray(threshold_idx, dtype="int16"),
        "mapping": np.asarray(mapping, dtype="int64"),
        "target_vol_scale": np.asarray(target_vol_scale, dtype="float64"),
        "gross_cap_scale": np.asarray(gross_cap_scale, dtype="float64"),
        "kill_switch_scale": np.asarray(kill_switch_scale, dtype="float64"),
        "cooldown_scale": np.asarray(cooldown_scale, dtype="float64"),
    }


def build_baseline_leaf_runtime_for_pairs(
    baseline_summary: dict[str, Any],
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
    library_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    selected_candidate = baseline_summary.get("selected_candidate", {})
    pair_configs = selected_candidate.get("pair_configs", {})
    out: dict[str, dict[str, np.ndarray]] = {}
    for pair in pairs:
        pair_config = pair_configs.get(pair)
        if pair_config is None:
            raise KeyError(f"Baseline summary does not contain pair config for {pair}")
        out[pair] = build_leaf_runtime_arrays_from_pair_configs(
            [{"pair_config": pair_config}],
            route_thresholds,
            library_size,
        )
    return out


def _fractal_fast_kernel_impl(
    close: np.ndarray,
    bucket_codes_matrix: np.ndarray,
    regime: np.ndarray,
    breadth: np.ndarray,
    vol_ann: np.ndarray,
    smooth_signal_matrix: np.ndarray,
    library_signal_pos: np.ndarray,
    library_rebalance_bars: np.ndarray,
    library_regime_threshold: np.ndarray,
    library_breadth_threshold: np.ndarray,
    library_target_vol_ann: np.ndarray,
    library_gross_cap: np.ndarray,
    library_kill_switch_pct: np.ndarray,
    library_cooldown_days: np.ndarray,
    leaf_threshold_idx: np.ndarray,
    leaf_mapping: np.ndarray,
    leaf_target_vol_scale: np.ndarray,
    leaf_gross_cap_scale: np.ndarray,
    leaf_kill_switch_scale: np.ndarray,
    leaf_cooldown_scale: np.ndarray,
    leaf_codes: np.ndarray,
    initial_cash: float,
    commission_pct: float,
    no_trade_band_pct: float,
    bars_per_day: int,
    daily_target: float,
    bar_factor: float,
) -> tuple[float, int, float, float, float, float, float, float, float, float]:
    equity = initial_cash
    peak_equity = initial_cash
    current_weight = 0.0
    cooldown_bars_left = 0
    n_trades = 0
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
    max_drawdown = 0.0

    for i in range(close.shape[0] - 1):
        leaf = leaf_codes[i]
        threshold_idx = leaf_threshold_idx[leaf]
        bucket = bucket_codes_matrix[threshold_idx, i]
        active_idx = leaf_mapping[leaf, bucket]

        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = smooth_signal_matrix[library_signal_pos[active_idx], i]
        signal_pct = min(500.0, max(-500.0, signal_pct))
        requested_weight = signal_pct / 100.0

        regime_score = regime[i]
        breadth_score = breadth[i]
        long_ok = regime_score >= library_regime_threshold[active_idx] and breadth_score >= library_breadth_threshold[active_idx]
        short_ok = regime_score <= -library_regime_threshold[active_idx] and breadth_score <= (1.0 - library_breadth_threshold[active_idx])
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = vol_ann[i]
        if bar_vol_ann == bar_vol_ann and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = library_target_vol_ann[active_idx] * leaf_target_vol_scale[leaf] / bar_vol_ann
            gross_scale = (library_gross_cap[active_idx] * leaf_gross_cap_scale[leaf]) / max(abs(requested_weight), 1e-8)
            if gross_scale < vol_scale:
                vol_scale = gross_scale
            requested_weight *= vol_scale

        leaf_gross_cap = library_gross_cap[active_idx] * leaf_gross_cap_scale[leaf]
        if requested_weight > leaf_gross_cap:
            requested_weight = leaf_gross_cap
        elif requested_weight < -leaf_gross_cap:
            requested_weight = -leaf_gross_cap

        drawdown = equity / max(peak_equity, 1e-8) - 1.0
        leaf_kill_switch = library_kill_switch_pct[active_idx] * leaf_kill_switch_scale[leaf]
        if drawdown <= -leaf_kill_switch and cooldown_bars_left == 0:
            cooldown_bars_left = max(1, int(round(library_cooldown_days[active_idx] * leaf_cooldown_scale[leaf]))) * bars_per_day

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
        equity *= 1.0 + bar_net
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

        day_accum *= 1.0 + bar_net
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

    sharpe = 0.0
    if bar_count > 1:
        variance = m2_bar / bar_count
        if variance > 1e-12:
            sharpe = mean_bar / np.sqrt(variance) * bar_factor

    avg_daily = 0.0 if day_count == 0 else day_sum / day_count
    daily_target_hit_rate = 0.0 if day_count == 0 else day_hits / day_count
    daily_win_rate = 0.0 if day_count == 0 else day_wins / day_count
    return (
        equity / initial_cash - 1.0,
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
    _fractal_fast_kernel = njit(cache=True)(_fractal_fast_kernel_impl)
else:  # pragma: no cover
    _fractal_fast_kernel = _fractal_fast_kernel_impl


def fast_fractal_replay_from_context(
    context: dict[str, Any],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
    leaf_runtime: dict[str, np.ndarray],
    leaf_codes: np.ndarray,
    commission_multiplier: float = 1.0,
) -> dict[str, Any]:
    bucket_codes_matrix = np.vstack([context["bucket_codes"][float(th)] for th in route_thresholds]).astype("int8")
    result = _fractal_fast_kernel(
        context["close"],
        bucket_codes_matrix,
        context["regime"],
        context["breadth"],
        context["vol_ann"],
        context["smooth_signal_matrix"],
        library_lookup["signal_pos"],
        library_lookup["rebalance_bars"],
        library_lookup["regime_threshold"],
        library_lookup["breadth_threshold"],
        library_lookup["target_vol_ann"],
        library_lookup["gross_cap"],
        library_lookup["kill_switch_pct"],
        library_lookup["cooldown_days"],
        leaf_runtime["threshold_idx"],
        leaf_runtime["mapping"],
        leaf_runtime["target_vol_scale"],
        leaf_runtime["gross_cap_scale"],
        leaf_runtime["kill_switch_scale"],
        leaf_runtime["cooldown_scale"],
        leaf_codes.astype("int16"),
        float(gp.INITIAL_CASH),
        float(gp.COMMISSION_PCT) * float(commission_multiplier),
        float(gp.NO_TRADE_BAND),
        int(BARS_PER_DAY),
        float(gp.DAILY_TARGET_PCT),
        float(BAR_FACTOR),
    )
    return summarize_single_result(
        {
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
    )


def _window_win_rate_stats(window: dict[str, Any]) -> tuple[float, float]:
    values = [float(metrics["daily_win_rate"]) for metrics in window["per_pair"].values()]
    if not values:
        return 0.0, 0.0
    return float(sum(values) / len(values)), float(min(values))


def _window_trade_count(window: dict[str, Any]) -> int:
    return int(sum(int(metrics["n_trades"]) for metrics in window["per_pair"].values()))


def build_baseline_relative_metrics(
    windows: dict[str, Any],
    baseline_windows: dict[str, Any],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key, window in windows.items():
        baseline_window = baseline_windows[key]
        cand_agg = window["aggregate"]
        base_agg = baseline_window["aggregate"]
        cand_mean_win, cand_worst_win = _window_win_rate_stats(window)
        base_mean_win, base_worst_win = _window_win_rate_stats(baseline_window)
        cand_trade_count = _window_trade_count(window)
        base_trade_count = _window_trade_count(baseline_window)
        out[key] = {
            "delta_worst_pair_avg_daily_return": float(cand_agg["worst_pair_avg_daily_return"] - base_agg["worst_pair_avg_daily_return"]),
            "delta_mean_avg_daily_return": float(cand_agg["mean_avg_daily_return"] - base_agg["mean_avg_daily_return"]),
            "delta_worst_pair_total_return": float(cand_agg["worst_pair_total_return"] - base_agg["worst_pair_total_return"]),
            "delta_mean_total_return": float(cand_agg["mean_total_return"] - base_agg["mean_total_return"]),
            "delta_worst_max_drawdown": float(abs(base_agg["worst_max_drawdown"]) - abs(cand_agg["worst_max_drawdown"])),
            "delta_pair_return_dispersion": float(base_agg["pair_return_dispersion"] - cand_agg["pair_return_dispersion"]),
            "delta_mean_daily_win_rate": float(cand_mean_win - base_mean_win),
            "delta_worst_daily_win_rate": float(cand_worst_win - base_worst_win),
            "trade_count_ratio": float(cand_trade_count / max(base_trade_count, 1)),
            "candidate_trade_count": float(cand_trade_count),
            "baseline_trade_count": float(base_trade_count),
        }
    return out


def relative_gate_pass(relative: dict[str, float], tolerance: float = 1e-12) -> bool:
    return bool(
        relative["delta_worst_pair_total_return"] >= -tolerance
        and relative["delta_worst_max_drawdown"] >= -tolerance
        and relative["delta_worst_daily_win_rate"] >= -tolerance
    )


def candidate_wf1_pass(item: dict[str, Any], tolerance: float = 1e-12) -> bool:
    robustness = item.get("robustness", {})
    return bool(
        float(robustness.get("latest_fold_delta_worst_pair_total_return", 0.0)) >= -tolerance
        and float(robustness.get("latest_fold_delta_worst_max_drawdown", 0.0)) >= -tolerance
        and float(robustness.get("latest_fold_delta_worst_daily_win_rate", 0.0)) >= -tolerance
    )


def candidate_stress_pass(item: dict[str, Any], tolerance: float = 1e-12) -> bool:
    robustness = item.get("robustness", {})
    threshold = float(robustness.get("stress_survival_threshold", 0.0))
    mean_survival = float(robustness.get("stress_survival_rate_mean", 0.0))
    min_survival = float(
        robustness.get("min_fold_stress_survival_rate", robustness.get("stress_survival_rate_min", 0.0))
    )
    reserve_score = float(robustness.get("latest_fold_stress_reserve_score", 0.0))
    return bool(
        mean_survival >= threshold - tolerance
        and min_survival >= threshold - tolerance
        and reserve_score >= -tolerance
    )


def candidate_cost_reserve_pass(item: dict[str, Any], tolerance: float = 1e-12) -> bool:
    robustness = item.get("robustness", {})
    threshold = float(robustness.get("stress_survival_threshold", 0.67))
    latest_rate = float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0))
    min_rate = float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0))
    latest_reserve = float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0))
    return bool(
        latest_rate >= threshold - tolerance
        and min_rate >= threshold - tolerance
        and latest_reserve >= -tolerance
    )


def defensive_leaf_gene_variant(gene: LeafGene, profile: str) -> LeafGene:
    if profile == "ultra":
        return LeafGene(
            route_threshold_bias=max(int(gene.route_threshold_bias), 2),
            mapping_shift=0,
            target_vol_scale=min(float(gene.target_vol_scale), 0.55),
            gross_cap_scale=min(float(gene.gross_cap_scale), 0.55),
            kill_switch_scale=min(float(gene.kill_switch_scale), 0.70),
            cooldown_scale=max(float(gene.cooldown_scale), 2.00),
        )
    if profile == "max_defense":
        return LeafGene(
            route_threshold_bias=max(int(gene.route_threshold_bias), 2),
            mapping_shift=0,
            target_vol_scale=min(float(gene.target_vol_scale), 0.40),
            gross_cap_scale=min(float(gene.gross_cap_scale), 0.40),
            kill_switch_scale=min(float(gene.kill_switch_scale), 0.70),
            cooldown_scale=max(float(gene.cooldown_scale), 2.50),
        )
    if profile == "strong":
        return LeafGene(
            route_threshold_bias=max(int(gene.route_threshold_bias), 1),
            mapping_shift=0,
            target_vol_scale=min(float(gene.target_vol_scale), 0.70),
            gross_cap_scale=min(float(gene.gross_cap_scale), 0.70),
            kill_switch_scale=min(float(gene.kill_switch_scale), 0.85),
            cooldown_scale=max(float(gene.cooldown_scale), 1.50),
        )
    return LeafGene(
        route_threshold_bias=max(int(gene.route_threshold_bias), 1),
        mapping_shift=0 if abs(int(gene.mapping_shift)) > 4 else int(gene.mapping_shift),
        target_vol_scale=min(float(gene.target_vol_scale), 0.85),
        gross_cap_scale=min(float(gene.gross_cap_scale), 0.85),
        kill_switch_scale=min(float(gene.kill_switch_scale), 0.85),
        cooldown_scale=max(float(gene.cooldown_scale), 1.25),
    )


def apply_leaf_gene_profile(node: TreeNode, profile: str) -> TreeNode:
    if isinstance(node, LeafNode):
        return LeafNode(node.expert_idx, defensive_leaf_gene_variant(getattr(node, "gene", LeafGene()), profile))
    assert isinstance(node, ConditionNode)
    return ConditionNode(
        condition=node.condition,
        if_true=apply_leaf_gene_profile(node.if_true, profile),
        if_false=apply_leaf_gene_profile(node.if_false, profile),
    )


def evaluate_defensive_variants(
    ranked: list[dict[str, Any]],
    evaluate_tree_fn: Any,
    candidate_limit: int = 12,
) -> list[dict[str, Any]]:
    seen = {item["tree_key"] for item in ranked}
    variants: list[dict[str, Any]] = []
    for item in ranked[:candidate_limit]:
        for profile in ("mild", "strong", "ultra", "max_defense"):
            variant_tree = apply_leaf_gene_profile(item["tree"], profile)
            variant_key = tree_key(variant_tree)
            if variant_key in seen:
                continue
            seen.add(variant_key)
            variants.append(evaluate_tree_fn(variant_tree))
    return variants


def summarize_robustness_folds(
    folds: list[dict[str, Any]],
    stress_survival_threshold: float,
) -> dict[str, Any]:
    def stress_run_delta_score(relative: dict[str, float]) -> float:
        score = 0.0
        score += float(relative["delta_worst_pair_avg_daily_return"]) * 640000.0
        score += float(relative["delta_mean_avg_daily_return"]) * 360000.0
        score += float(relative["delta_worst_pair_total_return"]) * 2400.0
        score += float(relative["delta_worst_max_drawdown"]) * 26000.0
        score += float(relative["delta_worst_daily_win_rate"]) * 26000.0
        return float(score)

    def summarize_stress_runs(stress_runs: list[dict[str, Any]]) -> dict[str, Any]:
        if not stress_runs:
            return {
                "count": 0,
                "pass_rate": 0.0,
                "reserve_score": 0.0,
                "by_multiplier": {},
            }
        by_multiplier: dict[str, dict[str, Any]] = {}
        scores: list[float] = []
        for run in stress_runs:
            multiplier_key = f"{float(run['commission_multiplier']):.1f}"
            summary = by_multiplier.setdefault(
                multiplier_key,
                {
                    "count": 0,
                    "pass_count": 0,
                    "delta_worst_pair_avg_daily_return": [],
                    "delta_mean_avg_daily_return": [],
                    "delta_worst_pair_total_return": [],
                    "delta_worst_max_drawdown": [],
                    "delta_worst_daily_win_rate": [],
                    "scores": [],
                },
            )
            relative = run["relative"]
            summary["count"] += 1
            summary["pass_count"] += 1 if bool(run["passed"]) else 0
            summary["delta_worst_pair_avg_daily_return"].append(float(relative["delta_worst_pair_avg_daily_return"]))
            summary["delta_mean_avg_daily_return"].append(float(relative["delta_mean_avg_daily_return"]))
            summary["delta_worst_pair_total_return"].append(float(relative["delta_worst_pair_total_return"]))
            summary["delta_worst_max_drawdown"].append(float(relative["delta_worst_max_drawdown"]))
            summary["delta_worst_daily_win_rate"].append(float(relative["delta_worst_daily_win_rate"]))
            run_score = stress_run_delta_score(relative)
            summary["scores"].append(run_score)
            scores.append(run_score)
        normalized: dict[str, Any] = {}
        for multiplier_key, summary in by_multiplier.items():
            count = max(int(summary["count"]), 1)
            normalized[multiplier_key] = {
                "count": int(summary["count"]),
                "pass_rate": float(summary["pass_count"] / count),
                "mean_delta_worst_pair_avg_daily_return": float(sum(summary["delta_worst_pair_avg_daily_return"]) / count),
                "mean_delta_mean_avg_daily_return": float(sum(summary["delta_mean_avg_daily_return"]) / count),
                "mean_delta_worst_pair_total_return": float(sum(summary["delta_worst_pair_total_return"]) / count),
                "mean_delta_worst_max_drawdown": float(sum(summary["delta_worst_max_drawdown"]) / count),
                "mean_delta_worst_daily_win_rate": float(sum(summary["delta_worst_daily_win_rate"]) / count),
                "min_delta_worst_pair_avg_daily_return": float(min(summary["delta_worst_pair_avg_daily_return"])),
                "min_delta_mean_avg_daily_return": float(min(summary["delta_mean_avg_daily_return"])),
                "min_delta_worst_pair_total_return": float(min(summary["delta_worst_pair_total_return"])),
                "min_delta_worst_max_drawdown": float(min(summary["delta_worst_max_drawdown"])),
                "min_delta_worst_daily_win_rate": float(min(summary["delta_worst_daily_win_rate"])),
                "mean_score": float(sum(summary["scores"]) / count),
                "min_score": float(min(summary["scores"])),
            }
        return {
            "count": len(stress_runs),
            "pass_rate": float(sum(1 for run in stress_runs if bool(run["passed"])) / len(stress_runs)),
            "reserve_score": float(min(scores)),
            "by_multiplier": normalized,
        }

    if not folds:
        return {
            "folds": [],
            "fold_pass_rate": 0.0,
            "stress_survival_rate_mean": 0.0,
            "stress_survival_rate_min": 0.0,
            "min_fold_stress_survival_rate": 0.0,
            "non_nominal_stress_survival_rate_mean": 0.0,
            "min_fold_non_nominal_stress_survival_rate": 0.0,
            "stress_survival_threshold": float(stress_survival_threshold),
            "stress_run_summary": {
                "count": 0,
                "pass_rate": 0.0,
                "reserve_score": 0.0,
                "by_multiplier": {},
            },
            "latest_fold_stress_run_summary": {
                "count": 0,
                "pass_rate": 0.0,
                "reserve_score": 0.0,
                "by_multiplier": {},
            },
            "latest_fold_stress_reserve_score": 0.0,
            "latest_non_nominal_stress_survival_rate": 0.0,
            "latest_non_nominal_stress_reserve_score": 0.0,
            "worst_fold_delta_worst_pair_avg_daily_return": 0.0,
            "worst_fold_delta_worst_pair_total_return": 0.0,
            "worst_fold_delta_worst_max_drawdown": 0.0,
            "worst_fold_delta_worst_daily_win_rate": 0.0,
            "latest_fold_delta_worst_pair_total_return": 0.0,
            "latest_fold_delta_worst_pair_avg_daily_return": 0.0,
            "latest_fold_delta_mean_avg_daily_return": 0.0,
            "latest_fold_delta_worst_max_drawdown": 0.0,
            "latest_fold_delta_worst_daily_win_rate": 0.0,
            "latest_fold_trade_count_ratio": 0.0,
            "mean_fold_trade_count_ratio": 0.0,
            "min_fold_trade_count_ratio": 0.0,
            "wf_1": None,
            "gate_passed": False,
        }

    relatives = [fold["relative"] for fold in folds]
    latest_fold = next((fold for fold in reversed(folds) if str(fold.get("fold")) == "wf_1"), folds[-1])
    latest = latest_fold["relative"]
    stress_run_summary = summarize_stress_runs([run for fold in folds for run in fold.get("stress", [])])
    latest_fold_stress_run_summary = summarize_stress_runs(list(latest_fold.get("stress", [])))
    fold_pass_rate = float(sum(1 for fold in folds if fold["passed"]) / len(folds))
    stress_survival_rate_mean = float(sum(float(fold["stress_survival_rate"]) for fold in folds) / len(folds))
    stress_survival_rate_min = float(min(float(fold["stress_survival_rate"]) for fold in folds))
    non_nominal_fold_pass_rates: list[float] = []
    non_nominal_fold_reserve_scores: list[float] = []
    latest_non_nominal_runs = [
        run
        for run in latest_fold.get("stress", [])
        if float(run.get("commission_multiplier", 1.0)) > 1.0
    ]
    latest_non_nominal_stress_run_summary = summarize_stress_runs(latest_non_nominal_runs)
    for fold in folds:
        non_nominal_runs = [
            run
            for run in fold.get("stress", [])
            if float(run.get("commission_multiplier", 1.0)) > 1.0
        ]
        non_nominal_summary = summarize_stress_runs(non_nominal_runs)
        non_nominal_fold_pass_rates.append(float(non_nominal_summary["pass_rate"]))
        non_nominal_fold_reserve_scores.append(float(non_nominal_summary["reserve_score"]))
    latest_fold_stress_reserve_score = float(latest_fold_stress_run_summary["reserve_score"])
    return {
        "folds": folds,
        "fold_pass_rate": fold_pass_rate,
        "stress_survival_rate_mean": stress_survival_rate_mean,
        "stress_survival_rate_min": stress_survival_rate_min,
        "min_fold_stress_survival_rate": stress_survival_rate_min,
        "non_nominal_stress_survival_rate_mean": float(
            sum(non_nominal_fold_pass_rates) / len(non_nominal_fold_pass_rates)
        ) if non_nominal_fold_pass_rates else 0.0,
        "min_fold_non_nominal_stress_survival_rate": float(
            min(non_nominal_fold_pass_rates)
        ) if non_nominal_fold_pass_rates else 0.0,
        "stress_survival_threshold": float(stress_survival_threshold),
        "stress_run_summary": stress_run_summary,
        "latest_fold_stress_run_summary": latest_fold_stress_run_summary,
        "latest_fold_stress_reserve_score": latest_fold_stress_reserve_score,
        "latest_non_nominal_stress_survival_rate": float(latest_non_nominal_stress_run_summary["pass_rate"]),
        "latest_non_nominal_stress_reserve_score": float(latest_non_nominal_stress_run_summary["reserve_score"]),
        "worst_fold_delta_worst_pair_avg_daily_return": float(min(rel["delta_worst_pair_avg_daily_return"] for rel in relatives)),
        "worst_fold_delta_worst_pair_total_return": float(min(rel["delta_worst_pair_total_return"] for rel in relatives)),
        "worst_fold_delta_worst_max_drawdown": float(min(rel["delta_worst_max_drawdown"] for rel in relatives)),
        "worst_fold_delta_worst_daily_win_rate": float(min(rel["delta_worst_daily_win_rate"] for rel in relatives)),
        "latest_fold_delta_worst_pair_avg_daily_return": float(latest["delta_worst_pair_avg_daily_return"]),
        "latest_fold_delta_worst_pair_total_return": float(latest["delta_worst_pair_total_return"]),
        "latest_fold_delta_mean_avg_daily_return": float(latest["delta_mean_avg_daily_return"]),
        "latest_fold_delta_worst_max_drawdown": float(latest["delta_worst_max_drawdown"]),
        "latest_fold_delta_worst_daily_win_rate": float(latest["delta_worst_daily_win_rate"]),
        "latest_fold_trade_count_ratio": float(latest["trade_count_ratio"]),
        "mean_fold_trade_count_ratio": float(sum(float(rel["trade_count_ratio"]) for rel in relatives) / len(relatives)),
        "min_fold_trade_count_ratio": float(min(float(rel["trade_count_ratio"]) for rel in relatives)),
        "wf_1": {
            "fold": latest_fold["fold"],
            "start": latest_fold["start"],
            "end": latest_fold["end"],
            "passed": bool(latest_fold["passed"]),
            "stress_survival_rate": float(latest_fold["stress_survival_rate"]),
            "stress_reserve_score": latest_fold_stress_reserve_score,
            "non_nominal_stress_survival_rate": float(latest_non_nominal_stress_run_summary["pass_rate"]),
            "non_nominal_stress_reserve_score": float(latest_non_nominal_stress_run_summary["reserve_score"]),
            "delta_worst_pair_avg_daily_return": float(latest["delta_worst_pair_avg_daily_return"]),
            "delta_worst_pair_total_return": float(latest["delta_worst_pair_total_return"]),
            "delta_worst_max_drawdown": float(latest["delta_worst_max_drawdown"]),
            "delta_worst_daily_win_rate": float(latest["delta_worst_daily_win_rate"]),
            "trade_count_ratio": float(latest["trade_count_ratio"]),
        },
        "gate_passed": bool(
            all(bool(fold["passed"]) for fold in folds)
            and all(float(fold["stress_survival_rate"]) >= float(stress_survival_threshold) for fold in folds)
        ),
    }


def leaf_gene_deviation_score(leaf_catalog: list[LeafNode]) -> float:
    if not leaf_catalog:
        return 0.0
    total = 0.0
    for leaf in leaf_catalog:
        gene = getattr(leaf, "gene", LeafGene())
        total += (
            0.85 * abs(int(gene.route_threshold_bias))
            + 0.16 * abs(int(gene.mapping_shift)) / 8.0
            + 1.15 * abs(float(gene.target_vol_scale) - 1.0) / 0.15
            + 1.05 * abs(float(gene.gross_cap_scale) - 1.0) / 0.15
            + 1.35 * abs(float(gene.kill_switch_scale) - 1.0) / 0.15
            + 0.95 * abs(float(gene.cooldown_scale) - 1.0) / 0.25
        )
    return float(total / len(leaf_catalog))


def fractal_fast_scalar_score(
    windows: dict[str, Any],
    baseline_windows: dict[str, Any],
    filter_decision: FilterDecision,
    node: TreeNode,
    robustness: dict[str, Any],
    leaf_gene_penalty: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    recent_2m_key, recent_6m_key, full_key = score_window_labels(windows)
    relative = build_baseline_relative_metrics(windows, baseline_windows)
    rel_2m = relative[recent_2m_key]
    rel_6m = relative[recent_6m_key]
    rel_4y = relative[full_key]
    score = 0.0
    score += rel_2m["delta_worst_pair_avg_daily_return"] * 320000.0
    score += rel_6m["delta_worst_pair_avg_daily_return"] * 240000.0
    score += rel_4y["delta_worst_pair_avg_daily_return"] * 180000.0
    score += rel_4y["delta_mean_avg_daily_return"] * 160000.0
    score += rel_2m["delta_mean_avg_daily_return"] * 50000.0
    score += rel_6m["delta_mean_avg_daily_return"] * 38000.0
    score += rel_2m["delta_worst_pair_total_return"] * 1800.0
    score += rel_6m["delta_worst_pair_total_return"] * 1200.0
    score += rel_4y["delta_worst_pair_total_return"] * 60.0
    score += rel_2m["delta_worst_max_drawdown"] * 20000.0
    score += rel_6m["delta_worst_max_drawdown"] * 15000.0
    score += rel_4y["delta_worst_max_drawdown"] * 11000.0
    score += rel_2m["delta_pair_return_dispersion"] * 50000.0
    score += rel_6m["delta_pair_return_dispersion"] * 35000.0
    score += rel_4y["delta_pair_return_dispersion"] * 25000.0
    score += rel_2m["delta_worst_daily_win_rate"] * 18000.0
    score += rel_6m["delta_worst_daily_win_rate"] * 14000.0
    score += rel_4y["delta_worst_daily_win_rate"] * 10000.0
    score += float(robustness["worst_fold_delta_worst_pair_avg_daily_return"]) * 220000.0
    score += float(robustness["worst_fold_delta_worst_pair_total_return"]) * 1600.0
    score += float(robustness["worst_fold_delta_worst_max_drawdown"]) * 16000.0
    score += float(robustness["worst_fold_delta_worst_daily_win_rate"]) * 14000.0
    score += float(robustness["latest_fold_delta_worst_pair_avg_daily_return"]) * 640000.0
    score += float(robustness["latest_fold_delta_mean_avg_daily_return"]) * 360000.0
    score += float(robustness["latest_fold_delta_worst_pair_total_return"]) * 2400.0
    score += float(robustness["latest_fold_delta_worst_max_drawdown"]) * 26000.0
    score += float(robustness["latest_fold_delta_worst_daily_win_rate"]) * 26000.0
    score += float(robustness["latest_fold_stress_reserve_score"]) * 1.0
    score += float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0)) * 2.2
    score += float(robustness["stress_survival_rate_mean"]) * 3200.0
    score += float(robustness["stress_survival_rate_min"]) * 5200.0
    score += float(robustness.get("non_nominal_stress_survival_rate_mean", 0.0)) * 6800.0
    score += float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)) * 9200.0
    score += (
        float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0))
        - float(robustness["stress_survival_threshold"])
    ) * 18000.0
    score += (
        float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0))
        - float(robustness["stress_survival_threshold"])
    ) * 22000.0
    score += float(robustness["fold_pass_rate"]) * 1000.0
    score += (
        float(robustness["stress_survival_rate_mean"]) - float(robustness["stress_survival_threshold"])
    ) * 9000.0
    score += (
        float(robustness["stress_survival_rate_min"]) - float(robustness["stress_survival_threshold"])
    ) * 12000.0
    score += (float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0)) - 0.5) * 12000.0
    score += (float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)) - 0.5) * 14000.0
    if float(robustness["fold_pass_rate"]) < 1.0:
        score -= (1.0 - float(robustness["fold_pass_rate"])) * 3500.0
    if float(robustness["latest_fold_stress_reserve_score"]) < 0.0:
        score -= abs(float(robustness["latest_fold_stress_reserve_score"])) * 4.5
    if float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0)) < 0.0:
        score -= abs(float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0))) * 8.0
    if float(robustness["stress_survival_rate_mean"]) < float(robustness["stress_survival_threshold"]):
        score -= (
            float(robustness["stress_survival_threshold"]) - float(robustness["stress_survival_rate_mean"])
        ) * 9000.0
    if float(robustness["stress_survival_rate_min"]) < float(robustness["stress_survival_threshold"]):
        score -= (
            float(robustness["stress_survival_threshold"]) - float(robustness["stress_survival_rate_min"])
        ) * 14000.0
    if float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0)) < 0.5:
        score -= (0.5 - float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0))) * 16000.0
    if float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)) < 0.5:
        score -= (0.5 - float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0))) * 18000.0
    if float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0)) < float(robustness["stress_survival_threshold"]):
        score -= (
            float(robustness["stress_survival_threshold"])
            - float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0))
        ) * 32000.0
    if float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)) < float(robustness["stress_survival_threshold"]):
        score -= (
            float(robustness["stress_survival_threshold"])
            - float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0))
        ) * 42000.0
    if float(robustness["latest_fold_delta_worst_pair_total_return"]) < 0.010:
        score -= (0.010 - float(robustness["latest_fold_delta_worst_pair_total_return"])) * 3800.0
    if float(robustness["latest_fold_delta_worst_daily_win_rate"]) < 0.010:
        score -= (0.010 - float(robustness["latest_fold_delta_worst_daily_win_rate"])) * 32000.0
    if float(robustness["latest_fold_delta_worst_pair_avg_daily_return"]) < 0.0:
        score -= abs(float(robustness["latest_fold_delta_worst_pair_avg_daily_return"])) * 520000.0
    if float(robustness["worst_fold_delta_worst_pair_total_return"]) < 0.0:
        score -= abs(float(robustness["worst_fold_delta_worst_pair_total_return"])) * 3000.0
    if float(robustness["latest_fold_delta_worst_pair_total_return"]) < 0.0:
        score -= abs(float(robustness["latest_fold_delta_worst_pair_total_return"])) * 2500.0
    if float(robustness["latest_fold_delta_worst_max_drawdown"]) < 0.0:
        score -= abs(float(robustness["latest_fold_delta_worst_max_drawdown"])) * 28000.0
    if float(robustness["latest_fold_delta_worst_daily_win_rate"]) < 0.0:
        score -= abs(float(robustness["latest_fold_delta_worst_daily_win_rate"])) * 22000.0
    if float(robustness["latest_fold_trade_count_ratio"]) > 1.0:
        score -= (float(robustness["latest_fold_trade_count_ratio"]) - 1.0) * 26000.0
    if float(robustness["mean_fold_trade_count_ratio"]) > 1.0:
        score -= (float(robustness["mean_fold_trade_count_ratio"]) - 1.0) * 14000.0
    if float(robustness["latest_fold_trade_count_ratio"]) > 0.9:
        score -= (float(robustness["latest_fold_trade_count_ratio"]) - 0.9) * 36000.0
    if float(robustness["mean_fold_trade_count_ratio"]) > 0.9:
        score -= (float(robustness["mean_fold_trade_count_ratio"]) - 0.9) * 22000.0
    for rel in (rel_2m, rel_6m, rel_4y):
        if rel["trade_count_ratio"] < 0.05:
            score -= (0.05 - rel["trade_count_ratio"]) * 150000.0
        if rel["trade_count_ratio"] > 1.0:
            score -= (rel["trade_count_ratio"] - 1.0) * 18000.0
    score -= tree_size(node) * 120.0
    score -= max(0, tree_depth(node) - 2) * 180.0
    score -= tree_logic_size(node) * 45.0
    score -= max(0, tree_logic_depth(node) - 1) * 90.0
    score -= float(leaf_gene_penalty) * 1800.0
    if not filter_decision.accepted:
        score -= 10_000_000.0
    return float(score), relative


def structural_bonus_from_metrics(
    tree_depth_value: int,
    logic_depth_value: int,
    leaf_cardinality: int,
    condition_count: int,
) -> float:
    bonus = 0.0
    bonus += max(0, tree_depth_value - 1) * 0.34
    bonus += max(0, logic_depth_value) * 0.28
    bonus += max(0, leaf_cardinality - 1) * 0.16
    bonus += max(0, condition_count - 1) * 0.09
    return bonus


def structural_bonus(node: TreeNode) -> float:
    depth = tree_depth(node)
    logic_depth_value = tree_logic_depth(node)
    leaf_cardinality = len(set(collect_leaf_keys(node)))
    condition_count = len(collect_specs(node))
    return structural_bonus_from_metrics(depth, logic_depth_value, leaf_cardinality, condition_count)


def find_condition_spec(
    condition_options: list[ConditionSpec],
    feature: str,
    threshold: float,
    comparator: str = ">=",
    invert: bool = False,
) -> ConditionSpec:
    for spec in condition_options:
        if (
            spec.feature == feature
            and spec.comparator == comparator
            and float(spec.threshold) == float(threshold)
            and bool(spec.invert) == bool(invert)
        ):
            return copy.deepcopy(spec)
    for spec in condition_options:
        if spec.feature == feature and spec.comparator == comparator and bool(spec.invert) == bool(invert):
            return copy.deepcopy(spec)
    return copy.deepcopy(condition_options[0])


def threshold(
    condition_options: list[ConditionSpec],
    feature: str,
    value: float,
    comparator: str = ">=",
    invert: bool = False,
) -> ThresholdCell:
    return ThresholdCell(spec=find_condition_spec(condition_options, feature, value, comparator=comparator, invert=invert))


def build_seed_trees(
    expert_pool: list[dict[str, Any]],
    condition_options: list[ConditionSpec],
    pairs: tuple[str, ...],
) -> list[TreeNode]:
    def named_threshold(feature: str, value: float, comparator: str = ">=", invert: bool = False) -> ThresholdCell:
        return threshold(condition_options, feature, value, comparator=comparator, invert=invert)

    seeds: list[TreeNode] = []
    top_count = min(4, len(expert_pool))
    for idx in range(top_count):
        seeds.append(LeafNode(idx))
    if len(expert_pool) >= 2:
        seeds.append(
            ConditionNode(
                condition=named_threshold("btc_regime", 0.0, comparator=">="),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            )
        )
        seeds.append(
            ConditionNode(
                condition=AndCell(
                    left=named_threshold("btc_rsi_14d", 55.0, comparator=">="),
                    right=named_threshold("btc_volume_z_7d", 0.0, comparator=">="),
                ),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            )
        )
        seeds.append(
            ConditionNode(
                condition=OrCell(
                    left=named_threshold("session_us_flag", 0.5, comparator=">="),
                    right=named_threshold("btc_atr_pct_14d", 0.02, comparator="<="),
                ),
                if_true=LeafNode(1),
                if_false=LeafNode(0),
            )
        )
        seeds.append(
            ConditionNode(
                condition=AndCell(
                    left=named_threshold("btc_macd_hist_12_26_9", 0.0, comparator=">="),
                    right=named_threshold("btc_bb_pct_b_20_2", 0.50, comparator=">="),
                ),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            )
        )
        seeds.append(
            ConditionNode(
                condition=OrCell(
                    left=named_threshold("btc_mfi_14d", 50.0, comparator=">="),
                    right=named_threshold("btc_cci_20d", 0.0, comparator=">="),
                ),
                if_true=LeafNode(1),
                if_false=LeafNode(0),
            )
        )
        seeds.append(
            ConditionNode(
                condition=AndCell(
                    left=named_threshold("btc_rsi_14_1h", 55.0, comparator=">="),
                    right=named_threshold("btc_volume_rel_1h", 1.0, comparator=">="),
                ),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            )
        )
        seeds.append(
            ConditionNode(
                condition=OrCell(
                    left=named_threshold("btc_dc_trend_05_1h", 0.5, comparator=">="),
                    right=named_threshold("btc_macd_h_pct_1h", 0.0, comparator=">="),
                ),
                if_true=LeafNode(1),
                if_false=LeafNode(0),
            )
        )
    if len(expert_pool) >= 4 and len(pairs) > 1:
        seeds.append(
            ConditionNode(
                condition=OrCell(
                    left=named_threshold("rsi_spread_btc_minus_bnb_14d", 0.0, comparator=">="),
                    right=named_threshold("volume_z_spread_btc_minus_bnb_7d", 0.0, comparator=">="),
                ),
                if_true=ConditionNode(
                    condition=AndCell(
                        left=named_threshold("bnb_regime", 0.0, comparator=">="),
                        right=NotCell(child=named_threshold("bnb_drawdown_7d", -0.10, comparator="<=")),
                    ),
                    if_true=LeafNode(0),
                    if_false=LeafNode(2),
                ),
                if_false=ConditionNode(
                    condition=AndCell(
                        left=named_threshold("btc_rsi_14d", 60.0, comparator=">="),
                        right=named_threshold("btc_atr_pct_14d", 0.03, comparator="<="),
                    ),
                    if_true=LeafNode(1),
                    if_false=LeafNode(3),
                ),
            )
        )
        seeds.append(
            ConditionNode(
                condition=AndCell(
                    left=named_threshold("macd_hist_spread_btc_minus_bnb_12_26_9", 0.0, comparator=">="),
                    right=named_threshold("bb_pct_b_spread_btc_minus_bnb_20_2", 0.0, comparator=">="),
                ),
                if_true=ConditionNode(
                    condition=OrCell(
                        left=named_threshold("btc_dc_pos_20d", 0.75, comparator=">="),
                        right=named_threshold("bnb_dc_pos_20d", 0.75, comparator=">="),
                    ),
                    if_true=LeafNode(0),
                    if_false=LeafNode(2),
                ),
                if_false=ConditionNode(
                    condition=AndCell(
                        left=named_threshold("return_spread_btc_minus_bnb_3d", 0.0, comparator=">="),
                        right=named_threshold("mfi_spread_btc_minus_bnb_14d", 0.0, comparator=">="),
                    ),
                    if_true=LeafNode(1),
                    if_false=LeafNode(3),
                ),
            )
        )
        seeds.append(
            ConditionNode(
                condition=AndCell(
                    left=named_threshold("macd_h_pct_spread_btc_minus_bnb_1h", 0.0, comparator=">="),
                    right=named_threshold("volume_rel_spread_btc_minus_bnb_1h", 0.0, comparator=">="),
                ),
                if_true=ConditionNode(
                    condition=OrCell(
                        left=named_threshold("bnb_rsi_14_1h", 55.0, comparator=">="),
                        right=named_threshold("bnb_dc_trend_05_1h", 0.5, comparator=">="),
                    ),
                    if_true=LeafNode(0),
                    if_false=LeafNode(2),
                ),
                if_false=ConditionNode(
                    condition=AndCell(
                        left=named_threshold("btc_intraday_return_6h", 0.0, comparator=">="),
                        right=named_threshold("dc_overshoot_spread_btc_minus_bnb_1h", 0.0, comparator=">="),
                    ),
                    if_true=LeafNode(1),
                    if_false=LeafNode(3),
                ),
            )
        )
    elif len(expert_pool) >= 4:
        seeds.append(
            ConditionNode(
                condition=OrCell(
                    left=named_threshold("btc_momentum_3d", 0.0, comparator=">="),
                    right=named_threshold("session_eu_flag", 0.5, comparator=">="),
                ),
                if_true=ConditionNode(
                    condition=AndCell(
                        left=named_threshold("btc_rsi_14d", 50.0, comparator=">="),
                        right=named_threshold("btc_volume_z_7d", 0.0, comparator=">="),
                    ),
                    if_true=LeafNode(0),
                    if_false=LeafNode(2),
                ),
                if_false=ConditionNode(
                    condition=AndCell(
                        left=named_threshold("btc_atr_pct_14d", 0.02, comparator="<="),
                        right=named_threshold("session_asia_flag", 0.5, comparator=">="),
                    ),
                    if_true=LeafNode(1),
                    if_false=LeafNode(3),
                ),
            )
        )
        seeds.append(
            ConditionNode(
                condition=AndCell(
                    left=named_threshold("btc_rsi_14_1h", 55.0, comparator=">="),
                    right=named_threshold("btc_dc_trend_05_1h", 0.5, comparator=">="),
                ),
                if_true=ConditionNode(
                    condition=OrCell(
                        left=named_threshold("btc_volume_rel_1h", 1.0, comparator=">="),
                        right=named_threshold("btc_intraday_return_6h", 0.0, comparator=">="),
                    ),
                    if_true=LeafNode(0),
                    if_false=LeafNode(2),
                ),
                if_false=ConditionNode(
                    condition=AndCell(
                        left=named_threshold("btc_atr_pct_1h", 0.003, comparator="<="),
                        right=named_threshold("btc_intraday_drawdown_24h", -0.03, comparator="<="),
                    ),
                    if_true=LeafNode(1),
                    if_false=LeafNode(3),
                ),
            )
        )
    return seeds


def tournament_select(population: list[dict[str, Any]], rng: random.Random, k: int = 3) -> TreeNode:
    sample = rng.sample(population, k=min(k, len(population)))
    sample.sort(key=lambda item: item["search_fitness"], reverse=True)
    return copy.deepcopy(sample[0]["tree"])


def export_llm_review_queue(path: str | None, candidates: list[dict[str, Any]], expert_pool: list[dict[str, Any]]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for item in candidates:
        tree = item["tree"]
        lines.append(
            json.dumps(
                {
                    "tree_key": tree_key(tree),
                    "tree": serialize_tree(tree),
                    "llm_prompt": item["filter"].llm_prompt or build_llm_prompt(tree, expert_pool),
                    "accepted": item["filter"].accepted,
                    "reason": item["filter"].reason,
                    "source": item["filter"].source,
                },
                ensure_ascii=False,
            )
        )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> None:
    args = parse_args()
    started = perf_counter()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    pairs = parse_csv_tuple(args.pairs, str)
    route_thresholds = parse_csv_tuple(args.route_thresholds, float)
    _ = resolve_fast_engine(args.fast_engine)

    expert_summary_paths = [part.strip() for part in args.expert_summaries.split(",") if part.strip()]
    expert_pool, expert_pool_diagnostics = build_expert_pool(expert_summary_paths, args.expert_pool_size, pairs)
    if len(expert_pool) < 2:
        raise RuntimeError("Need at least 2 experts to build recursive fractal trees.")

    baseline_summary = json.loads(Path(args.baseline_summary).read_text())
    llm_reviews = load_llm_review_map(args.llm_review_in)
    feature_specs = build_feature_specs(pairs)
    stress_values = parse_csv_tuple(args.commission_stress, float)

    library = list(iter_params())
    library_lookup = build_library_lookup(library)
    model, _ = load_model(Path(args.model))
    compiled = gp.toolbox.compile(expr=model)

    df_all = gp.load_all_pairs(pairs=list(pairs), start=None, end=None, refresh_cache=False)
    if df_all.empty:
        raise RuntimeError(f"No cached OHLCV data available for pairs={pairs}.")
    window_specs = compute_dynamic_windows(pd.DatetimeIndex(df_all.index))
    robustness_specs = build_walk_forward_ranges(
        pd.DatetimeIndex(df_all.index),
        args.robustness_folds,
        args.robustness_test_months,
    )
    baseline_windows = build_baseline_windows_for_pairs(
        baseline_summary["selected_candidate"]["windows"],
        pairs,
        window_specs,
    )
    baseline_runtime_arrays = build_baseline_leaf_runtime_for_pairs(
        baseline_summary,
        pairs,
        route_thresholds,
        len(library),
    )
    start_all = window_specs[-1]["start"]
    end_all = window_specs[-1]["end"]
    df_all = df_all.loc[start_all:end_all].copy()
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {
        pair: load_funding_from_cache_or_empty(pair, start_all, end_all)
        for pair in pairs
    }

    prepare_started = perf_counter()
    condition_options = build_condition_options(feature_specs)
    window_cache: dict[str, dict[str, Any]] = {}
    for spec in window_specs:
        label = spec["key"]
        start = spec["start"]
        end = spec["end"]
        df = df_all.loc[start:end].copy()
        feature_arrays = materialize_feature_arrays(build_market_features(df, pairs), pd.DatetimeIndex(df.index))
        pair_cache: dict[str, Any] = {}
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
                "fast_context": build_fast_context(
                    df=df,
                    pair=pair,
                    raw_signal=signal_slice,
                    overlay_inputs=overlay_inputs,
                    route_thresholds=route_thresholds,
                    library_lookup=library_lookup,
                    funding_df=funding_slice,
                ),
            }
        window_cache[label] = {
            "features": feature_arrays,
            "pair_cache": pair_cache,
            "bars": int(len(df)),
            "start": start,
            "end": end,
            "label": spec["label"],
            "description": spec["description"],
        }
    robustness_cache: dict[str, dict[str, Any]] = {}
    for spec in robustness_specs:
        label = spec["key"]
        start = spec["start"]
        end = spec["end"]
        df = df_all.loc[start:end].copy()
        if df.empty:
            continue
        feature_arrays = materialize_feature_arrays(build_market_features(df, pairs), pd.DatetimeIndex(df.index))
        pair_cache: dict[str, Any] = {}
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
                "fast_context": build_fast_context(
                    df=df,
                    pair=pair,
                    raw_signal=signal_slice,
                    overlay_inputs=overlay_inputs,
                    route_thresholds=route_thresholds,
                    library_lookup=library_lookup,
                    funding_df=funding_slice,
                ),
            }
        robustness_cache[label] = {
            "features": feature_arrays,
            "pair_cache": pair_cache,
            "bars": int(len(df)),
            "start": start,
            "end": end,
            "label": spec["label"],
            "description": spec["description"],
        }
    prepare_seconds = perf_counter() - prepare_started

    seed_trees = build_seed_trees(expert_pool, condition_options, pairs)
    population: list[TreeNode] = seed_trees[:]
    initial_tree_depth_budget = curriculum_budget(
        0,
        args.generations,
        args.curriculum_min_depth,
        args.max_depth,
    )
    initial_logic_depth_budget = curriculum_budget(
        0,
        args.generations,
        args.curriculum_min_logic_depth,
        args.logic_max_depth,
    )
    while len(population) < args.population:
        population.append(
            random_tree(
                rng,
                condition_options,
                len(expert_pool),
                initial_tree_depth_budget,
                logic_max_depth=initial_logic_depth_budget,
            )
        )

    fast_cache: dict[str, dict[str, Any]] = {}
    auto_llm_review_events: list[dict[str, Any]] = []
    generation_selection_diagnostics: list[dict[str, Any]] = []
    immigrant_injection_diagnostics: list[dict[str, Any]] = []

    def evaluate_tree(tree: TreeNode) -> dict[str, Any]:
        key = tree_key(tree)
        cached = fast_cache.get(key)
        if cached is not None:
            return cached
        filter_decision = semantic_filter(tree, expert_pool, args.max_depth, args.filter_mode, llm_reviews=llm_reviews)
        tree_depth_value = tree_depth(tree)
        tree_logic_depth_value = tree_logic_depth(tree)
        tree_size_value = tree_size(tree)
        tree_logic_size_value = tree_logic_size(tree)
        leaf_signature = _tree_leaf_signature(tree)
        condition_count = len(collect_specs(tree))
        reference_features = window_cache[window_specs[0]["key"]]["features"]
        _, leaf_catalog = evaluate_tree_leaf_codes(tree, reference_features)
        leaf_gene_penalty = leaf_gene_deviation_score(leaf_catalog)
        leaf_runtime_arrays = {
            pair: build_leaf_runtime_arrays_for_pair(
                pair,
                leaf_catalog,
                expert_pool,
                route_thresholds,
                len(library),
            )
            for pair in pairs
        }
        windows: dict[str, Any] = {}
        for spec in window_specs:
            label = spec["key"]
            window_state = window_cache[label]
            leaf_codes, _ = evaluate_tree_leaf_codes(tree, window_state["features"])
            per_pair = {}
            for pair in pairs:
                per_pair[pair] = fast_fractal_replay_from_context(
                    window_state["pair_cache"][pair]["fast_context"],
                    library_lookup,
                    route_thresholds,
                    leaf_runtime_arrays[pair],
                    leaf_codes,
                )
            windows[label] = {
                "window_label": window_state["label"],
                "window_description": window_state["description"],
                "start": window_state["start"],
                "end": window_state["end"],
                "bars": window_state["bars"],
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
        robustness_folds: list[dict[str, Any]] = []
        for spec in robustness_specs:
            label = spec["key"]
            fold_state = robustness_cache.get(label)
            if fold_state is None:
                continue
            leaf_codes, _ = evaluate_tree_leaf_codes(tree, fold_state["features"])
            per_pair = {}
            baseline_per_pair = {}
            stress_runs: list[dict[str, Any]] = []
            for pair in pairs:
                per_pair[pair] = fast_fractal_replay_from_context(
                    fold_state["pair_cache"][pair]["fast_context"],
                    library_lookup,
                    route_thresholds,
                    leaf_runtime_arrays[pair],
                    leaf_codes,
                )
                baseline_codes = np.zeros(len(leaf_codes), dtype="int16")
                baseline_per_pair[pair] = fast_fractal_replay_from_context(
                    fold_state["pair_cache"][pair]["fast_context"],
                    library_lookup,
                    route_thresholds,
                    baseline_runtime_arrays[pair],
                    baseline_codes,
                )
            candidate_window = {
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
            baseline_window = {
                "per_pair": baseline_per_pair,
                "aggregate": aggregate_metrics(baseline_per_pair),
            }
            relative = build_baseline_relative_metrics({"fold": candidate_window}, {"fold": baseline_window})["fold"]
            for multiplier in stress_values:
                stressed_per_pair = {}
                for pair in pairs:
                    stressed_per_pair[pair] = fast_fractal_replay_from_context(
                        fold_state["pair_cache"][pair]["fast_context"],
                        library_lookup,
                        route_thresholds,
                        leaf_runtime_arrays[pair],
                        leaf_codes,
                        commission_multiplier=float(multiplier),
                    )
                stressed_window = {
                    "per_pair": stressed_per_pair,
                    "aggregate": aggregate_metrics(stressed_per_pair),
                }
                stress_relative = build_baseline_relative_metrics({"fold": stressed_window}, {"fold": baseline_window})["fold"]
                stress_runs.append(
                    {
                        "commission_multiplier": float(multiplier),
                        "relative": stress_relative,
                        "passed": relative_gate_pass(stress_relative),
                    }
                )
            stress_survival_rate = float(
                sum(1 for item in stress_runs if bool(item["passed"])) / max(len(stress_runs), 1)
            )
            robustness_folds.append(
                {
                    "fold": label,
                    "start": fold_state["start"],
                    "end": fold_state["end"],
                    "relative": relative,
                    "passed": relative_gate_pass(relative),
                    "stress_survival_rate": stress_survival_rate,
                    "stress": stress_runs,
                }
            )
        robustness = summarize_robustness_folds(robustness_folds, args.stress_survival_threshold)
        validation = build_validation_bundle(windows, baseline_windows)
        fitness, baseline_relative = fractal_fast_scalar_score(
            windows,
            baseline_windows,
            filter_decision,
            tree,
            robustness,
            leaf_gene_penalty,
        )
        cached = {
            "tree": copy.deepcopy(tree),
            "tree_key": key,
            "filter": filter_decision,
            "windows": windows,
            "validation": validation,
            "fitness": fitness,
            "baseline_relative": baseline_relative,
            "robustness": robustness,
            "tree_depth": tree_depth_value,
            "logic_depth": tree_logic_depth_value,
            "tree_size": tree_size_value,
            "logic_size": tree_logic_size_value,
            "condition_count": condition_count,
            "leaf_signature": leaf_signature,
            "leaf_cardinality": len(leaf_signature),
            "leaf_gene_penalty": float(leaf_gene_penalty),
            "structural_score": structural_bonus_from_metrics(
                tree_depth_value,
                tree_logic_depth_value,
                len(leaf_signature),
                condition_count,
            ),
            "performance_score": float(fitness),
        }
        cached["search_fitness"] = float(cached["fitness"]) + 30.0 * float(cached["structural_score"])
        fast_cache[key] = cached
        return cached

    search_started = perf_counter()
    curriculum_schedule: list[dict[str, int]] = []
    for generation_idx in range(args.generations):
        generation_tree_depth_budget = curriculum_budget(
            generation_idx,
            args.generations,
            args.curriculum_min_depth,
            args.max_depth,
        )
        generation_logic_depth_budget = curriculum_budget(
            generation_idx,
            args.generations,
            args.curriculum_min_logic_depth,
            args.logic_max_depth,
        )
        curriculum_schedule.append(
            {
                "generation": generation_idx,
                "tree_depth_budget": generation_tree_depth_budget,
                "logic_depth_budget": generation_logic_depth_budget,
            }
        )
        evaluated = [evaluate_tree(tree) for tree in population]
        evaluated.sort(key=lambda item: item["search_fitness"], reverse=True)
        llm_review_event = auto_review_top_candidates(
            evaluated,
            expert_pool,
            llm_reviews,
            top_n=args.auto_llm_review_top_n,
            model=args.auto_llm_review_model,
            timeout_seconds=args.auto_llm_review_timeout_seconds,
        )
        llm_review_event["generation"] = generation_idx
        auto_llm_review_events.append(llm_review_event)
        reviewed_keys = llm_review_event.get("reviewed_keys", [])
        if reviewed_keys:
            for key in reviewed_keys:
                fast_cache.pop(key, None)
            evaluated = [evaluate_tree(tree) for tree in population]
            evaluated.sort(key=lambda item: item["search_fitness"], reverse=True)
        survivors, survivor_diag = select_generation_survivors(
            evaluated,
            args.elite_count,
            args.survivor_diversity_weight,
            args.survivor_depth_weight,
            target_tree_depth=generation_tree_depth_budget,
            target_logic_depth=generation_logic_depth_budget,
        )
        next_population = [copy.deepcopy(item["tree"]) for item in survivors]
        immigrant_count = 0
        if args.immigrant_rate > 0.0 and len(next_population) < args.population:
            immigrant_count = min(
                args.population - len(next_population),
                max(1, int(round(args.population * args.immigrant_rate))),
            )
            immigrant_tree_budget = max(1, min(args.max_depth, generation_tree_depth_budget + 1))
            immigrant_logic_budget = max(1, min(args.logic_max_depth, generation_logic_depth_budget + 1))
            for _ in range(immigrant_count):
                next_population.append(
                    random_tree(
                        rng,
                        condition_options,
                        len(expert_pool),
                        immigrant_tree_budget,
                        logic_max_depth=immigrant_logic_budget,
                    )
                )
        immigrant_injection_diagnostics.append(
            {
                "generation": generation_idx,
                "requested_rate": args.immigrant_rate,
                "injected_count": immigrant_count,
                "survivor_count": len(next_population) - immigrant_count,
                "tree_budget": generation_tree_depth_budget,
                "logic_budget": generation_logic_depth_budget,
                "immigrant_tree_budget": max(1, min(args.max_depth, generation_tree_depth_budget + 1)),
                "immigrant_logic_budget": max(1, min(args.logic_max_depth, generation_logic_depth_budget + 1)),
            }
        )
        parent_pool = survivors if survivors else evaluated
        while len(next_population) < args.population:
            parent_a = tournament_select(parent_pool, rng)
            if rng.random() < 0.65:
                parent_b = tournament_select(parent_pool, rng)
                child_a, child_b = crossover_tree(parent_a, parent_b, rng)
                candidate = child_a if rng.random() < 0.5 else child_b
            else:
                candidate = parent_a
            if rng.random() < 0.70:
                candidate = mutate_tree(
                    candidate,
                    rng,
                    condition_options,
                    len(expert_pool),
                    generation_tree_depth_budget,
                    logic_max_depth=generation_logic_depth_budget,
                )
            next_population.append(candidate)
        population = next_population[: args.population]
        generation_selection_diagnostics.append(
            {
                "generation": generation_idx,
                "tree_depth_budget": generation_tree_depth_budget,
                "logic_depth_budget": generation_logic_depth_budget,
                "survivor_selection": survivor_diag,
                "immigrant_injection": immigrant_injection_diagnostics[-1],
                "population_after_selection": len(next_population),
                "population_final": len(population),
            }
        )
    search_seconds = perf_counter() - search_started

    evaluated = [evaluate_tree(tree) for tree in population]
    ranked = sorted({tree_key(item["tree"]): item for item in evaluated}.values(), key=lambda item: item["search_fitness"], reverse=True)
    defensive_variants = evaluate_defensive_variants(ranked, evaluate_tree)
    if defensive_variants:
        ranked = sorted(
            {item["tree_key"]: item for item in [*ranked, *defensive_variants]}.values(),
            key=lambda item: item["search_fitness"],
            reverse=True,
        )
    top_candidates = ranked[: args.top_k]
    cost_reserve_candidates = [
        item
        for item in ranked
        if candidate_wf1_pass(item) and candidate_cost_reserve_pass(item)
    ]
    cost_reserve_archive_candidate = max(
        cost_reserve_candidates,
        key=lambda item: (
            int(item["tree_depth"]) >= 2,
            int(item["logic_depth"]) >= 2,
            float(item["robustness"].get("latest_non_nominal_stress_reserve_score", 0.0)),
            float(item["robustness"].get("latest_fold_stress_reserve_score", 0.0)),
            float(item["performance_score"]),
            float(item["search_fitness"]),
            item["tree_key"],
        ),
    ) if cost_reserve_candidates else None
    persistent_gate_candidates = [
        item
        for item in ranked
        if (int(item["tree_depth"]) >= 2 or int(item["logic_depth"]) >= 2)
        and candidate_wf1_pass(item)
        and candidate_stress_pass(item)
    ]
    strict_persistent_archive_candidate = max(
        persistent_gate_candidates,
        key=lambda item: (
            int(item["tree_depth"]) >= 3,
            int(item["logic_depth"]) >= 2,
            float(item["performance_score"]),
            float(item["search_fitness"]),
            float(item["structural_score"]),
            item["tree_key"],
        ),
    ) if persistent_gate_candidates else None
    persistent_candidates = [
        item
        for item in ranked
        if int(item["tree_depth"]) >= 3 or int(item["logic_depth"]) >= 2
    ]
    persistent_archive_candidate = max(
        persistent_candidates,
        key=lambda item: (
            int(item["tree_depth"]) >= 3,
            int(item["logic_depth"]) >= 2,
            1 if candidate_cost_reserve_pass(item) else 0,
            float(item["performance_score"]),
            float(item["search_fitness"]),
            float(item["structural_score"]),
            item["tree_key"],
        ),
    ) if persistent_candidates else None
    persistent_archive_injected_into_top_k = False
    if cost_reserve_archive_candidate is not None and cost_reserve_archive_candidate["tree_key"] not in {item["tree_key"] for item in top_candidates}:
        replacement_idx = min(
            range(len(top_candidates)),
            key=lambda idx: (
                float(top_candidates[idx]["search_fitness"]),
                float(top_candidates[idx]["performance_score"]),
                float(top_candidates[idx]["structural_score"]),
                top_candidates[idx]["tree_key"],
            ),
        ) if top_candidates else None
        if replacement_idx is not None:
            top_candidates[replacement_idx] = cost_reserve_archive_candidate
            top_candidates = sorted(
                {item["tree_key"]: item for item in top_candidates}.values(),
                key=lambda item: item["search_fitness"],
                reverse=True,
            )[: args.top_k]
            persistent_archive_injected_into_top_k = True
    elif strict_persistent_archive_candidate is not None and strict_persistent_archive_candidate["tree_key"] not in {item["tree_key"] for item in top_candidates}:
        replacement_idx = min(
            range(len(top_candidates)),
            key=lambda idx: (
                float(top_candidates[idx]["search_fitness"]),
                float(top_candidates[idx]["performance_score"]),
                float(top_candidates[idx]["structural_score"]),
                top_candidates[idx]["tree_key"],
            ),
        ) if top_candidates else None
        if replacement_idx is not None:
            top_candidates[replacement_idx] = strict_persistent_archive_candidate
            top_candidates = sorted(
                {item["tree_key"]: item for item in top_candidates}.values(),
                key=lambda item: item["search_fitness"],
                reverse=True,
            )[: args.top_k]
            persistent_archive_injected_into_top_k = True
    elif persistent_archive_candidate is not None and persistent_archive_candidate["tree_key"] not in {item["tree_key"] for item in top_candidates}:
        best_top_performance = max(float(item["performance_score"]) for item in top_candidates) if top_candidates else float(persistent_archive_candidate["performance_score"])
        persistent_frontier_band = max(250.0, abs(best_top_performance) * 0.080)
        if top_candidates and (
            best_top_performance - float(persistent_archive_candidate["performance_score"]) <= persistent_frontier_band
            or bool(persistent_archive_candidate["robustness"]["gate_passed"])
        ):
            replacement_idx = min(
                range(len(top_candidates)),
                key=lambda idx: (
                    float(top_candidates[idx]["search_fitness"]),
                    float(top_candidates[idx]["performance_score"]),
                    float(top_candidates[idx]["structural_score"]),
                    top_candidates[idx]["tree_key"],
                ),
            )
            top_candidates[replacement_idx] = persistent_archive_candidate
            top_candidates = sorted(
                {item["tree_key"]: item for item in top_candidates}.values(),
                key=lambda item: item["search_fitness"],
                reverse=True,
            )[: args.top_k]
            persistent_archive_injected_into_top_k = True
    progressive_candidates = [item for item in top_candidates if item["validation"]["profiles"]["progressive_improvement"]["passed"]]
    target_candidates = [item for item in top_candidates if item["validation"]["profiles"]["target_060"]["passed"]]
    robust_candidates = [item for item in top_candidates if bool(item["robustness"]["gate_passed"])]
    fallback_best = max(top_candidates, key=lambda item: float(item["performance_score"])) if top_candidates else None
    winner_pool = target_candidates or progressive_candidates or robust_candidates or top_candidates
    if cost_reserve_archive_candidate is not None:
        winner_pool = [item for item in winner_pool if item["tree_key"] != cost_reserve_archive_candidate["tree_key"]]
        winner_pool.append(cost_reserve_archive_candidate)
    elif strict_persistent_archive_candidate is not None:
        winner_pool = [item for item in winner_pool if item["tree_key"] != strict_persistent_archive_candidate["tree_key"]]
        winner_pool.append(strict_persistent_archive_candidate)
    elif persistent_archive_candidate is not None:
        winner_pool = [item for item in winner_pool if item["tree_key"] != persistent_archive_candidate["tree_key"]]
        winner_pool.append(persistent_archive_candidate)
    selected, selection_diagnostics = select_near_frontier_structural_winner(winner_pool)
    structural_champion_forced_into_top_k = False
    if selected is not None and selected["tree_key"] not in {item["tree_key"] for item in top_candidates}:
        top_candidate_frontier = float(selection_diagnostics.get("frontier_band", 0.0))
        best_top_performance = max(float(item["performance_score"]) for item in top_candidates) if top_candidates else float(selected["performance_score"])
        if top_candidates and best_top_performance - float(selected["performance_score"]) <= top_candidate_frontier:
            replacement_idx = min(
                range(len(top_candidates)),
                key=lambda idx: (
                    float(top_candidates[idx]["search_fitness"]),
                    float(top_candidates[idx]["performance_score"]),
                    float(top_candidates[idx]["structural_score"]),
                    top_candidates[idx]["tree_key"],
                ),
            )
            top_candidates[replacement_idx] = selected
            top_candidates = sorted(
                {item["tree_key"]: item for item in top_candidates}.values(),
                key=lambda item: item["search_fitness"],
                reverse=True,
            )[: args.top_k]
            structural_champion_forced_into_top_k = True
    selection_reason = "target_060_near_frontier_structural_pass"
    if not target_candidates and progressive_candidates:
        selection_reason = "progressive_near_frontier_structural_pass"
    elif not target_candidates and not progressive_candidates and robust_candidates:
        selection_reason = "robustness_gate_near_frontier_structural_pass"
    elif not target_candidates and not progressive_candidates and not robust_candidates:
        selection_reason = "no_gate_pass_near_frontier_structural"

    export_llm_review_queue(args.llm_review_out, top_candidates, expert_pool)
    feature_set = describe_feature_set(feature_specs)

    report = {
        "search": {
            "algorithm": "fractal_genome_fast_stage",
            "population": args.population,
            "generations": args.generations,
            "elite_count": args.elite_count,
            "max_depth": args.max_depth,
            "logic_max_depth": args.logic_max_depth,
            "curriculum_min_depth": args.curriculum_min_depth,
            "curriculum_min_logic_depth": args.curriculum_min_logic_depth,
            "seed": args.seed,
            "expert_pool_size": len(expert_pool),
            "filter_mode": args.filter_mode,
            "expert_summaries": expert_summary_paths,
            "model": str(args.model),
            "route_thresholds": [float(v) for v in route_thresholds],
            "llm_review_in": args.llm_review_in,
            "llm_review_out": args.llm_review_out,
            "auto_llm_review_top_n": args.auto_llm_review_top_n,
            "auto_llm_review_model": args.auto_llm_review_model,
            "survivor_diversity_weight": args.survivor_diversity_weight,
            "survivor_depth_weight": args.survivor_depth_weight,
            "immigrant_rate": args.immigrant_rate,
            "robustness_folds": args.robustness_folds,
            "robustness_test_months": args.robustness_test_months,
            "commission_stress": [float(v) for v in stress_values],
            "stress_survival_threshold": float(args.stress_survival_threshold),
            "feature_set": {
                "name": FEATURE_SET_NAME,
                "description": FEATURE_SET_DESCRIPTION,
                "condition_option_count": len(condition_options),
                "feature_count": len(feature_specs),
                "features": feature_set["features"],
                "feature_context": {
                    "primary_pair": pairs[0],
                    "secondary_pair": pairs[1] if len(pairs) > 1 else pairs[0],
                    "single_asset_mode": len(pairs) == 1,
                },
            },
        },
        "pairs": list(pairs),
        "backtest_windows": window_specs,
        "robustness_windows": robustness_specs,
        "curriculum_schedule": curriculum_schedule,
        "auto_llm_review_events": auto_llm_review_events,
        "expert_pool_diagnostics": expert_pool_diagnostics,
        "generation_selection_diagnostics": generation_selection_diagnostics,
        "immigrant_injection_diagnostics": immigrant_injection_diagnostics,
        "baseline_summary_path": str(args.baseline_summary),
        "baseline_candidate": baseline_summary["selected_candidate"],
        "expert_pool": expert_pool,
        "top_candidates": [
            {
                "tree_key": item["tree_key"],
                "tree": serialize_tree(item["tree"]),
                "fitness": float(item["fitness"]),
                "search_fitness": float(item["search_fitness"]),
                "performance_score": float(item["performance_score"]),
                "structural_score": float(item["structural_score"]),
                "tree_depth": int(tree_depth(item["tree"])),
                "tree_size": int(tree_size(item["tree"])),
                "logic_cell_depth": int(tree_logic_depth(item["tree"])),
                "logic_cell_count": int(tree_logic_size(item["tree"])),
                "condition_count": int(len(collect_specs(item["tree"]))),
                "leaf_cardinality": int(len(set(collect_leaf_keys(item["tree"])))),
                "leaf_gene_penalty": float(item["leaf_gene_penalty"]),
                "filter": {
                    "accepted": item["filter"].accepted,
                    "source": item["filter"].source,
                    "reason": item["filter"].reason,
                },
                "baseline_relative": item["baseline_relative"],
                "robustness": item["robustness"],
                "windows": item["windows"],
                "validation": item["validation"],
            }
            for item in top_candidates
        ],
        "promotion_candidates": {
            "target_060": [{"tree": serialize_tree(item["tree"])} for item in target_candidates],
            "progressive_improvement": [{"tree": serialize_tree(item["tree"])} for item in progressive_candidates],
            "robustness_gate": [{"tree": serialize_tree(item["tree"])} for item in robust_candidates],
        },
        "selection": {
            "reason": selection_reason,
            "target_060_pass_count": len(target_candidates),
            "progressive_pass_count": len(progressive_candidates),
            "robustness_gate_pass_count": len(robust_candidates),
            "top_k_wf1_pass_count": sum(1 for item in top_candidates if candidate_wf1_pass(item)),
            "top_k_stress_pass_count": sum(1 for item in top_candidates if candidate_stress_pass(item)),
            "top_k_cost_reserve_pass_count": sum(1 for item in top_candidates if candidate_cost_reserve_pass(item)),
            "top_k_stress_reserve_scores": [float(item["robustness"].get("latest_fold_stress_reserve_score", 0.0)) for item in top_candidates],
            "stress_run_summary": top_candidates[0]["robustness"].get("stress_run_summary", {}) if top_candidates else {},
            "latest_fold_stress_run_summary": top_candidates[0]["robustness"].get("latest_fold_stress_run_summary", {}) if top_candidates else {},
            "cost_reserve_archive_candidate": None if cost_reserve_archive_candidate is None else {
                "tree_key": cost_reserve_archive_candidate["tree_key"],
                "tree_depth": int(cost_reserve_archive_candidate["tree_depth"]),
                "logic_depth": int(cost_reserve_archive_candidate["logic_depth"]),
                "performance_score": float(cost_reserve_archive_candidate["performance_score"]),
                "stress_reserve_score": float(cost_reserve_archive_candidate["robustness"].get("latest_non_nominal_stress_reserve_score", 0.0)),
            },
            "persistent_archive_candidate": None if persistent_archive_candidate is None else {
                "tree_key": persistent_archive_candidate["tree_key"],
                "tree_depth": int(persistent_archive_candidate["tree_depth"]),
                "logic_depth": int(persistent_archive_candidate["logic_depth"]),
                "performance_score": float(persistent_archive_candidate["performance_score"]),
                "structural_score": float(persistent_archive_candidate["structural_score"]),
                "stress_reserve_score": float(persistent_archive_candidate["robustness"].get("latest_fold_stress_reserve_score", 0.0)),
            },
            "strict_persistent_archive_candidate": None if strict_persistent_archive_candidate is None else {
                "tree_key": strict_persistent_archive_candidate["tree_key"],
                "tree_depth": int(strict_persistent_archive_candidate["tree_depth"]),
                "logic_depth": int(strict_persistent_archive_candidate["logic_depth"]),
                "performance_score": float(strict_persistent_archive_candidate["performance_score"]),
                "structural_score": float(strict_persistent_archive_candidate["structural_score"]),
                "stress_reserve_score": float(strict_persistent_archive_candidate["robustness"].get("latest_fold_stress_reserve_score", 0.0)),
            },
            "persistent_archive_injected_into_top_k": persistent_archive_injected_into_top_k,
            "structural_champion_forced_into_top_k": structural_champion_forced_into_top_k,
            "selection_diagnostics": selection_diagnostics,
        },
        "fallback_best_candidate": None if fallback_best is None else {
            "tree": serialize_tree(fallback_best["tree"]),
            "fitness": float(fallback_best["fitness"]),
            "search_fitness": float(fallback_best["search_fitness"]),
            "performance_score": float(fallback_best["performance_score"]),
            "tree_depth": int(tree_depth(fallback_best["tree"])),
            "tree_size": int(tree_size(fallback_best["tree"])),
            "logic_cell_depth": int(tree_logic_depth(fallback_best["tree"])),
            "logic_cell_count": int(tree_logic_size(fallback_best["tree"])),
            "condition_count": int(len(collect_specs(fallback_best["tree"]))),
            "leaf_cardinality": int(len(set(collect_leaf_keys(fallback_best["tree"])))),
            "leaf_gene_penalty": float(fallback_best["leaf_gene_penalty"]),
            "structural_score": float(fallback_best["structural_score"]),
            "baseline_relative": fallback_best["baseline_relative"],
            "robustness": fallback_best["robustness"],
            "windows": fallback_best["windows"],
            "validation": fallback_best["validation"],
            "filter": {
                "accepted": fallback_best["filter"].accepted,
                "source": fallback_best["filter"].source,
                "reason": fallback_best["filter"].reason,
            },
        },
        "selected_candidate": None if selected is None else {
            "tree": serialize_tree(selected["tree"]),
            "fitness": float(selected["fitness"]),
            "search_fitness": float(selected["search_fitness"]),
            "performance_score": float(selected["performance_score"]),
            "structural_score": float(selected["structural_score"]),
            "tree_depth": int(tree_depth(selected["tree"])),
            "tree_size": int(tree_size(selected["tree"])),
            "logic_cell_depth": int(tree_logic_depth(selected["tree"])),
            "logic_cell_count": int(tree_logic_size(selected["tree"])),
            "condition_count": int(len(collect_specs(selected["tree"]))),
            "leaf_cardinality": int(len(set(collect_leaf_keys(selected["tree"])))),
            "leaf_gene_penalty": float(selected["leaf_gene_penalty"]),
            "baseline_relative": selected["baseline_relative"],
            "robustness": selected["robustness"],
            "windows": selected["windows"],
            "validation": selected["validation"],
            "filter": {
                "accepted": selected["filter"].accepted,
                "source": selected["filter"].source,
                "reason": selected["filter"].reason,
            },
        },
        "single_asset_backtest": None,
        "runtime": {
            "prepare_context_seconds": prepare_seconds,
            "search_seconds": search_seconds,
            "total_seconds": perf_counter() - started,
            "evaluated_unique_candidates": len(fast_cache),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }

    if len(pairs) == 1:
        pair = pairs[0]
        report["single_asset_backtest"] = {
            "pair": pair,
            "mode": "single_asset",
            "selected_candidate": None if selected is None else {
                key: value["per_pair"][pair]
                for key, value in selected["windows"].items()
            },
            "fallback_best_candidate": None if fallback_best is None else {
                key: value["per_pair"][pair]
                for key, value in fallback_best["windows"].items()
            },
        }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report["selection"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
