#!/usr/bin/env python3
"""First-stage fractal genome search over recursive If-Then-Else trees."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import pandas as pd

from derivative_market_data import load_derivative_bundle, slice_derivative_bundle
import gp_crypto_evolution as gp
from fractal_genome_core import (
    AndCell,
    ConditionNode,
    ConditionSpec,
    deserialize_tree,
    evaluate_logic_cell,
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
    serialize_logic,
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
    normalize_mapping_indices,
    normalize_route_state_mode,
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

FEATURE_SET_NAME = "fractal_market_feature_v4"
FEATURE_SET_DESCRIPTION = (
    "Expanded single-asset and pairwise inputs covering returns, momentum, "
    "RSI, ATR, MACD, Bollinger, MFI, CCI, Donchian, drawdown, volatility, "
    "volume, multi-threshold directional-change, order-imbalance, futures "
    "positioning, basis, session, and cross-asset spread features."
)
WINDOW_COMPAT_LABEL_FULL = "full_4y"
OBSERVATION_MODE_TIME = "time"
OBSERVATION_MODE_VOLUME = "volume"
OBSERVATION_MODE_IMBALANCE = "imbalance"
OBSERVATION_MODE_DIRECTIONAL_CHANGE = "directional_change"
OBSERVATION_MODE_ORDER = (
    OBSERVATION_MODE_TIME,
    OBSERVATION_MODE_VOLUME,
    OBSERVATION_MODE_IMBALANCE,
    OBSERVATION_MODE_DIRECTIONAL_CHANGE,
)
OBSERVATION_MODE_LABELS = {
    OBSERVATION_MODE_TIME: "time-driven",
    OBSERVATION_MODE_VOLUME: "volume-driven",
    OBSERVATION_MODE_IMBALANCE: "order-imbalance-driven",
    OBSERVATION_MODE_DIRECTIONAL_CHANGE: "directional-change-driven",
}
LABEL_HORIZON_ORDER = ("1m", "5m", "30m", "4h")
LABEL_HORIZON_LABELS = {
    "1m": "fast-proxy",
    "5m": "native-5m",
    "30m": "30m-decision-cadence",
    "4h": "4h-decision-cadence",
}
LABEL_HORIZON_BAR_COUNTS = {
    "1m": 1,
    "5m": 1,
    "30m": 6,
    "4h": 48,
}
COMMON_OBSERVATION_FEATURES = {
    "btc_regime",
    "bnb_regime",
    "regime_spread_btc_minus_bnb",
    "breadth",
    "breadth_change_1d",
}
DERIVATIVE_FEATURE_TOKENS = (
    "oi_rel",
    "basis_rate",
    "top_pos_log_ratio",
    "top_acct_log_ratio",
    "global_ls_log_ratio",
    "taker_buy_sell_log_ratio",
)

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
    ("btc_order_imbalance_1h", ">=", (-0.60, -0.30, 0.0, 0.30, 0.60)),
    ("btc_oi_rel_1h", ">=", (0.75, 0.90, 1.00, 1.10, 1.25)),
    ("btc_basis_rate_1h", ">=", (-0.0020, -0.0010, 0.0, 0.0010, 0.0020)),
    ("btc_top_pos_log_ratio_1h", ">=", (-0.08, -0.04, 0.0, 0.04, 0.08)),
    ("btc_top_acct_log_ratio_1h", ">=", (-0.15, -0.08, 0.0, 0.08, 0.15)),
    ("btc_global_ls_log_ratio_1h", ">=", (-0.15, -0.08, 0.0, 0.08, 0.15)),
    ("btc_taker_buy_sell_log_ratio_1h", ">=", (-0.40, -0.15, 0.0, 0.15, 0.40)),
    ("btc_cci_scaled_1h", ">=", (-1.50, -0.75, -0.25, 0.0, 0.25, 0.75, 1.50)),
    ("btc_dc_trend_015_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_event_015_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_trend_03_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_event_03_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_trend_05_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_event_05_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_overshoot_05_1h", ">=", (-0.015, -0.005, 0.0, 0.005, 0.015)),
    ("btc_dc_run_05_1h", ">=", (0.0, 0.01, 0.02, 0.05)),
    ("btc_dc_trend_10_1h", ">=", (-0.50, 0.50)),
    ("btc_dc_event_10_1h", ">=", (-0.50, 0.50)),
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
    ("bnb_order_imbalance_1h", ">=", (-0.60, -0.30, 0.0, 0.30, 0.60)),
    ("bnb_oi_rel_1h", ">=", (0.75, 0.90, 1.00, 1.10, 1.25)),
    ("bnb_basis_rate_1h", ">=", (-0.0020, -0.0010, 0.0, 0.0010, 0.0020)),
    ("bnb_top_pos_log_ratio_1h", ">=", (0.0, 0.04, 0.08, 0.12, 0.16)),
    ("bnb_top_acct_log_ratio_1h", ">=", (0.60, 0.70, 0.80, 0.90)),
    ("bnb_global_ls_log_ratio_1h", ">=", (0.60, 0.70, 0.80, 0.90)),
    ("bnb_taker_buy_sell_log_ratio_1h", ">=", (-0.40, -0.15, 0.0, 0.15, 0.40)),
    ("bnb_cci_scaled_1h", ">=", (-1.50, -0.75, -0.25, 0.0, 0.25, 0.75, 1.50)),
    ("bnb_dc_trend_015_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_event_015_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_trend_03_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_event_03_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_trend_05_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_event_05_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_overshoot_05_1h", ">=", (-0.015, -0.005, 0.0, 0.005, 0.015)),
    ("bnb_dc_run_05_1h", ">=", (0.0, 0.01, 0.02, 0.05)),
    ("bnb_dc_trend_10_1h", ">=", (-0.50, 0.50)),
    ("bnb_dc_event_10_1h", ">=", (-0.50, 0.50)),
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
    ("imbalance_spread_btc_minus_bnb_1h", ">=", (-1.0, -0.50, 0.0, 0.50, 1.0)),
    ("oi_rel_spread_btc_minus_bnb_1h", ">=", (-0.60, -0.25, 0.0, 0.25, 0.60)),
    ("basis_rate_spread_btc_minus_bnb_1h", ">=", (-0.0030, -0.0015, 0.0, 0.0015, 0.0030)),
    ("top_pos_log_ratio_spread_btc_minus_bnb_1h", ">=", (-0.10, -0.06, -0.03, 0.0, 0.03)),
    ("top_acct_log_ratio_spread_btc_minus_bnb_1h", ">=", (-1.00, -0.80, -0.60, -0.40, -0.20, 0.0)),
    ("global_ls_log_ratio_spread_btc_minus_bnb_1h", ">=", (-1.00, -0.80, -0.60, -0.40, -0.20, 0.0)),
    ("taker_buy_sell_log_ratio_spread_btc_minus_bnb_1h", ">=", (-0.50, -0.20, 0.0, 0.20, 0.50)),
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
    parser.add_argument(
        "--warm-start-summaries",
        default="",
        help="Optional comma-separated prior fractal summary JSON files used to warm-start local search.",
    )
    parser.add_argument(
        "--warm-start-candidate-limit",
        type=int,
        default=12,
        help="Maximum incumbent trees imported from prior fractal summaries per mode/horizon.",
    )
    parser.add_argument(
        "--warm-start-variant-budget",
        type=int,
        default=24,
        help="Maximum local neighbor variants generated from imported incumbent trees per mode/horizon.",
    )
    parser.add_argument(
        "--local-search-rate",
        type=float,
        default=0.35,
        help="Fraction of offspring generated from local incumbent mutations instead of generic breeding.",
    )
    parser.add_argument(
        "--local-search-mutation-burst",
        type=int,
        default=2,
        help="Maximum sequential mutation count applied during local incumbent search.",
    )
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
        "--observation-modes",
        default="time,volume,imbalance,directional_change",
        help="Observation families searched in parallel.",
    )
    parser.add_argument(
        "--label-horizons",
        default="1m,5m,30m,4h",
        help="Decision horizons searched in parallel on top of 5m execution data.",
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
    parser.add_argument(
        "--fetch-derivatives",
        action="store_true",
        help="Refresh optional Binance futures derivatives caches before search.",
    )
    parser.add_argument(
        "--enable-derivatives",
        action="store_true",
        help="Allow derivatives features into the search space. Default keeps main-equivalent price/indicator features only.",
    )
    parser.add_argument(
        "--disable-derivatives",
        action="store_true",
        help="Force the search to ignore derivatives caches and run on price/indicator inputs only.",
    )
    parser.add_argument(
        "--derivative-lookback-days",
        type=int,
        default=30,
        help="Recent-days window used when refreshing optional Binance derivatives caches.",
    )
    parser.add_argument(
        "--strict-external-asof",
        action="store_true",
        help="Align daily and sparse external signals to prior completed periods only. Default preserves main behavior.",
    )
    parser.add_argument(
        "--derivative-search-bonus-weight",
        type=float,
        default=0.0,
        help="Extra search-fitness weight for derivative-aware trees. Default 0 preserves main ranking.",
    )
    parser.add_argument(
        "--derivative-survivor-bonus-weight",
        type=float,
        default=0.0,
        help="Extra survivor-selection weight for derivative-aware trees. Default 0 preserves main ranking.",
    )
    parser.add_argument(
        "--derivative-frontier-bonus-weight",
        type=float,
        default=0.0,
        help="Extra near-frontier tie-break weight for derivative-aware trees. Default 0 preserves main ranking.",
    )
    return parser.parse_args()


def normalize_observation_modes(raw_modes: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in raw_modes:
        mode = str(raw).strip().lower()
        if not mode:
            continue
        if mode not in OBSERVATION_MODE_ORDER:
            raise ValueError(f"Unsupported observation mode: {raw}")
        if mode not in normalized:
            normalized.append(mode)
    return tuple(normalized) if normalized else OBSERVATION_MODE_ORDER


def normalize_label_horizons(raw_horizons: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in raw_horizons:
        horizon = str(raw).strip().lower()
        if not horizon:
            continue
        if horizon not in LABEL_HORIZON_ORDER:
            raise ValueError(f"Unsupported label horizon: {raw}")
        if horizon not in normalized:
            normalized.append(horizon)
    return tuple(normalized) if normalized else LABEL_HORIZON_ORDER


def classify_feature_observation_mode(feature: str) -> str:
    if feature in COMMON_OBSERVATION_FEATURES:
        return "common"
    if "imbalance" in feature:
        return OBSERVATION_MODE_IMBALANCE
    if "dc_" in feature:
        return OBSERVATION_MODE_DIRECTIONAL_CHANGE
    if (
        "volume" in feature
        or "vol_rel" in feature
        or "_vol_" in feature
        or "volatility" in feature
        or "mfi" in feature
    ):
        return OBSERVATION_MODE_VOLUME
    return OBSERVATION_MODE_TIME


def filter_feature_specs_by_observation_mode(
    feature_specs: tuple[tuple[str, str, tuple[float, ...]], ...],
    observation_mode: str,
) -> tuple[tuple[str, str, tuple[float, ...]], ...]:
    mode = str(observation_mode).strip().lower()
    if mode not in OBSERVATION_MODE_ORDER:
        raise ValueError(f"Unsupported observation mode: {observation_mode}")
    return tuple(
        spec
        for spec in feature_specs
        if classify_feature_observation_mode(spec[0]) in {"common", mode}
    )


def is_derivative_feature_name(name: str) -> bool:
    return any(token in str(name) for token in DERIVATIVE_FEATURE_TOKENS)


def build_feature_specs(
    pairs: tuple[str, ...],
    observation_mode: str | None = None,
    include_derivative_features: bool = True,
) -> tuple[tuple[str, str, tuple[float, ...]], ...]:
    feature_specs = BASE_FEATURE_SPECS if len(pairs) <= 1 else BASE_FEATURE_SPECS + MULTI_PAIR_FEATURE_SPECS
    if not include_derivative_features:
        feature_specs = tuple(spec for spec in feature_specs if not is_derivative_feature_name(spec[0]))
    if observation_mode is None:
        return feature_specs
    return filter_feature_specs_by_observation_mode(feature_specs, observation_mode)


def allocate_mode_budgets(total: int, modes: tuple[str, ...], minimum: int = 0) -> dict[str, int]:
    if not modes:
        return {}
    total_budget = max(int(total), 0)
    if minimum > 0 and total_budget >= len(modes) * minimum:
        total_budget = max(total_budget, len(modes) * minimum)
    base = total_budget // len(modes)
    remainder = total_budget % len(modes)
    budgets: dict[str, int] = {}
    for idx, mode in enumerate(modes):
        budgets[mode] = base + (1 if idx < remainder else 0)
    return budgets


def candidate_tree_key(observation_mode: str, label_horizon: str, node: TreeNode) -> str:
    return f"{observation_mode}::{label_horizon}::{tree_key(node)}"


def candidate_tree_key_from_raw(observation_mode: str, label_horizon: str, raw_tree_key: str) -> str:
    return f"{observation_mode}::{label_horizon}::{raw_tree_key}"


def _copy_feature_arrays(feature_arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values, dtype="float64").copy() for name, values in feature_arrays.items()}


def _feature_scope(name: str) -> str:
    if name.startswith("btc_"):
        return "btc"
    if name.startswith("bnb_"):
        return "bnb"
    return "pair"


def _scope_values(name: str, btc_values: np.ndarray, bnb_values: np.ndarray, pair_values: np.ndarray) -> np.ndarray:
    scope = _feature_scope(name)
    if scope == "btc":
        return btc_values
    if scope == "bnb":
        return bnb_values
    return pair_values


def project_feature_arrays_by_observation_mode(
    feature_arrays: dict[str, np.ndarray],
    observation_mode: str,
) -> dict[str, np.ndarray]:
    mode = str(observation_mode).strip().lower()
    if mode not in OBSERVATION_MODE_ORDER:
        raise ValueError(f"Unsupported observation mode: {observation_mode}")
    projected = _copy_feature_arrays(feature_arrays)
    if mode == OBSERVATION_MODE_TIME or not projected:
        return projected

    sample = next(iter(projected.values()))
    length = len(sample)
    zeros = np.zeros(length, dtype="float64")
    ones = np.ones(length, dtype="float64")

    btc_volume_rel = np.clip(np.asarray(projected.get("btc_volume_rel_1h", ones), dtype="float64"), 0.25, 4.0)
    bnb_volume_rel = np.clip(np.asarray(projected.get("bnb_volume_rel_1h", ones), dtype="float64"), 0.25, 4.0)
    pair_volume_rel = np.clip(0.5 * (btc_volume_rel + bnb_volume_rel), 0.25, 4.0)

    btc_imbalance = np.clip(np.asarray(projected.get("btc_order_imbalance_1h", zeros), dtype="float64"), -1.0, 1.0)
    bnb_imbalance = np.clip(np.asarray(projected.get("bnb_order_imbalance_1h", zeros), dtype="float64"), -1.0, 1.0)
    pair_imbalance = np.clip(
        np.asarray(projected.get("imbalance_spread_btc_minus_bnb_1h", btc_imbalance - bnb_imbalance), dtype="float64"),
        -1.0,
        1.0,
    )

    btc_dc_trend = np.asarray(projected.get("btc_dc_trend_05_1h", zeros), dtype="float64")
    bnb_dc_trend = np.asarray(projected.get("bnb_dc_trend_05_1h", zeros), dtype="float64")
    pair_dc_trend = np.asarray(
        projected.get("dc_trend_spread_btc_minus_bnb_1h", btc_dc_trend - bnb_dc_trend),
        dtype="float64",
    )
    btc_dc_event = np.asarray(projected.get("btc_dc_event_05_1h", zeros), dtype="float64")
    bnb_dc_event = np.asarray(projected.get("bnb_dc_event_05_1h", zeros), dtype="float64")
    pair_dc_event = np.asarray(
        projected.get("dc_event_spread_btc_minus_bnb_1h", btc_dc_event - bnb_dc_event),
        dtype="float64",
    )
    btc_dc_run = np.asarray(projected.get("btc_dc_run_05_1h", zeros), dtype="float64")
    bnb_dc_run = np.asarray(projected.get("bnb_dc_run_05_1h", zeros), dtype="float64")
    pair_dc_run = btc_dc_run - bnb_dc_run

    if mode == OBSERVATION_MODE_VOLUME:
        btc_factor = np.clip(0.75 + 0.25 * np.sqrt(btc_volume_rel), 0.80, 1.35)
        bnb_factor = np.clip(0.75 + 0.25 * np.sqrt(bnb_volume_rel), 0.80, 1.35)
        pair_factor = np.clip(0.75 + 0.25 * np.sqrt(pair_volume_rel), 0.80, 1.35)
    elif mode == OBSERVATION_MODE_IMBALANCE:
        btc_factor = np.clip(1.0 + 0.35 * btc_imbalance, 0.65, 1.35)
        bnb_factor = np.clip(1.0 + 0.35 * bnb_imbalance, 0.65, 1.35)
        pair_factor = np.clip(1.0 + 0.25 * pair_imbalance, 0.70, 1.30)
    else:
        btc_factor = np.clip(1.0 + 0.20 * btc_dc_trend + 0.15 * np.abs(btc_dc_event) + 0.10 * btc_dc_run, 0.60, 1.60)
        bnb_factor = np.clip(1.0 + 0.20 * bnb_dc_trend + 0.15 * np.abs(bnb_dc_event) + 0.10 * bnb_dc_run, 0.60, 1.60)
        pair_factor = np.clip(1.0 + 0.15 * pair_dc_trend + 0.10 * np.abs(pair_dc_event) + 0.05 * pair_dc_run, 0.60, 1.60)

    for name, values in projected.items():
        if name.startswith("session_"):
            continue
        scoped_factor = _scope_values(name, btc_factor, bnb_factor, pair_factor)
        adjusted = np.asarray(values, dtype="float64") * scoped_factor
        if "vol_rel" in name:
            neutral = 1.0
            adjusted = neutral + (np.asarray(values, dtype="float64") - neutral) * scoped_factor
        elif name == "breadth":
            adjusted = np.clip(adjusted, 0.0, 1.0)
        elif name == "breadth_change_1d":
            adjusted = np.clip(adjusted, -1.0, 1.0)
        projected[name] = adjusted.astype("float64")
    return projected


def apply_label_horizon_to_feature_arrays(
    feature_arrays: dict[str, np.ndarray],
    label_horizon: str,
) -> dict[str, np.ndarray]:
    horizon = str(label_horizon).strip().lower()
    bars = LABEL_HORIZON_BAR_COUNTS[horizon]
    if bars <= 1:
        return _copy_feature_arrays(feature_arrays)
    transformed: dict[str, np.ndarray] = {}
    for name, values in feature_arrays.items():
        array = np.asarray(values, dtype="float64")
        if len(array) == 0:
            transformed[name] = array.copy()
            continue
        anchor_values = array[::bars]
        stepped = np.repeat(anchor_values, bars)[: len(array)]
        transformed[name] = stepped.astype("float64")
    return transformed


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


def _derivative_metric_history_series(
    bundle: dict[str, pd.DataFrame] | None,
    metric_key: str,
    value_column: str,
) -> pd.Series:
    frame = (bundle or {}).get(metric_key)
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


def _derivative_metric_series(
    bundle: dict[str, pd.DataFrame] | None,
    metric_key: str,
    value_column: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    series = _derivative_metric_history_series(bundle, metric_key, value_column)
    if series.empty:
        return pd.Series(np.nan, index=index, dtype="float64")
    return series.reindex(index).ffill().replace([np.inf, -np.inf], np.nan).astype("float64")


def _log_ratio_feature(series: pd.Series) -> pd.Series:
    positive = pd.to_numeric(series, errors="coerce").where(lambda values: values > 0.0)
    logged = np.log(positive)
    return logged.replace([np.inf, -np.inf], np.nan).clip(-3.0, 3.0).astype("float64")


def _open_interest_relative(series: pd.Series) -> pd.Series:
    baseline = series.rolling(12 * 24 * 7, min_periods=12 * 24).median()
    relative = series / baseline.replace(0.0, np.nan)
    return relative.replace([np.inf, -np.inf], np.nan).clip(0.0, 5.0).astype("float64")


def _open_interest_relative_metric(
    bundle: dict[str, pd.DataFrame] | None,
    index: pd.DatetimeIndex,
) -> pd.Series:
    history_series = _derivative_metric_history_series(bundle, "open_interest", "open_interest")
    if history_series.empty:
        return pd.Series(np.nan, index=index, dtype="float64")
    combined_index = history_series.index.union(pd.DatetimeIndex(index)).sort_values()
    aligned = history_series.reindex(combined_index).ffill().replace([np.inf, -np.inf], np.nan).astype("float64")
    relative = _open_interest_relative(aligned)
    return relative.reindex(index).ffill().replace([np.inf, -np.inf], np.nan).astype("float64")


def _is_derivative_feature_name(name: str) -> bool:
    raw = str(name)
    return any(token in raw for token in DERIVATIVE_FEATURE_TOKENS)


def _logic_feature_names(cell: Any) -> set[str]:
    if isinstance(cell, ThresholdCell):
        return {str(cell.spec.feature)}
    if isinstance(cell, NotCell):
        return _logic_feature_names(cell.child)
    if isinstance(cell, (AndCell, OrCell)):
        return _logic_feature_names(cell.left) | _logic_feature_names(cell.right)
    return set()


def summarize_derivative_feature_coverage(
    features: dict[str, pd.Series],
    index: pd.DatetimeIndex,
) -> dict[str, Any]:
    per_feature: dict[str, Any] = {}
    target_index = pd.DatetimeIndex(index)
    for name, series in features.items():
        if not _is_derivative_feature_name(name):
            continue
        aligned = pd.to_numeric(series.reindex(target_index), errors="coerce")
        non_null = aligned.notna()
        non_null_count = int(non_null.sum())
        last_valid_ts = None
        if non_null_count:
            last_valid_ts = pd.Timestamp(aligned.index[non_null][-1]).isoformat()
        per_feature[name] = {
            "non_null_count": non_null_count,
            "non_null_ratio": float(non_null.mean()) if len(aligned) else 0.0,
            "min": None if non_null_count == 0 else float(aligned[non_null].min()),
            "max": None if non_null_count == 0 else float(aligned[non_null].max()),
            "last_valid_timestamp": last_valid_ts,
        }
    aggregate = {
        "feature_count": int(len(per_feature)),
        "features_with_signal_count": int(sum(1 for item in per_feature.values() if item["non_null_count"] > 0)),
        "mean_non_null_ratio": float(np.mean([item["non_null_ratio"] for item in per_feature.values()])) if per_feature else 0.0,
    }
    return {"aggregate": aggregate, "per_feature": per_feature}


def summarize_derivative_bundle_inventory(
    derivatives_by_pair: dict[str, dict[str, pd.DataFrame]],
    *,
    as_of: datetime,
) -> dict[str, Any]:
    as_of_ts = pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of, tz="UTC")
    pairs_summary: dict[str, Any] = {}
    for pair, bundle in (derivatives_by_pair or {}).items():
        metric_summary: dict[str, Any] = {}
        for metric_key, frame in (bundle or {}).items():
            if frame is None or frame.empty or "timestamp" not in frame.columns:
                metric_summary[metric_key] = {
                    "row_count": 0,
                    "non_null_rows": 0,
                    "first_timestamp": None,
                    "latest_timestamp": None,
                    "stale_hours": None,
                }
                continue
            value_columns = [column for column in frame.columns if column != "timestamp"]
            non_null_rows = int(frame[value_columns].notna().any(axis=1).sum()) if value_columns else int(len(frame))
            first_ts = pd.Timestamp(frame["timestamp"].iloc[0]) if len(frame) else None
            latest_ts = pd.Timestamp(frame["timestamp"].iloc[-1]) if len(frame) else None
            if first_ts is not None and first_ts.tzinfo is None:
                first_ts = first_ts.tz_localize("UTC")
            elif first_ts is not None:
                first_ts = first_ts.tz_convert("UTC")
            if latest_ts is not None and latest_ts.tzinfo is None:
                latest_ts = latest_ts.tz_localize("UTC")
            elif latest_ts is not None:
                latest_ts = latest_ts.tz_convert("UTC")
            metric_summary[metric_key] = {
                "row_count": int(len(frame)),
                "non_null_rows": non_null_rows,
                "first_timestamp": None if first_ts is None else first_ts.isoformat(),
                "latest_timestamp": None if latest_ts is None else latest_ts.isoformat(),
                "stale_hours": None if latest_ts is None else float((as_of_ts - latest_ts).total_seconds() / 3600.0),
            }
        pairs_summary[pair] = metric_summary
    return {
        "enabled": bool(any(bundle for bundle in (derivatives_by_pair or {}).values())),
        "as_of": as_of_ts.isoformat(),
        "pairs": pairs_summary,
    }


def summarize_tree_condition_activity(
    node: TreeNode,
    features: dict[str, np.ndarray],
) -> dict[str, Any]:
    first = next(iter(features.values()))
    total_bars = int(len(first))
    conditions: list[dict[str, Any]] = []

    def walk(current: TreeNode, mask: np.ndarray, path: str) -> None:
        if not np.any(mask) or isinstance(current, LeafNode):
            return
        cond_values = evaluate_logic_cell(current.condition, features)
        true_mask = mask & cond_values
        false_mask = mask & (~cond_values)
        feature_names = sorted(_logic_feature_names(current.condition))
        derivative_feature_names = [name for name in feature_names if _is_derivative_feature_name(name)]
        active_bars = int(mask.sum())
        conditions.append(
            {
                "path": path,
                "condition": serialize_logic(current.condition),
                "feature_names": feature_names,
                "derivative_feature_names": derivative_feature_names,
                "derivative_condition": bool(derivative_feature_names),
                "active_bars": active_bars,
                "active_ratio": float(active_bars / total_bars) if total_bars else 0.0,
                "true_count": int(true_mask.sum()),
                "false_count": int(false_mask.sum()),
                "true_ratio_within_active": float(true_mask.sum() / active_bars) if active_bars else 0.0,
            }
        )
        walk(current.if_true, true_mask, f"{path}.T")
        walk(current.if_false, false_mask, f"{path}.F")

    walk(node, np.ones(total_bars, dtype=bool), "root")
    derivative_conditions = [item for item in conditions if item["derivative_condition"]]
    return {
        "condition_count": int(len(conditions)),
        "derivative_condition_count": int(len(derivative_conditions)),
        "derivative_feature_names": sorted(
            {name for item in derivative_conditions for name in item["derivative_feature_names"]}
        ),
        "conditions": conditions,
    }


def summarize_derivative_selection_profile(
    activity: dict[str, Any] | None,
    full_window_coverage: dict[str, Any] | None,
    latest_fold_coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    activity = activity or {}
    derivative_conditions = [
        item
        for item in (activity.get("conditions") or [])
        if bool(item.get("derivative_condition", False))
    ]
    derivative_feature_names = sorted(
        {
            name
            for item in derivative_conditions
            for name in (item.get("derivative_feature_names") or [])
        }
    )

    def mean_feature_coverage(payload: dict[str, Any] | None) -> float:
        per_feature = (payload or {}).get("per_feature") or {}
        ratios = [
            float((per_feature.get(name) or {}).get("non_null_ratio", 0.0))
            for name in derivative_feature_names
            if name in per_feature
        ]
        if not ratios:
            return 0.0
        return float(sum(ratios) / len(ratios))

    def branch_balance(condition: dict[str, Any]) -> float:
        ratio = float(condition.get("true_ratio_within_active", 0.0))
        return max(0.0, 1.0 - abs(ratio - 0.5) * 2.0)

    mean_active_ratio = (
        float(sum(float(item.get("active_ratio", 0.0)) for item in derivative_conditions) / len(derivative_conditions))
        if derivative_conditions
        else 0.0
    )
    mean_branch_balance = (
        float(sum(branch_balance(item) for item in derivative_conditions) / len(derivative_conditions))
        if derivative_conditions
        else 0.0
    )
    full_coverage_ratio = mean_feature_coverage(full_window_coverage)
    latest_coverage_ratio = mean_feature_coverage(latest_fold_coverage)
    condition_count = len(derivative_conditions)
    feature_count = len(derivative_feature_names)
    score = 0.0
    if condition_count > 0:
        score += float(condition_count) * 0.55
        score += float(feature_count) * 0.20
        score += mean_active_ratio * 0.35
        score += mean_branch_balance * 0.45
        score += full_coverage_ratio * 1.10
        score += latest_coverage_ratio * 2.40
    return {
        "condition_count": int(condition_count),
        "feature_count": int(feature_count),
        "feature_names": derivative_feature_names,
        "mean_active_ratio": float(mean_active_ratio),
        "mean_branch_balance": float(mean_branch_balance),
        "full_feature_coverage": float(full_coverage_ratio),
        "latest_feature_coverage": float(latest_coverage_ratio),
        "score": float(min(score, 4.0)),
    }


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


def build_market_features(
    df: pd.DataFrame,
    pairs: tuple[str, ...],
    derivatives_by_pair: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> dict[str, pd.Series]:
    primary_pair = pairs[0]
    secondary_pair = pairs[1] if len(pairs) > 1 else primary_pair
    index = pd.DatetimeIndex(df.index)
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
    btc_order_imbalance_1h = _safe_numeric_feature(df, f"{primary_pair}_order_imbalance", 0.0).clip(-1.0, 1.0)
    bnb_order_imbalance_1h = _safe_numeric_feature(df, f"{secondary_pair}_order_imbalance", 0.0).clip(-1.0, 1.0)
    btc_cci_scaled_1h = (_safe_numeric_feature(df, f"{primary_pair}_cci_14", 0.0) / 100000.0).clip(-3.0, 3.0)
    bnb_cci_scaled_1h = (_safe_numeric_feature(df, f"{secondary_pair}_cci_14", 0.0) / 100000.0).clip(-3.0, 3.0)
    btc_dc_trend_015_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_trend_015", 0.0)
    bnb_dc_trend_015_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_trend_015", 0.0)
    btc_dc_event_015_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_event_015", 0.0)
    bnb_dc_event_015_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_event_015", 0.0)
    btc_dc_trend_03_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_trend_03", 0.0)
    bnb_dc_trend_03_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_trend_03", 0.0)
    btc_dc_event_03_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_event_03", 0.0)
    bnb_dc_event_03_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_event_03", 0.0)
    btc_dc_trend_05_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_trend_05", 0.0)
    bnb_dc_trend_05_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_trend_05", 0.0)
    btc_dc_event_05_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_event_05", 0.0)
    bnb_dc_event_05_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_event_05", 0.0)
    btc_dc_overshoot_05_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_overshoot_05", 0.0)
    bnb_dc_overshoot_05_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_overshoot_05", 0.0)
    btc_dc_run_05_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_run_05", 0.0)
    bnb_dc_run_05_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_run_05", 0.0)
    btc_dc_trend_10_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_trend_10", 0.0)
    bnb_dc_trend_10_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_trend_10", 0.0)
    btc_dc_event_10_1h = _safe_numeric_feature(df, f"{primary_pair}_dc_event_10", 0.0)
    bnb_dc_event_10_1h = _safe_numeric_feature(df, f"{secondary_pair}_dc_event_10", 0.0)
    primary_derivatives = (derivatives_by_pair or {}).get(primary_pair, {})
    secondary_derivatives = (derivatives_by_pair or {}).get(secondary_pair, {})
    btc_oi_rel_1h = _open_interest_relative_metric(primary_derivatives, index)
    bnb_oi_rel_1h = _open_interest_relative_metric(secondary_derivatives, index)
    btc_basis_rate_1h = _derivative_metric_series(primary_derivatives, "basis_perpetual", "basis_rate", index).clip(-0.05, 0.05)
    bnb_basis_rate_1h = _derivative_metric_series(secondary_derivatives, "basis_perpetual", "basis_rate", index).clip(-0.05, 0.05)
    btc_top_pos_log_ratio_1h = _log_ratio_feature(
        _derivative_metric_series(primary_derivatives, "top_trader_position_ratio", "long_short_ratio", index)
    )
    bnb_top_pos_log_ratio_1h = _log_ratio_feature(
        _derivative_metric_series(secondary_derivatives, "top_trader_position_ratio", "long_short_ratio", index)
    )
    btc_top_acct_log_ratio_1h = _log_ratio_feature(
        _derivative_metric_series(primary_derivatives, "top_trader_account_ratio", "long_short_ratio", index)
    )
    bnb_top_acct_log_ratio_1h = _log_ratio_feature(
        _derivative_metric_series(secondary_derivatives, "top_trader_account_ratio", "long_short_ratio", index)
    )
    btc_global_ls_log_ratio_1h = _log_ratio_feature(
        _derivative_metric_series(primary_derivatives, "global_long_short_ratio", "long_short_ratio", index)
    )
    bnb_global_ls_log_ratio_1h = _log_ratio_feature(
        _derivative_metric_series(secondary_derivatives, "global_long_short_ratio", "long_short_ratio", index)
    )
    btc_taker_buy_sell_log_ratio_1h = _log_ratio_feature(
        _derivative_metric_series(primary_derivatives, "taker_buy_sell_ratio", "buy_sell_ratio", index)
    )
    bnb_taker_buy_sell_log_ratio_1h = _log_ratio_feature(
        _derivative_metric_series(secondary_derivatives, "taker_buy_sell_ratio", "buy_sell_ratio", index)
    )

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
    imbalance_spread_1h = btc_order_imbalance_1h - bnb_order_imbalance_1h
    oi_rel_spread_1h = btc_oi_rel_1h - bnb_oi_rel_1h
    basis_rate_spread_1h = btc_basis_rate_1h - bnb_basis_rate_1h
    top_pos_log_ratio_spread_1h = btc_top_pos_log_ratio_1h - bnb_top_pos_log_ratio_1h
    top_acct_log_ratio_spread_1h = btc_top_acct_log_ratio_1h - bnb_top_acct_log_ratio_1h
    global_ls_log_ratio_spread_1h = btc_global_ls_log_ratio_1h - bnb_global_ls_log_ratio_1h
    taker_buy_sell_log_ratio_spread_1h = btc_taker_buy_sell_log_ratio_1h - bnb_taker_buy_sell_log_ratio_1h
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
        "btc_order_imbalance_1h": btc_order_imbalance_1h,
        "bnb_order_imbalance_1h": bnb_order_imbalance_1h,
        "btc_oi_rel_1h": btc_oi_rel_1h,
        "bnb_oi_rel_1h": bnb_oi_rel_1h,
        "btc_basis_rate_1h": btc_basis_rate_1h,
        "bnb_basis_rate_1h": bnb_basis_rate_1h,
        "btc_top_pos_log_ratio_1h": btc_top_pos_log_ratio_1h,
        "bnb_top_pos_log_ratio_1h": bnb_top_pos_log_ratio_1h,
        "btc_top_acct_log_ratio_1h": btc_top_acct_log_ratio_1h,
        "bnb_top_acct_log_ratio_1h": bnb_top_acct_log_ratio_1h,
        "btc_global_ls_log_ratio_1h": btc_global_ls_log_ratio_1h,
        "bnb_global_ls_log_ratio_1h": bnb_global_ls_log_ratio_1h,
        "btc_taker_buy_sell_log_ratio_1h": btc_taker_buy_sell_log_ratio_1h,
        "bnb_taker_buy_sell_log_ratio_1h": bnb_taker_buy_sell_log_ratio_1h,
        "btc_cci_scaled_1h": btc_cci_scaled_1h,
        "bnb_cci_scaled_1h": bnb_cci_scaled_1h,
        "btc_dc_trend_015_1h": btc_dc_trend_015_1h,
        "bnb_dc_trend_015_1h": bnb_dc_trend_015_1h,
        "btc_dc_event_015_1h": btc_dc_event_015_1h,
        "bnb_dc_event_015_1h": bnb_dc_event_015_1h,
        "btc_dc_trend_03_1h": btc_dc_trend_03_1h,
        "bnb_dc_trend_03_1h": bnb_dc_trend_03_1h,
        "btc_dc_event_03_1h": btc_dc_event_03_1h,
        "bnb_dc_event_03_1h": bnb_dc_event_03_1h,
        "btc_dc_trend_05_1h": btc_dc_trend_05_1h,
        "bnb_dc_trend_05_1h": bnb_dc_trend_05_1h,
        "btc_dc_event_05_1h": btc_dc_event_05_1h,
        "bnb_dc_event_05_1h": bnb_dc_event_05_1h,
        "btc_dc_overshoot_05_1h": btc_dc_overshoot_05_1h,
        "bnb_dc_overshoot_05_1h": bnb_dc_overshoot_05_1h,
        "btc_dc_run_05_1h": btc_dc_run_05_1h,
        "bnb_dc_run_05_1h": bnb_dc_run_05_1h,
        "btc_dc_trend_10_1h": btc_dc_trend_10_1h,
        "bnb_dc_trend_10_1h": bnb_dc_trend_10_1h,
        "btc_dc_event_10_1h": btc_dc_event_10_1h,
        "bnb_dc_event_10_1h": bnb_dc_event_10_1h,
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
        "imbalance_spread_btc_minus_bnb_1h": imbalance_spread_1h,
        "oi_rel_spread_btc_minus_bnb_1h": oi_rel_spread_1h,
        "basis_rate_spread_btc_minus_bnb_1h": basis_rate_spread_1h,
        "top_pos_log_ratio_spread_btc_minus_bnb_1h": top_pos_log_ratio_spread_1h,
        "top_acct_log_ratio_spread_btc_minus_bnb_1h": top_acct_log_ratio_spread_1h,
        "global_ls_log_ratio_spread_btc_minus_bnb_1h": global_ls_log_ratio_spread_1h,
        "taker_buy_sell_log_ratio_spread_btc_minus_bnb_1h": taker_buy_sell_log_ratio_spread_1h,
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


def _neutral_feature_fill(name: str) -> float:
    if "spread" in name:
        return 0.0
    if name in {"session_utc_phase", "session_asia_flag", "session_eu_flag", "session_us_flag"}:
        return 0.0
    if "oi_rel" in name:
        return 1.0
    if (name.endswith("_vol_rel") or "volume_rel" in name) and "spread" not in name:
        return 1.0
    if "rsi" in name or "mfi" in name:
        return 50.0
    if "bb_p" in name or "bb_pct_b" in name:
        return 0.5
    return 0.0


def materialize_feature_arrays(
    features: dict[str, pd.Series],
    index: pd.DatetimeIndex,
    *,
    strict_external_asof: bool = False,
) -> dict[str, np.ndarray]:
    day_index = index.normalize()
    completed_day_index = day_index - pd.Timedelta(days=1)
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
        neutral_fill = _neutral_feature_fill(name)
        if name in intraday_features or name.endswith(("_1h", "_6h", "_24h")):
            if strict_external_asof:
                values = series.reindex(index).ffill().replace([np.inf, -np.inf], np.nan).fillna(neutral_fill)
            else:
                values = series.reindex(index).ffill().bfill().replace([np.inf, -np.inf], np.nan).fillna(neutral_fill)
        else:
            effective_day_index = completed_day_index if strict_external_asof else day_index
            values = series.reindex(effective_day_index, method="ffill").replace([np.inf, -np.inf], np.nan).fillna(neutral_fill)
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
    derivative_bonus_weight: float = 0.0,
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
                1 if candidate_final_hard_gate_pass(item) else 0,
                1 if candidate_repair_hard_gate_pass(item) else 0,
                1 if candidate_joint_repair_min_floor_pass(item) else 0,
                1 if candidate_joint_repair_stress_pass(item) else 0,
                1 if candidate_cost_reserve_pass(item) else 0,
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
                    1 if candidate_final_hard_gate_pass(item) else 0,
                    1 if candidate_repair_hard_gate_pass(item) else 0,
                    1 if candidate_joint_repair_min_floor_pass(item) else 0,
                    1 if candidate_joint_repair_stress_pass(item) else 0,
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
            derivative_profile = item.get("derivative_profile") or {}
            derivative_score = float(derivative_profile.get("score", 0.0))
            derivative_condition_count = float(derivative_profile.get("condition_count", 0.0))
            final_hard_gate_pass = candidate_final_hard_gate_pass(item)
            repair_hard_gate_pass = candidate_repair_hard_gate_pass(item)
            joint_repair_balance_pass = candidate_joint_repair_balance_pass(item)
            joint_repair_min_floor_pass = candidate_joint_repair_min_floor_pass(item)
            joint_repair_stress_pass = candidate_joint_repair_stress_pass(item)
            robustness_bonus = 0.0
            robustness_bonus += 8.40 if final_hard_gate_pass else -5.20
            robustness_bonus += 5.20 if repair_hard_gate_pass else -7.20
            robustness_bonus += 0.90 if candidate_wf1_pass(item) else -0.90
            robustness_bonus += 2.20 if candidate_stress_pass(item) else -3.20
            robustness_bonus += 2.60 if candidate_cost_reserve_pass(item) else -3.40
            robustness_bonus += 3.80 if joint_repair_min_floor_pass else 0.0
            robustness_bonus += 4.40 if joint_repair_stress_pass else 0.0
            robustness_bonus += max(0.0, stress_mean - stress_threshold) * 3.80
            robustness_bonus += max(0.0, stress_floor - stress_threshold) * 4.60
            robustness_bonus += max(0.0, stress_reserve_score) / 6000.0
            robustness_bonus += max(0.0, non_nominal_stress_rate - stress_threshold) * 5.20
            robustness_bonus += max(0.0, non_nominal_stress_floor - stress_threshold) * 6.20
            robustness_bonus += max(0.0, non_nominal_stress_reserve) / 6000.0
            if joint_repair_balance_pass and stress_floor <= 0.0:
                robustness_bonus -= 6.40
            if joint_repair_balance_pass and non_nominal_stress_floor <= 0.0:
                robustness_bonus -= 6.80
            if joint_repair_balance_pass and stress_floor < stress_threshold:
                robustness_bonus -= (stress_threshold - stress_floor) * 8.20
            if joint_repair_balance_pass and non_nominal_stress_floor < stress_threshold:
                robustness_bonus -= (stress_threshold - non_nominal_stress_floor) * 9.20
            if joint_repair_balance_pass and not joint_repair_stress_pass:
                robustness_bonus -= 4.80
            if joint_repair_balance_pass and not candidate_cost_reserve_pass(item):
                robustness_bonus -= 2.20
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
            utility += float(derivative_bonus_weight) * derivative_score
            utility += 0.35 * float(item["structural_score"])
            tie_break = (
                utility,
                1.0 if final_hard_gate_pass else 0.0,
                1.0 if repair_hard_gate_pass else 0.0,
                1.0 if joint_repair_min_floor_pass else 0.0,
                1.0 if joint_repair_stress_pass else 0.0,
                1.0 if candidate_cost_reserve_pass(item) else 0.0,
                1.0 if candidate_wf1_pass(item) else 0.0,
                1.0 if candidate_stress_pass(item) else 0.0,
                float(item["search_fitness"]),
                derivative_score,
                derivative_condition_count,
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
        "final_hard_gate_pass_count": sum(1 for item in selected if candidate_final_hard_gate_pass(item)),
        "repair_hard_gate_pass_count": sum(1 for item in selected if candidate_repair_hard_gate_pass(item)),
        "joint_repair_min_floor_pass_count": sum(1 for item in selected if candidate_joint_repair_min_floor_pass(item)),
        "joint_repair_stress_pass_count": sum(1 for item in selected if candidate_joint_repair_stress_pass(item)),
        "top_fitness_depths": [int(item["tree_depth"]) for item in ranked[:survivor_count]],
        "top_fitness_logic_depths": [int(item["logic_depth"]) for item in ranked[:survivor_count]],
    }
    return selected, diagnostics


def select_near_frontier_structural_winner(
    candidates: list[dict[str, Any]],
    frontier_ratio: float = 0.080,
    frontier_floor: float = 250.0,
    derivative_bonus_weight: float = 0.0,
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
    final_hard_frontier = [item for item in wf1_frontier if candidate_final_hard_gate_pass(item)]
    repair_hard_frontier = [item for item in wf1_frontier if candidate_repair_hard_gate_pass(item)]
    joint_repair_min_floor_frontier = [item for item in repair_hard_frontier if candidate_joint_repair_min_floor_pass(item)]
    joint_repair_stress_frontier = [item for item in wf1_frontier if candidate_joint_repair_stress_pass(item)]
    stress_frontier = [item for item in wf1_frontier if candidate_stress_pass(item)]
    reserve_frontier = [item for item in wf1_frontier if candidate_cost_reserve_pass(item)]
    selection_frontier = (
        final_hard_frontier
        or
        joint_repair_min_floor_frontier
        or repair_hard_frontier
        or joint_repair_stress_frontier
        or stress_frontier
        or reserve_frontier
        or wf1_frontier
        or frontier
    )

    def key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, str]:
        robustness = item.get("robustness", {})
        derivative_profile = item.get("derivative_profile") or {}
        return (
            float(candidate_final_hard_gate_pass(item)),
            float(candidate_repair_hard_gate_pass(item)),
            float(candidate_joint_repair_min_floor_pass(item)),
            float(candidate_joint_repair_stress_pass(item)),
            float(candidate_wf1_pass(item)),
            float(candidate_stress_pass(item)),
            float(candidate_cost_reserve_pass(item)),
            float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)),
            float(robustness.get("min_fold_stress_survival_rate", robustness.get("stress_survival_rate_min", 0.0))),
            float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0)),
            float(robustness.get("latest_fold_stress_reserve_score", 0.0)),
            float(item["structural_score"]),
            float(derivative_bonus_weight) * float(derivative_profile.get("score", 0.0)),
            float(derivative_bonus_weight) * float(derivative_profile.get("latest_feature_coverage", 0.0)),
            float(derivative_bonus_weight) * float(derivative_profile.get("condition_count", 0.0)),
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
        "final_hard_frontier_count": len(final_hard_frontier),
        "repair_hard_frontier_count": len(repair_hard_frontier),
        "joint_repair_min_floor_frontier_count": len(joint_repair_min_floor_frontier),
        "joint_repair_stress_frontier_count": len(joint_repair_stress_frontier),
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
        "winner_final_hard_gate_pass": candidate_final_hard_gate_pass(winner),
        "winner_repair_hard_gate_pass": candidate_repair_hard_gate_pass(winner),
        "winner_joint_repair_min_floor_pass": candidate_joint_repair_min_floor_pass(winner),
        "winner_joint_repair_stress_pass": candidate_joint_repair_stress_pass(winner),
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
            "key": "recent_1y",
            "label": "recent_1y",
            "description": "Latest 1 year ending at the most recent bar",
            "start": max(first_day, end_day - pd.DateOffset(years=1)).date().isoformat(),
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
        if key not in baseline_windows:
            continue
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


def apply_leaf_gene_to_pair_config(
    pair_config: dict[str, Any],
    gene: LeafGene | None,
    route_thresholds: tuple[float, ...],
    library_size: int,
) -> dict[str, Any]:
    adjusted = copy.deepcopy(pair_config)
    leaf_gene = gene if gene is not None else LeafGene()
    route_state_mode = normalize_route_state_mode(adjusted.get("route_state_mode"))
    base_threshold_idx = route_thresholds.index(float(adjusted["route_breadth_threshold"]))
    biased_threshold_idx = min(max(base_threshold_idx + int(leaf_gene.route_threshold_bias), 0), len(route_thresholds) - 1)
    mapping = list(normalize_mapping_indices(adjusted["mapping_indices"], route_state_mode))
    adjusted["route_breadth_threshold"] = float(route_thresholds[biased_threshold_idx])
    adjusted["mapping_indices"] = [
        min(max(int(value) + int(leaf_gene.mapping_shift), 0), max(library_size - 1, 0))
        for value in mapping
    ]
    adjusted["route_state_mode"] = route_state_mode
    adjusted["leaf_gene"] = json_safe(leaf_gene)
    return adjusted


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
        if key not in baseline_windows:
            continue
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


def _safe_metric(mapping: dict[str, Any], field: str) -> float:
    try:
        return float(mapping.get(field, 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def build_pair_repair_metrics(
    windows: dict[str, Any],
    repair_pair: str | None,
    window_key: str = "recent_1y",
) -> dict[str, Any]:
    active_window_key = window_key if window_key in windows else "recent_6m"
    window = windows.get(active_window_key) or {}
    per_pair = window.get("per_pair") or {}
    resolved_pair = repair_pair if repair_pair in per_pair else None
    if resolved_pair is None and per_pair:
        resolved_pair = min(per_pair, key=lambda pair: _safe_metric(per_pair.get(pair) or {}, "avg_daily_return"))
    pair_metrics = per_pair.get(resolved_pair) or {}
    aggregate = window.get("aggregate") or {}
    pair_count = len(per_pair)
    positive_pair_count = int(_safe_metric(aggregate, "positive_pair_count"))
    return {
        "window": active_window_key,
        "repair_pair": resolved_pair,
        "repair_pair_avg_daily_return": _safe_metric(pair_metrics, "avg_daily_return"),
        "repair_pair_total_return": _safe_metric(pair_metrics, "total_return"),
        "repair_pair_max_drawdown": abs(_safe_metric(pair_metrics, "max_drawdown")),
        "worst_pair_avg_daily_return": _safe_metric(aggregate, "worst_pair_avg_daily_return"),
        "positive_pair_count": positive_pair_count,
        "pair_count": int(pair_count),
        "negative_pair_count": max(0, int(pair_count) - positive_pair_count),
        "pair_return_dispersion": _safe_metric(aggregate, "pair_return_dispersion"),
    }


def candidate_joint_repair_balance_pass(
    item: dict[str, Any],
    *,
    recent_6m_floor: float = 0.006,
    full_4y_floor: float = 0.0045,
    full_4y_mdd_cap: float = 0.15,
    tolerance: float = 1e-12,
) -> bool:
    if not candidate_repair_hard_gate_pass(item, tolerance=tolerance):
        return False
    if not candidate_wf1_pass(item, tolerance=tolerance):
        return False
    windows = item.get("windows") or {}
    agg_6m = ((windows.get("recent_6m") or {}).get("aggregate") or {})
    agg_4y = ((windows.get(WINDOW_COMPAT_LABEL_FULL) or {}).get("aggregate") or {})
    return bool(
        float(agg_6m.get("worst_pair_avg_daily_return", 0.0)) >= recent_6m_floor - tolerance
        and float(agg_4y.get("worst_pair_avg_daily_return", 0.0)) >= full_4y_floor - tolerance
        and abs(float(agg_4y.get("worst_max_drawdown", 0.0))) <= full_4y_mdd_cap + tolerance
    )


def candidate_repair_hard_gate_pass(item: dict[str, Any], tolerance: float = 1e-12) -> bool:
    validation_profiles = ((item.get("validation") or {}).get("profiles") or {})
    if not bool((validation_profiles.get("pair_repair_1y") or {}).get("passed", False)):
        return False
    repair_metrics = item.get("repair_metrics") or {}
    if not repair_metrics:
        repair_pair = (validation_profiles.get("pair_repair_1y") or {}).get("repair_pair")
        repair_metrics = build_pair_repair_metrics(item.get("windows") or {}, repair_pair)
    pair_count = int(repair_metrics.get("pair_count", 0) or 0)
    positive_pair_count = int(repair_metrics.get("positive_pair_count", 0) or 0)
    repair_pair_daily = float(repair_metrics.get("repair_pair_avg_daily_return", 0.0) or 0.0)
    repair_pair_total = float(repair_metrics.get("repair_pair_total_return", 0.0) or 0.0)
    repair_pair_mdd = abs(float(repair_metrics.get("repair_pair_max_drawdown", 0.0) or 0.0))
    return bool(
        pair_count > 0
        and positive_pair_count >= pair_count
        and repair_pair_daily >= -tolerance
        and repair_pair_total >= -tolerance
        and repair_pair_mdd <= 0.15 + tolerance
    )


def candidate_final_hard_gate_pass(item: dict[str, Any], tolerance: float = 1e-12) -> bool:
    validation_profiles = ((item.get("validation") or {}).get("profiles") or {})
    if not candidate_repair_hard_gate_pass(item, tolerance=tolerance):
        return False
    if not bool((validation_profiles.get("target_060") or {}).get("passed", False)):
        return False
    return bool(
        candidate_wf1_pass(item, tolerance=tolerance)
        and candidate_stress_pass(item, tolerance=tolerance)
        and candidate_cost_reserve_pass(item, tolerance=tolerance)
    )


def candidate_joint_repair_stress_pass(item: dict[str, Any], tolerance: float = 1e-12) -> bool:
    return bool(
        candidate_joint_repair_balance_pass(item, tolerance=tolerance)
        and candidate_stress_pass(item, tolerance=tolerance)
        and candidate_cost_reserve_pass(item, tolerance=tolerance)
    )


def candidate_joint_repair_min_floor_pass(item: dict[str, Any], tolerance: float = 1e-12) -> bool:
    if not candidate_joint_repair_balance_pass(item, tolerance=tolerance):
        return False
    robustness = item.get("robustness", {})
    threshold = float(robustness.get("stress_survival_threshold", 0.0))
    stress_min = float(
        robustness.get("min_fold_stress_survival_rate", robustness.get("stress_survival_rate_min", 0.0))
    )
    non_nominal_min = float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0))
    latest_reserve = float(robustness.get("latest_fold_stress_reserve_score", 0.0))
    latest_non_nominal_reserve = float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0))
    return bool(
        stress_min >= threshold - tolerance
        and non_nominal_min >= threshold - tolerance
        and latest_reserve >= -tolerance
        and latest_non_nominal_reserve >= -tolerance
    )


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
            variant_key = candidate_tree_key(item["observation_mode"], item["label_horizon"], variant_tree)
            if variant_key in seen:
                continue
            seen.add(variant_key)
            variants.append(evaluate_tree_fn(item["observation_mode"], item["label_horizon"], variant_tree))
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
    repair_pair: str | None = None,
) -> tuple[float, dict[str, dict[str, float]]]:
    recent_2m_key, recent_6m_key, full_key = score_window_labels(windows)
    relative = build_baseline_relative_metrics(windows, baseline_windows)
    rel_2m = relative[recent_2m_key]
    rel_6m = relative[recent_6m_key]
    rel_4y = relative[full_key]
    repair_metrics = build_pair_repair_metrics(windows, repair_pair)
    recent_6m_worst_daily = _safe_metric((windows.get(recent_6m_key) or {}).get("aggregate") or {}, "worst_pair_avg_daily_return")
    recent_6m_mdd = abs(_safe_metric((windows.get(recent_6m_key) or {}).get("aggregate") or {}, "worst_max_drawdown"))
    full_4y_worst_daily = _safe_metric((windows.get(full_key) or {}).get("aggregate") or {}, "worst_pair_avg_daily_return")
    full_4y_mdd = abs(_safe_metric((windows.get(full_key) or {}).get("aggregate") or {}, "worst_max_drawdown"))
    wf1_like_pass = bool(
        float(robustness["latest_fold_delta_worst_pair_total_return"]) >= 0.0
        and float(robustness["latest_fold_delta_worst_max_drawdown"]) >= 0.0
        and float(robustness["latest_fold_delta_worst_daily_win_rate"]) >= 0.0
    )
    repair_pair_count = int(repair_metrics["pair_count"])
    repair_positive_pair_count = int(repair_metrics["positive_pair_count"])
    repair_pair_daily = float(repair_metrics["repair_pair_avg_daily_return"])
    repair_pair_total = float(repair_metrics["repair_pair_total_return"])
    repair_pair_mdd = float(repair_metrics["repair_pair_max_drawdown"])
    repair_positive = bool(
        repair_pair_daily >= 0.0
        and int(repair_metrics["negative_pair_count"]) == 0
    )
    repair_hard_pass = bool(
        repair_positive
        and repair_pair_count > 0
        and repair_positive_pair_count >= repair_pair_count
        and repair_pair_total >= 0.0
        and repair_pair_mdd <= 0.15
    )
    balanced_joint_pass = bool(
        repair_hard_pass
        and wf1_like_pass
        and recent_6m_worst_daily >= 0.006
        and full_4y_worst_daily >= 0.0045
        and full_4y_mdd <= 0.15
    )
    stress_threshold = float(robustness["stress_survival_threshold"])
    stress_pass = bool(
        float(robustness["stress_survival_rate_mean"]) >= stress_threshold
        and float(robustness["stress_survival_rate_min"]) >= stress_threshold
        and float(robustness["latest_fold_stress_reserve_score"]) >= 0.0
    )
    cost_reserve_pass = bool(
        float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0)) >= stress_threshold
        and float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)) >= stress_threshold
        and float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0)) >= 0.0
    )
    min_floor_pass = bool(
        balanced_joint_pass
        and float(robustness["stress_survival_rate_min"]) >= stress_threshold
        and float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)) >= stress_threshold
        and float(robustness["latest_fold_stress_reserve_score"]) >= 0.0
        and float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0)) >= 0.0
    )
    joint_repair_stress_pass = bool(balanced_joint_pass and stress_pass and cost_reserve_pass)
    target_060_pass = bool(
        recent_6m_worst_daily >= 0.006
        and full_4y_worst_daily >= 0.006
        and recent_6m_mdd <= 0.17
        and full_4y_mdd <= 0.20
    )
    final_hard_pass = bool(repair_hard_pass and wf1_like_pass and target_060_pass and stress_pass and cost_reserve_pass)
    full_4y_floor_gap = max(0.0, 0.0045 - full_4y_worst_daily)
    recent_6m_floor_gap = max(0.0, 0.006 - recent_6m_worst_daily)
    target_060_full_gap = max(0.0, 0.006 - full_4y_worst_daily)
    target_060_recent_mdd_gap = max(0.0, recent_6m_mdd - 0.17)
    target_060_full_mdd_gap = max(0.0, full_4y_mdd - 0.20)
    stress_mean_gap = max(0.0, stress_threshold - float(robustness["stress_survival_rate_mean"]))
    stress_min_gap = max(0.0, stress_threshold - float(robustness["stress_survival_rate_min"]))
    non_nominal_latest_gap = max(
        0.0,
        stress_threshold - float(robustness.get("latest_non_nominal_stress_survival_rate", 0.0)),
    )
    non_nominal_min_gap = max(
        0.0,
        stress_threshold - float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)),
    )
    negative_stress_reserve = max(0.0, -float(robustness["latest_fold_stress_reserve_score"]))
    negative_non_nominal_reserve = max(
        0.0,
        -float(robustness.get("latest_non_nominal_stress_reserve_score", 0.0)),
    )
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
    score += float(repair_metrics["worst_pair_avg_daily_return"]) * 120000.0
    score += float(repair_metrics["repair_pair_avg_daily_return"]) * 220000.0
    score += float(repair_metrics["repair_pair_total_return"]) * 900.0
    score += (int(repair_metrics["positive_pair_count"]) - int(repair_metrics["pair_count"])) * 1800.0
    score -= float(repair_metrics["repair_pair_max_drawdown"]) * 12000.0
    score -= float(repair_metrics["pair_return_dispersion"]) * 40000.0
    score += full_4y_worst_daily * 420000.0
    score += recent_6m_worst_daily * 260000.0
    score -= recent_6m_mdd * 18000.0
    if final_hard_pass:
        score += 180000.0
    elif repair_hard_pass:
        score -= target_060_full_gap * 28_000_000.0
        score -= target_060_recent_mdd_gap * 900_000.0
        score -= target_060_full_mdd_gap * 650_000.0
        score -= stress_mean_gap * 260000.0
        score -= stress_min_gap * 320000.0
        score -= non_nominal_latest_gap * 360000.0
        score -= non_nominal_min_gap * 420000.0
        score -= negative_stress_reserve * 40.0
        score -= negative_non_nominal_reserve * 52.0
    if min_floor_pass:
        score += 62000.0
    elif balanced_joint_pass and float(robustness["stress_survival_rate_min"]) > 0.0:
        score += 12000.0
    if joint_repair_stress_pass:
        score += 48000.0
    elif balanced_joint_pass and stress_pass:
        score += 18000.0
    elif balanced_joint_pass:
        score -= stress_mean_gap * 42000.0
        score -= stress_min_gap * 52000.0
        score -= non_nominal_latest_gap * 62000.0
        score -= non_nominal_min_gap * 72000.0
        score -= negative_stress_reserve * 18.0
        score -= negative_non_nominal_reserve * 24.0
    repair_pair_coverage_gap = max(0, repair_pair_count - repair_positive_pair_count)
    repair_pair_total_gap = max(0.0, -repair_pair_total)
    repair_pair_mdd_gap = max(0.0, repair_pair_mdd - 0.15)
    if repair_hard_pass:
        score += 96000.0
        score -= repair_pair_mdd_gap * 220000.0
    else:
        score -= 180000.0
        score -= max(0.0, -repair_pair_daily) * 120_000_000.0
        score -= repair_pair_total_gap * 24000.0
        score -= repair_pair_coverage_gap * 90000.0
        score -= repair_pair_mdd_gap * 260000.0
    if repair_positive:
        score += 18000.0
        score -= full_4y_floor_gap * 9_000_000.0
        score -= recent_6m_floor_gap * 5_000_000.0
        score -= stress_min_gap * 120000.0
        score -= non_nominal_min_gap * 140000.0
        if float(robustness["stress_survival_rate_min"]) <= 0.0:
            score -= 36000.0
        if float(robustness.get("min_fold_non_nominal_stress_survival_rate", 0.0)) <= 0.0:
            score -= 42000.0
    else:
        score -= 70000.0
        score -= abs(float(repair_metrics["repair_pair_avg_daily_return"])) * 60_000_000.0
        score -= abs(float(repair_metrics["repair_pair_total_return"])) * 12000.0
        score -= int(repair_metrics["negative_pair_count"]) * 45000.0
    if balanced_joint_pass:
        score += 22000.0
    elif repair_hard_pass and wf1_like_pass and full_4y_worst_daily >= 0.0040 and full_4y_mdd <= 0.16:
        score += 10000.0
    elif repair_hard_pass and not wf1_like_pass:
        score -= 7000.0
    elif repair_hard_pass and full_4y_worst_daily < 0.0045:
        score -= (0.0045 - full_4y_worst_daily) * 420000.0
    elif repair_hard_pass and recent_6m_worst_daily < 0.006:
        score -= (0.006 - recent_6m_worst_daily) * 360000.0
    if repair_hard_pass and full_4y_mdd > 0.15:
        score -= (full_4y_mdd - 0.15) * 80000.0
    if repair_hard_pass and full_4y_worst_daily < 0.006:
        score -= (0.006 - full_4y_worst_daily) * 1_200_000.0
    if repair_hard_pass and recent_6m_mdd > 0.17:
        score -= (recent_6m_mdd - 0.17) * 180000.0
    if repair_hard_pass and full_4y_mdd > 0.20:
        score -= (full_4y_mdd - 0.20) * 140000.0
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
    if repair_pair_daily < 0.0:
        score -= abs(repair_pair_daily) * 1_600_000.0
    if repair_pair_total < 0.0:
        score -= abs(repair_pair_total) * 1500.0
    if int(repair_metrics["negative_pair_count"]) > 0:
        score -= int(repair_metrics["negative_pair_count"]) * 2200.0
    if repair_pair_coverage_gap > 0:
        score -= repair_pair_coverage_gap * 18000.0
    if repair_pair_mdd_gap > 0.0:
        score -= repair_pair_mdd_gap * 60000.0
    if not repair_positive and full_4y_worst_daily >= 0.0045 and wf1_like_pass:
        score -= 4500.0
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
    strict: bool = False,
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
    if strict:
        raise KeyError(f"Condition spec not available for feature={feature} comparator={comparator} invert={invert}")
    return copy.deepcopy(condition_options[0])


def threshold(
    condition_options: list[ConditionSpec],
    feature: str,
    value: float,
    comparator: str = ">=",
    invert: bool = False,
    strict: bool = False,
) -> ThresholdCell:
    return ThresholdCell(
        spec=find_condition_spec(
            condition_options,
            feature,
            value,
            comparator=comparator,
            invert=invert,
            strict=strict,
        )
    )


def build_seed_trees(
    expert_pool: list[dict[str, Any]],
    condition_options: list[ConditionSpec],
    pairs: tuple[str, ...],
) -> list[TreeNode]:
    def named_threshold(feature: str, value: float, comparator: str = ">=", invert: bool = False) -> ThresholdCell:
        return threshold(condition_options, feature, value, comparator=comparator, invert=invert, strict=True)

    def add_seed(factory: Any, seeds: list[TreeNode]) -> None:
        try:
            seeds.append(factory())
        except KeyError:
            return

    seeds: list[TreeNode] = []
    top_count = min(4, len(expert_pool))
    for idx in range(top_count):
        seeds.append(LeafNode(idx))
    if len(expert_pool) >= 2:
        add_seed(lambda: ConditionNode(
                condition=named_threshold("btc_regime", 0.0, comparator=">="),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            ), seeds)
        add_seed(lambda: ConditionNode(
                condition=AndCell(
                    left=named_threshold("btc_rsi_14d", 55.0, comparator=">="),
                    right=named_threshold("btc_volume_z_7d", 0.0, comparator=">="),
                ),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            ), seeds)
        add_seed(lambda: ConditionNode(
                condition=OrCell(
                    left=named_threshold("session_us_flag", 0.5, comparator=">="),
                    right=named_threshold("btc_atr_pct_14d", 0.02, comparator="<="),
                ),
                if_true=LeafNode(1),
                if_false=LeafNode(0),
            ), seeds)
        add_seed(lambda: ConditionNode(
                condition=AndCell(
                    left=named_threshold("btc_macd_hist_12_26_9", 0.0, comparator=">="),
                    right=named_threshold("btc_bb_pct_b_20_2", 0.50, comparator=">="),
                ),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            ), seeds)
        add_seed(lambda: ConditionNode(
                condition=OrCell(
                    left=named_threshold("btc_mfi_14d", 50.0, comparator=">="),
                    right=named_threshold("btc_cci_20d", 0.0, comparator=">="),
                ),
                if_true=LeafNode(1),
                if_false=LeafNode(0),
            ), seeds)
        add_seed(lambda: ConditionNode(
                condition=AndCell(
                    left=named_threshold("btc_rsi_14_1h", 55.0, comparator=">="),
                    right=named_threshold("btc_volume_rel_1h", 1.0, comparator=">="),
                ),
                if_true=LeafNode(0),
                if_false=LeafNode(1),
            ), seeds)
        add_seed(lambda: ConditionNode(
                condition=OrCell(
                    left=named_threshold("btc_dc_trend_05_1h", 0.5, comparator=">="),
                    right=named_threshold("btc_macd_h_pct_1h", 0.0, comparator=">="),
                ),
                if_true=LeafNode(1),
                if_false=LeafNode(0),
            ), seeds)
    if len(expert_pool) >= 4 and len(pairs) > 1:
        add_seed(lambda: ConditionNode(
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
            ), seeds)
        add_seed(lambda: ConditionNode(
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
            ), seeds)
        add_seed(lambda: ConditionNode(
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
            ), seeds)
    elif len(expert_pool) >= 4:
        add_seed(lambda: ConditionNode(
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
            ), seeds)
        add_seed(lambda: ConditionNode(
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
            ), seeds)
    return seeds


def load_warm_start_trees(
    summary_paths: list[Path],
    *,
    observation_mode: str,
    label_horizon: str,
    limit: int,
) -> list[TreeNode]:
    prioritized: list[tuple[int, str, TreeNode]] = []
    seen: set[str] = set()
    for path in summary_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        raw_candidates: list[dict[str, Any]] = []
        selected = payload.get("selected_candidate") or {}
        if selected:
            raw_candidates.append(selected)
        raw_candidates.extend(payload.get("top_candidates") or [])
        for raw in raw_candidates:
            if str(raw.get("observation_mode") or "") != observation_mode:
                continue
            if str(raw.get("label_horizon") or "") != label_horizon:
                continue
            raw_tree = raw.get("tree")
            if not isinstance(raw_tree, dict):
                continue
            try:
                tree = deserialize_tree(raw_tree)
            except Exception:
                continue
            key = tree_key(tree)
            if key in seen:
                continue
            seen.add(key)
            if candidate_final_hard_gate_pass(raw):
                priority = 0
            elif candidate_repair_hard_gate_pass(raw):
                priority = 1
            elif candidate_joint_repair_min_floor_pass(raw):
                priority = 2
            elif candidate_joint_repair_stress_pass(raw):
                priority = 3
            elif candidate_joint_repair_balance_pass(raw):
                priority = 4
            else:
                priority = 5
            prioritized.append((priority, key, tree))
    if limit <= 0 or not prioritized:
        return []
    prioritized.sort(key=lambda item: (item[0], item[1]))
    repair_candidates = [tree for priority, _, tree in prioritized if priority <= 1]
    if not repair_candidates:
        return [tree for _, _, tree in prioritized[: int(limit)]]
    return repair_candidates[: int(limit)]


def build_exploit_pool(
    parent_pool: list[dict[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    final_hard_pool = [
        item
        for item in parent_pool
        if candidate_final_hard_gate_pass(item, tolerance=tolerance)
    ]
    repair_pool = [
        item
        for item in parent_pool
        if candidate_repair_hard_gate_pass(item, tolerance=tolerance)
        and not candidate_final_hard_gate_pass(item, tolerance=tolerance)
    ]
    if final_hard_pool or repair_pool:
        return [*final_hard_pool, *repair_pool]
    balance_pool = [
        item
        for item in parent_pool
        if candidate_joint_repair_balance_pass(item, tolerance=tolerance)
    ]
    if balance_pool:
        return balance_pool
    return parent_pool[: max(1, min(4, len(parent_pool)))]


def build_local_variant_population(
    base_trees: list[TreeNode],
    *,
    rng: random.Random,
    condition_options: list[ConditionSpec],
    expert_count: int,
    max_depth: int,
    logic_max_depth: int,
    variant_budget: int,
) -> list[TreeNode]:
    if not base_trees or variant_budget <= 0:
        return []
    variants: list[TreeNode] = []
    seen: set[str] = set()

    def add(tree: TreeNode) -> None:
        key = tree_key(tree)
        if key in seen:
            return
        seen.add(key)
        variants.append(copy.deepcopy(tree))

    for tree in base_trees:
        add(tree)
        if len(variants) >= variant_budget:
            return variants[:variant_budget]

    profiles = ("mild", "strong", "ultra", "max_defense")
    attempts = 0
    max_attempts = max(variant_budget * 12, 24)
    while len(variants) < variant_budget and attempts < max_attempts:
        attempts += 1
        base = copy.deepcopy(base_trees[attempts % len(base_trees)])
        mode = attempts % 6
        if mode == 0:
            candidate = mutate_tree(
                base,
                rng,
                condition_options,
                expert_count,
                max_depth,
                logic_max_depth=logic_max_depth,
            )
        elif mode == 1:
            candidate = mutate_tree(
                mutate_tree(
                    base,
                    rng,
                    condition_options,
                    expert_count,
                    max_depth,
                    logic_max_depth=logic_max_depth,
                ),
                rng,
                condition_options,
                expert_count,
                max_depth,
                logic_max_depth=logic_max_depth,
            )
        elif mode == 2:
            candidate = apply_leaf_gene_profile(base, profiles[attempts % len(profiles)])
        elif mode == 3 and len(base_trees) >= 2:
            other = copy.deepcopy(base_trees[(attempts + 1) % len(base_trees)])
            candidate, _ = crossover_tree(base, other, rng)
        elif mode == 4:
            candidate = apply_leaf_gene_profile(
                mutate_tree(
                    base,
                    rng,
                    condition_options,
                    expert_count,
                    max_depth,
                    logic_max_depth=logic_max_depth,
                ),
                profiles[attempts % len(profiles)],
            )
        else:
            candidate = mutate_tree(
                apply_leaf_gene_profile(base, profiles[attempts % len(profiles)]),
                rng,
                condition_options,
                expert_count,
                max_depth,
                logic_max_depth=logic_max_depth,
            )
        add(candidate)
    return variants[:variant_budget]


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
                    "tree_key": item["tree_key"],
                    "structure_tree_key": item.get("structure_tree_key"),
                    "observation_mode": item.get("observation_mode"),
                    "label_horizon": item.get("label_horizon"),
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
    observation_modes = normalize_observation_modes(parse_csv_tuple(args.observation_modes, str))
    label_horizons = normalize_label_horizons(parse_csv_tuple(args.label_horizons, str))
    strict_external_asof = bool(args.strict_external_asof)
    derivatives_enabled = bool(args.enable_derivatives) and not bool(args.disable_derivatives)
    warm_start_summary_paths = [
        Path(part.strip())
        for part in parse_csv_tuple(args.warm_start_summaries, str)
        if str(part).strip()
    ]
    _ = resolve_fast_engine(args.fast_engine)

    expert_summary_paths = [part.strip() for part in args.expert_summaries.split(",") if part.strip()]
    expert_pool, expert_pool_diagnostics = build_expert_pool(expert_summary_paths, args.expert_pool_size, pairs)
    if len(expert_pool) < 2:
        raise RuntimeError("Need at least 2 experts to build recursive fractal trees.")

    baseline_summary = json.loads(Path(args.baseline_summary).read_text())
    llm_reviews = load_llm_review_map(args.llm_review_in)
    full_feature_specs = build_feature_specs(pairs, include_derivative_features=derivatives_enabled)
    feature_specs_by_mode = {
        mode: build_feature_specs(
            pairs,
            observation_mode=mode,
            include_derivative_features=derivatives_enabled,
        )
        for mode in observation_modes
    }
    condition_options_by_mode = {
        mode: build_condition_options(specs)
        for mode, specs in feature_specs_by_mode.items()
    }
    search_configs = [
        {"observation_mode": mode, "label_horizon": horizon}
        for mode in observation_modes
        for horizon in label_horizons
    ]
    config_keys = tuple(f"{cfg['observation_mode']}::{cfg['label_horizon']}" for cfg in search_configs)
    population_budget_by_config = allocate_mode_budgets(args.population, config_keys, minimum=0)
    elite_budget_by_config = {
        key: (
            0
            if population_budget_by_config[key] <= 0
            else min(
                population_budget_by_config[key],
                max(1, int(round(args.elite_count * population_budget_by_config[key] / max(args.population, 1)))),
            )
        )
        for key in config_keys
    }
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
    start_all_dt = datetime.fromisoformat(start_all).replace(tzinfo=UTC)
    end_all_dt = datetime.fromisoformat(end_all).replace(tzinfo=UTC) + timedelta(days=1)
    derivative_history_window = timedelta(days=max(8, int(args.derivative_lookback_days)))
    if not derivatives_enabled:
        derivatives_all = {pair: {} for pair in pairs}
    else:
        derivatives_all = {
            pair: load_derivative_bundle(
                pair,
                start_dt=start_all_dt,
                end_dt=end_all_dt,
                fetch=bool(args.fetch_derivatives),
                lookback_days=max(1, int(args.derivative_lookback_days)),
            )
            for pair in pairs
        }

    prepare_started = perf_counter()
    window_cache: dict[str, dict[str, Any]] = {}
    for spec in window_specs:
        label = spec["key"]
        start = spec["start"]
        end = spec["end"]
        df = df_all.loc[start:end].copy()
        derivative_slice = {
            pair: slice_derivative_bundle(
                derivatives_all[pair],
                start_dt=datetime.fromisoformat(start).replace(tzinfo=UTC),
                end_dt=datetime.fromisoformat(end).replace(tzinfo=UTC) + timedelta(days=1),
                history=derivative_history_window,
            )
            for pair in pairs
        }
        raw_features = build_market_features(df, pairs, derivatives_by_pair=derivative_slice)
        derivative_feature_coverage = summarize_derivative_feature_coverage(raw_features, pd.DatetimeIndex(df.index))
        base_feature_arrays = materialize_feature_arrays(
            raw_features,
            pd.DatetimeIndex(df.index),
            strict_external_asof=strict_external_asof,
        )
        feature_arrays_by_mode = {}
        for mode in observation_modes:
            projected_feature_arrays = project_feature_arrays_by_observation_mode(base_feature_arrays, mode)
            feature_arrays_by_mode[mode] = {
                horizon: apply_label_horizon_to_feature_arrays(projected_feature_arrays, horizon)
                for horizon in label_horizons
            }
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
                    strict_external_asof=strict_external_asof,
                ),
            }
        window_cache[label] = {
            "features_by_mode": feature_arrays_by_mode,
            "derivative_feature_coverage": derivative_feature_coverage,
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
        derivative_slice = {
            pair: slice_derivative_bundle(
                derivatives_all[pair],
                start_dt=datetime.fromisoformat(start).replace(tzinfo=UTC),
                end_dt=datetime.fromisoformat(end).replace(tzinfo=UTC) + timedelta(days=1),
                history=derivative_history_window,
            )
            for pair in pairs
        }
        raw_features = build_market_features(df, pairs, derivatives_by_pair=derivative_slice)
        derivative_feature_coverage = summarize_derivative_feature_coverage(raw_features, pd.DatetimeIndex(df.index))
        base_feature_arrays = materialize_feature_arrays(
            raw_features,
            pd.DatetimeIndex(df.index),
            strict_external_asof=strict_external_asof,
        )
        feature_arrays_by_mode = {}
        for mode in observation_modes:
            projected_feature_arrays = project_feature_arrays_by_observation_mode(base_feature_arrays, mode)
            feature_arrays_by_mode[mode] = {
                horizon: apply_label_horizon_to_feature_arrays(projected_feature_arrays, horizon)
                for horizon in label_horizons
            }
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
                    strict_external_asof=strict_external_asof,
                ),
            }
        robustness_cache[label] = {
            "features_by_mode": feature_arrays_by_mode,
            "derivative_feature_coverage": derivative_feature_coverage,
            "pair_cache": pair_cache,
            "bars": int(len(df)),
            "start": start,
            "end": end,
            "label": spec["label"],
            "description": spec["description"],
        }
    prepare_seconds = perf_counter() - prepare_started

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
    population_by_config: dict[str, list[TreeNode]] = {}
    warm_start_diagnostics: list[dict[str, Any]] = []
    for cfg in search_configs:
        mode = cfg["observation_mode"]
        horizon = cfg["label_horizon"]
        config_key = f"{mode}::{horizon}"
        target_population = population_budget_by_config[config_key]
        if target_population <= 0:
            population_by_config[config_key] = []
            continue
        condition_options = condition_options_by_mode[mode]
        warm_start_trees = load_warm_start_trees(
            warm_start_summary_paths,
            observation_mode=mode,
            label_horizon=horizon,
            limit=max(0, int(args.warm_start_candidate_limit)),
        )
        warm_start_variants = build_local_variant_population(
            warm_start_trees,
            rng=rng,
            condition_options=condition_options,
            expert_count=len(expert_pool),
            max_depth=initial_tree_depth_budget,
            logic_max_depth=initial_logic_depth_budget,
            variant_budget=max(0, int(args.warm_start_variant_budget)),
        )
        seed_trees = [*warm_start_variants, *build_seed_trees(expert_pool, condition_options, pairs)]
        deduped_seed_trees: list[TreeNode] = []
        seen_seed_keys: set[str] = set()
        for seed_tree in seed_trees:
            key = tree_key(seed_tree)
            if key in seen_seed_keys:
                continue
            seen_seed_keys.add(key)
            deduped_seed_trees.append(seed_tree)
        seed_trees = deduped_seed_trees
        warm_start_diagnostics.append(
            {
                "observation_mode": mode,
                "label_horizon": horizon,
                "imported_incumbents": len(warm_start_trees),
                "generated_local_variants": len(warm_start_variants),
                "summary_paths": [str(path) for path in warm_start_summary_paths],
            }
        )
        mode_population = seed_trees[:target_population]
        while len(mode_population) < target_population:
            mode_population.append(
                random_tree(
                    rng,
                    condition_options,
                    len(expert_pool),
                    initial_tree_depth_budget,
                    logic_max_depth=initial_logic_depth_budget,
                )
            )
        population_by_config[config_key] = mode_population

    fast_cache: dict[str, dict[str, Any]] = {}
    auto_llm_review_events: list[dict[str, Any]] = []
    generation_selection_diagnostics: list[dict[str, Any]] = []
    immigrant_injection_diagnostics: list[dict[str, Any]] = []
    latest_robustness_key = next(reversed(robustness_cache), None) if robustness_cache else None

    def evaluate_tree(observation_mode: str, label_horizon: str, tree: TreeNode) -> dict[str, Any]:
        structure_key = tree_key(tree)
        key = candidate_tree_key_from_raw(observation_mode, label_horizon, structure_key)
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
        full_window_state = window_cache[WINDOW_COMPAT_LABEL_FULL]
        reference_features = full_window_state["features_by_mode"][observation_mode][label_horizon]
        _, leaf_catalog = evaluate_tree_leaf_codes(tree, reference_features)
        leaf_gene_penalty = leaf_gene_deviation_score(leaf_catalog)
        latest_fold_state = robustness_cache.get(latest_robustness_key) if latest_robustness_key is not None else None
        condition_activity = summarize_tree_condition_activity(tree, reference_features)
        derivative_profile = summarize_derivative_selection_profile(
            condition_activity,
            full_window_state.get("derivative_feature_coverage", {}),
            None if latest_fold_state is None else latest_fold_state.get("derivative_feature_coverage", {}),
        )
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
            leaf_codes, _ = evaluate_tree_leaf_codes(
                tree,
                window_state["features_by_mode"][observation_mode][label_horizon],
            )
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
            leaf_codes, _ = evaluate_tree_leaf_codes(
                tree,
                fold_state["features_by_mode"][observation_mode][label_horizon],
            )
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
        repair_pair = pairs[1] if len(pairs) > 1 else pairs[0]
        repair_metrics = build_pair_repair_metrics(windows, repair_pair)
        validation = build_validation_bundle(windows, baseline_windows, repair_pair=repair_pair)
        fitness, baseline_relative = fractal_fast_scalar_score(
            windows,
            baseline_windows,
            filter_decision,
            tree,
            robustness,
            leaf_gene_penalty,
            repair_pair=repair_pair,
        )
        cached = {
            "tree": copy.deepcopy(tree),
            "candidate_kind": "fractal_tree",
            "observation_mode": observation_mode,
            "observation_mode_label": OBSERVATION_MODE_LABELS.get(observation_mode, observation_mode),
            "label_horizon": label_horizon,
            "label_horizon_label": LABEL_HORIZON_LABELS.get(label_horizon, label_horizon),
            "tree_key": key,
            "structure_tree_key": structure_key,
            "filter": filter_decision,
            "windows": windows,
            "validation": validation,
            "fitness": fitness,
            "baseline_relative": baseline_relative,
            "robustness": robustness,
            "repair_metrics": repair_metrics,
            "tree_depth": tree_depth_value,
            "logic_depth": tree_logic_depth_value,
            "tree_size": tree_size_value,
            "logic_size": tree_logic_size_value,
            "condition_count": condition_count,
            "leaf_signature": leaf_signature,
            "leaf_cardinality": len(leaf_signature),
            "leaf_gene_penalty": float(leaf_gene_penalty),
            "condition_activity": condition_activity,
            "derivative_profile": derivative_profile,
            "structural_score": structural_bonus_from_metrics(
                tree_depth_value,
                tree_logic_depth_value,
                len(leaf_signature),
                condition_count,
            ),
            "performance_score": float(fitness),
        }
        cached["search_fitness"] = (
            float(cached["fitness"])
            + 30.0 * float(cached["structural_score"])
            + float(args.derivative_search_bonus_weight) * float(derivative_profile.get("score", 0.0))
        )
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
        evaluated_by_config: dict[str, list[dict[str, Any]]] = {}
        evaluated_all: list[dict[str, Any]] = []
        for cfg in search_configs:
            mode = cfg["observation_mode"]
            horizon = cfg["label_horizon"]
            config_key = f"{mode}::{horizon}"
            evaluated = [evaluate_tree(mode, horizon, tree) for tree in population_by_config[config_key]]
            evaluated.sort(key=lambda item: item["search_fitness"], reverse=True)
            evaluated_by_config[config_key] = evaluated
            evaluated_all.extend(evaluated)
        evaluated_all.sort(key=lambda item: item["search_fitness"], reverse=True)
        llm_review_event = auto_review_top_candidates(
            evaluated_all,
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
            for cache_key, cached in list(fast_cache.items()):
                if cached.get("structure_tree_key") in reviewed_keys:
                    fast_cache.pop(cache_key, None)
            evaluated_by_config = {}
            evaluated_all = []
            for cfg in search_configs:
                mode = cfg["observation_mode"]
                horizon = cfg["label_horizon"]
                config_key = f"{mode}::{horizon}"
                reevaluated = [evaluate_tree(mode, horizon, tree) for tree in population_by_config[config_key]]
                reevaluated.sort(key=lambda item: item["search_fitness"], reverse=True)
                evaluated_by_config[config_key] = reevaluated
                evaluated_all.extend(reevaluated)
            evaluated_all.sort(key=lambda item: item["search_fitness"], reverse=True)
        generation_mode_summary: list[dict[str, Any]] = []
        for cfg in search_configs:
            mode = cfg["observation_mode"]
            horizon = cfg["label_horizon"]
            config_key = f"{mode}::{horizon}"
            condition_options = condition_options_by_mode[mode]
            evaluated = evaluated_by_config[config_key]
            target_population = population_budget_by_config[config_key]
            if target_population <= 0:
                generation_mode_summary.append(
                    {
                        "observation_mode": mode,
                        "observation_mode_label": OBSERVATION_MODE_LABELS.get(mode, mode),
                        "label_horizon": horizon,
                        "label_horizon_label": LABEL_HORIZON_LABELS.get(horizon, horizon),
                        "population_budget": 0,
                        "elite_budget": 0,
                        "candidate_count": 0,
                        "best_search_fitness": None,
                        "survivor_selection": None,
                        "immigrant_injection": None,
                        "population_after_selection": 0,
                        "population_final": 0,
                    }
                )
                continue
            survivors, survivor_diag = select_generation_survivors(
                evaluated,
                elite_budget_by_config[config_key],
                args.survivor_diversity_weight,
                args.survivor_depth_weight,
                derivative_bonus_weight=float(args.derivative_survivor_bonus_weight),
                target_tree_depth=generation_tree_depth_budget,
                target_logic_depth=generation_logic_depth_budget,
            )
            next_population = [copy.deepcopy(item["tree"]) for item in survivors]
            immigrant_count = 0
            immigrant_tree_budget = max(1, min(args.max_depth, generation_tree_depth_budget + 1))
            immigrant_logic_budget = max(1, min(args.logic_max_depth, generation_logic_depth_budget + 1))
            if args.immigrant_rate > 0.0 and len(next_population) < target_population:
                immigrant_count = min(
                    target_population - len(next_population),
                    max(1, int(round(target_population * args.immigrant_rate))),
                )
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
            immigrant_diag = {
                "generation": generation_idx,
                "observation_mode": mode,
                "label_horizon": horizon,
                "requested_rate": args.immigrant_rate,
                "injected_count": immigrant_count,
                "survivor_count": len(next_population) - immigrant_count,
                "tree_budget": generation_tree_depth_budget,
                "logic_budget": generation_logic_depth_budget,
                "immigrant_tree_budget": immigrant_tree_budget,
                "immigrant_logic_budget": immigrant_logic_budget,
            }
            immigrant_injection_diagnostics.append(immigrant_diag)
            parent_pool = survivors if survivors else evaluated
            exploit_pool = build_exploit_pool(parent_pool)
            local_search_count = 0
            repair_anchor_offspring = 0
            if exploit_pool and len(next_population) < target_population:
                repair_anchor_budget = min(
                    target_population - len(next_population),
                    max(1, min(3, len(exploit_pool))),
                )
                for base_item in exploit_pool[:repair_anchor_budget]:
                    candidate = copy.deepcopy(base_item["tree"])
                    if rng.random() < 0.85:
                        candidate = mutate_tree(
                            candidate,
                            rng,
                            condition_options,
                            len(expert_pool),
                            generation_tree_depth_budget,
                            logic_max_depth=generation_logic_depth_budget,
                        )
                    if rng.random() < 0.30:
                        candidate = apply_leaf_gene_profile(
                            candidate,
                            rng.choice(("mild", "strong", "ultra")),
                        )
                    next_population.append(candidate)
                    repair_anchor_offspring += 1
                    local_search_count += 1
            while len(next_population) < target_population:
                if exploit_pool and rng.random() < float(args.local_search_rate):
                    base_item = rng.choice(exploit_pool)
                    candidate = copy.deepcopy(base_item["tree"])
                    burst = max(1, min(int(args.local_search_mutation_burst), 4))
                    mutate_count = rng.randint(1, burst)
                    for _ in range(mutate_count):
                        candidate = mutate_tree(
                            candidate,
                            rng,
                            condition_options,
                            len(expert_pool),
                            generation_tree_depth_budget,
                            logic_max_depth=generation_logic_depth_budget,
                        )
                    if rng.random() < 0.45:
                        candidate = apply_leaf_gene_profile(
                            candidate,
                            rng.choice(("mild", "strong", "ultra", "max_defense")),
                        )
                    local_search_count += 1
                else:
                    parent_a = tournament_select(parent_pool, rng)
                    if len(parent_pool) >= 2 and rng.random() < 0.65:
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
            population_by_config[config_key] = next_population[:target_population]
            generation_mode_summary.append(
                {
                    "observation_mode": mode,
                    "observation_mode_label": OBSERVATION_MODE_LABELS.get(mode, mode),
                    "label_horizon": horizon,
                    "label_horizon_label": LABEL_HORIZON_LABELS.get(horizon, horizon),
                    "population_budget": target_population,
                    "elite_budget": elite_budget_by_config[config_key],
                    "candidate_count": len(evaluated),
                    "best_search_fitness": float(evaluated[0]["search_fitness"]) if evaluated else None,
                    "survivor_selection": survivor_diag,
                    "immigrant_injection": immigrant_diag,
                    "local_search_offspring": int(local_search_count),
                    "repair_anchor_offspring": int(repair_anchor_offspring),
                    "exploit_pool_size": len(exploit_pool),
                    "population_after_selection": len(next_population),
                    "population_final": len(population_by_config[config_key]),
                }
            )
        generation_selection_diagnostics.append(
            {
                "generation": generation_idx,
                "tree_depth_budget": generation_tree_depth_budget,
                "logic_depth_budget": generation_logic_depth_budget,
                "mode_coverage": generation_mode_summary,
            }
        )
    search_seconds = perf_counter() - search_started

    evaluated = [
        evaluate_tree(cfg["observation_mode"], cfg["label_horizon"], tree)
        for cfg in search_configs
        for tree in population_by_config[f"{cfg['observation_mode']}::{cfg['label_horizon']}"]
    ]
    ranked = sorted({item["tree_key"]: item for item in evaluated}.values(), key=lambda item: item["search_fitness"], reverse=True)
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
    final_hard_candidates = [item for item in top_candidates if candidate_final_hard_gate_pass(item)]
    repair_candidates = [item for item in top_candidates if item["validation"]["profiles"]["pair_repair_1y"]["passed"]]
    repair_hard_candidates = [item for item in top_candidates if candidate_repair_hard_gate_pass(item)]
    joint_repair_min_floor_candidates = [item for item in top_candidates if candidate_joint_repair_min_floor_pass(item)]
    joint_repair_stress_candidates = [item for item in top_candidates if candidate_joint_repair_stress_pass(item)]
    joint_repair_candidates = [item for item in top_candidates if candidate_joint_repair_balance_pass(item)]
    robust_candidates = [item for item in top_candidates if bool(item["robustness"]["gate_passed"])]
    fallback_best = max(top_candidates, key=lambda item: float(item["performance_score"])) if top_candidates else None
    winner_pool = (
        final_hard_candidates
        or
        target_candidates
        or progressive_candidates
        or repair_hard_candidates
        or joint_repair_min_floor_candidates
        or joint_repair_stress_candidates
        or joint_repair_candidates
        or repair_candidates
        or robust_candidates
        or top_candidates
    )
    if cost_reserve_archive_candidate is not None:
        winner_pool = [item for item in winner_pool if item["tree_key"] != cost_reserve_archive_candidate["tree_key"]]
        winner_pool.append(cost_reserve_archive_candidate)
    elif strict_persistent_archive_candidate is not None:
        winner_pool = [item for item in winner_pool if item["tree_key"] != strict_persistent_archive_candidate["tree_key"]]
        winner_pool.append(strict_persistent_archive_candidate)
    elif persistent_archive_candidate is not None:
        winner_pool = [item for item in winner_pool if item["tree_key"] != persistent_archive_candidate["tree_key"]]
        winner_pool.append(persistent_archive_candidate)
    selected, selection_diagnostics = select_near_frontier_structural_winner(
        winner_pool,
        derivative_bonus_weight=float(args.derivative_frontier_bonus_weight),
    )
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
    selection_reason = "final_hard_gate_near_frontier_structural_pass"
    if not final_hard_candidates and target_candidates:
        selection_reason = "target_060_near_frontier_structural_pass"
    elif not final_hard_candidates and not target_candidates and progressive_candidates:
        selection_reason = "progressive_near_frontier_structural_pass"
    elif not final_hard_candidates and not target_candidates and not progressive_candidates and repair_hard_candidates:
        selection_reason = "repair_hard_gate_near_frontier_structural_pass"
    elif not final_hard_candidates and not target_candidates and not progressive_candidates and joint_repair_min_floor_candidates:
        selection_reason = "joint_repair_min_floor_near_frontier_structural_pass"
    elif not final_hard_candidates and not target_candidates and not progressive_candidates and joint_repair_stress_candidates:
        selection_reason = "joint_repair_stress_near_frontier_structural_pass"
    elif not final_hard_candidates and not target_candidates and not progressive_candidates and joint_repair_candidates:
        selection_reason = "joint_repair_market_os_near_frontier_structural_pass"
    elif not final_hard_candidates and not target_candidates and not progressive_candidates and repair_candidates:
        selection_reason = "pair_repair_1y_near_frontier_structural_pass"
    elif not final_hard_candidates and not target_candidates and not progressive_candidates and robust_candidates:
        selection_reason = "robustness_gate_near_frontier_structural_pass"
    elif not final_hard_candidates and not target_candidates and not progressive_candidates and not robust_candidates:
        selection_reason = "no_gate_pass_near_frontier_structural"

    export_llm_review_queue(args.llm_review_out, top_candidates, expert_pool)
    feature_set = describe_feature_set(full_feature_specs)
    derivative_inventory = summarize_derivative_bundle_inventory(derivatives_all, as_of=end_all_dt)
    derivative_inventory["fetch_requested"] = bool(args.fetch_derivatives)
    derivative_inventory["lookback_days"] = int(args.derivative_lookback_days)
    latest_robustness_key = next(reversed(robustness_cache), None) if robustness_cache else None

    def build_candidate_derivative_diagnostics(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
        if candidate is None:
            return None
        full_window_state = window_cache.get(WINDOW_COMPAT_LABEL_FULL) or window_cache.get(window_specs[-1]["key"])
        latest_fold_state = robustness_cache.get(latest_robustness_key) if latest_robustness_key is not None else None
        full_window_features = (
            full_window_state["features_by_mode"][candidate["observation_mode"]][candidate["label_horizon"]]
            if full_window_state is not None
            else {}
        )
        return {
            "tree_condition_activity": (
                candidate.get("condition_activity")
                or summarize_tree_condition_activity(
                    candidate["tree"],
                    full_window_features,
                )
            ) if full_window_features else {"condition_count": 0, "derivative_condition_count": 0, "derivative_feature_names": [], "conditions": []},
            "selection_profile": candidate.get("derivative_profile"),
            "window_feature_coverage": {
                "full_4y": None if full_window_state is None else full_window_state.get("derivative_feature_coverage", {}),
                "latest_fold": None if latest_fold_state is None else {
                    "fold": latest_robustness_key,
                    **latest_fold_state.get("derivative_feature_coverage", {}),
                },
            },
        }

    fallback_derivative_diagnostics = build_candidate_derivative_diagnostics(fallback_best)
    selected_derivative_diagnostics = build_candidate_derivative_diagnostics(selected)

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
            "warm_start_summaries": [str(path) for path in warm_start_summary_paths],
            "warm_start_candidate_limit": int(args.warm_start_candidate_limit),
            "warm_start_variant_budget": int(args.warm_start_variant_budget),
            "local_search_rate": float(args.local_search_rate),
            "local_search_mutation_burst": int(args.local_search_mutation_burst),
            "immigrant_rate": args.immigrant_rate,
            "robustness_folds": args.robustness_folds,
            "robustness_test_months": args.robustness_test_months,
            "commission_stress": [float(v) for v in stress_values],
            "stress_survival_threshold": float(args.stress_survival_threshold),
            "derivatives_enabled": bool(derivatives_enabled),
            "fetch_derivatives": bool(args.fetch_derivatives),
            "derivative_lookback_days": int(args.derivative_lookback_days),
            "strict_external_asof": bool(strict_external_asof),
            "derivative_search_bonus_weight": float(args.derivative_search_bonus_weight),
            "derivative_survivor_bonus_weight": float(args.derivative_survivor_bonus_weight),
            "derivative_frontier_bonus_weight": float(args.derivative_frontier_bonus_weight),
            "observation_modes": [mode for mode in observation_modes],
            "observation_mode_labels": {
                mode: OBSERVATION_MODE_LABELS.get(mode, mode)
                for mode in observation_modes
            },
            "label_horizons": [horizon for horizon in label_horizons],
            "label_horizon_labels": {
                horizon: LABEL_HORIZON_LABELS.get(horizon, horizon)
                for horizon in label_horizons
            },
            "config_population_budgets": population_budget_by_config,
            "config_elite_budgets": elite_budget_by_config,
            "feature_set": {
                "name": FEATURE_SET_NAME,
                "description": FEATURE_SET_DESCRIPTION,
                "condition_option_count": sum(len(options) for options in condition_options_by_mode.values()),
                "feature_count": len(full_feature_specs),
                "features": feature_set["features"],
                "observation_modes": [
                    {
                        "mode": mode,
                        "label": OBSERVATION_MODE_LABELS.get(mode, mode),
                        "feature_count": len(feature_specs_by_mode[mode]),
                        "condition_option_count": len(condition_options_by_mode[mode]),
                        "features": [spec[0] for spec in feature_specs_by_mode[mode]],
                    }
                    for mode in observation_modes
                ],
                "label_horizons": [
                    {
                        "horizon": horizon,
                        "label": LABEL_HORIZON_LABELS.get(horizon, horizon),
                        "decision_stride_bars": LABEL_HORIZON_BAR_COUNTS[horizon],
                        "native": horizon == "5m",
                    }
                    for horizon in label_horizons
                ],
                "feature_context": {
                    "primary_pair": pairs[0],
                    "secondary_pair": pairs[1] if len(pairs) > 1 else pairs[0],
                    "single_asset_mode": len(pairs) == 1,
                },
            },
        },
        "derivatives": derivative_inventory,
        "pairs": list(pairs),
        "backtest_windows": window_specs,
        "robustness_windows": robustness_specs,
        "curriculum_schedule": curriculum_schedule,
        "auto_llm_review_events": auto_llm_review_events,
        "expert_pool_diagnostics": expert_pool_diagnostics,
        "warm_start_diagnostics": warm_start_diagnostics,
        "generation_selection_diagnostics": generation_selection_diagnostics,
        "immigrant_injection_diagnostics": immigrant_injection_diagnostics,
        "model_path": str(args.model),
        "baseline_summary_path": str(args.baseline_summary),
        "baseline_candidate": baseline_summary["selected_candidate"],
        "expert_pool": expert_pool,
        "top_candidates": [
            {
                "tree_key": item["tree_key"],
                "structure_tree_key": item["structure_tree_key"],
                "observation_mode": item["observation_mode"],
                "observation_mode_label": item["observation_mode_label"],
                "label_horizon": item["label_horizon"],
                "label_horizon_label": item["label_horizon_label"],
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
                "derivative_profile": item.get("derivative_profile"),
                "filter": {
                    "accepted": item["filter"].accepted,
                    "source": item["filter"].source,
                    "reason": item["filter"].reason,
                },
                "baseline_relative": item["baseline_relative"],
                "robustness": item["robustness"],
                "repair_metrics": item.get("repair_metrics", {}),
                "windows": item["windows"],
                "validation": item["validation"],
            }
            for item in top_candidates
        ],
        "promotion_candidates": {
            "final_hard_gate": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in final_hard_candidates],
            "target_060": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in target_candidates],
            "progressive_improvement": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in progressive_candidates],
            "repair_hard_gate": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in repair_hard_candidates],
            "joint_repair_min_floor": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in joint_repair_min_floor_candidates],
            "joint_repair_stress": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in joint_repair_stress_candidates],
            "joint_repair_market_os": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in joint_repair_candidates],
            "pair_repair_1y": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in repair_candidates],
            "robustness_gate": [{"tree": serialize_tree(item["tree"]), "observation_mode": item["observation_mode"], "label_horizon": item["label_horizon"]} for item in robust_candidates],
        },
        "selection": {
            "reason": selection_reason,
            "final_hard_gate_pass_count": len(final_hard_candidates),
            "target_060_pass_count": len(target_candidates),
            "progressive_pass_count": len(progressive_candidates),
            "repair_hard_gate_pass_count": len(repair_hard_candidates),
            "joint_repair_min_floor_pass_count": len(joint_repair_min_floor_candidates),
            "joint_repair_stress_pass_count": len(joint_repair_stress_candidates),
            "joint_repair_market_os_pass_count": len(joint_repair_candidates),
            "pair_repair_1y_pass_count": len(repair_candidates),
            "robustness_gate_pass_count": len(robust_candidates),
            "top_k_wf1_pass_count": sum(1 for item in top_candidates if candidate_wf1_pass(item)),
            "top_k_stress_pass_count": sum(1 for item in top_candidates if candidate_stress_pass(item)),
            "top_k_cost_reserve_pass_count": sum(1 for item in top_candidates if candidate_cost_reserve_pass(item)),
            "top_k_final_hard_gate_pass_count": sum(1 for item in top_candidates if candidate_final_hard_gate_pass(item)),
            "top_k_repair_hard_gate_pass_count": sum(1 for item in top_candidates if candidate_repair_hard_gate_pass(item)),
            "top_k_joint_repair_min_floor_pass_count": sum(1 for item in top_candidates if candidate_joint_repair_min_floor_pass(item)),
            "top_k_joint_repair_stress_pass_count": sum(1 for item in top_candidates if candidate_joint_repair_stress_pass(item)),
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
            "tree_key": fallback_best["tree_key"],
            "structure_tree_key": fallback_best["structure_tree_key"],
            "observation_mode": fallback_best["observation_mode"],
            "observation_mode_label": fallback_best["observation_mode_label"],
            "label_horizon": fallback_best["label_horizon"],
            "label_horizon_label": fallback_best["label_horizon_label"],
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
            "repair_metrics": fallback_best.get("repair_metrics", {}),
            "windows": fallback_best["windows"],
            "validation": fallback_best["validation"],
            "derivative_diagnostics": fallback_derivative_diagnostics,
            "filter": {
                "accepted": fallback_best["filter"].accepted,
                "source": fallback_best["filter"].source,
                "reason": fallback_best["filter"].reason,
            },
        },
        "selected_candidate": None if selected is None else {
            "tree_key": selected["tree_key"],
            "structure_tree_key": selected["structure_tree_key"],
            "observation_mode": selected["observation_mode"],
            "observation_mode_label": selected["observation_mode_label"],
            "label_horizon": selected["label_horizon"],
            "label_horizon_label": selected["label_horizon_label"],
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
            "repair_metrics": selected.get("repair_metrics", {}),
            "windows": selected["windows"],
            "validation": selected["validation"],
            "derivative_diagnostics": selected_derivative_diagnostics,
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
