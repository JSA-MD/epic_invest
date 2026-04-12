#!/usr/bin/env python3
"""Experimental MOO router search with specialist experts and execution genes."""

from __future__ import annotations

import argparse
import copy
import faulthandler
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools

import gp_crypto_evolution as gp
from execution_gene_utils import (
    SPECIALIST_ROLE_NAMES,
    build_stressed_execution_profile,
    derive_execution_profile,
    normalize_execution_gene,
)
from pairwise_validation_engine import (
    build_candidate_validation_bundle,
    build_return_frame,
    summarize_state_payload,
    build_validation_robustness_profile,
)
from replay_regime_mixture_realistic import load_model, resolve_candidate
from repair_pair_subset_pairwise_candidate import (
    assign_pairwise_pareto_metadata,
    build_candidate_cost_reference,
    build_candidate_validation_input,
    build_pairwise_pareto_vector,
    candidate_id,
    pair_metric_or_default,
    summarize_special_regime_payload_from_fast_context,
    unwrap_window_aggregate,
)
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    NUMBA_AVAILABLE,
    ROUTE_STATE_MODE_EQUITY_CORR,
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    fast_overlay_replay_from_context,
    json_safe,
    load_or_fetch_funding,
    parse_csv_tuple,
    realistic_overlay_replay_from_context,
    resolve_fast_engine,
    route_state_names,
)
from strategy_replay_dispatch import replay_candidate_from_context
from validate_pair_subset_summary import build_validation_bundle


UTC = timezone.utc
SEARCH_WINDOWS: tuple[tuple[str, str, str], ...] = (
    ("recent_2m", "2026-02-06", "2026-04-06"),
    ("recent_4m", "2025-12-07", "2026-04-06"),
    ("recent_6m", "2025-10-06", "2026-04-06"),
    ("recent_1y", "2025-04-07", "2026-04-06"),
    ("full_4y", "2022-04-06", "2026-04-06"),
)
SEARCH_WINDOWS_LABELS: tuple[str, ...] = tuple(label for label, _, _ in SEARCH_WINDOWS)
ROLE_SPECIAL_REGIME_KEYS: dict[str, str] = {
    "trend": "trend_specialist_regime",
    "range": "range_repair_regime",
    "panic": "panic_deleveraging_regime",
    "carry": "carry_basis_regime",
}
MOO_OBJECTIVES: tuple[tuple[str, bool], ...] = (
    ("dsr_oos", True),
    ("calmar_oos", True),
    ("median_fold_expectancy", True),
    ("recent_win_rate", True),
    ("durable_win_rate", True),
    ("win_rate_floor", True),
    ("regime_coverage", True),
    ("corr_state_robustness", True),
    ("special_regime_coverage", True),
    ("special_regime_robustness", True),
    ("max_drawdown_abs", False),
    ("cvar_95_abs", False),
    ("turnover_cost", False),
    ("slippage_sensitivity", False),
    ("parameter_instability", False),
)
EXECUTION_GENE_OPTIONS: dict[str, tuple[float, ...]] = {
    "maker_priority": (0.20, 0.45, 0.65, 0.85),
    "max_wait_bars": (0, 1, 2, 3),
    "chase_distance_bp": (1.0, 2.0, 4.0, 6.0),
    "cancel_replace_interval_bars": (1, 2, 3),
    "partial_fill_tolerance": (0.25, 0.50, 0.75, 1.00),
    "emergency_market_threshold_bp": (8.0, 15.0, 25.0, 40.0),
    "signal_gate_pct": (0.0, 0.15, 0.30, 0.50, 0.75, 1.0, 1.25, 1.50),
    "regime_buffer_mult": (0.0, 0.10, 0.25, 0.40, 0.60),
    "confirm_bars": (1, 2, 3, 4, 5),
    "flow_alignment_threshold": (0.0, 0.05, 0.10, 0.20, 0.35),
    "dc_alignment_threshold": (0.0, 0.05, 0.10, 0.20, 0.35),
    "min_alignment_votes": (0, 1, 2),
    "abstain_edge_pct": (0.0, 0.05, 0.10, 0.20),
    "specialist_isolation_mult": (0.0, 0.20, 0.40, 0.70),
    "trend_signal_gate_mult": (0.80, 1.00, 1.20),
    "range_signal_gate_mult": (0.80, 1.00, 1.20),
    "panic_signal_gate_mult": (0.70, 1.00, 1.30),
    "carry_signal_gate_mult": (0.80, 1.00, 1.20),
    "trend_regime_buffer_mult": (0.80, 1.00, 1.20),
    "range_regime_buffer_mult": (0.80, 1.00, 1.20),
    "panic_regime_buffer_mult": (0.80, 1.00, 1.20),
    "carry_regime_buffer_mult": (0.80, 1.00, 1.20),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experimental MOO router search for BTC/BNB pairwise overlay strategies.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--base-summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
    )
    parser.add_argument(
        "--library-source",
        choices=("summary", "full-grid"),
        default="full-grid",
    )
    parser.add_argument(
        "--baseline-summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"),
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_moo_router_summary.json"),
    )
    parser.add_argument("--subset-indices", default=None)
    parser.add_argument("--route-thresholds", default="0.35,0.50,0.65,0.80")
    parser.add_argument("--population", type=int, default=72)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--offspring", type=int, default=24)
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--migration-interval", type=int, default=2)
    parser.add_argument("--migrants", type=int, default=2)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--top-k-realistic", type=int, default=12)
    parser.add_argument("--cxpb", type=float, default=0.60)
    parser.add_argument("--mutpb", type=float, default=0.40)
    parser.add_argument("--gene-mutpb", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--role-pool-size", type=int, default=24)
    parser.add_argument(
        "--fast-engine",
        choices=("auto", "python", "numba"),
        default="auto",
    )
    return parser.parse_args()


def build_baseline_pair_configs(pairs: tuple[str, ...], summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = summary["selected_candidate"]
    pair_configs = {}
    for pair in pairs:
        cfg = dict((selected.get("pair_configs") or {}).get(pair) or {})
        cfg["route_state_mode"] = str(cfg.get("route_state_mode") or ROUTE_STATE_MODE_EQUITY_CORR)
        pair_configs[pair] = cfg
    return pair_configs


def build_baseline_seed_pair_configs(
    pairs: tuple[str, ...],
    baseline_pair_configs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    neutral_execution_gene = dict(derive_execution_profile(None)["gene"])
    seeded: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        cfg = copy.deepcopy(baseline_pair_configs[pair])
        cfg["execution_gene"] = dict(neutral_execution_gene)
        seeded[pair] = cfg
    return seeded


def candidate_id(candidate: dict[str, Any], pairs: tuple[str, ...]) -> str:
    parts: list[str] = [str(candidate.get("candidate_kind") or "pairwise_candidate")]
    pair_configs = candidate.get("pair_configs") or {}
    for pair in pairs:
        cfg = pair_configs.get(pair) or {}
        parts.append(pair)
        parts.append(str(float(cfg.get("route_breadth_threshold", 0.0))))
        parts.append(str(cfg.get("route_state_mode") or ROUTE_STATE_MODE_EQUITY_CORR))
        parts.extend(str(int(v)) for v in (cfg.get("mapping_indices") or ()))
        parts.extend(f"ss:{int(v)}" for v in (cfg.get("state_specialists") or ()))
        execution_gene = cfg.get("execution_gene")
        if isinstance(execution_gene, dict):
            normalized_gene = normalize_execution_gene(execution_gene)
            for key in sorted(k for k in normalized_gene.keys() if not k.endswith("_mults")):
                value = normalized_gene[key]
                if isinstance(value, tuple):
                    parts.append(f"{key}={','.join(str(v) for v in value)}")
                else:
                    parts.append(f"{key}={value}")
        else:
            parts.append("execution_gene=legacy")
    return "|".join(parts)


def decode_execution_gene(raw_values: list[int]) -> dict[str, Any]:
    keys = tuple(EXECUTION_GENE_OPTIONS)
    gene: dict[str, Any] = {}
    for idx, key in enumerate(keys):
        options = EXECUTION_GENE_OPTIONS[key]
        gene[key] = options[int(raw_values[idx]) % len(options)]
    return gene


def encode_candidate_from_individual(
    individual: list[int],
    *,
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    pair_configs: dict[str, dict[str, Any]] = {}
    offset = 0
    role_count = len(SPECIALIST_ROLE_NAMES)
    state_count = 12
    execution_gene_len = len(EXECUTION_GENE_OPTIONS)
    for pair in pairs:
        threshold_idx = int(individual[offset])
        specialist_indices = [int(v) for v in individual[offset + 1 : offset + 1 + role_count]]
        specialist_choices = [int(v) % role_count for v in individual[offset + 1 + role_count : offset + 1 + role_count + state_count]]
        execution_gene = decode_execution_gene(
            [int(v) for v in individual[offset + 1 + role_count + state_count : offset + 1 + role_count + state_count + execution_gene_len]]
        )
        mapping_indices = [int(specialist_indices[choice]) for choice in specialist_choices]
        pair_configs[pair] = {
            "route_breadth_threshold": float(route_thresholds[threshold_idx]),
            "route_state_mode": ROUTE_STATE_MODE_EQUITY_CORR,
            "mapping_indices": mapping_indices,
            "specialist_indices": specialist_indices,
            "state_specialists": specialist_choices,
            "state_specialist_roles": [SPECIALIST_ROLE_NAMES[int(choice)] for choice in specialist_choices],
            "execution_gene": execution_gene,
        }
        offset += 1 + role_count + state_count + execution_gene_len
    return {"candidate_kind": "pairwise_candidate", "pair_configs": pair_configs}


def individual_from_pair_configs(
    pair_configs: dict[str, dict[str, Any]],
    *,
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
) -> list[int]:
    genes: list[int] = []
    execution_defaults = derive_execution_profile(None)["gene"]
    for pair in pairs:
        cfg = pair_configs[pair]
        threshold = float(cfg["route_breadth_threshold"])
        specialist_indices = [int(v) for v in cfg.get("specialist_indices") or []]
        state_specialists = [int(v) for v in cfg.get("state_specialists") or []]
        mapping_indices = [int(v) for v in cfg.get("mapping_indices") or []]
        if threshold not in route_thresholds:
            raise ValueError(f"Missing route threshold {threshold} in candidate encoding.")
        if len(specialist_indices) != len(SPECIALIST_ROLE_NAMES) or len(state_specialists) != 12:
            unique_in_order: list[int] = []
            for idx in mapping_indices:
                if idx not in unique_in_order:
                    unique_in_order.append(int(idx))
            while len(unique_in_order) < len(SPECIALIST_ROLE_NAMES):
                unique_in_order.append(unique_in_order[-1] if unique_in_order else 0)
            specialist_indices = unique_in_order[: len(SPECIALIST_ROLE_NAMES)]
            position_by_index = {int(value): pos for pos, value in enumerate(specialist_indices)}
            state_specialists = [
                int(position_by_index.get(int(idx), 0))
                for idx in mapping_indices[:12]
            ]
        execution_gene = dict(execution_defaults)
        execution_gene.update(cfg.get("execution_gene") or {})
        genes.append(route_thresholds.index(threshold))
        genes.extend(int(v) for v in specialist_indices[: len(SPECIALIST_ROLE_NAMES)])
        genes.extend(int(v) for v in state_specialists[:12])
        for key in EXECUTION_GENE_OPTIONS:
            options = EXECUTION_GENE_OPTIONS[key]
            value = execution_gene.get(key, execution_defaults[key])
            option_idx = min(range(len(options)), key=lambda idx: abs(float(options[idx]) - float(value)))
            genes.append(int(option_idx))
    return genes


def compute_diversity_score(candidate: dict[str, Any], pairs: tuple[str, ...]) -> float:
    unique_indices: set[int] = set()
    role_entropy_values: list[float] = []
    for pair in pairs:
        cfg = candidate["pair_configs"][pair]
        specialist_indices = [int(v) for v in cfg.get("specialist_indices") or []]
        state_specialists = [int(v) for v in cfg.get("state_specialists") or []]
        unique_indices.update(specialist_indices)
        if state_specialists:
            counts = np.bincount(np.asarray(state_specialists, dtype="int64"), minlength=len(SPECIALIST_ROLE_NAMES))
            probs = counts / max(np.sum(counts), 1)
            entropy = -float(np.sum([p * math.log(max(p, 1e-12)) for p in probs if p > 0.0]))
            role_entropy_values.append(entropy / math.log(len(SPECIALIST_ROLE_NAMES)))
    max_unique = max(1, len(pairs) * len(SPECIALIST_ROLE_NAMES))
    unique_share = len(unique_indices) / max_unique
    mean_entropy = float(np.mean(role_entropy_values)) if role_entropy_values else 0.0
    return float(0.55 * unique_share + 0.45 * mean_entropy)


def compute_stability_penalty(windows: dict[str, Any]) -> float:
    worst_series = np.asarray(
        [
            float(unwrap_window_aggregate(windows[label])["worst_pair_avg_daily_return"])
            for label in ("recent_2m", "recent_4m", "recent_6m", "recent_1y", "full_4y")
        ],
        dtype="float64",
    )
    dispersion = np.asarray(
        [
            float(unwrap_window_aggregate(windows[label])["pair_return_dispersion"])
            for label in ("recent_2m", "recent_4m", "recent_6m", "recent_1y", "full_4y")
        ],
        dtype="float64",
    )
    return float(np.std(worst_series) + np.mean(dispersion))


def window_mean_daily_win_rate(window: dict[str, Any]) -> float:
    aggregate = unwrap_window_aggregate(window)
    value = aggregate.get("mean_daily_win_rate")
    if value is not None:
        return float(value)
    per_pair = window.get("per_pair") if isinstance(window, dict) else None
    if isinstance(per_pair, dict) and per_pair:
        return float(np.mean([float(metrics.get("daily_win_rate", 0.0)) for metrics in per_pair.values()]))
    return 0.0


def window_worst_daily_win_rate(window: dict[str, Any]) -> float:
    aggregate = unwrap_window_aggregate(window)
    value = aggregate.get("worst_pair_daily_win_rate")
    if value is not None:
        return float(value)
    per_pair = window.get("per_pair") if isinstance(window, dict) else None
    if isinstance(per_pair, dict) and per_pair:
        return float(min(float(metrics.get("daily_win_rate", 0.0)) for metrics in per_pair.values()))
    return 0.0


def build_constant_mapping(index: int, route_state_mode: str) -> tuple[int, ...]:
    state_count = 12 if str(route_state_mode) == ROUTE_STATE_MODE_EQUITY_CORR else 4
    return tuple([int(index)] * state_count)


def _special_regime_mean(profile: dict[str, Any], name: str, fallback: float) -> float:
    special = (profile.get("special_regime_mean_returns") or {})
    value = special.get(name)
    if value is None or not np.isfinite(float(value)):
        return float(fallback)
    return float(value)


def _special_regime_affinity(profile: dict[str, Any], role: str) -> float:
    special = {
        str(name): float(value)
        for name, value in ((profile.get("special_regime_mean_returns") or {}).items())
        if value is not None and np.isfinite(float(value))
    }
    regime_name = ROLE_SPECIAL_REGIME_KEYS[role]
    target = special.get(regime_name)
    if target is None:
        return 0.0
    peers = [value for name, value in special.items() if name != regime_name]
    if not peers:
        return float(target)
    return float(target - np.mean(np.asarray(peers, dtype="float64")))


def _special_regime_coverage(profile: dict[str, Any]) -> float:
    return float(profile.get("special_regime_coverage", 0.0))


def specialist_role_score(
    role: str,
    *,
    recent_2m: dict[str, Any],
    recent_2m_profile: dict[str, Any],
    recent_4m: dict[str, Any],
    recent_4m_profile: dict[str, Any],
    recent_6m: dict[str, Any],
    recent_6m_profile: dict[str, Any],
    recent_1y: dict[str, Any],
    recent_1y_profile: dict[str, Any],
    full_4y: dict[str, Any],
    full_4y_profile: dict[str, Any],
) -> float:
    if role == "trend":
        trend_6m = _special_regime_mean(recent_6m_profile, "trend_specialist_regime", recent_6m["avg_daily_return"])
        trend_1y = _special_regime_mean(recent_1y_profile, "trend_specialist_regime", recent_1y["avg_daily_return"])
        trend_4y = _special_regime_mean(full_4y_profile, "trend_specialist_regime", full_4y["avg_daily_return"])
        trend_affinity = (
            _special_regime_affinity(recent_6m_profile, "trend") * 170000.0
            + _special_regime_affinity(recent_1y_profile, "trend") * 145000.0
            + _special_regime_affinity(full_4y_profile, "trend") * 120000.0
        )
        return float(
            trend_6m * 260000.0
            + trend_1y * 220000.0
            + trend_4y * 180000.0
            + trend_affinity
            + recent_4m["avg_daily_return"] * 120000.0
            + recent_6m["daily_win_rate"] * 900.0
            + _special_regime_coverage(recent_6m_profile) * 500.0
            - abs(recent_6m["max_drawdown"]) * 9000.0
            - abs(recent_1y["max_drawdown"]) * 8000.0
            - abs(full_4y["max_drawdown"]) * 7000.0
        )
    if role == "range":
        range_2m = _special_regime_mean(recent_2m_profile, "range_repair_regime", recent_2m["avg_daily_return"])
        range_4m = _special_regime_mean(recent_4m_profile, "range_repair_regime", recent_4m["avg_daily_return"])
        range_6m = _special_regime_mean(recent_6m_profile, "range_repair_regime", recent_6m["avg_daily_return"])
        range_affinity = (
            _special_regime_affinity(recent_2m_profile, "range") * 190000.0
            + _special_regime_affinity(recent_4m_profile, "range") * 150000.0
            + _special_regime_affinity(recent_6m_profile, "range") * 120000.0
        )
        return float(
            range_2m * 210000.0
            + range_4m * 160000.0
            + range_6m * 140000.0
            + range_affinity
            + recent_2m["daily_win_rate"] * 1600.0
            + recent_6m["daily_win_rate"] * 1100.0
            + _special_regime_coverage(recent_2m_profile) * 700.0
            - abs(recent_2m["max_drawdown"]) * 9000.0
            - abs(recent_6m["max_drawdown"]) * 8000.0
            - abs(recent_2m["worst_day"]) * 2600.0
        )
    if role == "panic":
        panic_2m = _special_regime_mean(recent_2m_profile, "panic_deleveraging_regime", recent_2m["avg_daily_return"])
        panic_6m = _special_regime_mean(recent_6m_profile, "panic_deleveraging_regime", recent_6m["avg_daily_return"])
        panic_affinity = (
            _special_regime_affinity(recent_2m_profile, "panic") * 200000.0
            + _special_regime_affinity(recent_6m_profile, "panic") * 150000.0
        )
        return float(
            panic_2m * 240000.0
            + panic_6m * 180000.0
            + panic_affinity
            + recent_2m["best_day"] * 4200.0
            + recent_6m["best_day"] * 2600.0
            + _special_regime_coverage(recent_2m_profile) * 650.0
            - abs(recent_2m["worst_day"]) * 1800.0
            - abs(recent_2m["max_drawdown"]) * 8000.0
            - recent_2m["n_trades"] * 0.08
        )
    carry_6m = _special_regime_mean(recent_6m_profile, "carry_basis_regime", recent_6m["avg_daily_return"])
    carry_1y = _special_regime_mean(recent_1y_profile, "carry_basis_regime", recent_1y["avg_daily_return"])
    carry_4y = _special_regime_mean(full_4y_profile, "carry_basis_regime", full_4y["avg_daily_return"])
    carry_affinity = (
        _special_regime_affinity(recent_6m_profile, "carry") * 140000.0
        + _special_regime_affinity(recent_1y_profile, "carry") * 150000.0
        + _special_regime_affinity(full_4y_profile, "carry") * 190000.0
    )
    return float(
        carry_4y * 220000.0
        + carry_1y * 170000.0
        + carry_6m * 130000.0
        + carry_affinity
        + full_4y["daily_win_rate"] * 900.0
        + recent_6m["daily_win_rate"] * 400.0
        + _special_regime_coverage(recent_1y_profile) * 550.0
        - abs(full_4y["max_drawdown"]) * 9000.0
        - abs(recent_1y["max_drawdown"]) * 7000.0
        - full_4y["n_trades"] * 0.012
        - recent_1y["n_trades"] * 0.010
    )


def build_role_specific_specialist_pools(
    *,
    pairs: tuple[str, ...],
    library: list[Any],
    library_lookup: dict[str, Any],
    window_cache: dict[str, Any],
    baseline_pair_configs: dict[str, dict[str, Any]],
    fast_engine: str,
    pool_size: int,
    candidate_indices: tuple[int, ...] | list[int] | None = None,
) -> dict[str, dict[str, tuple[int, ...]]]:
    role_pools: dict[str, dict[str, tuple[int, ...]]] = {}
    screen_labels = ("recent_2m", "recent_4m", "recent_6m", "recent_1y", "full_4y")
    candidate_index_list = tuple(
        int(v) for v in (candidate_indices if candidate_indices is not None else tuple(range(len(library))))
    )
    shortlist_limit = min(len(candidate_index_list), max(int(pool_size) * 2, 48))
    for pair in pairs:
        baseline_cfg = baseline_pair_configs[pair]
        baseline_threshold = float(baseline_cfg["route_breadth_threshold"])
        route_state_mode = str(baseline_cfg.get("route_state_mode") or ROUTE_STATE_MODE_EQUITY_CORR)
        baseline_indices = [
            int(v)
            for v in (list(baseline_cfg.get("specialist_indices") or []) + list(baseline_cfg.get("mapping_indices") or []))
        ]
        scored_by_role: dict[str, list[tuple[float, int]]] = {role: [] for role in SPECIALIST_ROLE_NAMES}
        for library_idx in candidate_index_list:
            mapping = build_constant_mapping(library_idx, route_state_mode)
            metrics_by_window: dict[str, dict[str, Any]] = {}
            for label in screen_labels:
                pair_data = window_cache[label]["pairs"][pair]
                fast_result = fast_overlay_replay_from_context(
                    pair_data["fast_context"],
                    library,
                    library_lookup,
                    mapping,
                    baseline_threshold,
                    fast_engine,
                    signal_gate_pct=0.0,
                    regime_buffer_mult=0.0,
                    confirm_bars=1,
                )
                metrics_by_window[label] = summarize_single_fast_result(fast_result)
            for role in SPECIALIST_ROLE_NAMES:
                score = specialist_role_score(
                    role,
                    recent_2m=metrics_by_window["recent_2m"],
                    recent_2m_profile={},
                    recent_4m=metrics_by_window["recent_4m"],
                    recent_4m_profile={},
                    recent_6m=metrics_by_window["recent_6m"],
                    recent_6m_profile={},
                    recent_1y=metrics_by_window["recent_1y"],
                    recent_1y_profile={},
                    full_4y=metrics_by_window["full_4y"],
                    full_4y_profile={},
                )
                scored_by_role[role].append((score, int(library_idx)))
        base_order_by_role: dict[str, list[int]] = {}
        shortlist_indices: set[int] = set(int(v) for v in baseline_indices)
        for role in SPECIALIST_ROLE_NAMES:
            ordered = [idx for _, idx in sorted(scored_by_role[role], key=lambda item: item[0], reverse=True)]
            base_order_by_role[role] = ordered
            shortlist_indices.update(int(idx) for idx in ordered[: min(len(ordered), shortlist_limit)])
        enriched_scores_by_role: dict[str, list[tuple[float, int]]] = {role: [] for role in SPECIALIST_ROLE_NAMES}
        for library_idx in sorted(shortlist_indices):
            mapping = build_constant_mapping(library_idx, route_state_mode)
            metrics_by_window: dict[str, dict[str, Any]] = {}
            profiles_by_window: dict[str, dict[str, Any]] = {}
            for label in screen_labels:
                pair_data = window_cache[label]["pairs"][pair]
                fast_result = fast_overlay_replay_from_context(
                    pair_data["fast_context"],
                    library,
                    library_lookup,
                    mapping,
                    baseline_threshold,
                    fast_engine,
                    signal_gate_pct=0.0,
                    regime_buffer_mult=0.0,
                    confirm_bars=1,
                )
                metrics_by_window[label] = summarize_single_fast_result(fast_result)
                python_result = fast_overlay_replay_from_context(
                    pair_data["fast_context"],
                    library,
                    library_lookup,
                    mapping,
                    baseline_threshold,
                    "python",
                    signal_gate_pct=0.0,
                    regime_buffer_mult=0.0,
                    confirm_bars=1,
                )
                raw_daily_returns = (python_result.get("daily_metrics") or {}).get("daily_returns")
                special_payload = summarize_special_regime_payload_from_fast_context(
                    pair_data["fast_context"],
                    np.asarray([] if raw_daily_returns is None else raw_daily_returns, dtype="float64"),
                )
                profiles_by_window[label] = summarize_state_payload(
                    {
                        "special_regime_returns": (special_payload.get("special_regime_returns") or {}),
                        "total_route_states": 1,
                        "total_corr_buckets": 1,
                    }
                )
            for role in SPECIALIST_ROLE_NAMES:
                score = specialist_role_score(
                    role,
                    recent_2m=metrics_by_window["recent_2m"],
                    recent_2m_profile=profiles_by_window["recent_2m"],
                    recent_4m=metrics_by_window["recent_4m"],
                    recent_4m_profile=profiles_by_window["recent_4m"],
                    recent_6m=metrics_by_window["recent_6m"],
                    recent_6m_profile=profiles_by_window["recent_6m"],
                    recent_1y=metrics_by_window["recent_1y"],
                    recent_1y_profile=profiles_by_window["recent_1y"],
                    full_4y=metrics_by_window["full_4y"],
                    full_4y_profile=profiles_by_window["full_4y"],
                )
                enriched_scores_by_role[role].append((score, int(library_idx)))
        pair_pools: dict[str, tuple[int, ...]] = {}
        enriched_order_by_role = {
            role: [idx for _, idx in sorted(enriched_scores_by_role[role], key=lambda item: item[0], reverse=True)]
            for role in SPECIALIST_ROLE_NAMES
        }
        anchor_by_role: dict[str, int] = {}
        used_anchors: set[int] = set()
        for role in SPECIALIST_ROLE_NAMES:
            ordered = enriched_order_by_role[role]
            anchor = next((idx for idx in ordered if idx not in used_anchors), ordered[0] if ordered else None)
            if anchor is not None:
                anchor_by_role[role] = int(anchor)
                used_anchors.add(int(anchor))
        for role in SPECIALIST_ROLE_NAMES:
            ordered = list(enriched_order_by_role[role])
            anchor = anchor_by_role.get(role)
            other_anchors = {idx for other_role, idx in anchor_by_role.items() if other_role != role}
            prioritized: list[int] = []
            if anchor is not None:
                prioritized.append(int(anchor))
            prioritized.extend(int(idx) for idx in ordered if idx not in prioritized and idx not in other_anchors)
            prioritized.extend(int(idx) for idx in ordered if idx not in prioritized)
            unique = list(dict.fromkeys(prioritized + baseline_indices + base_order_by_role[role]))
            pair_pools[role] = tuple(int(idx) for idx in unique[: max(int(pool_size), len(SPECIALIST_ROLE_NAMES))])
        role_pools[pair] = pair_pools
    return role_pools


def build_neighbor_variants(
    candidate: dict[str, Any],
    *,
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    gene_neighbor_keys = tuple(EXECUTION_GENE_OPTIONS)
    default_gene = derive_execution_profile(None)["gene"]

    def append_bundle_variant(
        base_candidate: dict[str, Any],
        *,
        pair: str,
        base_gene: dict[str, Any],
        updates: dict[str, int],
    ) -> None:
        variant = copy.deepcopy(base_candidate)
        variant["pair_configs"][pair].setdefault("execution_gene", {})
        changed = False
        for key, step in updates.items():
            options = EXECUTION_GENE_OPTIONS[key]
            value = base_gene.get(key, default_gene[key])
            option_idx = min(range(len(options)), key=lambda idx: abs(float(options[idx]) - float(value)))
            new_idx = min(max(option_idx + int(step), 0), len(options) - 1)
            if new_idx == option_idx:
                continue
            variant["pair_configs"][pair]["execution_gene"][key] = options[new_idx]
            changed = True
        if changed:
            variants.append(variant)

    for pair in pairs:
        cfg = candidate["pair_configs"][pair]
        threshold = float(cfg["route_breadth_threshold"])
        if threshold in route_thresholds:
            threshold_idx = route_thresholds.index(threshold)
            for step in (-1, 1):
                new_idx = threshold_idx + step
                if 0 <= new_idx < len(route_thresholds):
                    variant = copy.deepcopy(candidate)
                    variant["pair_configs"][pair]["route_breadth_threshold"] = float(route_thresholds[new_idx])
                    variants.append(variant)
        gene = dict(default_gene)
        gene.update(cfg.get("execution_gene") or {})
        for key in gene_neighbor_keys:
            options = EXECUTION_GENE_OPTIONS[key]
            value = gene.get(key, default_gene[key])
            option_idx = min(range(len(options)), key=lambda idx: abs(float(options[idx]) - float(value)))
            for step in (-1, 1):
                new_idx = option_idx + step
                if 0 <= new_idx < len(options):
                    variant = copy.deepcopy(candidate)
                    variant["pair_configs"][pair].setdefault("execution_gene", {})
                    variant["pair_configs"][pair]["execution_gene"][key] = options[new_idx]
                    variants.append(variant)
        append_bundle_variant(
            candidate,
            pair=pair,
            base_gene=gene,
            updates={
                "signal_gate_pct": 1,
                "regime_buffer_mult": 1,
                "confirm_bars": 1,
                "abstain_edge_pct": 1,
            },
        )
        append_bundle_variant(
            candidate,
            pair=pair,
            base_gene=gene,
            updates={
                "maker_priority": 1,
                "max_wait_bars": 1,
                "partial_fill_tolerance": 1,
                "regime_buffer_mult": 1,
                "confirm_bars": 1,
            },
        )
        append_bundle_variant(
            candidate,
            pair=pair,
            base_gene=gene,
            updates={
                "signal_gate_pct": -1,
                "regime_buffer_mult": -1,
                "confirm_bars": -1,
                "abstain_edge_pct": -1,
                "min_alignment_votes": -1,
            },
        )
    return variants


def compute_neighbor_sensitivity_penalty(
    candidate: dict[str, Any],
    *,
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
    window_cache: dict[str, Any],
    library: list[Any],
    library_lookup: dict[str, Any],
    fast_engine: str,
) -> float:
    base_windows = {}
    for label in ("recent_6m", "recent_1y", "full_4y"):
        window_data = window_cache[label]
        per_pair = {}
        for pair in pairs:
            cfg = candidate["pair_configs"][pair]
            execution_profile = derive_execution_profile(cfg.get("execution_gene"))
            per_pair[pair] = summarize_single_fast_result(
                fast_overlay_replay_from_context(
                    window_data["pairs"][pair]["fast_context"],
                    library,
                    library_lookup,
                    tuple(int(v) for v in cfg["mapping_indices"]),
                    float(cfg["route_breadth_threshold"]),
                    fast_engine,
                    commission_pct=float(execution_profile["fast_commission_pct"]),
                    no_trade_band_pct=float(execution_profile["no_trade_band_pct"]),
                    signal_gate_pct=float(execution_profile["signal_gate_pct"]),
                    regime_buffer_mult=float(execution_profile["regime_buffer_mult"]),
                    confirm_bars=int(execution_profile["confirm_bars"]),
                    execution_gene=cfg.get("execution_gene"),
                    state_specialists=tuple(int(v) for v in cfg.get("state_specialists") or ()),
                )
            )
        base_windows[label] = {"aggregate": aggregate_metrics(per_pair)}
    base_score = float(
        unwrap_window_aggregate(base_windows["recent_6m"])["worst_pair_avg_daily_return"]
        + unwrap_window_aggregate(base_windows["recent_1y"])["worst_pair_avg_daily_return"]
        + unwrap_window_aggregate(base_windows["full_4y"])["worst_pair_avg_daily_return"]
        + (window_worst_daily_win_rate(base_windows["recent_6m"]) - 0.50) * 0.0025
        + (window_worst_daily_win_rate(base_windows["recent_1y"]) - 0.50) * 0.0025
        + (window_worst_daily_win_rate(base_windows["full_4y"]) - 0.50) * 0.0030
    )
    variants = build_neighbor_variants(candidate, pairs=pairs, route_thresholds=route_thresholds)
    if not variants:
        return 0.0
    degradations: list[float] = []
    for variant in variants:
        variant_score = 0.0
        for label in ("recent_6m", "recent_1y", "full_4y"):
            window_data = window_cache[label]
            per_pair = {}
            for pair in pairs:
                cfg = variant["pair_configs"][pair]
                execution_profile = derive_execution_profile(cfg.get("execution_gene"))
                per_pair[pair] = summarize_single_fast_result(
                    fast_overlay_replay_from_context(
                        window_data["pairs"][pair]["fast_context"],
                        library,
                        library_lookup,
                        tuple(int(v) for v in cfg["mapping_indices"]),
                        float(cfg["route_breadth_threshold"]),
                        fast_engine,
                        commission_pct=float(execution_profile["fast_commission_pct"]),
                        no_trade_band_pct=float(execution_profile["no_trade_band_pct"]),
                        signal_gate_pct=float(execution_profile["signal_gate_pct"]),
                        regime_buffer_mult=float(execution_profile["regime_buffer_mult"]),
                        confirm_bars=int(execution_profile["confirm_bars"]),
                        execution_gene=cfg.get("execution_gene"),
                        state_specialists=tuple(int(v) for v in cfg.get("state_specialists") or ()),
                    )
                )
            aggregate = aggregate_metrics(per_pair)
            variant_score += float(aggregate["worst_pair_avg_daily_return"])
            win_weight = 0.0025 if label != "full_4y" else 0.0030
            variant_score += (float(aggregate.get("worst_pair_daily_win_rate", 0.0)) - 0.50) * win_weight
        degradations.append(max(0.0, base_score - variant_score))
    return float(np.mean(np.asarray(degradations, dtype="float64"))) if degradations else 0.0


def fast_candidate_sort_score(item: dict[str, Any]) -> float:
    recent_2m = unwrap_window_aggregate(item["windows"]["recent_2m"])
    recent_4m = unwrap_window_aggregate(item["windows"]["recent_4m"])
    recent_6m = unwrap_window_aggregate(item["windows"]["recent_6m"])
    recent_1y = unwrap_window_aggregate(item["windows"]["recent_1y"])
    full_4y = unwrap_window_aggregate(item["windows"]["full_4y"])
    recent_win_rate = np.mean(
        [
            float(recent_2m.get("mean_daily_win_rate", 0.0)),
            float(recent_4m.get("mean_daily_win_rate", 0.0)),
            float(recent_6m.get("mean_daily_win_rate", 0.0)),
        ]
    )
    durable_win_rate = np.mean(
        [
            float(recent_6m.get("mean_daily_win_rate", 0.0)),
            float(recent_1y.get("mean_daily_win_rate", 0.0)),
            float(full_4y.get("mean_daily_win_rate", 0.0)),
        ]
    )
    win_rate_floor = min(
        float(recent_2m.get("worst_pair_daily_win_rate", 0.0)),
        float(recent_4m.get("worst_pair_daily_win_rate", 0.0)),
        float(recent_6m.get("worst_pair_daily_win_rate", 0.0)),
        float(recent_1y.get("worst_pair_daily_win_rate", 0.0)),
        float(full_4y.get("worst_pair_daily_win_rate", 0.0)),
    )
    score = 0.0
    score += float(item["reserve"]) * 300000.0
    score += float(recent_2m["worst_pair_avg_daily_return"]) * 180000.0
    score += float(recent_4m["worst_pair_avg_daily_return"]) * 120000.0
    score += float(recent_6m["worst_pair_avg_daily_return"]) * 150000.0
    score += float(recent_1y["worst_pair_avg_daily_return"]) * 150000.0
    score += float(full_4y["worst_pair_avg_daily_return"]) * 120000.0
    score += float(recent_win_rate) * 2200.0
    score += float(durable_win_rate) * 2600.0
    score += float(win_rate_floor) * 3200.0
    score += max(0.0, float(win_rate_floor) - 0.50) * 10000.0
    score -= max(0.0, 0.50 - float(durable_win_rate)) * 9000.0
    score += float(item["diversity_score"]) * 2400.0
    score -= float(item["turnover_proxy"]) * 1200.0
    score -= float(item["slippage_sensitivity_proxy"]) * 90000.0
    score -= float(item["stability_penalty"]) * 180000.0
    score -= abs(float(recent_4m["worst_max_drawdown"])) * 9000.0
    score -= abs(float(recent_6m["worst_max_drawdown"])) * 12000.0
    score -= abs(float(recent_1y["worst_max_drawdown"])) * 12000.0
    score -= abs(float(full_4y["worst_max_drawdown"])) * 9000.0
    return float(score)


def moo_raw_metrics(item: dict[str, Any]) -> dict[str, float]:
    validation_engine = item["validation_engine"]
    market_os = validation_engine.get("market_operating_system") or {}
    raw = (market_os.get("fitness") or {}).get("raw") or {}
    windows = item["windows"]
    recent_win_rate = float(
        np.mean([window_mean_daily_win_rate(windows[label]) for label in ("recent_2m", "recent_4m", "recent_6m")])
    )
    durable_win_rate = float(
        np.mean([window_mean_daily_win_rate(windows[label]) for label in ("recent_6m", "recent_1y", "full_4y")])
    )
    win_rate_floor = float(min(window_worst_daily_win_rate(windows[label]) for label in SEARCH_WINDOWS_LABELS))
    return {
        "dsr_oos": float(raw.get("dsr_oos", validation_engine.get("dsr_proxy", 0.0))),
        "calmar_oos": float(raw.get("calmar_oos", 0.0)),
        "median_fold_expectancy": float(raw.get("median_fold_expectancy", 0.0)),
        "recent_win_rate": recent_win_rate,
        "durable_win_rate": durable_win_rate,
        "win_rate_floor": win_rate_floor,
        "regime_coverage": float(raw.get("regime_coverage", 0.0)),
        "corr_state_robustness": float(raw.get("corr_state_robustness", 0.0)),
        "special_regime_coverage": float(raw.get("special_regime_coverage", 0.0)),
        "special_regime_robustness": float(raw.get("special_regime_robustness", 0.0)),
        "max_drawdown_abs": abs(float(raw.get("max_drawdown", 0.0))),
        "cvar_95_abs": abs(float(min(raw.get("cvar_95", 0.0), 0.0))),
        "turnover_cost": float(raw.get("turnover_cost", 0.0)),
        "slippage_sensitivity": float(item.get("slippage_sensitivity", 0.0)),
        "parameter_instability": float(max(raw.get("parameter_instability", 1.0), item.get("parameter_sensitivity_penalty", 0.0))),
    }


def realistic_selection_score(item: dict[str, Any]) -> float:
    metrics = moo_raw_metrics(item)
    robustness = item.get("validation_robustness") or build_validation_robustness_profile(item["validation_engine"])
    gate = item["validation_engine"].get("gate") or {}
    recent_6m = unwrap_window_aggregate(item["windows"]["recent_6m"])
    recent_1y = unwrap_window_aggregate(item["windows"]["recent_1y"])
    full_4y = unwrap_window_aggregate(item["windows"]["full_4y"])
    special_regime_means = ((item.get("state_summary") or {}).get("special_regime_mean_returns") or {})
    special_worst = min((float(v) for v in special_regime_means.values()), default=0.0)
    score = 0.0
    score += metrics["dsr_oos"] * 3400.0
    score += metrics["calmar_oos"] * 1200.0
    score += metrics["median_fold_expectancy"] * 140000.0
    score += metrics["recent_win_rate"] * 2600.0
    score += metrics["durable_win_rate"] * 3200.0
    score += metrics["win_rate_floor"] * 3600.0
    score += max(0.0, metrics["win_rate_floor"] - 0.50) * 14000.0
    score += metrics["regime_coverage"] * 1800.0
    score += metrics["corr_state_robustness"] * 1800.0
    score += metrics["special_regime_coverage"] * 900.0
    score += metrics["special_regime_robustness"] * 1200.0
    score += float(recent_6m["worst_pair_avg_daily_return"]) * 130000.0
    score += float(recent_1y["worst_pair_avg_daily_return"]) * 150000.0
    score += float(full_4y["worst_pair_avg_daily_return"]) * 100000.0
    score += float(robustness.get("score", 0.0)) * 1500.0
    score -= metrics["max_drawdown_abs"] * 1400.0
    score -= metrics["cvar_95_abs"] * 2200.0
    score -= metrics["turnover_cost"] * 4000.0
    score -= metrics["slippage_sensitivity"] * 3200.0
    score -= metrics["parameter_instability"] * 1200.0
    score -= max(0.0, 0.50 - metrics["durable_win_rate"]) * 16000.0
    score -= max(0.0, 0.50 - metrics["win_rate_floor"]) * 20000.0
    score -= abs(float(recent_1y["worst_max_drawdown"])) * 12000.0
    score -= abs(float(full_4y["worst_max_drawdown"])) * 9000.0
    score += special_worst * 120000.0
    if bool(gate.get("passed", False)):
        score += 2000.0
    else:
        score -= 1800.0
    return float(score)


def moo_dominates(left: dict[str, Any], right: dict[str, Any], *, eps: float = 1e-12) -> bool:
    left_vec = left["moo_vector"]
    right_vec = right["moo_vector"]
    better_or_equal = True
    strictly_better = False
    for name, maximize in MOO_OBJECTIVES:
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


def assign_moo_pareto_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    remaining = list(rows)
    fronts: list[list[dict[str, Any]]] = []
    while remaining:
        front: list[dict[str, Any]] = []
        for row in remaining:
            if not any(moo_dominates(other, row) for other in remaining if other["candidate_id"] != row["candidate_id"]):
                front.append(row)
        fronts.append(front)
        front_ids = {row["candidate_id"] for row in front}
        remaining = [row for row in remaining if row["candidate_id"] not in front_ids]

    metadata: dict[str, dict[str, Any]] = {}
    for rank, front in enumerate(fronts, start=1):
        distances = {row["candidate_id"]: 0.0 for row in front}
        if len(front) <= 2:
            for row in front:
                distances[row["candidate_id"]] = float("inf")
        else:
            for name, _ in MOO_OBJECTIVES:
                ordered = sorted(front, key=lambda row: float(row["moo_vector"][name]))
                min_value = float(ordered[0]["moo_vector"][name])
                max_value = float(ordered[-1]["moo_vector"][name])
                distances[ordered[0]["candidate_id"]] = float("inf")
                distances[ordered[-1]["candidate_id"]] = float("inf")
                scale = max(max_value - min_value, 1e-8)
                for idx in range(1, len(ordered) - 1):
                    candidate_id_value = ordered[idx]["candidate_id"]
                    if np.isinf(distances[candidate_id_value]):
                        continue
                    prev_value = float(ordered[idx - 1]["moo_vector"][name])
                    next_value = float(ordered[idx + 1]["moo_vector"][name])
                    distances[candidate_id_value] += (next_value - prev_value) / scale
        for row in front:
            crowding_distance = float(distances[row["candidate_id"]])
            metadata[row["candidate_id"]] = {
                "rank": int(rank),
                "is_nondominated": bool(rank == 1),
                "crowding_distance": crowding_distance,
                "crowding_sort_value": float(1e9 if np.isinf(crowding_distance) else crowding_distance),
            }
    return metadata


def default_state_specialists_for_router(route_names: tuple[str, ...]) -> list[int]:
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
    return assignments


def build_role_seed_individuals(
    *,
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
    role_pools_by_pair: dict[str, dict[str, tuple[int, ...]]] | None,
) -> list[list[int]]:
    if not role_pools_by_pair:
        return []
    default_execution_gene = dict(derive_execution_profile(None)["gene"])
    route_assignments = default_state_specialists_for_router(route_state_names(ROUTE_STATE_MODE_EQUITY_CORR))
    seeds: list[list[int]] = []
    max_rank_depth = min(
        3,
        max(
            (len(role_pools_by_pair.get(pair, {}).get(role, ())) for pair in pairs for role in SPECIALIST_ROLE_NAMES),
            default=0,
        ),
    )
    for rank in range(max_rank_depth):
        for threshold in route_thresholds:
            pair_configs: dict[str, dict[str, Any]] = {}
            for pair in pairs:
                pools = role_pools_by_pair.get(pair, {})
                specialist_indices: list[int] = []
                for role in SPECIALIST_ROLE_NAMES:
                    pool = tuple(int(v) for v in (pools.get(role) or (0,)))
                    specialist_indices.append(int(pool[min(rank, len(pool) - 1)]))
                mapping_indices = [int(specialist_indices[pos]) for pos in route_assignments]
                pair_configs[pair] = {
                    "route_breadth_threshold": float(threshold),
                    "route_state_mode": ROUTE_STATE_MODE_EQUITY_CORR,
                    "mapping_indices": mapping_indices,
                    "specialist_indices": specialist_indices,
                    "state_specialists": list(route_assignments),
                    "state_specialist_roles": [SPECIALIST_ROLE_NAMES[int(pos)] for pos in route_assignments],
                    "execution_gene": dict(default_execution_gene),
                }
            seeds.append(
                individual_from_pair_configs(
                    pair_configs,
                    pairs=pairs,
                    route_thresholds=route_thresholds,
                )
            )
    return seeds


def build_toolbox(
    *,
    pairs: tuple[str, ...],
    route_thresholds: tuple[float, ...],
    subset_indices: tuple[int, ...],
    role_pools_by_pair: dict[str, dict[str, tuple[int, ...]]] | None,
    rng: random.Random,
) -> base.Toolbox:
    if not hasattr(creator, "FitnessPairwiseMooRouterNSGA2"):
        creator.create(
            "FitnessPairwiseMooRouterNSGA2",
            base.Fitness,
            weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        )
    if not hasattr(creator, "IndividualPairwiseMooRouterNSGA2"):
        creator.create(
            "IndividualPairwiseMooRouterNSGA2",
            list,
            fitness=creator.FitnessPairwiseMooRouterNSGA2,
        )

    role_count = len(SPECIALIST_ROLE_NAMES)
    state_count = 12
    execution_gene_len = len(EXECUTION_GENE_OPTIONS)
    gene_span = 1 + role_count + state_count + execution_gene_len
    toolbox = base.Toolbox()

    def random_threshold_gene() -> int:
        return rng.randrange(len(route_thresholds))

    def specialist_pool(pair_idx: int, role_idx: int) -> tuple[int, ...]:
        if role_pools_by_pair:
            pair_name = pairs[pair_idx]
            role_name = SPECIALIST_ROLE_NAMES[role_idx]
            pool = tuple(int(v) for v in (role_pools_by_pair.get(pair_name, {}).get(role_name) or ()))
            if pool:
                return pool
        return subset_indices

    def random_specialist_gene(pair_idx: int, role_idx: int) -> int:
        pool = specialist_pool(pair_idx, role_idx)
        if len(pool) <= 1:
            return int(pool[0])
        head = pool[: min(len(pool), max(4, len(pool) // 3))]
        if len(head) < len(pool) and rng.random() < 0.70:
            return int(rng.choice(head))
        return int(rng.choice(pool))

    def random_role_assignment_gene() -> int:
        return rng.randrange(role_count)

    def random_execution_gene(gene_idx: int) -> int:
        key = tuple(EXECUTION_GENE_OPTIONS)[gene_idx]
        return rng.randrange(len(EXECUTION_GENE_OPTIONS[key]))

    def init_individual() -> creator.IndividualPairwiseMooRouterNSGA2:
        genes: list[int] = []
        for pair_idx, _ in enumerate(pairs):
            genes.append(random_threshold_gene())
            genes.extend(random_specialist_gene(pair_idx, role_idx) for role_idx in range(role_count))
            genes.extend(random_role_assignment_gene() for _ in range(state_count))
            genes.extend(random_execution_gene(idx) for idx in range(execution_gene_len))
        return creator.IndividualPairwiseMooRouterNSGA2(genes)

    def mate(ind1: list[int], ind2: list[int]) -> tuple[list[int], list[int]]:
        tools.cxUniform(ind1, ind2, indpb=0.5)
        return ind1, ind2

    def mutate(individual: list[int], indpb: float) -> tuple[list[int]]:
        for pair_idx in range(len(pairs)):
            start = pair_idx * gene_span
            for local_idx in range(gene_span):
                if rng.random() >= indpb:
                    continue
                absolute_idx = start + local_idx
                if local_idx == 0:
                    individual[absolute_idx] = random_threshold_gene()
                elif local_idx <= role_count:
                    individual[absolute_idx] = random_specialist_gene(pair_idx, local_idx - 1)
                elif local_idx <= role_count + state_count:
                    individual[absolute_idx] = random_role_assignment_gene()
                else:
                    exec_idx = local_idx - (1 + role_count + state_count)
                    individual[absolute_idx] = random_execution_gene(exec_idx)
        return (individual,)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate, indpb=0.20)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("clone", copy.deepcopy)
    return toolbox


def main() -> None:
    faulthandler.enable()
    args = parse_args()
    started = perf_counter()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    pairs = parse_csv_tuple(args.pairs, str)
    subset_indices = parse_csv_tuple(args.subset_indices, int) if args.subset_indices else None
    route_thresholds = parse_csv_tuple(args.route_thresholds, float)
    fast_engine = resolve_fast_engine(args.fast_engine)

    baseline_summary = json.loads(Path(args.baseline_summary).read_text())
    baseline_pair_configs = build_baseline_pair_configs(pairs, baseline_summary)
    for cfg in baseline_pair_configs.values():
        threshold = float(cfg["route_breadth_threshold"])
        if threshold not in route_thresholds:
            route_thresholds = tuple(sorted(set(route_thresholds + (threshold,))))

    _, compact_library, _ = resolve_candidate(Path(args.base_summary), None, None)
    if args.library_source == "full-grid":
        library = list(iter_params())
        full_index_by_params = {params: idx for idx, params in enumerate(library)}
        compact_subset_indices = tuple(int(full_index_by_params[params]) for params in compact_library)
    else:
        library = compact_library
        compact_subset_indices = tuple(range(len(library)))

    if subset_indices is None:
        subset_indices = compact_subset_indices

    model, _ = load_model(Path(args.model))
    compiled = gp.toolbox.compile(expr=model)
    library_lookup = build_library_lookup(library)

    start_all = SEARCH_WINDOWS[-1][1]
    end_all = SEARCH_WINDOWS[0][2]
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=False)
    print("phase:dataset_loaded", flush=True)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    print("phase:signals_ready", flush=True)
    funding_all = {pair: load_or_fetch_funding(pair, start_all, end_all) for pair in pairs}
    print("phase:funding_ready", flush=True)

    prepare_started = perf_counter()
    window_cache: dict[str, dict[str, Any]] = {}
    for label, start, end in SEARCH_WINDOWS:
        df = df_all.loc[start:end].copy()
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
                    route_state_mode=ROUTE_STATE_MODE_EQUITY_CORR,
                ),
            }
        window_cache[label] = {
            "start": start,
            "end": end,
            "df": df,
            "pairs": pair_cache,
        }
    prepare_seconds = perf_counter() - prepare_started
    print("phase:window_cache_ready", flush=True)

    baseline_realistic: dict[str, Any] = {}
    baseline_candidate = {
        "candidate_kind": "pairwise_candidate",
        "pair_configs": baseline_pair_configs,
    }
    baseline_started = perf_counter()
    for label, start, end in SEARCH_WINDOWS:
        window_data = window_cache[label]
        df = window_data["df"]
        per_pair = {}
        for pair in pairs:
            per_pair[pair] = replay_candidate_from_context(
                candidate=baseline_candidate,
                pair=pair,
                context=window_data["pairs"][pair]["fast_context"],
                library_lookup=library_lookup,
                route_thresholds=route_thresholds,
                leaf_runtime_array=None,
                leaf_codes=None,
            )
        baseline_realistic[label] = {
            "start": start,
            "end": end,
            "bars": int(len(df)),
            "per_pair": per_pair,
            "aggregate": aggregate_metrics(per_pair),
        }
    baseline_seconds = perf_counter() - baseline_started
    print("phase:baseline_ready", flush=True)

    role_pools_by_pair = None
    if args.library_source == "full-grid":
        role_pools_by_pair = build_role_specific_specialist_pools(
            pairs=pairs,
            library=library,
            library_lookup=library_lookup,
            window_cache=window_cache,
            baseline_pair_configs=baseline_pair_configs,
            fast_engine=fast_engine,
            pool_size=int(args.role_pool_size),
            candidate_indices=subset_indices,
        )
    print("phase:role_pools_ready", flush=True)

    fast_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    def evaluate_fast(individual: list[int]) -> tuple[float, ...]:
        key = tuple(int(v) for v in individual)
        cached = fast_cache.get(key)
        if cached is None:
            candidate = encode_candidate_from_individual(
                individual,
                pairs=pairs,
                route_thresholds=route_thresholds,
            )
            windows = {}
            slippage_sensitivity_proxy = 0.0
            turnover_values: list[float] = []
            for label, _, _ in SEARCH_WINDOWS:
                window_data = window_cache[label]
                per_pair = {}
                for pair in pairs:
                    cfg = candidate["pair_configs"][pair]
                    execution_profile = derive_execution_profile(cfg["execution_gene"])
                    state_specialists = tuple(int(v) for v in cfg.get("state_specialists") or ())
                    pair_data = window_data["pairs"][pair]
                    per_pair[pair] = summarize_single_fast_result(
                        fast_overlay_replay_from_context(
                            pair_data["fast_context"],
                            library,
                            library_lookup,
                            tuple(int(v) for v in cfg["mapping_indices"]),
                            float(cfg["route_breadth_threshold"]),
                            fast_engine,
                            commission_pct=float(execution_profile["fast_commission_pct"]),
                            no_trade_band_pct=float(execution_profile["no_trade_band_pct"]),
                            signal_gate_pct=float(execution_profile["signal_gate_pct"]),
                            regime_buffer_mult=float(execution_profile["regime_buffer_mult"]),
                            confirm_bars=int(execution_profile["confirm_bars"]),
                            execution_gene=cfg["execution_gene"],
                            state_specialists=state_specialists,
                        )
                    )
                    turnover_values.append(float(per_pair[pair]["n_trades"]))
                    if label == "recent_6m":
                        stressed_profile = build_stressed_execution_profile(cfg["execution_gene"])
                        stressed_result = summarize_single_fast_result(
                            fast_overlay_replay_from_context(
                                pair_data["fast_context"],
                                library,
                                library_lookup,
                                tuple(int(v) for v in cfg["mapping_indices"]),
                                float(cfg["route_breadth_threshold"]),
                                fast_engine,
                            commission_pct=float(stressed_profile["fast_commission_pct"]),
                            no_trade_band_pct=float(stressed_profile["no_trade_band_pct"]),
                            signal_gate_pct=float(stressed_profile["signal_gate_pct"]),
                            regime_buffer_mult=float(stressed_profile["regime_buffer_mult"]),
                            confirm_bars=int(stressed_profile["confirm_bars"]),
                            execution_gene=cfg["execution_gene"],
                            state_specialists=state_specialists,
                        )
                    )
                        slippage_sensitivity_proxy += max(
                            0.0,
                            float(per_pair[pair]["total_return"]) - float(stressed_result["total_return"]),
                        )
                windows[label] = {
                    "per_pair": per_pair,
                    "aggregate": aggregate_metrics(per_pair),
                }
            bnb_full_4y = pair_metric_or_default(
                windows,
                label="full_4y",
                pair="BNBUSDT",
                metric="avg_daily_return",
                default_metric="worst_pair_avg_daily_return",
            )
            reserve = float(
                min(
                    float(unwrap_window_aggregate(windows["recent_2m"])["worst_pair_avg_daily_return"]),
                    float(unwrap_window_aggregate(windows["recent_4m"])["worst_pair_avg_daily_return"]),
                    float(unwrap_window_aggregate(windows["recent_6m"])["worst_pair_avg_daily_return"]),
                    float(unwrap_window_aggregate(windows["recent_1y"])["worst_pair_avg_daily_return"]),
                    float(unwrap_window_aggregate(windows["full_4y"])["worst_pair_avg_daily_return"]),
                    float(bnb_full_4y),
                )
            )
            diversity_score = compute_diversity_score(candidate, pairs)
            turnover_proxy = float(np.mean(np.asarray(turnover_values, dtype="float64"))) if turnover_values else 0.0
            stability_penalty = compute_stability_penalty(windows)
            objectives = (
                reserve,
                float(unwrap_window_aggregate(windows["recent_2m"])["worst_pair_avg_daily_return"]),
                float(unwrap_window_aggregate(windows["recent_6m"])["worst_pair_avg_daily_return"]),
                float(unwrap_window_aggregate(windows["recent_1y"])["worst_pair_avg_daily_return"]),
                float(unwrap_window_aggregate(windows["full_4y"])["worst_pair_avg_daily_return"]),
                float(diversity_score),
                abs(float(unwrap_window_aggregate(windows["recent_6m"])["worst_max_drawdown"])),
                abs(float(unwrap_window_aggregate(windows["recent_1y"])["worst_max_drawdown"])),
                abs(float(unwrap_window_aggregate(windows["full_4y"])["worst_max_drawdown"])),
                float(turnover_proxy),
                float(slippage_sensitivity_proxy),
                float(stability_penalty),
            )
            cached = {
                "candidate": candidate,
                "candidate_id": candidate_id(candidate, pairs),
                "objectives": objectives,
                "windows": windows,
                "reserve": reserve,
                "diversity_score": diversity_score,
                "turnover_proxy": turnover_proxy,
                "slippage_sensitivity_proxy": slippage_sensitivity_proxy,
                "stability_penalty": stability_penalty,
            }
            cached["scalar_score"] = fast_candidate_sort_score(cached)
            fast_cache[key] = cached
        return cached["objectives"]

    toolbox = build_toolbox(
        pairs=pairs,
        route_thresholds=route_thresholds,
        subset_indices=subset_indices,
        role_pools_by_pair=role_pools_by_pair,
        rng=rng,
    )
    toolbox.register("evaluate", evaluate_fast)
    print("phase:toolbox_ready", flush=True)

    baseline_seed_pair_configs = build_baseline_seed_pair_configs(pairs, baseline_pair_configs)
    baseline_seed_genes = individual_from_pair_configs(
        baseline_seed_pair_configs,
        pairs=pairs,
        route_thresholds=route_thresholds,
    )
    role_seed_genes = build_role_seed_individuals(
        pairs=pairs,
        route_thresholds=route_thresholds,
        role_pools_by_pair=role_pools_by_pair,
    )

    island_count = max(1, int(args.islands))
    island_sizes = [args.population // island_count] * island_count
    for idx in range(args.population % island_count):
        island_sizes[idx] += 1
    islands = [toolbox.population(n=max(1, size)) for size in island_sizes]
    print("phase:population_initialized", flush=True)
    if islands and baseline_seed_genes is not None:
        islands[0][0] = creator.IndividualPairwiseMooRouterNSGA2(baseline_seed_genes)
    seeded_individuals = ([baseline_seed_genes] if baseline_seed_genes is not None else []) + role_seed_genes
    seed_cursor = 0
    for island in islands:
        for idx in range(len(island)):
            if seed_cursor >= len(seeded_individuals):
                break
            island[idx] = creator.IndividualPairwiseMooRouterNSGA2(seeded_individuals[seed_cursor])
            seed_cursor += 1

    search_started = perf_counter()
    for island in islands:
        invalid = [ind for ind in island if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit
        island[:] = toolbox.select(island, k=len(island))
    print("phase:initial_population_evaluated", flush=True)

    for generation in range(args.generations):
        print(f"phase:generation_start:{generation + 1}", flush=True)
        for island_idx, island in enumerate(islands):
            elites = tools.selBest(island, k=min(len(island), max(0, args.elite_count)))
            offspring = algorithms.varOr(
                island,
                toolbox,
                lambda_=max(1, int(args.offspring)),
                cxpb=args.cxpb,
                mutpb=args.mutpb,
            )
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
                ind.fitness.values = fit
            islands[island_idx] = toolbox.select(island + offspring + elites, k=len(island))
        if island_count > 1 and (generation + 1) % max(1, int(args.migration_interval)) == 0:
            migrants_per_island = []
            for island in islands:
                migrants = [toolbox.clone(ind) for ind in tools.selBest(island, k=min(len(island), max(1, int(args.migrants))))]
                migrants_per_island.append(migrants)
            for island_idx in range(island_count):
                incoming = [toolbox.clone(ind) for ind in migrants_per_island[(island_idx - 1) % island_count]]
                islands[island_idx] = toolbox.select(islands[island_idx] + incoming, k=len(islands[island_idx]))
    search_seconds = perf_counter() - search_started
    print("phase:search_complete", flush=True)

    population = [ind for island in islands for ind in island]
    unique_fast_candidates: dict[tuple[int, ...], dict[str, Any]] = {}
    for ind in population:
        key = tuple(int(v) for v in ind)
        unique_fast_candidates[key] = fast_cache[key]
    validation_pool_limit = min(len(unique_fast_candidates), max(int(args.top_k_realistic) * 2, 12))

    fronts = tools.sortNondominated(population, len(population), first_front_only=False)
    validation_pool: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for front in fronts:
        ordered_front = sorted(
            (fast_cache[tuple(int(v) for v in ind)] for ind in front),
            key=lambda item: item["scalar_score"],
            reverse=True,
        )
        for item in ordered_front:
            if item["candidate_id"] in seen_ids:
                continue
            validation_pool.append(
                {
                    "candidate_id": item["candidate_id"],
                    "pair_configs": copy.deepcopy(item["candidate"]["pair_configs"]),
                    "candidate_kind": "pairwise_candidate",
                    "windows": item["windows"],
                    "objectives": list(item["objectives"]),
                    "reserve": float(item["reserve"]),
                    "diversity_score": float(item["diversity_score"]),
                    "turnover_proxy": float(item["turnover_proxy"]),
                    "slippage_sensitivity_proxy": float(item["slippage_sensitivity_proxy"]),
                    "stability_penalty": float(item["stability_penalty"]),
                    "scalar_score": float(item["scalar_score"]),
                }
            )
            seen_ids.add(item["candidate_id"])
            if len(validation_pool) >= validation_pool_limit:
                break
        if len(validation_pool) >= validation_pool_limit:
            break
    print(f"phase:validation_pool_ready:{len(validation_pool)}", flush=True)

    validation_frames_by_key: dict[str, Any] = {}
    validation_state_payloads: dict[str, Any] = {}
    validation_inputs_by_key: dict[str, Any] = {}
    for item in validation_pool:
        candidate = {
            "candidate_kind": "pairwise_candidate",
            "pair_configs": item["pair_configs"],
        }
        validation_input = build_candidate_validation_input(
            candidate,
            pairs=pairs,
            window_cache=window_cache,
            library=library,
            library_lookup=library_lookup,
        )
        key = item["candidate_id"]
        validation_inputs_by_key[key] = validation_input
        validation_frames_by_key[key] = build_return_frame(validation_input["daily_returns"], validation_input["daily_index"])
        validation_state_payloads[key] = validation_input["state_payload"]
    print(f"phase:validation_inputs_ready:{len(validation_inputs_by_key)}", flush=True)

    realistic_top: list[dict[str, Any]] = []
    realistic_started = perf_counter()
    for item in validation_pool[: max(int(args.top_k_realistic), 1)]:
        pair_configs = item["pair_configs"]
        windows = {}
        slippage_sensitivity = 0.0
        for label, start, end in SEARCH_WINDOWS:
            window_data = window_cache[label]
            per_pair = {}
            for pair in pairs:
                cfg = pair_configs[pair]
                per_pair[pair] = replay_candidate_from_context(
                    candidate=candidate,
                    pair=pair,
                    context=window_data["pairs"][pair]["fast_context"],
                    library_lookup=library_lookup,
                    route_thresholds=route_thresholds,
                    leaf_runtime_array=None,
                    leaf_codes=None,
                )
                if label == "recent_6m":
                    stressed_cfg = dict(cfg)
                    stressed_cfg["execution_gene"] = build_stressed_execution_profile(cfg.get("execution_gene") or {}).get("gene") or cfg.get("execution_gene")
                    stressed_result = realistic_overlay_replay_from_context(
                        window_data["pairs"][pair]["fast_context"],
                        library_lookup,
                        tuple(int(v) for v in cfg["mapping_indices"]),
                        float(cfg["route_breadth_threshold"]),
                        execution_gene=stressed_cfg.get("execution_gene"),
                        state_specialists=tuple(int(v) for v in cfg.get("state_specialists") or ()),
                        engine="python",
                    )
                    slippage_sensitivity += max(
                        0.0,
                        float(per_pair[pair]["total_return"]) - float(stressed_result["total_return"]),
                    )
            windows[label] = {
                "start": start,
                "end": end,
                "bars": int(len(window_data["df"])),
                "per_pair": per_pair,
                "aggregate": aggregate_metrics(per_pair),
            }
        key = item["candidate_id"]
        validation_input = validation_inputs_by_key[key]
        validation_engine = build_candidate_validation_bundle(
            key,
            validation_input["daily_returns"],
            validation_input["daily_index"],
            trial_count=max(len(validation_pool), 1),
            peer_frames_by_key=validation_frames_by_key,
            state_payload=validation_state_payloads[key],
            cost_reference=build_candidate_cost_reference(windows),
        )
        parameter_sensitivity_penalty = compute_neighbor_sensitivity_penalty(
            {
                "candidate_kind": "pairwise_candidate",
                "pair_configs": pair_configs,
            },
            pairs=pairs,
            route_thresholds=route_thresholds,
            window_cache=window_cache,
            library=library,
            library_lookup=library_lookup,
            fast_engine=fast_engine,
        )
        row = {
            "candidate_kind": "pairwise_candidate",
            "candidate_id": key,
            "pair_configs": pair_configs,
            "windows": windows,
            "validation": build_validation_bundle(windows, baseline_realistic),
            "validation_engine": validation_engine,
            "validation_robustness": (
                validation_engine.get("robustness")
                or build_validation_robustness_profile(validation_engine)
            ),
            "state_summary": (validation_engine.get("market_operating_system") or {}).get("state_summary") or {},
            "scalar_score": float(item["scalar_score"]),
            "reserve": float(item["reserve"]),
            "diversity_score": float(item["diversity_score"]),
            "slippage_sensitivity": float(slippage_sensitivity),
            "parameter_sensitivity_penalty": float(parameter_sensitivity_penalty),
        }
        row["moo_vector"] = moo_raw_metrics(row)
        row["score"] = realistic_selection_score(row)
        row["pareto_vector"] = build_pairwise_pareto_vector(row)
        realistic_top.append(row)
    realistic_seconds = perf_counter() - realistic_started
    print("phase:realistic_top_ready", flush=True)

    pareto_metadata = assign_moo_pareto_metadata(realistic_top)
    pairwise_pareto_metadata = assign_pairwise_pareto_metadata(realistic_top) if realistic_top else {}
    for row in realistic_top:
        row["moo_pareto"] = pareto_metadata.get(row["candidate_id"], {})
        row["pareto"] = pairwise_pareto_metadata.get(row["candidate_id"], {})
    realistic_top.sort(
        key=lambda item: (
            -(item.get("moo_pareto") or {}).get("is_nondominated", False),
            float((item.get("moo_pareto") or {}).get("rank", 999.0)),
            -float((item.get("moo_pareto") or {}).get("crowding_sort_value", 0.0)),
            -float(item["score"]),
        ),
    )

    validation_pass = [
        item for item in realistic_top
        if bool((item.get("validation_engine") or {}).get("gate", {}).get("passed", False))
    ]
    nondominated_validation_pass = [
        item for item in validation_pass
        if bool((item.get("moo_pareto") or {}).get("is_nondominated", False))
    ]
    selected = None
    selection_reason = "no_gate_pass"
    if nondominated_validation_pass:
        selected = max(nondominated_validation_pass, key=realistic_selection_score)
        selection_reason = "moo_pareto_validation"
    elif validation_pass:
        selected = max(validation_pass, key=realistic_selection_score)
        selection_reason = "validation_only"
    elif realistic_top:
        selected = max(realistic_top, key=realistic_selection_score)
        selection_reason = "fallback_best"
    shadow_challengers = nondominated_validation_pass[:3] if nondominated_validation_pass else realistic_top[:3]

    report = {
        "architecture": {
            "data_layer": [
                "ohlcv",
                "event_bars_planned",
                "derivatives",
                "macro_cross_asset_existing",
            ],
            "validation_layer": [
                "purged_embargo",
                "cpcv",
                "pbo",
                "dsr",
                "stability_test",
            ],
            "strategy_evolution_layer": [
                "nsga_ii",
                "island_model",
                "novelty_diversity",
                "hierarchical_chromosome",
            ],
            "regime_layer": list(SPECIALIST_ROLE_NAMES),
            "execution_layer": [
                "fees",
                "funding",
                "slippage",
                "maker_taker_logic_proxy",
                "partial_fill_proxy",
            ],
            "deployment_layer": [
                "shadow_live_existing",
                "champion_challenger_existing",
                "kill_switch_existing",
                "risk_dashboard_existing",
            ],
        },
        "search": {
            "algorithm": "pairwise_moo_router_nsga2",
            "population": args.population,
            "generations": args.generations,
            "offspring": args.offspring,
            "islands": island_count,
            "migration_interval": args.migration_interval,
            "migrants": args.migrants,
            "elite_count": args.elite_count,
            "seed": args.seed,
            "library_source": args.library_source,
            "library_size": len(library),
            "route_thresholds": list(route_thresholds),
            "subset_indices": list(subset_indices),
            "role_pool_size": int(args.role_pool_size),
            "route_state_mode": ROUTE_STATE_MODE_EQUITY_CORR,
            "specialist_roles": list(SPECIALIST_ROLE_NAMES),
        },
        "pairs": list(pairs),
        "model_path": str(args.model),
        "baseline_summary_path": str(args.baseline_summary),
        "base_summary_path": str(args.base_summary),
        "baseline_candidate": {
            "candidate_kind": "pairwise_candidate",
            "pair_configs": baseline_pair_configs,
            "seeded_into_population": True,
            "baseline_seed_pair_configs": baseline_seed_pair_configs,
        },
        "baseline_realistic": baseline_realistic,
        "role_pools_by_pair": role_pools_by_pair,
        "top_fast_candidates": validation_pool,
        "realistic_top_candidates": realistic_top,
        "selection": {
            "reason": selection_reason,
            "validation_pass_count": len(validation_pass),
            "nondominated_validation_pass_count": len(nondominated_validation_pass),
            "realistic_top_count": len(realistic_top),
        },
        "shadow_challenger_candidates": shadow_challengers,
        "runtime": {
            "fast_engine": fast_engine,
            "numba_available": NUMBA_AVAILABLE,
            "prepare_context_seconds": prepare_seconds,
            "baseline_seconds": baseline_seconds,
            "search_seconds": search_seconds,
            "realistic_seconds": realistic_seconds,
            "total_seconds": perf_counter() - started,
            "evaluated_unique_candidates": len(fast_cache),
        },
        "selected_candidate": selected,
        "fallback_best_candidate": max(realistic_top, key=realistic_selection_score) if realistic_top else None,
        "created_at": datetime.now(UTC).isoformat(),
    }

    out_path = Path(args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print("phase:summary_written", flush=True)
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


def summarize_single_fast_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "avg_daily_return": float(result["daily_metrics"]["avg_daily_return"]),
        "daily_target_hit_rate": float(result["daily_metrics"]["daily_target_hit_rate"]),
        "daily_win_rate": float(result["daily_metrics"]["daily_win_rate"]),
        "worst_day": float(result["daily_metrics"]["worst_day"]),
        "best_day": float(result["daily_metrics"]["best_day"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]),
        "n_trades": int(result["n_trades"]),
    }


if __name__ == "__main__":
    main()
