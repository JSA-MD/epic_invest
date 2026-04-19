from __future__ import annotations

from typing import Any, Mapping

import gp_crypto_evolution as gp


DEFAULT_MAKER_FEE_RATE = 0.0002
DEFAULT_TAKER_FEE_RATE = 0.0004
DEFAULT_BASE_SLIPPAGE = 0.0002
DEFAULT_AMOUNT_STEP = 0.001
DEFAULT_MIN_QTY = 0.001

SPECIALIST_ROLE_NAMES: tuple[str, ...] = ("trend", "range", "panic", "carry")
ROLE_INDEX_BY_NAME: dict[str, int] = {name: idx for idx, name in enumerate(SPECIALIST_ROLE_NAMES)}


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def microstructure_alignment_score(order_imbalance: float, buy_volume_share: float) -> float:
    return _clip(
        0.70 * float(order_imbalance) + 0.30 * (2.0 * float(buy_volume_share) - 1.0),
        -1.0,
        1.0,
    )


def candle_microstructure_proxy_score(close_location_value: float, body_to_range: float, wick_skew: float) -> float:
    directional_body = (1.0 if float(close_location_value) >= 0.0 else -1.0) * float(body_to_range)
    return _clip(
        0.50 * float(close_location_value) + 0.30 * directional_body + 0.20 * float(wick_skew),
        -1.0,
        1.0,
    )


def blended_microstructure_score(
    order_imbalance: float,
    buy_volume_share: float,
    close_location_value: float,
    body_to_range: float,
    wick_skew: float,
) -> float:
    flow_score = microstructure_alignment_score(order_imbalance, buy_volume_share)
    candle_score = candle_microstructure_proxy_score(close_location_value, body_to_range, wick_skew)
    return _clip(0.65 * flow_score + 0.35 * candle_score, -1.0, 1.0)


def derivative_positioning_score(
    oi_rel: float,
    basis_rate: float,
    top_pos_log_ratio: float,
    taker_buy_sell_log_ratio: float,
) -> float:
    oi_component = _clip((float(oi_rel) - 1.0) / 0.10, -1.0, 1.0)
    basis_component = _clip(float(basis_rate) / 0.0015, -1.0, 1.0)
    return _clip(
        0.35 * float(top_pos_log_ratio)
        + 0.35 * float(taker_buy_sell_log_ratio)
        + 0.20 * oi_component
        + 0.10 * basis_component,
        -1.0,
        1.0,
    )


def blended_entry_quality_score(
    order_imbalance: float,
    buy_volume_share: float,
    close_location_value: float,
    body_to_range: float,
    wick_skew: float,
    oi_rel: float,
    basis_rate: float,
    top_pos_log_ratio: float,
    taker_buy_sell_log_ratio: float,
) -> float:
    flow_score = blended_microstructure_score(
        order_imbalance,
        buy_volume_share,
        close_location_value,
        body_to_range,
        wick_skew,
    )
    positioning_score = derivative_positioning_score(
        oi_rel,
        basis_rate,
        top_pos_log_ratio,
        taker_buy_sell_log_ratio,
    )
    return _clip(0.75 * flow_score + 0.25 * positioning_score, -1.0, 1.0)


def dc_alignment_score(dc_trend_05: float, dc_run_05: float) -> float:
    return _clip(
        0.75 * float(dc_trend_05) + 0.25 * float(dc_run_05),
        -1.0,
        1.0,
    )


def should_abstain_for_alignment(
    requested_side: int,
    microstructure_score: float,
    dc_score: float,
    microstructure_align_gate_pct: float,
    dc_align_gate_pct: float,
    min_alignment_votes: int = 2,
) -> bool:
    if requested_side == 0:
        return True
    if int(min_alignment_votes) <= 0:
        return False
    votes = 0
    if float(requested_side) * float(microstructure_score) >= float(microstructure_align_gate_pct):
        votes += 1
    if float(requested_side) * float(dc_score) >= float(dc_align_gate_pct):
        votes += 1
    return votes < max(1, min(int(min_alignment_votes), 2))


def should_abstain_for_liquidity(
    requested_side: int,
    current_weight: float,
    range_bps: float,
    volume_ratio: float,
    liquidity_range_gate_bp: float,
    liquidity_volume_ratio_floor: float,
) -> bool:
    if requested_side == 0:
        return False
    if float(current_weight) * float(requested_side) > 0.0:
        return False
    range_gate_enabled = float(liquidity_range_gate_bp) > 0.0
    volume_gate_enabled = float(liquidity_volume_ratio_floor) > 0.0
    if not range_gate_enabled and not volume_gate_enabled:
        return False
    range_bad = range_gate_enabled and float(range_bps) >= float(liquidity_range_gate_bp)
    volume_bad = volume_gate_enabled and float(volume_ratio) <= float(liquidity_volume_ratio_floor)
    if range_gate_enabled and volume_gate_enabled:
        return bool(range_bad and volume_bad)
    return bool(range_bad or volume_bad)


def should_abstain_for_weak_tape(
    requested_side: int,
    current_weight: float,
    microstructure_score: float,
    dc_score: float,
    range_bps: float,
    volume_ratio: float,
    equity_corr_gross_scale: float,
    short_horizon_abstain_mult: float,
) -> bool:
    if requested_side == 0:
        return False
    if float(current_weight) * float(requested_side) > 0.0:
        return False
    strength = float(short_horizon_abstain_mult)
    if strength <= 0.0:
        return False
    weak_votes = 0
    if float(requested_side) * float(microstructure_score) < 0.15 * strength:
        weak_votes += 1
    if float(requested_side) * float(dc_score) < 0.10 * strength:
        weak_votes += 1
    if float(range_bps) <= 45.0 and float(volume_ratio) <= 1.0:
        weak_votes += 1
    if float(equity_corr_gross_scale) < max(0.75, 1.0 - 0.10 * strength):
        weak_votes += 1
    return weak_votes >= 2


def should_allow_countertrend_entry(
    requested_side: int,
    signal_pct: float,
    regime_score: float,
    breadth_score: float,
    breadth_threshold: float,
    microstructure_score: float,
    positioning_score: float,
    dc_score: float,
    range_bps: float,
    volume_ratio: float,
    signal_floor_pct: float,
    regime_score_cap: float,
    breadth_slack: float,
    microstructure_floor: float,
    positioning_floor: float,
    dc_floor: float,
    range_bps_floor: float,
    volume_ratio_floor: float,
) -> bool:
    if int(requested_side) == 0:
        return False
    if float(signal_floor_pct) <= 0.0:
        return False
    if abs(float(signal_pct)) < float(signal_floor_pct):
        return False
    if float(range_bps) < float(range_bps_floor):
        return False
    if float(volume_ratio) < float(volume_ratio_floor):
        return False

    side = float(requested_side)
    if side > 0.0:
        if float(regime_score) < -float(regime_score_cap):
            return False
        if float(breadth_score) < max(0.0, float(breadth_threshold) - float(breadth_slack)):
            return False
    else:
        if float(regime_score) > float(regime_score_cap):
            return False
        if float(breadth_score) > min(1.0, (1.0 - float(breadth_threshold)) + float(breadth_slack)):
            return False

    if side * float(microstructure_score) < float(microstructure_floor):
        return False
    if side * float(positioning_score) < float(positioning_floor):
        return False
    if side * float(dc_score) < float(dc_floor):
        return False
    return True




def normalize_execution_gene(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    flow_alignment_threshold = _clip(
        float(payload.get("flow_alignment_threshold", payload.get("microstructure_align_gate_pct", 0.0))),
        0.0,
        1.0,
    )
    dc_alignment_threshold = _clip(
        float(payload.get("dc_alignment_threshold", payload.get("dc_align_gate_pct", 0.0))),
        0.0,
        1.0,
    )
    min_alignment_votes = max(min(int(payload.get("min_alignment_votes", 0)), 2), 0)
    role_signal_gate_mults = tuple(
        _clip(float(payload.get(f"{role}_signal_gate_mult", 1.0)), 0.35, 2.0)
        for role in SPECIALIST_ROLE_NAMES
    )
    role_regime_buffer_mults = tuple(
        _clip(float(payload.get(f"{role}_regime_buffer_mult", 1.0)), 0.35, 2.0)
        for role in SPECIALIST_ROLE_NAMES
    )
    return {
        "maker_priority": _clip(float(payload.get("maker_priority", 0.55)), 0.0, 1.0),
        "max_wait_bars": max(int(payload.get("max_wait_bars", 1)), 0),
        "chase_distance_bp": max(float(payload.get("chase_distance_bp", 2.0)), 0.0),
        "cancel_replace_interval_bars": max(int(payload.get("cancel_replace_interval_bars", 1)), 1),
        "partial_fill_tolerance": _clip(float(payload.get("partial_fill_tolerance", 0.60)), 0.05, 1.0),
        "emergency_market_threshold_bp": max(float(payload.get("emergency_market_threshold_bp", 25.0)), 1.0),
        "signal_gate_pct": _clip(float(payload.get("signal_gate_pct", 0.0)), 0.0, 2.0),
        "regime_buffer_mult": _clip(float(payload.get("regime_buffer_mult", 0.0)), 0.0, 1.5),
        "confirm_bars": max(int(payload.get("confirm_bars", 1)), 1),
        "abstain_edge_pct": _clip(float(payload.get("abstain_edge_pct", 0.0)), 0.0, 1.5),
        "specialist_isolation_mult": _clip(float(payload.get("specialist_isolation_mult", 0.0)), 0.0, 1.5),
        "liquidity_range_gate_bp": max(float(payload.get("liquidity_range_gate_bp", 0.0)), 0.0),
        "liquidity_volume_ratio_floor": _clip(float(payload.get("liquidity_volume_ratio_floor", 0.0)), 0.0, 2.0),
        "short_horizon_abstain_mult": _clip(float(payload.get("short_horizon_abstain_mult", 0.0)), 0.0, 1.5),
        "countertrend_weight_scale": _clip(float(payload.get("countertrend_weight_scale", 0.0)), 0.0, 1.0),
        "countertrend_rebalance_bypass": bool(payload.get("countertrend_rebalance_bypass", False)),
        "countertrend_signal_floor_pct": _clip(float(payload.get("countertrend_signal_floor_pct", 0.0)), 0.0, 500.0),
        "countertrend_regime_score_cap": _clip(float(payload.get("countertrend_regime_score_cap", 0.0)), 0.0, 0.25),
        "countertrend_breadth_slack": _clip(float(payload.get("countertrend_breadth_slack", 0.0)), 0.0, 0.5),
        "countertrend_microstructure_floor": _clip(float(payload.get("countertrend_microstructure_floor", 0.0)), -1.0, 1.0),
        "countertrend_positioning_floor": _clip(float(payload.get("countertrend_positioning_floor", 0.0)), -1.0, 1.0),
        "countertrend_dc_floor": _clip(float(payload.get("countertrend_dc_floor", 0.0)), -1.0, 1.0),
        "countertrend_range_bps_floor": max(float(payload.get("countertrend_range_bps_floor", 0.0)), 0.0),
        "countertrend_volume_ratio_floor": _clip(float(payload.get("countertrend_volume_ratio_floor", 0.0)), 0.0, 3.0),
        "flow_alignment_threshold": flow_alignment_threshold,
        "dc_alignment_threshold": dc_alignment_threshold,
        "min_alignment_votes": min_alignment_votes,
        "microstructure_align_gate_pct": flow_alignment_threshold,
        "dc_align_gate_pct": dc_alignment_threshold,
        "trend_signal_gate_mult": _clip(float(payload.get("trend_signal_gate_mult", 1.0)), 0.35, 2.0),
        "range_signal_gate_mult": _clip(float(payload.get("range_signal_gate_mult", 1.0)), 0.35, 2.0),
        "panic_signal_gate_mult": _clip(float(payload.get("panic_signal_gate_mult", 1.0)), 0.35, 2.0),
        "carry_signal_gate_mult": _clip(float(payload.get("carry_signal_gate_mult", 1.0)), 0.35, 2.0),
        "trend_regime_buffer_mult": _clip(float(payload.get("trend_regime_buffer_mult", 1.0)), 0.35, 2.0),
        "range_regime_buffer_mult": _clip(float(payload.get("range_regime_buffer_mult", 1.0)), 0.35, 2.0),
        "panic_regime_buffer_mult": _clip(float(payload.get("panic_regime_buffer_mult", 1.0)), 0.35, 2.0),
        "carry_regime_buffer_mult": _clip(float(payload.get("carry_regime_buffer_mult", 1.0)), 0.35, 2.0),
        "role_signal_gate_mults": role_signal_gate_mults,
        "role_regime_buffer_mults": role_regime_buffer_mults,
    }


def derive_execution_profile(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    gene = normalize_execution_gene(raw)
    maker_priority = float(gene["maker_priority"])
    chase_distance_bp = float(gene["chase_distance_bp"])
    max_wait_bars = int(gene["max_wait_bars"])
    cancel_replace_interval_bars = int(gene["cancel_replace_interval_bars"])
    partial_fill_tolerance = float(gene["partial_fill_tolerance"])
    emergency_market_threshold_bp = float(gene["emergency_market_threshold_bp"])

    urgency = (
        (1.0 - maker_priority) * 0.60
        + max(0.0, 0.55 - partial_fill_tolerance) * 0.90
        + max(0.0, 20.0 - emergency_market_threshold_bp) / 40.0 * 0.35
    )
    maker_fee_share = maker_priority * partial_fill_tolerance
    fee_rate = (
        DEFAULT_MAKER_FEE_RATE * maker_fee_share
        + DEFAULT_TAKER_FEE_RATE * (1.0 - maker_fee_share)
    )
    slippage = DEFAULT_BASE_SLIPPAGE * (
        1.0
        + urgency
        + chase_distance_bp / 10.0
        + max(0, cancel_replace_interval_bars - 1) * 0.08
    )
    no_trade_band_pct = float(gp.NO_TRADE_BAND) + max_wait_bars * 0.12 + cancel_replace_interval_bars * 0.10
    fast_commission_pct = fee_rate + slippage * 0.50
    fill_confidence = _clip(
        0.55
        + maker_priority * 0.20
        + partial_fill_tolerance * 0.20
        - max_wait_bars * 0.04
        - max(0, cancel_replace_interval_bars - 1) * 0.03,
        0.20,
        0.98,
    )
    return {
        "gene": gene,
        "fee_rate": float(fee_rate),
        "slippage": float(slippage),
        "no_trade_band_pct": float(no_trade_band_pct),
        "amount_step": float(DEFAULT_AMOUNT_STEP),
        "min_qty": float(DEFAULT_MIN_QTY),
        "fast_commission_pct": float(fast_commission_pct),
        "fill_confidence": float(fill_confidence),
        "signal_gate_pct": float(gene["signal_gate_pct"]),
        "regime_buffer_mult": float(gene["regime_buffer_mult"]),
        "confirm_bars": int(gene["confirm_bars"]),
        "abstain_edge_pct": float(gene["abstain_edge_pct"]),
        "specialist_isolation_mult": float(gene["specialist_isolation_mult"]),
        "liquidity_range_gate_bp": float(gene["liquidity_range_gate_bp"]),
        "liquidity_volume_ratio_floor": float(gene["liquidity_volume_ratio_floor"]),
        "short_horizon_abstain_mult": float(gene["short_horizon_abstain_mult"]),
        "countertrend_weight_scale": float(gene["countertrend_weight_scale"]),
        "countertrend_rebalance_bypass": bool(gene["countertrend_rebalance_bypass"]),
        "countertrend_signal_floor_pct": float(gene["countertrend_signal_floor_pct"]),
        "countertrend_regime_score_cap": float(gene["countertrend_regime_score_cap"]),
        "countertrend_breadth_slack": float(gene["countertrend_breadth_slack"]),
        "countertrend_microstructure_floor": float(gene["countertrend_microstructure_floor"]),
        "countertrend_positioning_floor": float(gene["countertrend_positioning_floor"]),
        "countertrend_dc_floor": float(gene["countertrend_dc_floor"]),
        "countertrend_range_bps_floor": float(gene["countertrend_range_bps_floor"]),
        "countertrend_volume_ratio_floor": float(gene["countertrend_volume_ratio_floor"]),
        "flow_alignment_threshold": float(gene["flow_alignment_threshold"]),
        "dc_alignment_threshold": float(gene["dc_alignment_threshold"]),
        "min_alignment_votes": int(gene["min_alignment_votes"]),
        "microstructure_align_gate_pct": float(gene["microstructure_align_gate_pct"]),
        "dc_align_gate_pct": float(gene["dc_align_gate_pct"]),
        "role_signal_gate_mults": tuple(float(v) for v in gene["role_signal_gate_mults"]),
        "role_regime_buffer_mults": tuple(float(v) for v in gene["role_regime_buffer_mults"]),
    }


def legacy_execution_profile() -> dict[str, Any]:
    """Execution settings that preserve legacy pairwise replay semantics."""
    neutral_mults = tuple(1.0 for _ in SPECIALIST_ROLE_NAMES)
    gene = {
        "maker_priority": 0.55,
        "max_wait_bars": 1,
        "chase_distance_bp": 2.0,
        "cancel_replace_interval_bars": 1,
        "partial_fill_tolerance": 0.60,
        "emergency_market_threshold_bp": 25.0,
        "signal_gate_pct": 0.0,
        "regime_buffer_mult": 0.0,
        "confirm_bars": 1,
        "abstain_edge_pct": 0.0,
        "specialist_isolation_mult": 0.0,
        "liquidity_range_gate_bp": 0.0,
        "liquidity_volume_ratio_floor": 0.0,
        "short_horizon_abstain_mult": 0.0,
        "countertrend_weight_scale": 0.0,
        "countertrend_rebalance_bypass": False,
        "countertrend_signal_floor_pct": 0.0,
        "countertrend_regime_score_cap": 0.0,
        "countertrend_breadth_slack": 0.0,
        "countertrend_microstructure_floor": 0.0,
        "countertrend_positioning_floor": 0.0,
        "countertrend_dc_floor": 0.0,
        "countertrend_range_bps_floor": 0.0,
        "countertrend_volume_ratio_floor": 0.0,
        "flow_alignment_threshold": 0.0,
        "dc_alignment_threshold": 0.0,
        "min_alignment_votes": 0,
        "microstructure_align_gate_pct": 0.0,
        "dc_align_gate_pct": 0.0,
        "trend_signal_gate_mult": 1.0,
        "range_signal_gate_mult": 1.0,
        "panic_signal_gate_mult": 1.0,
        "carry_signal_gate_mult": 1.0,
        "trend_regime_buffer_mult": 1.0,
        "range_regime_buffer_mult": 1.0,
        "panic_regime_buffer_mult": 1.0,
        "carry_regime_buffer_mult": 1.0,
        "role_signal_gate_mults": neutral_mults,
        "role_regime_buffer_mults": neutral_mults,
    }
    return {
        "gene": gene,
        "fee_rate": float(DEFAULT_TAKER_FEE_RATE),
        "slippage": float(DEFAULT_BASE_SLIPPAGE),
        "no_trade_band_pct": float(gp.NO_TRADE_BAND),
        "amount_step": float(DEFAULT_AMOUNT_STEP),
        "min_qty": float(DEFAULT_MIN_QTY),
        "fast_commission_pct": float(gp.COMMISSION_PCT),
        "fill_confidence": 1.0,
        "signal_gate_pct": 0.0,
        "regime_buffer_mult": 0.0,
        "confirm_bars": 1,
        "abstain_edge_pct": 0.0,
        "specialist_isolation_mult": 0.0,
        "liquidity_range_gate_bp": 0.0,
        "liquidity_volume_ratio_floor": 0.0,
        "short_horizon_abstain_mult": 0.0,
        "countertrend_weight_scale": 0.0,
        "countertrend_rebalance_bypass": False,
        "countertrend_signal_floor_pct": 0.0,
        "countertrend_regime_score_cap": 0.0,
        "countertrend_breadth_slack": 0.0,
        "countertrend_microstructure_floor": 0.0,
        "countertrend_positioning_floor": 0.0,
        "countertrend_dc_floor": 0.0,
        "countertrend_range_bps_floor": 0.0,
        "countertrend_volume_ratio_floor": 0.0,
        "flow_alignment_threshold": 0.0,
        "dc_alignment_threshold": 0.0,
        "min_alignment_votes": 0,
        "microstructure_align_gate_pct": 0.0,
        "dc_align_gate_pct": 0.0,
        "role_signal_gate_mults": neutral_mults,
        "role_regime_buffer_mults": neutral_mults,
    }


def build_stressed_execution_profile(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    profile = derive_execution_profile(raw)
    return {
        **profile,
        "fee_rate": float(profile["fee_rate"]) * 1.20,
        "slippage": float(profile["slippage"]) * 1.85,
        "fast_commission_pct": float(profile["fast_commission_pct"]) * 1.45,
    }


def extract_pair_execution_gene(candidate: Mapping[str, Any], pair: str) -> dict[str, Any] | None:
    pair_configs = candidate.get("pair_configs") or {}
    pair_config = pair_configs.get(pair) or {}
    if isinstance(pair_config.get("execution_gene"), Mapping):
        return normalize_execution_gene(pair_config.get("execution_gene"))
    execution_genes = candidate.get("execution_genes") or {}
    if isinstance(execution_genes, Mapping) and isinstance(execution_genes.get(pair), Mapping):
        return normalize_execution_gene(execution_genes.get(pair))
    return None
