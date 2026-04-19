from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from execution_gene_utils import blended_entry_quality_score, dc_alignment_score


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def breakout_support_score(
    *,
    order_imbalance: float,
    buy_volume_share: float,
    close_location_value: float,
    body_to_range: float,
    wick_skew: float,
    oi_rel: float,
    basis_rate: float,
    top_pos_log_ratio: float,
    taker_buy_sell_log_ratio: float,
    dc_trend_05: float,
    dc_run_05: float,
    range_bps: float,
    volume_ratio: float,
) -> float:
    entry_quality = blended_entry_quality_score(
        order_imbalance,
        buy_volume_share,
        close_location_value,
        body_to_range,
        wick_skew,
        oi_rel,
        basis_rate,
        top_pos_log_ratio,
        taker_buy_sell_log_ratio,
    )
    dc_score = dc_alignment_score(dc_trend_05, dc_run_05)
    volume_score = _clip((float(volume_ratio) - 1.0) / 1.5, -1.0, 1.0)
    # Breakout entries should reward expansion, unlike liquidity filters that reject it.
    range_expansion_score = _clip((float(range_bps) - 25.0) / 75.0, -1.0, 1.0)
    return _clip(
        0.45 * float(entry_quality)
        + 0.25 * float(dc_score)
        + 0.20 * float(volume_score)
        + 0.10 * float(range_expansion_score),
        -1.0,
        1.0,
    )


def build_breakout_override_trace(
    *,
    context: Mapping[str, Any],
    baseline_trace: Mapping[str, Any],
    alpha: float,
    breakout_weight: float,
    activation_mode: str,
    lookback_bars: int,
    breakout_buffer_bps: float,
    support_floor: float,
    volume_ratio_floor: float,
) -> dict[str, np.ndarray]:
    close = np.asarray(context["close"], dtype="float64")
    baseline_target = np.asarray(baseline_trace["target_weight"], dtype="float64")
    target = baseline_target.copy()
    alpha_trace = np.zeros_like(target)
    support_trace = np.zeros_like(target)
    breakout_side_trace = np.zeros_like(target)

    lookback = max(int(lookback_bars), 2)
    buffer_mult = float(breakout_buffer_bps) / 10_000.0
    alpha = float(alpha)
    breakout_weight = float(abs(breakout_weight))
    support_floor = float(support_floor)
    volume_ratio_floor = float(volume_ratio_floor)

    for i in range(len(target)):
        left = max(0, i - lookback)
        if i - left < 2:
            continue
        recent = close[left:i]
        prev_high = float(np.nanmax(recent))
        prev_low = float(np.nanmin(recent))
        price = float(close[i])
        volume_ratio = float(context["volume_ratio"][i])
        support = breakout_support_score(
            order_imbalance=float(context["order_imbalance"][i]),
            buy_volume_share=float(context["buy_volume_share"][i]),
            close_location_value=float(context["close_location_value"][i]),
            body_to_range=float(context["body_to_range"][i]),
            wick_skew=float(context["wick_skew"][i]),
            oi_rel=float(context["oi_rel"][i]),
            basis_rate=float(context["basis_rate"][i]),
            top_pos_log_ratio=float(context["top_pos_log_ratio"][i]),
            taker_buy_sell_log_ratio=float(context["taker_buy_sell_log_ratio"][i]),
            dc_trend_05=float(context["dc_trend_05"][i]),
            dc_run_05=float(context["dc_run_05"][i]),
            range_bps=float(context["range_bps"][i]),
            volume_ratio=volume_ratio,
        )
        support_trace[i] = support
        breakout_side = 0.0
        if (
            volume_ratio >= volume_ratio_floor
            and price >= prev_high * (1.0 + buffer_mult)
            and support >= support_floor
        ):
            breakout_side = 1.0
        elif (
            volume_ratio >= volume_ratio_floor
            and price <= prev_low * (1.0 - buffer_mult)
            and support <= -support_floor
        ):
            breakout_side = -1.0
        breakout_side_trace[i] = breakout_side
        if breakout_side == 0.0:
            continue

        base_weight = float(baseline_target[i])
        base_side = 0.0 if abs(base_weight) <= 1e-12 else float(np.sign(base_weight))
        if activation_mode == "flat_only" and base_side != 0.0:
            continue
        if activation_mode == "flat_or_disagree" and base_side != 0.0 and base_side == breakout_side:
            continue
        if activation_mode not in {"flat_only", "flat_or_disagree"}:
            raise ValueError(f"unsupported activation mode: {activation_mode}")

        override_weight = breakout_side * breakout_weight
        alpha_trace[i] = alpha
        target[i] = (1.0 - alpha) * base_weight + alpha * override_weight

    return {
        "target_weight": target,
        "alpha": alpha_trace,
        "support": support_trace,
        "breakout_side": breakout_side_trace,
    }
