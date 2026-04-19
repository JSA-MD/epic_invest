from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from btc_breakout_override import breakout_support_score


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    span = max(int(span), 1)
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(values, dtype="float64")
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * out[i - 1]
    return out


def build_event_orderflow_trace(
    *,
    context: Mapping[str, Any],
    baseline_trace: Mapping[str, Any],
    trigger_weight: float,
    fast_span: int,
    slow_span: int,
    retest_lookback: int,
    support_floor: float,
    volume_ratio_floor: float,
    activation_mode: str = "flat_only",
    hold_bars: int = 4,
) -> dict[str, np.ndarray]:
    close = np.asarray(context["close"], dtype="float64")
    baseline_target = np.asarray(baseline_trace["target_weight"], dtype="float64")
    trigger_weight = float(abs(trigger_weight))
    n = len(baseline_target)

    ema_fast = _ema(close, fast_span)
    ema_slow = _ema(close, slow_span)
    target = baseline_target.copy()
    trigger_side = np.zeros(n, dtype="float64")
    support_trace = np.zeros(n, dtype="float64")
    trend_trace = np.zeros(n, dtype="float64")

    hold_left = 0
    active_side = 0.0
    hold_bars = max(int(hold_bars), 1)
    lookback = max(int(retest_lookback), 3)

    for i in range(1, n):
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
            volume_ratio=float(context["volume_ratio"][i]),
        )
        support_trace[i] = support
        fast = float(ema_fast[i])
        slow = float(ema_slow[i])
        fast_prev = float(ema_fast[i - 1])
        slow_prev = float(ema_slow[i - 1])
        price = float(close[i])
        prev_price = float(close[i - 1])
        volume_ratio = float(context["volume_ratio"][i])
        trend_up = fast > slow and fast >= fast_prev and slow >= slow_prev
        trend_down = fast < slow and fast <= fast_prev and slow <= slow_prev
        trend_trace[i] = 1.0 if trend_up else (-1.0 if trend_down else 0.0)

        if active_side != 0.0 and hold_left > 0:
            if active_side > 0.0 and (not trend_up or support < 0.0):
                active_side = 0.0
                hold_left = 0
            elif active_side < 0.0 and (not trend_down or support > 0.0):
                active_side = 0.0
                hold_left = 0
            else:
                target[i] = active_side * trigger_weight
                trigger_side[i] = active_side
                hold_left -= 1
                continue

        base_weight = float(baseline_target[i])
        base_side = 0.0 if abs(base_weight) <= 1e-12 else float(np.sign(base_weight))

        left = max(0, i - lookback)
        recent = close[left:i]
        recent_high = float(np.nanmax(recent))
        recent_low = float(np.nanmin(recent))
        reclaim_long = prev_price <= fast_prev and price > fast and price >= recent_high
        reclaim_short = prev_price >= fast_prev and price < fast and price <= recent_low

        side = 0.0
        if trend_up and reclaim_long and support >= float(support_floor) and volume_ratio >= float(volume_ratio_floor):
            side = 1.0
        elif trend_down and reclaim_short and support <= -float(support_floor) and volume_ratio >= float(volume_ratio_floor):
            side = -1.0

        if side == 0.0:
            continue
        if activation_mode == "flat_only" and base_side != 0.0:
            continue
        if activation_mode == "flat_or_disagree" and base_side != 0.0 and base_side == side:
            continue
        if activation_mode not in {"flat_only", "flat_or_disagree"}:
            raise ValueError(f"unsupported activation mode: {activation_mode}")

        active_side = side
        hold_left = hold_bars - 1
        target[i] = side * trigger_weight
        trigger_side[i] = side

    return {
        "target_weight": target,
        "trigger_side": trigger_side,
        "support": support_trace,
        "trend": trend_trace,
    }
