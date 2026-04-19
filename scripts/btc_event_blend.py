from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from btc_convex_blend import BLEND_PAIR, build_blended_target_trace, replay_target_trace
from btc_convex_blend import blend_runtime_weight, get_btc_convex_blend, replay_btc_convex_blend_candidate
from btc_online_blend import get_btc_online_blend, replay_btc_online_blend_candidate
from btc_breakout_override import breakout_support_score
from event_orderflow_specialist import _ema, build_event_orderflow_trace
from search_pair_subset_regime_mixture import (
    _derivative_metric_series,
    _log_ratio_feature,
    _open_interest_relative_metric,
    realistic_overlay_replay_from_context,
)


def build_runtime_event_context_from_frame(
    df: Any,
    pair: str,
    *,
    derivative_bundle: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    idx = pd.DatetimeIndex(df.index)

    def pair_feature_array(suffix: str, *, fill_value: float) -> np.ndarray:
        column = f"{pair}_{suffix}"
        if column not in df.columns:
            return np.full(len(idx), float(fill_value), dtype="float64")
        return np.asarray(
            pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill_value),
            dtype="float64",
        )

    derivative_bundle = derivative_bundle or {}
    derivative_sources = {
        "oi_rel": bool(f"{pair}_oi_rel" in df.columns or ((derivative_bundle.get("open_interest") is not None) and not derivative_bundle["open_interest"].empty)),
        "basis_rate": bool(
            f"{pair}_basis_rate" in df.columns
            or ((derivative_bundle.get("basis_perpetual") is not None) and not derivative_bundle["basis_perpetual"].empty)
        ),
        "top_pos_log_ratio": bool(
            f"{pair}_top_pos_log_ratio" in df.columns
            or ((derivative_bundle.get("top_trader_position_ratio") is not None) and not derivative_bundle["top_trader_position_ratio"].empty)
        ),
        "taker_buy_sell_log_ratio": bool(
            f"{pair}_taker_buy_sell_log_ratio" in df.columns
            or ((derivative_bundle.get("taker_buy_sell_ratio") is not None) and not derivative_bundle["taker_buy_sell_ratio"].empty)
        ),
    }
    derivative_inputs_ready = all(derivative_sources.values())

    def derivative_feature_array(
        suffix: str,
        *,
        fill_value: float,
        series_builder: Callable[[], pd.Series],
    ) -> np.ndarray:
        column = f"{pair}_{suffix}"
        if column in df.columns:
            return pair_feature_array(suffix, fill_value=fill_value)
        series = series_builder()
        return np.asarray(series.fillna(fill_value), dtype="float64")

    close = pair_feature_array("close", fill_value=0.0)
    high = pair_feature_array("high", fill_value=0.0)
    low = pair_feature_array("low", fill_value=0.0)
    volume = pair_feature_array("volume", fill_value=0.0)
    vol_sma = pair_feature_array("vol_sma", fill_value=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        range_bps = np.nan_to_num(((high - low) / np.where(close == 0.0, np.nan, close)) * 10_000.0, nan=0.0, posinf=0.0, neginf=0.0)
        volume_ratio = np.nan_to_num(volume / np.where(vol_sma == 0.0, np.nan, vol_sma), nan=1.0, posinf=1.0, neginf=1.0)
    return {
        "close": close,
        "order_imbalance": pair_feature_array("order_imbalance", fill_value=0.0),
        "buy_volume_share": pair_feature_array("buy_volume_share", fill_value=0.5),
        "close_location_value": pair_feature_array("close_location_value", fill_value=0.0),
        "body_to_range": pair_feature_array("body_to_range", fill_value=0.0),
        "wick_skew": pair_feature_array("wick_skew", fill_value=0.0),
        "oi_rel": derivative_feature_array(
            "oi_rel",
            fill_value=1.0,
            series_builder=lambda: _open_interest_relative_metric(derivative_bundle.get("open_interest"), idx),
        ),
        "basis_rate": derivative_feature_array(
            "basis_rate",
            fill_value=0.0,
            series_builder=lambda: _derivative_metric_series(
                derivative_bundle.get("basis_perpetual"),
                "basis_rate",
                idx,
            ).clip(-0.01, 0.01),
        ),
        "top_pos_log_ratio": derivative_feature_array(
            "top_pos_log_ratio",
            fill_value=0.0,
            series_builder=lambda: _log_ratio_feature(
                _derivative_metric_series(
                    derivative_bundle.get("top_trader_position_ratio"),
                    "long_short_ratio",
                    idx,
                )
            ),
        ),
        "taker_buy_sell_log_ratio": derivative_feature_array(
            "taker_buy_sell_log_ratio",
            fill_value=0.0,
            series_builder=lambda: _log_ratio_feature(
                _derivative_metric_series(
                    derivative_bundle.get("taker_buy_sell_ratio"),
                    "buy_sell_ratio",
                    idx,
                )
            ),
        ),
        "dc_trend_05": pair_feature_array("dc_trend_05", fill_value=0.0),
        "dc_run_05": pair_feature_array("dc_run_05", fill_value=0.0),
        "range_bps": np.asarray(range_bps, dtype="float64"),
        "volume_ratio": np.asarray(volume_ratio, dtype="float64"),
        "derivative_inputs_ready": bool(derivative_inputs_ready),
        "derivative_input_sources": derivative_sources,
    }


def get_btc_event_blend(candidate: Mapping[str, Any] | None, pair: str | None = None) -> dict[str, Any] | None:
    if not isinstance(candidate, Mapping):
        return None
    raw = candidate.get("btc_event_blend")
    if not isinstance(raw, Mapping):
        return None
    payload = copy.deepcopy(dict(raw))
    blend_pair = str(payload.get("pair") or BLEND_PAIR)
    if pair is not None and str(pair) != blend_pair:
        return None
    payload["pair"] = blend_pair
    payload["alpha"] = float(payload.get("alpha", 0.0))
    payload["mode"] = str(payload.get("mode") or "disagree_narrow_only")
    payload["trigger_weight"] = float(payload.get("trigger_weight", 0.0))
    payload["fast_span"] = int(payload.get("fast_span", 6))
    payload["slow_span"] = int(payload.get("slow_span", 18))
    payload["retest_lookback"] = int(payload.get("retest_lookback", 6))
    payload["support_floor"] = float(payload.get("support_floor", 0.0))
    payload["volume_ratio_floor"] = float(payload.get("volume_ratio_floor", 1.0))
    payload["activation_mode"] = str(payload.get("activation_mode") or "flat_only")
    payload["hold_bars"] = int(payload.get("hold_bars", 2))
    return payload


def _baseline_replay_without_event(
    *,
    candidate: Mapping[str, Any],
    pair: str,
    context: Mapping[str, Any],
    library_lookup: Mapping[str, Any],
    return_trace: bool = False,
) -> dict[str, Any]:
    base_candidate = copy.deepcopy(dict(candidate))
    base_candidate.pop("btc_event_blend", None)
    if pair == "BTCUSDT" and base_candidate.get("btc_online_blend"):
        return replay_btc_online_blend_candidate(
            candidate=base_candidate,
            pair=pair,
            context=context,
            library_lookup=library_lookup,
            return_trace=return_trace,
        )
    if pair == "BTCUSDT" and base_candidate.get("btc_convex_blend"):
        return replay_btc_convex_blend_candidate(
            candidate=base_candidate,
            pair=pair,
            context=context,
            library_lookup=library_lookup,
            return_trace=return_trace,
        )
    pair_config = dict((base_candidate.get("pair_configs") or {}).get(pair) or {})
    return realistic_overlay_replay_from_context(
        context,
        library_lookup,
        tuple(int(v) for v in pair_config["mapping_indices"]),
        float(pair_config["route_breadth_threshold"]),
        execution_gene=pair_config.get("execution_gene"),
        engine="python",
        return_trace=return_trace,
    )


def replay_btc_event_blend_candidate(
    *,
    candidate: Mapping[str, Any],
    pair: str,
    context: Mapping[str, Any],
    library_lookup: Mapping[str, Any],
    return_trace: bool = False,
) -> dict[str, Any]:
    event = get_btc_event_blend(candidate, pair)
    if event is None:
        raise RuntimeError("Candidate has no btc_event_blend payload.")
    baseline = _baseline_replay_without_event(
        candidate=candidate,
        pair=pair,
        context=context,
        library_lookup=library_lookup,
        return_trace=True,
    )
    baseline_trace = baseline["trace"]
    specialist_trace = build_event_orderflow_trace(
        context=context,
        baseline_trace=baseline_trace,
        trigger_weight=float(event["trigger_weight"]),
        fast_span=int(event["fast_span"]),
        slow_span=int(event["slow_span"]),
        retest_lookback=int(event["retest_lookback"]),
        support_floor=float(event["support_floor"]),
        volume_ratio_floor=float(event["volume_ratio_floor"]),
        activation_mode=str(event["activation_mode"]),
        hold_bars=int(event["hold_bars"]),
    )
    pair_cfg = dict((candidate.get("pair_configs") or {}).get(pair) or {})
    target_trace = build_blended_target_trace(
        context=context,
        route_breadth_threshold=float(pair_cfg.get("route_breadth_threshold", 0.5)),
        baseline_trace=baseline_trace,
        specialist_trace=specialist_trace,
        alpha=float(event["alpha"]),
        mode=str(event["mode"]),
    )
    result = replay_target_trace(
        context=context,
        target_trace=np.asarray(target_trace, dtype="float64"),
        execution_gene=pair_cfg.get("execution_gene"),
        trace_template=baseline_trace,
        return_trace=return_trace,
    )
    result["event_blend"] = {
        "alpha": float(event["alpha"]),
        "mode": str(event["mode"]),
    }
    if return_trace:
        result["trace"]["event_specialist_target"] = np.asarray(specialist_trace["target_weight"], dtype="float64")
    return result


def build_runtime_event_specialist_plan(
    *,
    context: Mapping[str, Any],
    baseline_target_weight: float,
    pair_state: Mapping[str, Any] | None,
    event: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    close = np.asarray(context["close"], dtype="float64")
    if close.shape[0] < 2:
        return (
            {"requested_weight": 0.0, "target_weight": 0.0, "support": 0.0, "trend": 0.0},
            {"active_side": 0.0, "hold_left": 0},
        )

    fast_span = max(int(event["fast_span"]), 1)
    slow_span = max(int(event["slow_span"]), fast_span + 1)
    lookback = max(int(event["retest_lookback"]), 3)
    support_floor = float(event["support_floor"])
    volume_ratio_floor = float(event["volume_ratio_floor"])
    activation_mode = str(event["activation_mode"])
    hold_bars = max(int(event["hold_bars"]), 1)
    trigger_weight = float(abs(event["trigger_weight"]))

    ema_fast = _ema(close, fast_span)
    ema_slow = _ema(close, slow_span)
    i = int(close.shape[0] - 1)
    prev_i = int(i - 1)
    price = float(close[i])
    prev_price = float(close[prev_i])
    fast = float(ema_fast[i])
    slow = float(ema_slow[i])
    fast_prev = float(ema_fast[prev_i])
    slow_prev = float(ema_slow[prev_i])
    volume_ratio = float(context["volume_ratio"][i])
    trend_up = fast > slow and fast >= fast_prev and slow >= slow_prev
    trend_down = fast < slow and fast <= fast_prev and slow <= slow_prev
    trend = 1.0 if trend_up else (-1.0 if trend_down else 0.0)

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

    state = dict(pair_state or {})
    active_side = float(state.get("active_side", 0.0))
    hold_left = int(state.get("hold_left", 0))

    if active_side != 0.0 and hold_left > 0:
        if active_side > 0.0 and (not trend_up or support < 0.0):
            active_side = 0.0
            hold_left = 0
        elif active_side < 0.0 and (not trend_down or support > 0.0):
            active_side = 0.0
            hold_left = 0
        else:
            hold_left -= 1
            specialist_weight = float(active_side * trigger_weight)
            next_state = {
                "active_side": float(active_side),
                "hold_left": int(hold_left),
                "support": float(support),
                "trend": float(trend),
                "last_price": float(price),
            }
            return (
                {
                    "requested_weight": specialist_weight,
                    "target_weight": specialist_weight,
                    "support": float(support),
                    "trend": float(trend),
                },
                next_state,
            )

    base_side = 0.0 if abs(float(baseline_target_weight)) <= 1e-12 else float(np.sign(float(baseline_target_weight)))
    if activation_mode == "flat_only" and base_side != 0.0:
        return (
            {"requested_weight": 0.0, "target_weight": 0.0, "support": float(support), "trend": float(trend)},
            {"active_side": 0.0, "hold_left": 0, "support": float(support), "trend": float(trend), "last_price": float(price)},
        )

    left = max(0, i - lookback)
    recent = close[left:i]
    recent_high = float(np.nanmax(recent))
    recent_low = float(np.nanmin(recent))
    reclaim_long = prev_price <= fast_prev and price > fast and price >= recent_high
    reclaim_short = prev_price >= fast_prev and price < fast and price <= recent_low

    side = 0.0
    if trend_up and reclaim_long and support >= support_floor and volume_ratio >= volume_ratio_floor:
        side = 1.0
    elif trend_down and reclaim_short and support <= -support_floor and volume_ratio >= volume_ratio_floor:
        side = -1.0

    specialist_weight = 0.0
    if side != 0.0:
        specialist_weight = float(side * trigger_weight)
        active_side = float(side)
        hold_left = int(hold_bars - 1)

    next_state = {
        "active_side": float(active_side),
        "hold_left": int(hold_left),
        "support": float(support),
        "trend": float(trend),
        "last_price": float(price),
    }
    return (
        {
            "requested_weight": float(specialist_weight),
            "target_weight": float(specialist_weight),
            "support": float(support),
            "trend": float(trend),
        },
        next_state,
    )


def apply_runtime_event_blend(
    *,
    context: Mapping[str, Any],
    baseline_plan: Mapping[str, Any],
    pair_state: Mapping[str, Any] | None,
    event: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not bool(context.get("derivative_inputs_ready", True)):
        final_plan = dict(baseline_plan)
        missing_sources = [
            key for key, present in dict(context.get("derivative_input_sources") or {}).items()
            if not bool(present)
        ]
        close_values = context.get("close")
        close = np.asarray([] if close_values is None else close_values, dtype="float64")
        final_plan["event_blend"] = {
            "alpha": float(event["alpha"]),
            "mode": str(event["mode"]),
            "enabled": False,
            "reason": "missing_derivative_inputs",
            "missing_sources": missing_sources,
        }
        return (
            final_plan,
            {
                "active_side": 0.0,
                "hold_left": 0,
                "support": 0.0,
                "trend": 0.0,
                "last_price": float(close[-1]) if close.size else 0.0,
                "enabled": False,
                "reason": "missing_derivative_inputs",
            },
        )

    specialist_plan, next_state = build_runtime_event_specialist_plan(
        context=context,
        baseline_target_weight=float(baseline_plan["target_weight"]),
        pair_state=pair_state,
        event=event,
    )
    final_plan = dict(baseline_plan)
    final_plan["requested_weight"] = blend_runtime_weight(
        baseline_weight=float(baseline_plan["requested_weight"]),
        specialist_weight=float(specialist_plan["requested_weight"]),
        route_state_name=str(baseline_plan["route_state_name"]),
        alpha=float(event["alpha"]),
        mode=str(event["mode"]),
    )
    final_plan["target_weight"] = blend_runtime_weight(
        baseline_weight=float(baseline_plan["target_weight"]),
        specialist_weight=float(specialist_plan["target_weight"]),
        route_state_name=str(baseline_plan["route_state_name"]),
        alpha=float(event["alpha"]),
        mode=str(event["mode"]),
    )
    final_plan["event_blend"] = {
        "alpha": float(event["alpha"]),
        "mode": str(event["mode"]),
        "specialist_requested_weight": float(specialist_plan["requested_weight"]),
        "specialist_target_weight": float(specialist_plan["target_weight"]),
        "support": float(specialist_plan["support"]),
        "trend": float(specialist_plan["trend"]),
    }
    return final_plan, next_state
