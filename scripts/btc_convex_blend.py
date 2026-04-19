from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

import gp_crypto_evolution as gp
from execution_gene_utils import derive_execution_profile, legacy_execution_profile
from search_pair_subset_regime_mixture import realistic_overlay_replay_from_context, route_state_names


BLEND_PAIR = "BTCUSDT"
DEFAULT_BLEND_MODE = "always"
PAIR_CONVEX_BLEND_KEY = "pair_convex_blends"

def _normalize_convex_blend(raw: Mapping[str, Any] | None, *, default_pair: str) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    blend = copy.deepcopy(dict(raw))
    blend_pair = str(blend.get("pair") or default_pair or BLEND_PAIR)
    if not isinstance(blend.get("specialist_pair_config"), Mapping):
        return None
    blend["pair"] = blend_pair
    blend["alpha"] = float(blend.get("alpha", 0.0))
    blend["mode"] = str(blend.get("mode") or DEFAULT_BLEND_MODE)
    raw_state_alphas = blend.get("state_alphas")
    state_alphas: dict[str, float] = {}
    if isinstance(raw_state_alphas, Mapping):
        for state_name, value in raw_state_alphas.items():
            try:
                state_alphas[str(state_name)] = float(value)
            except (TypeError, ValueError):
                continue
    blend["state_alphas"] = state_alphas
    blend["specialist_pair_config"] = copy.deepcopy(dict(blend["specialist_pair_config"]))
    return blend


def get_btc_convex_blend(candidate: Mapping[str, Any] | None, pair: str | None = None) -> dict[str, Any] | None:
    if not isinstance(candidate, Mapping):
        return None

    raw: Mapping[str, Any] | None = None
    if pair is not None:
        pair_blends = candidate.get(PAIR_CONVEX_BLEND_KEY)
        if isinstance(pair_blends, Mapping):
            pair_raw = pair_blends.get(str(pair))
            if isinstance(pair_raw, Mapping):
                raw = pair_raw
                blend = _normalize_convex_blend(raw, default_pair=str(pair))
                if blend is not None and str(blend["pair"]) == str(pair):
                    return blend
    raw = candidate.get("btc_convex_blend")
    blend = _normalize_convex_blend(raw, default_pair=BLEND_PAIR if pair is None else str(pair))
    if blend is None:
        return None
    if pair is not None and str(blend["pair"]) != str(pair):
        return None
    return blend


def should_apply_runtime_blend(
    *,
    baseline_weight: float,
    specialist_weight: float,
    route_state_name: str,
    mode: str,
) -> bool:
    narrow = str(route_state_name).endswith("narrow")
    disagree = np.sign(float(baseline_weight)) != np.sign(float(specialist_weight))
    if mode == "always":
        return True
    if mode == "narrow_only":
        return bool(narrow)
    if mode == "disagree_only":
        return bool(disagree)
    if mode == "disagree_narrow_only":
        return bool(narrow and disagree)
    if mode == "state_alphas":
        return False
    raise ValueError(f"Unsupported convex blend mode: {mode}")


def resolve_blend_alpha(
    *,
    baseline_weight: float,
    specialist_weight: float,
    route_state_name: str,
    alpha: float,
    mode: str,
    state_alphas: Mapping[str, float] | None = None,
) -> float:
    if state_alphas is not None and route_state_name in state_alphas:
        return float(state_alphas[route_state_name])
    if mode == "state_alphas":
        return 0.0
    if should_apply_runtime_blend(
        baseline_weight=baseline_weight,
        specialist_weight=specialist_weight,
        route_state_name=route_state_name,
        mode=mode,
    ):
        return float(alpha)
    return 0.0


def blend_runtime_weight(
    *,
    baseline_weight: float,
    specialist_weight: float,
    route_state_name: str,
    alpha: float,
    mode: str,
    state_alphas: Mapping[str, float] | None = None,
) -> float:
    blend_alpha = resolve_blend_alpha(
        baseline_weight=baseline_weight,
        specialist_weight=specialist_weight,
        route_state_name=route_state_name,
        alpha=alpha,
        mode=mode,
        state_alphas=state_alphas,
    )
    if blend_alpha <= 0.0:
        return float(baseline_weight)
    return float((1.0 - float(blend_alpha)) * float(baseline_weight) + float(blend_alpha) * float(specialist_weight))


def build_blended_target_trace(
    *,
    context: Mapping[str, Any],
    route_breadth_threshold: float,
    baseline_trace: Mapping[str, Any],
    specialist_trace: Mapping[str, Any],
    alpha: float,
    mode: str,
    state_alphas: Mapping[str, float] | None = None,
) -> np.ndarray:
    base_target = np.asarray(baseline_trace["target_weight"], dtype="float64")
    specialist_target = np.asarray(specialist_trace["target_weight"], dtype="float64")
    blended = base_target.copy()
    if mode == "always" and not state_alphas:
        return (1.0 - float(alpha)) * base_target + float(alpha) * specialist_target

    route_state_mode = str(context.get("route_state_mode") or "equity_corr")
    names = route_state_names(route_state_mode)
    bucket_codes = np.asarray(context["bucket_codes"][float(route_breadth_threshold)], dtype="int64")[: len(base_target)]
    for i in range(len(base_target)):
        route_state_name = str(names[int(bucket_codes[i])])
        blend_alpha = resolve_blend_alpha(
            baseline_weight=float(base_target[i]),
            specialist_weight=float(specialist_target[i]),
            route_state_name=route_state_name,
            alpha=alpha,
            mode=mode,
            state_alphas=state_alphas,
        )
        if blend_alpha <= 0.0:
            continue
        blended[i] = (1.0 - float(blend_alpha)) * base_target[i] + float(blend_alpha) * specialist_target[i]
    return blended


def quantize_amount(value: float, amount_step: float, min_qty: float) -> float:
    if amount_step <= 0.0:
        quantized = float(value)
    else:
        quantized = float(np.trunc(value / amount_step) * amount_step)
    if abs(quantized) < min_qty:
        return 0.0
    return quantized


def replay_target_trace(
    *,
    context: Mapping[str, Any],
    target_trace: np.ndarray,
    execution_gene: Mapping[str, Any] | None,
    trace_template: Mapping[str, Any] | None = None,
    return_trace: bool = False,
) -> dict[str, Any]:
    profile = legacy_execution_profile() if execution_gene is None else derive_execution_profile(dict(execution_gene))
    fee_rate = float(profile["fee_rate"])
    slippage = float(profile["slippage"])
    amount_step = float(profile["amount_step"])
    min_qty = float(profile["min_qty"])

    open_p = np.asarray(context["open"], dtype="float64")
    funding_rates = np.asarray(context["funding_rates"], dtype="float64")

    cash = float(gp.INITIAL_CASH)
    qty = 0.0
    n_trades = 0
    fee_paid = 0.0
    slippage_paid = 0.0
    funding_paid = 0.0
    funding_events = 0
    net_ret: list[float] = []
    equity_curve: list[float] = [float(gp.INITIAL_CASH)]
    realized_target_trace: list[float] = []

    for exec_idx in range(1, open_p.shape[0] - 1):
        signal_idx = exec_idx - 1
        px_open = float(open_p[exec_idx])
        next_open = float(open_p[exec_idx + 1])
        funding_rate = float(funding_rates[exec_idx])

        if qty != 0.0 and funding_rate != 0.0:
            funding_cashflow = -qty * px_open * funding_rate
            cash += funding_cashflow
            funding_paid += funding_cashflow
            funding_events += 1

        equity_before = cash + qty * px_open
        if abs(equity_before) <= 1e-9:
            equity_before = 1e-9

        target_weight = float(target_trace[signal_idx])
        target_notional = equity_before * target_weight
        target_qty = 0.0
        if abs(px_open) > 1e-12:
            target_qty = quantize_amount(target_notional / px_open, amount_step, min_qty)
        diff_qty = quantize_amount(target_qty - qty, amount_step, min_qty)

        if abs(diff_qty) > 0.0:
            side = 1.0 if diff_qty > 0.0 else -1.0
            exec_price = px_open * (1.0 + slippage * side)
            trade_notional = diff_qty * exec_price
            fee = abs(diff_qty) * exec_price * fee_rate
            cash -= trade_notional
            cash -= fee
            qty += diff_qty
            n_trades += 1
            fee_paid += fee
            slippage_paid += abs(diff_qty) * px_open * slippage

        equity_after = cash + qty * next_open
        bar_net = equity_after / equity_before - 1.0
        net_ret.append(float(bar_net))
        equity_curve.append(float(equity_after))
        realized_target_trace.append(float(target_weight))

    daily_metrics = gp.compute_daily_metrics(np.asarray(net_ret, dtype="float64"))
    equity_arr = np.asarray(equity_curve, dtype="float64")
    max_drawdown = float(np.min(equity_arr / np.maximum.accumulate(equity_arr) - 1.0))
    sharpe = 0.0
    if len(net_ret) > 1 and np.std(net_ret) > 1e-12:
        sharpe = float(np.mean(net_ret) / np.std(net_ret) * np.sqrt(365.25 * 24.0 * 60.0 / 5.0))
    result = {
        "avg_daily_return": float(daily_metrics["avg_daily_return"]),
        "total_return": float(equity_arr[-1] / gp.INITIAL_CASH - 1.0),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "daily_target_hit_rate": float(daily_metrics["daily_target_hit_rate"]),
        "daily_win_rate": float(daily_metrics["daily_win_rate"]),
        "worst_day": float(daily_metrics["worst_day"]),
        "best_day": float(daily_metrics["best_day"]),
        "n_trades": int(n_trades),
        "fee_paid": float(fee_paid),
        "slippage_paid": float(slippage_paid),
        "funding_paid": float(funding_paid),
        "funding_events": int(funding_events),
        "final_equity": float(equity_arr[-1]),
    }
    if return_trace:
        trace_payload = {
            "bar_net": np.asarray(net_ret, dtype="float64"),
            "target_weight": np.asarray(realized_target_trace, dtype="float64"),
            "requested_weight": np.asarray(realized_target_trace, dtype="float64"),
        }
        if trace_template is not None:
            for key in ("signal_pct", "role_idx"):
                if key in trace_template:
                    arr = np.asarray(trace_template[key])
                    trace_payload[key] = arr[: len(realized_target_trace)]
        result["trace"] = trace_payload
    return result


def replay_btc_convex_blend_candidate(
    *,
    candidate: Mapping[str, Any],
    pair: str,
    context: Mapping[str, Any],
    library_lookup: Mapping[str, Any],
    return_trace: bool = False,
) -> dict[str, Any]:
    blend = get_btc_convex_blend(candidate, pair)
    if blend is None:
        raise ValueError(f"No convex blend configured for pair {pair}.")
    pair_configs = candidate.get("pair_configs") or {}
    baseline_cfg = copy.deepcopy(dict(pair_configs[pair]))
    specialist_cfg = copy.deepcopy(dict(blend["specialist_pair_config"]))
    route_breadth_threshold = float(baseline_cfg["route_breadth_threshold"])
    baseline = realistic_overlay_replay_from_context(
        dict(context),
        dict(library_lookup),
        tuple(int(v) for v in baseline_cfg["mapping_indices"]),
        route_breadth_threshold,
        execution_gene=baseline_cfg.get("execution_gene"),
        engine="python",
        return_trace=True,
    )
    specialist = realistic_overlay_replay_from_context(
        dict(context),
        dict(library_lookup),
        tuple(int(v) for v in specialist_cfg["mapping_indices"]),
        float(specialist_cfg["route_breadth_threshold"]),
        execution_gene=specialist_cfg.get("execution_gene"),
        engine="python",
        return_trace=True,
    )
    target_trace = build_blended_target_trace(
        context=context,
        route_breadth_threshold=route_breadth_threshold,
        baseline_trace=baseline["trace"],
        specialist_trace=specialist["trace"],
        alpha=float(blend["alpha"]),
        mode=str(blend["mode"]),
        state_alphas=blend.get("state_alphas"),
    )
    result = replay_target_trace(
        context=context,
        target_trace=target_trace,
        execution_gene=baseline_cfg.get("execution_gene"),
        trace_template=baseline.get("trace"),
        return_trace=return_trace,
    )
    result["blend"] = {
        "pair": pair,
        "alpha": float(blend["alpha"]),
        "mode": str(blend["mode"]),
        "state_alphas": dict(blend.get("state_alphas") or {}),
    }
    return result
