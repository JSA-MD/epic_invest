from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np

import gp_crypto_evolution as gp


def _read_gate_overrides(env=None) -> tuple[float, bool]:
    """Return (threshold_scale, gate_disabled) from env.

    Mirrors the live-side hook in pairwise_regime_live.compute_requested_weight
    so backtest replays running through the conformal gate honour the same
    PAIRWISE_REGIME_THRESHOLD_SCALE / PAIRWISE_REGIME_GATE_DISABLED settings.
    Without this, any operator who tightens or loosens the live gate breaks
    apples-to-apples parity until the next full restart.
    """
    e = env if env is not None else os.environ
    try:
        scale = float(e.get("PAIRWISE_REGIME_THRESHOLD_SCALE", "1.0") or 1.0)
    except (TypeError, ValueError):
        scale = 1.0
    disabled = str(e.get("PAIRWISE_REGIME_GATE_DISABLED", "0")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    return scale, disabled
from execution_gene_utils import (
    dc_alignment_score,
    derive_execution_profile,
    legacy_execution_profile,
    microstructure_alignment_score,
    should_abstain_for_alignment,
)
from search_pair_subset_regime_mixture import (
    BAR_FACTOR,
    BARS_PER_DAY,
    _quantize_amount_kernel,
    default_state_specialists_for_router,
    normalize_mapping_indices,
    normalize_route_state_mode,
    realistic_overlay_replay_from_context,
    route_state_names,
)


def normalize_conformal_gate_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(raw or {})
    enabled_pairs = payload.get("enabled_pairs") or ()
    return {
        "alpha": float(payload.get("alpha", 0.45)),
        "lookback_days": int(payload.get("lookback_days", 63)),
        "min_samples": int(payload.get("min_samples", 40)),
        "fallback_min_samples": int(payload.get("fallback_min_samples", 20)),
        "enabled_pairs": tuple(str(pair) for pair in enabled_pairs),
        "state_conditioned": bool(payload.get("state_conditioned", True)),
        "side_conditioned": bool(payload.get("side_conditioned", True)),
    }


def risk_control_threshold(
    events: list[tuple[int, float, int]],
    *,
    alpha: float,
    min_samples: int,
    min_index: int,
) -> float:
    eligible = [(score, error) for event_index, score, error in events if int(event_index) >= int(min_index)]
    if len(eligible) < int(min_samples):
        return 0.0
    eligible.sort(key=lambda item: float(item[0]), reverse=True)
    cum_errors = 0
    cum_total = 0
    threshold = float("inf")
    found = False
    for score, error in eligible:
        cum_total += 1
        cum_errors += int(error)
        if (cum_errors / max(cum_total, 1)) <= float(alpha):
            threshold = float(score)
            found = True
    return float(threshold if found else float("inf"))


def _append_resolved_event(
    history: dict[tuple[Any, ...], list[tuple[int, float, int]]],
    *,
    gate: dict[str, Any],
    state_code: int,
    side: int,
    score: float,
    error: int,
    event_index: int,
) -> None:
    if bool(gate["state_conditioned"]) and bool(gate["side_conditioned"]):
        history[("state_side", int(state_code), int(side))].append((int(event_index), float(score), int(error)))
    if bool(gate["side_conditioned"]):
        history[("side", int(side))].append((int(event_index), float(score), int(error)))
    history[("global",)].append((int(event_index), float(score), int(error)))


def _resolve_crc_threshold(
    history: dict[tuple[Any, ...], list[tuple[int, float, int]]],
    *,
    gate: dict[str, Any],
    state_code: int,
    side: int,
    current_index: int,
) -> float:
    lookback_bars = int(gate["lookback_days"]) * int(BARS_PER_DAY)
    min_index = int(current_index) - int(lookback_bars)
    keys: list[tuple[Any, ...]] = []
    if bool(gate["state_conditioned"]) and bool(gate["side_conditioned"]):
        keys.append(("state_side", int(state_code), int(side)))
    if bool(gate["side_conditioned"]):
        keys.append(("side", int(side)))
    keys.append(("global",))
    for key in keys:
        threshold = risk_control_threshold(
            history.get(key, []),
            alpha=float(gate["alpha"]),
            min_samples=int(gate["min_samples"] if key != ("global",) else gate["fallback_min_samples"]),
            min_index=min_index,
        )
        if threshold > 0.0:
            return float(threshold)
    return 0.0


def candidate_uses_conformal_gate(candidate: dict[str, Any], pair: str) -> bool:
    gate = normalize_conformal_gate_config(candidate.get("conformal_gate"))
    enabled_pairs = set(gate.get("enabled_pairs") or ())
    return bool(enabled_pairs and str(pair) in enabled_pairs)


def replay_candidate_with_conformal_gate_from_context(
    *,
    candidate: dict[str, Any],
    pair: str,
    context: dict[str, Any],
    library_lookup: dict[str, Any],
    route_breadth_threshold: float,
    mapping: tuple[int, ...],
) -> dict[str, Any]:
    _gate_threshold_scale, _gate_disabled = _read_gate_overrides()
    pair_cfg = (candidate.get("pair_configs") or {}).get(pair) or {}
    execution_gene = pair_cfg.get("execution_gene")
    state_specialists_source = pair_cfg.get("state_specialists") or context.get("route_state_specialists")
    route_state_mode = normalize_route_state_mode(context.get("route_state_mode"))
    route_names = route_state_names(route_state_mode)
    mapping = normalize_mapping_indices(mapping, route_state_mode)
    if state_specialists_source is None or len(state_specialists_source) != len(route_names):
        state_specialists_source = default_state_specialists_for_router(route_names)

    if not candidate_uses_conformal_gate(candidate, pair):
        return realistic_overlay_replay_from_context(
            context,
            library_lookup,
            mapping,
            route_breadth_threshold,
            execution_gene=execution_gene,
            state_specialists=state_specialists_source,
            engine="python",
        )

    gate = normalize_conformal_gate_config(candidate.get("conformal_gate"))
    execution_profile = legacy_execution_profile() if execution_gene is None else derive_execution_profile(execution_gene)

    fee_rate = 0.0004
    slippage = 0.0002
    amount_step = 0.001
    min_qty = 0.001
    if execution_gene is not None:
        fee_rate = float(execution_profile["fee_rate"])
        slippage = float(execution_profile["slippage"])
        amount_step = float(execution_profile["amount_step"])
        min_qty = float(execution_profile["min_qty"])
        no_trade_band_pct = float(execution_profile["no_trade_band_pct"])
        signal_gate_pct = float(execution_profile["signal_gate_pct"])
        regime_buffer_mult = float(execution_profile["regime_buffer_mult"])
        confirm_bars = int(execution_profile["confirm_bars"])
    else:
        no_trade_band_pct = float(gp.NO_TRADE_BAND)
        signal_gate_pct = 0.0
        regime_buffer_mult = 0.0
        confirm_bars = 1

    corr_gross_scale = np.ones_like(context["regime"])
    corr_regime_mult = np.ones_like(context["regime"])
    if bool(pair_cfg.get("use_equity_corr_risk")):
        corr_gross_scale = context["equity_corr_gross_scale"]
        corr_regime_mult = context["equity_corr_regime_mult"]

    open_p = context["open"]
    close_p = context["close"]
    funding_rates = context["funding_rates"]
    bucket_codes = context["bucket_codes"][float(route_breadth_threshold)]
    regime = context["regime"]
    breadth = context["breadth"]
    vol_ann = context["vol_ann"]
    order_imbalance = context["order_imbalance"]
    buy_volume_share = context["buy_volume_share"]
    dc_trend_05 = context["dc_trend_05"]
    dc_run_05 = context["dc_run_05"]
    smooth_signal_matrix = context["smooth_signal_matrix"]

    library_signal_pos = np.asarray(library_lookup["signal_pos"], dtype="int64")
    library_rebalance_bars = np.asarray(library_lookup["rebalance_bars"], dtype="int64")
    library_regime_threshold = np.asarray(library_lookup["regime_threshold"], dtype="float64")
    library_breadth_threshold = np.asarray(library_lookup["breadth_threshold"], dtype="float64")
    library_target_vol_ann = np.asarray(library_lookup["target_vol_ann"], dtype="float64")
    library_gross_cap = np.asarray(library_lookup["gross_cap"], dtype="float64")
    library_kill_switch_pct = np.asarray(library_lookup["kill_switch_pct"], dtype="float64")
    library_cooldown_days = np.asarray(library_lookup["cooldown_days"], dtype="int64")
    effective_state_specialists = np.asarray(state_specialists_source, dtype="int64")
    role_signal_gate_mults = np.asarray(execution_profile["role_signal_gate_mults"], dtype="float64")
    role_regime_buffer_mults = np.asarray(execution_profile["role_regime_buffer_mults"], dtype="float64")
    abstain_edge_pct = float(execution_profile["abstain_edge_pct"])
    specialist_isolation_mult = float(execution_profile["specialist_isolation_mult"])
    microstructure_align_gate_pct = float(execution_profile["microstructure_align_gate_pct"])
    dc_align_gate_pct = float(execution_profile["dc_align_gate_pct"])
    min_alignment_votes = int(execution_profile["min_alignment_votes"])

    cash = float(gp.INITIAL_CASH)
    qty = 0.0
    n_trades = 0
    fee_paid = 0.0
    slippage_paid = 0.0
    funding_paid = 0.0
    funding_events = 0
    peak_equity = float(gp.INITIAL_CASH)
    max_drawdown = 0.0
    cooldown_bars_left = 0

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
    confirm_side = 0
    confirm_count = 0
    last_role_idx = -1

    pending_events: list[tuple[int, int, int, float, int, int]] = []
    resolved_history: dict[tuple[Any, ...], list[tuple[int, float, int]]] = defaultdict(list)

    for exec_idx in range(1, open_p.shape[0] - 1):
        signal_idx = exec_idx - 1
        px_open = open_p[exec_idx]
        next_open = open_p[exec_idx + 1]
        prev_close = close_p[signal_idx]

        if pending_events:
            still_pending: list[tuple[int, int, int, float, int, int]] = []
            for due_exec_idx, entry_exec_idx, event_index, score, side, state_code in pending_events:
                if int(due_exec_idx) <= int(exec_idx):
                    future_ret = float(open_p[int(due_exec_idx)] / open_p[int(entry_exec_idx)] - 1.0)
                    error = 1 if float(side) * float(future_ret) <= 0.0 else 0
                    _append_resolved_event(
                        resolved_history,
                        gate=gate,
                        state_code=int(state_code),
                        side=int(side),
                        score=float(score),
                        error=int(error),
                        event_index=int(event_index),
                    )
                else:
                    still_pending.append((due_exec_idx, entry_exec_idx, event_index, score, side, state_code))
            pending_events = still_pending

        funding_rate = funding_rates[exec_idx]
        if qty != 0.0 and funding_rate != 0.0:
            funding_cashflow = -qty * px_open * funding_rate
            cash += funding_cashflow
            funding_paid += funding_cashflow
            funding_events += 1

        equity_before = cash + qty * px_open
        if equity_before <= 1e-9:
            equity_before = 1e-9
        if equity_before > peak_equity:
            peak_equity = equity_before

        active_idx = int(mapping[int(bucket_codes[signal_idx])])
        role_idx = int(effective_state_specialists[int(bucket_codes[signal_idx])])
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        role_changed = role_idx != last_role_idx
        if role_changed:
            confirm_side = 0
            confirm_count = 0
            last_role_idx = role_idx

        signal_pct = float(np.clip(smooth_signal_matrix[library_signal_pos[active_idx], signal_idx], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(regime[signal_idx])
        breadth_score = float(breadth[signal_idx])
        role_signal_gate_pct = float(signal_gate_pct) * float(role_signal_gate_mults[role_idx])
        role_regime_buffer_mult = float(regime_buffer_mult) * float(role_regime_buffer_mults[role_idx])
        effective_regime_threshold = float(library_regime_threshold[active_idx]) * float(corr_regime_mult[signal_idx]) * (
            1.0 + float(role_regime_buffer_mult)
        ) * _gate_threshold_scale
        effective_gross_cap = float(library_gross_cap[active_idx]) * float(corr_gross_scale[signal_idx])
        if _gate_disabled:
            long_ok = True
            short_ok = True
        else:
            long_ok = regime_score >= effective_regime_threshold and breadth_score >= float(library_breadth_threshold[active_idx])
            short_ok = regime_score <= -effective_regime_threshold and breadth_score <= (1.0 - float(library_breadth_threshold[active_idx]))
        if abs(signal_pct) < float(role_signal_gate_pct + abstain_edge_pct):
            requested_weight = 0.0
        elif requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0
        else:
            requested_side = 0
            if requested_weight > 1e-12:
                requested_side = 1
            elif requested_weight < -1e-12:
                requested_side = -1
            if should_abstain_for_alignment(
                requested_side,
                microstructure_alignment_score(
                    float(order_imbalance[signal_idx]),
                    float(buy_volume_share[signal_idx]),
                ),
                dc_alignment_score(
                    float(dc_trend_05[signal_idx]),
                    float(dc_run_05[signal_idx]),
                ),
                float(microstructure_align_gate_pct),
                float(dc_align_gate_pct),
                int(min_alignment_votes),
            ):
                requested_weight = 0.0

        if signal_idx % int(library_rebalance_bars[active_idx]) == 0:
            requested_side = 0
            if requested_weight > 1e-12:
                requested_side = 1
            elif requested_weight < -1e-12:
                requested_side = -1
            if requested_side != 0:
                due_exec_idx = min(exec_idx + int(library_rebalance_bars[active_idx]), open_p.shape[0] - 1)
                pending_events.append(
                    (
                        int(due_exec_idx),
                        int(exec_idx),
                        int(signal_idx),
                        float(abs(signal_pct)),
                        int(requested_side),
                        int(bucket_codes[signal_idx]),
                    )
                )
                threshold = _resolve_crc_threshold(
                    resolved_history,
                    gate=gate,
                    state_code=int(bucket_codes[signal_idx]),
                    side=int(requested_side),
                    current_index=int(signal_idx),
                )
                if float(abs(signal_pct)) < float(threshold):
                    requested_weight = 0.0

        bar_vol_ann = float(vol_ann[signal_idx])
        if bar_vol_ann == bar_vol_ann and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = float(library_target_vol_ann[active_idx]) / bar_vol_ann
            gross_scale = effective_gross_cap / max(abs(requested_weight), 1e-8)
            if gross_scale < vol_scale:
                vol_scale = gross_scale
            requested_weight *= vol_scale

        gross_cap = effective_gross_cap
        if requested_weight > gross_cap:
            requested_weight = gross_cap
        elif requested_weight < -gross_cap:
            requested_weight = -gross_cap

        drawdown = equity_before / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -float(library_kill_switch_pct[active_idx]) and cooldown_bars_left == 0:
            cooldown_bars_left = int(library_cooldown_days[active_idx]) * int(BARS_PER_DAY)

        current_weight = 0.0
        if abs(equity_before) > 1e-9:
            current_weight = qty * px_open / equity_before
        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif signal_idx % int(library_rebalance_bars[active_idx]) == 0:
            role_confirm_bars = int(confirm_bars)
            if role_changed:
                role_confirm_bars = int(confirm_bars) + int(np.rint(specialist_isolation_mult * 2.0))
            requested_side = 0
            if requested_weight > 1e-12:
                requested_side = 1
            elif requested_weight < -1e-12:
                requested_side = -1
            if requested_side == 0:
                confirm_side = 0
                confirm_count = 0
            elif requested_side == confirm_side:
                confirm_count += 1
            else:
                confirm_side = requested_side
                confirm_count = 1
            if requested_side != 0 and current_weight * requested_side <= 0.0 and confirm_count < role_confirm_bars:
                requested_weight = 0.0
            target_weight = requested_weight

        if abs(target_weight - current_weight) < float(no_trade_band_pct) / 100.0:
            target_weight = current_weight

        target_notional = equity_before * target_weight
        target_qty = 0.0
        if abs(prev_close) > 1e-12:
            target_qty = _quantize_amount_kernel(target_notional / prev_close, amount_step, min_qty)
        diff_qty = _quantize_amount_kernel(target_qty - qty, amount_step, min_qty)

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
        if equity_after > peak_equity:
            peak_equity = equity_after
        dd = equity_after / peak_equity - 1.0
        if dd < max_drawdown:
            max_drawdown = dd

        bar_net = equity_after / equity_before - 1.0
        bar_count += 1
        delta = bar_net - mean_bar
        mean_bar += delta / bar_count
        m2_bar += delta * (bar_net - mean_bar)

        day_accum *= (1.0 + bar_net)
        day_len += 1
        if day_len == int(BARS_PER_DAY) or exec_idx == open_p.shape[0] - 2:
            day_ret = day_accum - 1.0
            day_sum += day_ret
            day_count += 1
            if day_ret > 0.0:
                day_wins += 1
            if day_ret >= float(gp.DAILY_TARGET_PCT):
                day_hits += 1
            if day_count == 1 or day_ret < worst_day:
                worst_day = day_ret
            if day_count == 1 or day_ret > best_day:
                best_day = day_ret
            day_accum = 1.0
            day_len = 0

    total_return = cash + qty * open_p[-1]
    total_return = total_return / float(gp.INITIAL_CASH) - 1.0
    sharpe = 0.0
    if bar_count > 1:
        variance = m2_bar / bar_count
        if variance > 1e-12:
            sharpe = mean_bar / np.sqrt(variance) * float(BAR_FACTOR)

    avg_daily = 0.0 if day_count == 0 else day_sum / day_count
    daily_target_hit_rate = 0.0 if day_count == 0 else day_hits / day_count
    daily_win_rate = 0.0 if day_count == 0 else day_wins / day_count
    final_equity = cash + qty * open_p[-1]

    return {
        "avg_daily_return": float(avg_daily),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
        "daily_target_hit_rate": float(daily_target_hit_rate),
        "daily_win_rate": float(daily_win_rate),
        "worst_day": float(worst_day),
        "best_day": float(best_day),
        "n_trades": int(n_trades),
        "fee_paid": float(fee_paid),
        "slippage_paid": float(slippage_paid),
        "funding_paid": float(funding_paid),
        "funding_events": int(funding_events),
        "final_equity": float(final_equity),
    }
