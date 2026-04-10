#!/usr/bin/env python3
"""Stress test pair-specific BTC/BNB overlay candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from replay_regime_mixture_realistic import load_model, resolve_candidate, summarize
from run_pair_subset_stress_matrix import (
    SCENARIOS,
    StressScenario,
    build_progressive_stress_profile,
    build_target_060_stress_profile,
    comparison_block,
    quantize_amount,
)
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    aggregate_metrics,
    build_overlay_inputs,
    build_route_bucket_codes,
    json_safe,
    load_or_fetch_funding,
    parse_csv_tuple,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stress matrix for pair-specific BTC/BNB candidates.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"),
    )
    parser.add_argument(
        "--base-summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
        help="Summary file containing the overlay library.",
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--report-out",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_validated_stress_report.json"),
    )
    return parser.parse_args()


def _failed_check_names(block: dict[str, Any] | None, prefix: str) -> list[str]:
    if not block:
        return [f"{prefix}_missing"]
    failed_checks = list(block.get("failed_checks") or [])
    if failed_checks:
        return [f"{prefix}.{name}" for name in failed_checks]
    checks = block.get("checks")
    if isinstance(checks, list):
        named_failures = [item.get("name") for item in checks if isinstance(item, dict) and not bool(item.get("passed", False)) and item.get("name")]
        if named_failures:
            return [f"{prefix}.{name}" for name in named_failures]
    if not bool(block.get("passed", False)):
        return [prefix]
    return []


def build_promotion_decision(summary: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    selected_candidate = summary.get("selected_candidate") or {}
    validation_engine = selected_candidate.get("validation_engine") or {}
    validation_gate = validation_engine.get("gate") or {}
    market_operating_system = validation_engine.get("market_operating_system") or {}
    market_os_gate = market_operating_system.get("gate") or {}
    final_oos_audit = market_operating_system.get("audit") or {}

    target_060_stress = report.get("profiles", {}).get("target_060_stress", {}).get("selected") or {}
    progressive_stress = report.get("profiles", {}).get("progressive_stress", {}).get("selected") or {}

    validation_gate_passed = bool(validation_gate.get("passed", False))
    market_os_gate_passed = bool(market_os_gate.get("passed", False))
    final_oos_audit_passed = bool(final_oos_audit.get("passed", False))
    target_060_passed = bool(target_060_stress.get("passed", False))
    progressive_passed = bool(progressive_stress.get("passed", False))

    ready_for_merge = validation_gate_passed and market_os_gate_passed and final_oos_audit_passed and target_060_passed
    ready_for_live = validation_gate_passed and market_os_gate_passed and final_oos_audit_passed and progressive_passed

    if not validation_gate_passed:
        status = "validation_fail"
    elif not market_os_gate_passed:
        status = "market_os_fail"
    elif not final_oos_audit_passed:
        status = "final_oos_audit_fail"
    elif ready_for_merge:
        status = "ready_for_merge"
    elif ready_for_live:
        status = "ready_for_live"
    else:
        status = "stress_fail"

    if status == "validation_fail":
        failed_checks = _failed_check_names(validation_gate, "validation_gate")
    elif status == "market_os_fail":
        failed_checks = _failed_check_names(market_os_gate, "market_os_gate")
    elif status == "final_oos_audit_fail":
        failed_checks = _failed_check_names(final_oos_audit, "final_oos_audit")
    elif status == "ready_for_live":
        failed_checks = _failed_check_names(target_060_stress, "target_060_stress")
    elif status == "stress_fail":
        failed_checks = []
        failed_checks.extend(_failed_check_names(target_060_stress, "target_060_stress"))
        failed_checks.extend(_failed_check_names(progressive_stress, "progressive_stress"))
    else:
        failed_checks = []

    return {
        "validation_gate": validation_gate,
        "market_os_gate": market_os_gate,
        "final_oos_audit": final_oos_audit,
        "target_060_stress": target_060_stress,
        "progressive_stress": progressive_stress,
        "ready_for_live": ready_for_live,
        "ready_for_merge": ready_for_merge,
        "status": status,
        "failed_checks": failed_checks,
    }


def stress_overlay_replay_pairwise(
    df: pd.DataFrame,
    trade_pair: str,
    raw_signal: pd.Series,
    pair_config: dict[str, Any],
    overlay_inputs: dict[str, pd.Series],
    funding_df: pd.DataFrame,
    library: list[Any],
    scenario: StressScenario,
    seed: int,
) -> dict[str, Any]:
    idx = pd.DatetimeIndex(df.index)
    open_p = df[f"{trade_pair}_open"].to_numpy(dtype="float64")
    close_p = df[f"{trade_pair}_close"].to_numpy(dtype="float64")
    vol_ann = overlay_inputs["vol_ann_bar"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype="float64")
    day_index = idx.normalize()
    regime = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    breadth = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    bucket_codes = build_route_bucket_codes(idx, overlay_inputs, float(pair_config["route_breadth_threshold"]))
    mapping_indices = tuple(int(v) for v in pair_config["mapping_indices"])

    spans = sorted({params.signal_span for params in library})
    smooth_signals = {
        span: raw_signal.ewm(span=span, adjust=False).mean().to_numpy(dtype="float64")
        for span in spans
    }

    funding_map = {}
    if not funding_df.empty:
        for _, row in funding_df.iterrows():
            funding_map[pd.Timestamp(row["fundingTime"]).tz_convert("UTC")] = float(row["fundingRate"])

    rng = np.random.default_rng(seed)
    cash = float(gp.INITIAL_CASH)
    qty = 0.0
    n_trades = 0
    fee_paid = 0.0
    slippage_paid = 0.0
    funding_paid = 0.0
    funding_events = 0
    rejected_orders = 0
    partial_fills = 0
    net_ret: list[float] = []
    equity_curve = [float(gp.INITIAL_CASH)]
    peak_equity = float(gp.INITIAL_CASH)
    cooldown_bars_left = 0

    start_exec = max(1, scenario.delay_bars)
    end_exec = len(df) - 1
    for exec_idx in range(start_exec, end_exec):
        signal_idx = exec_idx - scenario.delay_bars
        ts_open = pd.Timestamp(idx[exec_idx])
        px_open = float(open_p[exec_idx])
        next_open = float(open_p[exec_idx + 1])
        prev_close = float(close_p[signal_idx])

        if qty != 0.0 and ts_open in funding_map:
            funding_rate = funding_map[ts_open]
            funding_cashflow = -qty * px_open * funding_rate
            cash += funding_cashflow
            funding_paid += funding_cashflow
            funding_events += 1

        equity_before = cash + qty * px_open
        if equity_before <= 1e-9:
            equity_before = 1e-9
        peak_equity = max(peak_equity, equity_before)

        active_idx = int(mapping_indices[int(bucket_codes[signal_idx])])
        params = library[active_idx]
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = float(np.clip(smooth_signals[params.signal_span][signal_idx], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(regime[signal_idx])
        breadth_score = float(breadth[signal_idx])
        long_ok = regime_score >= params.regime_threshold and breadth_score >= params.breadth_threshold
        short_ok = regime_score <= -params.regime_threshold and breadth_score <= (1.0 - params.breadth_threshold)
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = float(vol_ann[signal_idx])
        if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = min(
                params.target_vol_ann / bar_vol_ann,
                params.gross_cap / max(abs(requested_weight), 1e-8),
            )
            requested_weight *= float(vol_scale)
        requested_weight = float(np.clip(requested_weight, -params.gross_cap, params.gross_cap))

        drawdown = equity_before / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -params.kill_switch_pct and cooldown_bars_left == 0:
            cooldown_bars_left = params.cooldown_days * gp.periods_per_day(gp.TIMEFRAME)

        current_weight = qty * px_open / equity_before if abs(equity_before) > 1e-9 else 0.0
        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif signal_idx % params.rebalance_bars == 0:
            target_weight = requested_weight

        if abs(target_weight - current_weight) < gp.NO_TRADE_BAND / 100.0:
            target_weight = current_weight

        target_notional = equity_before * target_weight
        target_qty = quantize_amount(
            target_notional / prev_close if abs(prev_close) > 1e-12 else 0.0,
            0.001,
            0.001,
        )
        desired_diff_qty = quantize_amount(target_qty - qty, 0.001, 0.001)

        if abs(desired_diff_qty) > 0.0:
            if scenario.reject_prob > 0.0 and rng.random() < scenario.reject_prob:
                rejected_orders += 1
            else:
                filled_qty = quantize_amount(desired_diff_qty * scenario.partial_fill_ratio, 0.001, 0.001)
                if abs(filled_qty) > 0.0:
                    if abs(filled_qty) + 1e-12 < abs(desired_diff_qty):
                        partial_fills += 1
                    slip = scenario.base_slippage
                    if (
                        scenario.vol_spike_threshold is not None
                        and np.isfinite(bar_vol_ann)
                        and bar_vol_ann >= scenario.vol_spike_threshold
                    ):
                        slip *= scenario.vol_spike_mult
                    side = 1.0 if filled_qty > 0.0 else -1.0
                    exec_price = px_open * (1.0 + slip * side)
                    trade_notional = filled_qty * exec_price
                    fee = abs(filled_qty) * exec_price * scenario.fee_rate
                    cash -= trade_notional
                    cash -= fee
                    qty += filled_qty
                    n_trades += 1
                    fee_paid += fee
                    slippage_paid += abs(filled_qty) * px_open * slip

        equity_after = cash + qty * next_open
        net_ret.append(float(equity_after / equity_before - 1.0))
        equity_curve.append(float(equity_after))

    result = summarize(
        np.asarray(net_ret, dtype="float64"),
        np.asarray(equity_curve, dtype="float64"),
        n_trades=n_trades,
        fee_paid=fee_paid,
        slippage_paid=slippage_paid,
        funding_paid=funding_paid,
        funding_events=funding_events,
    )
    result["rejected_orders"] = int(rejected_orders)
    result["partial_fills"] = int(partial_fills)
    result["scenario"] = scenario.name
    return result


def build_candidate_metrics(
    df_all: pd.DataFrame,
    raw_signal_all: dict[str, pd.Series],
    funding_all: dict[str, pd.DataFrame],
    library: list[Any],
    candidate: dict[str, Any],
    pairs: tuple[str, ...],
    scenario: StressScenario,
    seed_offset: int,
) -> dict[str, Any]:
    windows = {}
    for window_idx, (label, start, end) in enumerate(DEFAULT_WINDOWS):
        df = df_all.loc[start:end].copy()
        per_pair = {}
        for pair_idx, pair in enumerate(pairs):
            overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
            funding_slice = funding_all[pair]
            if not funding_slice.empty:
                funding_slice = funding_slice[
                    (funding_slice["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                    & (funding_slice["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
                ].copy()
            per_pair[pair] = stress_overlay_replay_pairwise(
                df=df,
                trade_pair=pair,
                raw_signal=raw_signal_all[pair].loc[start:end].copy(),
                pair_config=candidate["pair_configs"][pair],
                overlay_inputs=overlay_inputs,
                funding_df=funding_slice,
                library=library,
                scenario=scenario,
                seed=seed_offset + window_idx * 100 + pair_idx,
            )
        windows[label] = {
            "start": start,
            "end": end,
            "bars": int(len(df)),
            "per_pair": per_pair,
            "aggregate": aggregate_metrics(per_pair),
        }
    return windows


def main() -> None:
    args = parse_args()
    pairs = parse_csv_tuple(args.pairs, str)
    summary = json.loads(Path(args.summary).read_text())
    library_source = summary.get("search", {}).get("library_source", "summary")
    if library_source == "full-grid":
        library = list(iter_params())
    else:
        _, library, _ = resolve_candidate(Path(args.base_summary), None, None)
    model, _ = load_model(Path(args.model))
    compiled = gp.toolbox.compile(expr=model)

    start_all = DEFAULT_WINDOWS[-1][1]
    end_all = DEFAULT_WINDOWS[-1][2]
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=False)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {pair: load_or_fetch_funding(pair, start_all, end_all) for pair in pairs}

    baseline_candidate = summary["baseline_candidate"]
    selected_candidate = summary["selected_candidate"]

    scenarios_report = {}
    for scenario_idx, scenario in enumerate(SCENARIOS):
        baseline_windows = build_candidate_metrics(
            df_all,
            raw_signal_all,
            funding_all,
            library,
            baseline_candidate,
            pairs,
            scenario,
            seed_offset=1000 + scenario_idx * 10000,
        )
        selected_windows = build_candidate_metrics(
            df_all,
            raw_signal_all,
            funding_all,
            library,
            selected_candidate,
            pairs,
            scenario,
            seed_offset=500000 + scenario_idx * 10000,
        )
        scenarios_report[scenario.name] = {
            "assumptions": {
                "name": scenario.name,
                "fee_rate": scenario.fee_rate,
                "base_slippage": scenario.base_slippage,
                "delay_bars": scenario.delay_bars,
                "partial_fill_ratio": scenario.partial_fill_ratio,
                "reject_prob": scenario.reject_prob,
                "vol_spike_threshold": scenario.vol_spike_threshold,
                "vol_spike_mult": scenario.vol_spike_mult,
            },
            "baseline_candidate": baseline_windows,
            "selected_candidate": selected_windows,
            "comparison": {
                window: comparison_block(
                    baseline_windows[window]["aggregate"],
                    selected_windows[window]["aggregate"],
                )
                for window, _, _ in DEFAULT_WINDOWS
            },
        }

    report = {
        "pairs": list(pairs),
        "summary_path": args.summary,
        "base_summary_path": args.base_summary,
        "library_source": library_source,
        "baseline_candidate": summary["baseline_candidate"],
        "selected_candidate": summary["selected_candidate"],
        "scenarios": scenarios_report,
    }
    report["profiles"] = {
        "progressive_stress": {
            "baseline": build_progressive_stress_profile(report, "baseline_candidate"),
            "selected": build_progressive_stress_profile(report, "selected_candidate"),
        },
        "target_060_stress": {
            "baseline": build_target_060_stress_profile(report, "baseline_candidate"),
            "selected": build_target_060_stress_profile(report, "selected_candidate"),
        },
    }
    report["promotion_decision"] = build_promotion_decision(summary, report)

    out_path = Path(args.report_out)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
