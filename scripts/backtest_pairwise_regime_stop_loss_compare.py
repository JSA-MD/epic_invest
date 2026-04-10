#!/usr/bin/env python3
"""Compare the pairwise regime-mixture selected candidate vs entry-anchored stop-loss."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from replay_regime_mixture_realistic import load_model, resolve_candidate, summarize
from run_pair_subset_pairwise_stress import build_candidate_metrics
from run_pair_subset_stress_matrix import SCENARIOS, StressScenario
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    aggregate_metrics,
    build_overlay_inputs,
    build_route_bucket_codes,
    load_or_fetch_funding,
    parse_csv_tuple,
)


DEFAULT_STOP_LOSS_PCT = 0.02
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pairwise regime-mixture selected candidate vs entry stop-loss.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"),
    )
    parser.add_argument(
        "--base-summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
        help="Summary file containing the overlay library when the pairwise summary references it.",
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--report-out",
        default=str(gp.MODELS_DIR / "pairwise_regime_stop_loss_2pct_compare.json"),
    )
    parser.add_argument("--scenario", default="baseline_realistic")
    parser.add_argument("--stop-loss-pct", type=float, default=DEFAULT_STOP_LOSS_PCT)
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh missing 5m pair caches from Binance.")
    parser.add_argument("--fetch-funding", action="store_true", help="Fetch missing funding caches from Binance.")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StressScenario):
        return asdict(value)
    return value


def quantize_amount(value: float, step: float, min_qty: float) -> float:
    sign = 1.0 if value >= 0.0 else -1.0
    raw = abs(float(value))
    if raw < min_qty:
        return 0.0
    precise = np.floor(raw / step + 1e-12) * step
    if precise < min_qty:
        return 0.0
    return sign * float(precise)


def resolve_scenario(name: str) -> StressScenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    valid = ", ".join(item.name for item in SCENARIOS)
    raise ValueError(f"Unknown scenario {name!r}. Expected one of: {valid}")


def compute_trade_slippage(bar_vol_ann: float, scenario: StressScenario) -> float:
    slip = float(scenario.base_slippage)
    if (
        scenario.vol_spike_threshold is not None
        and np.isfinite(bar_vol_ann)
        and bar_vol_ann >= float(scenario.vol_spike_threshold)
    ):
        slip *= float(scenario.vol_spike_mult)
    return float(slip)


def stop_trigger_price(
    qty: float,
    entry_basis: float,
    px_open: float,
    px_high: float,
    px_low: float,
    stop_loss_pct: float,
    *,
    include_open_gap: bool,
) -> tuple[bool, float]:
    if abs(qty) <= EPS or not np.isfinite(entry_basis) or entry_basis <= 0.0:
        return False, np.nan

    if qty > 0.0:
        stop_price = float(entry_basis) * (1.0 - float(stop_loss_pct))
        if include_open_gap and np.isfinite(px_open) and float(px_open) <= stop_price:
            return True, float(px_open)
        if np.isfinite(px_low) and float(px_low) <= stop_price:
            return True, stop_price
        return False, np.nan

    stop_price = float(entry_basis) * (1.0 + float(stop_loss_pct))
    if include_open_gap and np.isfinite(px_open) and float(px_open) >= stop_price:
        return True, float(px_open)
    if np.isfinite(px_high) and float(px_high) >= stop_price:
        return True, stop_price
    return False, np.nan


def update_entry_basis(old_qty: float, old_basis: float, filled_qty: float, exec_price: float, new_qty: float) -> float:
    if abs(new_qty) <= EPS:
        return np.nan
    if not np.isfinite(exec_price) or exec_price <= 0.0:
        return old_basis if np.isfinite(old_basis) else np.nan
    if abs(old_qty) <= EPS or not np.isfinite(old_basis) or old_basis <= 0.0:
        return float(exec_price)
    if old_qty * new_qty < 0.0:
        return float(exec_price)
    if old_qty * filled_qty > 0.0 and abs(new_qty) > abs(old_qty) + EPS:
        added_qty = abs(new_qty) - abs(old_qty)
        return float((abs(old_qty) * old_basis + added_qty * exec_price) / abs(new_qty))
    return float(old_basis)


def execute_forced_exit(
    cash: float,
    qty: float,
    exit_ref_price: float,
    slippage: float,
    fee_rate: float,
) -> tuple[float, float, float, float]:
    if abs(qty) <= EPS:
        return cash, 0.0, 0.0, 0.0
    exit_qty = -float(qty)
    side = 1.0 if exit_qty > 0.0 else -1.0
    exec_price = float(exit_ref_price) * (1.0 + float(slippage) * side)
    trade_notional = exit_qty * exec_price
    fee = abs(exit_qty) * exec_price * float(fee_rate)
    cash_after = float(cash) - trade_notional - fee
    slippage_paid = abs(exit_qty) * float(exit_ref_price) * float(slippage)
    return cash_after, exec_price, fee, slippage_paid


def stress_overlay_replay_pairwise_with_stop(
    df: pd.DataFrame,
    trade_pair: str,
    raw_signal: pd.Series,
    pair_config: dict[str, Any],
    overlay_inputs: dict[str, pd.Series],
    funding_df: pd.DataFrame,
    library: list[Any],
    scenario: StressScenario,
    seed: int,
    stop_loss_pct: float,
) -> dict[str, Any]:
    idx = pd.DatetimeIndex(df.index)
    open_p = df[f"{trade_pair}_open"].to_numpy(dtype="float64")
    high_p = df[f"{trade_pair}_high"].to_numpy(dtype="float64")
    low_p = df[f"{trade_pair}_low"].to_numpy(dtype="float64")
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

    funding_map: dict[pd.Timestamp, float] = {}
    if not funding_df.empty:
        for _, row in funding_df.iterrows():
            funding_map[pd.Timestamp(row["fundingTime"]).tz_convert("UTC")] = float(row["fundingRate"])

    rng = np.random.default_rng(seed)
    cash = float(gp.INITIAL_CASH)
    qty = 0.0
    entry_basis = np.nan
    n_trades = 0
    fee_paid = 0.0
    slippage_paid = 0.0
    funding_paid = 0.0
    funding_events = 0
    rejected_orders = 0
    partial_fills = 0
    stop_event_count = 0
    stop_bar_count = 0
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
        px_high = float(high_p[exec_idx])
        px_low = float(low_p[exec_idx])
        next_open = float(open_p[exec_idx + 1])
        prev_close = float(close_p[signal_idx])
        bar_vol_ann = float(vol_ann[signal_idx])
        trade_slippage = compute_trade_slippage(bar_vol_ann, scenario)

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

        stop_triggered = False
        if stop_loss_pct > 0.0 and abs(qty) > EPS:
            hit, exit_ref = stop_trigger_price(
                qty,
                entry_basis,
                px_open,
                px_high,
                px_low,
                stop_loss_pct,
                include_open_gap=True,
            )
            if hit:
                cash, _, stop_fee, stop_slip = execute_forced_exit(
                    cash,
                    qty,
                    exit_ref,
                    trade_slippage,
                    scenario.fee_rate,
                )
                fee_paid += stop_fee
                slippage_paid += stop_slip
                qty = 0.0
                entry_basis = np.nan
                n_trades += 1
                stop_event_count += 1
                stop_bar_count += 1
                stop_triggered = True

        if not stop_triggered:
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

            if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > EPS:
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

            if abs(desired_diff_qty) > EPS:
                if scenario.reject_prob > 0.0 and rng.random() < scenario.reject_prob:
                    rejected_orders += 1
                else:
                    filled_qty = quantize_amount(desired_diff_qty * scenario.partial_fill_ratio, 0.001, 0.001)
                    if abs(filled_qty) > EPS:
                        if abs(filled_qty) + EPS < abs(desired_diff_qty):
                            partial_fills += 1
                        old_qty = qty
                        side = 1.0 if filled_qty > 0.0 else -1.0
                        exec_price = px_open * (1.0 + trade_slippage * side)
                        trade_notional = filled_qty * exec_price
                        fee = abs(filled_qty) * exec_price * scenario.fee_rate
                        cash -= trade_notional
                        cash -= fee
                        qty += filled_qty
                        n_trades += 1
                        fee_paid += fee
                        slippage_paid += abs(filled_qty) * px_open * trade_slippage
                        entry_basis = update_entry_basis(old_qty, entry_basis, filled_qty, exec_price, qty)

            if stop_loss_pct > 0.0 and abs(qty) > EPS:
                hit, exit_ref = stop_trigger_price(
                    qty,
                    entry_basis,
                    px_open,
                    px_high,
                    px_low,
                    stop_loss_pct,
                    include_open_gap=False,
                )
                if hit:
                    cash, _, stop_fee, stop_slip = execute_forced_exit(
                        cash,
                        qty,
                        exit_ref,
                        trade_slippage,
                        scenario.fee_rate,
                    )
                    fee_paid += stop_fee
                    slippage_paid += stop_slip
                    qty = 0.0
                    entry_basis = np.nan
                    n_trades += 1
                    stop_event_count += 1
                    stop_bar_count += 1

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
    result["stop_event_count"] = int(stop_event_count)
    result["stop_bar_count"] = int(stop_bar_count)
    return result


def build_candidate_metrics_with_stop(
    df_all: pd.DataFrame,
    raw_signal_all: dict[str, pd.Series],
    funding_all: dict[str, pd.DataFrame],
    library: list[Any],
    candidate: dict[str, Any],
    pairs: tuple[str, ...],
    scenario: StressScenario,
    seed_offset: int,
    stop_loss_pct: float,
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for window_idx, (label, start, end) in enumerate(DEFAULT_WINDOWS):
        df = df_all.loc[start:end].copy()
        per_pair: dict[str, Any] = {}
        for pair_idx, pair in enumerate(pairs):
            overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
            funding_slice = funding_all[pair]
            if not funding_slice.empty:
                funding_slice = funding_slice[
                    (funding_slice["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                    & (funding_slice["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
                ].copy()
            per_pair[pair] = stress_overlay_replay_pairwise_with_stop(
                df=df,
                trade_pair=pair,
                raw_signal=raw_signal_all[pair].loc[start:end].copy(),
                pair_config=candidate["pair_configs"][pair],
                overlay_inputs=overlay_inputs,
                funding_df=funding_slice,
                library=library,
                scenario=scenario,
                seed=seed_offset + window_idx * 100 + pair_idx,
                stop_loss_pct=stop_loss_pct,
            )
        windows[label] = {
            "start": start,
            "end": end,
            "bars": int(len(df)),
            "per_pair": per_pair,
            "aggregate": aggregate_metrics(per_pair),
        }
    return windows


def window_delta(baseline: dict[str, Any], stop2: dict[str, Any]) -> dict[str, Any]:
    return {
        "worst_pair_total_return_delta": float(stop2["aggregate"]["worst_pair_total_return"] - baseline["aggregate"]["worst_pair_total_return"]),
        "mean_total_return_delta": float(stop2["aggregate"]["mean_total_return"] - baseline["aggregate"]["mean_total_return"]),
        "worst_max_drawdown_delta": float(stop2["aggregate"]["worst_max_drawdown"] - baseline["aggregate"]["worst_max_drawdown"]),
        "worst_pair_avg_daily_return_delta": float(stop2["aggregate"]["worst_pair_avg_daily_return"] - baseline["aggregate"]["worst_pair_avg_daily_return"]),
        "mean_avg_daily_return_delta": float(stop2["aggregate"]["mean_avg_daily_return"] - baseline["aggregate"]["mean_avg_daily_return"]),
    }


def main() -> None:
    args = parse_args()
    pairs = parse_csv_tuple(args.pairs, str)
    scenario = resolve_scenario(args.scenario)
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
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=bool(args.refresh_cache))
    missing_ohlc = []
    for pair in pairs:
        required = [f"{pair}_{field}" for field in ("open", "high", "low", "close")]
        if not all(col in df_all.columns for col in required):
            missing_ohlc.append(pair)
    if missing_ohlc:
        raise RuntimeError(
            "Missing pair OHLC columns for "
            + ", ".join(missing_ohlc)
            + ". Re-run with --refresh-cache after backfilling those pairs."
        )
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    if not args.fetch_funding:
        missing_funding = [
            pair for pair in pairs
            if not (gp.DATA_DIR / f"{pair}_funding_{start_all}_{end_all}.csv").exists()
        ]
        if missing_funding:
            raise RuntimeError(
                "Missing funding caches for "
                + ", ".join(missing_funding)
                + ". Re-run with --fetch-funding to build them."
            )
    funding_all = {pair: load_or_fetch_funding(pair, start_all, end_all) for pair in pairs}

    selected_candidate = summary["selected_candidate"]
    baseline_windows = build_candidate_metrics(
        df_all,
        raw_signal_all,
        funding_all,
        library,
        selected_candidate,
        pairs,
        scenario,
        seed_offset=500000,
    )
    stop_windows = build_candidate_metrics_with_stop(
        df_all,
        raw_signal_all,
        funding_all,
        library,
        selected_candidate,
        pairs,
        scenario,
        seed_offset=900000,
        stop_loss_pct=float(args.stop_loss_pct),
    )

    windows_report: list[dict[str, Any]] = []
    summary_windows = summary.get("selected_candidate", {}).get("windows", {})
    for label, _, _ in DEFAULT_WINDOWS:
        baseline = baseline_windows[label]
        stop2 = stop_windows[label]
        windows_report.append(
            {
                "label": label,
                "baseline": baseline,
                "stop_loss_2pct": stop2,
                "delta": window_delta(baseline, stop2),
                "baseline_matches_summary": (
                    abs(
                        float(baseline["aggregate"]["worst_pair_total_return"])
                        - float(summary_windows.get(label, {}).get("aggregate", {}).get("worst_pair_total_return", baseline["aggregate"]["worst_pair_total_return"]))
                    )
                    < 1e-6
                ),
            }
        )

    report = {
        "strategy_class": "pairwise_regime_entry_stop_compare",
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "summary_path": str(Path(args.summary).resolve()),
        "base_summary_path": str(Path(args.base_summary).resolve()),
        "model_path": str(Path(args.model).resolve()),
        "pairs": list(pairs),
        "selected_candidate": selected_candidate,
        "scenario": asdict(scenario),
        "stop_loss_pct": float(args.stop_loss_pct),
        "dataset": {
            "start": start_all,
            "end": end_all,
            "bars": int(len(df_all)),
            "refresh_cache": bool(args.refresh_cache),
            "fetch_funding": bool(args.fetch_funding),
        },
        "windows": windows_report,
    }

    output_path = Path(args.report_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_safe(report), f, indent=2)

    print("=" * 96)
    print("Pairwise Regime Mixture Selected Candidate vs Entry-Anchored 2% Stop-Loss")
    print("=" * 96)
    print(f"Scenario : {scenario.name}")
    print(f"Pairs    : {', '.join(pairs)}")
    print(f"Data     : {start_all} -> {end_all} | bars={len(df_all)}")
    print(f"Stop     : {float(args.stop_loss_pct):.2%}")
    print("-" * 96)
    for row in windows_report:
        base = row["baseline"]["aggregate"]
        stop2 = row["stop_loss_2pct"]["aggregate"]
        stop_events = sum(
            int(metrics.get("stop_event_count", 0))
            for metrics in row["stop_loss_2pct"]["per_pair"].values()
        )
        print(
            f"{row['label']:>9} | "
            f"baseline worst={base['worst_pair_total_return']*100:+10.2f}% mdd={base['worst_max_drawdown']*100:+7.2f}% | "
            f"stop2 worst={stop2['worst_pair_total_return']*100:+10.2f}% mdd={stop2['worst_max_drawdown']*100:+7.2f}% "
            f"stops={stop_events}"
        )
    print("-" * 96)
    print(f"Summary saved: {output_path}")


if __name__ == "__main__":
    main()
