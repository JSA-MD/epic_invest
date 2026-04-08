#!/usr/bin/env python3
"""Run deterministic adverse execution stress tests for subset-pair candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from replay_regime_mixture_realistic import Candidate, load_model, resolve_candidate, summarize
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    aggregate_metrics,
    build_overlay_inputs,
    build_route_bucket_codes,
    load_or_fetch_funding,
    parse_csv_tuple,
)


@dataclass(frozen=True)
class StressScenario:
    name: str
    fee_rate: float
    base_slippage: float
    delay_bars: int
    partial_fill_ratio: float
    reject_prob: float
    vol_spike_threshold: float | None
    vol_spike_mult: float


SCENARIOS = (
    StressScenario(
        name="baseline_realistic",
        fee_rate=0.0004,
        base_slippage=0.0002,
        delay_bars=1,
        partial_fill_ratio=1.0,
        reject_prob=0.0,
        vol_spike_threshold=None,
        vol_spike_mult=1.0,
    ),
    StressScenario(
        name="conservative_book",
        fee_rate=0.0005,
        base_slippage=0.00035,
        delay_bars=1,
        partial_fill_ratio=0.90,
        reject_prob=0.01,
        vol_spike_threshold=0.90,
        vol_spike_mult=1.6,
    ),
    StressScenario(
        name="latency_partial",
        fee_rate=0.0005,
        base_slippage=0.00030,
        delay_bars=2,
        partial_fill_ratio=0.80,
        reject_prob=0.03,
        vol_spike_threshold=0.80,
        vol_spike_mult=1.8,
    ),
    StressScenario(
        name="ultra_conservative",
        fee_rate=0.0007,
        base_slippage=0.00055,
        delay_bars=3,
        partial_fill_ratio=0.65,
        reject_prob=0.08,
        vol_spike_threshold=0.70,
        vol_spike_mult=2.4,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adverse execution stress tests for baseline and selected subset-pair candidates.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--summary",
        default="models/gp_regime_mixture_btc_bnb_search_summary.json",
    )
    parser.add_argument(
        "--base-summary",
        default="models/gp_regime_mixture_search_summary.json",
        help="Summary file containing the overlay library.",
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--report-out",
        default="models/gp_regime_mixture_btc_bnb_stress_report.json",
    )
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
    return float(sign * precise)


def candidate_from_mapping(mapping_indices: list[int], route_breadth_threshold: float) -> Candidate:
    return Candidate(
        route_breadth_threshold=float(route_breadth_threshold),
        mapping_indices=tuple(int(v) for v in mapping_indices),
        mapping={},
    )


def stress_overlay_replay(
    df: pd.DataFrame,
    trade_pair: str,
    raw_signal: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    funding_df: pd.DataFrame,
    library: list[Any],
    candidate: Candidate,
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
    bucket_codes = build_route_bucket_codes(idx, overlay_inputs, candidate.route_breadth_threshold)
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

        active_idx = int(candidate.mapping_indices[int(bucket_codes[signal_idx])])
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
        target_qty = quantize_amount(target_notional / prev_close if abs(prev_close) > 1e-12 else 0.0, 0.001, 0.001)
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
                    if scenario.vol_spike_threshold is not None and np.isfinite(bar_vol_ann) and bar_vol_ann >= scenario.vol_spike_threshold:
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


def comparison_block(base_agg: dict[str, Any], sel_agg: dict[str, Any]) -> dict[str, Any]:
    return {
        "delta_mean_avg_daily_return": float(sel_agg["mean_avg_daily_return"] - base_agg["mean_avg_daily_return"]),
        "delta_worst_pair_avg_daily_return": float(sel_agg["worst_pair_avg_daily_return"] - base_agg["worst_pair_avg_daily_return"]),
        "delta_worst_max_drawdown": float(sel_agg["worst_max_drawdown"] - base_agg["worst_max_drawdown"]),
        "selected_beats_baseline_on_worst_pair": bool(
            sel_agg["worst_pair_avg_daily_return"] > base_agg["worst_pair_avg_daily_return"]
        ),
    }


def make_check(name: str, passed: bool, actual: float, target: float, comparator: str, note: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": float(actual),
        "target": float(target),
        "comparator": comparator,
        "note": note,
    }


def build_progressive_stress_profile(report: dict[str, Any], candidate_key: str) -> dict[str, Any]:
    scenarios = report["scenarios"]
    scenario_names = list(scenarios.keys())
    recent_2m_worst = [
        float(scenarios[name][candidate_key]["recent_2m"]["aggregate"]["worst_pair_avg_daily_return"])
        for name in scenario_names
    ]
    recent_6m_worst = [
        float(scenarios[name][candidate_key]["recent_6m"]["aggregate"]["worst_pair_avg_daily_return"])
        for name in scenario_names
    ]
    full_4y_worst = [
        float(scenarios[name][candidate_key]["full_4y"]["aggregate"]["worst_pair_avg_daily_return"])
        for name in scenario_names
    ]
    recent_2m_mdd = [
        abs(float(scenarios[name][candidate_key]["recent_2m"]["aggregate"]["worst_max_drawdown"]))
        for name in scenario_names
    ]
    recent_6m_mdd = [
        abs(float(scenarios[name][candidate_key]["recent_6m"]["aggregate"]["worst_max_drawdown"]))
        for name in scenario_names
    ]
    full_4y_mdd = [
        abs(float(scenarios[name][candidate_key]["full_4y"]["aggregate"]["worst_max_drawdown"]))
        for name in scenario_names
    ]
    recent_2m_delta = [
        float(scenarios[name]["comparison"]["recent_2m"]["delta_worst_pair_avg_daily_return"])
        for name in scenario_names
    ]
    recent_6m_delta = [
        float(scenarios[name]["comparison"]["recent_6m"]["delta_worst_pair_avg_daily_return"])
        for name in scenario_names
    ]
    full_4y_mean_delta = [
        float(scenarios[name]["comparison"]["full_4y"]["delta_mean_avg_daily_return"])
        for name in scenario_names
    ]
    full_4y_worst_delta = [
        float(scenarios[name]["comparison"]["full_4y"]["delta_worst_pair_avg_daily_return"])
        for name in scenario_names
    ]

    checks = [
        make_check(
            "recent_2m_worst_pair_positive_all",
            min(recent_2m_worst) >= 0.0,
            min(recent_2m_worst),
            0.0,
            ">=",
            "모든 스트레스 시나리오에서 최근 2개월 최악 코인이 음수가 아니어야 함",
        ),
        make_check(
            "recent_2m_worst_pair_vs_baseline_all",
            min(recent_2m_delta) > 0.0,
            min(recent_2m_delta),
            0.0,
            ">",
            "모든 스트레스 시나리오에서 최근 2개월 최악 코인이 baseline보다 좋아야 함",
        ),
        make_check(
            "recent_6m_worst_pair_positive_all",
            min(recent_6m_worst) > 0.0,
            min(recent_6m_worst),
            0.0,
            ">",
            "모든 스트레스 시나리오에서 최근 6개월 최악 코인이 양수여야 함",
        ),
        make_check(
            "recent_6m_worst_pair_vs_baseline_all",
            min(recent_6m_delta) > 0.0,
            min(recent_6m_delta),
            0.0,
            ">",
            "모든 스트레스 시나리오에서 최근 6개월 최악 코인이 baseline보다 좋아야 함",
        ),
        make_check(
            "full_4y_worst_pair_positive_all",
            min(full_4y_worst) > 0.0,
            min(full_4y_worst),
            0.0,
            ">",
            "모든 스트레스 시나리오에서 4년 전체 최악 코인이 양수여야 함",
        ),
        make_check(
            "full_4y_mean_vs_baseline_all",
            min(full_4y_mean_delta) > 0.0,
            min(full_4y_mean_delta),
            0.0,
            ">",
            "모든 스트레스 시나리오에서 4년 평균 일수익률이 baseline보다 좋아야 함",
        ),
        make_check(
            "full_4y_worst_pair_vs_baseline_majority",
            sum(delta > 0.0 for delta in full_4y_worst_delta) >= max(1, len(full_4y_worst_delta) - 1),
            sum(delta > 0.0 for delta in full_4y_worst_delta),
            max(1, len(full_4y_worst_delta) - 1),
            ">=",
            "4년 전체 최악 코인 기준 baseline 우위가 대부분 시나리오에서 유지돼야 함",
        ),
        make_check(
            "recent_2m_worst_mdd_cap_all",
            max(recent_2m_mdd) <= 0.18,
            max(recent_2m_mdd),
            0.18,
            "<=",
            "모든 스트레스 시나리오에서 최근 2개월 최악 MDD 18% 이내",
        ),
        make_check(
            "recent_6m_worst_mdd_cap_all",
            max(recent_6m_mdd) <= 0.17,
            max(recent_6m_mdd),
            0.17,
            "<=",
            "모든 스트레스 시나리오에서 최근 6개월 최악 MDD 17% 이내",
        ),
        make_check(
            "full_4y_worst_mdd_cap_all",
            max(full_4y_mdd) <= 0.26,
            max(full_4y_mdd),
            0.26,
            "<=",
            "모든 스트레스 시나리오에서 4년 전체 최악 MDD 26% 이내",
        ),
    ]
    return {
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


def build_target_060_stress_profile(report: dict[str, Any], candidate_key: str) -> dict[str, Any]:
    scenarios = report["scenarios"]
    scenario_names = list(scenarios.keys())
    recent_2m_worst = [
        float(scenarios[name][candidate_key]["recent_2m"]["aggregate"]["worst_pair_avg_daily_return"])
        for name in scenario_names
    ]
    recent_6m_worst = [
        float(scenarios[name][candidate_key]["recent_6m"]["aggregate"]["worst_pair_avg_daily_return"])
        for name in scenario_names
    ]
    full_4y_worst = [
        float(scenarios[name][candidate_key]["full_4y"]["aggregate"]["worst_pair_avg_daily_return"])
        for name in scenario_names
    ]
    recent_6m_mdd = [
        abs(float(scenarios[name][candidate_key]["recent_6m"]["aggregate"]["worst_max_drawdown"]))
        for name in scenario_names
    ]
    full_4y_mdd = [
        abs(float(scenarios[name][candidate_key]["full_4y"]["aggregate"]["worst_max_drawdown"]))
        for name in scenario_names
    ]

    checks = [
        make_check(
            "recent_2m_worst_pair_target_060_all",
            min(recent_2m_worst) >= 0.006,
            min(recent_2m_worst),
            0.006,
            ">=",
            "모든 스트레스 시나리오에서 최근 2개월 최악 코인 0.6%/day 이상",
        ),
        make_check(
            "recent_6m_worst_pair_target_060_all",
            min(recent_6m_worst) >= 0.006,
            min(recent_6m_worst),
            0.006,
            ">=",
            "모든 스트레스 시나리오에서 최근 6개월 최악 코인 0.6%/day 이상",
        ),
        make_check(
            "full_4y_worst_pair_target_060_all",
            min(full_4y_worst) >= 0.006,
            min(full_4y_worst),
            0.006,
            ">=",
            "모든 스트레스 시나리오에서 4년 전체 최악 코인 0.6%/day 이상",
        ),
        make_check(
            "recent_6m_worst_mdd_cap_all",
            max(recent_6m_mdd) <= 0.17,
            max(recent_6m_mdd),
            0.17,
            "<=",
            "모든 스트레스 시나리오에서 최근 6개월 최악 MDD 17% 이내",
        ),
        make_check(
            "full_4y_worst_mdd_cap_all",
            max(full_4y_mdd) <= 0.20,
            max(full_4y_mdd),
            0.20,
            "<=",
            "모든 스트레스 시나리오에서 4년 전체 최악 MDD 20% 이내",
        ),
    ]
    return {
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }


def build_candidate_metrics(
    df_all: pd.DataFrame,
    raw_signal_all: dict[str, pd.Series],
    funding_all: dict[str, pd.DataFrame],
    library: list[Any],
    candidate: Candidate,
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
            per_pair[pair] = stress_overlay_replay(
                df,
                trade_pair=pair,
                raw_signal=raw_signal_all[pair].loc[start:end].copy(),
                overlay_inputs=overlay_inputs,
                funding_df=funding_slice,
                library=library,
                candidate=candidate,
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

    baseline_candidate = candidate_from_mapping(
        summary["baseline_candidate"]["mapping_indices"],
        summary["baseline_candidate"]["route_breadth_threshold"],
    )
    selected_candidate = candidate_from_mapping(
        summary["selected_candidate"]["mapping_indices"],
        summary["selected_candidate"]["route_breadth_threshold"],
    )

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
            "assumptions": asdict(scenario),
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
        "baseline_candidate": summary["baseline_candidate"],
        "selected_candidate": {
            "route_breadth_threshold": summary["selected_candidate"]["route_breadth_threshold"],
            "mapping_indices": summary["selected_candidate"]["mapping_indices"],
        },
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
    report["promotion_decision"] = {
        "selected_passes_target_060_stress": report["profiles"]["target_060_stress"]["selected"]["passed"],
        "selected_passes_progressive_stress": report["profiles"]["progressive_stress"]["selected"]["passed"],
        "selected_candidate_ready_for_merge": report["profiles"]["target_060_stress"]["selected"]["passed"],
        "status": (
            "target_060_stress_pass"
            if report["profiles"]["target_060_stress"]["selected"]["passed"]
            else (
                "progressive_stress_pass"
                if report["profiles"]["progressive_stress"]["selected"]["passed"]
                else "stress_fail"
            )
        ),
    }
    out_path = Path(args.report_out)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
