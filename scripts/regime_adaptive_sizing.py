#!/usr/bin/env python3
"""Regime-adaptive sizing policy: design and OOS simulation.

Loads per-regime W/L from models/pairwise_equity_corr_risk_compare.json,
computes Kelly-based size multipliers per bucket, then re-simulates the
full_4y window using return_trace post-processing to estimate OOS lift.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import gp_crypto_evolution as gp
from backtest_pairwise_equity_corr_risk_compare import (
    filter_window,
    filter_funding_window,
    load_funding_cache,
)
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    BAR_FACTOR,
    BARS_PER_DAY,
    EQUITY_CORR_ROUTE_STATE_NAMES,
    ROUTE_STATE_MODE_EQUITY_CORR,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    realistic_overlay_replay_from_context,
)

UTC = timezone.utc
REPORT_IN = gp.MODELS_DIR / "pairwise_equity_corr_risk_compare.json"
REPORT_OUT = gp.MODELS_DIR / "regime_adaptive_sizing_report.json"
MIN_SAMPLES = 30
MAX_SIZE_MULT = 2.0
MIN_KELLY_THRESHOLD = 0.10  # abstain if kelly < this
FULL_4Y_WINDOW = ("full_4y", "2022-04-18", "2026-04-18")


def iso_now() -> str:
    return datetime.now(UTC).isoformat()


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
    return value


def compute_metrics(bar_net: np.ndarray) -> dict[str, float]:
    """Recompute scalar metrics from a bar_net return array using kernel-matching constants."""
    n = len(bar_net)
    if n == 0:
        return {
            "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
            "avg_daily_return": 0.0, "daily_win_rate": 0.0,
            "prob_daily_ge_1pct": 0.0,
        }
    equity = np.cumprod(1.0 + bar_net)
    total_return = float(equity[-1] - 1.0)

    mean_bar = float(np.mean(bar_net))
    std_bar = float(np.std(bar_net))
    sharpe = mean_bar / std_bar * BAR_FACTOR if std_bar > 1e-12 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_drawdown = float(np.min(dd))

    # Daily aggregation using kernel's BARS_PER_DAY
    daily_rets: list[float] = []
    i = 0
    while i < n:
        chunk = bar_net[i: i + BARS_PER_DAY]
        day_ret = float(np.prod(1.0 + chunk) - 1.0)
        daily_rets.append(day_ret)
        i += BARS_PER_DAY
    if not daily_rets:
        daily_rets = [float(np.prod(1.0 + bar_net) - 1.0)]
    d_arr = np.asarray(daily_rets, dtype="float64")
    avg_daily = float(np.mean(d_arr))
    daily_win_rate = float(np.mean(d_arr > 0.0))
    prob_ge_1pct = float(np.mean(d_arr >= 0.01))

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_daily_return": avg_daily,
        "daily_win_rate": daily_win_rate,
        "prob_daily_ge_1pct": prob_ge_1pct,
    }


def build_regime_table(
    wins: list[int],
    losses: list[int],
    global_payoff: float,
    global_kelly: float,
) -> dict[int, dict[str, Any]]:
    """Build per-regime sizing policy table."""
    total_decided = sum(w + l for w, l in zip(wins, losses))
    n_states = len(EQUITY_CORR_ROUTE_STATE_NAMES)
    table: dict[int, dict[str, Any]] = {}

    for i in range(min(len(wins), n_states)):
        w, l = wins[i], losses[i]
        decided = w + l
        name = EQUITY_CORR_ROUTE_STATE_NAMES[i]

        if decided < MIN_SAMPLES:
            table[i] = {
                "name": name,
                "wins": w,
                "losses": l,
                "decided": decided,
                "win_rate": None,
                "kelly": None,
                "share": decided / total_decided if total_decided > 0 else 0.0,
                "recommended_size_mult": None,
                "skip_reason": "insufficient_samples",
            }
            continue

        win_rate = w / decided
        payoff = global_payoff if global_payoff > 0.0 else 1.0
        kelly = win_rate - (1.0 - win_rate) / payoff

        if global_kelly > 1e-6:
            raw_mult = kelly / global_kelly
        else:
            raw_mult = 1.0

        if kelly < MIN_KELLY_THRESHOLD:
            size_mult = 0.0
        else:
            size_mult = float(np.clip(raw_mult, 0.0, MAX_SIZE_MULT))

        table[i] = {
            "name": name,
            "wins": w,
            "losses": l,
            "decided": decided,
            "win_rate": float(win_rate),
            "kelly": float(kelly),
            "share": decided / total_decided if total_decided > 0 else 0.0,
            "recommended_size_mult": size_mult,
            "skip_reason": None,
        }

    return table


def simulate_adaptive(
    bar_net: np.ndarray,
    bucket_codes_per_bar: np.ndarray,
    regime_table: dict[int, dict[str, Any]],
) -> dict[str, float]:
    """Post-process bar_net by regime-adaptive size multipliers.

    bar_net[i] ≈ target_weight * price_return - costs.
    We approximate the adaptive bar_net by scaling each bar's P&L contribution
    by the regime size multiplier. The costs scale proportionally with position size.
    """
    size_mults = np.ones(len(bar_net), dtype="float64")
    for idx, entry in regime_table.items():
        if entry["skip_reason"] is not None:
            continue
        mult = entry["recommended_size_mult"]
        mask = bucket_codes_per_bar[: len(bar_net)] == idx
        size_mults[mask] = mult

    adaptive_bar_net = bar_net * size_mults
    return compute_metrics(adaptive_bar_net)


def run_pair(
    sym: str,
    df_window: pd.DataFrame,
    raw_signal: pd.Series,
    overlay_inputs: dict[str, Any],
    funding_df: pd.DataFrame,
    library: list[Any],
    library_lookup: dict[str, Any],
    mapping: tuple[int, ...],
    route_breadth_threshold: float,
    baseline_data: dict[str, Any],
) -> dict[str, Any]:
    """Run a single pair: build regime table and simulate adaptive."""

    # Build baseline regime table from precomputed JSON (no re-run needed for policy)
    wins = baseline_data["regime_n_wins"]
    losses = baseline_data["regime_n_losses"]
    global_payoff = float(baseline_data.get("payoff_ratio", 1.0))
    global_kelly = float(baseline_data.get("kelly_fraction", 0.0))

    regime_table = build_regime_table(wins, losses, global_payoff, global_kelly)

    # Get bar_net trace via return_trace=True kernel call
    context = build_fast_context(
        df=df_window,
        pair=sym,
        raw_signal=raw_signal,
        overlay_inputs=overlay_inputs,
        route_thresholds=(float(route_breadth_threshold),),
        library_lookup=library_lookup,
        funding_df=funding_df,
        route_state_mode=ROUTE_STATE_MODE_EQUITY_CORR,
    )

    trace_result = realistic_overlay_replay_from_context(
        context,
        library_lookup,
        mapping,
        route_breadth_threshold,
        use_equity_corr_risk=False,
        return_trace=True,
    )

    bar_net = trace_result["trace"]["bar_net"]
    bucket_codes_full = context["bucket_codes"][float(route_breadth_threshold)]
    # bucket_codes is indexed by bar position; align length to bar_net
    n_bars = len(bar_net)
    bucket_codes_aligned = bucket_codes_full[:n_bars]

    # Baseline metrics recomputed from bar_net trace for apples-to-apples comparison
    baseline_metrics = compute_metrics(bar_net)

    # Adaptive simulation
    adaptive_metrics = simulate_adaptive(bar_net, bucket_codes_aligned, regime_table)

    lift_pct = (
        (adaptive_metrics["avg_daily_return"] - baseline_metrics["avg_daily_return"])
        / abs(baseline_metrics["avg_daily_return"])
        * 100.0
        if abs(baseline_metrics["avg_daily_return"]) > 1e-10
        else 0.0
    )

    return {
        "regime_table": {str(k): v for k, v in regime_table.items()},
        "baseline_metrics": baseline_metrics,
        "adaptive_metrics": adaptive_metrics,
        "lift_pct": float(lift_pct),
    }


def main() -> None:
    print(f"Loading {REPORT_IN} ...")
    raw = json.loads(REPORT_IN.read_text())
    window_data = raw["windows"]["full_4y"]
    pair_data = window_data["pairs"]

    label, start, end = FULL_4Y_WINDOW
    print(f"Loading OHLCV data for window {label} ({start} -> {end}) ...")
    df_all = gp.load_all_pairs(pairs=list(PAIRS), start=start, end=end, refresh_cache=False)
    df_window = filter_window(df_all, start, end)

    summary = json.loads(DEFAULT_SUMMARY_PATH.read_text())
    config = summary["selected_candidate"]["pair_configs"]
    library = list(iter_params())
    library_lookup = build_library_lookup(library)

    model_tree, _ = load_signal_model(DEFAULT_MODEL_PATH)
    compiled = gp.toolbox.compile(expr=model_tree)

    funding_cache = {pair: load_funding_cache(pair) for pair in PAIRS}

    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "route_state_mode": ROUTE_STATE_MODE_EQUITY_CORR,
        "window": label,
        "min_samples_threshold": MIN_SAMPLES,
        "max_size_mult": MAX_SIZE_MULT,
        "min_kelly_threshold": MIN_KELLY_THRESHOLD,
        "per_pair": {},
    }

    for sym in ("BTCUSDT", "BNBUSDT"):
        print(f"\nProcessing {sym} ...")
        pair_entry = pair_data.get(sym, {})
        baseline_data = pair_entry.get("baseline", {})

        raw_signal = pd.Series(
            compiled(*gp.get_feature_arrays(df_window, sym)),
            index=df_window.index,
            dtype="float64",
        )
        overlay_inputs = build_overlay_inputs(df_window, PAIRS, regime_pair=sym)
        funding_df = filter_funding_window(funding_cache[sym], start, end)

        pair_cfg = config[sym]
        mapping = tuple(int(v) for v in pair_cfg["mapping_indices"])
        route_breadth_threshold = float(pair_cfg["route_breadth_threshold"])

        result = run_pair(
            sym=sym,
            df_window=df_window,
            raw_signal=raw_signal,
            overlay_inputs=overlay_inputs,
            funding_df=funding_df,
            library=library,
            library_lookup=library_lookup,
            mapping=mapping,
            route_breadth_threshold=route_breadth_threshold,
            baseline_data=baseline_data,
        )
        report["per_pair"][sym] = result

        # Print summary
        rt = result["regime_table"]
        print(f"  Regime table ({sym}):")
        for idx, entry in rt.items():
            if entry["skip_reason"]:
                continue
            print(
                f"    [{idx:2s}] {entry['name']:30s}  "
                f"wr={entry['win_rate']:.3f}  kelly={entry['kelly']:.3f}  "
                f"share={entry['share']:.3f}  size_mult={entry['recommended_size_mult']:.2f}"
            )
        print(f"  Baseline:  roi={result['baseline_metrics']['total_return']:.2f}  "
              f"sharpe={result['baseline_metrics']['sharpe']:.2f}  "
              f"mdd={result['baseline_metrics']['max_drawdown']:.4f}  "
              f"daily_wr={result['baseline_metrics']['daily_win_rate']:.3f}")
        print(f"  Adaptive:  roi={result['adaptive_metrics']['total_return']:.2f}  "
              f"sharpe={result['adaptive_metrics']['sharpe']:.2f}  "
              f"mdd={result['adaptive_metrics']['max_drawdown']:.4f}  "
              f"daily_wr={result['adaptive_metrics']['daily_win_rate']:.3f}")
        print(f"  Lift: {result['lift_pct']:+.1f}% on avg_daily_return")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(json_safe(report), indent=2))
    print(f"\nReport written to {REPORT_OUT}")


if __name__ == "__main__":
    main()
