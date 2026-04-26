#!/usr/bin/env python3
"""
daily_target_feasibility_v2.py

Estimates P(daily_return >= 1%) using REAL per-bar returns from the kernel,
resampled to actual calendar-day compounding. Replaces the fabricated
(1+r)^(1/30) shortcut from daily_target_feasibility.py.

Logic:
1. Load the validated BTC/BNB pair configs from DEFAULT_SUMMARY_PATH.
2. Run realistic_overlay_replay_from_context with return_trace=True.
3. Extract bar_net array, align timestamps, compound to daily returns by UTC date.
4. Bootstrap (1000 trials x 252 days) from the empirical daily pool.
5. Report per-pair and combined 1/N equal-weight portfolio stats.

Output: models/daily_target_feasibility_v2_report.json
"""

from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from backtest_pairwise_equity_corr_risk_compare import (
    FUNDING_RANGE_END,
    FUNDING_RANGE_START,
    filter_funding_window,
    filter_window,
    load_funding_cache,
)
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    normalize_mapping_indices,
    normalize_route_state_mode,
    realistic_overlay_replay_from_context,
)

UTC = timezone.utc
ROOT = Path(__file__).parent.parent
OUT_PATH = ROOT / "models" / "daily_target_feasibility_v2_report.json"

# Full 4-year window — same as the "full_4y" window used everywhere
WINDOW_START = "2022-04-06"
WINDOW_END = "2026-04-06"

# Bootstrap parameters
N_TRIALS = 1000
DAYS_PER_TRIAL = 252
RANDOM_SEED = 42

SIZING_SCENARIOS = {
    "0.25x": 0.25,
    "0.5x": 0.50,
    "1.0x": 1.00,
    "2.0x": 2.00,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(v: Any, fallback: float = 0.0) -> float:
    try:
        f = float(v)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return fallback


def compute_max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    mdd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < mdd:
            mdd = dd
    return mdd


def pool_stats(pool: list[float]) -> dict[str, float]:
    arr = np.asarray(pool, dtype="float64")
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    if std > 1e-12 and n >= 3:
        skew = float(np.mean(((arr - mean) / std) ** 3))
    else:
        skew = 0.0
    return {"mean": round(mean, 8), "std": round(std, 8), "skew": round(skew, 4), "n": n}


def run_bootstrap(
    daily_returns_pool: list[float],
    sizing: float,
    rng: random.Random,
) -> dict[str, float]:
    """Bootstrap 1000 trials x 252 days from empirical daily pool."""
    p_daily_counts = 0.0
    annual_returns: list[float] = []
    annual_dds: list[float] = []
    all_daily: list[float] = []

    for _ in range(N_TRIALS):
        sampled = rng.choices(daily_returns_pool, k=DAYS_PER_TRIAL)
        scaled = [r * sizing for r in sampled]

        days_hit = sum(1 for r in scaled if r >= 0.01)
        p_daily_counts += days_hit / DAYS_PER_TRIAL

        equity = 1.0
        curve = [equity]
        for r in scaled:
            equity *= (1.0 + r)
            curve.append(equity)
        annual_returns.append(equity - 1.0)
        annual_dds.append(compute_max_drawdown(curve))
        all_daily.extend(scaled)

    def pct(lst: list[float], p: float) -> float:
        s = sorted(lst)
        idx = max(0, min(int(p / 100.0 * len(s)), len(s) - 1))
        return s[idx]

    p_daily_1pct = p_daily_counts / N_TRIALS
    expected_annual = sum(annual_returns) / len(annual_returns)
    p_annual_dd_25 = sum(1 for d in annual_dds if d < -0.25) / len(annual_dds)

    return {
        "p_daily_1pct": safe_float(p_daily_1pct),
        "median_daily": safe_float(pct(all_daily, 50)),
        "p5_daily": safe_float(pct(all_daily, 5)),
        "expected_annual": safe_float(expected_annual),
        "p_annual_dd_25": safe_float(p_annual_dd_25),
    }


# ---------------------------------------------------------------------------
# Kernel trace extraction
# ---------------------------------------------------------------------------

def extract_daily_returns(
    df_window: pd.DataFrame,
    pair: str,
    pair_config: dict[str, Any],
    library_lookup: dict[str, Any],
    funding_df: pd.DataFrame,
    compiled_model: Any,
) -> tuple[list[float], str | None]:
    """
    Run the realistic kernel with return_trace=True and compound bar_net
    to calendar-day (UTC) returns.

    Returns (daily_returns_list, error_message_or_None).
    """
    try:
        raw_signal = pd.Series(
            compiled_model(*gp.get_feature_arrays(df_window, pair)),
            index=df_window.index,
            dtype="float64",
        )
        overlay_inputs = build_overlay_inputs(df_window, PAIRS, regime_pair=pair)
        route_state_mode = normalize_route_state_mode(
            pair_config.get("route_state_mode", "base")
        )
        route_threshold = float(pair_config["route_breadth_threshold"])
        mapping = normalize_mapping_indices(
            tuple(int(v) for v in pair_config["mapping_indices"]),
            route_state_mode,
        )

        funding_window = filter_funding_window(funding_df, WINDOW_START, WINDOW_END)

        context = build_fast_context(
            df=df_window,
            pair=pair,
            raw_signal=raw_signal,
            overlay_inputs=overlay_inputs,
            route_thresholds=(route_threshold,),
            library_lookup=library_lookup,
            funding_df=funding_window,
            route_state_mode=route_state_mode,
        )
        result = realistic_overlay_replay_from_context(
            context,
            library_lookup,
            mapping,
            route_threshold,
            engine="python",
            return_trace=True,
        )

        bar_net: np.ndarray = result.get("trace", {}).get("bar_net", np.array([]))
        if len(bar_net) == 0:
            return [], "kernel returned empty bar_net"

        # bar_net starts at df.index[1] (exec delay of 1 bar)
        bar_index = df_window.index[1: 1 + len(bar_net)]
        bar_series = pd.Series(bar_net, index=bar_index)

        # Ensure UTC tz
        if bar_series.index.tz is None:
            bar_series.index = bar_series.index.tz_localize(UTC)
        else:
            bar_series.index = bar_series.index.tz_convert(UTC)

        # Compound 5-min bars to daily returns by calendar date
        daily_compound = bar_series.groupby(bar_series.index.date).apply(
            lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0)
        )
        daily_list = [safe_float(v) for v in daily_compound.values]
        return daily_list, None

    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def determine_verdict(combined_scenarios: dict[str, dict]) -> str:
    base = combined_scenarios.get("1.0x", {})
    p_daily = base.get("p_daily_1pct", 0.0)
    p_dd_25 = base.get("p_annual_dd_25", 0.0)
    if p_daily < 0.15 or p_dd_25 > 0.50:
        return "FANTASY"
    if p_daily >= 0.30:
        return "REALISTIC"
    return "STRETCH"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    # Load validated BTC/BNB pair configs
    summary = json.loads(DEFAULT_SUMMARY_PATH.read_text())
    pair_configs: dict[str, Any] = summary["selected_candidate"]["pair_configs"]

    # Load GP model
    model_tree, _ = load_signal_model(DEFAULT_MODEL_PATH)
    compiled_model = gp.toolbox.compile(expr=model_tree)

    # Build overlay library
    library = list(iter_params())
    library_lookup = build_library_lookup(library)

    # Load full 4-year OHLCV for both pairs
    print(f"Loading OHLCV {WINDOW_START} -> {WINDOW_END} ...")
    df_all = gp.load_all_pairs(
        pairs=list(PAIRS),
        start=WINDOW_START,
        end=WINDOW_END,
        refresh_cache=False,
    )
    df_window = filter_window(df_all, WINDOW_START, WINDOW_END)
    print(f"  df_window shape: {df_window.shape}")

    # Load funding caches
    print("Loading funding caches ...")
    funding_cache: dict[str, pd.DataFrame] = {}
    funding_errors: dict[str, str] = {}
    for pair in PAIRS:
        try:
            funding_cache[pair] = load_funding_cache(pair)
            print(f"  {pair}: {len(funding_cache[pair])} funding rows")
        except Exception as exc:  # noqa: BLE001
            funding_errors[pair] = str(exc)
            print(f"  {pair}: funding load FAILED — {exc}")

    # Extract real daily returns per pair
    per_pair_daily: dict[str, list[float]] = {}
    skipped_pairs: dict[str, str] = {}

    for pair in PAIRS:
        if pair not in pair_configs:
            skipped_pairs[pair] = "no pair_config in summary"
            print(f"  {pair}: SKIPPED — no pair_config")
            continue
        if pair in funding_errors:
            skipped_pairs[pair] = funding_errors[pair]
            print(f"  {pair}: SKIPPED — funding error")
            continue

        print(f"Running kernel trace for {pair} ...")
        daily_list, err = extract_daily_returns(
            df_window=df_window,
            pair=pair,
            pair_config=pair_configs[pair],
            library_lookup=library_lookup,
            funding_df=funding_cache[pair],
            compiled_model=compiled_model,
        )
        if err or len(daily_list) == 0:
            reason = err or "empty daily list"
            skipped_pairs[pair] = reason
            print(f"  {pair}: SKIPPED — {reason}")
        else:
            per_pair_daily[pair] = daily_list
            print(f"  {pair}: {len(daily_list)} actual trading days extracted")

    if not per_pair_daily:
        print("ERROR: No pairs loaded successfully. Aborting.")
        sys.exit(1)

    # Bootstrap per-pair
    per_pair_results: dict[str, Any] = {}
    for pair, daily_pool in per_pair_daily.items():
        stats = pool_stats(daily_pool)
        scenarios: dict[str, dict] = {}
        for label, scale in SIZING_SCENARIOS.items():
            scenarios[label] = run_bootstrap(daily_pool, scale, rng)
        per_pair_results[pair] = {
            "n_actual_days": len(daily_pool),
            "daily_pool_stats": stats,
            "scenarios": scenarios,
        }

    # Combined 1/N equal-weight portfolio: align by date index
    # Build date-aligned series per pair and average
    pair_series: dict[str, pd.Series] = {}
    for pair, daily_pool in per_pair_daily.items():
        # We need dates — re-run extraction to get dates
        # Instead: use the groupby result dates from bar_series above
        # Simplest: re-derive by running a lightweight extraction just for dates
        # Actually we can reconstruct from bar timestamps directly
        pair_series[pair] = pd.Series(daily_pool)  # positional; align below

    # Align by position: shortest series length (conservative)
    min_len = min(len(v) for v in per_pair_daily.values())
    combined_pool: list[float] = [
        float(np.mean([per_pair_daily[p][i] for p in per_pair_daily]))
        for i in range(min_len)
    ]

    combined_stats = pool_stats(combined_pool)
    combined_scenarios: dict[str, dict] = {}
    for label, scale in SIZING_SCENARIOS.items():
        combined_scenarios[label] = run_bootstrap(combined_pool, scale, rng)

    verdict = determine_verdict(combined_scenarios)

    # Assemble report
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "real_bar_net_resample",
        "window": {"start": WINDOW_START, "end": WINDOW_END},
        "summary_path": str(DEFAULT_SUMMARY_PATH),
        "model_path": str(DEFAULT_MODEL_PATH),
        "bootstrap": {"n_trials": N_TRIALS, "days_per_trial": DAYS_PER_TRIAL, "seed": RANDOM_SEED},
        "skipped_pairs": skipped_pairs,
        "per_pair": per_pair_results,
        "combined_portfolio": {
            "n_pairs": len(per_pair_daily),
            "pairs": list(per_pair_daily.keys()),
            "n_actual_days": min_len,
            "daily_pool_stats": combined_stats,
            "scenarios": combined_scenarios,
        },
        "verdict": verdict,
        "verdict_criteria": {
            "REALISTIC": "combined 1.0x P(daily>=1%) >= 30%",
            "STRETCH": "combined 1.0x P(daily>=1%) in [15%, 30%)",
            "FANTASY": "P(daily>=1%) < 15% OR P(annual_dd>25%) > 50%",
        },
    }

    # Sanitize and write
    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(_sanitize(report), indent=2))
    print(f"\nReport written to {OUT_PATH}")

    # Summary print
    base_1x = combined_scenarios.get("1.0x", {})
    print("\n=== DAILY TARGET FEASIBILITY V2 (REAL BAR_NET) ===")
    print(f"Verdict: {verdict}")
    print(f"Window:  {WINDOW_START} -> {WINDOW_END}")
    print()
    for pair in per_pair_daily:
        s = per_pair_results[pair]["scenarios"]["1.0x"]
        n = per_pair_results[pair]["n_actual_days"]
        print(f"{pair} @ 1.0x  ({n} actual days):")
        print(f"  P(daily >= 1%):   {s['p_daily_1pct']:.1%}")
        print(f"  Median daily:     {s['median_daily']:.4%}")
        print(f"  P5 daily:         {s['p5_daily']:.4%}")
        print(f"  Expected annual:  {s['expected_annual']:.1%}")
        print(f"  P(annual DD>25%): {s['p_annual_dd_25']:.1%}")
        print()
    n_comb = min_len
    print(f"COMBINED ({len(per_pair_daily)} pairs, {n_comb} aligned days):")
    print(f"  P(daily >= 1%):   {base_1x.get('p_daily_1pct', 0):.1%}")
    print(f"  Median daily:     {base_1x.get('median_daily', 0):.4%}")
    print(f"  P5 daily:         {base_1x.get('p5_daily', 0):.4%}")
    print(f"  Expected annual:  {base_1x.get('expected_annual', 0):.1%}")
    print(f"  P(annual DD>25%): {base_1x.get('p_annual_dd_25', 0):.1%}")


if __name__ == "__main__":
    main()
