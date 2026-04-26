#!/usr/bin/env python3
"""Kelly / vol-target sizing optimization using the realistic pairwise kernel.

Re-runs the full_4y kernel for BTCUSDT and BNBUSDT with return_trace=True,
then rescales the bar-level net returns to simulate Quarter / Half / Full /
2x Kelly and vol-targeting scenarios.  Reports daily-return distributions,
hit rates, MDD, Sharpe, and risk-of-ruin per scenario.

Output: models/sizing_optimization_report.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gp_crypto_evolution as gp
from backtest_pairwise_equity_corr_risk_compare import (
    load_funding_cache,
    filter_window,
    filter_funding_window,
)
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    realistic_overlay_replay_from_context,
)

UTC = timezone.utc
FULL_4Y_LABEL = "full_4y"
KELLY_MULTS = [0.25, 0.5, 1.0, 2.0]
KELLY_NAMES = {0.25: "quarter_kelly", 0.5: "half_kelly", 1.0: "full_kelly", 2.0: "two_x_kelly"}
VOL_TARGETS_ANN = [0.30, 0.50, 1.00]
BARS_PER_DAY = gp.periods_per_day(gp.TIMEFRAME)
ANNUAL_TRADING_DAYS = 365.25
RUIN_THRESHOLD = -0.50  # equity drawdown that constitutes ruin


# ---------------------------------------------------------------------------
# Helper: daily compounding from bar-level returns
# ---------------------------------------------------------------------------

def _bar_to_daily(bar_net: np.ndarray) -> np.ndarray:
    """Resample 5-min bar returns to daily compounded returns."""
    n = len(bar_net)
    bars_per = int(BARS_PER_DAY)
    n_full = (n // bars_per) * bars_per
    if n_full == 0:
        return np.array([], dtype="float64")
    reshaped = bar_net[:n_full].reshape(-1, bars_per)
    daily = np.prod(1.0 + reshaped, axis=1) - 1.0
    return daily


def _equity_curve_from_bar(bar_net: np.ndarray) -> np.ndarray:
    equity = np.empty(len(bar_net) + 1, dtype="float64")
    equity[0] = 1.0
    np.cumprod(1.0 + bar_net, out=equity[1:])
    return equity


def _mdd(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak > 1e-12, peak, 1e-12) - 1.0
    return float(np.min(dd))


def _sharpe_annualized(daily_rets: np.ndarray) -> float:
    if len(daily_rets) < 2:
        return 0.0
    std = float(np.std(daily_rets))
    if std < 1e-12:
        return 0.0
    return float(np.mean(daily_rets) / std * np.sqrt(ANNUAL_TRADING_DAYS))


def _ruin_prob(bar_net: np.ndarray) -> float:
    """Fraction of rolling equity paths that ever breach -50% from peak."""
    equity = _equity_curve_from_bar(bar_net)
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak > 1e-12, peak, 1e-12) - 1.0
    ever_ruined = float(np.any(dd <= RUIN_THRESHOLD))
    return ever_ruined


def scenario_stats(bar_net_base: np.ndarray, scale: float) -> dict[str, Any]:
    """Compute all metrics for a rescaled bar-net array."""
    bar_net = bar_net_base * scale
    daily = _bar_to_daily(bar_net)
    equity = _equity_curve_from_bar(bar_net)
    avg_daily = float(np.mean(daily)) if len(daily) else 0.0
    std_daily = float(np.std(daily)) if len(daily) else 0.0
    p_daily_1pct = float(np.mean(daily >= 0.01)) if len(daily) else 0.0
    p_tail_neg3pct = float(np.mean(daily <= -0.03)) if len(daily) else 0.0
    mdd = _mdd(equity)
    sharpe = _sharpe_annualized(daily)
    ruin = _ruin_prob(bar_net)
    return {
        "scale": float(scale),
        "avg_daily_return": avg_daily,
        "std_daily_return": std_daily,
        "p_daily_1pct": p_daily_1pct,
        "p_tail_neg3pct": p_tail_neg3pct,
        "mdd": mdd,
        "sharpe": sharpe,
        "ruin_prob": ruin,
    }


def vol_target_stats(bar_net_base: np.ndarray, target_vol_ann: float) -> dict[str, Any]:
    """Scale bar returns to hit target_vol_ann, then compute metrics."""
    ann_vol_base = float(np.std(bar_net_base) * np.sqrt(BARS_PER_DAY * ANNUAL_TRADING_DAYS))
    if ann_vol_base < 1e-8:
        scale = 1.0
    else:
        scale = target_vol_ann / ann_vol_base
    stats = scenario_stats(bar_net_base, scale)
    stats["target_vol_ann"] = target_vol_ann
    stats["realized_scale"] = float(scale)
    return stats


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def _recommend(pair: str, kelly_scenarios: dict[str, Any]) -> dict[str, Any]:
    """Pick the best Kelly multiplier: highest Sharpe with MDD > -60% and ruin < 5%."""
    best_key = None
    best_sharpe = -1e9
    for k, v in kelly_scenarios.items():
        if v["mdd"] < -0.60 or v["ruin_prob"] > 0.05:
            continue
        if v["sharpe"] > best_sharpe:
            best_sharpe = v["sharpe"]
            best_key = k
    if best_key is None:
        best_key = "quarter_kelly"
    v = kelly_scenarios[best_key]
    return {
        "pair": pair,
        "multiplier": best_key,
        "sharpe": v["sharpe"],
        "p_daily_1pct": v["p_daily_1pct"],
        "mdd": v["mdd"],
        "ruin_prob": v["ruin_prob"],
        "why": (
            f"Highest Sharpe ({v['sharpe']:.2f}) among scenarios with MDD > -60% "
            f"and ruin < 5%.  P(daily>=1%) = {v['p_daily_1pct']:.1%}."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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


def main() -> None:
    summary = json.loads(DEFAULT_SUMMARY_PATH.read_text())
    config = summary["selected_candidate"]["pair_configs"]
    library = list(iter_params())
    library_lookup = build_library_lookup(library)

    model_tree, _ = load_signal_model(DEFAULT_MODEL_PATH)
    compiled = gp.toolbox.compile(expr=model_tree)

    # Determine full_4y window
    full_4y_window = next(w for w in DEFAULT_WINDOWS if w[0] == FULL_4Y_LABEL)
    _, start_all, end_all = full_4y_window

    print(f"Loading OHLCV for {PAIRS} [{start_all} .. {end_all}] ...")
    df_all = gp.load_all_pairs(pairs=list(PAIRS), start=start_all, end=end_all, refresh_cache=False)

    print("Loading funding caches ...")
    funding_cache = {pair: load_funding_cache(pair) for pair in PAIRS}

    df_window = filter_window(df_all, start_all, end_all)

    per_pair: dict[str, Any] = {}
    recommendations: list[dict[str, Any]] = []

    for pair in PAIRS:
        print(f"\n--- {pair} ---")
        pair_cfg = config[pair]
        mapping = tuple(int(v) for v in pair_cfg["mapping_indices"])
        route_breadth_threshold = float(pair_cfg["route_breadth_threshold"])

        raw_signal = pd.Series(
            compiled(*gp.get_feature_arrays(df_window, pair)),
            index=df_window.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        overlay_inputs = build_overlay_inputs(df_window, list(PAIRS), regime_pair=pair)
        funding_df = filter_funding_window(funding_cache[pair], start_all, end_all)

        context = build_fast_context(
            df=df_window,
            pair=pair,
            raw_signal=raw_signal,
            overlay_inputs=overlay_inputs,
            route_thresholds=(route_breadth_threshold,),
            library_lookup=library_lookup,
            funding_df=funding_df,
        )

        result = realistic_overlay_replay_from_context(
            context,
            library_lookup,
            mapping,
            route_breadth_threshold,
            return_trace=True,
        )

        trace = result["trace"]
        bar_net: np.ndarray = trace["bar_net"].astype("float64")

        print(f"  Bars: {len(bar_net)}, base avg_daily_return: {result.get('avg_daily_return', 'n/a')}")

        # Kelly scenarios
        kelly_scenarios: dict[str, Any] = {}
        for mult in KELLY_MULTS:
            name = KELLY_NAMES[mult]
            kelly_scenarios[name] = scenario_stats(bar_net, mult)
            print(f"  {name}: Sharpe={kelly_scenarios[name]['sharpe']:.2f}, "
                  f"P(>=1%)={kelly_scenarios[name]['p_daily_1pct']:.1%}, "
                  f"MDD={kelly_scenarios[name]['mdd']:.1%}")

        # Vol-target scenarios
        vol_target_scenarios: dict[str, Any] = {}
        for vt in VOL_TARGETS_ANN:
            key = f"vol_{int(vt*100)}pct"
            vol_target_scenarios[key] = vol_target_stats(bar_net, vt)
            v = vol_target_scenarios[key]
            print(f"  {key} (scale={v['realized_scale']:.2f}): "
                  f"Sharpe={v['sharpe']:.2f}, MDD={v['mdd']:.1%}")

        per_pair[pair] = {
            "kelly_scenarios": kelly_scenarios,
            "vol_target_scenarios": vol_target_scenarios,
            "base_stats": {
                "bars": int(len(bar_net)),
                "avg_daily_return": float(result.get("avg_daily_return", 0.0)),
                "max_drawdown": float(result.get("max_drawdown", 0.0)),
                "sharpe": float(result.get("sharpe", 0.0)),
            },
        }

        recommendations.append(_recommend(pair, kelly_scenarios))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window": FULL_4Y_LABEL,
        "window_start": start_all,
        "window_end": end_all,
        "summary_path": str(DEFAULT_SUMMARY_PATH),
        "model_path": str(DEFAULT_MODEL_PATH),
        "per_pair": per_pair,
        "recommended_sizing": recommendations,
    }

    out_path = gp.MODELS_DIR / "sizing_optimization_report.json"
    out_path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False))
    print(f"\nReport written to {out_path}")

    # Summary printout
    print("\n=== RECOMMENDED SIZING ===")
    for rec in recommendations:
        print(f"  {rec['pair']}: {rec['multiplier']} | "
              f"P(daily>=1%)={rec['p_daily_1pct']:.1%} | "
              f"Sharpe={rec['sharpe']:.2f} | MDD={rec['mdd']:.1%} | "
              f"Ruin={rec['ruin_prob']:.0%}")
        print(f"    {rec['why']}")


if __name__ == "__main__":
    main()
