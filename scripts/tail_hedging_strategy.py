#!/usr/bin/env python3
"""Tail-hedging strategy simulation for the pairwise GP overlay system.

Compares three post-hoc hedging strategies against the baseline full_4y replay
for BTCUSDT and BNBUSDT given extreme fat tails (Hill alpha 1.31 / 1.73).

Strategies:
  1. cvar_99_cut   - cut to 0 for 24h when rolling 30d return < CVaR-99 threshold
  2. vol_of_vol    - halve position for 24h when 1d realised vol > 2x rolling 30d median
  3. gross_cap     - hard cap target_weight to ±0.5 (vs ±1.0 default)

All hedging is applied post-hoc to the bar_net / target_weight trace returned
by the replay kernel.  The kernel itself is not modified.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── project path setup ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import gp_crypto_evolution as gp
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    load_or_fetch_funding,
    realistic_overlay_replay_from_context,
)
from search_gp_drawdown_overlay import iter_params

# ── constants ─────────────────────────────────────────────────────────────────
FULL_4Y_LABEL = "full_4y"
FULL_4Y_WINDOW = next(w for w in DEFAULT_WINDOWS if w[0] == FULL_4Y_LABEL)
BARS_PER_DAY = gp.periods_per_day(gp.TIMEFRAME)          # 288 bars per day (5m)
FUNDING_START = "2022-04-06"
FUNDING_END = "2026-04-06"
UTC = timezone.utc


# ── JSON serialiser ───────────────────────────────────────────────────────────
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


# ── metrics from a bar_net array ─────────────────────────────────────────────
def metrics_from_bar_net(bar_net: np.ndarray) -> dict[str, float]:
    """Compute ROI, MDD, Sharpe, CVaR-99, worst_day from 5m bar returns."""
    if len(bar_net) == 0:
        return {
            "roi": 0.0,
            "mdd": 0.0,
            "sharpe": 0.0,
            "cvar_99": 0.0,
            "worst_day": 0.0,
            "calmar": 0.0,
        }
    equity = np.cumprod(1.0 + bar_net) * gp.INITIAL_CASH
    roi = float(equity[-1] / gp.INITIAL_CASH - 1.0)

    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    mdd = float(np.min(dd))

    if np.std(bar_net) > 1e-12:
        bar_factor = math.sqrt(365.25 * 24.0 * 60.0 / 5.0)
        sharpe = float(np.mean(bar_net) / np.std(bar_net) * bar_factor)
    else:
        sharpe = 0.0

    # CVaR-99: mean of the worst 1% of bar returns
    sorted_ret = np.sort(bar_net)
    cutoff = max(1, int(math.floor(0.01 * len(sorted_ret))))
    cvar_99 = float(np.mean(sorted_ret[:cutoff]))

    # worst single day
    daily_ret = []
    for i in range(0, len(bar_net), BARS_PER_DAY):
        seg = bar_net[i : i + BARS_PER_DAY]
        if len(seg) > 0:
            daily_ret.append(float(np.prod(1.0 + seg) - 1.0))
    worst_day = float(min(daily_ret)) if daily_ret else 0.0

    calmar = float(roi / abs(mdd)) if abs(mdd) > 1e-10 else 0.0

    return {
        "roi": roi,
        "mdd": mdd,
        "sharpe": sharpe,
        "cvar_99": cvar_99,
        "worst_day": worst_day,
        "calmar": calmar,
    }


# ── strategy 1: CVaR-99 cut ───────────────────────────────────────────────────
def apply_cvar_99_cut(
    bar_net: np.ndarray,
    target_weight: np.ndarray,
    cvar_99_threshold: float,
    lookback_bars: int = 30 * BARS_PER_DAY,
    lockout_bars: int = BARS_PER_DAY,  # 24h
) -> tuple[np.ndarray, int]:
    """When rolling 30d return < CVaR-99 threshold, cut to 0 for 24h.

    Returns modified bar_net and count of triggered (protected) bars.
    """
    n = len(bar_net)
    out = bar_net.copy()
    days_protected = 0
    lockout_remaining = 0

    # rolling 30d return: compound product of last lookback_bars bars
    for i in range(n):
        # first compute rolling return up to bar i
        start = max(0, i - lookback_bars + 1)
        roll_ret = float(np.prod(1.0 + bar_net[start : i + 1]) - 1.0)

        if lockout_remaining > 0:
            # scale bar return by weight ratio (position cut to 0)
            tw = float(target_weight[i]) if i < len(target_weight) else 0.0
            if abs(tw) > 1e-10:
                out[i] = 0.0
            days_protected += 1
            lockout_remaining -= 1
        else:
            if roll_ret < cvar_99_threshold:
                # trigger: cut this bar and lock out for next lockout_bars
                tw = float(target_weight[i]) if i < len(target_weight) else 0.0
                if abs(tw) > 1e-10:
                    out[i] = 0.0
                lockout_remaining = lockout_bars
                days_protected += 1

    return out, days_protected


# ── strategy 2: vol-of-vol trigger ───────────────────────────────────────────
def apply_vol_of_vol(
    bar_net: np.ndarray,
    target_weight: np.ndarray,
    lookback_bars: int = 30 * BARS_PER_DAY,
    lockout_bars: int = BARS_PER_DAY,
    vol_mult: float = 2.0,
) -> tuple[np.ndarray, int]:
    """When 1d realised vol > vol_mult × rolling 30d median vol, halve position.

    Returns modified bar_net and count of triggered (protected) bars.
    """
    n = len(bar_net)
    out = bar_net.copy()
    days_protected = 0
    lockout_remaining = 0

    # precompute daily vol series (std of 5m returns within each day)
    daily_vols: list[float] = []
    for d in range(0, n, BARS_PER_DAY):
        seg = bar_net[d : d + BARS_PER_DAY]
        if len(seg) >= 2:
            daily_vols.append(float(np.std(seg)))
        else:
            daily_vols.append(0.0)

    # map bar index → day index
    bar_to_day = np.arange(n) // BARS_PER_DAY

    # rolling 30d median of daily vol (at the day level)
    lookback_days = lookback_bars // BARS_PER_DAY

    for i in range(n):
        d = int(bar_to_day[i])
        if lockout_remaining > 0:
            # halve position: scale bar return by 0.5
            tw = float(target_weight[i]) if i < len(target_weight) else 0.0
            if abs(tw) > 1e-10:
                out[i] = bar_net[i] * 0.5
            days_protected += 1
            lockout_remaining -= 1
        else:
            # compute today's vol and rolling median
            today_vol = daily_vols[d] if d < len(daily_vols) else 0.0
            hist_start = max(0, d - lookback_days)
            hist_vols = daily_vols[hist_start:d]
            if len(hist_vols) > 0:
                median_vol = float(np.median(hist_vols))
            else:
                median_vol = today_vol
            if median_vol > 1e-12 and today_vol > vol_mult * median_vol:
                tw = float(target_weight[i]) if i < len(target_weight) else 0.0
                if abs(tw) > 1e-10:
                    out[i] = bar_net[i] * 0.5
                lockout_remaining = lockout_bars
                days_protected += 1

    return out, days_protected


# ── strategy 3: gross cap ─────────────────────────────────────────────────────
def apply_gross_cap(
    bar_net: np.ndarray,
    target_weight: np.ndarray,
    cap: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Hard cap target_weight to ±cap.  Scale bar return proportionally.

    Returns modified bar_net and count of capped bars.
    """
    n = len(bar_net)
    out = bar_net.copy()
    days_protected = 0

    for i in range(n):
        tw = float(target_weight[i]) if i < len(target_weight) else 0.0
        if abs(tw) > cap + 1e-10:
            # scale the bar return by cap / |tw|
            scale = cap / abs(tw)
            out[i] = bar_net[i] * scale
            days_protected += 1

    return out, days_protected


# ── funding CSV loader (permissive: return empty if missing) ──────────────────
def _load_funding_csv(pair: str) -> pd.DataFrame:
    path = gp.DATA_DIR / f"{pair}_funding_{FUNDING_START}_{FUNDING_END}.csv"
    if not path.exists():
        candidates = sorted(gp.DATA_DIR.glob(f"{pair}_funding_{FUNDING_START}_*.csv"))
        if not candidates:
            return pd.DataFrame(columns=["fundingTime", "fundingRate"])
        path = candidates[-1]
    df = pd.read_csv(path)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, format="mixed")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return (
        df.dropna(subset=["fundingTime", "fundingRate"])
        .sort_values("fundingTime")
        .reset_index(drop=True)
    )


def load_funding(pair: str, start: str, end: str) -> pd.DataFrame:
    """Load funding from CSV/DB; return empty frame rather than raising."""
    try:
        pg = gp.load_funding_rates(pair, start, end)
    except Exception:
        pg = pd.DataFrame(columns=["fundingTime", "fundingRate"])
    csv = _load_funding_csv(pair)
    frames = [f for f in (pg, csv) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["fundingTime"], keep="first")
        .sort_values("fundingTime")
        .reset_index(drop=True)
    )
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return merged[(merged["fundingTime"] >= start_ts) & (merged["fundingTime"] < end_ts)].copy()


# ── recommend strategy by Calmar ──────────────────────────────────────────────
def recommend(results: dict[str, dict[str, float]]) -> str:
    """Return the strategy name with the highest Calmar ratio."""
    best = max(results.items(), key=lambda kv: kv[1].get("calmar", 0.0))
    return best[0]


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    summary = json.loads(DEFAULT_SUMMARY_PATH.read_text())
    config = summary["selected_candidate"]["pair_configs"]

    # load model
    model_tree, _ = load_signal_model(DEFAULT_MODEL_PATH)
    compiled = gp.toolbox.compile(expr=model_tree)

    # load tail risk for CVaR-99 thresholds
    tail_risk_path = ROOT / "models" / "tail_risk_report.json"
    tail_risk = json.loads(tail_risk_path.read_text())["per_pair"]

    # build library once
    library = list(iter_params())
    library_lookup = build_library_lookup(library)

    label, start, end = FULL_4Y_WINDOW
    df_all = gp.load_all_pairs(
        pairs=list(PAIRS), start=start, end=end, refresh_cache=False
    )

    per_pair_report: dict[str, Any] = {}

    for pair in PAIRS:
        print(f"\n=== {pair} ===")
        cvar_99_threshold = float(tail_risk[f"{pair}USDT" if "USDT" not in pair else pair]["CVaR_99"])

        # signal
        raw_signal = pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        overlay_inputs = build_overlay_inputs(df_all, PAIRS, regime_pair=pair)
        funding_df = load_funding(pair, start, end)

        mapping = tuple(int(v) for v in config[pair]["mapping_indices"])
        route_breadth_threshold = float(config[pair]["route_breadth_threshold"])
        route_state_mode = config[pair].get("route_state_mode", "equity_corr")

        context = build_fast_context(
            df=df_all,
            pair=pair,
            raw_signal=raw_signal,
            overlay_inputs=overlay_inputs,
            route_thresholds=(route_breadth_threshold,),
            library_lookup=library_lookup,
            funding_df=funding_df,
            route_state_mode=route_state_mode,
        )

        # run kernel with return_trace=True
        print(f"  Running baseline replay for {pair}...")
        trace_result = realistic_overlay_replay_from_context(
            context,
            library_lookup,
            mapping,
            route_breadth_threshold,
            use_equity_corr_risk=False,
            return_trace=True,
        )

        bar_net = trace_result["trace"]["bar_net"]           # shape (N,)
        target_weight = trace_result["trace"]["target_weight"]  # shape (N,)

        print(f"  Trace bars: {len(bar_net)}")

        # baseline metrics
        baseline_metrics = metrics_from_bar_net(bar_net)
        print(f"  Baseline: ROI={baseline_metrics['roi']:.3f}  MDD={baseline_metrics['mdd']:.3f}  "
              f"Sharpe={baseline_metrics['sharpe']:.2f}  Calmar={baseline_metrics['calmar']:.2f}")

        # strategy 1: CVaR-99 cut
        bar_net_s1, dp1 = apply_cvar_99_cut(
            bar_net, target_weight, cvar_99_threshold
        )
        s1_metrics = metrics_from_bar_net(bar_net_s1)
        s1_metrics["days_protected"] = dp1
        print(f"  CVaR-cut: ROI={s1_metrics['roi']:.3f}  MDD={s1_metrics['mdd']:.3f}  "
              f"Sharpe={s1_metrics['sharpe']:.2f}  Calmar={s1_metrics['calmar']:.2f}  dp={dp1}")

        # strategy 2: vol-of-vol
        bar_net_s2, dp2 = apply_vol_of_vol(bar_net, target_weight)
        s2_metrics = metrics_from_bar_net(bar_net_s2)
        s2_metrics["days_protected"] = dp2
        print(f"  Vol-of-vol: ROI={s2_metrics['roi']:.3f}  MDD={s2_metrics['mdd']:.3f}  "
              f"Sharpe={s2_metrics['sharpe']:.2f}  Calmar={s2_metrics['calmar']:.2f}  dp={dp2}")

        # strategy 3: gross cap ±0.5
        bar_net_s3, dp3 = apply_gross_cap(bar_net, target_weight, cap=0.5)
        s3_metrics = metrics_from_bar_net(bar_net_s3)
        s3_metrics["days_protected"] = dp3
        print(f"  Gross-cap: ROI={s3_metrics['roi']:.3f}  MDD={s3_metrics['mdd']:.3f}  "
              f"Sharpe={s3_metrics['sharpe']:.2f}  Calmar={s3_metrics['calmar']:.2f}  dp={dp3}")

        pair_results = {
            "baseline": baseline_metrics,
            "cvar_99_cut": s1_metrics,
            "vol_of_vol": s2_metrics,
            "gross_cap": s3_metrics,
        }
        per_pair_report[pair] = pair_results

    # build recommended per pair (best Calmar among hedged strategies only)
    recommended_per_pair: dict[str, str] = {}
    for pair in PAIRS:
        hedged = {
            k: v
            for k, v in per_pair_report[pair].items()
            if k != "baseline"
        }
        recommended_per_pair[pair] = recommend(hedged)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window": FULL_4Y_LABEL,
        "window_start": FULL_4Y_WINDOW[1],
        "window_end": FULL_4Y_WINDOW[2],
        "summary_path": str(DEFAULT_SUMMARY_PATH),
        "model_path": str(DEFAULT_MODEL_PATH),
        "tail_risk_path": str(tail_risk_path),
        "strategy_descriptions": {
            "cvar_99_cut": "Cut position to 0 for 24h when rolling 30d return < CVaR-99 threshold",
            "vol_of_vol": "Halve position for 24h when 1d realised vol > 2x rolling 30d median vol",
            "gross_cap": "Hard cap target_weight to ±0.5 (vs ±1.0 default)",
        },
        "per_pair": per_pair_report,
        "recommended_per_pair": recommended_per_pair,
    }

    out_path = ROOT / "models" / "tail_hedging_strategy_report.json"
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    print(f"\nReport written to {out_path}")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
