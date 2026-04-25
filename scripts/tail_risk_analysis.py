#!/usr/bin/env python3
"""Tail / extreme risk quantification for the live pairwise strategy (BTC + BNB).

Addresses Taleb's critique that worst_day alone is insufficient. Computes:
- VaR-95, VaR-99, CVaR-95, CVaR-99 (empirical, on daily returns)
- Skewness, excess kurtosis
- Hill tail index (lower tail)
- Rolling drawdown horizons: 1d, 7d, 30d
- Gaussian-implied MDD vs actual MDD (tail fatness ratio)

Usage:
    python tail_risk_analysis.py [--window full_4y] [--pairs BTCUSDT,BNBUSDT]
                                  [--report-out models/tail_risk_report.json]
                                  [--summary-path ...] [--model-path ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup – scripts/ is the cwd when running directly, but imports live
# alongside this file so we ensure the directory is on sys.path.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gp_crypto_evolution as gp
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    build_overlay_inputs,
    realistic_overlay_replay_from_context,
    build_fast_context,
    build_library_lookup,
    ROUTE_STATE_MODE_BASE,
)

UTC = timezone.utc
FUNDING_RANGE_START = "2022-04-06"
FUNDING_RANGE_END = "2026-04-06"
MIN_BARS_PER_DAY = 12  # skip days with fewer bars (incomplete sessions)


# ---------------------------------------------------------------------------
# Helpers – funding / data loading (mirrors backtest_pairwise_equity_corr_risk_compare)
# ---------------------------------------------------------------------------

def _load_funding_csv(pair: str) -> pd.DataFrame:
    path = gp.DATA_DIR / f"{pair}_funding_{FUNDING_RANGE_START}_{FUNDING_RANGE_END}.csv"
    if not path.exists():
        candidates = sorted(gp.DATA_DIR.glob(f"{pair}_funding_{FUNDING_RANGE_START}_*.csv"))
        if not candidates:
            return pd.DataFrame(columns=["fundingTime", "fundingRate"])
        path = candidates[-1]
    df = pd.read_csv(path)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, format="mixed")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df.dropna(subset=["fundingTime", "fundingRate"]).sort_values("fundingTime").reset_index(drop=True)


def load_funding_cache(pair: str) -> pd.DataFrame:
    range_start = pd.Timestamp(FUNDING_RANGE_START, tz=UTC)
    range_end = pd.Timestamp(FUNDING_RANGE_END, tz=UTC) + pd.Timedelta(days=1)
    pg_frame = gp.load_funding_rates(pair, FUNDING_RANGE_START, FUNDING_RANGE_END)
    csv_frame = _load_funding_csv(pair)
    frames = [f for f in (pg_frame, csv_frame) if not f.empty]
    if not frames:
        raise RuntimeError(f"No funding data available for {pair}")
    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["fundingTime"], keep="first")
        .sort_values("fundingTime")
        .reset_index(drop=True)
    )
    actual_end = merged["fundingTime"].max()
    expected_min_count = max(1, int((range_end - range_start).total_seconds() // (8 * 3600)) - 2)
    if pd.isna(actual_end) or actual_end < range_end - pd.Timedelta(days=1):
        raise RuntimeError(
            f"Funding coverage for {pair} ends at {actual_end} but window requires "
            f">= {range_end - pd.Timedelta(days=1)}; refresh CSV/DB before running."
        )
    if len(merged) < expected_min_count:
        raise RuntimeError(
            f"Funding rows for {pair}: {len(merged)} < expected ~{expected_min_count} "
            f"for window {FUNDING_RANGE_START}..{FUNDING_RANGE_END}."
        )
    return merged


def filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def filter_funding_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1)
    return df.loc[(df["fundingTime"] >= start_ts) & (df["fundingTime"] < end_ts)].copy()


# ---------------------------------------------------------------------------
# Tail risk statistics
# ---------------------------------------------------------------------------

def bar_net_to_daily_returns(
    bar_net: np.ndarray,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    """Compound 5-min bar_net returns within each calendar day.

    Days with fewer than MIN_BARS_PER_DAY bars are dropped.
    Returns an array of daily returns (as fractions, e.g. -0.012 = -1.2%).
    """
    if len(bar_net) == 0:
        return np.array([], dtype="float64")

    dates = index.normalize()  # UTC date at midnight
    unique_dates = pd.DatetimeIndex(np.unique(dates))
    daily_returns: list[float] = []

    for date in unique_dates:
        mask = dates == date
        bars = bar_net[mask]
        if len(bars) < MIN_BARS_PER_DAY:
            continue
        # compound: (1 + r1)(1 + r2)... - 1
        daily_ret = float(np.prod(1.0 + bars) - 1.0)
        daily_returns.append(daily_ret)

    return np.array(daily_returns, dtype="float64")


def hill_tail_index(returns: np.ndarray, k: int) -> float:
    """Hill estimator for the lower tail index alpha.

    Uses the k most extreme losses. Alpha < 2 → extreme fat tail (infinite variance).
    Alpha 2-3 → fat, 3-4 → moderate, >4 → thin.

    Returns np.nan if k < 2.
    """
    if k < 2 or len(returns) < k + 1:
        return float("nan")
    # Work on losses (positive values for negative returns)
    losses = -np.sort(returns)  # sorted ascending; losses are positive for negative returns
    # We want the k largest losses; sort descending
    losses_sorted = np.sort(losses)[::-1]  # largest first
    # Hill estimator: 1/alpha = (1/k) * sum(log(X_i / X_{k+1})) for i=1..k
    threshold = losses_sorted[k]  # (k+1)-th largest (0-indexed: index k)
    if threshold <= 0:
        return float("nan")
    log_ratios = np.log(losses_sorted[:k] / threshold)
    mean_log = float(np.mean(log_ratios))
    if mean_log <= 0:
        return float("nan")
    return float(1.0 / mean_log)


def rolling_max_drawdown(daily_returns: np.ndarray, window_days: int) -> float:
    """Maximum cumulative loss over any rolling window of `window_days` consecutive days.

    For window_days=1 returns the worst single-day return directly (avoids
    the trivial equity[i]/equity[i]-1=0 identity).
    """
    if len(daily_returns) < window_days:
        return float("nan")
    if window_days == 1:
        return float(np.min(daily_returns))
    # Build cumulative equity
    equity = np.cumprod(1.0 + daily_returns)
    worst = 0.0
    for i in range(len(equity) - window_days + 1):
        segment = equity[i: i + window_days]
        # drawdown within segment: end/start - 1
        seg_dd = segment[-1] / segment[0] - 1.0
        if seg_dd < worst:
            worst = seg_dd
    return float(worst)


def actual_mdd(daily_returns: np.ndarray) -> float:
    """Peak-to-trough max drawdown over the full series (negative fraction)."""
    if len(daily_returns) == 0:
        return float("nan")
    equity = np.cumprod(1.0 + daily_returns)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.min(dd))


def gaussian_implied_mdd(mean_d: float, std_d: float) -> float:
    """Gaussian MDD proxy over a 30-day horizon: mu_30d - 4*sigma_30d.

    Projects drift and volatility to a 30-day window (a typical drawdown
    horizon) and takes the 4-sigma downside. Always returns a negative number;
    if drift is implausibly large we use the bare -4*sigma_30d floor.

    tail_fatness_ratio = |actual_mdd| / |gaussian_implied_mdd|;
    > 1 means actual MDD was worse than Gaussian 4-sigma 30-day prediction.
    """
    horizon = 30
    mu_h = mean_d * horizon
    sigma_h = std_d * math.sqrt(horizon)
    implied = mu_h - 4.0 * sigma_h
    if implied >= 0.0:
        implied = -4.0 * sigma_h  # conservative floor
    return float(implied)


def verdict_from_hill(alpha: float) -> str:
    if math.isnan(alpha):
        return "unknown"
    if alpha < 2.0:
        return "extreme_fat"
    if alpha < 3.0:
        return "fat_tail"
    if alpha < 4.0:
        return "moderate"
    return "thin_tail"


def compute_tail_stats(daily_returns: np.ndarray) -> dict[str, Any]:
    """Compute the full tail risk stat block for one series of daily returns."""
    n = len(daily_returns)
    if n < 20:
        return {
            "n_days": n,
            "error": f"Insufficient data: {n} days (need >= 20)",
        }

    mean_d = float(np.mean(daily_returns))
    std_d = float(np.std(daily_returns, ddof=1))
    # Skewness and excess kurtosis (pure numpy, matches scipy fisher=True)
    diffs = daily_returns - mean_d
    m2 = float(np.mean(diffs ** 2))
    m3 = float(np.mean(diffs ** 3))
    m4 = float(np.mean(diffs ** 4))
    skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    kurt_excess = m4 / (m2 ** 2) - 3.0 if m2 > 0 else 0.0

    # VaR / CVaR (empirical, losses are negative returns)
    var_95 = float(np.percentile(daily_returns, 5))   # 5th percentile = 95% VaR
    var_99 = float(np.percentile(daily_returns, 1))   # 1st percentile = 99% VaR
    cvar_95 = float(np.mean(daily_returns[daily_returns <= var_95]))
    cvar_99 = float(np.mean(daily_returns[daily_returns <= var_99]))

    # Hill estimator
    k = math.ceil(0.05 * n)
    alpha_hill = hill_tail_index(daily_returns, k)

    # Rolling drawdowns
    dd_1d = rolling_max_drawdown(daily_returns, 1)
    dd_7d = rolling_max_drawdown(daily_returns, 7)
    dd_30d = rolling_max_drawdown(daily_returns, 30)

    # Gaussian MDD vs actual MDD
    g_mdd = gaussian_implied_mdd(mean_d, std_d)
    a_mdd = actual_mdd(daily_returns)
    # tail fatness ratio: |actual_mdd| / |gaussian_implied_mdd|
    # ratio > 1 means the actual loss was worse than Gaussian prediction (fatter tails)
    if g_mdd != 0.0 and not math.isnan(g_mdd) and not math.isnan(a_mdd):
        tfr = float(abs(a_mdd) / abs(g_mdd))
    else:
        tfr = float("nan")

    return {
        "n_days": n,
        "mean_daily": mean_d,
        "std_daily": std_d,
        "skew": skew,
        "kurtosis_excess": kurt_excess,
        "VaR_95": var_95,
        "VaR_99": var_99,
        "CVaR_95": cvar_95,
        "CVaR_99": cvar_99,
        "tail_index_hill": alpha_hill if not math.isnan(alpha_hill) else None,
        "tail_index_k": k,
        "rolling_dd_1d": dd_1d,
        "rolling_dd_7d": dd_7d,
        "rolling_dd_30d": dd_30d,
        "gaussian_implied_mdd": g_mdd,
        "actual_mdd": a_mdd,
        "tail_fatness_ratio": tfr if not math.isnan(tfr) else None,
    }


# ---------------------------------------------------------------------------
# Main replay + analysis
# ---------------------------------------------------------------------------

def run_pair_analysis(
    pair: str,
    df_window: pd.DataFrame,
    all_pairs: tuple[str, ...],
    compiled_model: Any,
    overlay_library: list[Any],
    funding_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run realistic_overlay_replay with return_trace=True and extract daily returns."""
    raw_signal = pd.Series(
        compiled_model(*gp.get_feature_arrays(df_window, pair)),
        index=df_window.index,
        dtype="float64",
    )
    overlay_inputs = build_overlay_inputs(df_window, all_pairs, regime_pair=pair)
    library_lookup = build_library_lookup(overlay_library)
    mapping = tuple(int(v) for v in config[pair]["mapping_indices"])
    route_breadth_threshold = float(config[pair]["route_breadth_threshold"])

    context = build_fast_context(
        df=df_window,
        pair=pair,
        raw_signal=raw_signal,
        overlay_inputs=overlay_inputs,
        route_thresholds=(route_breadth_threshold,),
        library_lookup=library_lookup,
        funding_df=funding_df,
        route_state_mode=ROUTE_STATE_MODE_BASE,
    )

    result = realistic_overlay_replay_from_context(
        context,
        library_lookup,
        mapping,
        route_breadth_threshold,
        use_equity_corr_risk=False,
        return_trace=True,
    )

    if not isinstance(result, dict) or "trace" not in result:
        return {"error": "return_trace did not produce a trace dict"}

    bar_net = result["trace"]["bar_net"]
    if len(bar_net) == 0:
        return {"error": "Empty bar_net trace"}

    # The kernel may skip the first 1–2 bars; align by taking the tail of the index
    trace_index = df_window.index[-len(bar_net):]
    daily_returns = bar_net_to_daily_returns(bar_net, trace_index)
    return compute_tail_stats(daily_returns)


def overall_verdict(per_pair: dict[str, dict[str, Any]]) -> str:
    """Worst-case verdict across both pairs based on Hill alpha."""
    alphas = []
    for stats in per_pair.values():
        a = stats.get("tail_index_hill")
        if a is not None and not (isinstance(a, float) and math.isnan(a)):
            alphas.append(float(a))
    if not alphas:
        return "unknown"
    worst_alpha = min(alphas)
    return verdict_from_hill(worst_alpha)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def print_pair_summary(pair: str, stats: dict[str, Any]) -> None:
    if "error" in stats:
        print(f"  {pair}: ERROR – {stats['error']}")
        return
    alpha = stats.get("tail_index_hill")
    alpha_str = f"{alpha:.2f}" if alpha is not None else "n/a"
    v = verdict_from_hill(alpha if alpha is not None else float("nan"))
    tfr = stats.get("tail_fatness_ratio")
    tfr_str = f"{tfr:.2f}x" if tfr is not None else "n/a"
    print(
        f"  {pair}: "
        f"VaR-95={stats['VaR_95']*100:.2f}%  "
        f"CVaR-95={stats['CVaR_95']*100:.2f}%  "
        f"VaR-99={stats['VaR_99']*100:.2f}%  "
        f"CVaR-99={stats['CVaR_99']*100:.2f}%  "
        f"skew={stats['skew']:.2f}  kurt={stats['kurtosis_excess']:.2f}  "
        f"Hill α={alpha_str} → {v}  "
        f"MDD={stats['actual_mdd']*100:.2f}%  gauss_MDD={stats['gaussian_implied_mdd']*100:.2f}%  "
        f"tail_fatness={tfr_str}  "
        f"roll_dd_1d={stats['rolling_dd_1d']*100:.2f}%  "
        f"roll_dd_7d={stats['rolling_dd_7d']*100:.2f}%  "
        f"roll_dd_30d={stats['rolling_dd_30d']*100:.2f}%  "
        f"n_days={stats['n_days']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tail risk analysis for pairwise strategy.")
    parser.add_argument(
        "--window",
        default="full_4y",
        choices=[w[0] for w in DEFAULT_WINDOWS],
        help="Backtest window label (default: full_4y)",
    )
    parser.add_argument(
        "--pairs",
        default=",".join(PAIRS),
        help="Comma-separated pair list (default: BTCUSDT,BNBUSDT)",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=gp.MODELS_DIR / "tail_risk_report.json",
    )
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs: tuple[str, ...] = tuple(p.strip() for p in args.pairs.split(",") if p.strip())

    # Resolve window dates
    window_map = {w[0]: (w[1], w[2]) for w in DEFAULT_WINDOWS}
    if args.window not in window_map:
        raise ValueError(f"Unknown window '{args.window}'. Choose from: {list(window_map)}")
    win_start, win_end = window_map[args.window]

    print(f"Loading summary: {args.summary_path}")
    summary = json.loads(args.summary_path.read_text())
    config = summary["selected_candidate"]["pair_configs"]

    print(f"Loading model: {args.model_path}")
    model_tree, _ = load_signal_model(args.model_path)
    compiled = gp.toolbox.compile(expr=model_tree)

    overlay_library = list(iter_params())

    print(f"Loading OHLCV data for window {args.window} [{win_start} → {win_end}] pairs={pairs}")
    df_all = gp.load_all_pairs(pairs=list(pairs), start=win_start, end=win_end, refresh_cache=False)
    df_window = filter_window(df_all, win_start, win_end)

    print("Loading funding rate caches...")
    funding_cache: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        funding_cache[pair] = load_funding_cache(pair)

    per_pair: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        print(f"\nRunning replay + tail analysis for {pair}...")
        funding_df = filter_funding_window(funding_cache[pair], win_start, win_end)
        stats = run_pair_analysis(
            pair=pair,
            df_window=df_window,
            all_pairs=pairs,
            compiled_model=compiled,
            overlay_library=overlay_library,
            funding_df=funding_df,
            config=config,
        )
        per_pair[pair] = stats

    verdict = overall_verdict(per_pair)

    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "window": args.window,
        "window_start": win_start,
        "window_end": win_end,
        "summary_path": str(args.summary_path),
        "model_path": str(args.model_path),
        "per_pair": per_pair,
        "verdict": verdict,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(json_safe(report), indent=2))

    print("\n" + "=" * 80)
    print(f"TAIL RISK ANALYSIS  window={args.window}  verdict={verdict.upper()}")
    print("=" * 80)
    for pair in pairs:
        print_pair_summary(pair, per_pair[pair])
    print(f"\nReport written to: {args.report_out}")


if __name__ == "__main__":
    main()
