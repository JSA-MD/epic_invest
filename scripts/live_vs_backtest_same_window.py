#!/usr/bin/env python3
"""Apples-to-apples live vs backtest comparison over the SAME calendar window.

Reads scripts/live_actual_pnl_30d.json (which has per-date per-pair live P&L)
and runs realistic_overlay_replay over the exact same date range, then aligns
day-by-day to compute per-day delta in basis points.

This is the CORRECT way to measure live-vs-backtest drift. Do NOT use the
aggregate CAGR-derived daily from backtest_today_windows.py for this purpose.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gp_crypto_evolution as gp
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    realistic_overlay_replay_from_context,
)
from backtest_pairwise_equity_corr_risk_compare import (
    load_funding_cache,
    filter_window,
    filter_funding_window,
)

UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--live-pnl-report", type=Path, default=ROOT / "models" / "live_actual_pnl_30d.json")
    p.add_argument("--report-out", type=Path, default=ROOT / "models" / "live_vs_backtest_same_window.json")
    p.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument(
        "--initial",
        type=float,
        default=None,
        help="notional for backtest sizing; defaults to live initial_equity_estimate from the live report",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.live_pnl_report.exists():
        print(f"Live P&L report missing: {args.live_pnl_report}", file=sys.stderr)
        sys.exit(2)
    live = json.loads(args.live_pnl_report.read_text())
    daily_rows = live.get("daily_pnl_live") or []
    if not daily_rows:
        print("Live P&L report has no daily_pnl_live rows", file=sys.stderr)
        sys.exit(2)

    live_initial = float(live.get("initial_equity_estimate") or 0.0)
    if args.initial is not None:
        notional = float(args.initial)
        if abs(notional - live_initial) > 1.0:
            print(
                f"WARN: --initial={notional:.2f} differs from live initial_equity_estimate={live_initial:.2f};"
                " dollar comparison will be biased.",
                file=sys.stderr,
            )
    elif live_initial > 0:
        notional = live_initial
        print(f"Using live initial_equity_estimate as backtest notional: ${notional:,.2f}")
    else:
        notional = 100000.0
        print(f"WARN: live initial_equity_estimate missing; defaulting to $100,000 — dollar comparison NOT meaningful", file=sys.stderr)
    args.initial = notional

    # Resolve common live date window
    dates = sorted({r["date"] for r in daily_rows if r.get("date")})
    start = dates[0]
    end = dates[-1]
    print(f"Live window: {start} → {end} ({len(dates)} dates)")

    # Aggregate live per (date, pair) into a {date: {pair: total}} structure
    live_by_date: dict[str, dict[str, float]] = defaultdict(dict)
    for r in daily_rows:
        live_by_date[r["date"]][r["pair"]] = float(r.get("total", 0.0))

    # Load model + library + summary
    summary = json.loads(args.summary_path.read_text())
    config = summary["selected_candidate"]["pair_configs"]
    library = list(iter_params())
    model_tree, _ = load_signal_model(args.model_path)
    compiled = gp.toolbox.compile(expr=model_tree)

    df_all = gp.load_all_pairs(pairs=list(PAIRS), start=start, end=end, refresh_cache=False)
    funding_cache = {pair: load_funding_cache(pair) for pair in PAIRS}

    df_window = filter_window(df_all, start, end)
    if df_window.empty:
        print("No backtest bars for window", file=sys.stderr)
        sys.exit(2)

    backtest_daily: dict[str, dict[str, float]] = {}
    library_lookup = build_library_lookup(library)
    for pair in PAIRS:
        raw_signal = pd.Series(
            compiled(*gp.get_feature_arrays(df_window, pair)),
            index=df_window.index,
            dtype="float64",
        )
        overlay_inputs = build_overlay_inputs(df_window, PAIRS, regime_pair=pair)
        funding_df = filter_funding_window(funding_cache[pair], start, end)
        rb_threshold = float(config[pair]["route_breadth_threshold"])
        context = build_fast_context(
            df=df_window,
            pair=pair,
            raw_signal=raw_signal,
            overlay_inputs=overlay_inputs,
            route_thresholds=(rb_threshold,),
            library_lookup=library_lookup,
            funding_df=funding_df,
        )
        result = realistic_overlay_replay_from_context(
            context,
            library_lookup,
            tuple(int(v) for v in config[pair]["mapping_indices"]),
            rb_threshold,
            use_equity_corr_risk=False,
            return_trace=True,
        )
        bar_net = result.get("trace", {}).get("bar_net")
        if bar_net is None or len(bar_net) == 0:
            print(f"  {pair}: no trace, skip", file=sys.stderr)
            continue
        bar_idx = df_window.index[-len(bar_net):]
        bar_returns = pd.Series(bar_net, index=bar_idx)
        # Compound bars to daily returns by UTC calendar date
        daily = bar_returns.groupby(bar_returns.index.date).apply(
            lambda x: float((1.0 + x).prod() - 1.0)
        )
        for date, ret in daily.items():
            date_str = date.isoformat()
            backtest_daily.setdefault(date_str, {})[pair] = float(ret) * args.initial
        print(f"  {pair}: {len(daily)} backtest dates; example: {daily.index[0]}={daily.iloc[0]*100:.2f}%, {daily.index[-1]}={daily.iloc[-1]*100:.2f}%")

    # Align live vs backtest per (date, pair). All percentages are pct-of-base;
    # dollars use the SAME notional for both sides so $ comparison is fair.
    rows: list[dict[str, Any]] = []
    common_dates = sorted(set(live_by_date.keys()) & set(backtest_daily.keys()))
    for date in common_dates:
        live_pairs = live_by_date.get(date, {})
        bt_pairs = backtest_daily.get(date, {})
        for pair in PAIRS:
            live_p = live_pairs.get(pair, 0.0)
            bt_p = bt_pairs.get(pair, 0.0)
            live_pct = live_p / args.initial * 100.0
            bt_pct = bt_p / args.initial * 100.0
            rows.append({
                "date": date,
                "pair": pair,
                "live_pnl_usd": live_p,
                "backtest_pnl_usd": bt_p,
                "live_pnl_pct_of_base": live_pct,
                "backtest_pnl_pct_of_base": bt_pct,
                "diff_usd": live_p - bt_p,
                "diff_pct_of_base": live_pct - bt_pct,
                "diff_bps_of_base": (live_p - bt_p) / args.initial * 1e4,
            })

    pair_day_diffs_bps = [abs(r["diff_bps_of_base"]) for r in rows]
    live_total_usd = sum(r["live_pnl_usd"] for r in rows)
    backtest_total_usd = sum(r["backtest_pnl_usd"] for r in rows)

    # Per-DATE aggregation (sum across pairs first, then take abs). This is
    # the "daily drift" the user actually feels in account equity — it does
    # NOT cancel BTC vs BNB signs within a single calendar day, and it does
    # NOT get diluted by zero-trade pair-days. Codex 16th-round fix.
    per_date_live: dict[str, float] = defaultdict(float)
    per_date_backtest: dict[str, float] = defaultdict(float)
    for r in rows:
        per_date_live[r["date"]] += r["live_pnl_usd"]
        per_date_backtest[r["date"]] += r["backtest_pnl_usd"]
    per_date_diffs_usd = [per_date_live[d] - per_date_backtest[d] for d in common_dates]
    per_date_abs_bps = [abs(diff_usd) / args.initial * 1e4 for diff_usd in per_date_diffs_usd]

    drift_summary = {
        "base_notional_usd": float(args.initial),
        "live_initial_equity_estimate_usd": live_initial,
        "n_dates_compared": len(common_dates),
        "n_pair_days": len(rows),
        # Pair-day basis (kept for backwards compatibility but flagged as the
        # weaker metric — diluted by per-date sign offsets)
        "pair_day_mean_abs_diff_bps": float(sum(pair_day_diffs_bps) / len(pair_day_diffs_bps)) if pair_day_diffs_bps else 0.0,
        "pair_day_max_abs_diff_bps": float(max(pair_day_diffs_bps)) if pair_day_diffs_bps else 0.0,
        "pair_day_count_gt_50bps": sum(1 for d in pair_day_diffs_bps if d > 50),
        # Per-date basis (PRIMARY metric for live drift)
        "per_date_mean_abs_diff_bps": float(sum(per_date_abs_bps) / len(per_date_abs_bps)) if per_date_abs_bps else 0.0,
        "per_date_max_abs_diff_bps": float(max(per_date_abs_bps)) if per_date_abs_bps else 0.0,
        "per_date_days_drift_gt_50bps": sum(1 for d in per_date_abs_bps if d > 50),
        "per_date_days_drift_gt_100bps": sum(1 for d in per_date_abs_bps if d > 100),
        # Totals
        "live_total_usd": live_total_usd,
        "backtest_total_usd": backtest_total_usd,
        "gap_total_usd": live_total_usd - backtest_total_usd,
        "live_total_pct_of_base": live_total_usd / args.initial * 100.0,
        "backtest_total_pct_of_base": backtest_total_usd / args.initial * 100.0,
        "gap_total_pct_of_base": (live_total_usd - backtest_total_usd) / args.initial * 100.0,
        "avg_daily_gap_bps": float(sum(per_date_diffs_usd)) / len(per_date_diffs_usd) / args.initial * 1e4 if per_date_diffs_usd else 0.0,
        "primary_metric_note": (
            "USE per_date_mean_abs_diff_bps as the daily drift figure."
            " pair_day_mean_abs_diff_bps is diluted by zero-trade pair-rows and by"
            " within-date sign offsets between BTC/BNB and will materially understate"
            " the drift the user observes in account equity."
        ),
    }

    out = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "window": {"start": start, "end": end, "n_dates": len(dates)},
        "initial_notional": args.initial,
        "method": "Apples-to-apples per-date comparison: live arithmetic daily P&L from logs vs backtest bar_net resampled to UTC calendar daily over the SAME window.",
        "drift_summary": drift_summary,
        "per_date_per_pair": rows,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nDrift summary: {json.dumps(drift_summary, indent=2)}")
    print(f"Report: {args.report_out}")


if __name__ == "__main__":
    main()
