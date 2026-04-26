#!/usr/bin/env python3
"""Run pairwise realistic-overlay backtest for 5 trailing windows ending today.

Windows (in days from today): 2m=60, 4m=120, 6m=180, 1y=365, 4y=1460.
Pairs: BTCUSDT, BNBUSDT.
Reports: n_trades, n_wins, n_losses, roundtrip_win_rate, daily_win_rate,
ROI, MDD, Sharpe, profit on $100k notional, fees, open-position state.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
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
    build_overlay_inputs,
    realistic_overlay_replay,
)
from backtest_pairwise_equity_corr_risk_compare import load_funding_cache, filter_window, filter_funding_window

UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today UTC)")
    p.add_argument("--report-out", type=Path, default=ROOT / "models" / "backtest_today_windows.json")
    p.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    p.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    today = datetime.fromisoformat(args.end).date() if args.end else datetime.now(tz=UTC).date()
    end_str = today.isoformat()
    windows = [
        ("2m", today - timedelta(days=60), today),
        ("4m", today - timedelta(days=120), today),
        ("6m", today - timedelta(days=180), today),
        ("1y", today - timedelta(days=365), today),
        ("4y", today - timedelta(days=1460), today),
    ]
    print(f"Today: {today}")
    for label, start, end in windows:
        print(f"  {label}: {start} → {end}")

    summary = json.loads(args.summary_path.read_text())
    config = summary["selected_candidate"]["pair_configs"]
    library = list(iter_params())
    model_tree, _ = load_signal_model(args.model_path)
    compiled = gp.toolbox.compile(expr=model_tree)

    longest_start = (today - timedelta(days=1460)).isoformat()
    df_all = gp.load_all_pairs(pairs=list(PAIRS), start=longest_start, end=end_str, refresh_cache=False)
    funding_cache = {pair: load_funding_cache(pair) for pair in PAIRS}

    report: dict[str, Any] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "end_date": end_str,
        "summary_path": str(args.summary_path),
        "model_path": str(args.model_path),
        "windows": {},
    }

    for label, ws, we in windows:
        ws_str = ws.isoformat()
        we_str = we.isoformat()
        df_window = filter_window(df_all, ws_str, we_str)
        if df_window.empty or len(df_window) < 100:
            print(f"[{label}] skip: bars={len(df_window)}")
            continue
        pair_reports: dict[str, Any] = {}
        for pair in PAIRS:
            raw_signal = pd.Series(
                compiled(*gp.get_feature_arrays(df_window, pair)),
                index=df_window.index,
                dtype="float64",
            )
            overlay_inputs = build_overlay_inputs(df_window, PAIRS, regime_pair=pair)
            funding_df = filter_funding_window(funding_cache[pair], ws_str, we_str)
            result = realistic_overlay_replay(
                df_window,
                pair,
                raw_signal,
                overlay_inputs,
                funding_df,
                library,
                tuple(int(v) for v in config[pair]["mapping_indices"]),
                float(config[pair]["route_breadth_threshold"]),
                use_equity_corr_risk=False,
            )
            pair_reports[pair] = result
        report["windows"][label] = {
            "start": ws_str,
            "end": we_str,
            "bars": int(len(df_window)),
            "pairs": pair_reports,
        }
        print(f"[{label}] done: bars={len(df_window)} {[(p, pair_reports[p]['total_return']) for p in PAIRS]}")

    report["caveats"] = {
        "comparable_to_live_pnl": False,
        "daily_metric_basis": (
            "Each window reports total_return, n_trades, n_wins/n_losses, daily_win_rate,"
            " sharpe, max_drawdown, fee_paid, slippage_paid, funding_paid, final_equity."
            " A daily 'CAGR-equivalent' MUST NOT be inferred by (1+total_return)**(1/days)"
            " and compared to live realised daily P&L: live is an arithmetic per-day series"
            " over a different (typically shorter) window with its own entry/exit timing."
            " To compute a like-for-like live-vs-backtest gap, run a separate replay over"
            " the EXACT same calendar window as live (see scripts/live_actual_pnl_30d.py for"
            " the daily live series; replay the kernel over those same dates and compare"
            " day by day, not by aggregate CAGR)."
        ),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport: {args.report_out}")
    print(
        "\nNote: total_return per window is cumulative compound. Do NOT derive a"
        " daily-mean from it to compare against live realised P&L; run a same-window"
        " replay against scripts/live_actual_pnl_30d.py output for an apples-to-apples"
        " drift comparison."
    )


if __name__ == "__main__":
    main()
