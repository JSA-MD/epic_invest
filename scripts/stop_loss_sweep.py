#!/usr/bin/env python3
"""Sweep stop-loss percent across multiple values and aggregate results per pair/window.

Runs scripts/backtest_pairwise_regime_stop_loss_compare.py for each pct value,
then aggregates into models/stop_loss_sweep_report.json with optimal pct selection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")
BACKTEST_SCRIPT = str(REPO_ROOT / "scripts" / "backtest_pairwise_regime_stop_loss_compare.py")
REPORT_OUT = REPO_ROOT / "models" / "stop_loss_sweep_report.json"

STOP_LOSS_PCTS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]
WINDOWS = ["recent_2m", "recent_6m", "full_4y"]
PAIRS = ["BTCUSDT", "BNBUSDT"]
MAX_WORKERS = 2  # 2-3 parallel runs to stay within memory budget


def run_single(pct: float, tmp_path: str) -> dict | None:
    """Run one backtest for the given stop-loss pct, return parsed JSON or None on failure."""
    pct_str = f"{pct:.4f}".rstrip("0").rstrip(".")
    env = {**os.environ, "EPIC_MARKET_DATA_SOURCE": "postgres"}
    cmd = [
        PYTHON,
        BACKTEST_SCRIPT,
        "--stop-loss-pct", str(pct),
        "--report-out", tmp_path,
    ]
    print(f"[sweep] Starting pct={pct_str} ...", flush=True)
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True,
            cwd=str(REPO_ROOT / "scripts"),
            timeout=600,
        )
        if result.returncode != 0:
            print(f"[sweep] ERROR: pct={pct_str} exited with code {result.returncode}", flush=True)
            return None
        data = json.loads(Path(tmp_path).read_text())
        print(f"[sweep] Done pct={pct_str}", flush=True)
        return data
    except subprocess.TimeoutExpired:
        print(f"[sweep] TIMEOUT: pct={pct_str} exceeded 600s", flush=True)
        return None
    except Exception as exc:
        print(f"[sweep] EXCEPTION: pct={pct_str}: {exc}", flush=True)
        return None


def extract_per_pair(report: dict, window_label: str) -> dict[str, dict]:
    """Extract per-pair metrics (baseline + stop) for a given window label."""
    windows = {w["label"]: w for w in report.get("windows", [])}
    if window_label not in windows:
        return {}
    w = windows[window_label]
    baseline_pp = w["baseline"]["per_pair"]
    stop_pp = w["stop_loss_2pct"]["per_pair"]
    result: dict[str, dict] = {}
    for pair in PAIRS:
        if pair not in baseline_pp:
            continue
        b = baseline_pp[pair]
        s = stop_pp.get(pair, {})
        result[pair] = {
            "baseline_roi": b.get("total_return"),
            "baseline_mdd": b.get("max_drawdown"),
            "baseline_sharpe": b.get("sharpe"),
            "baseline_avg_daily": b.get("avg_daily_return"),
            "stop_roi": s.get("total_return"),
            "stop_mdd": s.get("max_drawdown"),
            "stop_sharpe": s.get("sharpe"),
            "stop_avg_daily": s.get("avg_daily_return"),
            "stop_event_count": s.get("stop_event_count", 0),
            "delta_roi": (
                s.get("total_return", 0.0) - b.get("total_return", 0.0)
                if s.get("total_return") is not None and b.get("total_return") is not None
                else None
            ),
            "delta_mdd": (
                s.get("max_drawdown", 0.0) - b.get("max_drawdown", 0.0)
                if s.get("max_drawdown") is not None and b.get("max_drawdown") is not None
                else None
            ),
            "delta_sharpe": (
                s.get("sharpe", 0.0) - b.get("sharpe", 0.0)
                if s.get("sharpe") is not None and b.get("sharpe") is not None
                else None
            ),
        }
    return result


def find_optimal(per_pct: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """For each pair/window, find the pct that maximises delta_sharpe."""
    optimal: dict[str, dict[str, dict]] = {}
    for pair in PAIRS:
        optimal[pair] = {}
        for window in WINDOWS:
            best_pct = None
            best_delta_sharpe = float("-inf")
            for pct_str, window_data in per_pct.items():
                pp = window_data.get("per_pair", {}).get(pair, {})
                ds = pp.get("delta_sharpe")
                if ds is None:
                    continue
                if ds > best_delta_sharpe:
                    best_delta_sharpe = ds
                    best_pct = pct_str
            if best_pct is not None:
                optimal[pair][window] = {
                    "pct": best_pct,
                    "delta_sharpe": best_delta_sharpe,
                }
    return optimal


def build_summary(optimal: dict, per_pct: dict) -> str:
    lines = []
    lines.append("Optimal stop-loss pct per pair/window (maximize ΔSharpe):")
    for pair in PAIRS:
        parts = []
        for window in WINDOWS:
            o = optimal.get(pair, {}).get(window, {})
            if o:
                parts.append(f"{window}={o['pct']} (ΔSharpe={o['delta_sharpe']:+.4f})")
        lines.append(f"  {pair}: " + ", ".join(parts))

    # Best global config
    best_config = None
    best_ds = float("-inf")
    for pct_str, window_data in per_pct.items():
        for pair in PAIRS:
            for window in WINDOWS:
                ds = window_data.get("per_pair", {}).get(pair, {}).get("delta_sharpe")
                if ds is not None and ds > best_ds:
                    best_ds = ds
                    best_config = (pct_str, pair, window)
    if best_config:
        lines.append(f"Best global config: pct={best_config[0]}, pair={best_config[1]}, window={best_config[2]}, ΔSharpe={best_ds:+.4f}")

    # BTC full_4y summary (the key case from prior analysis)
    lines.append("BTC full_4y stop-loss impact:")
    for pct_str in sorted(per_pct.keys(), key=float):
        btc = per_pct[pct_str].get("per_pair", {}).get("BTCUSDT", {})
        ds = btc.get("delta_sharpe")
        dm = btc.get("delta_mdd")
        stops = btc.get("stop_event_count", 0)
        if ds is not None:
            lines.append(f"  pct={pct_str}: ΔSharpe={ds:+.4f}, ΔMDD={dm:+.4f}, stops={stops}")

    return " | ".join(lines[:2]) + "\n" + "\n".join(lines[2:])


def main() -> None:
    print(f"[sweep] Stop-loss sweep: {STOP_LOSS_PCTS}", flush=True)
    print(f"[sweep] Max workers: {MAX_WORKERS}", flush=True)

    # Map pct -> report dict
    results: dict[float, dict | None] = {}

    # Use temp files per run; run in parallel with MAX_WORKERS
    tmp_files: dict[float, str] = {}
    for pct in STOP_LOSS_PCTS:
        fd, path = tempfile.mkstemp(suffix=f"_stop_{pct}.json")
        os.close(fd)
        tmp_files[pct] = path

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(run_single, pct, tmp_files[pct]): pct
                for pct in STOP_LOSS_PCTS
            }
            for fut in as_completed(futures):
                pct = futures[fut]
                results[pct] = fut.result()
    finally:
        for path in tmp_files.values():
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    # Aggregate
    per_pct: dict[str, dict] = {}
    for pct in STOP_LOSS_PCTS:
        pct_str = str(pct)
        report = results.get(pct)
        if report is None:
            print(f"[sweep] WARNING: pct={pct} has no data (run failed), skipping", flush=True)
            continue
        window_data: dict[str, dict] = {}
        for window in WINDOWS:
            pp = extract_per_pair(report, window)
            window_data[window] = {"per_pair": pp}
        per_pct[pct_str] = window_data

    # Flatten per_pct for optimal search: {pct_str: {per_pair: {pair: metrics}}} across all windows
    # Need a flat view: per_pct_flat[pct_str]["per_pair"][pair] keyed by window
    # Build per-window view for optimal search
    per_pct_by_window: dict[str, dict] = {}  # {pct_str: {"per_pair": {pair: metrics}}} for each window
    for pct_str, wd in per_pct.items():
        for window in WINDOWS:
            key = f"{pct_str}:{window}"
            per_pct_by_window[key] = wd.get(window, {})

    # optimal: for each pair x window, find best pct
    optimal: dict[str, dict[str, dict]] = {}
    for pair in PAIRS:
        optimal[pair] = {}
        for window in WINDOWS:
            best_pct_str = None
            best_ds = float("-inf")
            for pct_str, wd in per_pct.items():
                pp = wd.get(window, {}).get("per_pair", {}).get(pair, {})
                ds = pp.get("delta_sharpe")
                if ds is None:
                    continue
                if ds > best_ds:
                    best_ds = ds
                    best_pct_str = pct_str
            if best_pct_str is not None:
                optimal[pair][window] = {
                    "pct": best_pct_str,
                    "delta_sharpe": best_ds,
                }

    # Build summary text using full_4y per pct per pair for the narrative
    full4y_per_pct: dict[str, dict] = {}
    for pct_str, wd in per_pct.items():
        full4y_per_pct[pct_str] = wd.get("full_4y", {})

    summary_text = build_summary(optimal, full4y_per_pct)

    report_out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stop_loss_pcts": STOP_LOSS_PCTS,
        "per_pct": per_pct,
        "optimal_per_pair_window": optimal,
        "summary": summary_text,
    }

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report_out, f, indent=2)

    print(f"\n[sweep] Report saved: {REPORT_OUT}", flush=True)
    print("\n" + "=" * 80)
    print(summary_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
