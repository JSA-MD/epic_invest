#!/usr/bin/env python3
"""Live-vs-backtest reconciliation: Taleb live-parity check.

Loads the live decision log, runs a backtest replay over the same 30-day window
using the same summary/model/config as the live service, then compares signal_pct,
target_weight, and route_state_name at each bar where live made a decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add scripts dir to path so local imports work.
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gp_crypto_evolution as gp
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
    route_state_names,
)

UTC = timezone.utc

# Paths
ROOT = Path(__file__).parent.parent
DEFAULT_DECISION_LOG = ROOT / "logs" / "pairwise_regime_decisions.jsonl"
DEFAULT_REPORT_OUT = ROOT / "models" / "live_vs_backtest_reconciliation.json"

# Funding CSV date range (matches backtest_pairwise_equity_corr_risk_compare.py)
FUNDING_RANGE_START = "2022-04-06"
FUNDING_RANGE_END = "2026-04-06"

# Divergence threshold for "high divergence" samples
HIGH_DIVERGENCE_THRESHOLD = 0.30  # 30% normalised magnitude diff
# Max samples to include in the high-divergence list per pair
MAX_HIGH_DIVERGENCE_SAMPLES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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


def load_funding(pair: str) -> pd.DataFrame:
    """Merge PostgreSQL and CSV funding data, tolerating partial DB coverage."""
    pg_frame = gp.load_funding_rates(pair, FUNDING_RANGE_START, FUNDING_RANGE_END)
    csv_frame = _load_funding_csv(pair)
    frames = [f for f in (pg_frame, csv_frame) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["fundingTime"], keep="first")
        .sort_values("fundingTime")
        .reset_index(drop=True)
    )


def filter_funding_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df.loc[(df["fundingTime"] >= start) & (df["fundingTime"] < end + pd.Timedelta(days=1))].copy()


# ---------------------------------------------------------------------------
# Decision log loading
# ---------------------------------------------------------------------------

def load_live_decisions(log_path: Path, cutoff: datetime) -> list[dict[str, Any]]:
    """Load JSONL records from the last `days` days that have signal_pct populated."""
    records: list[dict[str, Any]] = []
    with log_path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            at_str = rec.get("at", "")
            if not at_str:
                continue
            try:
                at = datetime.fromisoformat(at_str)
            except ValueError:
                continue
            if at < cutoff:
                continue
            # Only use records that have per-pair signal_pct (newer schema)
            plan = rec.get("plan") or {}
            pair_plans = plan.get("pair_plans") or {}
            has_signal_pct = any(
                pp.get("signal_pct") is not None
                for pp in pair_plans.values()
            )
            if not has_signal_pct:
                continue
            signal_ts_str = plan.get("signal_timestamp")
            if not signal_ts_str:
                continue
            try:
                signal_ts = datetime.fromisoformat(signal_ts_str)
            except ValueError:
                continue
            records.append({
                "at": at,
                "signal_ts": signal_ts,
                "mode": rec.get("mode", "unknown"),
                "summary_path": plan.get("summary_path", ""),
                "pair_plans": pair_plans,
            })
    return records


# ---------------------------------------------------------------------------
# Summary config loading (with caching)
# ---------------------------------------------------------------------------

_summary_cache: dict[str, dict[str, Any]] = {}


def load_summary_config(summary_path_str: str) -> dict[str, Any] | None:
    """Load pair_configs from a summary JSON, cached by path string."""
    if summary_path_str in _summary_cache:
        return _summary_cache[summary_path_str]
    p = Path(summary_path_str)
    if not p.exists():
        # Try the default if path doesn't exist
        p = DEFAULT_SUMMARY_PATH
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    cand = data.get("selected_candidate", data)
    pair_configs = cand.get("pair_configs", {})
    _summary_cache[summary_path_str] = pair_configs
    return pair_configs


# ---------------------------------------------------------------------------
# Backtest replay (per pair, with trace)
# ---------------------------------------------------------------------------

def run_backtest_with_trace(
    df: pd.DataFrame,
    pair: str,
    compiled_model,
    library: list,
    pair_config: dict[str, Any],
    funding_df: pd.DataFrame,
) -> dict[str, Any]:
    """Run a realistic overlay replay with trace=True, return result with trace."""
    raw_signal = pd.Series(
        compiled_model(*gp.get_feature_arrays(df, pair)),
        index=df.index,
        dtype="float64",
    )
    overlay_inputs = build_overlay_inputs(df, PAIRS, regime_pair=pair)
    route_state_mode = normalize_route_state_mode(pair_config.get("route_state_mode", "base"))
    route_threshold = float(pair_config["route_breadth_threshold"])
    mapping = normalize_mapping_indices(
        tuple(int(v) for v in pair_config["mapping_indices"]),
        route_state_mode,
    )
    library_lookup = build_library_lookup(library)
    context = build_fast_context(
        df=df,
        pair=pair,
        raw_signal=raw_signal,
        overlay_inputs=overlay_inputs,
        route_thresholds=(route_threshold,),
        library_lookup=library_lookup,
        funding_df=funding_df,
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
    return {
        "result": result,
        "context": context,
        "df_index": df.index,
        "route_state_mode": route_state_mode,
        "route_threshold": route_threshold,
    }


# ---------------------------------------------------------------------------
# Per-pair comparison
# ---------------------------------------------------------------------------

def compare_pair(
    pair: str,
    live_records: list[dict[str, Any]],
    backtest_data: dict[str, Any],
) -> dict[str, Any]:
    """Compare live decisions vs backtest trace for one pair."""
    result = backtest_data["result"]
    context = backtest_data["context"]
    df_index = backtest_data["df_index"]
    route_threshold = backtest_data["route_threshold"]
    route_state_mode = backtest_data["route_state_mode"]
    state_names = route_state_names(route_state_mode)

    trace = result.get("trace") or {}
    bt_signal_pct: np.ndarray = trace.get("signal_pct", np.array([]))
    bt_target_weight: np.ndarray = trace.get("target_weight", np.array([]))
    bt_bucket_codes: np.ndarray = context.get("bucket_codes", {}).get(float(route_threshold), np.array([]))

    # Build a mapping from timestamp -> bar index for fast lookup
    ts_to_idx: dict[pd.Timestamp, int] = {ts: i for i, ts in enumerate(df_index)}

    sign_mismatches: list[dict] = []
    magnitude_diffs: list[float] = []
    route_state_mismatches: list[dict] = []
    high_divergence_samples: list[dict] = []
    n_matched = 0
    n_not_found = 0

    for rec in live_records:
        pp = rec["pair_plans"].get(pair)
        if pp is None:
            continue
        live_signal_pct = pp.get("signal_pct")
        if live_signal_pct is None:
            continue

        live_target_weight = pp.get("target_weight", 0.0)
        live_route_state = pp.get("route_state_name", "")
        signal_ts = rec["signal_ts"]

        # Align: find the bar in df_index that matches signal_ts
        bar_idx = ts_to_idx.get(pd.Timestamp(signal_ts))
        if bar_idx is None:
            # Try floor to nearest bar
            ts_arr = df_index.get_indexer([pd.Timestamp(signal_ts)], method="nearest")
            if ts_arr[0] < 0 or ts_arr[0] >= len(bt_signal_pct):
                n_not_found += 1
                continue
            bar_idx = int(ts_arr[0])

        if bar_idx >= len(bt_signal_pct) or bar_idx < 0:
            n_not_found += 1
            continue

        n_matched += 1
        bt_sig = float(bt_signal_pct[bar_idx])
        bt_tw = float(bt_target_weight[bar_idx]) if bar_idx < len(bt_target_weight) else 0.0

        # Route state from bucket_codes
        bt_route_state = ""
        if bar_idx < len(bt_bucket_codes):
            bucket_code = int(bt_bucket_codes[bar_idx])
            if 0 <= bucket_code < len(state_names):
                bt_route_state = state_names[bucket_code]

        # Sign mismatch: live and backtest have opposite signs (ignoring zero)
        live_sign = 1 if float(live_target_weight) > 1e-6 else (-1 if float(live_target_weight) < -1e-6 else 0)
        bt_sign = 1 if bt_tw > 1e-6 else (-1 if bt_tw < -1e-6 else 0)
        sign_mismatch = (live_sign != 0 or bt_sign != 0) and (live_sign != bt_sign)

        # Magnitude diff normalised
        denom = max(abs(float(live_signal_pct)), abs(bt_sig), 0.01)
        mag_diff = abs(float(live_signal_pct) - bt_sig) / denom
        magnitude_diffs.append(mag_diff)

        # Route state mismatch
        route_mismatch = bool(live_route_state and bt_route_state and live_route_state != bt_route_state)

        sample = {
            "signal_ts": signal_ts.isoformat() if hasattr(signal_ts, "isoformat") else str(signal_ts),
            "bar_idx": bar_idx,
            "live_signal_pct": float(live_signal_pct),
            "bt_signal_pct": float(bt_sig),
            "live_target_weight": float(live_target_weight),
            "bt_target_weight": float(bt_tw),
            "live_route_state": live_route_state,
            "bt_route_state": bt_route_state,
            "sign_mismatch": sign_mismatch,
            "mag_diff": float(mag_diff),
            "route_state_mismatch": route_mismatch,
            "mode": rec.get("mode", ""),
        }

        if sign_mismatch:
            sign_mismatches.append(sample)
        if route_mismatch:
            route_state_mismatches.append(sample)
        if mag_diff >= HIGH_DIVERGENCE_THRESHOLD:
            high_divergence_samples.append(sample)

    n_live = n_matched + n_not_found
    n_bt_decisions = int(np.sum(np.abs(bt_target_weight) > 1e-6)) if len(bt_target_weight) > 0 else 0
    mean_diff = float(np.mean(magnitude_diffs)) if magnitude_diffs else 0.0
    p95_diff = float(np.percentile(magnitude_diffs, 95)) if magnitude_diffs else 0.0
    sign_mismatch_pct = float(len(sign_mismatches) / n_matched * 100.0) if n_matched > 0 else 0.0

    # Sort high-divergence by mag_diff desc, cap list
    high_divergence_samples.sort(key=lambda x: x["mag_diff"], reverse=True)
    high_divergence_samples = high_divergence_samples[:MAX_HIGH_DIVERGENCE_SAMPLES]

    return {
        "n_live_decisions": n_live,
        "n_live_matched_to_bar": n_matched,
        "n_live_not_found": n_not_found,
        "n_backtest_decisions": n_bt_decisions,
        "sign_mismatch_count": len(sign_mismatches),
        "sign_mismatch_pct": sign_mismatch_pct,
        "mean_abs_signal_diff_pct": float(mean_diff * 100.0),
        "p95_abs_signal_diff_pct": float(p95_diff * 100.0),
        "route_state_mismatch_count": len(route_state_mismatches),
        "samples_with_high_divergence": high_divergence_samples,
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def compute_verdict(per_pair: dict[str, dict[str, Any]]) -> str:
    all_sign_pct = [v["sign_mismatch_pct"] for v in per_pair.values() if v["n_live_matched_to_bar"] > 0]
    if not all_sign_pct:
        return "no_data"
    max_sign_pct = max(all_sign_pct)
    if max_sign_pct < 5.0:
        return "tight_match"
    if max_sign_pct < 15.0:
        return "moderate_drift"
    return "significant_drift"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare live decision log vs backtest replay (Taleb live-parity check)."
    )
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default 30).")
    parser.add_argument(
        "--decision-log",
        type=Path,
        default=DEFAULT_DECISION_LOG,
        help="Path to live decision JSONL log.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=DEFAULT_REPORT_OUT,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Override summary/config path for backtest (default: live service default).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Override GP model path (default: live service default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path: Path = args.decision_log
    report_out: Path = args.report_out

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=args.days)
    window_start = cutoff.isoformat()
    window_end = now.isoformat()

    # --- Stub report if no log exists or is empty ---
    if not log_path.exists() or log_path.stat().st_size == 0:
        stub = {
            "generated_at": iso_now(),
            "status": "no_live_data",
            "note": f"Decision log not found or empty: {log_path}",
            "window": {"start": window_start, "end": window_end},
        }
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(stub, indent=2))
        print(f"[reconciliation] No live data found. Stub report written to {report_out}")
        return

    # --- Load live decisions ---
    print(f"[reconciliation] Loading live decisions from {log_path} (last {args.days} days)...")
    live_records = load_live_decisions(log_path, cutoff)
    if not live_records:
        stub = {
            "generated_at": iso_now(),
            "status": "no_live_data",
            "note": (
                f"No records with signal_pct in the last {args.days} days "
                f"(cutoff: {cutoff.isoformat()})"
            ),
            "window": {"start": window_start, "end": window_end},
        }
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(stub, indent=2))
        print(f"[reconciliation] No qualifying live decisions. Stub report written to {report_out}")
        return

    print(f"[reconciliation] Found {len(live_records)} live decisions with signal_pct in window.")

    # --- Load model ---
    model_path: Path = args.model_path
    print(f"[reconciliation] Loading GP model from {model_path}...")
    model_tree, _ = load_signal_model(model_path)
    compiled_model = gp.toolbox.compile(expr=model_tree)

    # --- Load overlay library ---
    library = list(iter_params())

    # --- Determine date range for data load ---
    # Use a slightly wider window for warmup (signals need history)
    data_start_ts = cutoff - timedelta(days=60)  # 60-day warmup
    data_start = data_start_ts.strftime("%Y-%m-%d")
    data_end = now.strftime("%Y-%m-%d")

    print(f"[reconciliation] Loading OHLCV data {data_start} -> {data_end}...")
    df_all = gp.load_all_pairs(pairs=list(PAIRS), start=data_start, end=data_end, refresh_cache=False)

    # Filter to exact window (keep warmup for context)
    window_start_ts = pd.Timestamp(cutoff)

    # --- Load funding data ---
    print("[reconciliation] Loading funding data...")
    funding_cache: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        fd = load_funding(pair)
        funding_cache[pair] = filter_funding_window(
            fd, window_start_ts, pd.Timestamp(now)
        )

    # --- Determine config for backtest ---
    # Use the summary path from CLI (default = live service default)
    summary_path_str = str(args.summary_path)
    pair_configs = load_summary_config(summary_path_str)
    if pair_configs is None:
        print(f"[reconciliation] ERROR: Cannot load summary config from {summary_path_str}")
        sys.exit(1)

    # --- Run backtest with trace for each pair ---
    print("[reconciliation] Running backtest replay with trace (engine=python)...")
    backtest_data: dict[str, Any] = {}
    for pair in PAIRS:
        if pair not in pair_configs:
            print(f"[reconciliation] WARNING: {pair} not in pair_configs, skipping.")
            continue
        cfg = pair_configs[pair]
        print(f"  {pair}: route_state_mode={cfg.get('route_state_mode','base')}, "
              f"route_breadth_threshold={cfg.get('route_breadth_threshold')}")
        backtest_data[pair] = run_backtest_with_trace(
            df=df_all,
            pair=pair,
            compiled_model=compiled_model,
            library=library,
            pair_config=cfg,
            funding_df=funding_cache.get(pair, pd.DataFrame()),
        )

    # --- Compare live vs backtest per pair ---
    print("[reconciliation] Comparing live decisions vs backtest trace...")
    per_pair: dict[str, dict[str, Any]] = {}
    for pair in PAIRS:
        if pair not in backtest_data:
            continue
        pair_recs = [r for r in live_records if pair in r["pair_plans"]]
        print(f"  {pair}: {len(pair_recs)} live decisions to compare...")
        per_pair[pair] = compare_pair(pair, pair_recs, backtest_data[pair])

    # --- Verdict ---
    verdict = compute_verdict(per_pair)

    # --- Build report ---
    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "window": {"start": window_start, "end": window_end},
        "days": args.days,
        "decision_log": str(log_path),
        "summary_path": summary_path_str,
        "model_path": str(model_path),
        "per_pair": per_pair,
        "verdict": verdict,
    }

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(json_safe(report), indent=2))

    # --- Print per-pair table ---
    print("\n" + "=" * 72)
    print("LIVE vs BACKTEST RECONCILIATION — PER-PAIR MISMATCH TABLE")
    print("=" * 72)
    print(f"{'Pair':<10} {'LiveDec':>8} {'Matched':>8} {'SignMis%':>9} "
          f"{'MeanDiff%':>10} {'P95Diff%':>9} {'RouteMis':>9}")
    print("-" * 72)
    for pair, pdata in per_pair.items():
        print(
            f"{pair:<10} "
            f"{pdata['n_live_decisions']:>8} "
            f"{pdata['n_live_matched_to_bar']:>8} "
            f"{pdata['sign_mismatch_pct']:>9.1f} "
            f"{pdata['mean_abs_signal_diff_pct']:>10.1f} "
            f"{pdata['p95_abs_signal_diff_pct']:>9.1f} "
            f"{pdata['route_state_mismatch_count']:>9}"
        )
    print("=" * 72)
    print(f"Verdict: {verdict.upper()}")
    print(f"Report written to: {report_out}")


if __name__ == "__main__":
    main()
