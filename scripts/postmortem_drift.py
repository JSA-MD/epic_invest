#!/usr/bin/env python3
"""Postmortem on a single drift-spike day — Prompt 6.

Trigger: a daily_diagnosis with `drift.max_abs_diff_bps > 100` or any
single bar with `|live_pnl_pct - bt_pnl_pct| > 1%`. Run as

    python scripts/postmortem_drift.py --date 2026-04-25

The output classifies the day into one of the 5-cause taxonomy from
`docs/live_drift_diagnostic_20260426.md`:

    C1 router_suppression       — gate killed signal; live=0, bt=non-zero
    C2a sizing_oversize         — live tw >> bt tw same bar
    C2b open_carry              — position held into adverse swing
    C3 stale_price              — price feed froze, no intraday rebalance
    C4 regime_gate_normal       — gate worked as designed; not a bug

The single most useful output: a one-line `single_line_fix` string that
suggests the smallest code or env change that would have prevented the
incident, designed to be pasted directly into a PR title.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

log = logging.getLogger("postmortem_drift")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


def _classify(live_pct: float, bt_pct: float, has_open_carry: bool, stale_pct: float) -> tuple[str, str]:
    """Return (cause_code, one_line_fix)."""
    if stale_pct > 0.5:
        return (
            "C3_stale_price",
            "wire safety_guards.refresh_latest_prices_from_rest into D-pre on every cycle",
        )
    if abs(live_pct) > 0.5 and abs(bt_pct) < 0.1:
        return (
            "C1_router_suppression",
            "lower regime_threshold or wire adaptive_threshold via PAIRWISE_LIVE_OVERLAYS=1",
        )
    if abs(live_pct) > abs(bt_pct) * 5 and abs(live_pct) > 0.3:
        return (
            "C2a_sizing_oversize",
            "tighten PAIRWISE_LIVE_MAX_GROSS_CAP (Stage 0 lockdown=0.01); enforce backtest_overlay_enforcer",
        )
    if has_open_carry:
        return (
            "C2b_open_carry",
            "set PAIRWISE_MAX_HOLD_BARS=288 + RECON_FORCE_CLOSE_ON_MISMATCH=1",
        )
    if (live_pct - bt_pct) * (live_pct - bt_pct) < 0.01:
        return ("C4_regime_gate_normal", "no fix needed — gate behaved as designed")
    return ("UNKNOWN", "drift cause unclear; run /graphify path src=live_pnl tgt=bt_pnl")


def postmortem(
    *,
    date: str,
    live_pnl_path: Path,
    drift_path: Path,
    decision_log: Path,
) -> dict[str, Any]:
    pnl_src = json.loads(live_pnl_path.read_text()) if live_pnl_path.exists() else {}
    drift_src = json.loads(drift_path.read_text()) if drift_path.exists() else {}

    daily_rows = [r for r in (pnl_src.get("daily_pnl_live") or []) if r.get("date") == date]
    drift_rows = [r for r in (drift_src.get("per_date_per_pair") or []) if r.get("date") == date]

    base = float(pnl_src.get("initial_equity_estimate") or 0.0) or 1.0
    total_live = sum(float(r.get("total") or 0.0) for r in daily_rows)
    live_pct = (total_live / base) * 100.0

    bt_pct_per_pair: dict[str, float] = {}
    for r in drift_rows:
        bt_pct_per_pair[r.get("pair")] = float(r.get("backtest_pnl_pct_of_base") or 0.0)
    bt_pct_total = sum(bt_pct_per_pair.values())

    # Coarse staleness proxy from decision log: count consecutive duplicate prices
    stale_pct = 0.0
    if decision_log.exists():
        prices_seen: dict[str, list[float]] = {}
        with decision_log.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_raw = row.get("at") or ""
                if not str(ts_raw).startswith(date):
                    continue
                for pair, pp in (row.get("plan", {}).get("pair_plans") or {}).items():
                    p = pp.get("price")
                    if p is None:
                        continue
                    prices_seen.setdefault(pair, []).append(float(p))
        if prices_seen:
            ratios = []
            for pair, prices in prices_seen.items():
                if len(prices) < 5:
                    continue
                unique = len(set(prices))
                ratios.append(1.0 - unique / max(len(prices), 1))
            if ratios:
                stale_pct = max(ratios)

    has_open_carry = any(
        abs(float(r.get("live_pnl_usd") or 0.0)) > 1.0
        and abs(float(r.get("backtest_pnl_usd") or 0.0)) < 1e-6
        for r in drift_rows
    )

    cause, fix = _classify(live_pct, bt_pct_total, has_open_carry, stale_pct)

    return {
        "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "live_pct": live_pct,
        "backtest_pct": bt_pct_total,
        "diff_pct": live_pct - bt_pct_total,
        "stale_price_ratio": stale_pct,
        "has_open_carry": has_open_carry,
        "drift_rows": drift_rows,
        "cause_code": cause,
        "single_line_fix": fix,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Postmortem on a drift-spike day")
    p.add_argument("--date", required=True, help="UTC date YYYY-MM-DD")
    p.add_argument("--live-pnl", type=Path, default=ROOT / "models" / "live_actual_pnl_30d.json")
    p.add_argument("--drift", type=Path, default=ROOT / "models" / "live_vs_backtest_same_window.json")
    p.add_argument("--decision-log", type=Path, default=ROOT / "logs" / "pairwise_regime_decisions.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "models")
    return p


def main() -> int:
    args = build_parser().parse_args()
    report = postmortem(
        date=args.date,
        live_pnl_path=args.live_pnl,
        drift_path=args.drift,
        decision_log=args.decision_log,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"postmortem_{args.date}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", out)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
