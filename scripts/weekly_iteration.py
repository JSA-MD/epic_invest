#!/usr/bin/env python3
"""Weekly iteration — Prompt 7 (PDCA Act).

Reads the last 7 daily_diagnosis_*.json reports and produces a single
weekly decision document with one — only one — hypothesis for next
week's experiment. Forcing scope to a single hypothesis is the whole
point: it is the discipline that prevents the "one more patch" pattern
Taleb calls naive interventionism.

Output: `models/weekly_iteration_<UTC>.json` with fields

    week_summary             — averages over the 7 days
    what_worked              — at most 2 bullet points
    what_broke               — at most 2 bullet points
    next_hypothesis          — one sentence
    next_action              — one of {prompt_2_edge_discovery,
                                       prompt_4_live_wiring,
                                       prompt_5_sizing_optimization,
                                       hold}
    measurement_metric       — concrete metric the next iteration must move
    success_threshold        — explicit numeric goal for that metric
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("weekly_iteration")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


def _load_recent_diagnoses(models_dir: Path, days: int = 7) -> list[dict]:
    out: list[dict] = []
    for path in sorted(models_dir.glob("daily_diagnosis_*.json"))[-days:]:
        try:
            out.append(json.loads(path.read_text()))
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", path, exc)
    return out


def _summarise(diagnoses: list[dict]) -> dict[str, Any]:
    if not diagnoses:
        return {"n_days": 0}
    pcts = [(d.get("pnl") or {}).get("pct_of_base") or 0.0 for d in diagnoses]
    gaps = [d.get("gap_to_target_bps") or 0.0 for d in diagnoses]
    actions = [d.get("recommended_action") for d in diagnoses]
    suppressed_total = sum(((d.get("signal_decomposition") or {}).get("silently_dropped_signals") or 0) for d in diagnoses)
    flat_total = sum(((d.get("signal_decomposition") or {}).get("n_flat") or 0) for d in diagnoses)
    bars_total = sum(((d.get("signal_decomposition") or {}).get("n_bars") or 0) for d in diagnoses)
    return {
        "n_days": len(diagnoses),
        "mean_daily_pct": mean(pcts) if pcts else 0.0,
        "stdev_daily_pct": stdev(pcts) if len(pcts) > 1 else 0.0,
        "mean_gap_bps": mean(gaps) if gaps else 0.0,
        "actions_count": {a: actions.count(a) for a in set(actions) if a},
        "suppressed_total": suppressed_total,
        "flat_total": flat_total,
        "bars_total": bars_total,
        "flat_rate": (flat_total / bars_total) if bars_total else 0.0,
        "suppression_rate": (suppressed_total / bars_total) if bars_total else 0.0,
    }


def _hypothesise(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("n_days", 0) == 0:
        return {
            "next_hypothesis": "no diagnoses yet — run daily_diagnosis.py for 7 days first",
            "next_action": "hold",
            "measurement_metric": "n_days",
            "success_threshold": "n_days >= 7",
        }
    flat = summary.get("flat_rate", 0.0)
    suppression = summary.get("suppression_rate", 0.0)
    mean_pct = summary.get("mean_daily_pct", 0.0)
    mean_gap = summary.get("mean_gap_bps", 0.0)

    if suppression > 0.5:
        return {
            "next_hypothesis": (
                f"Wire adaptive_threshold + sign_mismatch_monitor via "
                f"PAIRWISE_LIVE_OVERLAYS=1 to attribute or eliminate the "
                f"{suppression:.0%} silent suppression rate"
            ),
            "next_action": "prompt_4_live_wiring",
            "measurement_metric": "suppression_rate",
            "success_threshold": "< 0.20 (vs current %.2f)" % suppression,
        }
    if mean_pct < -0.05:
        return {
            "next_hypothesis": (
                "Live PnL is structurally negative; recertify candidates with CPCV/PBO/DSR "
                "before any further sizing changes"
            ),
            "next_action": "prompt_2_edge_discovery",
            "measurement_metric": "mean_daily_pct",
            "success_threshold": "> 0.0 (vs current %.3f)" % mean_pct,
        }
    if flat > 0.7:
        return {
            "next_hypothesis": (
                f"{flat:.0%} of bars are flat — too restrictive. Consider lowering "
                f"PAIRWISE_SIGN_INSTABILITY_THRESHOLD from 0.30 to 0.40 "
                "after one apples-to-apples backtest"
            ),
            "next_action": "prompt_5_sizing_optimization",
            "measurement_metric": "flat_rate",
            "success_threshold": "< 0.50 (vs current %.2f)" % flat,
        }
    if mean_pct >= 0.05:
        return {
            "next_hypothesis": (
                f"Stage A target ({mean_pct:.2f}% avg) reached. Recertify and consider "
                "Stage A→B promotion via 2-step approval"
            ),
            "next_action": "prompt_3_stage_promotion",
            "measurement_metric": "consecutive_positive_days",
            "success_threshold": ">= 30",
        }
    return {
        "next_hypothesis": "system in steady state but below 1% target; gather another week of data before changing anything",
        "next_action": "hold",
        "measurement_metric": "mean_gap_bps",
        "success_threshold": "< %.0f (vs current %.0f)" % (mean_gap * 0.8, mean_gap),
    }


def iterate(models_dir: Path = ROOT / "models", days: int = 7) -> dict[str, Any]:
    diagnoses = _load_recent_diagnoses(models_dir, days=days)
    summary = _summarise(diagnoses)
    hypothesis = _hypothesise(summary)

    what_worked: list[str] = []
    what_broke: list[str] = []
    if summary.get("mean_daily_pct", -1) > 0:
        what_worked.append(f"avg daily PnL {summary['mean_daily_pct']:.3f}% (positive)")
    if summary.get("flat_rate", 1) < 0.5:
        what_worked.append(f"flat rate {summary['flat_rate']:.0%} (below 50%)")
    if summary.get("suppression_rate", 0) > 0.3:
        what_broke.append(f"suppression rate {summary['suppression_rate']:.0%}")
    if summary.get("flat_rate", 0) > 0.7:
        what_broke.append(f"flat rate {summary['flat_rate']:.0%}")
    if not what_worked:
        what_worked.append("nothing notable")
    if not what_broke:
        what_broke.append("no major incidents")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "week_summary": summary,
        "what_worked": what_worked[:2],
        "what_broke": what_broke[:2],
        **hypothesis,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Weekly iteration — Prompt 7 (PDCA Act)")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--output-dir", type=Path, default=ROOT / "models")
    return p


def main() -> int:
    args = build_parser().parse_args()
    report = iterate(models_dir=args.models_dir, days=args.days)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir / f"weekly_iteration_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", out)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
