#!/usr/bin/env python3
"""Stage promotion decision — Prompt 3.

Phased path to the 1%-day-target:

    Stage A  gross_cap=0.01,  daily target +0.10%, hold ≥ 30 trading days
    Stage B  gross_cap=0.05,  daily target +0.30%, hold ≥ 60
    Stage C  gross_cap=0.10,  daily target +0.50%, hold ≥ 90
    Stage D  gross_cap=0.20,  daily target +1.00%, hold ≥ 180

This script reads the last N daily_diagnosis_*.json reports plus
`candidate_recertification_report.json` and `apples_to_apples_report.json`
(when present) and emits one of:

    promote   — bump Stage A→B, B→C, etc. plus the env command to run
    hold      — keep current stage, document why
    rollback  — drop one stage; mostly drawdown / parity violations

Outputs `models/stage_promotion_<UTC>.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("stage_promotion")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


STAGE_SPEC: dict[str, dict[str, Any]] = {
    "A": {"gross_cap": 0.01, "daily_target_pct": 0.10, "min_days": 30, "max_dd_pct": 2.0},
    "B": {"gross_cap": 0.05, "daily_target_pct": 0.30, "min_days": 60, "max_dd_pct": 5.0},
    "C": {"gross_cap": 0.10, "daily_target_pct": 0.50, "min_days": 90, "max_dd_pct": 10.0},
    "D": {"gross_cap": 0.20, "daily_target_pct": 1.00, "min_days": 180, "max_dd_pct": 15.0},
}


def _load_diagnoses(models_dir: Path, days: int) -> list[dict]:
    out: list[dict] = []
    for path in sorted(models_dir.glob("daily_diagnosis_*.json"))[-days:]:
        try:
            out.append(json.loads(path.read_text()))
        except Exception:
            continue
    return out


def _running_drawdown(daily_pcts: list[float]) -> float:
    if not daily_pcts:
        return 0.0
    eq = 100.0
    peak = 100.0
    max_dd = 0.0
    for pct in daily_pcts:
        eq *= 1.0 + pct / 100.0
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100.0
        max_dd = max(max_dd, dd)
    return max_dd


def decide(
    *,
    current_stage: str,
    diagnoses: list[dict],
    recert_report: dict | None,
    parity_report: dict | None,
) -> dict[str, Any]:
    spec = STAGE_SPEC.get(current_stage)
    if spec is None:
        return {
            "decide": "hold",
            "reason": f"unknown current_stage {current_stage!r}",
        }
    stages = list(STAGE_SPEC.keys())
    next_stage = stages[stages.index(current_stage) + 1] if current_stage != stages[-1] else current_stage

    daily_pcts = [(d.get("pnl") or {}).get("pct_of_base") or 0.0 for d in diagnoses]
    n_days = len(daily_pcts)
    avg_pct = mean(daily_pcts) if daily_pcts else 0.0
    max_dd = _running_drawdown(daily_pcts)

    parity_holds = bool((parity_report or {}).get("parity_holds", False)) if parity_report else False
    recert_decision = (recert_report or {}).get("decision") or {}
    recert_ready = bool(recert_decision.get("ready_for_promotion", False))

    # rollback gates
    if max_dd > spec["max_dd_pct"]:
        prior = stages[max(0, stages.index(current_stage) - 1)]
        return {
            "decide": "rollback",
            "from_stage": current_stage,
            "to_stage": prior,
            "reason": f"max_dd {max_dd:.2f}% > stage cap {spec['max_dd_pct']}%",
            "n_days": n_days,
            "avg_daily_pct": avg_pct,
        }
    if parity_report and not parity_holds:
        return {
            "decide": "rollback",
            "from_stage": current_stage,
            "to_stage": stages[max(0, stages.index(current_stage) - 1)],
            "reason": "apples_to_apples parity violations",
            "n_days": n_days,
            "avg_daily_pct": avg_pct,
        }

    # promote gates
    if (
        n_days >= spec["min_days"]
        and avg_pct >= spec["daily_target_pct"]
        and recert_ready
        and (parity_report is None or parity_holds)
    ):
        return {
            "decide": "promote",
            "from_stage": current_stage,
            "to_stage": next_stage,
            "reason": (
                f"avg {avg_pct:.3f}% >= target {spec['daily_target_pct']}% over {n_days} days; "
                f"recert ready_for_promotion=true; parity {'ok' if parity_report is None else 'verified'}"
            ),
            "n_days": n_days,
            "avg_daily_pct": avg_pct,
            "max_dd_pct": max_dd,
            "next_env": {
                "PAIRWISE_GROSS_CAP": str(STAGE_SPEC[next_stage]["gross_cap"]),
                "PAIRWISE_LIVE_MAX_GROSS_CAP": str(STAGE_SPEC[next_stage]["gross_cap"]),
            },
            "approval_command": (
                "python -c \"from promotion_approval import request_approval; "
                "print(request_approval('Stage %s -> %s', requester='OPERATOR_A', "
                "ttl_minutes=60).token)\"" % (current_stage, next_stage)
            ),
        }

    # otherwise hold
    missing = []
    if n_days < spec["min_days"]:
        missing.append(f"days {n_days}/{spec['min_days']}")
    if avg_pct < spec["daily_target_pct"]:
        missing.append(f"avg {avg_pct:.3f}% < target {spec['daily_target_pct']}%")
    if not recert_ready:
        missing.append("recert not green")
    if parity_report is not None and not parity_holds:
        missing.append("parity violation")
    return {
        "decide": "hold",
        "current_stage": current_stage,
        "reason": "; ".join(missing) if missing else "checks passed but no promotion criterion satisfied",
        "n_days": n_days,
        "avg_daily_pct": avg_pct,
        "max_dd_pct": max_dd,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage promotion decision — Prompt 3")
    p.add_argument("--current-stage", required=True, choices=list(STAGE_SPEC.keys()))
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--recert-report", type=Path, default=ROOT / "models" / "candidate_recertification_report.json")
    p.add_argument("--parity-report", type=Path, default=ROOT / "models" / "apples_to_apples_report.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "models")
    return p


def main() -> int:
    args = build_parser().parse_args()
    diagnoses = _load_diagnoses(args.models_dir, days=args.days)
    recert = json.loads(args.recert_report.read_text()) if args.recert_report.exists() else None
    parity = json.loads(args.parity_report.read_text()) if args.parity_report.exists() else None
    decision = decide(
        current_stage=args.current_stage,
        diagnoses=diagnoses,
        recert_report=recert,
        parity_report=parity,
    )
    decision["generated_at"] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir / f"stage_promotion_{stamp}.json"
    out.write_text(json.dumps(decision, indent=2, default=str))
    log.info("Wrote %s", out)
    print(json.dumps(decision, indent=2, default=str))
    return 0 if decision["decide"] in {"promote", "hold"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
