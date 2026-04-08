#!/usr/bin/env python3
"""Run the subset-pair promotion pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
MODELS_DIR = ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run search, validation, and stress gates for subset-pair promotion.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument(
        "--search-summary-out",
        default=str(MODELS_DIR / "gp_regime_mixture_btc_bnb_search_summary.json"),
    )
    parser.add_argument(
        "--validation-report-out",
        default=str(MODELS_DIR / "gp_regime_mixture_btc_bnb_validation_report.json"),
    )
    parser.add_argument(
        "--stress-report-out",
        default=str(MODELS_DIR / "gp_regime_mixture_btc_bnb_stress_report.json"),
    )
    parser.add_argument(
        "--pipeline-report-out",
        default=str(MODELS_DIR / "gp_regime_mixture_btc_bnb_promotion_pipeline_report.json"),
    )
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def run_step(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    search_summary_path = Path(args.search_summary_out)
    validation_report_path = Path(args.validation_report_out)
    stress_report_path = Path(args.stress_report_out)
    pipeline_report_path = Path(args.pipeline_report_out)

    executed_steps: list[dict[str, Any]] = []
    python = sys.executable

    if not args.skip_search:
        cmd = [
            python,
            "-u",
            str(SCRIPTS_DIR / "search_pair_subset_regime_mixture.py"),
            "--pairs",
            args.pairs,
            "--summary-out",
            str(search_summary_path),
        ]
        run_step(cmd, ROOT)
        executed_steps.append({"step": "search", "status": "executed", "cmd": cmd})
    else:
        executed_steps.append({"step": "search", "status": "skipped"})

    if not args.skip_validation:
        cmd = [
            python,
            "-u",
            str(SCRIPTS_DIR / "validate_pair_subset_summary.py"),
            "--summary",
            str(search_summary_path),
            "--report-out",
            str(validation_report_path),
        ]
        run_step(cmd, ROOT)
        executed_steps.append({"step": "validation", "status": "executed", "cmd": cmd})
    else:
        executed_steps.append({"step": "validation", "status": "skipped"})

    if not args.skip_stress:
        cmd = [
            python,
            "-u",
            str(SCRIPTS_DIR / "run_pair_subset_stress_matrix.py"),
            "--summary",
            str(search_summary_path),
            "--report-out",
            str(stress_report_path),
        ]
        run_step(cmd, ROOT)
        executed_steps.append({"step": "stress", "status": "executed", "cmd": cmd})
    else:
        executed_steps.append({"step": "stress", "status": "skipped"})

    search_summary = json.loads(search_summary_path.read_text())
    validation_report = json.loads(validation_report_path.read_text())
    stress_report = json.loads(stress_report_path.read_text())

    selected_candidate = search_summary.get("selected_candidate")
    promotion_decision = stress_report.get("promotion_decision", {})
    pipeline_status = promotion_decision.get("status", "unknown")
    ready_for_merge = bool(promotion_decision.get("selected_candidate_ready_for_merge", False))

    report = {
        "pairs": search_summary["pairs"],
        "executed_steps": executed_steps,
        "artifacts": {
            "search_summary": str(search_summary_path),
            "validation_report": str(validation_report_path),
            "stress_report": str(stress_report_path),
        },
        "selected_candidate": None if selected_candidate is None else {
            "route_breadth_threshold": selected_candidate["route_breadth_threshold"],
            "mapping_indices": selected_candidate["mapping_indices"],
        },
        "selection": search_summary.get("selection"),
        "validation_profiles": validation_report.get("profiles"),
        "stress_profiles": stress_report.get("profiles"),
        "promotion_decision": promotion_decision,
        "ready_for_merge": ready_for_merge,
        "status": pipeline_status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    pipeline_report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
