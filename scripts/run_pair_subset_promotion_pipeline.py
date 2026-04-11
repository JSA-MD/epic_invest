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

PIPELINE_MODE_LEGACY_SHARED = "legacy_shared"
PIPELINE_MODE_PAIRWISE_MARKET_OS = "pairwise_market_os"
PIPELINE_MODE_PAIRWISE_MARKET_OS_FRACTAL = "pairwise_market_os_fractal"

PAIRWISE_DEFAULT_SEARCH_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_market_os_candidate_summary.json"
PAIRWISE_DEFAULT_VALIDATION_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_market_os_validation_report.json"
PAIRWISE_DEFAULT_STRESS_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_market_os_stress_report.json"
PAIRWISE_DEFAULT_PIPELINE_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_market_os_pipeline_report.json"
PAIRWISE_DEFAULT_CANDIDATE_SUMMARY_PATHS = (
    MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json",
    MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_summary.json",
    MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json",
    MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_nsga3_summary.json",
    MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_fullgrid_seed_pool.json",
)
PAIRWISE_DEFAULT_CANDIDATE_SUMMARIES = ",".join(str(path) for path in PAIRWISE_DEFAULT_CANDIDATE_SUMMARY_PATHS)
PAIRWISE_DEFAULT_BASELINE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_nsga3_summary.json"

LEGACY_DEFAULT_SEARCH_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_search_summary.json"
LEGACY_DEFAULT_VALIDATION_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_validation_report.json"
LEGACY_DEFAULT_STRESS_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_stress_report.json"
LEGACY_DEFAULT_PIPELINE_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_promotion_pipeline_report.json"

FRACTAL_DEFAULT_SEARCH_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_fractal_genome_market_os_pipeline_search_summary.json"
FRACTAL_DEFAULT_VALIDATION_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_fractal_genome_market_os_validation_report.json"
FRACTAL_DEFAULT_STRESS_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_fractal_genome_market_os_stress_report.json"
FRACTAL_DEFAULT_PIPELINE_REPORT = MODELS_DIR / "gp_regime_mixture_btc_bnb_fractal_genome_market_os_pipeline_report.json"
FRACTAL_DEFAULT_BASELINE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run search, validation, and stress gates for subset-pair promotion.",
    )
    parser.add_argument(
        "--pipeline-mode",
        default=PIPELINE_MODE_PAIRWISE_MARKET_OS,
        choices=(PIPELINE_MODE_LEGACY_SHARED, PIPELINE_MODE_PAIRWISE_MARKET_OS, PIPELINE_MODE_PAIRWISE_MARKET_OS_FRACTAL),
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument("--candidate-summaries", default=None)
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--search-summary-out", default=None)
    parser.add_argument("--validation-report-out", default=None)
    parser.add_argument("--stress-report-out", default=None)
    parser.add_argument("--pipeline-report-out", default=None)
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


def resolve_mode_defaults(mode: str) -> dict[str, Any]:
    if mode == PIPELINE_MODE_PAIRWISE_MARKET_OS:
        return {
            "search_summary_out": PAIRWISE_DEFAULT_SEARCH_SUMMARY,
            "validation_report_out": PAIRWISE_DEFAULT_VALIDATION_REPORT,
            "stress_report_out": PAIRWISE_DEFAULT_STRESS_REPORT,
            "pipeline_report_out": PAIRWISE_DEFAULT_PIPELINE_REPORT,
            "candidate_summaries": PAIRWISE_DEFAULT_CANDIDATE_SUMMARIES,
            "baseline_summary": PAIRWISE_DEFAULT_BASELINE_SUMMARY,
        }
    if mode == PIPELINE_MODE_PAIRWISE_MARKET_OS_FRACTAL:
        return {
            "search_summary_out": FRACTAL_DEFAULT_SEARCH_SUMMARY,
            "validation_report_out": FRACTAL_DEFAULT_VALIDATION_REPORT,
            "stress_report_out": FRACTAL_DEFAULT_STRESS_REPORT,
            "pipeline_report_out": FRACTAL_DEFAULT_PIPELINE_REPORT,
            "candidate_summaries": PAIRWISE_DEFAULT_CANDIDATE_SUMMARIES,
            "baseline_summary": FRACTAL_DEFAULT_BASELINE_SUMMARY,
        }
    return {
        "search_summary_out": LEGACY_DEFAULT_SEARCH_SUMMARY,
        "validation_report_out": LEGACY_DEFAULT_VALIDATION_REPORT,
        "stress_report_out": LEGACY_DEFAULT_STRESS_REPORT,
        "pipeline_report_out": LEGACY_DEFAULT_PIPELINE_REPORT,
        "candidate_summaries": LEGACY_DEFAULT_SEARCH_SUMMARY,
        "baseline_summary": LEGACY_DEFAULT_SEARCH_SUMMARY,
    }


def resolve_mode_paths(args: argparse.Namespace) -> dict[str, Any]:
    defaults = resolve_mode_defaults(str(args.pipeline_mode))
    return {
        "search_summary_out": Path(args.search_summary_out) if args.search_summary_out else defaults["search_summary_out"],
        "validation_report_out": Path(args.validation_report_out) if args.validation_report_out else defaults["validation_report_out"],
        "stress_report_out": Path(args.stress_report_out) if args.stress_report_out else defaults["stress_report_out"],
        "pipeline_report_out": Path(args.pipeline_report_out) if args.pipeline_report_out else defaults["pipeline_report_out"],
        "candidate_summaries": str(args.candidate_summaries) if args.candidate_summaries else str(defaults["candidate_summaries"]),
        "baseline_summary": Path(args.baseline_summary) if args.baseline_summary else defaults["baseline_summary"],
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def build_pairwise_validation_report(search_summary: dict[str, Any]) -> dict[str, Any]:
    selected = search_summary.get("selected_candidate") or {}
    validation = selected.get("validation") or {}
    validation_engine = selected.get("validation_engine") or {}
    market_operating_system = validation_engine.get("market_operating_system") or {}
    gate = validation_engine.get("gate") or {}
    market_os_gate = market_operating_system.get("gate") or {}
    market_os_audit = market_operating_system.get("audit") or {}
    final_oos = (validation.get("profiles") or {}).get("final_oos") or {}
    return {
        "pairs": search_summary.get("pairs"),
        "summary_path": search_summary.get("summary_path"),
        "selected_candidate": {
            "route_breadth_threshold": selected.get("route_breadth_threshold"),
            "mapping_indices": selected.get("mapping_indices"),
            "pair_configs": selected.get("pair_configs"),
            "candidate_id": selected.get("candidate_id"),
        },
        "validation_engine": validation_engine,
        "validation_gate": gate,
        "market_os_gate": market_os_gate,
        "market_os_audit": market_os_audit,
        "final_oos_audit": final_oos,
        "selection": search_summary.get("selection"),
    }


def build_fractal_validation_report(search_summary: dict[str, Any]) -> dict[str, Any]:
    selected = search_summary.get("selected_candidate") or {}
    validation = selected.get("validation") or {}
    profiles = validation.get("profiles") or {}
    robustness = selected.get("robustness") or {}
    return {
        "pairs": search_summary.get("pairs"),
        "summary_path": search_summary.get("summary_path"),
        "selected_candidate": {
            "candidate_kind": selected.get("candidate_kind", "fractal_tree"),
            "tree_key": selected.get("tree_key"),
            "observation_mode": selected.get("observation_mode"),
            "label_horizon": selected.get("label_horizon"),
            "tree_depth": selected.get("tree_depth"),
            "logic_depth": selected.get("logic_depth"),
        },
        "validation_engine": {
            "candidate_kind": selected.get("candidate_kind", "fractal_tree"),
            "observation_mode": selected.get("observation_mode"),
            "label_horizon": selected.get("label_horizon"),
            "tree_key": selected.get("tree_key"),
            "robustness": robustness,
        },
        "validation_gate": profiles.get("progressive_improvement", {}),
        "market_os_gate": profiles.get("target_060", {}),
        "market_os_audit": robustness,
        "final_oos_audit": profiles.get("final_oos", {}),
        "selection": search_summary.get("selection"),
    }


def build_fractal_stress_report(search_summary: dict[str, Any]) -> dict[str, Any]:
    selected = search_summary.get("selected_candidate") or {}
    robustness = selected.get("robustness") or {}
    validation = selected.get("validation") or {}
    profiles = validation.get("profiles") or {}
    wf1 = robustness.get("wf_1") or {}
    full_gate_passed = bool(robustness.get("gate_passed", False))
    shadow_live_ready = bool(wf1.get("passed", False))
    if full_gate_passed:
        status = "ready_for_merge"
    elif shadow_live_ready:
        status = "shadow_ready_only"
    else:
        status = "stress_fail"
    return {
        "selected_candidate": {
            "candidate_kind": selected.get("candidate_kind", "fractal_tree"),
            "tree_key": selected.get("tree_key"),
            "observation_mode": selected.get("observation_mode"),
            "label_horizon": selected.get("label_horizon"),
        },
        "robustness": robustness,
        "validation_profiles": profiles,
        "promotion_decision": {
            "status": status,
            "validation_gate": profiles.get("progressive_improvement", {}),
            "market_os_gate": profiles.get("target_060", {}),
            "final_oos_audit": profiles.get("final_oos", {}),
            "robustness_gate": {
                "passed": full_gate_passed,
                "wf_1_passed": shadow_live_ready,
                "stress_survival_rate_mean": robustness.get("stress_survival_rate_mean"),
                "stress_survival_rate_min": robustness.get("stress_survival_rate_min"),
                "stress_survival_threshold": robustness.get("stress_survival_threshold"),
                "latest_fold_stress_reserve_score": robustness.get("latest_fold_stress_reserve_score"),
            },
            "ready_for_merge": full_gate_passed,
            "ready_for_live": shadow_live_ready,
            "selected_candidate_ready_for_merge": full_gate_passed,
            "selected_candidate_ready_for_live": shadow_live_ready,
            "failed_checks": [] if full_gate_passed else ["robustness_gate"] if not shadow_live_ready else ["full_robustness_gate"],
        },
    }


def build_legacy_validation_report(validation_report: dict[str, Any]) -> dict[str, Any]:
    selected_profiles = (validation_report.get("profiles") or {})
    return {
        "pairs": validation_report.get("pairs"),
        "summary_path": validation_report.get("summary_path"),
        "selected_candidate": validation_report.get("selected_candidate"),
        "comparison": validation_report.get("comparison"),
        "validation_gate": selected_profiles.get("progressive_improvement", {}).get("selected", {}),
        "market_os_gate": None,
        "market_os_audit": None,
        "final_oos_audit": selected_profiles.get("final_oos", {}).get("selected", {}),
        "selection": validation_report.get("comparison"),
    }


def build_final_report(
    *,
    mode: str,
    search_summary: dict[str, Any],
    validation_report: dict[str, Any],
    stress_report: dict[str, Any],
    executed_steps: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    selected = search_summary.get("selected_candidate") or {}
    validation_engine = selected.get("validation_engine") or {}
    market_operating_system = validation_engine.get("market_operating_system") or {}
    validation_gate = validation_engine.get("gate") or validation_report.get("validation_gate") or {}
    market_os_gate = validation_report.get("market_os_gate")
    if market_os_gate is None:
        market_os_gate = market_operating_system.get("gate")
    final_oos_audit = validation_report.get("final_oos_audit") or {}
    if not final_oos_audit:
        final_oos_audit = (selected.get("validation") or {}).get("profiles", {}).get("final_oos", {})
    stress_decision = stress_report.get("promotion_decision") or {}
    stress_ready_for_merge = bool(
        stress_decision.get("ready_for_merge", stress_decision.get("selected_candidate_ready_for_merge", False))
    )
    stress_ready_for_shadow_live = bool(
        stress_decision.get(
            "ready_for_live",
            stress_decision.get("selected_candidate_ready_for_live", stress_ready_for_merge),
        )
    )
    stress_gate = {
        "passed": stress_ready_for_merge,
        "ready_for_shadow_live": stress_ready_for_shadow_live,
        "ready_for_merge": stress_ready_for_merge,
        "status": stress_decision.get("status", "unknown"),
        "promotion_decision": stress_decision,
    }
    validation_gate_passed = bool(validation_gate.get("passed", False))
    market_os_gate_passed = True if market_os_gate is None else bool(market_os_gate.get("passed", False))
    final_oos_passed = bool(final_oos_audit.get("passed", False))
    stress_gate_passed = bool(stress_gate["passed"])
    ready_for_shadow_live = bool(
        validation_gate_passed and market_os_gate_passed and final_oos_passed and stress_ready_for_shadow_live
    )
    ready_for_live = bool(validation_gate_passed and market_os_gate_passed and final_oos_passed and stress_gate_passed)
    ready_for_merge = ready_for_live
    if ready_for_live:
        status = "ready_for_live"
    elif ready_for_shadow_live:
        status = "shadow_ready_only"
    elif not validation_gate_passed:
        status = "validation_gate_blocked"
    elif not market_os_gate_passed:
        status = "market_os_gate_blocked"
    elif not final_oos_passed:
        status = "final_oos_audit_blocked"
    elif not stress_gate_passed:
        status = "stress_gate_blocked"
    else:
        status = "blocked"
    report = {
        "pipeline_mode": mode,
        "pairs": search_summary.get("pairs"),
        "executed_steps": executed_steps,
        "artifacts": {
            "search_summary": str(paths["search_summary_out"]),
            "validation_report": str(paths["validation_report_out"]),
            "stress_report": str(paths["stress_report_out"]),
        },
        "selected_candidate": None if selected is None else {
            "candidate_kind": selected.get("candidate_kind"),
            "route_breadth_threshold": selected.get("route_breadth_threshold"),
            "mapping_indices": selected.get("mapping_indices"),
            "pair_configs": selected.get("pair_configs"),
            "candidate_id": selected.get("candidate_id"),
            "tree_key": selected.get("tree_key"),
            "observation_mode": selected.get("observation_mode"),
            "label_horizon": selected.get("label_horizon"),
            "tree_depth": selected.get("tree_depth"),
            "logic_depth": selected.get("logic_depth"),
        },
        "selection": search_summary.get("selection"),
        "validation_engine": validation_engine,
        "validation_gate": validation_gate,
        "market_os_gate": market_os_gate,
        "final_oos_audit": final_oos_audit,
        "stress_gate": stress_gate,
        "ready_for_shadow_live": ready_for_shadow_live,
        "ready_for_live": ready_for_live,
        "ready_for_merge": ready_for_merge,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validation_report": validation_report,
        "stress_report": stress_report,
    }
    report["decision"] = {
        "status": status,
        "ready_for_shadow_live": ready_for_shadow_live,
        "ready_for_live": ready_for_live,
        "ready_for_merge": ready_for_merge,
        "validation_gate_passed": validation_gate_passed,
        "market_os_gate_passed": market_os_gate_passed,
        "final_oos_audit_passed": final_oos_passed,
        "stress_gate_passed": stress_gate_passed,
    }
    return report


def run_pipeline_step(mode: str, paths: dict[str, Path], args: argparse.Namespace, step: str) -> None:
    python = sys.executable
    if mode == PIPELINE_MODE_PAIRWISE_MARKET_OS:
        if step == "search":
            cmd = [
                python,
                "-u",
                str(SCRIPTS_DIR / "repair_pair_subset_pairwise_candidate.py"),
                "--pairs",
                args.pairs,
                "--candidate-summaries",
                str(paths["candidate_summaries"]),
                "--baseline-summary",
                str(paths["baseline_summary"]),
                "--summary-out",
                str(paths["search_summary_out"]),
            ]
            run_step(cmd, ROOT)
            return
        if step == "validation":
            return
        if step == "stress":
            cmd = [
                python,
                "-u",
                str(SCRIPTS_DIR / "run_pair_subset_pairwise_stress.py"),
                "--pairs",
                args.pairs,
                "--summary",
                str(paths["search_summary_out"]),
                "--report-out",
                str(paths["stress_report_out"]),
            ]
            run_step(cmd, ROOT)
            return
        raise ValueError(f"Unsupported step for pairwise_market_os: {step}")
    if mode == PIPELINE_MODE_PAIRWISE_MARKET_OS_FRACTAL:
        if step == "search":
            cmd = [
                python,
                "-u",
                str(SCRIPTS_DIR / "search_pair_subset_fractal_genome.py"),
                "--pairs",
                args.pairs,
                "--expert-summaries",
                str(paths["candidate_summaries"]),
                "--baseline-summary",
                str(paths["baseline_summary"]),
                "--summary-out",
                str(paths["search_summary_out"]),
            ]
            run_step(cmd, ROOT)
            return
        if step in {"validation", "stress"}:
            return
        raise ValueError(f"Unsupported step for pairwise_market_os_fractal: {step}")

    if step == "search":
        cmd = [
            python,
            "-u",
            str(SCRIPTS_DIR / "search_pair_subset_regime_mixture.py"),
            "--pairs",
            args.pairs,
            "--summary-out",
            str(paths["search_summary_out"]),
        ]
        run_step(cmd, ROOT)
        return
    if step == "validation":
        cmd = [
            python,
            "-u",
            str(SCRIPTS_DIR / "validate_pair_subset_summary.py"),
            "--summary",
            str(paths["search_summary_out"]),
            "--report-out",
            str(paths["validation_report_out"]),
        ]
        run_step(cmd, ROOT)
        return
    if step == "stress":
        cmd = [
            python,
            "-u",
            str(SCRIPTS_DIR / "run_pair_subset_stress_matrix.py"),
            "--pairs",
            args.pairs,
            "--summary",
            str(paths["search_summary_out"]),
            "--report-out",
            str(paths["stress_report_out"]),
        ]
        run_step(cmd, ROOT)
        return
    raise ValueError(f"Unsupported step: {step}")


def main() -> None:
    args = parse_args()
    mode = str(args.pipeline_mode)
    paths = resolve_mode_paths(args)

    executed_steps: list[dict[str, Any]] = []

    if not args.skip_search:
        run_pipeline_step(mode, paths, args, "search")
        executed_steps.append({"step": "search", "status": "executed"})
    else:
        executed_steps.append({"step": "search", "status": "skipped"})

    search_summary = load_json(paths["search_summary_out"])

    if mode == PIPELINE_MODE_PAIRWISE_MARKET_OS:
        validation_report = build_pairwise_validation_report(search_summary)
        if not args.skip_validation:
            write_json(paths["validation_report_out"], validation_report)
            executed_steps.append({"step": "validation", "status": "embedded"})
        else:
            executed_steps.append({"step": "validation", "status": "skipped"})
        if not args.skip_stress:
            run_pipeline_step(mode, paths, args, "stress")
            executed_steps.append({"step": "stress", "status": "executed"})
        else:
            executed_steps.append({"step": "stress", "status": "skipped"})
        stress_report = load_json(paths["stress_report_out"])
    elif mode == PIPELINE_MODE_PAIRWISE_MARKET_OS_FRACTAL:
        validation_report = build_fractal_validation_report(search_summary)
        stress_report = build_fractal_stress_report(search_summary)
        if not args.skip_validation:
            write_json(paths["validation_report_out"], validation_report)
            executed_steps.append({"step": "validation", "status": "embedded"})
        else:
            executed_steps.append({"step": "validation", "status": "skipped"})
        if not args.skip_stress:
            write_json(paths["stress_report_out"], stress_report)
            executed_steps.append({"step": "stress", "status": "embedded"})
        else:
            executed_steps.append({"step": "stress", "status": "skipped"})
    else:
        if not args.skip_validation:
            run_pipeline_step(mode, paths, args, "validation")
            executed_steps.append({"step": "validation", "status": "executed"})
        else:
            executed_steps.append({"step": "validation", "status": "skipped"})
        validation_report = load_json(paths["validation_report_out"])
        if not args.skip_stress:
            run_pipeline_step(mode, paths, args, "stress")
            executed_steps.append({"step": "stress", "status": "executed"})
        else:
            executed_steps.append({"step": "stress", "status": "skipped"})
        stress_report = load_json(paths["stress_report_out"])
    if mode == PIPELINE_MODE_LEGACY_SHARED:
        validation_report = build_legacy_validation_report(validation_report)

    report = build_final_report(
        mode=mode,
        search_summary=search_summary,
        validation_report=validation_report,
        stress_report=stress_report,
        executed_steps=executed_steps,
        paths=paths,
    )

    write_json(paths["pipeline_report_out"], report)
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
