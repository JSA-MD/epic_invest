#!/usr/bin/env python3
"""Run a parallel multi-seed fractal market OS search campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
MODELS_DIR = ROOT / "models"
PIPELINE_MODE_FRACTAL = "pairwise_market_os_fractal"


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("at least one seed is required")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel fractal market OS re-search across multiple seeds.",
    )
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT")
    parser.add_argument("--seeds", default="20260415,20260416,20260417")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--elite-count", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--expert-pool-size", type=int, default=16)
    parser.add_argument("--filter-mode", default="heuristic")
    parser.add_argument("--observation-modes", default="")
    parser.add_argument("--label-horizons", default="")
    parser.add_argument("--warm-start-summaries", default="")
    parser.add_argument("--warm-start-candidate-limit", type=int, default=12)
    parser.add_argument("--warm-start-variant-budget", type=int, default=24)
    parser.add_argument("--local-search-rate", type=float, default=0.35)
    parser.add_argument("--local-search-mutation-burst", type=int, default=2)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--summary-prefix",
        default=str(MODELS_DIR / "gp_regime_mixture_btc_bnb_fractal_genome_market_os_campaign"),
    )
    parser.add_argument(
        "--campaign-report-out",
        default=str(MODELS_DIR / "gp_regime_mixture_btc_bnb_fractal_genome_market_os_campaign_report.json"),
    )
    parser.add_argument("--top-n-report", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metric_value(selected: dict[str, Any], window_key: str, field: str) -> float | None:
    return float_or_none((((selected.get("windows") or {}).get(window_key) or {}).get("aggregate") or {}).get(field))


def robustness_stress_passed(robustness: dict[str, Any], tolerance: float = 1e-12) -> bool:
    threshold = float_or_none(robustness.get("stress_survival_threshold"))
    mean_survival = float_or_none(robustness.get("stress_survival_rate_mean"))
    min_survival = float_or_none(robustness.get("min_fold_stress_survival_rate"))
    if min_survival is None:
        min_survival = float_or_none(robustness.get("stress_survival_rate_min"))
    latest_reserve = float_or_none(robustness.get("latest_fold_stress_reserve_score"))
    if threshold is None or mean_survival is None or min_survival is None or latest_reserve is None:
        return False
    return bool(
        mean_survival >= threshold - tolerance
        and min_survival >= threshold - tolerance
        and latest_reserve >= -tolerance
    )


def robustness_cost_reserve_passed(robustness: dict[str, Any], tolerance: float = 1e-12) -> bool:
    threshold = float_or_none(robustness.get("stress_survival_threshold"))
    latest_rate = float_or_none(robustness.get("latest_non_nominal_stress_survival_rate"))
    min_rate = float_or_none(robustness.get("min_fold_non_nominal_stress_survival_rate"))
    latest_reserve = float_or_none(robustness.get("latest_non_nominal_stress_reserve_score"))
    if threshold is None or latest_rate is None or min_rate is None or latest_reserve is None:
        return False
    return bool(
        latest_rate >= threshold - tolerance
        and min_rate >= threshold - tolerance
        and latest_reserve >= -tolerance
    )


def robustness_stress_min_passed(robustness: dict[str, Any], tolerance: float = 1e-12) -> bool:
    threshold = float_or_none(robustness.get("stress_survival_threshold"))
    min_survival = float_or_none(robustness.get("min_fold_stress_survival_rate"))
    if min_survival is None:
        min_survival = float_or_none(robustness.get("stress_survival_rate_min"))
    latest_reserve = float_or_none(robustness.get("latest_fold_stress_reserve_score"))
    if threshold is None or min_survival is None or latest_reserve is None:
        return False
    return bool(min_survival >= threshold - tolerance and latest_reserve >= -tolerance)


def robustness_non_nominal_min_passed(robustness: dict[str, Any], tolerance: float = 1e-12) -> bool:
    threshold = float_or_none(robustness.get("stress_survival_threshold"))
    min_rate = float_or_none(robustness.get("min_fold_non_nominal_stress_survival_rate"))
    latest_reserve = float_or_none(robustness.get("latest_non_nominal_stress_reserve_score"))
    if threshold is None or min_rate is None or latest_reserve is None:
        return False
    return bool(min_rate >= threshold - tolerance and latest_reserve >= -tolerance)


def build_artifact_paths(summary_prefix: str, seed: int) -> dict[str, Path]:
    base = f"{summary_prefix}_seed{seed}"
    return {
        "search_summary": Path(f"{base}_summary.json"),
        "validation_report": Path(f"{base}_validation.json"),
        "stress_report": Path(f"{base}_stress.json"),
        "pipeline_report": Path(f"{base}_pipeline.json"),
    }


def run_command(cmd: list[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "elapsed_seconds": elapsed,
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))
    return payload


def build_campaign_entry(
    *,
    seed: int,
    artifact_paths: dict[str, Path],
    search_summary: dict[str, Any],
    pipeline_report: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    selected = search_summary.get("selected_candidate") or {}
    decision = pipeline_report.get("decision") or {}
    robustness = selected.get("robustness") or {}
    validation = (selected.get("validation") or {}).get("profiles") or {}
    wf1 = robustness.get("wf_1") or {}
    joint_repair_market_os_passed = bool((validation.get("joint_repair_market_os") or {}).get("passed", False))
    pair_repair_1y_passed = bool((validation.get("pair_repair_1y") or {}).get("passed", False))
    robustness_stress_gate = robustness_stress_passed(robustness)
    robustness_cost_reserve_gate = robustness_cost_reserve_passed(robustness)
    robustness_stress_min_gate = robustness_stress_min_passed(robustness)
    robustness_non_nominal_min_gate = robustness_non_nominal_min_passed(robustness)
    repair_metrics = selected.get("repair_metrics") or {}
    recent_6m_worst_daily = metric_value(selected, "recent_6m", "worst_pair_avg_daily_return")
    full_4y_worst_daily = metric_value(selected, "full_4y", "worst_pair_avg_daily_return")
    full_4y_mdd = abs(metric_value(selected, "full_4y", "worst_max_drawdown") or 0.0)
    recent_6m_floor_060_passed = bool((recent_6m_worst_daily or float("-inf")) >= 0.006)
    full_4y_floor_045_passed = bool((full_4y_worst_daily or float("-inf")) >= 0.0045)
    full_4y_mdd_015_passed = bool(full_4y_mdd <= 0.15 + 1e-12)
    repair_pair_daily = float_or_none(repair_metrics.get("repair_pair_avg_daily_return"))
    repair_pair_total = float_or_none(repair_metrics.get("repair_pair_total_return"))
    repair_pair_mdd = float_or_none(repair_metrics.get("repair_pair_max_drawdown"))
    positive_pair_count = int_or_none(repair_metrics.get("positive_pair_count"))
    pair_count = int_or_none(repair_metrics.get("pair_count"))
    repair_hard_gate_passed = bool(
        pair_repair_1y_passed
        and repair_pair_daily is not None
        and repair_pair_total is not None
        and repair_pair_mdd is not None
        and pair_count is not None
        and positive_pair_count is not None
        and pair_count > 0
        and positive_pair_count >= pair_count
        and repair_pair_daily >= -1e-12
        and repair_pair_total >= -1e-12
        and abs(repair_pair_mdd) <= 0.15 + 1e-12
    )
    final_hard_gate_passed = bool(
        repair_hard_gate_passed
        and bool((validation.get("target_060") or {}).get("passed", False))
        and bool(wf1.get("passed", False))
        and robustness_stress_gate
        and robustness_cost_reserve_gate
    )
    joint_repair_min_floor_passed = bool(
        repair_hard_gate_passed
        and bool(wf1.get("passed", False))
        and recent_6m_floor_060_passed
        and full_4y_floor_045_passed
        and full_4y_mdd_015_passed
        and robustness_stress_min_gate
        and robustness_non_nominal_min_gate
    )
    joint_repair_stress_passed = bool(
        joint_repair_market_os_passed
        and bool(wf1.get("passed", False))
        and robustness_stress_gate
        and robustness_cost_reserve_gate
    )

    gate_flags = {
        "validation_gate_passed": bool(decision.get("validation_gate_passed", False)),
        "market_os_gate_passed": bool(decision.get("market_os_gate_passed", False)),
        "final_oos_audit_passed": bool(decision.get("final_oos_audit_passed", False)),
        "stress_gate_passed": bool(decision.get("stress_gate_passed", False)),
        "ready_for_live": bool(decision.get("ready_for_live", False)),
        "ready_for_merge": bool(decision.get("ready_for_merge", False)),
        "wf1_passed": bool(wf1.get("passed", False)),
    }
    gate_pass_count = sum(1 for passed in gate_flags.values() if passed)

    return {
        "seed": int(seed),
        "status": "ok",
        "elapsed_seconds": float(elapsed_seconds),
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "decision_status": decision.get("status"),
        "candidate_kind": selected.get("candidate_kind", "fractal_tree"),
        "tree_key": selected.get("tree_key"),
        "observation_mode": selected.get("observation_mode"),
        "label_horizon": selected.get("label_horizon"),
        "tree_depth": int_or_none(selected.get("tree_depth")),
        "logic_depth": int_or_none(selected.get("logic_depth")),
        "gate_flags": gate_flags,
        "gate_pass_count": int(gate_pass_count),
        "recent_2m_worst_daily": metric_value(selected, "recent_2m", "worst_pair_avg_daily_return"),
        "recent_6m_worst_daily": recent_6m_worst_daily,
        "full_4y_worst_daily": full_4y_worst_daily,
        "recent_2m_mdd": abs(metric_value(selected, "recent_2m", "worst_max_drawdown") or 0.0),
        "recent_6m_mdd": abs(metric_value(selected, "recent_6m", "worst_max_drawdown") or 0.0),
        "full_4y_mdd": full_4y_mdd,
        "stress_survival_mean": float_or_none(robustness.get("stress_survival_rate_mean")),
        "stress_survival_min": float_or_none(robustness.get("stress_survival_rate_min")),
        "stress_survival_threshold": float_or_none(robustness.get("stress_survival_threshold")),
        "latest_fold_stress_reserve_score": float_or_none(robustness.get("latest_fold_stress_reserve_score")),
        "latest_fold_non_nominal_survival": float_or_none(robustness.get("latest_non_nominal_stress_survival_rate")),
        "latest_non_nominal_stress_reserve_score": float_or_none(robustness.get("latest_non_nominal_stress_reserve_score")),
        "target_060_passed": bool((validation.get("target_060") or {}).get("passed", False)),
        "progressive_passed": bool((validation.get("progressive_improvement") or {}).get("passed", False)),
        "final_hard_gate_passed": final_hard_gate_passed,
        "repair_hard_gate_passed": repair_hard_gate_passed,
        "joint_repair_min_floor_passed": joint_repair_min_floor_passed,
        "joint_repair_stress_passed": joint_repair_stress_passed,
        "joint_repair_market_os_passed": joint_repair_market_os_passed,
        "pair_repair_1y_passed": pair_repair_1y_passed,
        "robustness_stress_passed": robustness_stress_gate,
        "robustness_stress_min_passed": robustness_stress_min_gate,
        "cost_reserve_passed": robustness_cost_reserve_gate,
        "robustness_non_nominal_min_passed": robustness_non_nominal_min_gate,
        "recent_6m_floor_060_passed": recent_6m_floor_060_passed,
        "full_4y_floor_045_passed": full_4y_floor_045_passed,
        "full_4y_mdd_015_passed": full_4y_mdd_015_passed,
        "final_oos_passed": bool((validation.get("final_oos") or {}).get("passed", False)),
        "repair_pair": repair_metrics.get("repair_pair"),
        "recent_1y_repair_pair_daily": repair_pair_daily,
        "recent_1y_repair_pair_total_return": repair_pair_total,
        "recent_1y_repair_pair_mdd": None if repair_pair_mdd is None else abs(repair_pair_mdd),
        "recent_1y_positive_pair_count": positive_pair_count,
        "recent_1y_pair_count": pair_count,
    }


def campaign_rank_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(bool((entry.get("gate_flags") or {}).get("ready_for_merge", False))),
        int(bool(entry.get("final_hard_gate_passed", False))),
        int(bool(entry.get("repair_hard_gate_passed", False))),
        int(bool(entry.get("joint_repair_min_floor_passed", False))),
        int(bool(entry.get("joint_repair_stress_passed", False))),
        int(bool((entry.get("gate_flags") or {}).get("stress_gate_passed", False))),
        int(bool((entry.get("gate_flags") or {}).get("market_os_gate_passed", False))),
        int(bool((entry.get("gate_flags") or {}).get("final_oos_audit_passed", False))),
        int(bool((entry.get("gate_flags") or {}).get("validation_gate_passed", False))),
        int(bool(entry.get("joint_repair_market_os_passed", False))),
        int(bool((entry.get("gate_flags") or {}).get("wf1_passed", False))),
        int(bool(entry.get("pair_repair_1y_passed", False))),
        int(bool(entry.get("full_4y_floor_045_passed", False))),
        int(bool(entry.get("robustness_stress_min_passed", False))),
        int(bool(entry.get("robustness_non_nominal_min_passed", False))),
        int(bool(entry.get("cost_reserve_passed", False))),
        float_or_none(entry.get("full_4y_worst_daily")) or float("-inf"),
        float_or_none(entry.get("recent_6m_worst_daily")) or float("-inf"),
        float_or_none(entry.get("stress_survival_min")) or float("-inf"),
        float_or_none(entry.get("recent_1y_repair_pair_daily")) or float("-inf"),
        float_or_none(entry.get("recent_2m_worst_daily")) or float("-inf"),
        float_or_none(entry.get("stress_survival_mean")) or float("-inf"),
        float_or_none(entry.get("latest_fold_non_nominal_survival")) or float("-inf"),
        float_or_none(entry.get("latest_non_nominal_stress_reserve_score")) or float("-inf"),
        float_or_none(entry.get("latest_fold_stress_reserve_score")) or float("-inf"),
        -(float_or_none(entry.get("full_4y_mdd")) or float("inf")),
        -(float_or_none(entry.get("recent_6m_mdd")) or float("inf")),
        -(float_or_none(entry.get("recent_2m_mdd")) or float("inf")),
        -(float_or_none(entry.get("elapsed_seconds")) or float("inf")),
    )


def rank_campaign_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [entry for entry in entries if entry.get("status") == "ok"]
    return sorted(successful, key=campaign_rank_key, reverse=True)


def pick_best_by_key(entries: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for entry in rank_campaign_entries(entries):
        value = entry.get(field)
        if value is None:
            continue
        value_str = str(value)
        grouped.setdefault(value_str, entry)
    return grouped


def build_campaign_report(
    *,
    config: dict[str, Any],
    entries: list[dict[str, Any]],
    top_n_report: int,
) -> dict[str, Any]:
    ranked = rank_campaign_entries(entries)
    failed = [entry for entry in entries if entry.get("status") != "ok"]
    gate_counts = {
        "ready_for_merge": sum(int(bool((entry.get("gate_flags") or {}).get("ready_for_merge", False))) for entry in ranked),
        "ready_for_live": sum(int(bool((entry.get("gate_flags") or {}).get("ready_for_live", False))) for entry in ranked),
        "validation_gate_passed": sum(int(bool((entry.get("gate_flags") or {}).get("validation_gate_passed", False))) for entry in ranked),
        "market_os_gate_passed": sum(int(bool((entry.get("gate_flags") or {}).get("market_os_gate_passed", False))) for entry in ranked),
        "final_oos_audit_passed": sum(int(bool((entry.get("gate_flags") or {}).get("final_oos_audit_passed", False))) for entry in ranked),
        "stress_gate_passed": sum(int(bool((entry.get("gate_flags") or {}).get("stress_gate_passed", False))) for entry in ranked),
        "wf1_passed": sum(int(bool((entry.get("gate_flags") or {}).get("wf1_passed", False))) for entry in ranked),
        "final_hard_gate_passed": sum(int(bool(entry.get("final_hard_gate_passed", False))) for entry in ranked),
        "repair_hard_gate_passed": sum(int(bool(entry.get("repair_hard_gate_passed", False))) for entry in ranked),
        "joint_repair_min_floor_passed": sum(int(bool(entry.get("joint_repair_min_floor_passed", False))) for entry in ranked),
        "joint_repair_stress_passed": sum(int(bool(entry.get("joint_repair_stress_passed", False))) for entry in ranked),
        "joint_repair_market_os_passed": sum(int(bool(entry.get("joint_repair_market_os_passed", False))) for entry in ranked),
        "pair_repair_1y_passed": sum(int(bool(entry.get("pair_repair_1y_passed", False))) for entry in ranked),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "job_count": len(entries),
        "completed_count": len(ranked),
        "failed_count": len(failed),
        "gate_counts": gate_counts,
        "best_candidate": ranked[0] if ranked else None,
        "top_candidates": ranked[: max(int(top_n_report), 1)],
        "best_by_observation_mode": pick_best_by_key(ranked, "observation_mode"),
        "best_by_label_horizon": pick_best_by_key(ranked, "label_horizon"),
        "failed_jobs": failed,
        "jobs": entries,
    }


def run_seed_campaign_job(
    *,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifact_paths = build_artifact_paths(str(args.summary_prefix), seed)
    started = time.perf_counter()
    try:
        search_summary_path = artifact_paths["search_summary"]
        if not (args.skip_existing and search_summary_path.exists()):
            search_cmd = [
                str(args.python_bin),
                str(SCRIPTS_DIR / "search_pair_subset_fractal_genome.py"),
                "--pairs",
                str(args.pairs),
                "--expert-pool-size",
                str(int(args.expert_pool_size)),
                "--population",
                str(int(args.population)),
                "--generations",
                str(int(args.generations)),
                "--elite-count",
                str(int(args.elite_count)),
                "--top-k",
                str(int(args.top_k)),
                "--filter-mode",
                str(args.filter_mode),
                "--seed",
                str(int(seed)),
                "--summary-out",
                str(search_summary_path),
            ]
            if str(args.observation_modes).strip():
                search_cmd.extend(["--observation-modes", str(args.observation_modes)])
            if str(args.label_horizons).strip():
                search_cmd.extend(["--label-horizons", str(args.label_horizons)])
            if str(args.warm_start_summaries).strip():
                search_cmd.extend(["--warm-start-summaries", str(args.warm_start_summaries)])
            search_cmd.extend(["--warm-start-candidate-limit", str(int(args.warm_start_candidate_limit))])
            search_cmd.extend(["--warm-start-variant-budget", str(int(args.warm_start_variant_budget))])
            search_cmd.extend(["--local-search-rate", str(float(args.local_search_rate))])
            search_cmd.extend(["--local-search-mutation-burst", str(int(args.local_search_mutation_burst))])
            run_command(search_cmd, ROOT)

        pipeline_path = artifact_paths["pipeline_report"]
        if not (args.skip_existing and pipeline_path.exists()):
            pipeline_cmd = [
                str(args.python_bin),
                str(SCRIPTS_DIR / "run_pair_subset_promotion_pipeline.py"),
                "--pipeline-mode",
                PIPELINE_MODE_FRACTAL,
                "--skip-search",
                "--search-summary-out",
                str(artifact_paths["search_summary"]),
                "--validation-report-out",
                str(artifact_paths["validation_report"]),
                "--stress-report-out",
                str(artifact_paths["stress_report"]),
                "--pipeline-report-out",
                str(artifact_paths["pipeline_report"]),
            ]
            run_command(pipeline_cmd, ROOT)

        elapsed = time.perf_counter() - started
        return build_campaign_entry(
            seed=seed,
            artifact_paths=artifact_paths,
            search_summary=load_json(artifact_paths["search_summary"]),
            pipeline_report=load_json(artifact_paths["pipeline_report"]),
            elapsed_seconds=elapsed,
        )
    except Exception as exc:  # pragma: no cover - defensive shell wrapper
        elapsed = time.perf_counter() - started
        return {
            "seed": int(seed),
            "status": "failed",
            "elapsed_seconds": float(elapsed),
            "error": str(exc),
            "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        }


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    max_workers = max(1, min(int(args.max_workers), len(seeds)))

    entries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fractal-campaign") as executor:
        futures = {
            executor.submit(run_seed_campaign_job, seed=seed, args=args): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            entries.append(future.result())

    entries.sort(key=lambda item: int(item.get("seed", 0)))
    report = build_campaign_report(
        config={
            "pairs": str(args.pairs),
            "seeds": [int(seed) for seed in seeds],
            "max_workers": max_workers,
            "population": int(args.population),
            "generations": int(args.generations),
            "elite_count": int(args.elite_count),
            "top_k": int(args.top_k),
            "expert_pool_size": int(args.expert_pool_size),
            "filter_mode": str(args.filter_mode),
            "observation_modes": str(args.observation_modes),
            "label_horizons": str(args.label_horizons),
            "warm_start_summaries": str(args.warm_start_summaries),
            "warm_start_candidate_limit": int(args.warm_start_candidate_limit),
            "warm_start_variant_budget": int(args.warm_start_variant_budget),
            "local_search_rate": float(args.local_search_rate),
            "local_search_mutation_burst": int(args.local_search_mutation_burst),
            "summary_prefix": str(args.summary_prefix),
            "python_bin": str(args.python_bin),
            "skip_existing": bool(args.skip_existing),
        },
        entries=entries,
        top_n_report=int(args.top_n_report),
    )
    output_path = Path(args.campaign_report_out)
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
