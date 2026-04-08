#!/usr/bin/env python3
"""Gate strategy promotion against the accepted target_050 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "models" / "rotation_target_050_baseline.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject candidate summaries that regress from the locked target_050 baseline.",
    )
    parser.add_argument(
        "--candidate-summary",
        required=True,
        help="Path to the candidate summary JSON.",
    )
    parser.add_argument(
        "--baseline-manifest",
        default=str(DEFAULT_BASELINE),
        help="Path to the baseline manifest JSON.",
    )
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="Fail if the candidate only matches baseline and does not improve any tracked metric.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def extract_stage_metrics(summary: dict[str, Any], stage_group: str) -> dict[str, dict[str, float]]:
    stages = summary["stages"][stage_group]
    metrics: dict[str, dict[str, float]] = {}
    for stage_name in ("validation", "test", "oos"):
        stage = stages[stage_name]
        daily = stage["daily_metrics"]
        metrics[stage_name] = {
            "total_return": float(stage["total_return"]),
            "avg_daily_return": float(daily["avg_daily_return"]),
            "daily_win_rate": float(daily["daily_win_rate"]),
            "daily_target_hit_rate": float(daily["daily_target_hit_rate"]),
            "max_drawdown": float(stage["max_drawdown"]),
        }
    return metrics


def check_target_flags(summary: dict[str, Any], required_flags: list[str]) -> dict[str, bool]:
    target_check = summary.get("target_check", {})
    return {flag: bool(target_check.get(flag, False)) for flag in required_flags}


def evaluate_candidate(
    baseline: dict[str, Any],
    candidate_summary: dict[str, Any],
    require_improvement: bool,
) -> dict[str, Any]:
    epsilon = float(baseline["rules"].get("epsilon", 0.0))
    stage_group = baseline["stage_group"]
    tracked_metrics = baseline["rules"]["non_regression_metrics"]
    strict_stages = set(baseline["rules"]["strict_improvement_stages"])

    baseline_metrics = baseline["stages"]
    candidate_metrics = extract_stage_metrics(candidate_summary, stage_group)
    target_flags = check_target_flags(candidate_summary, baseline["required_target_checks"])

    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    deltas: dict[str, dict[str, float]] = {}

    for stage_name, baseline_stage in baseline_metrics.items():
        candidate_stage = candidate_metrics[stage_name]
        deltas[stage_name] = {}
        for metric_name in tracked_metrics:
            delta = float(candidate_stage[metric_name] - baseline_stage[metric_name])
            deltas[stage_name][metric_name] = delta

            if delta < -epsilon:
                regressions.append(
                    {
                        "stage": stage_name,
                        "metric": metric_name,
                        "baseline": baseline_stage[metric_name],
                        "candidate": candidate_stage[metric_name],
                        "delta": delta,
                    }
                )
            elif stage_name in strict_stages and delta > epsilon:
                improvements.append(
                    {
                        "stage": stage_name,
                        "metric": metric_name,
                        "baseline": baseline_stage[metric_name],
                        "candidate": candidate_stage[metric_name],
                        "delta": delta,
                    }
                )

    target_flags_pass = all(target_flags.values())
    non_regression_pass = not regressions
    strict_improvement_pass = bool(improvements)
    promotion_eligible = target_flags_pass and non_regression_pass and (
        strict_improvement_pass if require_improvement else True
    )

    return {
        "baseline_name": baseline["name"],
        "baseline_commit": baseline["baseline_commit"],
        "baseline_summary_path": baseline["baseline_summary_path"],
        "candidate_stage_group": stage_group,
        "candidate_target_flags": target_flags,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": deltas,
        "regressions": regressions,
        "improvements": improvements,
        "target_flags_pass": target_flags_pass,
        "non_regression_pass": non_regression_pass,
        "strict_improvement_pass": strict_improvement_pass,
        "require_improvement": require_improvement,
        "promotion_eligible": promotion_eligible,
    }


def print_human_summary(result: dict[str, Any]) -> None:
    print("=" * 80)
    print("Strategy Promotion Gate")
    print("=" * 80)
    print(f"Baseline:  {result['baseline_name']} @ {result['baseline_commit']}")
    print(f"Eligible:  {result['promotion_eligible']}")
    print(f"Target OK: {result['target_flags_pass']}")
    print(f"No Regr.:  {result['non_regression_pass']}")
    print(f"Improved:  {result['strict_improvement_pass']}")

    if result["regressions"]:
        print("\nRegressions:")
        for item in result["regressions"]:
            print(
                f"  - {item['stage']}.{item['metric']}: "
                f"{item['candidate']:.12f} < {item['baseline']:.12f} "
                f"(delta {item['delta']:+.12f})"
            )

    if result["improvements"]:
        print("\nImprovements:")
        for item in result["improvements"]:
            print(
                f"  - {item['stage']}.{item['metric']}: "
                f"{item['candidate']:.12f} > {item['baseline']:.12f} "
                f"(delta {item['delta']:+.12f})"
            )

    print("\nJSON:")
    print(json.dumps(result, indent=2))


def main() -> int:
    args = parse_args()
    baseline = load_json(args.baseline_manifest)
    candidate_summary = load_json(args.candidate_summary)
    result = evaluate_candidate(baseline, candidate_summary, require_improvement=args.require_improvement)
    print_human_summary(result)
    return 0 if result["promotion_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
