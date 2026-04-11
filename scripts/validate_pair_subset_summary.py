#!/usr/bin/env python3
"""Validate subset-pair search results against staged gate profiles."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    actual: float
    target: float | None = None
    comparator: str | None = None
    note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a subset-pair regime-mixture summary against staged gate profiles.",
    )
    parser.add_argument(
        "--summary",
        default="models/gp_regime_mixture_btc_bnb_search_summary.json",
    )
    parser.add_argument(
        "--report-out",
        default="models/gp_regime_mixture_btc_bnb_validation_report.json",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, CheckResult):
        return {
            "name": value.name,
            "passed": value.passed,
            "actual": value.actual,
            "target": value.target,
            "comparator": value.comparator,
            "note": value.note,
        }
    return value


def check_at_least(name: str, actual: float, target: float, note: str | None = None) -> CheckResult:
    return CheckResult(
        name=name,
        passed=actual >= target,
        actual=float(actual),
        target=float(target),
        comparator=">=",
        note=note,
    )


def check_at_most(name: str, actual: float, target: float, note: str | None = None) -> CheckResult:
    return CheckResult(
        name=name,
        passed=actual <= target,
        actual=float(actual),
        target=float(target),
        comparator="<=",
        note=note,
    )


def get_agg(windows: dict[str, Any], window: str) -> dict[str, Any]:
    return windows[window]["aggregate"]


def get_pair_metrics(windows: dict[str, Any], window: str, pair: str | None) -> dict[str, Any]:
    if not pair:
        return {}
    return (((windows.get(window) or {}).get("per_pair") or {}).get(pair) or {})


def build_progressive_profile_for_windows(candidate_windows: dict[str, Any], baseline_windows: dict[str, Any]) -> dict[str, Any]:
    agg_6m = get_agg(candidate_windows, "recent_6m")
    agg_4y = get_agg(candidate_windows, "full_4y")
    base_6m = get_agg(baseline_windows, "recent_6m")
    base_4y = get_agg(baseline_windows, "full_4y")

    checks = [
        check_at_least(
            "recent_6m_worst_pair_min",
            agg_6m["worst_pair_avg_daily_return"],
            0.006,
            "최근 6개월 최악 코인은 목표 0.6%/day 이상",
        ),
        check_at_least(
            "recent_6m_worst_pair_vs_baseline",
            agg_6m["worst_pair_avg_daily_return"],
            base_6m["worst_pair_avg_daily_return"],
            "최근 6개월 최악 코인 기준 baseline 이상",
        ),
        check_at_least(
            "full_4y_worst_pair_positive",
            agg_4y["worst_pair_avg_daily_return"],
            0.0,
            "4년 전체 최악 코인도 음수가 아니어야 함",
        ),
        check_at_least(
            "full_4y_mean_vs_baseline",
            agg_4y["mean_avg_daily_return"],
            base_4y["mean_avg_daily_return"],
            "4년 평균 일수익률은 baseline 이상",
        ),
        check_at_most(
            "recent_6m_worst_mdd_cap",
            abs(agg_6m["worst_max_drawdown"]),
            0.17,
            "최근 6개월 최악 MDD 17% 이내",
        ),
        check_at_most(
            "full_4y_worst_mdd_cap",
            abs(agg_4y["worst_max_drawdown"]),
            0.20,
            "4년 전체 최악 MDD 20% 이내",
        ),
    ]
    return {
        "checks": checks,
        "passed": all(item.passed for item in checks),
    }


def build_final_oos_profile_for_windows(candidate_windows: dict[str, Any], baseline_windows: dict[str, Any]) -> dict[str, Any]:
    agg_2m = get_agg(candidate_windows, "recent_2m")
    base_2m = get_agg(baseline_windows, "recent_2m")
    checks = [
        check_at_least(
            "recent_2m_worst_pair_positive",
            agg_2m["worst_pair_avg_daily_return"],
            0.0,
            "최근 2개월 최악 코인은 음수가 아니어야 함",
        ),
        check_at_least(
            "recent_2m_worst_pair_vs_baseline",
            agg_2m["worst_pair_avg_daily_return"],
            base_2m["worst_pair_avg_daily_return"] * 0.97,
            "최근 2개월 최악 코인은 baseline의 97% 이상 유지",
        ),
        check_at_most(
            "recent_2m_worst_mdd_cap",
            abs(agg_2m["worst_max_drawdown"]),
            0.18,
            "최근 2개월 최악 MDD 18% 이내",
        ),
    ]
    return {
        "checks": checks,
        "passed": all(item.passed for item in checks),
    }


def build_target_060_profile_for_windows(candidate_windows: dict[str, Any]) -> dict[str, Any]:
    agg_6m = get_agg(candidate_windows, "recent_6m")
    agg_4y = get_agg(candidate_windows, "full_4y")
    checks = [
        check_at_least(
            "recent_6m_worst_pair_target_060",
            agg_6m["worst_pair_avg_daily_return"],
            0.006,
            "최근 6개월 최악 코인 0.6%/day 이상",
        ),
        check_at_least(
            "full_4y_worst_pair_target_060",
            agg_4y["worst_pair_avg_daily_return"],
            0.006,
            "4년 전체 최악 코인 0.6%/day 이상",
        ),
        check_at_most(
            "recent_6m_worst_mdd_cap",
            abs(agg_6m["worst_max_drawdown"]),
            0.17,
            "최근 6개월 최악 MDD 17% 이내",
        ),
        check_at_most(
            "full_4y_worst_mdd_cap",
            abs(agg_4y["worst_max_drawdown"]),
            0.20,
            "4년 전체 최악 MDD 20% 이내",
        ),
    ]
    return {
        "checks": checks,
        "passed": all(item.passed for item in checks),
    }


def build_pair_repair_profile_for_windows(candidate_windows: dict[str, Any], repair_pair: str | None) -> dict[str, Any]:
    if not repair_pair or "recent_1y" not in candidate_windows:
        return {
            "repair_pair": repair_pair,
            "checks": [],
            "passed": False,
            "skipped": True,
        }

    agg_1y = get_agg(candidate_windows, "recent_1y")
    pair_metrics = get_pair_metrics(candidate_windows, "recent_1y", repair_pair)
    pair_count = len(((candidate_windows.get("recent_1y") or {}).get("per_pair") or {}))
    checks = [
        check_at_least(
            "recent_1y_repair_pair_non_negative",
            float(pair_metrics.get("avg_daily_return", 0.0)),
            0.0,
            f"최근 1년 {repair_pair} 일수익률은 음수가 아니어야 함",
        ),
        check_at_least(
            "recent_1y_positive_pair_coverage",
            float(agg_1y.get("positive_pair_count", 0.0)),
            float(pair_count),
            "최근 1년에는 전체 pair가 모두 양수여야 함",
        ),
        check_at_most(
            "recent_1y_worst_mdd_cap",
            abs(float(agg_1y.get("worst_max_drawdown", 0.0))),
            0.15,
            "최근 1년 최악 MDD 15% 이내",
        ),
        check_at_least(
            "recent_1y_repair_pair_total_return_positive",
            float(pair_metrics.get("total_return", 0.0)),
            0.0,
            f"최근 1년 {repair_pair} 누적수익률은 음수가 아니어야 함",
        ),
    ]
    return {
        "repair_pair": repair_pair,
        "checks": checks,
        "passed": all(item.passed for item in checks),
        "skipped": False,
    }


def build_joint_repair_market_os_profile_for_windows(
    candidate_windows: dict[str, Any],
    baseline_windows: dict[str, Any],
    repair_pair: str | None,
) -> dict[str, Any]:
    repair_profile = build_pair_repair_profile_for_windows(candidate_windows, repair_pair)
    if repair_profile.get("skipped"):
        return {
            "repair_pair": repair_pair,
            "checks": [],
            "passed": False,
            "skipped": True,
        }

    agg_6m = get_agg(candidate_windows, "recent_6m")
    agg_4y = get_agg(candidate_windows, "full_4y")
    base_6m = get_agg(baseline_windows, "recent_6m")
    checks = list(repair_profile["checks"])
    checks.extend(
        [
            check_at_least(
                "recent_6m_worst_pair_joint_floor",
                float(agg_6m["worst_pair_avg_daily_return"]),
                0.006,
                "최근 6개월 최악 코인 0.6%/day 이상",
            ),
            check_at_least(
                "recent_6m_worst_pair_joint_vs_baseline",
                float(agg_6m["worst_pair_avg_daily_return"]),
                float(base_6m["worst_pair_avg_daily_return"]),
                "최근 6개월 최악 코인은 baseline 이상",
            ),
            check_at_least(
                "full_4y_worst_pair_joint_floor",
                float(agg_4y["worst_pair_avg_daily_return"]),
                0.0045,
                "4년 전체 최악 코인 0.45%/day 이상",
            ),
            check_at_most(
                "full_4y_worst_mdd_joint_cap",
                abs(float(agg_4y["worst_max_drawdown"])),
                0.15,
                "4년 전체 최악 MDD 15% 이내",
            ),
        ]
    )
    return {
        "repair_pair": repair_pair,
        "checks": checks,
        "passed": all(item.passed for item in checks),
        "skipped": False,
    }


def build_comparison_for_windows(candidate_windows: dict[str, Any], baseline_windows: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for window in ("recent_2m", "recent_6m", "full_4y"):
        base = get_agg(baseline_windows, window)
        sel = get_agg(candidate_windows, window)
        out[window] = {
            "baseline_mean_avg_daily_return": float(base["mean_avg_daily_return"]),
            "selected_mean_avg_daily_return": float(sel["mean_avg_daily_return"]),
            "delta_mean_avg_daily_return": float(sel["mean_avg_daily_return"] - base["mean_avg_daily_return"]),
            "baseline_worst_pair_avg_daily_return": float(base["worst_pair_avg_daily_return"]),
            "selected_worst_pair_avg_daily_return": float(sel["worst_pair_avg_daily_return"]),
            "delta_worst_pair_avg_daily_return": float(sel["worst_pair_avg_daily_return"] - base["worst_pair_avg_daily_return"]),
            "baseline_worst_max_drawdown": float(base["worst_max_drawdown"]),
            "selected_worst_max_drawdown": float(sel["worst_max_drawdown"]),
            "delta_worst_max_drawdown": float(sel["worst_max_drawdown"] - base["worst_max_drawdown"]),
        }
    return out


def build_validation_bundle(
    candidate_windows: dict[str, Any],
    baseline_windows: dict[str, Any],
    repair_pair: str | None = None,
) -> dict[str, Any]:
    return {
        "comparison": build_comparison_for_windows(candidate_windows, baseline_windows),
        "profiles": {
            "progressive_improvement": build_progressive_profile_for_windows(candidate_windows, baseline_windows),
            "final_oos": build_final_oos_profile_for_windows(candidate_windows, baseline_windows),
            "target_060": build_target_060_profile_for_windows(candidate_windows),
            "pair_repair_1y": build_pair_repair_profile_for_windows(candidate_windows, repair_pair),
            "joint_repair_market_os": build_joint_repair_market_os_profile_for_windows(
                candidate_windows,
                baseline_windows,
                repair_pair,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    report_out = Path(args.report_out)
    summary = json.loads(summary_path.read_text())
    baseline_windows = summary["baseline_realistic"]
    selected_windows = summary["selected_candidate"]["windows"] if summary.get("selected_candidate") else None

    report = {
        "summary_path": str(summary_path),
        "pairs": summary["pairs"],
        "baseline_candidate": summary["baseline_candidate"],
        "selected_candidate": None if summary.get("selected_candidate") is None else {
            "route_breadth_threshold": summary["selected_candidate"]["route_breadth_threshold"],
            "mapping_indices": summary["selected_candidate"]["mapping_indices"],
        },
        "comparison": None if selected_windows is None else build_comparison_for_windows(selected_windows, baseline_windows),
        "profiles": {
            "progressive_improvement": {
                "baseline": build_progressive_profile_for_windows(baseline_windows, baseline_windows),
                "selected": None if selected_windows is None else build_progressive_profile_for_windows(selected_windows, baseline_windows),
            },
            "final_oos": {
                "baseline": build_final_oos_profile_for_windows(baseline_windows, baseline_windows),
                "selected": None if selected_windows is None else build_final_oos_profile_for_windows(selected_windows, baseline_windows),
            },
            "target_060": {
                "baseline": build_target_060_profile_for_windows(baseline_windows),
                "selected": None if selected_windows is None else build_target_060_profile_for_windows(selected_windows),
            },
            "pair_repair_1y": {
                "baseline": build_pair_repair_profile_for_windows(baseline_windows, "BNBUSDT"),
                "selected": None if selected_windows is None else build_pair_repair_profile_for_windows(selected_windows, "BNBUSDT"),
            },
            "joint_repair_market_os": {
                "baseline": build_joint_repair_market_os_profile_for_windows(baseline_windows, baseline_windows, "BNBUSDT"),
                "selected": None if selected_windows is None else build_joint_repair_market_os_profile_for_windows(selected_windows, baseline_windows, "BNBUSDT"),
            },
        },
    }
    report_out.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
