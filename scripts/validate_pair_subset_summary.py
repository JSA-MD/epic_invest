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


def get_agg(summary: dict[str, Any], mode: str, window: str) -> dict[str, Any]:
    if mode == "baseline":
        return summary["baseline_realistic"][window]["aggregate"]
    if mode == "selected":
        return summary["selected_candidate"]["windows"][window]["aggregate"]
    raise ValueError(mode)


def build_progressive_profile(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    agg_2m = get_agg(summary, mode, "recent_2m")
    agg_6m = get_agg(summary, mode, "recent_6m")
    agg_4y = get_agg(summary, mode, "full_4y")
    base_2m = get_agg(summary, "baseline", "recent_2m")
    base_6m = get_agg(summary, "baseline", "recent_6m")
    base_4y = get_agg(summary, "baseline", "full_4y")

    checks = [
        check_at_least(
            "recent_2m_worst_pair_positive",
            agg_2m["worst_pair_avg_daily_return"],
            0.0,
            "최근 2개월 최악 코인도 음수가 아니어야 함",
        ),
        check_at_least(
            "recent_2m_worst_pair_vs_baseline",
            agg_2m["worst_pair_avg_daily_return"],
            base_2m["worst_pair_avg_daily_return"],
            "최근 2개월 최악 코인 기준으로 baseline 이상",
        ),
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
            "recent_2m_worst_mdd_cap",
            abs(agg_2m["worst_max_drawdown"]),
            0.18,
            "최근 2개월 최악 MDD 18% 이내",
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


def build_target_060_profile(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    agg_2m = get_agg(summary, mode, "recent_2m")
    agg_6m = get_agg(summary, mode, "recent_6m")
    agg_4y = get_agg(summary, mode, "full_4y")
    checks = [
        check_at_least(
            "recent_2m_worst_pair_target_060",
            agg_2m["worst_pair_avg_daily_return"],
            0.006,
            "최근 2개월 최악 코인 0.6%/day 이상",
        ),
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


def build_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for window in ("recent_2m", "recent_6m", "full_4y"):
        base = get_agg(summary, "baseline", window)
        sel = get_agg(summary, "selected", window)
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


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    report_out = Path(args.report_out)
    summary = json.loads(summary_path.read_text())

    report = {
        "summary_path": str(summary_path),
        "pairs": summary["pairs"],
        "baseline_candidate": summary["baseline_candidate"],
        "selected_candidate": {
            "route_breadth_threshold": summary["selected_candidate"]["route_breadth_threshold"],
            "mapping_indices": summary["selected_candidate"]["mapping_indices"],
        },
        "comparison": build_comparison(summary),
        "profiles": {
            "progressive_improvement": {
                "baseline": build_progressive_profile(summary, "baseline"),
                "selected": build_progressive_profile(summary, "selected"),
            },
            "target_060": {
                "baseline": build_target_060_profile(summary, "baseline"),
                "selected": build_target_060_profile(summary, "selected"),
            },
        },
    }
    report_out.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
