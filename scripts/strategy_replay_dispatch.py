from __future__ import annotations

from typing import Any

from pairwise_regime_mixture_shadow_live import detect_candidate_kind
from search_pair_subset_fractal_genome import fast_fractal_replay_from_context
from search_pair_subset_regime_mixture import realistic_overlay_replay_from_context


SUMMARY_WINDOW_LABELS = {
    "2m": "recent_2m",
    "6m": "recent_6m",
    "4y": "full_4y",
}
AUDIT_PAIR_FLOAT_FIELDS = ("total_return", "daily_win_rate", "max_drawdown", "avg_daily_return")
AUDIT_PAIR_INT_FIELDS = ("n_trades",)
AUDIT_AGG_FLOAT_FIELDS = (
    "mean_total_return",
    "worst_pair_total_return",
    "mean_avg_daily_return",
    "worst_pair_avg_daily_return",
    "worst_max_drawdown",
)


def resolve_pairwise_route_state_mode(candidate: dict[str, Any], pair: str) -> str:
    pair_configs = candidate.get("pair_configs") or {}
    pair_config = pair_configs.get(pair) or {}
    return str(pair_config.get("route_state_mode") or "base")


def replay_candidate_from_context(
    *,
    candidate: dict[str, Any],
    pair: str,
    context: dict[str, Any],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...],
    leaf_runtime_array: Any | None,
    leaf_codes: Any | None,
) -> dict[str, Any]:
    candidate_kind = detect_candidate_kind(candidate)
    if candidate_kind == "pairwise_candidate":
        pair_config = candidate["pair_configs"][pair]
        return realistic_overlay_replay_from_context(
            context,
            library_lookup,
            tuple(int(v) for v in pair_config["mapping_indices"]),
            float(pair_config["route_breadth_threshold"]),
        )
    if candidate_kind == "fractal_tree":
        if leaf_runtime_array is None or leaf_codes is None:
            raise RuntimeError("Fractal replay requires leaf runtime arrays and leaf codes.")
        return fast_fractal_replay_from_context(
            context,
            library_lookup,
            route_thresholds,
            leaf_runtime_array,
            leaf_codes,
        )
    raise RuntimeError(f"Unsupported candidate kind for replay: {candidate_kind}")


def audit_replay_against_candidate_windows(
    period_reports: list[dict[str, Any]],
    candidate: dict[str, Any],
    *,
    float_tolerance: float = 1e-9,
) -> dict[str, Any]:
    candidate_windows = candidate.get("windows") or {}
    checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for period in period_reports:
        summary_label = SUMMARY_WINDOW_LABELS.get(str(period.get("label")))
        if not summary_label:
            continue
        expected = candidate_windows.get(summary_label)
        if not isinstance(expected, dict):
            continue

        actual_start = str(period.get("start"))
        actual_end = str(period.get("end"))
        expected_start = str(expected.get("start"))
        expected_end = str(expected.get("end"))
        if actual_start != expected_start or actual_end != expected_end:
            checks.append(
                {
                    "label": str(period.get("label")),
                    "status": "skipped_misaligned",
                    "actual_window": {"start": actual_start, "end": actual_end},
                    "expected_window": {"start": expected_start, "end": expected_end},
                }
            )
            continue

        window_mismatches: list[dict[str, Any]] = []
        actual_pairs = period.get("per_pair") or {}
        expected_pairs = expected.get("per_pair") or {}
        for pair, expected_metrics in expected_pairs.items():
            actual_metrics = actual_pairs.get(pair) or {}
            for field in AUDIT_PAIR_FLOAT_FIELDS:
                expected_value = expected_metrics.get(field)
                actual_value = actual_metrics.get(field)
                if expected_value is None or actual_value is None:
                    continue
                delta = abs(float(actual_value) - float(expected_value))
                if delta > float_tolerance:
                    window_mismatches.append(
                        {
                            "scope": "per_pair",
                            "pair": pair,
                            "field": field,
                            "expected": float(expected_value),
                            "actual": float(actual_value),
                            "delta": float(delta),
                        }
                    )
            for field in AUDIT_PAIR_INT_FIELDS:
                expected_value = expected_metrics.get(field)
                actual_value = actual_metrics.get(field)
                if expected_value is None or actual_value is None:
                    continue
                if int(actual_value) != int(expected_value):
                    window_mismatches.append(
                        {
                            "scope": "per_pair",
                            "pair": pair,
                            "field": field,
                            "expected": int(expected_value),
                            "actual": int(actual_value),
                            "delta": int(actual_value) - int(expected_value),
                        }
                    )

        actual_aggregate = period.get("aggregate") or {}
        expected_aggregate = expected.get("aggregate") or {}
        for field in AUDIT_AGG_FLOAT_FIELDS:
            expected_value = expected_aggregate.get(field)
            actual_value = actual_aggregate.get(field)
            if expected_value is None or actual_value is None:
                continue
            delta = abs(float(actual_value) - float(expected_value))
            if delta > float_tolerance:
                window_mismatches.append(
                    {
                        "scope": "aggregate",
                        "field": field,
                        "expected": float(expected_value),
                        "actual": float(actual_value),
                        "delta": float(delta),
                    }
                )

        status = "passed" if not window_mismatches else "mismatch"
        checks.append({"label": str(period.get("label")), "status": status, "mismatch_count": len(window_mismatches)})
        if window_mismatches:
            mismatches.append({"label": str(period.get("label")), "details": window_mismatches})

    status = "passed"
    if mismatches:
        status = "mismatch"
    elif not checks:
        status = "skipped"

    return {
        "status": status,
        "checked_windows": len(checks),
        "checks": checks,
        "mismatches": mismatches,
    }
