#!/usr/bin/env python3
"""Build a final decision report for fractal market OS finalists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
import backtest_fractal_campaign_winner_periods as winner_periods
import run_fractal_market_os_deep_validation as deep_validation


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_BASE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
DEFAULT_MODEL_PATH = MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pressure fractal finalists through a 1y/4y executive gate.",
    )
    parser.add_argument("--campaign-report", required=True)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--base-summary", default=str(DEFAULT_BASE_SUMMARY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--report-out",
        default=str(MODELS_DIR / "gp_regime_mixture_btc_bnb_fractal_genome_market_os_finalist_gate.json"),
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_period_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    periods = {str(item.get("label")): item for item in (report.get("periods") or []) if isinstance(item, dict)}

    def period_metric(label: str, field: str) -> float | None:
        aggregate = ((periods.get(label) or {}).get("aggregate") or {})
        return float_or_none(aggregate.get(field))

    one_year = periods.get("1y") or {}
    one_year_pairs = one_year.get("per_pair") or {}
    weakest_pair_name = None
    weakest_pair_payload: dict[str, Any] = {}
    if one_year_pairs:
        weakest_pair_name, weakest_pair_payload = min(
            one_year_pairs.items(),
            key=lambda item: float_or_none((item[1] or {}).get("avg_daily_return")) or float("inf"),
        )

    return {
        "recent_2m_worst_daily": period_metric("2m", "worst_pair_avg_daily_return"),
        "recent_4m_worst_daily": period_metric("4m", "worst_pair_avg_daily_return"),
        "recent_6m_worst_daily": period_metric("6m", "worst_pair_avg_daily_return"),
        "one_year_worst_daily": period_metric("1y", "worst_pair_avg_daily_return"),
        "one_year_positive_pair_count": int(period_metric("1y", "positive_pair_count") or 0),
        "one_year_worst_max_drawdown": abs(period_metric("1y", "worst_max_drawdown") or 0.0),
        "one_year_weakest_pair": weakest_pair_name,
        "one_year_weakest_pair_avg_daily": float_or_none(weakest_pair_payload.get("avg_daily_return")),
        "one_year_weakest_pair_total_return": float_or_none(weakest_pair_payload.get("total_return")),
        "one_year_weakest_pair_max_drawdown": abs(float_or_none(weakest_pair_payload.get("max_drawdown")) or 0.0),
        "full_4y_worst_daily": period_metric("4y", "worst_pair_avg_daily_return"),
        "full_4y_positive_pair_count": int(period_metric("4y", "positive_pair_count") or 0),
        "full_4y_mdd": abs(period_metric("4y", "worst_max_drawdown") or 0.0),
    }


def classify_finalist_record(record: dict[str, Any]) -> tuple[str, str, list[str]]:
    decision = record.get("decision") or {}
    metrics = record.get("metrics") or {}
    period_audit = record.get("period_audit") or {}
    blockers = list(record.get("blockers") or [])

    one_year_worst = float_or_none(period_audit.get("one_year_worst_daily")) or 0.0
    one_year_positive_pairs = int(period_audit.get("one_year_positive_pair_count") or 0)
    full_4y_worst = float_or_none(period_audit.get("full_4y_worst_daily")) or 0.0
    full_4y_mdd = float_or_none(period_audit.get("full_4y_mdd")) or 1.0
    latest_reserve = float_or_none(metrics.get("latest_fold_stress_reserve_score")) or 0.0
    weak_pair = str(period_audit.get("one_year_weakest_pair") or "")

    if one_year_positive_pairs < 2:
        blockers.append("one_year_pair_breadth_failed")
    if one_year_worst < 0.0:
        blockers.append("one_year_worst_daily_failed")

    if weak_pair == "BNBUSDT" and one_year_worst < 0.0:
        repair_theme = "bnb_1y_repair"
    elif full_4y_worst < 0.0045 or full_4y_mdd > 0.15:
        repair_theme = "long_horizon_return_repair"
    else:
        repair_theme = "final_gate_repair"

    if bool(decision.get("ready_for_merge", False)) and one_year_positive_pairs >= 2 and one_year_worst >= 0.0:
        return "keep", repair_theme, sorted(set(blockers))
    if (
        bool(metrics.get("wf1_passed", False))
        and one_year_worst >= -0.0005
        and full_4y_worst >= 0.0045
        and full_4y_mdd <= 0.15
    ):
        return "watch", repair_theme, sorted(set(blockers))
    if (
        bool(metrics.get("wf1_passed", False))
        and latest_reserve > 0.0
        and one_year_worst >= -0.0003
        and full_4y_worst >= 0.0035
        and full_4y_mdd <= 0.19
    ):
        return "watch", repair_theme, sorted(set(blockers))
    return "drop", repair_theme, sorted(set(blockers))


def build_finalist_record(
    entry: dict[str, Any],
    *,
    df_all: pd.DataFrame,
    base_summary: Path,
    model_path: Path,
    equity: float,
    leverage: int,
) -> dict[str, Any]:
    core = deep_validation.build_candidate_record(
        entry,
        df_all=df_all,
        base_summary=base_summary,
        model_path=model_path,
        equity=equity,
        leverage=leverage,
    )
    artifacts = core.get("artifacts") or {}
    period_report = winner_periods.evaluate_summary_periods(
        summary_path=Path(artifacts["search_summary"]),
        pipeline_path=Path(artifacts["pipeline_report"]),
        base_summary=base_summary,
        model_path=model_path,
    )
    period_audit = extract_period_snapshot(period_report)
    band, repair_theme, blockers = classify_finalist_record(
        {
            **core,
            "period_audit": period_audit,
        }
    )
    core["period_audit"] = period_audit
    core["decision_band"] = band
    core["repair_theme"] = repair_theme
    core["blockers"] = blockers
    return core


def finalist_rank_key(record: dict[str, Any]) -> tuple[Any, ...]:
    metrics = record.get("metrics") or {}
    audit = record.get("period_audit") or {}
    return (
        1 if record.get("decision_band") == "keep" else 0,
        1 if record.get("decision_band") == "watch" else 0,
        1 if int(audit.get("one_year_positive_pair_count") or 0) >= 2 else 0,
        1 if (float_or_none(audit.get("one_year_worst_daily")) or float("-inf")) >= 0.0 else 0,
        float_or_none(audit.get("one_year_worst_daily")) or float("-inf"),
        float_or_none(audit.get("full_4y_worst_daily")) or float("-inf"),
        float_or_none(metrics.get("recent_6m_worst_daily")) or float("-inf"),
        float_or_none(metrics.get("latest_fold_stress_reserve_score")) or float("-inf"),
        -(float_or_none(audit.get("full_4y_mdd")) or float("inf")),
    )


def rank_finalist_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=finalist_rank_key, reverse=True)


def main() -> None:
    args = parse_args()
    campaign_report = load_json(args.campaign_report)
    entries = deep_validation.collect_candidate_entries(campaign_report, int(args.top_n))
    if not entries:
        raise RuntimeError("No finalist entries found in campaign report.")

    first_summary = load_json(entries[0]["artifacts"]["search_summary"])
    pairs = tuple(first_summary.get("pairs") or ("BTCUSDT", "BNBUSDT"))
    df_all = gp.load_all_pairs(pairs=list(pairs), start=None, end=None, refresh_cache=False)

    records = [
        build_finalist_record(
            entry,
            df_all=df_all,
            base_summary=Path(args.base_summary),
            model_path=Path(args.model),
            equity=float(args.equity),
            leverage=int(args.leverage),
        )
        for entry in entries
    ]
    ranked = rank_finalist_records(records)
    report = {
        "strategy_class": "fractal_market_os_finalist_gate",
        "campaign_report": str(Path(args.campaign_report).resolve()),
        "candidate_count": len(ranked),
        "kept_count": sum(1 for item in ranked if item.get("decision_band") == "keep"),
        "watch_count": sum(1 for item in ranked if item.get("decision_band") == "watch"),
        "drop_count": sum(1 for item in ranked if item.get("decision_band") == "drop"),
        "best_candidate": ranked[0] if ranked else None,
        "records": ranked,
    }
    output_path = Path(args.report_out)
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
