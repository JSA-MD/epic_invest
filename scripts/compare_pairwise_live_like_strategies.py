#!/usr/bin/env python3
"""Compare deployed pairwise logic vs full research module under one live-like replay."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import gp_crypto_evolution as gp
from btc_convex_blend import get_btc_convex_blend, replay_btc_convex_blend_candidate
from btc_event_blend import replay_btc_event_blend_candidate
from btc_online_blend import replay_btc_online_blend_candidate
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS, load_live_frame
from replay_regime_mixture_realistic import load_model as load_signal_model
from safety_guards import enforce_runtime_gross_cap_ceiling
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    realistic_overlay_replay,
    realistic_overlay_replay_from_context,
)

UTC = timezone.utc


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def _filter_funding_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC) + pd.Timedelta(days=1)
    return df.loc[(df["fundingTime"] >= start_ts) & (df["fundingTime"] < end_ts)].copy()


def _standard_windows(index: pd.DatetimeIndex, anchor_end: str | None) -> list[tuple[str, str, str]]:
    if len(index) == 0:
        raise RuntimeError("No data available for window construction.")
    first_day = pd.Timestamp(index[0]).tz_convert(UTC).normalize()
    end_day = pd.Timestamp(anchor_end, tz=UTC).normalize() if anchor_end else pd.Timestamp(index[-1]).tz_convert(UTC).normalize()

    def month_start(months: int) -> str:
        return max(first_day, end_day - pd.DateOffset(months=months)).date().isoformat()

    def year_start(years: int) -> str:
        return max(first_day, end_day - pd.DateOffset(years=years)).date().isoformat()

    return [
        ("recent_2m", month_start(2), end_day.date().isoformat()),
        ("recent_4m", month_start(4), end_day.date().isoformat()),
        ("recent_6m", month_start(6), end_day.date().isoformat()),
        ("recent_1y", year_start(1), end_day.date().isoformat()),
        ("full", first_day.date().isoformat(), end_day.date().isoformat()),
    ]


def _day_count_for_frame(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(pd.DatetimeIndex(df.index).normalize().nunique())


def _summarize_result(result: dict[str, Any], day_count: int) -> dict[str, Any]:
    daily = result.get("daily_metrics") or {}
    if not daily:
        daily = gp.compute_daily_metrics(np.asarray(result.get("trace", {}).get("bar_net", []), dtype="float64"))
    total_return = float(result["total_return"])
    return {
        "days": int(day_count),
        "daily_win_rate": float(daily.get("daily_win_rate", 0.0)),
        "daily_target_hit_rate": float(daily.get("daily_target_hit_rate", 0.0)),
        "profit_usdt": float(total_return * gp.INITIAL_CASH),
        "total_return": total_return,
        "return_pct": float(total_return * 100.0),
        "final_equity": float(result.get("final_equity", gp.INITIAL_CASH * (1.0 + total_return))),
        "max_drawdown": float(result.get("max_drawdown", 0.0)),
        "max_drawdown_pct": float(result.get("max_drawdown", 0.0) * 100.0),
        "n_trades": int(result.get("n_trades", 0)),
        "fee_paid": float(result.get("fee_paid", 0.0)),
        "slippage_paid": float(result.get("slippage_paid", 0.0)),
        "funding_paid": float(result.get("funding_paid", 0.0)),
        "funding_events": int(result.get("funding_events", 0)),
    }


def _aggregate(per_pair: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not per_pair:
        return {}
    vals = list(per_pair.values())
    return {
        "days": float(np.mean([v["days"] for v in vals])),
        "daily_win_rate": float(np.mean([v["daily_win_rate"] for v in vals])),
        "daily_target_hit_rate": float(np.mean([v["daily_target_hit_rate"] for v in vals])),
        "profit_usdt": float(np.mean([v["profit_usdt"] for v in vals])),
        "total_return": float(np.mean([v["total_return"] for v in vals])),
        "return_pct": float(np.mean([v["return_pct"] for v in vals])),
        "final_equity": float(np.mean([v["final_equity"] for v in vals])),
        "max_drawdown": float(np.min([v["max_drawdown"] for v in vals])),
        "max_drawdown_pct": float(np.min([v["max_drawdown_pct"] for v in vals])),
        "n_trades": float(np.mean([v["n_trades"] for v in vals])),
        "fee_paid": float(np.mean([v["fee_paid"] for v in vals])),
        "slippage_paid": float(np.mean([v["slippage_paid"] for v in vals])),
        "funding_paid": float(np.mean([v["funding_paid"] for v in vals])),
    }


def _strip_research_modules(candidate: dict[str, Any]) -> dict[str, Any]:
    stripped = copy.deepcopy(candidate)
    for key in ("btc_convex_blend", "pair_convex_blends", "btc_online_blend", "btc_event_blend"):
        stripped.pop(key, None)
    return stripped


def _replay_pair(
    *,
    strategy_name: str,
    candidate: dict[str, Any],
    pair: str,
    df_window: pd.DataFrame,
    raw_signal: pd.Series,
    funding_df: pd.DataFrame,
    library: list[Any],
    library_lookup: dict[str, Any],
    use_equity_corr_risk: bool,
    min_notional_usd: float,
    max_hold_bars: int,
    runtime_gross_cap: float,
    return_trace: bool,
) -> dict[str, Any]:
    pair_cfg = candidate["pair_configs"][pair]
    route_state_mode = str(pair_cfg.get("route_state_mode") or "base")
    overlay_inputs = build_overlay_inputs(df_window, PAIRS, regime_pair=pair)
    context = build_fast_context(
        df=df_window,
        pair=pair,
        raw_signal=raw_signal,
        overlay_inputs=overlay_inputs,
        route_thresholds=(float(pair_cfg["route_breadth_threshold"]),),
        library_lookup=library_lookup,
        funding_df=funding_df,
        route_state_mode=route_state_mode,
    )
    if strategy_name == "research_full" and pair == "BTCUSDT" and candidate.get("btc_event_blend"):
        return replay_btc_event_blend_candidate(
            candidate=candidate,
            pair=pair,
            context=context,
            library_lookup=library_lookup,
            use_equity_corr_risk=use_equity_corr_risk,
            min_notional_usd=min_notional_usd,
            max_hold_bars=max_hold_bars,
            runtime_gross_cap=runtime_gross_cap,
            return_trace=return_trace,
        )
    if strategy_name == "research_full" and pair == "BTCUSDT" and candidate.get("btc_online_blend"):
        return replay_btc_online_blend_candidate(
            candidate=candidate,
            pair=pair,
            context=context,
            library_lookup=library_lookup,
            use_equity_corr_risk=use_equity_corr_risk,
            min_notional_usd=min_notional_usd,
            max_hold_bars=max_hold_bars,
            runtime_gross_cap=runtime_gross_cap,
            return_trace=return_trace,
        )
    if strategy_name == "research_full" and get_btc_convex_blend(candidate, pair) is not None:
        return replay_btc_convex_blend_candidate(
            candidate=candidate,
            pair=pair,
            context=context,
            library_lookup=library_lookup,
            use_equity_corr_risk=use_equity_corr_risk,
            min_notional_usd=min_notional_usd,
            max_hold_bars=max_hold_bars,
            runtime_gross_cap=runtime_gross_cap,
            return_trace=return_trace,
        )
    return realistic_overlay_replay_from_context(
        context,
        library_lookup,
        tuple(int(v) for v in pair_cfg["mapping_indices"]),
        float(pair_cfg["route_breadth_threshold"]),
        use_equity_corr_risk=use_equity_corr_risk,
        execution_gene=pair_cfg.get("execution_gene"),
        min_notional_usd=min_notional_usd,
        max_hold_bars=max_hold_bars,
        runtime_gross_cap=runtime_gross_cap,
        engine="python",
        return_trace=return_trace,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare current deployed pairwise strategy vs full research module.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-out", type=Path, default=Path("/tmp/pairwise_live_like_strategy_ab.json"))
    parser.add_argument("--history-start", default="2023-03-04")
    parser.add_argument("--anchor-end", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_candidate = json.loads(args.summary_path.read_text())["selected_candidate"]
    strategies = {
        "current_deployed": _strip_research_modules(selected_candidate),
        "research_full": copy.deepcopy(selected_candidate),
    }
    library = list(iter_params())
    library_lookup = build_library_lookup(library)
    model_tree, _ = load_signal_model(args.model_path)
    compiled = gp.toolbox.compile(expr=model_tree)

    refresh_live_data = _env_bool("PAIRWISE_REFRESH_LIVE_DATA", True)
    if refresh_live_data:
        df_all = load_live_frame(PAIRS, refresh_live_data=True)
        df_all = df_all.loc[pd.Timestamp(args.history_start, tz=UTC):].copy()
    else:
        df_all = gp.load_all_pairs(pairs=list(PAIRS), start=args.history_start, end=None, refresh_cache=False)
    if df_all.empty:
        raise RuntimeError("No OHLCV data loaded.")

    windows = _standard_windows(pd.DatetimeIndex(df_all.index), args.anchor_end)
    funding_all = {pair: gp.load_funding_rates(pair, windows[-1][1], windows[-1][2]) for pair in PAIRS}
    effective_gross_cap, gross_cap_warning = enforce_runtime_gross_cap_ceiling()
    min_notional_usd = _env_float("REBALANCE_NOTIONAL_BAND_USD", 25.0)
    max_hold_bars = _env_int("PAIRWISE_MAX_HOLD_BARS", 288)
    use_equity_corr_risk = _env_bool("PAIRWISE_EQUITY_CORR_RISK", False)

    raw_signal_all = {
        pair: pd.Series(
            np.asarray(compiled(*gp.get_feature_arrays(df_all, pair)), dtype="float64"),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in PAIRS
    }

    report: dict[str, Any] = {
        "data_source": os.getenv("EPIC_MARKET_DATA_SOURCE", "csv"),
        "summary_path": str(args.summary_path),
        "model_path": str(args.model_path),
        "initial_cash": float(gp.INITIAL_CASH),
        "data_range": {
            "start": pd.Timestamp(df_all.index[0]).isoformat(),
            "end": pd.Timestamp(df_all.index[-1]).isoformat(),
            "rows": int(len(df_all)),
        },
        "runtime_env": {
            "PAIRWISE_GROSS_CAP": _env_float("PAIRWISE_GROSS_CAP", 1.0),
            "PAIRWISE_LIVE_MAX_GROSS_CAP": _env_float("PAIRWISE_LIVE_MAX_GROSS_CAP", 0.05),
            "PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP": _env_bool("PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP", False),
            "PAIRWISE_EFFECTIVE_GROSS_CAP": float(effective_gross_cap),
            "PAIRWISE_GROSS_CAP_WARNING": gross_cap_warning,
            "REBALANCE_NOTIONAL_BAND_USD": float(min_notional_usd),
            "PAIRWISE_MAX_HOLD_BARS": int(max_hold_bars),
            "PAIRWISE_CVAR_CUT": _env_bool("PAIRWISE_CVAR_CUT", True),
            "PAIRWISE_EQUITY_CORR_RISK": bool(use_equity_corr_risk),
            "PAIRWISE_REFRESH_LIVE_DATA": bool(refresh_live_data),
        },
        "strategies": {},
        "comparison": {},
    }

    for strategy_name, candidate in strategies.items():
        strategy_windows: dict[str, Any] = {}
        latest_targets: dict[str, float] = {}
        for label, start, end in windows:
            df_window = _filter_window(df_all, start, end)
            day_count = _day_count_for_frame(df_window)
            per_pair: dict[str, Any] = {}
            for pair in PAIRS:
                funding_df = _filter_funding_window(funding_all[pair], start, end)
                raw_signal = raw_signal_all[pair].reindex(df_window.index).fillna(0.0)
                result = _replay_pair(
                    strategy_name=strategy_name,
                    candidate=candidate,
                    pair=pair,
                    df_window=df_window,
                    raw_signal=raw_signal,
                    funding_df=funding_df,
                    library=library,
                    library_lookup=library_lookup,
                    use_equity_corr_risk=use_equity_corr_risk,
                    min_notional_usd=min_notional_usd,
                    max_hold_bars=max_hold_bars,
                    runtime_gross_cap=float(effective_gross_cap),
                    return_trace=label == windows[-1][0],
                )
                per_pair[pair] = _summarize_result(result, day_count)
                if label == windows[-1][0]:
                    trace = result.get("trace") or {}
                    target = np.asarray(trace.get("target_weight", []), dtype="float64")
                    latest_targets[pair] = float(target[-1]) if target.size else 0.0
            strategy_windows[label] = {
                "start": start,
                "end": end,
                "per_pair": per_pair,
                "aggregate": _aggregate(per_pair),
            }
        report["strategies"][strategy_name] = {
            "windows": strategy_windows,
            "latest_targets": latest_targets,
            "latest_gross_target": float(sum(abs(v) for v in latest_targets.values())),
        }

    for label, _, _ in windows:
        current = report["strategies"]["current_deployed"]["windows"][label]["aggregate"]
        research = report["strategies"]["research_full"]["windows"][label]["aggregate"]
        report["comparison"][label] = {
            "current_return_pct": current["return_pct"],
            "research_return_pct": research["return_pct"],
            "return_delta_pct": research["return_pct"] - current["return_pct"],
            "current_mdd_pct": current["max_drawdown_pct"],
            "research_mdd_pct": research["max_drawdown_pct"],
            "mdd_delta_pct": research["max_drawdown_pct"] - current["max_drawdown_pct"],
            "current_trades": current["n_trades"],
            "research_trades": research["n_trades"],
            "trade_delta": research["n_trades"] - current["n_trades"],
            "winner_by_return": "research_full" if research["return_pct"] > current["return_pct"] else "current_deployed",
        }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True))
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
