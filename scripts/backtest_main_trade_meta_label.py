#!/usr/bin/env python3
"""Backtest trade-level meta-labeling on top of the current pairwise main strategy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from btc_convex_blend import replay_btc_convex_blend_candidate
from derivative_market_data import load_derivative_bundle
from pairwise_regime_mixture_shadow_live import load_strategy_bundle
from search_main_execution_beam import compare_to_baseline
from search_pair_subset_fractal_genome import load_funding_from_cache_or_empty
from search_pair_subset_pairwise_moo_router import SEARCH_WINDOWS
from search_pair_subset_regime_mixture import (
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    realistic_overlay_replay_from_context,
)
from strategy_replay_dispatch import replay_candidate_from_context


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DEFAULT_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json"
DEFAULT_BASE_SUMMARY = MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_summary.json"
DEFAULT_MODEL = MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"
DEFAULT_OUT = MODELS_DIR / "main_trade_meta_label_20260413.json"


@dataclass
class TradeMetaGate:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    train_end: str
    validation_end: str

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(np.nan_to_num(np.asarray(x, dtype="float64"), nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
        x_norm = np.nan_to_num((x - self.mean) / self.scale, nan=0.0, posinf=0.0, neginf=0.0)
        x_norm = np.clip(x_norm, -12.0, 12.0)
        weights = np.clip(np.nan_to_num(self.weights, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        bias = float(np.clip(np.nan_to_num(self.bias, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0))
        z = np.clip(x_norm @ weights + bias, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest trade-level meta-labeling on top of pairwise main.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--base-summary", default=str(DEFAULT_BASE_SUMMARY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--summary-out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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
    if isinstance(value, TradeMetaGate):
        return {
            "feature_names": list(value.feature_names),
            "mean": value.mean.tolist(),
            "scale": value.scale.tolist(),
            "weights": value.weights.tolist(),
            "bias": float(value.bias),
            "threshold": float(value.threshold),
            "train_end": value.train_end,
            "validation_end": value.validation_end,
        }
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n")


def fit_balanced_logistic(
    x: np.ndarray,
    y: np.ndarray,
    reg: float,
    *,
    epochs: int = 2000,
    lr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.clip(np.nan_to_num(np.asarray(x, dtype="float64"), nan=0.0, posinf=50.0, neginf=-50.0), -50.0, 50.0)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    x_norm = np.clip(x_norm, -12.0, 12.0)

    positive_rate = float(np.clip(y.mean(), 1e-6, 1.0 - 1e-6))
    weights = np.zeros(x.shape[1], dtype="float64")
    bias = float(np.log(positive_rate / (1.0 - positive_rate)))

    pos_weight = len(y) / max(1.0, 2.0 * y.sum())
    neg_weight = len(y) / max(1.0, 2.0 * (len(y) - y.sum()))
    sample_weight = np.where(y > 0.5, pos_weight, neg_weight)

    for _ in range(epochs):
        z = np.clip(x_norm @ weights + bias, -40.0, 40.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        error = (prob - y) * sample_weight
        grad_w = (x_norm.T @ error) / len(y) + reg * weights
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b
        weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0))

    return mean, scale, weights, bias


def normalize_day(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def build_window_cache(
    *,
    summary_path: Path,
    base_summary: Path,
    model_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...], tuple[float, ...], dict[str, Any], dict[str, Any]]:
    summary = load_json(summary_path)
    pairs = tuple(summary.get("pairs") or ("BTCUSDT", "BNBUSDT"))
    route_thresholds = tuple(float(v) for v in (summary.get("search", {}).get("route_thresholds") or (0.35, 0.50, 0.65, 0.80)))
    bundle = load_strategy_bundle(summary_path, base_summary, model_path, candidate_key="selected_candidate")
    library_lookup = build_library_lookup(bundle["library"])

    start_all = SEARCH_WINDOWS[-1][1]
    end_all = SEARCH_WINDOWS[0][2]
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=False)
    compiled = bundle["compiled_model"]
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {pair: load_funding_from_cache_or_empty(pair, start_all, end_all) for pair in pairs}
    derivative_all = {
        pair: load_derivative_bundle(
            pair,
            start_dt=pd.Timestamp(start_all, tz="UTC").to_pydatetime(),
            end_dt=pd.Timestamp(end_all, tz="UTC").to_pydatetime(),
            fetch=False,
            lookback_days=3650,
        )
        for pair in pairs
    }

    window_cache: dict[str, Any] = {}
    selected = summary["selected_candidate"]
    for label, start, end in SEARCH_WINDOWS:
        df = df_all.loc[start:end].copy()
        pair_cache: dict[str, Any] = {}
        for pair in pairs:
            overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
            signal_slice = raw_signal_all[pair].loc[start:end].copy()
            funding_slice = funding_all[pair]
            if not funding_slice.empty:
                funding_slice = funding_slice[
                    (funding_slice["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                    & (funding_slice["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
                ].copy()
            pair_cache[pair] = {
                "fast_context": build_fast_context(
                    df=df,
                    pair=pair,
                    raw_signal=signal_slice,
                    overlay_inputs=overlay_inputs,
                    route_thresholds=route_thresholds,
                    library_lookup=library_lookup,
                    funding_df=funding_slice,
                    derivative_bundle=derivative_all[pair],
                    route_state_mode=str((selected.get("pair_configs") or {}).get(pair, {}).get("route_state_mode") or "equity_corr"),
                )
            }
        window_cache[label] = {"df": df, "pairs": pair_cache}
    return summary, bundle, pairs, route_thresholds, window_cache, library_lookup


def replay_pair(
    *,
    candidate: dict[str, Any],
    pair: str,
    context: dict[str, Any],
    library_lookup: dict[str, Any],
    route_thresholds: tuple[float, ...] | None = None,
    entry_keep_flags: np.ndarray | None = None,
    return_trace: bool = False,
) -> dict[str, Any]:
    if pair == "BTCUSDT" and candidate.get("btc_event_blend") and entry_keep_flags is None:
        from btc_event_blend import replay_btc_event_blend_candidate

        return replay_btc_event_blend_candidate(
            candidate=candidate,
            pair=pair,
            context=context,
            library_lookup=library_lookup,
            return_trace=return_trace,
        )
    if pair == "BTCUSDT" and candidate.get("btc_online_blend") and entry_keep_flags is None:
        from btc_online_blend import replay_btc_online_blend_candidate

        return replay_btc_online_blend_candidate(
            candidate=candidate,
            pair=pair,
            context=context,
            library_lookup=library_lookup,
            return_trace=return_trace,
        )
    if pair == "BTCUSDT" and candidate.get("btc_convex_blend") and entry_keep_flags is None:
        return replay_btc_convex_blend_candidate(
            candidate=candidate,
            pair=pair,
            context=context,
            library_lookup=library_lookup,
            return_trace=return_trace,
        )
    replay = replay_candidate_from_context(
        candidate=candidate,
        pair=pair,
        context=context,
        library_lookup=library_lookup,
        route_thresholds=tuple(route_thresholds or (0.5,)),
        leaf_runtime_array=None,
        leaf_codes=None,
    )
    if entry_keep_flags is None and not return_trace:
        return replay
    pair_cfg = ((candidate.get("pair_configs") or {}).get(pair) or {})
    return realistic_overlay_replay_from_context(
        context,
        library_lookup,
        tuple(int(v) for v in pair_cfg["mapping_indices"]),
        float(pair_cfg["route_breadth_threshold"]),
        execution_gene=pair_cfg.get("execution_gene"),
        engine="python",
        entry_keep_flags=entry_keep_flags,
        return_trace=return_trace,
    )


def build_entry_feature_names() -> list[str]:
    return [
        "regime",
        "breadth",
        "vol_ann",
        "equity_corr",
        "btc_qqq_corr_5d",
        "btc_qqq_corr_20d",
        "btc_spy_beta_20d",
        "btc_dxy_corr_20d",
        "btc_gold_corr_20d",
        "order_imbalance",
        "buy_volume_share",
        "candle_micro_score",
        "range_bps",
        "volume_ratio",
        "oi_rel",
        "basis_rate",
        "top_pos_log_ratio",
        "taker_buy_sell_log_ratio",
        "dc_trend_05",
        "dc_run_05",
        "signal_abs",
        "signal_signed",
        "requested_abs",
        "role_idx",
        "side",
    ]


def build_entry_feature_row(context: dict[str, Any], trace: dict[str, Any], idx: int) -> list[float]:
    return [
        float(context["regime"][idx]),
        float(context["breadth"][idx]),
        float(context["vol_ann"][idx]),
        float(context["equity_corr"][idx]),
        float(context["btc_qqq_corr_5d"][idx]),
        float(context["btc_qqq_corr_20d"][idx]),
        float(context["btc_spy_beta_20d"][idx]),
        float(context["btc_dxy_corr_20d"][idx]),
        float(context["btc_gold_corr_20d"][idx]),
        float(context["order_imbalance"][idx]),
        float(context["buy_volume_share"][idx]),
        float(context["candle_micro_score"][idx]),
        float(context["range_bps"][idx]),
        float(context["volume_ratio"][idx]),
        float(context["oi_rel"][idx]),
        float(context["basis_rate"][idx]),
        float(context["top_pos_log_ratio"][idx]),
        float(context["taker_buy_sell_log_ratio"][idx]),
        float(context["dc_trend_05"][idx]),
        float(context["dc_run_05"][idx]),
        abs(float(trace["signal_pct"][idx])) / 100.0,
        float(trace["signal_pct"][idx]) / 100.0,
        abs(float(trace["requested_weight"][idx])),
        float(trace["role_idx"][idx]),
        1.0 if float(trace["target_weight"][idx]) > 0.0 else -1.0,
    ]


def extract_trade_rows(context: dict[str, Any], replay_result: dict[str, Any]) -> list[dict[str, Any]]:
    trace = replay_result.get("trace") or {}
    target_raw = trace.get("target_weight")
    bar_net_raw = trace.get("bar_net")
    target = np.asarray([] if target_raw is None else target_raw, dtype="float64")
    bar_net = np.asarray([] if bar_net_raw is None else bar_net_raw, dtype="float64")
    if len(target) == 0 or len(bar_net) == 0:
        return []
    days = pd.DatetimeIndex(context["bar_day_index"][: len(target)]).tz_localize(None)
    feature_names = build_entry_feature_names()
    rows: list[dict[str, Any]] = []
    current_side = 0
    trade_start = None
    for i in range(len(target)):
        weight = float(target[i])
        side = 0
        if weight > 1e-12:
            side = 1
        elif weight < -1e-12:
            side = -1

        if current_side == 0 and side != 0:
            trade_start = i
            current_side = side
            continue

        if current_side != 0 and side != current_side:
            if trade_start is not None and i > trade_start:
                segment = bar_net[trade_start:i]
                trade_ret = float(np.prod(1.0 + segment) - 1.0)
                rows.append(
                    {
                        "entry_index": int(trade_start),
                        "entry_day": normalize_day(days[trade_start]),
                        "side": int(current_side),
                        "total_return": trade_ret,
                        "label": int(trade_ret > 0.0),
                        "features": build_entry_feature_row(context, trace, trade_start),
                        "feature_names": feature_names,
                    }
                )
            trade_start = i if side != 0 else None
            current_side = side

    if current_side != 0 and trade_start is not None and len(target) > trade_start:
        segment = bar_net[trade_start:]
        trade_ret = float(np.prod(1.0 + segment) - 1.0)
        rows.append(
            {
                "entry_index": int(trade_start),
                "entry_day": normalize_day(days[trade_start]),
                "side": int(current_side),
                "total_return": trade_ret,
                "label": int(trade_ret > 0.0),
                "features": build_entry_feature_row(context, trace, trade_start),
                "feature_names": feature_names,
            }
        )
    return rows


def train_pair_meta_gate(
    *,
    pair: str,
    trade_rows: list[dict[str, Any]],
) -> tuple[TradeMetaGate, dict[str, Any], np.ndarray]:
    feature_names = build_entry_feature_names()
    rows = sorted(trade_rows, key=lambda row: row["entry_day"])
    x = np.asarray([row["features"] for row in rows], dtype="float64")
    y = np.asarray([row["label"] for row in rows], dtype="float64")
    entry_days = pd.DatetimeIndex([normalize_day(row["entry_day"]) for row in rows])
    train_end = max(int(len(rows) * 0.60), 12)
    val_end = max(int(len(rows) * 0.80), train_end + 6)

    x_train = x[:train_end]
    y_train = y[:train_end]
    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]
    val_returns = np.asarray([float(row["total_return"]) for row in rows[train_end:val_end]], dtype="float64")
    baseline_validation_win_rate = float(np.mean(y_val)) if len(y_val) else 0.0
    baseline_total_return = float(np.prod(1.0 + val_returns) - 1.0) if len(val_returns) else 0.0

    best: dict[str, Any] | None = None
    best_relaxed: dict[str, Any] | None = None
    for reg in (0.01, 0.05, 0.10, 0.20, 0.50, 1.0):
        mean, scale, weights, bias = fit_balanced_logistic(x_train, y_train, float(reg))
        stub = TradeMetaGate(
            feature_names=feature_names,
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
            threshold=0.5,
            train_end=str(entry_days[train_end - 1].date()),
            validation_end=str(entry_days[val_end - 1].date()),
        )
        probs = stub.predict_proba(x_val)
        for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            keep = probs >= threshold
            keep_rate = float(np.mean(keep)) if len(keep) else 0.0
            kept_win_rate = float(np.mean(y_val[keep])) if np.any(keep) else 0.0
            kept_total_return = float(np.prod(1.0 + val_returns[keep]) - 1.0) if np.any(keep) else 0.0
            total_return_retention = (
                kept_total_return / baseline_total_return if baseline_total_return > 1e-12 else 1.0
            )
            score = (
                kept_total_return * 200.0
                + kept_win_rate * 40.0
                - abs(keep_rate - 0.65) * 10.0
            )
            candidate = {
                "reg": float(reg),
                "threshold": float(threshold),
                "keep_rate": keep_rate,
                "win_rate": kept_win_rate,
                "total_return": kept_total_return,
                "baseline_total_return": baseline_total_return,
                "baseline_validation_win_rate": baseline_validation_win_rate,
                "total_return_retention": float(total_return_retention),
                "score": float(score),
                "mean": mean,
                "scale": scale,
                "weights": weights,
                "bias": bias,
            }
            relaxed_valid = keep_rate >= 0.35
            strict_valid = (
                relaxed_valid
                and kept_win_rate >= max(0.0, baseline_validation_win_rate - 0.02)
                and total_return_retention >= 0.85
            )
            if relaxed_valid and (best_relaxed is None or float(candidate["score"]) > float(best_relaxed["score"])):
                best_relaxed = candidate
            if strict_valid and (best is None or float(candidate["score"]) > float(best["score"])):
                best = candidate
    if best is None:
        best = best_relaxed
    assert best is not None

    gate = TradeMetaGate(
        feature_names=feature_names,
        mean=np.asarray(best["mean"], dtype="float64"),
        scale=np.asarray(best["scale"], dtype="float64"),
        weights=np.asarray(best["weights"], dtype="float64"),
        bias=float(best["bias"]),
        threshold=float(best["threshold"]),
        train_end=str(entry_days[train_end - 1].date()),
        validation_end=str(entry_days[val_end - 1].date()),
    )

    diagnostics = {
        "pair": pair,
        "trade_count": int(len(rows)),
        "train_trades": int(train_end),
        "validation_trades": int(val_end - train_end),
        "test_trades": int(len(rows) - val_end),
        "selected_reg": float(best["reg"]),
        "selected_threshold": float(best["threshold"]),
        "selected_validation_keep_rate": float(best["keep_rate"]),
        "selected_validation_win_rate": float(best["win_rate"]),
        "selected_validation_total_return": float(best["total_return"]),
        "baseline_validation_total_return": float(best["baseline_total_return"]),
    }
    return gate, diagnostics, entry_days


def build_entry_keep_flags(
    *,
    context: dict[str, Any],
    baseline_replay: dict[str, Any],
    gate: TradeMetaGate,
) -> np.ndarray:
    trace = baseline_replay["trace"]
    target = np.asarray(trace["target_weight"], dtype="float64")
    flags = np.ones(len(target), dtype=bool)
    current_side = 0
    for i in range(len(target)):
        weight = float(target[i])
        side = 0
        if weight > 1e-12:
            side = 1
        elif weight < -1e-12:
            side = -1
        new_entry = current_side == 0 and side != 0
        reversal = current_side != 0 and side != current_side and side != 0
        if new_entry or reversal:
            x = np.asarray([build_entry_feature_row(context, trace, i)], dtype="float64")
            prob = float(gate.predict_proba(x)[0])
            flags[i] = prob >= gate.threshold
        if side != current_side:
            current_side = side
    return flags


def evaluate_windows(
    *,
    candidate: dict[str, Any],
    pairs: tuple[str, ...],
    window_cache: dict[str, Any],
    library_lookup: dict[str, Any],
    pair_entry_keep_flags: dict[str, dict[str, np.ndarray]] | None = None,
    pair_return_trace: bool = False,
) -> dict[str, Any]:
    windows: dict[str, Any] = {}
    for label, start, end in SEARCH_WINDOWS:
        per_pair: dict[str, dict[str, Any]] = {}
        for pair in pairs:
            context = window_cache[label]["pairs"][pair]["fast_context"]
            replay = replay_pair(
                candidate=candidate,
                pair=pair,
                context=context,
                library_lookup=library_lookup,
                entry_keep_flags=None if pair_entry_keep_flags is None else pair_entry_keep_flags[pair][label],
                return_trace=pair_return_trace,
            )
            payload = {
                "total_return": replay["total_return"],
                "n_trades": replay["n_trades"],
                "sharpe": replay["sharpe"],
                "max_drawdown": replay["max_drawdown"],
                "final_equity": replay["final_equity"],
                "daily_metrics": {
                    "avg_daily_return": replay["avg_daily_return"],
                    "daily_target_hit_rate": replay["daily_target_hit_rate"],
                    "daily_win_rate": replay["daily_win_rate"],
                    "worst_day": replay["worst_day"],
                    "best_day": replay["best_day"],
                },
                "daily_win_rate": replay["daily_win_rate"],
                "avg_daily_return": replay["avg_daily_return"],
            }
            if pair_return_trace:
                payload["trace"] = replay.get("trace") or {}
            per_pair[pair] = payload
        windows[label] = {
            "start": start,
            "end": end,
            "per_pair": per_pair,
            "aggregate": aggregate_metrics(per_pair),
        }
    return windows


def main() -> None:
    args = parse_args()
    summary, bundle, pairs, _, window_cache, library_lookup = build_window_cache(
        summary_path=Path(args.summary),
        base_summary=Path(args.base_summary),
        model_path=Path(args.model),
    )
    candidate = summary["selected_candidate"]
    baseline_windows = evaluate_windows(
        candidate=candidate,
        pairs=pairs,
        window_cache=window_cache,
        library_lookup=library_lookup,
        pair_return_trace=False,
    )
    full_contexts = {pair: window_cache["full_4y"]["pairs"][pair]["fast_context"] for pair in pairs}
    baseline_full = {
        pair: replay_pair(
            candidate=candidate,
            pair=pair,
            context=full_contexts[pair],
            library_lookup=library_lookup,
            return_trace=True,
        )
        for pair in pairs
    }

    gates: dict[str, TradeMetaGate] = {}
    diagnostics: dict[str, Any] = {}
    pair_entry_keep_flags: dict[str, dict[str, np.ndarray]] = {}
    for pair in pairs:
        trade_rows = extract_trade_rows(full_contexts[pair], baseline_full[pair])
        gate, diag, _ = train_pair_meta_gate(pair=pair, trade_rows=trade_rows)
        gates[pair] = gate
        diagnostics[pair] = diag
        pair_entry_keep_flags[pair] = {}
        for label, _, _ in SEARCH_WINDOWS:
            context = window_cache[label]["pairs"][pair]["fast_context"]
            baseline_replay = replay_pair(
                candidate=candidate,
                pair=pair,
                context=context,
                library_lookup=library_lookup,
                return_trace=True,
            )
            pair_entry_keep_flags[pair][label] = build_entry_keep_flags(
                context=context,
                baseline_replay=baseline_replay,
                gate=gate,
            )

    gated_windows = evaluate_windows(
        candidate=candidate,
        pairs=pairs,
        window_cache=window_cache,
        library_lookup=library_lookup,
        pair_entry_keep_flags=pair_entry_keep_flags,
        pair_return_trace=False,
    )
    comparison = compare_to_baseline(gated_windows, baseline_windows)
    payload = {
        "strategy": {
            "summary": str(args.summary),
            "base_summary": str(args.base_summary),
            "model": str(args.model),
            "method": "trade_meta_label",
        },
        "gates": gates,
        "gate_diagnostics": diagnostics,
        "baseline_windows": baseline_windows,
        "gated_windows": gated_windows,
        "compare_to_main": comparison,
    }
    write_json(args.summary_out, payload)
    print(args.summary_out)


if __name__ == "__main__":
    main()
