#!/usr/bin/env python3
"""Search regime-conditioned overlay mixtures on top of a GP signal."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from search_gp_drawdown_overlay import (
    BARS_PER_DAY,
    BAR_FACTOR,
    OverlayParams,
    build_overlay_inputs,
    json_safe,
    load_model,
    summarize_result,
)


ROUTE_BREADTH_THRESHOLDS = (0.50, 0.65)
STATE_LABELS = {
    0: "down_narrow",
    1: "down_broad",
    2: "up_narrow",
    3: "up_broad",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search regime-mixture overlays for a GP signal.",
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--overlay-summary",
        default=str(gp.MODELS_DIR / "gp_drawdown_overlay_search_summary.json"),
    )
    parser.add_argument("--recent-start", default="2025-10-06")
    parser.add_argument("--recent-end", default="2026-04-06")
    parser.add_argument("--full-start", default="2022-04-06")
    parser.add_argument("--full-end", default="2026-04-06")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--max-library", type=int, default=8)
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
    )
    return parser.parse_args()


def load_overlay_library(path: Path, max_library: int) -> list[OverlayParams]:
    if not path.exists():
        return [
            OverlayParams(1, 3, 0.0, 0.50, 0.80, 1.50, 0.16, 1),
            OverlayParams(1, 3, 0.01, 0.65, 0.40, 0.75, 0.08, 1),
            OverlayParams(12, 1, 0.0, 0.65, 0.80, 1.50, 0.16, 1),
            OverlayParams(1, 3, 0.0, 0.50, 0.40, 0.75, 0.08, 1),
        ][:max_library]

    raw = json.loads(path.read_text())
    seen: dict[tuple[Any, ...], OverlayParams] = {}
    for group_name in ("top_recent_score", "top_recent_low_mdd"):
        for item in raw.get(group_name, []):
            params = OverlayParams(**item["params"])
            key = tuple(asdict(params).items())
            if key not in seen:
                seen[key] = params

    archetypes: dict[tuple[Any, ...], OverlayParams] = {}
    for params in seen.values():
        key = (
            params.signal_span,
            params.rebalance_bars,
            params.regime_threshold,
            params.breadth_threshold,
            params.target_vol_ann,
            params.gross_cap,
            params.kill_switch_pct,
        )
        current = archetypes.get(key)
        if current is None or params.cooldown_days < current.cooldown_days:
            archetypes[key] = params
    return list(archetypes.values())[:max_library]


def build_route_bucket_codes(
    index: pd.DatetimeIndex,
    overlay_inputs: dict[str, pd.Series],
    breadth_threshold: float,
) -> np.ndarray:
    day_index = index.normalize()
    regime_daily = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0)
    breadth_daily = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0)
    is_up = (regime_daily >= 0.0).astype(np.int8)
    is_broad = (breadth_daily >= breadth_threshold).astype(np.int8)
    return (is_up * 2 + is_broad).to_numpy(dtype="int8")


def high_target_score(metrics: dict[str, Any], target_daily: float = 0.006) -> float:
    avg_daily = float(metrics["avg_daily_return"])
    total_return = float(metrics["total_return"])
    max_dd = abs(float(metrics["max_drawdown"]))
    sharpe = float(metrics["sharpe"])
    target_hit = float(metrics["daily_target_hit_rate"])
    worst_day = abs(min(float(metrics["worst_day"]), 0.0))

    score = 0.0
    score += max_dd * 18000.0
    score += worst_day * 15000.0
    score += max(0.0, target_daily - avg_daily) * 340000.0
    score += max(0.0, -total_return) * 100000.0
    score -= avg_daily * 170000.0
    score -= total_return * 12000.0
    score -= sharpe * 140.0
    score -= target_hit * 2500.0
    return float(score)


def replay_regime_mixture(
    df: pd.DataFrame,
    raw_signal_pct: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    library: list[OverlayParams],
    mapping: tuple[int, int, int, int],
    route_breadth_threshold: float,
    pair: str = gp.PRIMARY_PAIR,
    initial_cash: float = gp.INITIAL_CASH,
    commission: float = gp.COMMISSION_PCT,
    dead_band: float = gp.NO_TRADE_BAND,
) -> dict[str, Any]:
    close = df[f"{pair}_close"].to_numpy(dtype="float64")
    idx = pd.DatetimeIndex(df.index)
    day_index = idx.normalize()
    bucket_codes = build_route_bucket_codes(idx, overlay_inputs, route_breadth_threshold)

    regime = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    breadth = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    vol_ann = overlay_inputs["vol_ann_bar"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype="float64")

    spans = sorted({params.signal_span for params in library})
    smooth_signals = {
        span: raw_signal_pct.ewm(span=span, adjust=False).mean().to_numpy(dtype="float64")
        for span in spans
    }

    equity = float(initial_cash)
    peak_equity = float(initial_cash)
    current_weight = 0.0
    cooldown_bars_left = 0
    net_ret: list[float] = []
    equity_curve = [float(initial_cash)]
    n_trades = 0

    for i in range(len(df) - 1):
        active_idx = int(mapping[int(bucket_codes[i])])
        params = library[active_idx]

        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = float(np.clip(smooth_signals[params.signal_span][i], -500.0, 500.0))
        requested_weight = signal_pct / 100.0

        regime_score = float(regime[i])
        breadth_score = float(breadth[i])
        long_ok = (
            regime_score >= params.regime_threshold
            and breadth_score >= params.breadth_threshold
        )
        short_ok = (
            regime_score <= -params.regime_threshold
            and breadth_score <= (1.0 - params.breadth_threshold)
        )
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = float(vol_ann[i])
        if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = min(
                params.target_vol_ann / bar_vol_ann,
                params.gross_cap / max(abs(requested_weight), 1e-8),
            )
            requested_weight *= float(vol_scale)
        requested_weight = float(np.clip(requested_weight, -params.gross_cap, params.gross_cap))

        drawdown = equity / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -params.kill_switch_pct and cooldown_bars_left == 0:
            cooldown_bars_left = params.cooldown_days * BARS_PER_DAY

        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif i % params.rebalance_bars == 0:
            target_weight = requested_weight

        if abs(target_weight - current_weight) < dead_band / 100.0:
            target_weight = current_weight

        turnover = abs(target_weight - current_weight)
        if turnover > 0.001:
            n_trades += 1

        price_ret = float(close[i + 1] / close[i] - 1.0)
        bar_net = target_weight * price_ret - turnover * commission * 2
        equity *= (1.0 + bar_net)
        peak_equity = max(peak_equity, equity)
        current_weight = target_weight
        net_ret.append(bar_net)
        equity_curve.append(float(equity))

    net_ret_arr = np.asarray(net_ret, dtype="float64")
    equity_curve_arr = np.asarray(equity_curve, dtype="float64")
    if len(net_ret_arr) > 1 and np.std(net_ret_arr) > 1e-12:
        sharpe = float(np.mean(net_ret_arr) / np.std(net_ret_arr) * BAR_FACTOR)
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve_arr)
    return {
        "total_return": float(equity / initial_cash - 1.0),
        "n_trades": int(n_trades),
        "sharpe": sharpe,
        "max_drawdown": float(np.min(equity_curve_arr / peak - 1.0)),
        "final_equity": float(equity),
        "equity_curve": equity_curve_arr,
        "net_ret": net_ret_arr,
        "daily_metrics": gp.compute_daily_metrics(net_ret_arr),
    }


def mapping_to_labels(mapping: tuple[int, int, int, int], library: list[OverlayParams]) -> dict[str, dict[str, Any]]:
    labeled = {}
    for state_code, lib_idx in enumerate(mapping):
        labeled[STATE_LABELS[state_code]] = asdict(library[int(lib_idx)])
    return labeled


def main() -> None:
    args = parse_args()
    gp.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    model, payload = load_model(model_path)
    compiled = gp.toolbox.compile(expr=model)

    library = load_overlay_library(Path(args.overlay_summary), args.max_library)
    if not library:
        raise RuntimeError("No overlay parameter library available")

    recent_df = gp.load_all_pairs(
        start=args.recent_start,
        end=args.recent_end,
        refresh_cache=False,
    )
    full_df = gp.load_all_pairs(
        start=args.full_start,
        end=args.full_end,
        refresh_cache=False,
    )

    recent_signal = pd.Series(
        compiled(*gp.get_feature_arrays(recent_df, gp.PRIMARY_PAIR)),
        index=recent_df.index,
        dtype="float64",
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    full_signal = pd.Series(
        compiled(*gp.get_feature_arrays(full_df, gp.PRIMARY_PAIR)),
        index=full_df.index,
        dtype="float64",
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    recent_inputs = build_overlay_inputs(recent_df)
    full_inputs = build_overlay_inputs(full_df)

    candidates: list[dict[str, Any]] = []
    all_mappings = list(itertools.product(range(len(library)), repeat=4))
    total = len(all_mappings) * len(ROUTE_BREADTH_THRESHOLDS)
    seen = 0
    print(f"Regime-mixture library: {len(library)} overlays")
    print(f"Regime-mixture combinations: {total}")

    for route_breadth_threshold in ROUTE_BREADTH_THRESHOLDS:
        for mapping in all_mappings:
            result = replay_regime_mixture(
                recent_df,
                recent_signal,
                recent_inputs,
                library=library,
                mapping=mapping,
                route_breadth_threshold=route_breadth_threshold,
            )
            metrics = summarize_result(result)
            metrics["score"] = high_target_score(metrics)
            candidates.append(
                {
                    "route_breadth_threshold": float(route_breadth_threshold),
                    "mapping_indices": list(mapping),
                    "mapping": mapping_to_labels(mapping, library),
                    "recent": metrics,
                }
            )
            seen += 1
            if seen % 100 == 0 or seen == total:
                best = min(candidates, key=lambda item: item["recent"]["score"])
                print(
                    f"[{seen}/{total}] best_recent"
                    f" avg_daily={best['recent']['avg_daily_return']*100:+.3f}%"
                    f" total={best['recent']['total_return']*100:+.2f}%"
                    f" mdd={best['recent']['max_drawdown']*100:.2f}%"
                )

    candidates.sort(key=lambda item: item["recent"]["score"])
    top_recent = candidates[: args.top_k]

    recent_mid = recent_df.index[len(recent_df) // 2]
    for item in top_recent:
        mapping = tuple(int(v) for v in item["mapping_indices"])
        route_threshold = float(item["route_breadth_threshold"])
        full_result = replay_regime_mixture(
            full_df,
            full_signal,
            full_inputs,
            library=library,
            mapping=mapping,
            route_breadth_threshold=route_threshold,
        )
        first_half_result = replay_regime_mixture(
            recent_df.loc[:recent_mid].copy(),
            recent_signal.loc[:recent_mid].copy(),
            build_overlay_inputs(recent_df.loc[:recent_mid].copy()),
            library=library,
            mapping=mapping,
            route_breadth_threshold=route_threshold,
        )
        second_half_result = replay_regime_mixture(
            recent_df.loc[recent_mid:].copy(),
            recent_signal.loc[recent_mid:].copy(),
            build_overlay_inputs(recent_df.loc[recent_mid:].copy()),
            library=library,
            mapping=mapping,
            route_breadth_threshold=route_threshold,
        )
        item["full"] = summarize_result(full_result)
        item["full"]["score"] = high_target_score(item["full"])
        item["recent_first_half"] = summarize_result(first_half_result)
        item["recent_second_half"] = summarize_result(second_half_result)

    summary = {
        "model_path": str(model_path),
        "payload_meta": {
            "algorithm": payload.get("algorithm") if isinstance(payload, dict) else None,
            "window_start": payload.get("window_start") if isinstance(payload, dict) else None,
            "window_end": payload.get("window_end") if isinstance(payload, dict) else None,
        },
        "library_size": len(library),
        "overlay_library": [asdict(params) for params in library],
        "search_space": {
            "route_breadth_thresholds": list(ROUTE_BREADTH_THRESHOLDS),
            "state_labels": STATE_LABELS,
            "mapping_count": len(all_mappings),
            "total_combinations": total,
        },
        "top_recent_score": top_recent,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Regime mixture search uses one GP alpha signal with state-dependent overlay genes. "
            "All evaluations use cache-only sequential replay."
        ),
    }
    out_path = Path(args.summary_out)
    out_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print(f"Saved summary: {out_path}")
    if top_recent:
        print(json.dumps(json_safe(top_recent[0]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
