#!/usr/bin/env python3
"""Search risk overlays that reduce GP drawdown using live-like replay."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dill
import numpy as np
import pandas as pd

import gp_crypto_evolution as gp


BARS_PER_DAY = gp.periods_per_day(gp.TIMEFRAME)
BAR_FACTOR = math.sqrt(365.25 * 24.0 * 60.0 / 5.0)


@dataclass(frozen=True)
class OverlayParams:
    signal_span: int
    rebalance_bars: int
    regime_threshold: float
    breadth_threshold: float
    target_vol_ann: float
    gross_cap: float
    kill_switch_pct: float
    cooldown_days: int
    vol_window_bars: int = 12 * 24 * 3
    regime_fast_days: int = 3
    regime_slow_days: int = 14
    breadth_lookback_days: int = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search drawdown overlays for a GP candidate with live-like replay.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--recent-start", default="2025-10-06")
    parser.add_argument("--recent-end", default="2026-04-06")
    parser.add_argument("--full-start", default="2022-04-06")
    parser.add_argument("--full-end", default="2026-04-06")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--summary-out",
        default=str(gp.MODELS_DIR / "gp_drawdown_overlay_search_summary.json"),
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def load_model(path: Path):
    with open(path, "rb") as f:
        payload = dill.load(f)
    if isinstance(payload, dict) and "tree" in payload:
        return payload["tree"], payload
    return payload, None


def build_overlay_inputs(df: pd.DataFrame, pair: str = gp.PRIMARY_PAIR) -> dict[str, pd.Series]:
    close = pd.concat(
        [df[f"{asset}_close"].rename(asset) for asset in gp.PAIRS],
        axis=1,
    ).sort_index()
    daily_close = close.resample("1D").last().dropna()
    btc_regime = (
        0.60 * daily_close[pair].pct_change(3)
        + 0.40 * daily_close[pair].pct_change(14)
    )
    breadth = (daily_close.pct_change(3) > 0.0).mean(axis=1)
    bar_ret = close[pair].pct_change()
    vol_ann = bar_ret.rolling(12 * 24 * 3).std() * BAR_FACTOR

    return {
        "btc_regime_daily": btc_regime,
        "breadth_daily": breadth,
        "vol_ann_bar": vol_ann,
    }


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    daily = result["daily_metrics"]
    return {
        "avg_daily_return": float(daily["avg_daily_return"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "sharpe": float(result["sharpe"]),
        "daily_target_hit_rate": float(daily["daily_target_hit_rate"]),
        "daily_win_rate": float(daily["daily_win_rate"]),
        "worst_day": float(daily["worst_day"]),
        "best_day": float(daily["best_day"]),
        "n_trades": int(result["n_trades"]),
    }


def overlay_score(metrics: dict[str, Any]) -> float:
    avg_daily = float(metrics["avg_daily_return"])
    total_return = float(metrics["total_return"])
    max_dd = abs(float(metrics["max_drawdown"]))
    sharpe = float(metrics["sharpe"])
    target_hit = float(metrics["daily_target_hit_rate"])
    worst_day = abs(min(float(metrics["worst_day"]), 0.0))

    score = 0.0
    score += max_dd * 18000.0
    score += worst_day * 14000.0
    score += max(0.0, 0.002 - avg_daily) * 280000.0
    score += max(0.0, -total_return) * 90000.0
    score -= avg_daily * 120000.0
    score -= total_return * 12000.0
    score -= sharpe * 120.0
    score -= target_hit * 2000.0
    return float(score)


def replay_with_overlay(
    df: pd.DataFrame,
    raw_signal_pct: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    params: OverlayParams,
    pair: str = gp.PRIMARY_PAIR,
    initial_cash: float = gp.INITIAL_CASH,
    commission: float = gp.COMMISSION_PCT,
    dead_band: float = gp.NO_TRADE_BAND,
) -> dict[str, Any]:
    smooth_signal = raw_signal_pct.ewm(span=params.signal_span, adjust=False).mean()
    close = df[f"{pair}_close"].to_numpy(dtype="float64")
    idx = pd.DatetimeIndex(df.index)
    day_index = idx.normalize()

    btc_regime = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    breadth = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    vol_ann = overlay_inputs["vol_ann_bar"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype="float64")

    equity = float(initial_cash)
    peak_equity = float(initial_cash)
    current_weight = 0.0
    cooldown_bars_left = 0
    net_ret: list[float] = []
    equity_curve = [float(initial_cash)]
    n_trades = 0

    for i in range(len(df) - 1):
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = float(np.clip(smooth_signal.iloc[i], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(btc_regime[i])
        breadth_score = float(breadth[i])

        long_ok = regime_score >= params.regime_threshold and breadth_score >= params.breadth_threshold
        short_ok = regime_score <= -params.regime_threshold and breadth_score <= (1.0 - params.breadth_threshold)
        if requested_weight > 0.0 and not long_ok:
            requested_weight = 0.0
        elif requested_weight < 0.0 and not short_ok:
            requested_weight = 0.0

        bar_vol_ann = float(vol_ann[i])
        if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = min(params.target_vol_ann / bar_vol_ann, params.gross_cap / max(abs(requested_weight), 1e-8))
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
    result = {
        "total_return": float(equity / initial_cash - 1.0),
        "n_trades": int(n_trades),
        "sharpe": sharpe,
        "max_drawdown": float(np.min(equity_curve_arr / peak - 1.0)),
        "final_equity": float(equity),
        "equity_curve": equity_curve_arr,
        "net_ret": net_ret_arr,
        "daily_metrics": gp.compute_daily_metrics(net_ret_arr),
    }
    return result


def iter_params() -> list[OverlayParams]:
    combos = []
    for signal_span, rebalance_bars, regime_threshold, breadth_threshold, target_vol_ann, gross_cap, kill_switch_pct, cooldown_days in itertools.product(
        [1, 12, 36],
        [1, 3, 12],
        [0.0, 0.01, 0.02],
        [0.50, 0.65],
        [0.40, 0.60, 0.80],
        [0.75, 1.00, 1.50],
        [0.08, 0.12, 0.16],
        [1, 3, 5],
    ):
        combos.append(
            OverlayParams(
                signal_span=signal_span,
                rebalance_bars=rebalance_bars,
                regime_threshold=regime_threshold,
                breadth_threshold=breadth_threshold,
                target_vol_ann=target_vol_ann,
                gross_cap=gross_cap,
                kill_switch_pct=kill_switch_pct,
                cooldown_days=cooldown_days,
            )
        )
    return combos


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    gp.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model, payload = load_model(model_path)
    compiled = gp.toolbox.compile(expr=model)

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
    all_params = iter_params()
    print(f"Overlay search combos: {len(all_params)}")
    for idx, params in enumerate(all_params, start=1):
        recent_result = replay_with_overlay(recent_df, recent_signal, recent_inputs, params)
        recent_metrics = summarize_result(recent_result)
        recent_metrics["score"] = overlay_score(recent_metrics)
        candidate = {
            "params": asdict(params),
            "recent": recent_metrics,
        }
        candidates.append(candidate)
        if idx % 100 == 0 or idx == len(all_params):
            best_so_far = min(candidates, key=lambda item: item["recent"]["score"])
            print(
                f"[{idx}/{len(all_params)}] best_recent"
                f" avg_daily={best_so_far['recent']['avg_daily_return']*100:+.3f}%"
                f" total={best_so_far['recent']['total_return']*100:+.2f}%"
                f" mdd={best_so_far['recent']['max_drawdown']*100:.2f}%"
            )

    candidates.sort(key=lambda item: item["recent"]["score"])
    top_recent = candidates[: args.top_k]

    for item in top_recent:
        params = OverlayParams(**item["params"])
        full_result = replay_with_overlay(full_df, full_signal, full_inputs, params)
        item["full"] = summarize_result(full_result)
        item["full"]["score"] = overlay_score(item["full"])

    # Also keep a conservative leaderboard sorted by MDD among profitable recent configs.
    conservative = [
        item for item in candidates
        if item["recent"]["total_return"] > 0.0 and item["recent"]["avg_daily_return"] > 0.0
    ]
    conservative.sort(key=lambda item: (abs(item["recent"]["max_drawdown"]), -item["recent"]["avg_daily_return"]))

    summary = {
        "model_path": str(model_path),
        "payload_meta": {
            "algorithm": payload.get("algorithm") if isinstance(payload, dict) else None,
            "window_start": payload.get("window_start") if isinstance(payload, dict) else None,
            "window_end": payload.get("window_end") if isinstance(payload, dict) else None,
        },
        "search_space": {
            "combos": len(all_params),
        },
        "top_recent_score": top_recent,
        "top_recent_low_mdd": conservative[: args.top_k],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "All evaluations use live-like sequential replay, not the fast vectorized shortcut.",
    }

    out_path = Path(args.summary_out)
    out_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    print(f"Saved summary: {out_path}")
    if top_recent:
        best = top_recent[0]
        print(json.dumps(json_safe(best), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
