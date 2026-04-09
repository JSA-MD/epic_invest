#!/usr/bin/env python3
"""Realistic live-like replay for the regime-mixture GP overlay strategy."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

import gp_crypto_evolution as gp
from search_gp_drawdown_overlay import OverlayParams, build_overlay_inputs


API_BASE = "https://fapi.binance.com"
DEFAULT_WINDOWS = (
    ("recent_2m", "2026-02-06", "2026-04-06"),
    ("recent_6m", "2025-10-06", "2026-04-06"),
    ("full_4y", "2022-04-06", "2026-04-06"),
)


@dataclass(frozen=True)
class Candidate:
    route_breadth_threshold: float
    mapping_indices: tuple[int, int, int, int]
    mapping: dict[str, dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the regime-mixture GP strategy with live-like execution assumptions.",
    )
    parser.add_argument(
        "--summary",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"),
    )
    parser.add_argument(
        "--model",
        default=str(gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"),
    )
    parser.add_argument(
        "--funding-cache",
        default=str(gp.DATA_DIR / "BTCUSDT_funding_2022-04-06_2026-04-06.csv"),
    )
    parser.add_argument("--mapping", default=None, help="Optional mapping indices like 0,7,0,5")
    parser.add_argument("--route-breadth-threshold", type=float, default=None)
    parser.add_argument("--fee-rate", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--amount-step", type=float, default=0.001)
    parser.add_argument("--min-qty", type=float, default=0.001)
    parser.add_argument("--dead-band-pp", type=float, default=gp.NO_TRADE_BAND)
    parser.add_argument("--delay-bars", type=int, default=1)
    parser.add_argument("--fetch-funding", action="store_true")
    parser.add_argument(
        "--report-out",
        default=str(gp.MODELS_DIR / "gp_regime_mixture_realistic_report.json"),
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


def http_get_json(path: str, params: dict[str, Any]) -> Any:
    url = f"{API_BASE}{path}?{urllib.parse.urlencode(params)}"
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        cp = subprocess.run(
            ["curl", "-sS", "--max-time", "20", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(cp.stdout)


def fetch_funding_rates(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    rows_all: list[dict[str, Any]] = []
    cur = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1000,
        }
        rows = http_get_json("/fapi/v1/fundingRate", params)
        if not rows:
            break
        rows_all.extend(rows)
        last_t = int(rows[-1]["fundingTime"])
        nxt = last_t + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)

    if not rows_all:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    out = pd.DataFrame(rows_all)
    out["fundingTime"] = pd.to_datetime(out["fundingTime"].astype(np.int64), unit="ms", utc=True)
    out["fundingRate"] = pd.to_numeric(out["fundingRate"], errors="coerce")
    out = out.dropna(subset=["fundingTime", "fundingRate"]).drop_duplicates(subset=["fundingTime"])
    return out.sort_values("fundingTime").reset_index(drop=True)


def load_or_fetch_funding(path: Path, fetch: bool) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, parse_dates=["fundingTime"])
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True)
        df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df.dropna(subset=["fundingTime", "fundingRate"]).sort_values("fundingTime").reset_index(drop=True)
    if not fetch:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    df = fetch_funding_rates(
        "BTCUSDT",
        datetime(2022, 4, 6, tzinfo=timezone.utc),
        datetime(2026, 4, 7, tzinfo=timezone.utc),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_model(path: Path):
    import dill

    with open(path, "rb") as f:
        payload = dill.load(f)
    if isinstance(payload, dict) and "tree" in payload:
        return payload["tree"], payload
    return payload, None


def resolve_candidate(summary_path: Path, mapping_arg: str | None, route_threshold: float | None) -> tuple[Candidate, list[OverlayParams], dict[str, Any]]:
    raw = json.loads(summary_path.read_text())
    library = [OverlayParams(**item) for item in raw["overlay_library"]]
    if mapping_arg:
        mapping = tuple(int(x.strip()) for x in mapping_arg.split(","))
        if len(mapping) != 4:
            raise ValueError("--mapping must contain exactly four indices")
        chosen = None
        for item in raw["top_recent_score"]:
            if tuple(item["mapping_indices"]) == mapping:
                if route_threshold is None or float(item["route_breadth_threshold"]) == float(route_threshold):
                    chosen = item
                    break
        if chosen is None:
            raise ValueError("Requested mapping not found in summary")
    else:
        chosen = max(
            raw["top_recent_score"],
            key=lambda item: (
                float(item["full"]["total_return"]),
                float(item["full"]["avg_daily_return"]),
            ),
        )
    candidate = Candidate(
        route_breadth_threshold=float(chosen["route_breadth_threshold"]),
        mapping_indices=tuple(int(v) for v in chosen["mapping_indices"]),
        mapping=chosen["mapping"],
    )
    return candidate, library, raw


def quantize_amount(value: float, step: float, min_qty: float) -> float:
    sign = 1.0 if value >= 0.0 else -1.0
    raw = abs(float(value))
    if raw < min_qty:
        return 0.0
    precise = math.floor(raw / step + 1e-12) * step
    if precise < min_qty:
        return 0.0
    return sign * precise


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


def summarize(net_ret: np.ndarray, equity_curve: np.ndarray, n_trades: int, fee_paid: float, slippage_paid: float, funding_paid: float, funding_events: int) -> dict[str, Any]:
    final_equity = float(equity_curve[-1]) if len(equity_curve) else float(gp.INITIAL_CASH)
    total_return = final_equity / gp.INITIAL_CASH - 1.0
    if len(net_ret) > 1 and np.std(net_ret) > 1e-12:
        sharpe = float(np.mean(net_ret) / np.std(net_ret) * np.sqrt(365.25 * 24 * 60.0 / 5.0))
    else:
        sharpe = 0.0
    peak = np.maximum.accumulate(equity_curve)
    max_drawdown = float(np.min(equity_curve / peak - 1.0))
    daily = gp.compute_daily_metrics(net_ret)
    return {
        "avg_daily_return": float(daily["avg_daily_return"]),
        "total_return": float(total_return),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "daily_target_hit_rate": float(daily["daily_target_hit_rate"]),
        "daily_win_rate": float(daily["daily_win_rate"]),
        "worst_day": float(daily["worst_day"]),
        "best_day": float(daily["best_day"]),
        "n_trades": int(n_trades),
        "fee_paid": float(fee_paid),
        "slippage_paid": float(slippage_paid),
        "funding_paid": float(funding_paid),
        "funding_events": int(funding_events),
        "final_equity": final_equity,
    }


def replay_realistic(
    df: pd.DataFrame,
    raw_signal: pd.Series,
    overlay_inputs: dict[str, pd.Series],
    funding_df: pd.DataFrame,
    library: list[OverlayParams],
    candidate: Candidate,
    fee_rate: float,
    slippage: float,
    amount_step: float,
    min_qty: float,
    dead_band_pp: float,
    delay_bars: int,
) -> dict[str, Any]:
    idx = pd.DatetimeIndex(df.index)
    open_p = df[f"{gp.PRIMARY_PAIR}_open"].to_numpy(dtype="float64")
    close_p = df[f"{gp.PRIMARY_PAIR}_close"].to_numpy(dtype="float64")
    vol_ann = overlay_inputs["vol_ann_bar"].reindex(idx).ffill().bfill().fillna(0.0).to_numpy(dtype="float64")
    day_index = idx.normalize()
    regime = overlay_inputs["btc_regime_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    breadth = overlay_inputs["breadth_daily"].reindex(day_index, method="ffill").fillna(0.0).to_numpy(dtype="float64")
    bucket_codes = build_route_bucket_codes(idx, overlay_inputs, candidate.route_breadth_threshold)
    spans = sorted({params.signal_span for params in library})
    smooth_signals = {
        span: raw_signal.ewm(span=span, adjust=False).mean().to_numpy(dtype="float64")
        for span in spans
    }

    funding_map = {}
    if not funding_df.empty:
        for _, row in funding_df.iterrows():
            funding_map[pd.Timestamp(row["fundingTime"]).tz_convert("UTC")] = float(row["fundingRate"])

    cash = float(gp.INITIAL_CASH)
    qty = 0.0
    n_trades = 0
    fee_paid = 0.0
    slippage_paid = 0.0
    funding_paid = 0.0
    funding_events = 0
    net_ret: list[float] = []
    equity_curve = [float(gp.INITIAL_CASH)]
    peak_equity = float(gp.INITIAL_CASH)
    cooldown_bars_left = 0

    start_exec = max(1, delay_bars)
    end_exec = len(df) - 1
    for exec_idx in range(start_exec, end_exec):
        signal_idx = exec_idx - delay_bars
        ts_open = pd.Timestamp(idx[exec_idx])
        next_ts_open = pd.Timestamp(idx[exec_idx + 1])
        px_open = float(open_p[exec_idx])
        next_open = float(open_p[exec_idx + 1])
        prev_close = float(close_p[signal_idx])

        if qty != 0.0 and ts_open in funding_map:
            funding_rate = funding_map[ts_open]
            funding_cashflow = -qty * px_open * funding_rate
            cash += funding_cashflow
            funding_paid += funding_cashflow
            funding_events += 1

        equity_before = cash + qty * px_open
        if equity_before <= 1e-9:
            equity_before = 1e-9
        peak_equity = max(peak_equity, equity_before)

        active_idx = int(candidate.mapping_indices[int(bucket_codes[signal_idx])])
        params = library[active_idx]
        if cooldown_bars_left > 0:
            cooldown_bars_left -= 1

        signal_pct = float(np.clip(smooth_signals[params.signal_span][signal_idx], -500.0, 500.0))
        requested_weight = signal_pct / 100.0
        regime_score = float(regime[signal_idx])
        breadth_score = float(breadth[signal_idx])
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

        bar_vol_ann = float(vol_ann[signal_idx])
        if np.isfinite(bar_vol_ann) and bar_vol_ann > 1e-8 and abs(requested_weight) > 1e-12:
            vol_scale = min(
                params.target_vol_ann / bar_vol_ann,
                params.gross_cap / max(abs(requested_weight), 1e-8),
            )
            requested_weight *= float(vol_scale)
        requested_weight = float(np.clip(requested_weight, -params.gross_cap, params.gross_cap))

        drawdown = equity_before / max(peak_equity, 1e-8) - 1.0
        if drawdown <= -params.kill_switch_pct and cooldown_bars_left == 0:
            cooldown_bars_left = params.cooldown_days * gp.periods_per_day(gp.TIMEFRAME)

        current_weight = (qty * px_open / equity_before) if abs(equity_before) > 1e-9 else 0.0
        target_weight = current_weight
        if cooldown_bars_left > 0:
            target_weight = 0.0
        elif signal_idx % params.rebalance_bars == 0:
            target_weight = requested_weight

        if abs(target_weight - current_weight) < dead_band_pp / 100.0:
            target_weight = current_weight

        target_notional = equity_before * target_weight
        target_qty = quantize_amount(target_notional / prev_close if abs(prev_close) > 1e-12 else 0.0, amount_step, min_qty)
        diff_qty = quantize_amount(target_qty - qty, amount_step, min_qty)

        if abs(diff_qty) > 0.0:
            side = 1.0 if diff_qty > 0.0 else -1.0
            exec_price = px_open * (1.0 + slippage * side)
            trade_notional = diff_qty * exec_price
            fee = abs(diff_qty) * exec_price * fee_rate
            cash -= trade_notional
            cash -= fee
            qty += diff_qty
            n_trades += 1
            fee_paid += fee
            slippage_paid += abs(diff_qty) * px_open * slippage

        equity_after_interval = cash + qty * next_open
        interval_ret = equity_after_interval / equity_before - 1.0
        net_ret.append(float(interval_ret))
        equity_curve.append(float(equity_after_interval))

    return summarize(
        np.asarray(net_ret, dtype="float64"),
        np.asarray(equity_curve, dtype="float64"),
        n_trades=n_trades,
        fee_paid=fee_paid,
        slippage_paid=slippage_paid,
        funding_paid=funding_paid,
        funding_events=funding_events,
    )


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    model_path = Path(args.model)
    funding_cache_path = Path(args.funding_cache)

    candidate, library, summary_raw = resolve_candidate(
        summary_path,
        args.mapping,
        args.route_breadth_threshold,
    )
    model, _ = load_model(model_path)
    compiled = gp.toolbox.compile(expr=model)

    start_all = DEFAULT_WINDOWS[-1][1]
    end_all = DEFAULT_WINDOWS[-1][2]
    df_all = gp.load_all_pairs(start=start_all, end=end_all, refresh_cache=False)
    raw_signal_all = pd.Series(
        compiled(*gp.get_feature_arrays(df_all, gp.PRIMARY_PAIR)),
        index=df_all.index,
        dtype="float64",
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    overlay_inputs_all = build_overlay_inputs(df_all)
    funding_df = load_or_fetch_funding(funding_cache_path, fetch=args.fetch_funding)

    windows_report = {}
    for label, start, end in DEFAULT_WINDOWS:
        df = df_all.loc[start:end].copy()
        raw_signal = raw_signal_all.loc[start:end].copy()
        overlay_inputs = build_overlay_inputs(df)
        if funding_df.empty:
            funding_slice = funding_df
        else:
            funding_slice = funding_df[
                (funding_df["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                & (funding_df["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
            ].copy()
        windows_report[label] = replay_realistic(
            df,
            raw_signal,
            overlay_inputs,
            funding_slice,
            library=library,
            candidate=candidate,
            fee_rate=args.fee_rate,
            slippage=args.slippage,
            amount_step=args.amount_step,
            min_qty=args.min_qty,
            dead_band_pp=args.dead_band_pp,
            delay_bars=args.delay_bars,
        )
        windows_report[label]["start"] = start
        windows_report[label]["end"] = end
        windows_report[label]["bars"] = int(len(df))

    report = {
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "candidate": {
            "route_breadth_threshold": candidate.route_breadth_threshold,
            "mapping_indices": list(candidate.mapping_indices),
            "mapping": candidate.mapping,
        },
        "assumptions": {
            "execution_timing": "Use the just-closed 5m bar for signal, execute on the next 5m bar open.",
            "fee_rate_per_side": args.fee_rate,
            "slippage_per_side": args.slippage,
            "amount_step": args.amount_step,
            "min_qty": args.min_qty,
            "dead_band_pp": args.dead_band_pp,
            "delay_bars": args.delay_bars,
            "funding_cache_path": str(funding_cache_path),
            "funding_loaded": not funding_df.empty,
        },
        "windows": windows_report,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.report_out)
    out_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
