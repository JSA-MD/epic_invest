#!/usr/bin/env python3
"""Backtest for trend-following volatility breakout short strategy on Binance USDT-M."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from zoneinfo import ZoneInfo

API_BASE = "https://fapi.binance.com"
NY_TZ = ZoneInfo("America/New_York")


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


@dataclass
class Params:
    k: float = 0.5
    stop_loss_pct: float = 0.02
    trail_activate_pct: float = 0.03
    trail_retrace_pct: float = 0.01
    vol_mult: float = 1.5
    fee_rate: float = 0.0004
    slippage: float = 0.0002
    position_fraction: float = 0.10
    leverage: float = 3.0


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def interval_ms(interval: str) -> int:
    mapping = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}")
    return mapping[interval]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    data_all: list[list[Any]] = []
    cur = start_ms
    step_ms = interval_ms(interval)

    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1500,
        }
        data = http_get_json("/fapi/v1/klines", params)
        if not data:
            break

        data_all.extend(data)
        last_open = int(data[-1][0])
        nxt = last_open + step_ms
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)

    if not data_all:
        return pd.DataFrame()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_base",
        "taker_quote",
        "ignore",
    ]
    df = pd.DataFrame(data_all, columns=cols)
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_base",
        "taker_quote",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    cur = start_ms

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

        all_rows.extend(rows)
        last_t = int(rows[-1]["fundingTime"])
        nxt = last_t + 1
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)

    if not all_rows:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])

    out = pd.DataFrame(all_rows)
    out["fundingTime"] = pd.to_datetime(out["fundingTime"].astype(np.int64), unit="ms", utc=True)
    out["fundingRate"] = pd.to_numeric(out["fundingRate"], errors="coerce")
    out = out.dropna(subset=["fundingTime", "fundingRate"]).drop_duplicates(subset=["fundingTime"])
    out = out.sort_values("fundingTime").reset_index(drop=True)
    return out


def select_symbol(
    candidates: list[str],
    start_dt: datetime,
    lookback_days: int = 260,
    vol_window_days: int = 30,
) -> tuple[str, pd.DataFrame]:
    stats_rows: list[dict[str, Any]] = []
    start_ms = dt_to_ms(start_dt - timedelta(days=lookback_days))
    end_ms = dt_to_ms(start_dt)

    for sym in candidates:
        try:
            df = fetch_klines(sym, "1d", start_ms, end_ms)
        except Exception:
            continue
        if df.empty or len(df) < 220:
            continue

        close = df["close"]
        ema200 = close.ewm(span=200, adjust=False).mean()
        last_close = float(close.iloc[-1])
        last_ema200 = float(ema200.iloc[-1])
        above_ema = last_close > last_ema200
        avg_qv_30 = float(df["quote_volume"].tail(vol_window_days).mean())

        stats_rows.append(
            {
                "symbol": sym,
                "last_close": last_close,
                "ema200": last_ema200,
                "above_ema200": above_ema,
                "avg_quote_volume_30d": avg_qv_30,
            }
        )

    if not stats_rows:
        raise RuntimeError("No candidates had sufficient daily data for symbol selection.")

    stats = pd.DataFrame(stats_rows)
    stats = stats.sort_values("avg_quote_volume_30d", ascending=False).reset_index(drop=True)

    above = stats[stats["above_ema200"]]
    if not above.empty:
        selected = above.iloc[0]["symbol"]
    else:
        selected = stats.iloc[0]["symbol"]

    return str(selected), stats


def prepare_hourly_features(df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()
    df = df.set_index("open_time")

    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    ny_idx = df.index.tz_convert(NY_TZ)
    df["ny_date"] = ny_idx.date
    df["ny_hour"] = ny_idx.hour

    daily = df.groupby("ny_date").agg(day_open=("open", "first"), day_high=("high", "max"), day_low=("low", "min"))
    daily["prev_range"] = (daily["day_high"] - daily["day_low"]).shift(1)

    df = df.join(daily[["day_open", "prev_range"]], on="ny_date")
    return df


def run_backtest(
    df: pd.DataFrame,
    funding_df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    params: Params,
    ny_entry_hour: int = 10,
    initial_equity: float = 100_000.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"error": "empty dataframe"}

    start_ms = dt_to_ms(start_dt)
    end_ms = dt_to_ms(end_dt)

    work = df.copy()
    work["breakout"] = work["day_open"] - (work["prev_range"] * params.k)

    close = work["close"]
    breakout = work["breakout"]

    cross_down = (close < breakout) & (close.shift(1) >= breakout.shift(1))
    entry_signal = (
        (work["close"] < work["ema50"])
        & cross_down
        & (work["volume"] >= work["vol_ma20"] * params.vol_mult)
        & (work["ny_hour"] >= ny_entry_hour)
        & work["breakout"].notna()
        & work["vol_ma20"].notna()
    )

    open_ms = (work.index.view("int64") // 1_000_000).astype(np.int64)
    open_p = work["open"].to_numpy(dtype=float)
    high_p = work["high"].to_numpy(dtype=float)
    low_p = work["low"].to_numpy(dtype=float)
    close_p = work["close"].to_numpy(dtype=float)
    signal = entry_signal.to_numpy(dtype=bool)

    funding_map: dict[int, float] = {}
    if not funding_df.empty:
        for _, r in funding_df.iterrows():
            t_ms = int(r["fundingTime"].value // 1_000_000)
            funding_map[t_ms] = float(r["fundingRate"])

    equity = float(initial_equity)
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []

    pending_entry = False
    pending_signal_time: int | None = None
    pos: dict[str, Any] | None = None

    for i in range(1, len(work)):
        t_ms = int(open_ms[i])

        # Funding is applied if position was already open at bar open.
        if pos is not None:
            fr = funding_map.get(t_ms)
            if fr is not None:
                funding_pnl = pos["qty"] * open_p[i] * fr
                equity += funding_pnl
                pos["funding_pnl"] += funding_pnl

        # Execute pending entry at current bar open.
        if pending_entry and pos is None and t_ms <= end_ms:
            entry_exec = open_p[i] * (1.0 - params.slippage)
            entry_equity = equity
            notional = entry_equity * params.position_fraction * params.leverage
            if notional > 0:
                qty = notional / entry_exec
                entry_fee = notional * params.fee_rate
                equity -= entry_fee
                pos = {
                    "entry_time_ms": t_ms,
                    "entry_signal_time_ms": pending_signal_time,
                    "entry_price": entry_exec,
                    "qty": qty,
                    "entry_equity": entry_equity,
                    "entry_notional": notional,
                    "entry_fee": entry_fee,
                    "funding_pnl": 0.0,
                    "trail_active": False,
                    "trough": math.inf,
                }
            pending_entry = False
            pending_signal_time = None

        # Position management.
        if pos is not None:
            stop_price = pos["entry_price"] * (1.0 + params.stop_loss_pct)
            exit_reason = None
            exit_raw = None

            # Conservative order: stop-loss precedence on same bar.
            if high_p[i] >= stop_price:
                exit_reason = "stop_loss"
                exit_raw = stop_price
            else:
                if (not pos["trail_active"]) and (low_p[i] <= pos["entry_price"] * (1.0 - params.trail_activate_pct)):
                    pos["trail_active"] = True
                    pos["trough"] = low_p[i]

                if pos["trail_active"]:
                    if low_p[i] < pos["trough"]:
                        pos["trough"] = low_p[i]
                    trail_stop = pos["trough"] * (1.0 + params.trail_retrace_pct)
                    if high_p[i] >= trail_stop:
                        exit_reason = "trailing_stop"
                        exit_raw = trail_stop

            if exit_reason is None and t_ms >= end_ms:
                exit_reason = "end_of_test"
                exit_raw = close_p[i]

            if exit_reason is not None and exit_raw is not None:
                exit_exec = exit_raw * (1.0 + params.slippage)
                exit_notional = pos["qty"] * exit_exec
                exit_fee = exit_notional * params.fee_rate
                gross_pnl = (pos["entry_price"] - exit_exec) * pos["qty"]
                equity += gross_pnl - exit_fee
                total_pnl = gross_pnl - pos["entry_fee"] - exit_fee + pos["funding_pnl"]
                margin_used = pos["entry_equity"] * params.position_fraction

                trades.append(
                    {
                        "entry_time": ms_to_dt(pos["entry_time_ms"]).isoformat(),
                        "exit_time": ms_to_dt(t_ms).isoformat(),
                        "signal_time": ms_to_dt(pos["entry_signal_time_ms"]).isoformat()
                        if pos["entry_signal_time_ms"] is not None
                        else None,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_exec,
                        "qty": pos["qty"],
                        "entry_notional": pos["entry_notional"],
                        "gross_pnl": gross_pnl,
                        "funding_pnl": pos["funding_pnl"],
                        "entry_fee": pos["entry_fee"],
                        "exit_fee": exit_fee,
                        "total_pnl": total_pnl,
                        "return_on_margin": total_pnl / margin_used if margin_used > 0 else np.nan,
                        "holding_hours": (t_ms - pos["entry_time_ms"]) / 3_600_000,
                        "reason": exit_reason,
                    }
                )
                pos = None

        # Mark-to-market equity curve.
        if pos is not None:
            mtm_equity = equity + (pos["entry_price"] - close_p[i]) * pos["qty"]
        else:
            mtm_equity = equity
        curve.append({"time": ms_to_dt(t_ms).isoformat(), "equity": mtm_equity})

        # Signal generation at bar close for next bar entry.
        if (
            (pos is None)
            and (not pending_entry)
            and (start_ms <= t_ms <= end_ms)
            and signal[i]
            and (i < len(work) - 1)
            and (open_ms[i + 1] <= end_ms)
        ):
            pending_entry = True
            pending_signal_time = t_ms

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve)

    metrics = summarize_metrics(
        trades_df=trades_df,
        curve_df=curve_df,
        initial_equity=initial_equity,
        final_equity=(float(curve_df["equity"].iloc[-1]) if not curve_df.empty else equity),
        start_dt=start_dt,
        end_dt=end_dt,
    )
    return trades_df, curve_df, metrics


def summarize_metrics(
    trades_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    initial_equity: float,
    final_equity: float,
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, Any]:
    total_return = final_equity / initial_equity - 1.0
    years = max((end_dt - start_dt).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (final_equity / initial_equity) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0

    if trades_df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "total_return": total_return,
            "cagr": cagr,
            "calmar": 0.0,
            "monthly_win_ratio": 0.0,
            "max_consecutive_losses": 0,
            "final_equity": final_equity,
        }

    wins = trades_df[trades_df["total_pnl"] > 0]
    losses = trades_df[trades_df["total_pnl"] <= 0]
    gross_profit = float(wins["total_pnl"].sum())
    gross_loss = float(abs(losses["total_pnl"].sum()))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    win_rate = float(len(wins) / len(trades_df))
    expectancy = float(trades_df["total_pnl"].mean())

    max_dd = 0.0
    monthly_win_ratio = 0.0

    if not curve_df.empty:
        eq = curve_df.copy()
        eq["time"] = pd.to_datetime(eq["time"], utc=True)
        eq = eq.sort_values("time")
        eq["peak"] = eq["equity"].cummax()
        eq["dd"] = eq["equity"] / eq["peak"] - 1.0
        max_dd = float(eq["dd"].min())

        monthly = eq.set_index("time")["equity"].resample("ME").last().dropna()
        if len(monthly) >= 2:
            mret = monthly.pct_change().dropna()
            if len(mret) > 0:
                monthly_win_ratio = float((mret > 0).mean())

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    max_consec_losses = 0
    cur_losses = 0
    for pnl in trades_df["total_pnl"].tolist():
        if pnl <= 0:
            cur_losses += 1
            max_consec_losses = max(max_consec_losses, cur_losses)
        else:
            cur_losses = 0

    return {
        "trades": int(len(trades_df)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_dd,
        "total_return": total_return,
        "cagr": cagr,
        "calmar": calmar,
        "monthly_win_ratio": monthly_win_ratio,
        "max_consecutive_losses": int(max_consec_losses),
        "final_equity": final_equity,
    }


def choose_best_params(grid_df: pd.DataFrame) -> dict[str, float] | None:
    if grid_df.empty:
        return None

    valid = grid_df[grid_df["trades"] >= 25].copy()
    if valid.empty:
        valid = grid_df.copy()

    valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=["calmar", "profit_factor", "total_return"])
    if valid.empty:
        return None

    valid = valid.sort_values(
        by=["calmar", "profit_factor", "total_return", "max_drawdown"],
        ascending=[False, False, False, False],
    )
    top = valid.iloc[0]
    return {
        "k": float(top["k"]),
        "stop_loss_pct": float(top["stop_loss_pct"]),
        "trail_activate_pct": float(top["trail_activate_pct"]),
        "trail_retrace_pct": float(top["trail_retrace_pct"]),
    }


def run_grid_search(
    df: pd.DataFrame,
    funding_df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    base_params: Params,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for k in [0.4, 0.5, 0.6]:
        for stop in [0.015, 0.02, 0.025]:
            for act in [0.025, 0.03, 0.035]:
                for retr in [0.008, 0.01, 0.012]:
                    p = Params(
                        k=k,
                        stop_loss_pct=stop,
                        trail_activate_pct=act,
                        trail_retrace_pct=retr,
                        vol_mult=base_params.vol_mult,
                        fee_rate=base_params.fee_rate,
                        slippage=base_params.slippage,
                        position_fraction=base_params.position_fraction,
                        leverage=base_params.leverage,
                    )
                    _, _, m = run_backtest(df, funding_df, start_dt, end_dt, p)
                    rows.append(
                        {
                            "k": k,
                            "stop_loss_pct": stop,
                            "trail_activate_pct": act,
                            "trail_retrace_pct": retr,
                            **m,
                        }
                    )

    return pd.DataFrame(rows)


def build_judgement(metrics_oos: dict[str, Any]) -> str:
    if metrics_oos.get("trades", 0) < 25:
        return "NO-GO: OOS 거래 수 부족"

    pf = metrics_oos.get("profit_factor", 0.0)
    mdd = abs(metrics_oos.get("max_drawdown", 0.0))
    calmar = metrics_oos.get("calmar", 0.0)

    if pf >= 1.25 and mdd <= 0.15 and calmar >= 1.0:
        return "GO"
    if pf >= 1.05 and mdd <= 0.22:
        return "CONDITIONAL GO"
    return "NO-GO"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest short volatility breakout strategy")
    p.add_argument("--start", default="2024-09-01")
    p.add_argument("--end", default="2026-02-27")
    p.add_argument("--initial-equity", type=float, default=100000.0)
    p.add_argument("--outdir", default="results")
    p.add_argument(
        "--candidates",
        default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT,TRXUSDT",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

    split_dt = datetime(2025, 9, 1, tzinfo=timezone.utc)

    candidates = [c.strip().upper() for c in args.candidates.split(",") if c.strip()]

    selected_symbol, selection_stats = select_symbol(candidates, start_dt)

    warmup_start = start_dt - timedelta(days=14)
    df_1h = fetch_klines(selected_symbol, "1h", dt_to_ms(warmup_start), dt_to_ms(end_dt + timedelta(days=1)))
    if df_1h.empty:
        raise RuntimeError(f"No 1h data fetched for {selected_symbol}")

    funding_df = fetch_funding_rates(selected_symbol, dt_to_ms(start_dt), dt_to_ms(end_dt + timedelta(days=1)))

    prepared = prepare_hourly_features(df_1h)

    base_params = Params()

    grid_is = run_grid_search(prepared, funding_df, start_dt, split_dt - timedelta(seconds=1), base_params)
    best = choose_best_params(grid_is)
    if best is None:
        best_params = base_params
    else:
        best_params = Params(
            k=best["k"],
            stop_loss_pct=best["stop_loss_pct"],
            trail_activate_pct=best["trail_activate_pct"],
            trail_retrace_pct=best["trail_retrace_pct"],
            vol_mult=base_params.vol_mult,
            fee_rate=base_params.fee_rate,
            slippage=base_params.slippage,
            position_fraction=base_params.position_fraction,
            leverage=base_params.leverage,
        )

    trades_full, curve_full, metrics_full = run_backtest(
        prepared,
        funding_df,
        start_dt,
        end_dt,
        best_params,
        initial_equity=args.initial_equity,
    )

    trades_oos, curve_oos, metrics_oos = run_backtest(
        prepared,
        funding_df,
        split_dt,
        end_dt,
        best_params,
        initial_equity=args.initial_equity,
    )

    judgement = build_judgement(metrics_oos)

    summary = {
        "selected_symbol": selected_symbol,
        "period": {"start": start_dt.isoformat(), "end": end_dt.isoformat(), "split_oos_start": split_dt.isoformat()},
        "assumptions": {
            "exchange": "Binance USDT-M Perp",
            "timeframe": "1h",
            "ny_entry_hour": 10,
            "position_fraction": best_params.position_fraction,
            "leverage": best_params.leverage,
            "fee_rate_per_side": best_params.fee_rate,
            "slippage_per_side": best_params.slippage,
        },
        "best_params_from_is": {
            "k": best_params.k,
            "stop_loss_pct": best_params.stop_loss_pct,
            "trail_activate_pct": best_params.trail_activate_pct,
            "trail_retrace_pct": best_params.trail_retrace_pct,
            "vol_mult": best_params.vol_mult,
        },
        "metrics_full": metrics_full,
        "metrics_oos": metrics_oos,
        "judgement": judgement,
    }

    selection_stats.to_csv(f"{outdir}/symbol_selection.csv", index=False)
    grid_is.to_csv(f"{outdir}/grid_is.csv", index=False)
    trades_full.to_csv(f"{outdir}/trades_full.csv", index=False)
    curve_full.to_csv(f"{outdir}/equity_curve_full.csv", index=False)
    trades_oos.to_csv(f"{outdir}/trades_oos.csv", index=False)
    curve_oos.to_csv(f"{outdir}/equity_curve_oos.csv", index=False)

    with open(f"{outdir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
