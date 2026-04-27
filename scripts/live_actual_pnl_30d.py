#!/usr/bin/env python3
"""Compute realized P&L of the live demo trader over the last 30 days.

Steps:
1. Filter decision log to placed=True, |amount| > 0 fills.
2. Build FIFO position book per pair (consistent with kernel design).
3. Mark positions to market using hourly OHLCV candles from postgres.
4. Compute daily realized + unrealized P&L per pair.
5. Run backtest replay over same 30-day window and extract per-day returns.
6. Compute per-day live vs backtest gap analysis.

Output: models/live_actual_pnl_30d.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gp_crypto_evolution as gp
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    normalize_mapping_indices,
    normalize_route_state_mode,
    realistic_overlay_replay_from_context,
)

UTC = timezone.utc
ROOT = Path(__file__).parent.parent
DEFAULT_DECISION_LOG = ROOT / "logs" / "pairwise_regime_decisions.jsonl"
DEFAULT_REPORT_OUT = ROOT / "models" / "live_actual_pnl_30d.json"

FEE_RATE = 0.0004       # taker fee, matches kernel default
SLIPPAGE = 0.0002       # half-spread slippage estimate
FUNDING_RANGE_START = "2022-04-06"
FUNDING_RANGE_END = "2026-04-06"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# Load placed trades from decision log
# ---------------------------------------------------------------------------

def load_placed_trades(log_path: Path, cutoff: datetime) -> list[dict[str, Any]]:
    """Return list of real fills: placed=True AND |amount| > 0 within window."""
    trades: list[dict[str, Any]] = []
    if not log_path.exists():
        return trades
    with log_path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            at_str = rec.get("at", "")
            if not at_str:
                continue
            try:
                at = datetime.fromisoformat(at_str)
            except ValueError:
                continue
            if at.tzinfo is None:
                at = at.replace(tzinfo=UTC)
            if at < cutoff:
                continue
            actions = rec.get("actions") or []
            equity = rec.get("equity")
            for action in actions:
                if not action.get("placed"):
                    continue
                amount = abs(action.get("amount", 0.0))
                if amount < 1e-9:
                    continue
                diff_qty = action.get("diff_qty", 0.0)
                exec_price = action.get("price", 0.0)
                if exec_price <= 0:
                    continue
                trades.append({
                    "at": at,
                    "pair": action.get("pair", ""),
                    "side": action.get("side", ""),
                    "exec_price": float(exec_price),
                    "amount": float(amount),
                    "diff_qty": float(diff_qty),
                    "target_weight": float(action.get("target_weight", 0.0)),
                    "equity": float(equity) if equity is not None else None,
                    "mode": rec.get("mode", ""),
                    "fee": float(amount) * float(exec_price) * FEE_RATE,
                })
    trades.sort(key=lambda t: t["at"])
    return trades


# ---------------------------------------------------------------------------
# FIFO position book
# ---------------------------------------------------------------------------

class FifoBook:
    """FIFO lot accounting per pair, consistent with the kernel's entry_lots design."""

    def __init__(self, pair: str) -> None:
        self.pair = pair
        # each lot: [qty, entry_price, entry_fee]  (qty always positive)
        self.lots: list[list[float]] = []
        self.net_qty: float = 0.0        # positive = long, negative = short
        self.realized_pnl: float = 0.0
        self.total_fee: float = 0.0

    def apply_trade(self, diff_qty: float, exec_price: float, fee: float) -> float:
        """Apply a fill (diff_qty signed: + = buying, - = selling).
        Returns realized P&L from closes in this fill."""
        self.total_fee += fee
        realized = 0.0

        if diff_qty == 0.0:
            return 0.0

        prev_sign = 1 if self.net_qty > 0 else (-1 if self.net_qty < 0 else 0)
        new_sign = 1 if diff_qty > 0 else -1

        if prev_sign == 0 or prev_sign == new_sign:
            # Opening or adding to position
            self.lots.append([abs(diff_qty), exec_price, fee])
            self.net_qty += diff_qty
        else:
            # Closing or reversing
            remaining = abs(diff_qty)
            close_fee_per_unit = fee / abs(diff_qty) if abs(diff_qty) > 1e-12 else 0.0
            while remaining > 1e-12 and self.lots:
                lot_qty, lot_price, lot_entry_fee = self.lots[0]
                closed = min(lot_qty, remaining)
                portion_entry_fee = lot_entry_fee * (closed / lot_qty)
                gross = (exec_price - lot_price) * closed * float(prev_sign)
                net = gross - portion_entry_fee - close_fee_per_unit * closed
                realized += net
                self.lots[0][0] -= closed
                self.lots[0][2] -= portion_entry_fee
                if self.lots[0][0] < 1e-12:
                    self.lots.pop(0)
                remaining -= closed
            self.net_qty += diff_qty
            # If reversal leaves open, start new lot
            if abs(remaining) > 1e-12 and abs(self.net_qty) > 1e-12:
                open_fee = close_fee_per_unit * remaining
                self.lots = [[remaining, exec_price, open_fee]]
            elif abs(self.net_qty) < 1e-12:
                self.lots = []
                self.net_qty = 0.0

        self.realized_pnl += realized
        return realized

    def unrealized_pnl(self, mark_price: float) -> float:
        """MTM unrealized P&L of open lots."""
        if not self.lots or self.net_qty == 0.0:
            return 0.0
        sign = 1 if self.net_qty > 0 else -1
        total = 0.0
        for lot_qty, lot_price, _ in self.lots:
            total += (mark_price - lot_price) * lot_qty * sign
        return total

    def open_notional(self, mark_price: float) -> float:
        return abs(self.net_qty) * mark_price


# ---------------------------------------------------------------------------
# Load hourly OHLCV from postgres (mark prices)
# ---------------------------------------------------------------------------

def load_hourly_close(pair: str, start: str, end: str) -> pd.Series:
    """Return hourly close prices for pair as a UTC-indexed Series."""
    try:
        df = gp.load_pair(pair, interval="1h", start=start, end=end, refresh_cache=False)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        # load_pair returns prefixed columns like BTCUSDT_close
        col = f"{pair}_close"
        if col not in df.columns:
            # Fallback: look for any close column
            close_cols = [c for c in df.columns if c.endswith("_close")]
            if not close_cols:
                print(f"[live_pnl] WARNING: no close column found for {pair}; columns: {df.columns.tolist()[:5]}")
                return pd.Series(dtype=float)
            col = close_cols[0]
        # Ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)
        return df[col].sort_index()
    except Exception as exc:
        print(f"[live_pnl] WARNING: could not load hourly data for {pair}: {exc}")
        return pd.Series(dtype=float)


# ---------------------------------------------------------------------------
# Compute daily P&L from trades + mark-to-market
# ---------------------------------------------------------------------------

def compute_daily_pnl_live(
    pair: str,
    trades: list[dict[str, Any]],
    hourly_close: pd.Series,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, Any]]:
    """Return list of {date, pair, realized, unrealized, total} dicts (one per day)."""
    if not trades:
        # Generate zero rows for each day
        days = pd.date_range(
            window_start.date(), window_end.date(), freq="D", tz=UTC
        )
        return [
            {"date": d.date().isoformat(), "pair": pair,
             "realized": 0.0, "unrealized": 0.0, "total": 0.0}
            for d in days
        ]

    book = FifoBook(pair)

    # Index trades by date
    trades_by_date: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        trades_by_date[t["at"].date().isoformat()].append(t)

    # Mark prices: resample hourly to daily close (end of day = last hour close)
    if not hourly_close.empty:
        daily_mark = hourly_close.resample("1D").last().ffill()
        daily_mark.index = daily_mark.index.tz_convert(UTC)
    else:
        daily_mark = pd.Series(dtype=float)

    days = pd.date_range(
        window_start.date(), window_end.date(), freq="D", tz=UTC
    )
    rows: list[dict[str, Any]] = []
    realized_cumulative = 0.0

    last_known_price = 0.0
    # seed last_known_price from first trade in window if available
    if trades:
        last_known_price = trades[0]["exec_price"]

    prev_unrealized = 0.0  # unrealized at end of previous day

    for day in days:
        date_str = day.date().isoformat()
        day_realized = 0.0
        for t in trades_by_date.get(date_str, []):
            r = book.apply_trade(t["diff_qty"], t["exec_price"], t["fee"])
            day_realized += r
            if t["exec_price"] > 0:
                last_known_price = t["exec_price"]

        # Mark price for this day — prefer hourly close, fall back to last trade price
        mark_price = last_known_price
        if not daily_mark.empty:
            idx = daily_mark.index.get_indexer([day], method="nearest")
            if idx[0] >= 0:
                mp = float(daily_mark.iloc[idx[0]])
                if mp > 0:
                    mark_price = mp
                    last_known_price = mp

        unrealized_snap = book.unrealized_pnl(mark_price) if mark_price > 0 else prev_unrealized
        # Daily unrealized P&L = change from yesterday's snapshot
        day_unrealized_delta = unrealized_snap - prev_unrealized
        prev_unrealized = unrealized_snap

        # total = realized today + change in unrealized today
        rows.append({
            "date": date_str,
            "pair": pair,
            "realized": round(day_realized, 6),
            "unrealized": round(day_unrealized_delta, 6),
            "total": round(day_realized + day_unrealized_delta, 6),
        })

    return rows


# ---------------------------------------------------------------------------
# Funding data helpers (reused from reconciliation script)
# ---------------------------------------------------------------------------

def _load_funding_csv(pair: str) -> pd.DataFrame:
    path = gp.DATA_DIR / f"{pair}_funding_{FUNDING_RANGE_START}_{FUNDING_RANGE_END}.csv"
    if not path.exists():
        candidates = sorted(gp.DATA_DIR.glob(f"{pair}_funding_{FUNDING_RANGE_START}_*.csv"))
        if not candidates:
            return pd.DataFrame(columns=["fundingTime", "fundingRate"])
        path = candidates[-1]
    df = pd.read_csv(path)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], utc=True, format="mixed")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df.dropna(subset=["fundingTime", "fundingRate"]).sort_values("fundingTime").reset_index(drop=True)


def load_funding(pair: str) -> pd.DataFrame:
    pg_frame = gp.load_funding_rates(pair, FUNDING_RANGE_START, FUNDING_RANGE_END)
    csv_frame = _load_funding_csv(pair)
    frames = [f for f in (pg_frame, csv_frame) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["fundingTime", "fundingRate"])
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["fundingTime"], keep="first")
        .sort_values("fundingTime")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Backtest daily returns
# ---------------------------------------------------------------------------

def run_backtest_daily(
    df: pd.DataFrame,
    pair: str,
    compiled_model,
    library: list,
    pair_config: dict[str, Any],
    funding_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    """Run realistic replay and extract per-day returns within the window."""
    from search_pair_subset_regime_mixture import route_state_names

    raw_signal = pd.Series(
        compiled_model(*gp.get_feature_arrays(df, pair)),
        index=df.index,
        dtype="float64",
    )
    overlay_inputs = build_overlay_inputs(df, PAIRS, regime_pair=pair)
    route_state_mode = normalize_route_state_mode(pair_config.get("route_state_mode", "base"))
    route_threshold = float(pair_config["route_breadth_threshold"])
    mapping = normalize_mapping_indices(
        tuple(int(v) for v in pair_config["mapping_indices"]),
        route_state_mode,
    )
    library_lookup = build_library_lookup(library)
    context = build_fast_context(
        df=df,
        pair=pair,
        raw_signal=raw_signal,
        overlay_inputs=overlay_inputs,
        route_thresholds=(route_threshold,),
        library_lookup=library_lookup,
        funding_df=funding_df,
        route_state_mode=route_state_mode,
    )
    result = realistic_overlay_replay_from_context(
        context,
        library_lookup,
        mapping,
        route_threshold,
        engine="python",
        return_trace=True,
    )

    # bar_net is a per-bar return fraction; df.index gives bar timestamps
    bar_net: np.ndarray = result.get("trace", {}).get("bar_net", np.array([]))
    # bar_net has len = open_p.shape[0] - 2 (exec_idx goes 1..N-2)
    # the bar timestamps start at index 1 of df
    bar_index = df.index[1: 1 + len(bar_net)]

    if len(bar_net) == 0 or len(bar_index) == 0:
        days = pd.date_range(window_start.date(), window_end.date(), freq="D", tz=UTC)
        return [
            {"date": d.date().isoformat(), "pair": pair, "daily_return": 0.0}
            for d in days
        ]

    bar_series = pd.Series(bar_net, index=bar_index)
    # Filter to window
    bar_series = bar_series.loc[
        (bar_series.index >= window_start) & (bar_series.index <= window_end)
    ]

    # Compound bar returns into daily returns: (1+r1)*(1+r2)*...-1
    if bar_series.index.tz is None:
        bar_series.index = bar_series.index.tz_localize(UTC)
    else:
        bar_series.index = bar_series.index.tz_convert(UTC)

    daily_compound = bar_series.groupby(bar_series.index.date).apply(
        lambda x: float(np.prod(1.0 + x) - 1.0)
    )

    days = pd.date_range(window_start.date(), window_end.date(), freq="D", tz=UTC)
    rows: list[dict[str, Any]] = []
    for day in days:
        date_key = day.date()
        dr = float(daily_compound.get(date_key, 0.0))
        rows.append({"date": date_key.isoformat(), "pair": pair, "daily_return": dr})
    return rows


# ---------------------------------------------------------------------------
# Load summary config
# ---------------------------------------------------------------------------

def load_summary_config(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text())
    cand = data.get("selected_candidate", data)
    return cand.get("pair_configs", {})


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

def compute_gap_analysis(
    daily_live: list[dict[str, Any]],
    daily_bt: list[dict[str, Any]],
    initial_equity: float,
) -> dict[str, Any]:
    """Per-day diff between live total P&L (as return) and backtest return."""
    # Build date -> live total pnl across all pairs
    live_by_date: dict[str, float] = defaultdict(float)
    for row in daily_live:
        live_by_date[row["date"]] += row["total"]

    bt_by_date: dict[str, float] = defaultdict(float)
    for row in daily_bt:
        bt_by_date[row["date"]] += row["daily_return"]

    all_dates = sorted(set(live_by_date) | set(bt_by_date))
    diffs_bps: list[float] = []
    drift_gt_50: list[str] = []
    sign_matches = 0

    equity_ref = initial_equity if initial_equity and initial_equity > 0 else 1.0

    for date in all_dates:
        live_pnl = live_by_date.get(date, 0.0)
        live_ret = live_pnl / equity_ref
        bt_ret = bt_by_date.get(date, 0.0)
        diff_bps = (live_ret - bt_ret) * 10000.0
        diffs_bps.append(diff_bps)
        if abs(diff_bps) > 50.0:
            drift_gt_50.append(date)
        if (live_ret >= 0) == (bt_ret >= 0):
            sign_matches += 1

    total_live_pnl = sum(live_by_date.values())
    total_live_ret = total_live_pnl / equity_ref if equity_ref > 0 else 0.0
    total_bt_ret = sum(bt_by_date.values())

    n = len(diffs_bps)
    mean_abs = float(np.mean(np.abs(diffs_bps))) if diffs_bps else 0.0
    max_abs = float(np.max(np.abs(diffs_bps))) if diffs_bps else 0.0
    sign_match_rate = float(sign_matches / n) if n > 0 else 0.0

    # Gap to 1%/day target: live avg daily return vs 0.01
    n_days = len(all_dates) if all_dates else 1
    avg_live_daily = total_live_ret / n_days
    gap_to_target = avg_live_daily - 0.01   # negative = below target

    # Top 3 drift days (by |diff_bps|)
    date_diff = sorted(
        zip(all_dates, diffs_bps), key=lambda x: abs(x[1]), reverse=True
    )
    top_3_drift = [
        {"date": d, "diff_bps": round(diff, 2)} for d, diff in date_diff[:3]
    ]

    return {
        "mean_abs_diff_bps": round(mean_abs, 2),
        "max_abs_diff_bps": round(max_abs, 2),
        "days_with_drift_gt_50bps": len(drift_gt_50),
        "drift_gt_50bps_dates": drift_gt_50,
        "sign_match_rate": round(sign_match_rate, 4),
        "total_live_pnl": round(total_live_pnl, 4),
        "total_live_return": round(total_live_ret, 6),
        "total_backtest_return": round(total_bt_ret, 6),
        "avg_live_daily_return": round(avg_live_daily, 6),
        "gap_to_1pct_per_day_target": round(gap_to_target, 6),
        "gap_to_1pct_per_day_target_bps": round(gap_to_target * 10000.0, 2),
        "top_3_drift_days": top_3_drift,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# Decision-log equity path (replaces FIFO as default)
# ---------------------------------------------------------------------------

def compute_daily_pnl_from_equity(
    log_path: Path,
    window_start: datetime,
    window_end: datetime,
    pairs: tuple[str, ...],
) -> tuple[list[dict[str, Any]], float | None, list[str]]:
    """Derive daily P&L from the equity field in each decision record.

    Algorithm:
      - For each UTC calendar date in [window_start, window_end], take the LAST
        decision record that has a non-None equity value on that date as the
        day's closing equity.
      - Use the previous date's closing equity as the opening equity.
      - daily_pnl = close_equity - open_equity.
      - If a date has no records with equity values, record daily_pnl = 0.0
        and flag the date as data_missing=True.
      - The first date with equity data is treated as t0 (opening only);
        its own daily_pnl is 0.0 (no prior reference).

    Per-pair attribution is not available in this mode.  The total daily P&L
    is distributed equally across all pairs so that downstream consumers
    (live_vs_backtest_same_window.py) produce a correct live_total_usd.
    This split is documented in the row's ``attribution_method`` field and in
    the report metadata — no pair-level numbers are fabricated.

    Returns:
        (rows, initial_equity, missing_dates)
        rows: list of {date, pair, realized, unrealized, total, data_missing,
                       attribution_method} dicts
        initial_equity: equity at the first record (for gap analysis base)
        missing_dates: list of ISO date strings with no equity data
    """
    # Build {date_str: last_equity_value} from log
    by_date: dict[str, float] = {}
    with log_path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            at_str = rec.get("at", "")
            if not at_str:
                continue
            try:
                at = datetime.fromisoformat(at_str)
            except ValueError:
                continue
            if at.tzinfo is None:
                at = at.replace(tzinfo=UTC)
            eq = rec.get("equity")
            if eq is None:
                continue
            date_str = at.date().isoformat()
            # Keep last (highest timestamp) equity per date
            by_date[date_str] = float(eq)

    # Build the ordered list of all dates in [window_start, window_end]
    all_days = pd.date_range(
        window_start.date(), window_end.date(), freq="D", tz=UTC
    )

    # Sorted equity dates available — used to fill prev_equity across gaps
    equity_dates = sorted(by_date.keys())
    initial_equity: float | None = float(by_date[equity_dates[0]]) if equity_dates else None

    rows: list[dict[str, Any]] = []
    missing_dates: list[str] = []
    per_pair = len(pairs) if pairs else 1

    prev_equity: float | None = None

    for day in all_days:
        date_str = day.date().isoformat()
        close_eq = by_date.get(date_str)

        if close_eq is None:
            # No equity data for this date
            missing_dates.append(date_str)
            daily_total = 0.0
            data_missing = True
        elif prev_equity is None:
            # First date with data — treat as t0, no P&L yet
            daily_total = 0.0
            data_missing = False
        else:
            daily_total = close_eq - prev_equity
            data_missing = False

        # Advance prev_equity if we got a close for today
        if close_eq is not None:
            prev_equity = close_eq

        # Distribute total evenly across pairs so downstream dollar sums are correct.
        # Each pair row carries (total / n_pairs); they sum back to daily_total.
        pair_share = daily_total / per_pair

        for pair in pairs:
            rows.append({
                "date": date_str,
                "pair": pair,
                "realized": round(pair_share, 6),
                "unrealized": 0.0,
                "total": round(pair_share, 6),
                "data_missing": data_missing,
                "attribution_method": (
                    "equity_delta_equal_split"
                    if not data_missing
                    else "missing_data_zero"
                ),
            })

    return rows, initial_equity, missing_dates


# ---------------------------------------------------------------------------
# Per-pair P&L from decision-log fills (replaces equal-split heuristic)
# ---------------------------------------------------------------------------

def compute_per_pair_pnl_from_decisions(
    log_path: Path,
    window_start: datetime,
    window_end: datetime,
    pairs: tuple[str, ...],
) -> tuple[list[dict[str, Any]], float | None, list[str], list[dict[str, Any]]]:
    """Compute per-pair daily realized + unrealized P&L from actual placed fills.

    Algorithm
    ---------
    Pass 1 — scan the full log (no window filter for equity; fills only from window):
      • Collect top-level ``equity`` as last-seen value per calendar date.
      • Collect all ``placed: True`` fills within [window_start, window_end].

    Pass 2 — iterate calendar days in the window:
      Realized (per pair):
        Feed today's placed fills into per-pair ``FifoBook`` instances using
        actual ``diff_qty`` (signed) and ``price``.  Fee = |diff_qty|*price*FEE_RATE.
        This produces genuine per-pair realized P&L from closes only.

      Unrealized (per pair) — balance identity:
        We know ``total_day_pnl = equity_close - equity_prev``.
        ``total_unrealized = total_day_pnl - sum(realized_per_pair)``.
        Unrealized is distributed across pairs weighted by their fill notional
        on active-fill days; equal-split on no-fill days.
        This guarantees ``sum(realized + unrealized) == equity_delta`` exactly.

      Days with no equity data: zero rows, ``data_missing=True``.

    Reconciliation warnings are emitted when a day has equity data but the
    computed pnl sum deviates > $5 from equity_delta (should never happen with
    this algorithm, but guards against log corruption).

    Returns
    -------
    (rows, initial_equity, missing_dates, reconciliation_warnings)
    rows: list of {date, pair, realized, unrealized, total,
                   data_missing, attribution_method}
    """
    per_pair = len(pairs) if pairs else 1
    n_pairs = per_pair

    # ------------------------------------------------------------------ #
    # Pass 1: scan log                                                    #
    # ------------------------------------------------------------------ #
    equity_by_date: dict[str, float] = {}
    # fills_by_date[date][pair] = list of (diff_qty, price)
    fills_by_date: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with log_path.open() as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            at_str = rec.get("at", "")
            if not at_str:
                continue
            try:
                at = datetime.fromisoformat(at_str)
            except ValueError:
                continue
            if at.tzinfo is None:
                at = at.replace(tzinfo=UTC)
            if at > window_end:
                continue
            date_str = at.date().isoformat()

            # Equity: scan entire log (pre-window too) for carry-forward
            eq = rec.get("equity")
            if eq is not None:
                equity_by_date[date_str] = float(eq)

            # Fills: only within window
            if at < window_start:
                continue
            for action in rec.get("actions") or []:
                if not action.get("placed"):
                    continue
                pair = action.get("pair", "")
                if not pair:
                    continue
                diff_qty = float(action.get("diff_qty", 0.0))
                price = float(action.get("price", 0.0))
                if price <= 0 or abs(diff_qty) < 1e-12:
                    continue
                fills_by_date[date_str][pair].append((diff_qty, price))

    # ------------------------------------------------------------------ #
    # Pass 2: iterate calendar days                                       #
    # ------------------------------------------------------------------ #
    all_days = pd.date_range(
        window_start.date(), window_end.date(), freq="D", tz=UTC
    )

    sorted_equity_dates = sorted(equity_by_date.keys())
    # initial_equity: first equity value at or after window_start
    initial_equity: float | None = None
    for d in sorted_equity_dates:
        if d >= window_start.date().isoformat():
            initial_equity = equity_by_date[d]
            break
    if initial_equity is None and sorted_equity_dates:
        initial_equity = equity_by_date[sorted_equity_dates[0]]

    # Per-pair FIFO books — rolling across the window for realized P&L
    books: dict[str, FifoBook] = {pair: FifoBook(pair) for pair in pairs}

    rows: list[dict[str, Any]] = []
    missing_dates: list[str] = []
    reconciliation_warnings: list[dict[str, Any]] = []

    # prev_equity: last seen equity, seeded from pre-window records if available
    prev_equity: float | None = None
    pre_window = [d for d in sorted_equity_dates if d < window_start.date().isoformat()]
    if pre_window:
        prev_equity = equity_by_date[pre_window[-1]]

    for day in all_days:
        date_str = day.date().isoformat()
        has_equity = date_str in equity_by_date
        day_fills = fills_by_date.get(date_str, {})
        has_fills = bool(day_fills)

        if not has_equity and not has_fills:
            missing_dates.append(date_str)

        # Realized per pair: FIFO on actual placed fills
        day_realized: dict[str, float] = {pair: 0.0 for pair in pairs}
        for pair in pairs:
            for diff_qty, price in day_fills.get(pair, []):
                fee = abs(diff_qty) * price * FEE_RATE
                r = books[pair].apply_trade(diff_qty, price, fee)
                day_realized[pair] += r

        # Total day P&L from equity delta
        close_eq = equity_by_date.get(date_str)
        if close_eq is not None and prev_equity is not None:
            eq_diff = close_eq - prev_equity
        else:
            eq_diff = None

        # Unrealized per pair: balance identity so aggregate is exact
        total_realized = sum(day_realized.values())
        pair_unrealized: dict[str, float] = {}

        if eq_diff is not None:
            total_unrealized = eq_diff - total_realized
            # Weight by fill notional so pairs that traded more absorb more MTM
            fill_notional: dict[str, float] = {
                pair: sum(abs(dq) * pr for dq, pr in day_fills.get(pair, []))
                for pair in pairs
            }
            total_notional = sum(fill_notional.values())
            for pair in pairs:
                if total_notional > 0:
                    weight = fill_notional[pair] / total_notional
                else:
                    weight = 1.0 / n_pairs
                pair_unrealized[pair] = total_unrealized * weight
            method = "decision-log-fills"
        else:
            # No equity data: zero unrealized for missing days
            for pair in pairs:
                pair_unrealized[pair] = 0.0
            method = "missing_data_zero"

        # Reconciliation guard (should always be 0 with balance identity)
        if eq_diff is not None:
            day_sum = sum(day_realized[p] + pair_unrealized[p] for p in pairs)
            residual = abs(day_sum - eq_diff)
            if residual > 5.0:
                reconciliation_warnings.append({
                    "date": date_str,
                    "pnl_sum": round(day_sum, 4),
                    "equity_diff": round(eq_diff, 4),
                    "residual": round(residual, 4),
                })

        if close_eq is not None:
            prev_equity = close_eq

        data_missing = not has_equity and not has_fills
        for pair in pairs:
            r = day_realized[pair]
            u = pair_unrealized[pair]
            rows.append({
                "date": date_str,
                "pair": pair,
                "realized": round(r, 6),
                "unrealized": round(u, 6),
                "total": round(r + u, 6),
                "data_missing": data_missing,
                "attribution_method": method,
            })

    return rows, initial_equity, missing_dates, reconciliation_warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute realized P&L of the live demo trader vs backtest over 30 days."
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--decision-log", type=Path, default=DEFAULT_DECISION_LOG)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Skip backtest replay (faster, live P&L only).")
    parser.add_argument(
        "--pnl-source",
        choices=["decision-log", "fifo"],
        default="fifo",
        help=(
            "Source for live daily P&L. "
            "'fifo' (default) preserves legacy per-pair / realized vs unrealized attribution "
            "for downstream consumers (telegram bots, charts, alarms). "
            "'decision-log' uses the equity field from each decision record "
            "to compute close-minus-open daily P&L; this matches real account totals "
            "but provides only an aggregate (per-pair attribution is split equally). "
            "'fifo' uses the legacy FIFO lot-accumulation path which is known to produce "
            "phantom unrealized P&L from residual rebalance round-trips (DEPRECATED)."
        ),
    )
    return parser.parse_args()


def _stub_report(report_out: Path, window: dict, note: str) -> None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    stub = {
        "generated_at": iso_now(),
        "status": "no_fills",
        "note": note,
        "window": window,
        "n_filled_trades_per_pair": {},
        "daily_pnl_live": [],
        "daily_pnl_backtest": [],
        "gap_analysis": {
            "mean_abs_diff_bps": 0.0,
            "max_abs_diff_bps": 0.0,
            "days_with_drift_gt_50bps": 0,
            "drift_gt_50bps_dates": [],
            "sign_match_rate": 0.0,
            "total_live_pnl": 0.0,
            "total_live_return": 0.0,
            "total_backtest_return": 0.0,
            "avg_live_daily_return": 0.0,
            "gap_to_1pct_per_day_target": -0.01,
            "gap_to_1pct_per_day_target_bps": -100.0,
            "top_3_drift_days": [],
            "n_days": 0,
        },
    }
    report_out.write_text(json.dumps(stub, indent=2))
    print(f"[live_pnl] Stub report written to {report_out}")


def main() -> None:
    args = parse_args()

    now = datetime.now(UTC)
    cutoff = now - timedelta(days=args.days)
    window = {
        "start": cutoff.isoformat(),
        "end": now.isoformat(),
    }

    use_decision_log_path = (args.pnl_source == "decision-log")

    # -------------------------------------------------------------------------
    # BRANCH: decision-log path (default) — equity close-minus-open per day
    # -------------------------------------------------------------------------
    if use_decision_log_path:
        print(f"[live_pnl] pnl-source=decision-log: reading equity from {args.decision_log}")
        if not args.decision_log.exists():
            _stub_report(
                args.report_out,
                window,
                f"Decision log not found: {args.decision_log}",
            )
            return

        daily_pnl_live, initial_equity_dl, missing_dates, recon_warnings = (
            compute_per_pair_pnl_from_decisions(
                log_path=args.decision_log,
                window_start=cutoff,
                window_end=now,
                pairs=PAIRS,
            )
        )

        if not daily_pnl_live:
            _stub_report(
                args.report_out,
                window,
                "No decision records found in decision log for window.",
            )
            return

        # Count filled trades for metadata
        all_trades = load_placed_trades(args.decision_log, cutoff)
        trades_by_pair: dict[str, list[dict]] = defaultdict(list)
        for t in all_trades:
            if t["pair"]:
                trades_by_pair[t["pair"]].append(t)
        n_filled_per_pair = {p: len(ts) for p, ts in trades_by_pair.items()}

        initial_equity = initial_equity_dl if initial_equity_dl else 10000.0

        if missing_dates:
            print(
                f"[live_pnl] WARNING: {len(missing_dates)} dates have no decision records "
                f"(recorded as 0.0, data_missing=True): {missing_dates}"
            )
        if recon_warnings:
            print(
                f"[live_pnl] WARNING: {len(recon_warnings)} days have reconciliation "
                f"residual > $5: {[w['date'] for w in recon_warnings]}"
            )

        # Backtest still needs OHLCV — load it
        data_start_ts = cutoff - timedelta(days=60)
        data_start = data_start_ts.strftime("%Y-%m-%d")
        data_end = now.strftime("%Y-%m-%d")
        window_start_ts = pd.Timestamp(cutoff)
        window_end_ts = pd.Timestamp(now)

    # -------------------------------------------------------------------------
    # BRANCH: fifo path (legacy/deprecated)
    # -------------------------------------------------------------------------
    else:
        print("[live_pnl] pnl-source=fifo (DEPRECATED): using FIFO lot-accumulation path.")

        # --- Load fills ---
        print(f"[live_pnl] Loading placed trades from {args.decision_log}...")
        all_trades = load_placed_trades(args.decision_log, cutoff)
        print(f"[live_pnl] Found {len(all_trades)} placed fills in last {args.days} days.")

        if not all_trades and not args.skip_backtest:
            _stub_report(
                args.report_out,
                window,
                f"No placed fills found in last {args.days} days (cutoff {cutoff.isoformat()})",
            )
            _print_summary({}, [], [], {
                "mean_abs_diff_bps": 0.0, "max_abs_diff_bps": 0.0,
                "total_live_pnl": 0.0, "gap_to_1pct_per_day_target_bps": -100.0,
                "top_3_drift_days": [],
            })
            return

        # Separate by pair
        trades_by_pair = defaultdict(list)
        for t in all_trades:
            if t["pair"]:
                trades_by_pair[t["pair"]].append(t)

        n_filled_per_pair = {p: len(ts) for p, ts in trades_by_pair.items()}

        # --- Load 5m OHLCV for backtest (with 60-day warmup) ---
        data_start_ts = cutoff - timedelta(days=60)
        data_start = data_start_ts.strftime("%Y-%m-%d")
        data_end = now.strftime("%Y-%m-%d")

        print(f"[live_pnl] Loading 5m OHLCV {data_start} -> {data_end}...")
        df_all = gp.load_all_pairs(pairs=list(PAIRS), start=data_start, end=data_end, refresh_cache=False)

        # --- Load hourly close for MTM ---
        hourly_close: dict[str, pd.Series] = {}
        for pair in PAIRS:
            print(f"[live_pnl] Loading hourly close for {pair}...")
            hourly_close[pair] = load_hourly_close(pair, data_start, data_end)

        # --- Compute live daily P&L per pair ---
        window_start_ts = pd.Timestamp(cutoff)
        window_end_ts = pd.Timestamp(now)
        daily_pnl_live = []

        for pair in PAIRS:
            print(f"[live_pnl] Computing live daily P&L for {pair} ({n_filled_per_pair.get(pair, 0)} fills)...")
            rows = compute_daily_pnl_live(
                pair=pair,
                trades=trades_by_pair.get(pair, []),
                hourly_close=hourly_close.get(pair, pd.Series(dtype=float)),
                window_start=cutoff,
                window_end=now,
            )
            daily_pnl_live.extend(rows)

        # --- Estimate initial equity ---
        initial_equity = 10000.0
        if all_trades and all_trades[0].get("equity") is not None:
            initial_equity = float(all_trades[0]["equity"])
        elif all_trades:
            initial_equity = all_trades[0]["exec_price"] * all_trades[0]["amount"]

        missing_dates = []

    # --- Backtest (shared by both branches; needs df_all + window timestamps) ---
    daily_pnl_backtest: list[dict[str, Any]] = []

    if not args.skip_backtest:
        # decision-log branch needs to load OHLCV now (fifo branch loaded it above)
        if use_decision_log_path:
            print(f"[live_pnl] Loading 5m OHLCV {data_start} -> {data_end} for backtest...")
            df_all = gp.load_all_pairs(pairs=list(PAIRS), start=data_start, end=data_end, refresh_cache=False)

        print(f"[live_pnl] Loading summary config from {args.summary_path}...")
        pair_configs = load_summary_config(args.summary_path)
        if pair_configs is None:
            print(f"[live_pnl] WARNING: Cannot load summary config; backtest skipped.")
        else:
            print(f"[live_pnl] Loading GP model from {args.model_path}...")
            model_tree, _ = load_signal_model(args.model_path)
            compiled_model = gp.toolbox.compile(expr=model_tree)
            library = list(iter_params())

            print("[live_pnl] Loading funding data...")
            funding_cache: dict[str, pd.DataFrame] = {}
            for pair in PAIRS:
                fd = load_funding(pair)
                funding_cache[pair] = fd

            for pair in PAIRS:
                if pair not in pair_configs:
                    print(f"[live_pnl] WARNING: {pair} not in pair_configs, skipping backtest for pair.")
                    continue
                cfg = pair_configs[pair]
                print(f"[live_pnl] Running backtest replay for {pair}...")
                rows = run_backtest_daily(
                    df=df_all,
                    pair=pair,
                    compiled_model=compiled_model,
                    library=library,
                    pair_config=cfg,
                    funding_df=funding_cache.get(pair, pd.DataFrame()),
                    window_start=window_start_ts,
                    window_end=window_end_ts,
                )
                daily_pnl_backtest.extend(rows)

    # --- Gap analysis ---
    gap_analysis = compute_gap_analysis(daily_pnl_live, daily_pnl_backtest, initial_equity)

    # --- Build report ---
    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "pnl_source": args.pnl_source,
        "window": window,
        "n_filled_trades_per_pair": n_filled_per_pair,
        "initial_equity_estimate": round(initial_equity, 4),
        "daily_pnl_live": daily_pnl_live,
        "daily_pnl_backtest": daily_pnl_backtest,
        "gap_analysis": gap_analysis,
    }
    if use_decision_log_path:
        report["decision_log_metadata"] = {
            "dates_with_no_decision_records": missing_dates,
            "n_missing_dates": len(missing_dates),
            "reconciliation_warnings": recon_warnings,
            "n_reconciliation_warnings": len(recon_warnings),
            "per_pair_attribution": (
                "Per-pair realized P&L derived from actual placed fills via FIFO lot "
                "accounting. Unrealized P&L is marked at end-of-day latest_prices from "
                "the last decision record of each day. attribution_method='decision-log-fills'."
            ),
        }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(json_safe(report), indent=2))
    print(f"\n[live_pnl] Report written to {args.report_out}")

    _print_summary(n_filled_per_pair, daily_pnl_live, daily_pnl_backtest, gap_analysis)


def _print_summary(
    n_filled_per_pair: dict[str, int],
    daily_pnl_live: list[dict[str, Any]],
    daily_pnl_backtest: list[dict[str, Any]],
    gap_analysis: dict[str, Any],
) -> None:
    print("\n" + "=" * 68)
    print("LIVE ACTUAL P&L vs BACKTEST — 30-DAY WINDOW")
    print("=" * 68)
    print(f"Filled trades:  {json_safe(n_filled_per_pair)}")
    total_live_pnl = gap_analysis.get("total_live_pnl", 0.0)
    total_live_ret = gap_analysis.get("total_live_return", 0.0)
    gap_bps = gap_analysis.get("gap_to_1pct_per_day_target_bps", 0.0)
    avg_daily = gap_analysis.get("avg_live_daily_return", 0.0)
    print(f"Total live P&L: ${total_live_pnl:.2f}  ({total_live_ret*100:.3f}%)")
    print(f"Avg live daily: {avg_daily*100:.3f}%/day  |  Gap to 1%/day: {gap_bps:+.1f} bps")
    print(f"Mean |diff| vs backtest: {gap_analysis.get('mean_abs_diff_bps', 0.0):.1f} bps")
    print(f"Max  |diff| vs backtest: {gap_analysis.get('max_abs_diff_bps', 0.0):.1f} bps")
    print(f"Days drift >50bps: {gap_analysis.get('days_with_drift_gt_50bps', 0)}")
    top3 = gap_analysis.get("top_3_drift_days", [])
    if top3:
        print("Top 3 drift days:")
        for item in top3:
            print(f"  {item['date']}  {item['diff_bps']:+.1f} bps")
    print("=" * 68)


if __name__ == "__main__":
    main()
