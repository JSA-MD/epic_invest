"""
Genetic-Programming Crypto Trading Strategy
============================================
DEAP GP that evolves trading signal trees for Binance USDT-M Futures.
Adapted from: github.com/ZiadFrancis/Genetics_Trading_Part_1

Key adaptations for crypto:
- Trades BTCUSDT as primary instrument
- Uses BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT as signal inputs (16 features)
- Vectorized numpy backtester (no external backtesting library needed)
- 1-hour timeframe with crypto-specific parameters
"""

import math
import operator
import random
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, timezone

import functools

import dill
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from deap import base, creator, gp, tools

# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────
RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
PRIMARY_PAIR = "BTCUSDT"
BASIC_FEATURES = (
    "open", "high", "low", "close",
    "rsi_14", "atr_14", "macd_h", "bb_p", "vol_sma", "cci_14", "mfi_14",
    "dc_trend_05", "dc_event_05", "dc_overshoot_05", "dc_run_05",
)
ARG_NAMES = list(BASIC_FEATURES)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "binance_futures"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_START = "2025-10-01"
TRAIN_END = "2026-01-01"
VAL_START = "2026-01-01"
VAL_END = "2026-02-15"
TEST_START = "2026-02-15"
TEST_END = "2026-04-06"

POP_SIZE = 500
N_GEN = 5
P_CX = 0.90
P_MUT = 0.15
MAX_DEPTH = 8
MAX_LEN = 60

INITIAL_CASH = 100_000
COMMISSION_PCT = 0.0004   # 0.04% per side (Binance futures taker)
NO_TRADE_BAND = 10        # +/-10pp dead-band
TIMEFRAME = "5m"
DAILY_TARGET_PCT = 0.005
DAILY_MAX_LOSS_PCT = -0.005
DAILY_CVAR_ALPHA = 0.10
DAILY_ENTRY_THRESHOLD = 0.0
DEFAULT_REWARD_MULTIPLE = 3.0
ROBUST_REWARD_MULTIPLES = (2.0, 3.0, 4.0)
TRAIL_ACTIVATION_PCT = 0.005
TRAIL_DISTANCE_PCT = 0.005
TRAIL_FLOOR_PCT = 0.0025
DC_THRESHOLD_PCT = 0.005

API_BASE = "https://fapi.binance.com"

_eval_count = 0
_start_time = None

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data Fetching & Caching
# ─────────────────────────────────────────────────────────────────────────────
INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}
RAW_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
DERIVED_FEATURE_COLUMNS = tuple(
    c for c in BASIC_FEATURES if c not in {"open", "high", "low", "close"}
)
CACHE_COLUMNS = RAW_OHLCV_COLUMNS + DERIVED_FEATURE_COLUMNS


def fetch_klines(symbol: str, interval: str,
                 start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch OHLCV klines and return a cache-ready frame with features."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    step_ms = INTERVAL_MS[interval]
    all_data: list = []
    cur = start_ms

    while cur < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": end_ms, "limit": 1500,
        }
        data = None
        for attempt in range(5):
            try:
                resp = requests.get(
                    f"{API_BASE}/fapi/v1/klines", params=params, timeout=20,
                )
                if resp.status_code == 429:
                    wait_seconds = min(30.0, 2.0 * (attempt + 1))
                    print(f"  Rate limit for {symbol}; retrying in {wait_seconds:.0f}s")
                    time.sleep(wait_seconds)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 429 and attempt < 4:
                    wait_seconds = min(30.0, 2.0 * (attempt + 1))
                    print(f"  Rate limit for {symbol}; retrying in {wait_seconds:.0f}s")
                    time.sleep(wait_seconds)
                    continue
                print(f"  API error for {symbol}: {e}")
                data = None
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                print(f"  API error for {symbol}: {e}")
                data = None
                break
        if data is None:
            break
        if not data:
            break

        all_data.extend(data)
        last_open = int(data[-1][0])
        nxt = last_open + step_ms
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.05)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_base", "taker_quote", "ignore",
    ])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = (df.drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .set_index("open_time"))
    df = df[["open", "high", "low", "close", "volume"]]

    return enrich_features(df)


def add_directional_change_features(
    df: pd.DataFrame,
    threshold: float = DC_THRESHOLD_PCT,
) -> pd.DataFrame:
    """Add event-based directional-change features using close prices."""
    if df.empty:
        for c in ("dc_trend_05", "dc_event_05", "dc_overshoot_05", "dc_run_05"):
            df[c] = 0.0
        return df

    close = df["close"].to_numpy(dtype="float64")
    trend = np.zeros(len(df), dtype="float64")
    event = np.zeros(len(df), dtype="float64")
    overshoot = np.zeros(len(df), dtype="float64")
    run_mag = np.zeros(len(df), dtype="float64")

    state = 0.0
    last_extreme = float(close[0])
    confirm_price = float(close[0])

    for i, price in enumerate(close):
        if state == 0.0:
            up_move = price / last_extreme - 1.0
            down_move = price / last_extreme - 1.0
            if up_move >= threshold:
                state = 1.0
                event[i] = 1.0
                confirm_price = float(price)
                last_extreme = float(price)
            elif down_move <= -threshold:
                state = -1.0
                event[i] = -1.0
                confirm_price = float(price)
                last_extreme = float(price)
            else:
                last_extreme = max(last_extreme, float(price))
        elif state > 0.0:
            last_extreme = max(last_extreme, float(price))
            if price <= last_extreme * (1.0 - threshold):
                state = -1.0
                event[i] = -1.0
                confirm_price = float(price)
                last_extreme = float(price)
        else:
            last_extreme = min(last_extreme, float(price))
            if price >= last_extreme * (1.0 + threshold):
                state = 1.0
                event[i] = 1.0
                confirm_price = float(price)
                last_extreme = float(price)

        trend[i] = state
        signed_move = state * (float(price) / confirm_price - 1.0) if confirm_price else 0.0
        overshoot[i] = signed_move
        if state > 0.0 and last_extreme:
            run_mag[i] = max(float(price) / last_extreme - 1.0, 0.0)
        elif state < 0.0 and last_extreme:
            run_mag[i] = max(last_extreme / float(price) - 1.0, 0.0)
        else:
            run_mag[i] = 0.0

    df["dc_trend_05"] = trend
    df["dc_event_05"] = event
    df["dc_overshoot_05"] = overshoot
    df["dc_run_05"] = run_mag
    return df


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute technical and event-based features from OHLCV."""
    base_cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    base = df[base_cols].copy()
    if "volume" not in base.columns:
        base["volume"] = 0.0
    for c in ("rsi_14", "atr_14", "macd_h", "bb_p", "vol_sma", "cci_14", "mfi_14"):
        if c in df.columns and c not in base.columns:
            base[c] = df[c]

    if len(base) >= 30:
        try:
            base["rsi_14"] = base.ta.rsi(length=14)
            base["atr_14"] = base.ta.atr(length=14)
            macd = base.ta.macd(fast=12, slow=26, signal=9)
            if macd is not None and "MACDh_12_26_9" in macd:
                base["macd_h"] = macd["MACDh_12_26_9"]
            else:
                base["macd_h"] = 0.0
            bb = base.ta.bbands(length=20)
            if bb is not None and "BBP_20_2.0" in bb:
                base["bb_p"] = bb["BBP_20_2.0"]
            else:
                base["bb_p"] = 0.5
            if "volume" in base.columns:
                base["vol_sma"] = base.ta.sma(close=base["volume"], length=20)
            cci = base.ta.cci(
                high=base["high"], low=base["low"], close=base["close"], length=14,
            )
            if cci is not None:
                base["cci_14"] = cci
            if "volume" in base.columns:
                mfi = base.ta.mfi(
                    high=base["high"], low=base["low"], close=base["close"],
                    volume=base["volume"], length=14,
                )
                if mfi is not None:
                    base["mfi_14"] = mfi
        except Exception:
            pass

    for c, fallback in (
        ("rsi_14", 0.0), ("atr_14", 0.0), ("macd_h", 0.0), ("bb_p", 0.5),
        ("vol_sma", 0.0), ("cci_14", 0.0), ("mfi_14", 0.0),
    ):
        if c not in base.columns:
            base[c] = fallback

    base = add_directional_change_features(base)
    for c in BASIC_FEATURES:
        if c not in base.columns:
            base[c] = 0.0

    base.bfill(inplace=True)
    base.fillna(0, inplace=True)
    return base[list(CACHE_COLUMNS)]


def _parse_time_boundary(value: str | None) -> datetime | None:
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _normalize_cache_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in RAW_OHLCV_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _canonical_cache_file(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol}_{interval}.csv"


def _legacy_cache_file(
    symbol: str,
    interval: str,
    start: str | None,
    end: str | None,
) -> Path | None:
    if not start or not end or ":" in start or ":" in end:
        return None
    return DATA_DIR / f"{symbol}_{interval}_{start}_{end}.csv"


def _resolve_existing_cache_file(
    symbol: str,
    interval: str,
    start: str | None,
    end: str | None,
) -> Path | None:
    canonical = _canonical_cache_file(symbol, interval)
    if canonical.exists():
        return canonical

    legacy_exact = _legacy_cache_file(symbol, interval, start, end)
    if legacy_exact is not None and legacy_exact.exists():
        return legacy_exact

    matches = sorted(DATA_DIR.glob(f"{symbol}_{interval}_*.csv"))
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _read_cache_frame(
    symbol: str,
    interval: str,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, Path | None]:
    path = _resolve_existing_cache_file(symbol, interval, start, end)
    if path is None:
        return pd.DataFrame(), None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return _normalize_cache_frame(df), path


def _write_cache_frame(path: Path, df: pd.DataFrame) -> None:
    out = _normalize_cache_frame(df)
    out.index.name = "open_time"
    out.to_csv(path)


def _raw_cache_view(df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in RAW_OHLCV_COLUMNS if col in df.columns]
    if not cols:
        return pd.DataFrame()
    return _normalize_cache_frame(df[cols])


def _find_missing_ranges(
    df: pd.DataFrame,
    interval: str,
) -> list[tuple[datetime, datetime]]:
    if df.empty:
        return []
    expected_delta = pd.Timedelta(milliseconds=INTERVAL_MS[interval])
    diffs = df.index.to_series().diff().dropna()
    ranges: list[tuple[datetime, datetime]] = []
    for current_ts, delta in diffs[diffs > expected_delta].items():
        current_pos = df.index.get_loc(current_ts)
        prev_ts = df.index[current_pos - 1]
        gap_start = (prev_ts + expected_delta).to_pydatetime()
        gap_end = current_ts.to_pydatetime()
        ranges.append((gap_start, gap_end))
    return ranges


def _complete_raw_cache(
    symbol: str,
    interval: str,
    df: pd.DataFrame,
    request_start_dt: datetime | None,
    refresh_end_dt: datetime,
) -> pd.DataFrame:
    raw_df = _raw_cache_view(df)
    if raw_df.empty:
        fetch_start_dt = request_start_dt or refresh_end_dt
        raw_df = _raw_cache_view(fetch_klines(symbol, interval, fetch_start_dt, refresh_end_dt))
        if raw_df.empty:
            return raw_df

    interval_delta = timedelta(milliseconds=INTERVAL_MS[interval])
    for _ in range(4):
        changed = False
        first_cached_dt = raw_df.index[0].to_pydatetime()
        last_cached_dt = raw_df.index[-1].to_pydatetime()

        if request_start_dt is not None and first_cached_dt - request_start_dt >= interval_delta:
            backfill = _raw_cache_view(fetch_klines(symbol, interval, request_start_dt, first_cached_dt))
            if not backfill.empty:
                raw_df = _normalize_cache_frame(pd.concat([backfill, raw_df], axis=0))
                changed = True

        first_cached_dt = raw_df.index[0].to_pydatetime()
        last_cached_dt = raw_df.index[-1].to_pydatetime()
        if refresh_end_dt - last_cached_dt >= interval_delta:
            refreshed = _raw_cache_view(fetch_klines(symbol, interval, last_cached_dt, refresh_end_dt))
            if not refreshed.empty:
                raw_df = _normalize_cache_frame(pd.concat([raw_df, refreshed], axis=0))
                changed = True

        gap_ranges = _find_missing_ranges(raw_df, interval)
        for gap_start_dt, gap_end_dt in gap_ranges:
            print(f"  {symbol}: filling gap {gap_start_dt.isoformat()} -> {gap_end_dt.isoformat()}")
            gap_df = _raw_cache_view(fetch_klines(symbol, interval, gap_start_dt, gap_end_dt))
            if gap_df.empty:
                continue
            raw_df = _normalize_cache_frame(pd.concat([raw_df, gap_df], axis=0))
            changed = True

        if not changed:
            break

    return raw_df


def _sync_pair_cache(
    symbol: str,
    interval: str,
    start: str | None,
    end: str | None,
    refresh_cache: bool = True,
) -> pd.DataFrame:
    if not refresh_cache:
        cached_df, _ = _read_cache_frame(symbol, interval, start, end)
        if cached_df.empty:
            print(f"  {symbol}: cache miss")
        else:
            print(f"  {symbol}: cache-only hit")
        return cached_df

    request_start_dt = _parse_time_boundary(start)
    request_end_dt = _parse_time_boundary(end)
    refresh_end_dt = max(
        [dt for dt in (request_end_dt, datetime.now(timezone.utc)) if dt is not None]
    )
    interval_delta = timedelta(milliseconds=INTERVAL_MS[interval])

    cached_df, cache_source = _read_cache_frame(symbol, interval, start, end)
    fallback_df = cached_df.copy()
    had_cache = not cached_df.empty
    had_raw = had_cache and all(col in cached_df.columns for col in RAW_OHLCV_COLUMNS)
    before_len = len(cached_df)

    if not had_cache:
        fetch_start_dt = request_start_dt or refresh_end_dt
        print(f"  {symbol}: fetching from Binance API...")
        cached_df = fetch_klines(symbol, interval, fetch_start_dt, refresh_end_dt)
    elif not had_raw:
        fetch_start_dt = request_start_dt or cached_df.index[0].to_pydatetime()
        print(f"  {symbol}: migrating legacy cache -> {_canonical_cache_file(symbol, interval).name}")
        refreshed = fetch_klines(symbol, interval, fetch_start_dt, refresh_end_dt)
        if not refreshed.empty:
            cached_df = refreshed
        else:
            cached_df = fallback_df

    if cached_df.empty:
        return fallback_df

    if all(col in cached_df.columns for col in RAW_OHLCV_COLUMNS):
        raw_completed = _complete_raw_cache(
            symbol,
            interval,
            cached_df,
            request_start_dt=request_start_dt,
            refresh_end_dt=refresh_end_dt,
        )
        if not raw_completed.empty:
            cached_df = enrich_features(raw_completed)
        else:
            cached_df = cached_df[list(CACHE_COLUMNS)] if all(
                col in cached_df.columns for col in CACHE_COLUMNS
            ) else enrich_features(cached_df)
        canonical = _canonical_cache_file(symbol, interval)
        _write_cache_frame(canonical, cached_df)
        if not had_cache:
            print(f"  {symbol}: {len(cached_df)} bars cached -> {canonical.name}")
        elif cache_source != canonical:
            print(f"  {symbol}: cache migrated -> {canonical.name}")
        elif len(cached_df) > before_len:
            print(f"  {symbol}: cache extended (+{len(cached_df) - before_len} bars)")
        else:
            print(f"  {symbol}: cache hit")
        return cached_df

    if cache_source is not None:
        print(f"  {symbol}: using legacy cache ({cache_source.name})")
    return fallback_df


def load_pair(symbol: str, interval: str = TIMEFRAME,
              start: str = TRAIN_START, end: str | None = TEST_END,
              refresh_cache: bool = True) -> pd.DataFrame:
    """Load pair data, while persisting newly available bars into the cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = _sync_pair_cache(symbol, interval, start, end, refresh_cache=refresh_cache)

    if not df.empty:
        start_dt = _parse_time_boundary(start)
        end_dt = _parse_time_boundary(end)
        if start_dt is not None:
            df = df[df.index >= pd.Timestamp(start_dt)]
        if end_dt is not None:
            df = df[df.index <= pd.Timestamp(end_dt)]

    df.columns = [f"{symbol}_{c}" for c in df.columns]
    return df


def load_all_pairs(
    pairs: List[str] = PAIRS,
    start: str = TRAIN_START,
    end: str | None = TEST_END,
    refresh_cache: bool = True,
) -> pd.DataFrame:
    """Load all pairs and merge into wide DataFrame."""
    print(f"Loading {len(pairs)} crypto pairs...")
    dfs = [
        load_pair(pair, start=start, end=end, refresh_cache=refresh_cache)
        for pair in pairs
    ]
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, axis=1).dropna()
    if combined.empty:
        print("Dataset: 0 bars")
        return combined
    print(f"Dataset: {len(combined)} bars "
          f"({combined.index[0].date()} to {combined.index[-1].date()})")
    return combined


def split_dataset(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / validation / test."""
    train = df.loc[TRAIN_START:TRAIN_END].copy()
    val = df.loc[VAL_START:VAL_END].copy()
    test = df.loc[TEST_START:TEST_END].copy()
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)} bars")
    return train, val, test


def get_feature_arrays(
    df_slice: pd.DataFrame,
    pair: str = PRIMARY_PAIR,
) -> List[np.ndarray]:
    """Return model input arrays from either prefixed or plain feature columns."""
    prefixed_cols = [f"{pair}_{c}" for c in ARG_NAMES]
    if all(c in df_slice.columns for c in prefixed_cols):
        return [df_slice[c].to_numpy(dtype="float64") for c in prefixed_cols]
    if all(c in df_slice.columns for c in ARG_NAMES):
        return [df_slice[c].to_numpy(dtype="float64") for c in ARG_NAMES]
    missing = [c for c in prefixed_cols if c not in df_slice.columns]
    raise KeyError(
        f"Missing feature columns for {pair}: {', '.join(missing[:3])}"
    )


def get_feature_values(
    row,
    pair: str = PRIMARY_PAIR,
) -> List[float]:
    """Return model input scalars from either prefixed or plain feature keys."""
    prefixed_cols = [f"{pair}_{c}" for c in ARG_NAMES]
    if all(c in row for c in prefixed_cols):
        return [float(row[c]) for c in prefixed_cols]
    if all(c in row for c in ARG_NAMES):
        return [float(row[c]) for c in ARG_NAMES]
    missing = [c for c in prefixed_cols if c not in row]
    raise KeyError(
        f"Missing feature values for {pair}: {', '.join(missing[:3])}"
    )


def periods_per_day(interval: str = TIMEFRAME) -> int:
    """Infer the number of bars per day from the timeframe string."""
    if interval.endswith("m"):
        minutes = int(interval[:-1])
        return max(1, 1440 // minutes)
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return max(1, 24 // hours)
    if interval.endswith("d"):
        return 1
    raise ValueError(f"Unsupported timeframe: {interval}")


def compute_daily_metrics(
    net_ret: np.ndarray,
    target_return: float = DAILY_TARGET_PCT,
    max_loss: float = DAILY_MAX_LOSS_PCT,
    interval: str = TIMEFRAME,
    cvar_alpha: float = DAILY_CVAR_ALPHA,
) -> Dict:
    """Aggregate bar-level returns into daily target-oriented metrics."""
    bars = periods_per_day(interval)
    daily_returns = []
    for start in range(0, len(net_ret), bars):
        seg = net_ret[start:start + bars]
        if len(seg) == 0:
            continue
        daily_returns.append(float(np.prod(1.0 + seg) - 1.0))

    daily_returns = np.asarray(daily_returns, dtype="float64")
    return summarize_period_returns(
        daily_returns,
        target_return=target_return,
        max_loss=max_loss,
        cvar_alpha=cvar_alpha,
    )


def summarize_period_returns(
    period_returns: np.ndarray,
    target_return: float = DAILY_TARGET_PCT,
    max_loss: float = DAILY_MAX_LOSS_PCT,
    cvar_alpha: float = DAILY_CVAR_ALPHA,
) -> Dict:
    """Summarize a sequence of period returns against a target-return objective."""
    daily_returns = np.asarray(period_returns, dtype="float64")
    if len(daily_returns) == 0:
        return {
            "daily_returns": daily_returns,
            "avg_daily_return": 0.0,
            "median_daily_return": 0.0,
            "daily_win_rate": 0.0,
            "daily_target_hit_rate": 0.0,
            "daily_shortfall_sum": 0.0,
            "daily_shortfall_mean": 0.0,
            "daily_excess_sum": 0.0,
            "worst_day": 0.0,
            "best_day": 0.0,
            "cvar_alpha": cvar_alpha,
            "cvar": 0.0,
            "max_loss_breach_rate": 0.0,
        }

    shortfall = np.maximum(target_return - daily_returns, 0.0)
    excess = np.maximum(daily_returns - target_return, 0.0)
    tail_n = max(1, int(np.ceil(len(daily_returns) * cvar_alpha)))
    cvar = float(np.mean(np.sort(daily_returns)[:tail_n]))

    return {
        "daily_returns": daily_returns,
        "avg_daily_return": float(np.mean(daily_returns)),
        "median_daily_return": float(np.median(daily_returns)),
        "daily_win_rate": float(np.mean(daily_returns > 0.0)),
        "daily_target_hit_rate": float(np.mean(daily_returns >= target_return)),
        "daily_shortfall_sum": float(np.sum(shortfall)),
        "daily_shortfall_mean": float(np.mean(shortfall)),
        "daily_excess_sum": float(np.sum(excess)),
        "worst_day": float(np.min(daily_returns)),
        "best_day": float(np.max(daily_returns)),
        "cvar_alpha": cvar_alpha,
        "cvar": cvar,
        "max_loss_breach_rate": float(np.mean(daily_returns <= max_loss)),
    }


def summarize_monthly_returns(
    daily_returns: np.ndarray,
    daily_index,
    target_daily_return: float = DAILY_TARGET_PCT,
) -> Dict:
    """Summarize calendar-month performance against a daily target."""
    month_returns = np.asarray(daily_returns, dtype="float64")
    if len(month_returns) == 0:
        return {
            "n_months": 0,
            "month_labels": [],
            "month_days": [],
            "monthly_returns": [],
            "monthly_avg_daily_returns": [],
            "target_monthly_returns": [],
            "monthly_target_hit_rate": 0.0,
            "monthly_avg_daily_target_hit_rate": 0.0,
            "monthly_shortfall_sum": 0.0,
            "monthly_shortfall_mean": 0.0,
            "worst_month": 0.0,
            "best_month": 0.0,
            "avg_monthly_return": 0.0,
        }

    dates = pd.DatetimeIndex(daily_index)
    daily_series = pd.Series(month_returns, index=dates)
    month_groups = daily_series.groupby(pd.Grouper(freq="MS"))

    month_labels = []
    month_days = []
    monthly_returns = []
    monthly_avg_daily_returns = []
    target_monthly_returns = []

    for month_start, seg in month_groups:
        if len(seg) == 0:
            continue
        days = int(len(seg))
        month_labels.append(month_start.strftime("%Y-%m"))
        month_days.append(days)
        monthly_returns.append(float(np.prod(1.0 + seg.to_numpy()) - 1.0))
        monthly_avg_daily_returns.append(float(seg.mean()))
        target_monthly_returns.append(float((1.0 + target_daily_return) ** days - 1.0))

    monthly_returns = np.asarray(monthly_returns, dtype="float64")
    monthly_avg_daily_returns = np.asarray(monthly_avg_daily_returns, dtype="float64")
    target_monthly_returns = np.asarray(target_monthly_returns, dtype="float64")
    shortfall = np.maximum(target_monthly_returns - monthly_returns, 0.0)

    return {
        "n_months": int(len(monthly_returns)),
        "month_labels": month_labels,
        "month_days": month_days,
        "monthly_returns": monthly_returns,
        "monthly_avg_daily_returns": monthly_avg_daily_returns,
        "target_monthly_returns": target_monthly_returns,
        "monthly_target_hit_rate": float(np.mean(monthly_returns >= target_monthly_returns)),
        "monthly_avg_daily_target_hit_rate": float(
            np.mean(monthly_avg_daily_returns >= target_daily_return)
        ),
        "monthly_shortfall_sum": float(np.sum(shortfall)),
        "monthly_shortfall_mean": float(np.mean(shortfall)),
        "worst_month": float(np.min(monthly_returns)),
        "best_month": float(np.max(monthly_returns)),
        "avg_monthly_return": float(np.mean(monthly_returns)),
    }


def prepare_weights(
    desired_pcts: np.ndarray,
    dead_band: float = NO_TRADE_BAND,
) -> np.ndarray:
    """Clip GP outputs and apply the dead-band rebalancing rule."""
    desired_pcts = np.where(np.isfinite(desired_pcts), desired_pcts, 0.0)
    desired_pcts = np.clip(desired_pcts, -500.0, 500.0)
    weights = desired_pcts / 100.0
    delta = np.abs(np.diff(weights, prepend=0.0))
    weights[delta < dead_band / 100.0] = np.nan
    return pd.Series(weights).ffill().fillna(0.0).values


# ─────────────────────────────────────────────────────────────────────────────
# 2. Vectorized Backtester
# ─────────────────────────────────────────────────────────────────────────────
def vectorized_backtest(
    close: np.ndarray,
    desired_pcts: np.ndarray,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION_PCT,
    dead_band: float = NO_TRADE_BAND,
    interval: str = TIMEFRAME,
) -> Dict:
    """Fast vectorized backtest using target-weight rebalancing.

    Parameters
    ----------
    close : price series of traded instrument
    desired_pcts : GP output (-100..+100), target exposure as % of equity
    initial_cash : starting capital
    commission : fee rate per side
    dead_band : minimum pct-point change to trigger a rebalance

    Returns
    -------
    dict with total_return, n_trades, sharpe, max_drawdown, final_equity
    """
    weights = prepare_weights(desired_pcts, dead_band)

    # Price returns
    price_ret = np.diff(close) / close[:-1]

    # Strategy returns: weight[t] determines exposure to return[t+1]
    strat_ret = weights[:-1] * price_ret

    # Transaction costs (round-trip cost on each turnover)
    turnover = np.abs(np.diff(weights, prepend=0.0))
    costs = turnover[:-1] * commission * 2

    net_ret = strat_ret - costs

    # Equity curve
    equity = initial_cash * np.cumprod(1 + net_ret)
    final_equity = float(equity[-1]) if len(equity) > 0 else initial_cash
    total_return = final_equity / initial_cash - 1.0

    n_trades = int(np.sum(turnover > 0.001))

    # Annualised Sharpe (hourly bars → 8766 bars/year)
    if len(net_ret) > 1 and np.std(net_ret) > 1e-12:
        sharpe = float(
            np.mean(net_ret) / np.std(net_ret) * np.sqrt(365.25 * 24)
        )
    else:
        sharpe = 0.0

    # Max drawdown
    cum_eq = np.concatenate([[initial_cash], equity])
    peak = np.maximum.accumulate(cum_eq)
    max_dd = float(np.min(cum_eq / peak - 1.0))

    return {
        "total_return": total_return,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": final_equity,
        "equity_curve": cum_eq,
        "net_ret": net_ret,
        "daily_metrics": compute_daily_metrics(net_ret, interval=interval),
    }


def sequential_vectorized_backtest(
    close: np.ndarray,
    desired_pcts: np.ndarray,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION_PCT,
    dead_band: float = NO_TRADE_BAND,
    interval: str = TIMEFRAME,
) -> Dict:
    """Bar-by-bar replay of target-weight execution.

    This follows the same semantics as ``vectorized_backtest`` but processes one
    bar at a time so it can be used as a live-like validation path.
    """
    desired_pcts = np.where(np.isfinite(desired_pcts), desired_pcts, 0.0)
    desired_pcts = np.clip(desired_pcts, -500.0, 500.0)
    target_weights = desired_pcts / 100.0

    equity = float(initial_cash)
    equity_curve = [float(initial_cash)]
    net_ret = []
    current_weight = 0.0
    n_trades = 0

    for i in range(len(close) - 1):
        target_weight = float(target_weights[i])
        if abs(target_weight - current_weight) < dead_band / 100.0:
            target_weight = current_weight

        turnover = abs(target_weight - current_weight)
        if turnover > 0.001:
            n_trades += 1

        price_ret = float(close[i + 1] / close[i] - 1.0)
        bar_net = target_weight * price_ret - turnover * commission * 2
        equity *= (1.0 + bar_net)
        net_ret.append(bar_net)
        equity_curve.append(float(equity))
        current_weight = target_weight

    net_ret = np.asarray(net_ret, dtype="float64")
    equity_curve = np.asarray(equity_curve, dtype="float64")
    final_equity = float(equity_curve[-1]) if len(equity_curve) else float(initial_cash)
    total_return = final_equity / initial_cash - 1.0

    if len(net_ret) > 1 and np.std(net_ret) > 1e-12:
        sharpe = float(
            np.mean(net_ret) / np.std(net_ret) * np.sqrt(365.25 * 24)
        )
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    max_dd = float(np.min(equity_curve / peak - 1.0))

    return {
        "total_return": total_return,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": final_equity,
        "equity_curve": equity_curve,
        "net_ret": net_ret,
        "daily_metrics": compute_daily_metrics(net_ret, interval=interval),
    }


def daily_target_control_backtest(
    close: np.ndarray,
    desired_pcts: np.ndarray,
    index,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION_PCT,
    dead_band: float = NO_TRADE_BAND,
    interval: str = TIMEFRAME,
    daily_target_pct: float = DAILY_TARGET_PCT,
    daily_stop_pct: float = DAILY_MAX_LOSS_PCT,
) -> Dict:
    """Path-dependent backtest that flattens after reaching daily target/stop."""
    if len(close) < 2:
        return {
            "total_return": 0.0,
            "n_trades": 0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "final_equity": initial_cash,
            "equity_curve": np.asarray([initial_cash], dtype="float64"),
            "net_ret": np.asarray([], dtype="float64"),
            "daily_metrics": compute_daily_metrics(np.asarray([], dtype="float64"), interval=interval),
        }

    weights = prepare_weights(desired_pcts, dead_band)
    idx = pd.DatetimeIndex(index)
    equity = float(initial_cash)
    current_weight = 0.0
    net_ret = []
    equity_curve = [initial_cash]
    n_trades = 0
    day_start_equity = float(initial_cash)
    current_day = idx[0].date()
    gated_flat = False

    for i in range(len(close) - 1):
        bar_day = idx[i].date()
        if bar_day != current_day:
            current_day = bar_day
            day_start_equity = equity
            gated_flat = False

        target_weight = 0.0 if gated_flat else float(weights[i])
        turnover = abs(target_weight - current_weight)
        if turnover > 0.001:
            n_trades += 1

        bar_ret = float(close[i + 1] / close[i] - 1.0)
        bar_net = target_weight * bar_ret - turnover * commission * 2
        equity *= (1.0 + bar_net)
        net_ret.append(bar_net)
        equity_curve.append(equity)
        current_weight = target_weight

        day_return = equity / day_start_equity - 1.0
        if day_return >= daily_target_pct or day_return <= daily_stop_pct:
            gated_flat = True

    net_ret = np.asarray(net_ret, dtype="float64")
    equity_curve = np.asarray(equity_curve, dtype="float64")
    total_return = float(equity / initial_cash - 1.0)

    if len(net_ret) > 1 and np.std(net_ret) > 1e-12:
        sharpe = float(
            np.mean(net_ret) / np.std(net_ret) * np.sqrt(365.25 * 24)
        )
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    max_dd = float(np.min(equity_curve / peak - 1.0))

    return {
        "total_return": total_return,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(equity),
        "equity_curve": equity_curve,
        "net_ret": net_ret,
        "daily_metrics": compute_daily_metrics(net_ret, interval=interval),
    }


def daily_session_backtest(
    df_slice: pd.DataFrame,
    desired_pcts: np.ndarray,
    pair: str = PRIMARY_PAIR,
    initial_cash: float = INITIAL_CASH,
    commission: float = COMMISSION_PCT,
    daily_target_pct: float = DAILY_TARGET_PCT,
    daily_stop_pct: float = DAILY_MAX_LOSS_PCT,
    reward_multiple: float = DEFAULT_REWARD_MULTIPLE,
    trail_activation_pct: float = TRAIL_ACTIVATION_PCT,
    trail_distance_pct: float = TRAIL_DISTANCE_PCT,
    trail_floor_pct: float = TRAIL_FLOOR_PCT,
    entry_threshold: float = DAILY_ENTRY_THRESHOLD,
) -> Dict:
    """One trade per day with fixed risk, trailing profit lock, and hard RR target."""
    if df_slice.empty:
        return {
            "total_return": 0.0,
            "n_trades": 0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "final_equity": initial_cash,
            "equity_curve": np.asarray([initial_cash], dtype="float64"),
            "net_ret": np.asarray([], dtype="float64"),
            "daily_metrics": summarize_period_returns(np.asarray([], dtype="float64")),
            "monthly_metrics": summarize_monthly_returns(
                np.asarray([], dtype="float64"),
                pd.DatetimeIndex([]),
            ),
            "win_rate": 0.0,
            "target_hit_rate": 0.0,
            "trade_log": [],
        }

    idx = pd.DatetimeIndex(df_slice.index)
    close = df_slice[f"{pair}_close"].to_numpy(dtype="float64")
    high = df_slice[f"{pair}_high"].to_numpy(dtype="float64")
    low = df_slice[f"{pair}_low"].to_numpy(dtype="float64")
    signal = np.where(np.isfinite(desired_pcts), desired_pcts, 0.0)
    signal = np.clip(signal, -100.0, 100.0)

    risk_pct = abs(daily_stop_pct)
    gross_target = reward_multiple * risk_pct + 2 * commission
    gross_stop = -risk_pct + 2 * commission
    gross_trail_activation = trail_activation_pct + 2 * commission
    gross_trail_floor = trail_floor_pct + 2 * commission

    equity = float(initial_cash)
    equity_curve = [initial_cash]
    daily_returns = []
    trade_log = []

    unique_days = pd.Index(idx.normalize().unique())
    for day in unique_days:
        pos = np.where(idx.normalize() == day)[0]
        if len(pos) < 2:
            equity_curve.append(equity)
            daily_returns.append(0.0)
            continue

        start = pos[0]
        end = pos[-1]
        entry_signal = float(signal[start])
        if abs(entry_signal) <= entry_threshold:
            equity_curve.append(equity)
            daily_returns.append(0.0)
            trade_log.append({
                "date": str(day.date()),
                "direction": "FLAT",
                "gross_return": 0.0,
                "net_return": 0.0,
                "exit_reason": "no_signal",
            })
            continue

        direction = 1.0 if entry_signal > 0 else -1.0
        entry_price = float(close[start])
        exit_price = float(close[end])
        exit_reason = "eod"
        gross_return = direction * (exit_price / entry_price - 1.0)
        best_favorable = 0.0
        trail_active = False

        for j in pos[1:]:
            bar_high = direction * (float(high[j]) / entry_price - 1.0)
            bar_low = direction * (float(low[j]) / entry_price - 1.0)
            favorable = max(bar_high, bar_low)
            adverse = min(bar_high, bar_low)
            best_favorable = max(best_favorable, favorable)

            dynamic_stop = gross_stop
            if best_favorable >= gross_trail_activation:
                trail_active = True
                dynamic_stop = max(
                    gross_stop,
                    gross_trail_floor,
                    best_favorable - trail_distance_pct,
                )

            stop_hit = adverse <= dynamic_stop
            target_hit = favorable >= gross_target

            if stop_hit and target_hit:
                gross_return = dynamic_stop
                exit_reason = "trail_stop_and_target_same_bar" if trail_active else "stop_and_target_same_bar"
                exit_price = entry_price * (1.0 + direction * gross_return)
                break
            if stop_hit:
                gross_return = dynamic_stop
                exit_reason = "trail_stop" if trail_active and dynamic_stop > gross_stop else "stop"
                exit_price = entry_price * (1.0 + direction * gross_return)
                break
            if target_hit:
                gross_return = gross_target
                exit_reason = "target"
                exit_price = entry_price * (1.0 + direction * gross_target)
                break

        net_return = gross_return - 2 * commission
        equity *= (1.0 + net_return)
        equity_curve.append(equity)
        daily_returns.append(net_return)
        trade_log.append({
            "date": str(day.date()),
            "direction": "LONG" if direction > 0 else "SHORT",
            "signal": entry_signal,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross_return,
            "net_return": net_return,
            "exit_reason": exit_reason,
            "reward_multiple": reward_multiple,
            "trail_active": trail_active,
        })

    daily_returns = np.asarray(daily_returns, dtype="float64")
    equity_curve = np.asarray(equity_curve, dtype="float64")
    total_return = float(equity / initial_cash - 1.0)
    n_trades = int(sum(1 for t in trade_log if t["direction"] != "FLAT"))

    if len(daily_returns) > 1 and np.std(daily_returns) > 1e-12:
        sharpe = float(
            np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365.25)
        )
    else:
        sharpe = 0.0

    peak = np.maximum.accumulate(equity_curve)
    max_dd = float(np.min(equity_curve / peak - 1.0))
    daily_metrics = summarize_period_returns(daily_returns)
    daily_index = pd.DatetimeIndex(unique_days)
    monthly_metrics = summarize_monthly_returns(daily_returns, daily_index)
    win_rate = float(np.mean(daily_returns > 0.0)) if len(daily_returns) else 0.0

    return {
        "total_return": total_return,
        "n_trades": n_trades,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "final_equity": float(equity),
        "equity_curve": equity_curve,
        "net_ret": daily_returns,
        "daily_metrics": daily_metrics,
        "monthly_metrics": monthly_metrics,
        "win_rate": win_rate,
        "target_hit_rate": daily_metrics["daily_target_hit_rate"],
        "reward_multiple": reward_multiple,
        "trade_log": trade_log,
    }


def robust_rr_backtest(
    df_slice: pd.DataFrame,
    desired_pcts: np.ndarray,
    pair: str = PRIMARY_PAIR,
    reward_multiples = ROBUST_REWARD_MULTIPLES,
) -> Dict:
    """Evaluate the same rule across several reward multiples and aggregate results."""
    scenario_results = []
    for multiple in reward_multiples:
        scenario_results.append(
            daily_session_backtest(
                df_slice,
                desired_pcts,
                pair=pair,
                reward_multiple=float(multiple),
            )
        )

    primary = next(
        (r for r in scenario_results if abs(r["reward_multiple"] - DEFAULT_REWARD_MULTIPLE) < 1e-9),
        scenario_results[0],
    )
    monthly_shortfall_sum = float(
        np.mean([r["monthly_metrics"]["monthly_shortfall_sum"] for r in scenario_results])
    )
    monthly_target_hit_rate = float(
        np.mean([r["monthly_metrics"]["monthly_target_hit_rate"] for r in scenario_results])
    )
    max_drawdown = float(np.mean([abs(r["max_drawdown"]) for r in scenario_results]))
    cvar = float(np.mean([r["daily_metrics"]["cvar"] for r in scenario_results]))

    return {
        "primary": primary,
        "scenario_results": scenario_results,
        "aggregate": {
            "monthly_shortfall_sum": monthly_shortfall_sum,
            "monthly_target_hit_rate": monthly_target_hit_rate,
            "avg_abs_max_drawdown": max_drawdown,
            "avg_cvar": cvar,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. GP Primitives & Framework
# ─────────────────────────────────────────────────────────────────────────────
def pdiv(a, b):
    return np.divide(a, b, out=np.copy(a).astype(float), where=np.abs(b) > 1e-8)


def gt_signal(a, b):
    return np.where(a > b, 100.0, -100.0)


def neg(a):
    return np.negative(a)


pset = gp.PrimitiveSet("CRYPTO", len(ARG_NAMES), prefix="inp")

for op in (np.add, np.subtract, np.multiply):
    pset.addPrimitive(op, 2)
pset.addPrimitive(pdiv, 2, name="pdiv")

for f, name in [(np.sin, "sin"), (np.cos, "cos"), (np.tanh, "tanh")]:
    pset.addPrimitive(f, 1, name=name)

pset.addPrimitive(gt_signal, 2, name="gt")
pset.addPrimitive(neg, 1, name="neg")
pset.addPrimitive(np.abs, 1, name="abs")

pset.addEphemeralConstant("rand", functools.partial(random.uniform, -1.0, 1.0))

for i, name in enumerate(ARG_NAMES):
    pset.renameArguments(**{f"inp{i}": name})

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=MAX_DEPTH)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=len, max_value=MAX_LEN))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=MAX_LEN))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fitness Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_individual(
    ind, df_slice: pd.DataFrame,
) -> Tuple[float]:
    """Evaluate a GP individual via vectorized backtest."""
    global _eval_count
    _eval_count += 1

    try:
        func = toolbox.compile(expr=ind)
        cols = get_feature_arrays(df_slice, PRIMARY_PAIR)
        desired_pcts = func(*cols)
        robust = robust_rr_backtest(df_slice, desired_pcts, PRIMARY_PAIR)
        result = robust["primary"]
        aggregate = robust["aggregate"]
        daily = result["daily_metrics"]
        monthly = result["monthly_metrics"]

        if len(daily["daily_returns"]) < 30 or result["n_trades"] < 5:
            return (1e6,)

        if result["max_drawdown"] < -0.25:
            return (1e6,)

        if daily["worst_day"] <= -0.03:
            return (1e6,)

        score = 0.0
        # 하루 0.5% 수익 달성을 강제하기 위한 극단적 페널티/보상
        score += daily["daily_shortfall_sum"] * 100000.0
        score += monthly["monthly_shortfall_sum"] * 50000.0
        score += (1.0 - daily["daily_target_hit_rate"]) * 100000.0
        score += max(0.0, 0.005 - daily["avg_daily_return"]) * 100000.0
        
        score += max(0.0, -aggregate["avg_cvar"]) * 750.0
        score += max(0.0, -daily["worst_day"]) * 1000.0
        score += max(0.0, -daily["avg_daily_return"]) * 2000.0
        score += daily["max_loss_breach_rate"] * 10000.0
        score += max(0.0, aggregate["avg_abs_max_drawdown"] - 0.15) * 250.0
        score += (1.0 - aggregate["monthly_target_hit_rate"]) * 10000.0
        
        score -= daily["daily_target_hit_rate"] * 20000.0
        score -= daily["daily_win_rate"] * 2000.0
        score -= daily["avg_daily_return"] * 15000.0
        score -= result["total_return"] * 500.0

        # Bloat Control Penalty
        tree_size = len(ind)
        if tree_size > 40:
            score += (tree_size - 40) * 0.005

        return (score,)
    except Exception:
        return (1e6,)


def _evaluate_with_train(ind):
    return evaluate_individual(ind, _evaluate_with_train.df)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evolution Loop
# ─────────────────────────────────────────────────────────────────────────────
def run_evolution(
    train_df: pd.DataFrame,
    pop_size: int = POP_SIZE,
    n_gen: int = N_GEN,
) -> tools.HallOfFame:
    """Run GP evolution and return Hall of Fame (top 10)."""
    global _eval_count, _start_time
    _eval_count = 0
    _start_time = time.time()

    _evaluate_with_train.df = train_df
    toolbox.register("evaluate", _evaluate_with_train)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(10, similar=lambda a, b: a == b)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    print(f"\nEvolution: pop={pop_size}, gen={n_gen}, cx={P_CX}, mut={P_MUT}")
    print(f"{'Gen':>4} | {'Min Fitness':>12} | {'Avg Fitness':>12} | "
          f"{'Evals':>8} | {'Time':>8}")
    print("-" * 62)

    for gen in range(1, n_gen + 1):
        # Evaluate individuals missing fitness
        invalid = [i for i in pop if not i.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)
        hof.update(pop)

        # Selection & variation
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CX:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for m in offspring:
            if random.random() < P_MUT:
                toolbox.mutate(m)
                del m.fitness.values

        pop[:] = offspring

        # Evaluate newly created offspring
        invalid = [i for i in pop if not i.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        record = stats.compile(pop)
        elapsed = time.time() - _start_time
        print(
            f"{gen:4d} | {record['min']:12.6f} | {record['avg']:12.6f} | "
            f"{_eval_count:8,d} | {elapsed:7.1f}s"
        )

    total_time = time.time() - _start_time
    print(f"\nEvolution complete: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Evaluations: {_eval_count:,}")
    print(f"Best fitness: {hof[0].fitness.values[0]:.6f} "
          f"(tree: {len(hof[0])} nodes)")
    return hof


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validation & Testing
# ─────────────────────────────────────────────────────────────────────────────
def backtest_on_slice(
    individual, df_slice: pd.DataFrame, label: str,
) -> Dict:
    """Run full backtest of an individual on a data slice."""
    func = toolbox.compile(expr=individual)
    cols = get_feature_arrays(df_slice, PRIMARY_PAIR)
    desired_pcts = func(*cols)
    result = daily_session_backtest(df_slice, desired_pcts, PRIMARY_PAIR)

    print(f"\n=== {label} ===")
    print(f"  Return:       {result['total_return']*100:+.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
    print(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
    print(f"  Trades:       {result['n_trades']}")
    print(f"  Win Rate:     {result.get('win_rate', 0.0)*100:.1f}%")
    print(f"  Reward R:     1:{result.get('reward_multiple', DEFAULT_REWARD_MULTIPLE):.1f}")
    print(f"  Final Equity: ${result['final_equity']:,.2f}")
    daily = result["daily_metrics"]
    print(f"  Avg Daily:    {daily['avg_daily_return']*100:+.2f}%")
    print(f"  Daily Win:    {daily['daily_win_rate']*100:.1f}%")
    print(f"  Target Hit:   {daily['daily_target_hit_rate']*100:.1f}% "
          f"(>= {DAILY_TARGET_PCT*100:.2f}%/day)")
    print(f"  Worst Day:    {daily['worst_day']*100:.2f}%")
    monthly = result["monthly_metrics"]
    if monthly["n_months"] > 0:
        print(f"  Monthly Hit:  {monthly['monthly_target_hit_rate']*100:.1f}%")
        print(f"  Avg Month:    {monthly['avg_monthly_return']*100:+.2f}%")
        print(f"  Worst Month:  {monthly['worst_month']*100:.2f}%")
    return result


def select_best_on_validation(
    hof: tools.HallOfFame, val_df: pd.DataFrame,
):
    """Pick best individual from HoF by validation performance."""
    print("\nEvaluating Hall of Fame on validation set...")
    scores = []
    for i, ind in enumerate(hof):
        score = evaluate_individual(ind, val_df)[0]
        scores.append(score)
        print(f"  #{i+1}: fitness={score:.6f}, size={len(ind)} nodes")

    best_idx = int(np.argmin(scores))
    print(f"Best: #{best_idx+1} (val fitness: {scores[best_idx]:.6f})")
    return hof[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Genetic Programming Crypto Trading Strategy")
    print("=" * 62)

    # 1) Load data
    print("\n[Phase 1] Data Loading")
    df_all = load_all_pairs()
    train_df, val_df, test_df = split_dataset(df_all)

    # 2) Evolution
    print("\n[Phase 2] GP Evolution")
    hof = run_evolution(train_df)

    # 3) Validation
    print("\n[Phase 3] Validation")
    best = select_best_on_validation(hof, val_df)

    # 4) Out-of-sample test
    print("\n[Phase 4] Out-of-Sample Test")
    test_result = backtest_on_slice(best, test_df, "TEST (Out-of-Sample)")

    # 5) Full-period backtest for reference
    print("\n[Phase 5] Full-Period Backtest")
    backtest_on_slice(best, df_all, "FULL PERIOD")

    # 6) Save model & metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_crypto_gp.dill"
    with open(model_path, "wb") as f:
        dill.dump(best, f)
    print(f"\nModel saved: {model_path}")

    meta = {
        "pairs": PAIRS,
        "primary_pair": PRIMARY_PAIR,
        "timeframe": TIMEFRAME,
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period": f"{VAL_START} ~ {VAL_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "pop_size": POP_SIZE,
        "n_gen": N_GEN,
        "tree_size": len(best),
        "fitness": float(best.fitness.values[0]),
        "test_return": test_result["total_return"],
        "test_sharpe": test_result["sharpe"],
        "test_max_dd": test_result["max_drawdown"],
        "test_daily_win_rate": test_result["daily_metrics"]["daily_win_rate"],
        "test_daily_target_hit_rate": test_result["daily_metrics"]["daily_target_hit_rate"],
        "daily_target_pct": DAILY_TARGET_PCT,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = MODELS_DIR / "best_crypto_gp_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    print(f"\nGP tree expression:\n  {best}")


if __name__ == "__main__":
    main()
