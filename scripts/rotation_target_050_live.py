#!/usr/bin/env python3
"""Live/demo execution for the target-0.5% rotation strategy.

The strategy has two layers:
1. Daily aggressive rotation core across BTC/ETH/SOL/XRP.
2. BTC intraday overlay only on core-flat days.

This runner uses Binance USD-M via CCXT and defaults to Binance Demo Trading,
which replaces the deprecated USD-M futures sandbox/testnet path.
"""

from __future__ import annotations

import atexit
import argparse
import json
import os
import signal
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import ccxt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backtest_cash_filtered_rotation import build_daily_close, build_target_weights
from backtest_rotation_target_050 import (
    BEST_CORE_PARAMS,
    BEST_OVERLAY_PARAMS,
    MEAN_TARGET_LEVERAGE,
)
from gp_crypto_evolution import (
    COMMISSION_PCT,
    DAILY_MAX_LOSS_PCT,
    INITIAL_CASH,
    MODELS_DIR,
    PAIRS,
    TIMEFRAME,
    fetch_klines,
    load_all_pairs,
)

load_dotenv()

PAIR_TO_MARKET = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
    "SOLUSDT": "SOL/USDT:USDT",
    "XRPUSDT": "XRP/USDT:USDT",
}
MARKET_TO_PAIR = {v: k for k, v in PAIR_TO_MARKET.items()}
STATE_PATH = MODELS_DIR / "rotation_target_050_live_state.json"
DEFAULT_EXCHANGE_LEVERAGE = 5
REBALANCE_NOTIONAL_BAND_USD = float(os.getenv("REBALANCE_NOTIONAL_BAND_USD", "25"))
CORE_KILL_SWITCH_SIGMA_MULTIPLE = 1.5
CORE_KILL_SWITCH_MIN_PCT = 0.06
CORE_KILL_SWITCH_MAX_PCT = 0.10
SHUTDOWN_PROTECTION_RISK_PCT = abs(DAILY_MAX_LOSS_PCT)
SHUTDOWN_PROTECTION_REWARD_MULTIPLE = 2.5
SHUTDOWN_PROTECTION_WORKING_TYPE = "MARK_PRICE"
PROTECTION_CLIENT_PREFIX = "epiP"
UTC = timezone.utc
EPSILON = 1e-12
RUNTIME_CONTEXT: dict[str, Any] = {
    "args": None,
    "state": None,
    "exchange": None,
    "shutdown_requested": False,
    "shutdown_reason": None,
}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_NOTIFICATIONS_ENABLED = os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def resolve_telegram_chat_ids() -> list[int]:
    values: list[int] = []
    raw_values = [
        os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""),
        os.getenv("TELEGRAM_CHAT_ID", ""),
    ]
    for raw in raw_values:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(int(item))
            except ValueError:
                continue
    deduped: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


TELEGRAM_NOTIFY_CHAT_IDS = resolve_telegram_chat_ids()


def resolve_strategy_profile() -> str:
    return "mean"


def resolve_default_leverage() -> float:
    explicit = os.getenv("STRATEGY_LEVERAGE")
    if explicit:
        return float(explicit)
    return MEAN_TARGET_LEVERAGE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the target-0.5% rotation strategy on Binance Demo Trading.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Show the current strategy plan without placing orders.")
    status.add_argument("--equity", type=float, default=INITIAL_CASH)
    status.add_argument("--effective-date", default=None, help="Override the trading date in YYYY-MM-DD.")

    run_once = subparsers.add_parser("run-once", help="Reconcile demo positions with the strategy once.")
    run_once.add_argument("--execute", action="store_true", help="Actually place demo orders.")
    run_once.add_argument("--equity", type=float, default=None, help="Override equity if exchange balance is unavailable.")
    run_once.add_argument("--effective-date", default=None, help="Override the trading date in YYYY-MM-DD.")
    run_once.add_argument("--mode", choices=["demo", "live"], default=os.getenv("BINANCE_MODE", "demo"))
    run_once.add_argument("--leverage", type=float, default=resolve_default_leverage())
    run_once.add_argument("--state-path", default=str(STATE_PATH))

    loop = subparsers.add_parser("loop", help="Keep reconciling the strategy on a schedule.")
    loop.add_argument("--execute", action="store_true", help="Actually place demo orders.")
    loop.add_argument("--mode", choices=["demo", "live"], default=os.getenv("BINANCE_MODE", "demo"))
    loop.add_argument("--leverage", type=float, default=resolve_default_leverage())
    loop.add_argument("--poll-seconds", type=int, default=60)
    loop.add_argument("--state-path", default=str(STATE_PATH))
    loop.add_argument("--equity", type=float, default=None)

    sync_state = subparsers.add_parser("sync-state", help="Sync local state with exchange positions and cancel managed protective orders.")
    sync_state.add_argument("--execute", action="store_true", help="Actually cancel managed protective orders on the exchange.")
    sync_state.add_argument("--mode", choices=["demo", "live"], default=os.getenv("BINANCE_MODE", "demo"))
    sync_state.add_argument("--leverage", type=float, default=resolve_default_leverage())
    sync_state.add_argument("--effective-date", default=None, help="Override the trading date in YYYY-MM-DD.")
    sync_state.add_argument("--state-path", default=str(STATE_PATH))

    shutdown_protect = subparsers.add_parser("shutdown-protect", help="Install server-side reduce-only SL/TP protection for open positions.")
    shutdown_protect.add_argument("--execute", action="store_true", help="Actually place protective orders.")
    shutdown_protect.add_argument("--mode", choices=["demo", "live"], default=os.getenv("BINANCE_MODE", "demo"))
    shutdown_protect.add_argument("--state-path", default=str(STATE_PATH))

    close_all = subparsers.add_parser("close-all", help="Close all open positions and clear local strategy state.")
    close_all.add_argument("--execute", action="store_true", help="Actually place flatten orders.")
    close_all.add_argument("--mode", choices=["demo", "live"], default=os.getenv("BINANCE_MODE", "demo"))
    close_all.add_argument("--state-path", default=str(STATE_PATH))

    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(UTC)


def telegram_ready() -> bool:
    return TELEGRAM_NOTIFICATIONS_ENABLED and bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_NOTIFY_CHAT_IDS)


def send_telegram_notification(text: str) -> bool:
    if not telegram_ready():
        return False

    payload_text = text.strip()
    if not payload_text:
        return False

    delivered = False
    for chat_id in TELEGRAM_NOTIFY_CHAT_IDS:
        params = urlencode(
            {
                "chat_id": str(chat_id),
                "text": payload_text,
                "disable_web_page_preview": "true",
            }
        ).encode()
        request = Request(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            if not data.get("ok"):
                print(f"[WARN] Telegram send failed for chat_id={chat_id}: {data}")
                continue
            delivered = True
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"[WARN] Telegram notification error for chat_id={chat_id}: {exc}")
    return delivered


def side_label_from_qty(qty: float) -> str:
    if qty > EPSILON:
        return "롱"
    if qty < -EPSILON:
        return "숏"
    return "플랫"


def side_label_from_name(side: str) -> str:
    return {"LONG": "롱", "SHORT": "숏"}.get(str(side).upper(), str(side))


def format_notification_price(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${float(value):,.4f}"


def format_notification_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:+.2f}%"


def build_core_rebalance_notification(actions: list[dict[str, Any]], title: str = "코어 리밸런싱 체결") -> str | None:
    lines: list[str] = []
    for action in actions:
        if not action.get("placed"):
            continue
        current_qty = float(action.get("current_qty", 0.0))
        target_qty = float(action.get("target_qty", 0.0))
        if abs(current_qty) <= EPSILON and abs(target_qty) > EPSILON:
            event = "신규 진입"
        elif abs(current_qty) > EPSILON and abs(target_qty) <= EPSILON:
            event = "전량 청산"
        elif current_qty * target_qty < 0:
            event = "포지션 전환"
        elif abs(target_qty) > abs(current_qty):
            event = "포지션 증액"
        else:
            event = "포지션 감축"
        lines.append(
            f"- {action['pair']}: {event} | 목표 {side_label_from_qty(target_qty)} {abs(target_qty):.6f} | "
            f"체결 {action['side']} {float(action.get('amount', 0.0)):.6f} | 기준가 {format_notification_price(action.get('price'))}"
        )
    if not lines:
        return None
    return "\n".join([title, *lines])


def build_overlay_entry_notification(opened: dict[str, Any], effective_day: str, leverage: float) -> str | None:
    if not opened.get("placed"):
        return None
    direction = "롱" if opened.get("side") == "buy" else "숏"
    return "\n".join(
        [
            "오버레이 진입",
            f"- 적용일: {effective_day}",
            f"- 방향: {direction}",
            f"- 수량: {float(opened.get('amount', 0.0)):.6f}",
            f"- 체결가: {format_notification_price(opened.get('price'))}",
            f"- 전략 레버리지: {float(leverage):.2f}x",
        ]
    )


def translate_overlay_exit_reason(reason: str) -> str:
    mapping = {
        "target": "익절",
        "stop": "손절",
        "trail_stop": "트레일링 손절",
        "stop_and_target_same_bar": "손절/익절 동시 충돌",
        "trail_stop_and_target_same_bar": "트레일링 손절/익절 동시 충돌",
        "forced_eod_close": "일자 변경 강제 청산",
        "would_force_eod_close": "일자 변경 예정 청산",
    }
    return mapping.get(reason, reason)


def build_overlay_exit_notification(result: dict[str, Any]) -> str | None:
    close_action = result.get("close_action") or {}
    if not close_action.get("placed"):
        return None
    reason = translate_overlay_exit_reason(str(result.get("exit_reason") or result.get("status") or "청산"))
    return "\n".join(
        [
            "오버레이 청산",
            f"- 사유: {reason}",
            f"- 방향: {side_label_from_name(str(result.get('overlay_side', '')))}",
            f"- 수량: {abs(float(close_action.get('current_qty', 0.0))):.6f}",
            f"- 예상 손익률: {format_notification_pct(result.get('exit_return'))}",
        ]
    )


def build_kill_switch_notification(report: dict[str, Any]) -> str | None:
    if report.get("status") != "triggered":
        return None
    flatten_actions = report.get("flatten_actions") or []
    closed_pairs = [action.get("pair") for action in flatten_actions if action.get("placed")]
    return "\n".join(
        [
            "코어 킬 스위치 발동",
            f"- 손실률: {format_notification_pct(report.get('trigger_return'))}",
            f"- 임계값: {format_notification_pct(report.get('kill_switch_pct'))}",
            f"- 청산 종목: {', '.join(closed_pairs) if closed_pairs else '없음'}",
        ]
    )


def build_daily_briefing(plan: dict[str, Any], equity: float) -> str:
    lines = [
        "일일 시작 브리핑",
        f"- 적용일: {plan['effective_day']}",
        f"- 세션: {str(plan['session_type']).upper()}",
        f"- 자산: ${float(equity):,.2f}",
        f"- 전략 레버리지: {float(plan['leverage']):.2f}x",
        f"- 코어 총 레버리지: {float(plan['core_gross_leverage']):.3f}x",
    ]
    if plan["session_type"] == "core":
        active = [f"{pair} {float(weight):+.2%}" for pair, weight in plan["core_weights"].items() if abs(float(weight)) > EPSILON]
        lines.append(f"- 코어 진입: {', '.join(active) if active else '없음'}")
    elif plan["session_type"] == "overlay":
        overlay = plan["overlay"]
        lines.append(f"- 오버레이 방향: {overlay['side']}")
        lines.append(f"- 오버레이 신호: {float(overlay['signal_pct']):+.1f}%")
    else:
        lines.append("- 오늘은 관망 예정")
    return "\n".join(lines)


def build_daily_summary(effective_day: str, start_equity: float, end_equity: float) -> str:
    pnl = float(end_equity) - float(start_equity)
    ret = float(end_equity / start_equity - 1.0) if abs(start_equity) > EPSILON else 0.0
    return "\n".join(
        [
            "일일 마감 요약",
            f"- 적용일: {effective_day}",
            f"- 시작 자산: ${float(start_equity):,.2f}",
            f"- 종료 자산: ${float(end_equity):,.2f}",
            f"- 손익: ${pnl:,.2f}",
            f"- 수익률: {ret * 100:+.2f}%",
        ]
    )


def build_session_change_notification(previous_session: str, current_session: str, effective_day: str) -> str:
    return "\n".join(
        [
            "세션 상태 변경",
            f"- 적용일: {effective_day}",
            f"- 이전 세션: {previous_session.upper()}",
            f"- 현재 세션: {current_session.upper()}",
        ]
    )


def dispatch_notifications(messages: list[str]) -> None:
    for message in messages:
        message = message.strip()
        if not message:
            continue
        send_telegram_notification(message)


def normalize_day(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def get_demo_credentials() -> tuple[str, str]:
    api_key = os.getenv("BINANCE_DEMO_API_KEY", "") or os.getenv("BINANCE_API_KEY", "")
    secret = (
        os.getenv("BINANCE_DEMO_API_SECRET", "")
        or os.getenv("BINANCE_SECRET", "")
        or os.getenv("BINANCE_SECRET_KEY", "")
    )
    return api_key, secret


def get_exchange(mode: str) -> ccxt.binanceusdm:
    api_key, secret = get_demo_credentials()
    exchange = ccxt.binanceusdm(
        {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
    )
    if mode == "demo":
        exchange.enable_demo_trading(True)
    return exchange


def register_runtime_context(args: argparse.Namespace, state: dict[str, Any], exchange: ccxt.binanceusdm | None) -> None:
    RUNTIME_CONTEXT["args"] = args
    RUNTIME_CONTEXT["state"] = state
    RUNTIME_CONTEXT["exchange"] = exchange


def clear_runtime_context() -> None:
    shutdown_requested = bool(RUNTIME_CONTEXT.get("shutdown_requested"))
    shutdown_reason = RUNTIME_CONTEXT.get("shutdown_reason")
    RUNTIME_CONTEXT["args"] = None
    RUNTIME_CONTEXT["state"] = None
    RUNTIME_CONTEXT["exchange"] = None
    if not shutdown_requested:
        RUNTIME_CONTEXT["shutdown_requested"] = False
        RUNTIME_CONTEXT["shutdown_reason"] = None
    else:
        RUNTIME_CONTEXT["shutdown_requested"] = True
        RUNTIME_CONTEXT["shutdown_reason"] = shutdown_reason


def request_shutdown(signum: int, _frame: Any) -> None:
    try:
        signame = signal.Signals(signum).name
    except Exception:
        signame = str(signum)
    RUNTIME_CONTEXT["shutdown_requested"] = True
    RUNTIME_CONTEXT["shutdown_reason"] = signame
    raise SystemExit(128 + int(signum))


def build_exit_client_id(tag: str, pair: str) -> str:
    stamp = utc_now().strftime("%m%d%H%M%S")
    symbol_code = pair.replace("USDT", "")[:4]
    return f"{PROTECTION_CLIENT_PREFIX}{tag}{symbol_code}{stamp}"[:32]


def extract_client_order_id(order: dict[str, Any]) -> str:
    info = order.get("info", {}) if isinstance(order.get("info"), dict) else {}
    for key in ("clientOrderId", "clientAlgoId", "newClientStrategyId", "newClientOrderId"):
        value = order.get(key)
        if value:
            return str(value)
        value = info.get(key)
        if value:
            return str(value)
    return ""


def is_managed_protection_order(order: dict[str, Any]) -> bool:
    client_order_id = extract_client_order_id(order)
    return client_order_id.startswith(PROTECTION_CLIENT_PREFIX)


def load_state(path: str | Path) -> dict[str, Any]:
    state_file = Path(path)
    if not state_file.exists():
        return {
            "overlay": None,
            "overlay_completed_day": None,
            "core_day_state": None,
            "shutdown_protection": [],
            "notification_state": {
                "last_briefing_effective_day": None,
                "last_briefing_equity": None,
                "last_summary_effective_day": None,
                "last_session_effective_day": None,
                "last_session_type": None,
            },
            "updated_at": None,
        }
    with open(state_file, "r") as f:
        state = json.load(f)
    state.setdefault("overlay", None)
    state.setdefault("overlay_completed_day", None)
    state.setdefault("core_day_state", None)
    state.setdefault("shutdown_protection", [])
    state.setdefault("notification_state", {})
    state["notification_state"].setdefault("last_briefing_effective_day", None)
    state["notification_state"].setdefault("last_briefing_equity", None)
    state["notification_state"].setdefault("last_summary_effective_day", None)
    state["notification_state"].setdefault("last_session_effective_day", None)
    state["notification_state"].setdefault("last_session_type", None)
    state.setdefault("updated_at", None)
    return state


def save_state(path: str | Path, state: dict[str, Any]) -> None:
    state_file = Path(path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now().isoformat()
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def clear_strategy_state(state: dict[str, Any]) -> None:
    state["overlay"] = None
    state["overlay_completed_day"] = None
    state["core_day_state"] = None
    state["shutdown_protection"] = []


def latest_closed_day(close: pd.DataFrame) -> pd.Timestamp:
    return normalize_day(close.index.max())


def resolve_effective_day(close: pd.DataFrame, effective_date: str | None) -> tuple[pd.Timestamp, pd.Timestamp]:
    market_day = latest_closed_day(close)
    if effective_date is None:
        effective_day = market_day + pd.Timedelta(days=1)
    else:
        effective_day = normalize_day(effective_date)
    return market_day, effective_day


def compute_overlay_signal(close: pd.DataFrame, market_day: pd.Timestamp) -> dict[str, Any]:
    btc_ret1 = float(close["BTCUSDT"].pct_change(BEST_OVERLAY_PARAMS.momentum_lookback).loc[market_day])
    breadth3 = float((close.pct_change(BEST_OVERLAY_PARAMS.breadth_lookback) > 0.0).mean(axis=1).loc[market_day])
    signal = 0.0
    side = "FLAT"
    if np.isfinite(breadth3) and breadth3 >= BEST_OVERLAY_PARAMS.breadth_threshold:
        if btc_ret1 >= BEST_OVERLAY_PARAMS.momentum_threshold:
            signal = 100.0
            side = "LONG"
        else:
            signal = -100.0
            side = "SHORT"
    return {
        "btc_ret1": btc_ret1,
        "breadth3": breadth3,
        "signal_pct": signal,
        "side": side,
    }


def load_live_close(recent_days: int = 10) -> tuple[pd.DataFrame, dict[str, float]]:
    df_all = load_all_pairs()
    close_series: dict[str, pd.Series] = {}
    for pair in PAIRS:
        close_series[pair] = df_all[f"{pair}_close"].rename(pair).sort_index()

    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=int(recent_days))
    for pair in PAIRS:
        try:
            recent = fetch_klines(pair, TIMEFRAME, start_dt, end_dt)
        except Exception:
            recent = pd.DataFrame()
        if recent.empty:
            continue
        recent_close = recent["close"].rename(pair)
        merged = pd.concat([close_series[pair], recent_close], axis=0).sort_index()
        close_series[pair] = merged[~merged.index.duplicated(keep="last")]

    intraday_close = pd.concat([close_series[pair] for pair in PAIRS], axis=1).sort_index()
    intraday_close = intraday_close.sort_index()
    daily_close = intraday_close.resample("1D").last().dropna().sort_index()
    cutoff_day = normalize_day(utc_now())
    if daily_close.index.tz is not None:
        daily_close = daily_close[daily_close.index < cutoff_day.tz_localize("UTC")]
        daily_close.index = daily_close.index.tz_convert(None)
    else:
        daily_close = daily_close[daily_close.index < cutoff_day]

    latest_prices = {}
    for pair in PAIRS:
        latest_prices[pair] = float(intraday_close[pair].dropna().iloc[-1])
    return daily_close, latest_prices


def build_strategy_plan(effective_date: str | None, leverage: float) -> dict[str, Any]:
    try:
        close, latest_prices = load_live_close()
    except Exception:
        df_all = load_all_pairs()
        close = build_daily_close(df_all)
        latest_prices = {pair: float(df_all[f"{pair}_close"].iloc[-1]) for pair in PAIRS}
    if close.empty:
        raise ValueError("No daily close history available")
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)

    market_day, effective_day = resolve_effective_day(close, effective_date)
    if effective_day > market_day + pd.Timedelta(days=1):
        raise ValueError("effective_date cannot be more than one day after the latest available daily close")

    target_weights = build_target_weights(close, BEST_CORE_PARAMS)
    if market_day not in target_weights.index:
        raise ValueError(f"Missing core weights for {market_day.date()}")

    base_core = target_weights.loc[market_day].fillna(0.0).astype("float64")
    core_weights = base_core * float(leverage)
    core_active = bool(core_weights.sum() > 0.0)
    overlay = compute_overlay_signal(close, market_day)
    session_type = "core" if core_active else "overlay" if overlay["signal_pct"] != 0.0 else "flat"

    return {
        "strategy_class": "rotation_target_050_live",
        "strategy_profile": resolve_strategy_profile(),
        "market_day": market_day.date().isoformat(),
        "effective_day": effective_day.date().isoformat(),
        "leverage": float(leverage),
        "core_params": asdict(BEST_CORE_PARAMS),
        "overlay_params": asdict(BEST_OVERLAY_PARAMS),
        "session_type": session_type,
        "core_active": core_active,
        "core_weights": {pair: float(core_weights[pair]) for pair in core_weights.index},
        "core_gross_leverage": float(core_weights.abs().sum()),
        "overlay": overlay,
        "latest_prices": latest_prices,
    }


def print_plan(plan: dict[str, Any], equity: float) -> None:
    print("=" * 72)
    print("  Rotation Target 0.5% Live Plan")
    print("=" * 72)
    print(f"Market Day       : {plan['market_day']}")
    print(f"Effective Day    : {plan['effective_day']}")
    print(f"Strategy Leverage: {plan['leverage']:.2f}x")
    print(f"Session Type     : {plan['session_type'].upper()}")
    print(f"Equity Basis     : ${equity:,.2f}")
    print(f"Core Gross Lev   : {plan['core_gross_leverage']:.3f}x")

    print("\nCore Targets")
    for pair in PAIRS:
        weight = float(plan["core_weights"][pair])
        notional = weight * float(equity)
        print(f"  {pair:8s}  {weight:+8.3%}  ${notional:>11,.2f}")

    overlay = plan["overlay"]
    print("\nOverlay Signal")
    print(f"  btc_ret1      : {overlay['btc_ret1']:+.4%}")
    print(f"  breadth3      : {overlay['breadth3']:.2%}")
    print(f"  signal_pct    : {overlay['signal_pct']:+.1f}%")
    print(f"  overlay_side  : {overlay['side']}")


def extract_total_usdt(balance: dict[str, Any]) -> float:
    if "USDT" in balance and isinstance(balance["USDT"], dict):
        for key in ("total", "free"):
            value = balance["USDT"].get(key)
            if value is not None:
                return float(value)
    for outer in ("total", "free"):
        inner = balance.get(outer, {})
        if isinstance(inner, dict) and "USDT" in inner:
            return float(inner["USDT"])
    raise ValueError("Could not determine USDT equity from exchange balance")


def fetch_equity(exchange: ccxt.binanceusdm) -> float:
    balance = exchange.fetch_balance()
    return extract_total_usdt(balance)


def fetch_last_price(exchange: ccxt.binanceusdm, symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    for key in ("last", "mark", "close"):
        value = ticker.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"No usable last price for {symbol}")


def load_markets(exchange: ccxt.binanceusdm) -> dict[str, Any]:
    return exchange.load_markets()


def fetch_open_position_map(exchange: ccxt.binanceusdm) -> dict[str, dict[str, Any]]:
    positions_by_pair: dict[str, dict[str, Any]] = {}
    try:
        positions = exchange.fetch_positions(list(PAIR_TO_MARKET.values()))
    except Exception:
        return positions_by_pair

    for pos in positions:
        symbol = pos.get("symbol")
        pair = MARKET_TO_PAIR.get(symbol)
        if pair is None:
            continue
        info = pos.get("info", {}) if isinstance(pos.get("info"), dict) else {}
        qty = info.get("positionAmt")
        if qty is None:
            qty = pos.get("contracts", 0.0)
        qty_value = float(qty)
        if abs(qty_value) <= EPSILON:
            continue
        entry_price = info.get("entryPrice")
        if entry_price is None:
            entry_price = pos.get("entryPrice", 0.0)
        mark_price = info.get("markPrice")
        if mark_price is None:
            mark_price = pos.get("markPrice", 0.0)
        margin_mode = info.get("marginType")
        if margin_mode is None:
            margin_mode = pos.get("marginMode")
        leverage = info.get("leverage")
        if leverage is None:
            leverage = pos.get("leverage")
        positions_by_pair[pair] = {
            "pair": pair,
            "symbol": symbol,
            "qty": qty_value,
            "side": "LONG" if qty_value > 0.0 else "SHORT",
            "entry_price": float(entry_price) if entry_price is not None else 0.0,
            "mark_price": float(mark_price) if mark_price is not None else 0.0,
            "margin_mode": str(margin_mode).lower() if margin_mode is not None else None,
            "leverage": float(leverage) if leverage not in (None, "") else None,
            "position_side": info.get("positionSide"),
        }
    return positions_by_pair


def fetch_position_qty_map(exchange: ccxt.binanceusdm) -> dict[str, float]:
    qty_map = {pair: 0.0 for pair in PAIRS}
    for pair, position in fetch_open_position_map(exchange).items():
        qty_map[pair] = float(position["qty"])
    return qty_map


def quantize_price(exchange: ccxt.binanceusdm, symbol: str, price: float) -> float:
    precise = exchange.price_to_precision(symbol, float(price))
    return float(precise)


def ensure_symbol_margin_settings(exchange: ccxt.binanceusdm, symbol: str, leverage: int = DEFAULT_EXCHANGE_LEVERAGE) -> dict[str, Any]:
    result: dict[str, Any] = {"symbol": symbol, "margin_mode": None, "leverage": None, "warnings": []}
    try:
        exchange.set_margin_mode("isolated", symbol)
        result["margin_mode"] = "isolated"
    except Exception as exc:
        result["warnings"].append(f"margin_mode:{exc}")
    try:
        exchange.set_leverage(leverage, symbol)
        result["leverage"] = leverage
    except Exception as exc:
        result["warnings"].append(f"leverage:{exc}")
    return result


def fetch_strategy_protection_orders(exchange: ccxt.binanceusdm, pairs: list[str] | None = None) -> list[dict[str, Any]]:
    open_orders: list[dict[str, Any]] = []
    for pair in PAIRS if pairs is None else pairs:
        symbol = PAIR_TO_MARKET[pair]
        try:
            orders = exchange.fetch_open_orders(symbol, params={"conditional": True, "type": "swap"})
        except Exception:
            continue
        for order in orders:
            if is_managed_protection_order(order):
                open_orders.append(order)
    return open_orders


def cancel_strategy_protection_orders(
    exchange: ccxt.binanceusdm,
    orders: list[dict[str, Any]],
    execute: bool,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for order in orders:
        action = {
            "id": order.get("id"),
            "client_order_id": extract_client_order_id(order),
            "symbol": order.get("symbol"),
            "type": order.get("type"),
            "side": order.get("side"),
            "cancelled": False,
        }
        if execute:
            try:
                exchange.cancel_order(str(order["id"]), str(order["symbol"]), params={"conditional": True, "type": "swap"})
                action["cancelled"] = True
            except Exception as exc:
                action["error"] = str(exc)
        actions.append(action)
    return actions


def quantize_amount(exchange: ccxt.binanceusdm, symbol: str, amount: float) -> float:
    raw_amount = abs(float(amount))
    if raw_amount <= 1e-12:
        return 0.0
    market = exchange.market(symbol)
    amount_precision = market.get("precision", {}).get("amount")
    if amount_precision is not None and raw_amount < float(amount_precision):
        return 0.0
    try:
        precise = float(exchange.amount_to_precision(symbol, raw_amount))
    except ccxt.InvalidOrder:
        return 0.0
    min_amount = (
        market.get("limits", {}).get("amount", {}).get("min")
        if isinstance(market.get("limits"), dict)
        else None
    )
    if min_amount is not None and precise < float(min_amount):
        return 0.0
    return precise


def build_shutdown_protection_prices(side: str, reference_price: float) -> tuple[float, float]:
    risk_pct = SHUTDOWN_PROTECTION_RISK_PCT
    reward_pct = SHUTDOWN_PROTECTION_REWARD_MULTIPLE * risk_pct
    if side == "LONG":
        stop_price = reference_price * (1.0 - risk_pct)
        take_price = reference_price * (1.0 + reward_pct)
    else:
        stop_price = reference_price * (1.0 + risk_pct)
        take_price = reference_price * (1.0 - reward_pct)
    return stop_price, take_price


def place_shutdown_protection_for_position(
    exchange: ccxt.binanceusdm,
    position: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    symbol = str(position["symbol"])
    pair = str(position["pair"])
    qty = float(position["qty"])
    close_side = "sell" if qty > 0.0 else "buy"
    amount = quantize_amount(exchange, symbol, qty)
    reference_price = float(position["mark_price"] or position["entry_price"])
    if amount <= 0.0 or reference_price <= 0.0:
        return {
            "pair": pair,
            "symbol": symbol,
            "status": "skipped",
            "reason": "no_amount_or_reference_price",
        }

    stop_price_raw, take_price_raw = build_shutdown_protection_prices(position["side"], reference_price)
    stop_price = quantize_price(exchange, symbol, stop_price_raw)
    take_price = quantize_price(exchange, symbol, take_price_raw)
    result: dict[str, Any] = {
        "pair": pair,
        "symbol": symbol,
        "qty": qty,
        "reference_price": reference_price,
        "stop_price": stop_price,
        "take_price": take_price,
        "close_side": close_side,
        "amount": amount,
        "status": "planned",
        "orders": [],
    }
    if not execute:
        return result

    stop_params = {
        "stopLossPrice": stop_price,
        "reduceOnly": True,
        "workingType": SHUTDOWN_PROTECTION_WORKING_TYPE,
        "clientOrderId": build_exit_client_id("SL", pair),
    }
    take_params = {
        "takeProfitPrice": take_price,
        "reduceOnly": True,
        "workingType": SHUTDOWN_PROTECTION_WORKING_TYPE,
        "clientOrderId": build_exit_client_id("TP", pair),
    }
    stop_order = exchange.create_order(symbol, "market", close_side, amount, None, stop_params)
    take_order = exchange.create_order(symbol, "market", close_side, amount, None, take_params)
    result["status"] = "placed"
    result["orders"] = [
        {
            "id": stop_order.get("id"),
            "client_order_id": extract_client_order_id(stop_order),
            "type": stop_order.get("type"),
            "side": stop_order.get("side"),
            "stop_price": stop_price,
        },
        {
            "id": take_order.get("id"),
            "client_order_id": extract_client_order_id(take_order),
            "type": take_order.get("type"),
            "side": take_order.get("side"),
            "stop_price": take_price,
        },
    ]
    return result


def sync_overlay_state_with_exchange(
    state: dict[str, Any],
    plan: dict[str, Any],
    position_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "unchanged"}
    overlay = state.get("overlay")
    btc_position = position_map.get("BTCUSDT")
    non_btc_positions = [pair for pair in position_map if pair != "BTCUSDT"]

    if overlay is not None and btc_position is None:
        state["overlay"] = None
        state["shutdown_protection"] = []
        report["status"] = "cleared_missing_btc_position"
        return report

    if overlay is not None and btc_position is not None:
        overlay["symbol"] = btc_position["symbol"]
        overlay["qty"] = float(btc_position["qty"])
        overlay["side"] = "LONG" if float(btc_position["qty"]) > 0.0 else "SHORT"
        overlay["entry_price"] = float(btc_position["entry_price"] or overlay.get("entry_price", 0.0))
        report["status"] = "refreshed_existing_overlay"
        report["qty"] = overlay["qty"]
        return report

    if overlay is None and btc_position is not None and not non_btc_positions and plan["session_type"] == "overlay":
        state["overlay"] = {
            "effective_day": plan["effective_day"],
            "symbol": btc_position["symbol"],
            "side": "LONG" if float(btc_position["qty"]) > 0.0 else "SHORT",
            "qty": float(btc_position["qty"]),
            "entry_price": float(btc_position["entry_price"] or btc_position["mark_price"]),
            "best_favorable": 0.0,
            "trail_active": False,
            "last_bar_ts": None,
            "opened_at": utc_now().isoformat(),
            "recovered": True,
        }
        report["status"] = "recovered_overlay_from_exchange"
        report["qty"] = state["overlay"]["qty"]
        return report

    return report


def sync_exchange_state(
    exchange: ccxt.binanceusdm,
    state: dict[str, Any],
    plan: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    position_map = fetch_open_position_map(exchange)
    protection_orders = fetch_strategy_protection_orders(exchange)
    cancel_actions = cancel_strategy_protection_orders(exchange, protection_orders, execute)
    overlay_report = sync_overlay_state_with_exchange(state, plan, position_map)
    if execute and cancel_actions:
        state["shutdown_protection"] = []
    return {
        "positions": [
            {
                "pair": pair,
                "symbol": pos["symbol"],
                "qty": pos["qty"],
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "mark_price": pos["mark_price"],
                "margin_mode": pos["margin_mode"],
            }
            for pair, pos in position_map.items()
        ],
        "managed_protection_orders": [
            {
                "id": order.get("id"),
                "client_order_id": extract_client_order_id(order),
                "symbol": order.get("symbol"),
                "type": order.get("type"),
                "side": order.get("side"),
            }
            for order in protection_orders
        ],
        "cancel_actions": cancel_actions,
        "overlay_sync": overlay_report,
    }


def install_shutdown_protection(
    exchange: ccxt.binanceusdm,
    state: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    positions = fetch_open_position_map(exchange)
    existing_orders = fetch_strategy_protection_orders(exchange)
    cancel_actions = cancel_strategy_protection_orders(exchange, existing_orders, execute)
    installed = [
        place_shutdown_protection_for_position(exchange, position, execute)
        for _, position in positions.items()
    ]
    if execute:
        state["shutdown_protection"] = installed
    return {
        "status": "placed" if execute else "planned",
        "positions": [
            {
                "pair": pair,
                "symbol": pos["symbol"],
                "qty": pos["qty"],
                "side": pos["side"],
                "entry_price": pos["entry_price"],
                "mark_price": pos["mark_price"],
                "margin_mode": pos["margin_mode"],
            }
            for pair, pos in positions.items()
        ],
        "cancel_actions": cancel_actions,
        "protections": installed,
    }


def compute_core_kill_switch_pct(plan: dict[str, Any]) -> float:
    core_params = plan.get("core_params", {})
    target_vol_ann = float(core_params.get("target_vol_ann", 0.8))
    target_daily_vol = target_vol_ann / np.sqrt(365.25)
    return float(np.clip(
        CORE_KILL_SWITCH_SIGMA_MULTIPLE * target_daily_vol,
        CORE_KILL_SWITCH_MIN_PCT,
        CORE_KILL_SWITCH_MAX_PCT,
    ))


def refresh_core_day_state(
    state: dict[str, Any],
    plan: dict[str, Any],
    equity: float,
) -> dict[str, Any] | None:
    if plan["session_type"] != "core":
        state["core_day_state"] = None
        return None

    threshold = compute_core_kill_switch_pct(plan)
    day_state = state.get("core_day_state")
    if not isinstance(day_state, dict) or day_state.get("effective_day") != plan["effective_day"]:
        day_state = {
            "effective_day": plan["effective_day"],
            "baseline_equity": float(equity),
            "kill_switch_pct": float(threshold),
            "kill_triggered": False,
            "triggered_at": None,
            "trigger_return": None,
            "trigger_equity": None,
            "last_equity": float(equity),
            "last_return": 0.0,
        }
        state["core_day_state"] = day_state
        return day_state

    if day_state.get("baseline_equity") in (None, 0):
        day_state["baseline_equity"] = float(equity)
    day_state["kill_switch_pct"] = float(threshold)
    day_state["last_equity"] = float(equity)
    baseline_equity = float(day_state["baseline_equity"])
    day_state["last_return"] = float(equity / baseline_equity - 1.0) if abs(baseline_equity) > EPSILON else 0.0
    return day_state


def evaluate_core_emergency_kill_switch(
    exchange: ccxt.binanceusdm,
    state: dict[str, Any],
    plan: dict[str, Any],
    equity: float,
    execute: bool,
) -> dict[str, Any]:
    day_state = refresh_core_day_state(state, plan, equity)
    if day_state is None:
        return {"status": "inactive"}

    baseline_equity = float(day_state["baseline_equity"])
    kill_switch_pct = float(day_state["kill_switch_pct"])
    current_return = float(equity / baseline_equity - 1.0) if abs(baseline_equity) > EPSILON else 0.0
    day_state["last_equity"] = float(equity)
    day_state["last_return"] = current_return

    report = {
        "status": "ok",
        "effective_day": plan["effective_day"],
        "baseline_equity": baseline_equity,
        "current_equity": float(equity),
        "current_return": current_return,
        "kill_switch_pct": kill_switch_pct,
        "kill_triggered": bool(day_state.get("kill_triggered", False)),
    }

    if bool(day_state.get("kill_triggered", False)):
        report["status"] = "locked"
        report["triggered_at"] = day_state.get("triggered_at")
        report["trigger_return"] = day_state.get("trigger_return")
        return report

    if current_return > -kill_switch_pct:
        return report

    flat_actions = reconcile_target_positions(exchange, equity, {pair: 0.0 for pair in PAIRS}, execute)
    day_state["kill_triggered"] = True
    day_state["triggered_at"] = utc_now().isoformat()
    day_state["trigger_return"] = current_return
    day_state["trigger_equity"] = float(equity)
    report["status"] = "triggered"
    report["kill_triggered"] = True
    report["triggered_at"] = day_state["triggered_at"]
    report["trigger_return"] = current_return
    report["flatten_actions"] = flat_actions
    return report


def reconcile_target_positions(
    exchange: ccxt.binanceusdm | None,
    equity: float,
    target_weights: dict[str, float],
    execute: bool,
    pairs: list[str] | None = None,
) -> list[dict[str, Any]]:
    pairs = PAIRS if pairs is None else pairs
    current_qtys = {pair: 0.0 for pair in PAIRS}
    latest_prices: dict[str, float] = {}
    actions: list[dict[str, Any]] = []

    if exchange is not None:
        load_markets(exchange)
        current_qtys = fetch_position_qty_map(exchange)

    for pair in pairs:
        symbol = PAIR_TO_MARKET[pair]
        price = fetch_last_price(exchange, symbol) if exchange is not None else None
        if price is None:
            raise ValueError(f"Price unavailable for {pair}")
        latest_prices[pair] = price
        target_notional = float(target_weights.get(pair, 0.0)) * float(equity)
        target_qty = target_notional / price if abs(price) > 1e-12 else 0.0
        current_qty = float(current_qtys.get(pair, 0.0))
        diff_qty = target_qty - current_qty
        diff_notional = abs(diff_qty * price)
        amount = quantize_amount(exchange, symbol, diff_qty) if exchange is not None else abs(diff_qty)
        action = {
            "pair": pair,
            "symbol": symbol,
            "price": price,
            "target_weight": float(target_weights.get(pair, 0.0)),
            "target_notional": target_notional,
            "current_qty": current_qty,
            "target_qty": target_qty,
            "diff_qty": diff_qty,
            "diff_notional": diff_notional,
            "amount": amount,
            "side": "buy" if diff_qty > 0 else "sell",
            "placed": False,
        }
        if diff_notional < REBALANCE_NOTIONAL_BAND_USD:
            action["amount"] = 0.0
            actions.append(action)
            continue
        if amount <= 0.0:
            actions.append(action)
            continue
        if execute and exchange is not None:
            margin_action = ensure_symbol_margin_settings(exchange, symbol)
            if margin_action["warnings"]:
                action["margin_warnings"] = margin_action["warnings"]
            order = exchange.create_market_order(symbol, action["side"], amount)
            action["placed"] = True
            action["order_id"] = order.get("id")
        actions.append(action)
    return actions


def flatten_pairs(
    exchange: ccxt.binanceusdm,
    pairs: list[str],
    execute: bool,
) -> list[dict[str, Any]]:
    qty_map = fetch_position_qty_map(exchange)
    actions = []
    for pair in pairs:
        current_qty = float(qty_map.get(pair, 0.0))
        actions.append(close_pair_position(exchange, pair, current_qty, execute))
    return actions


def fetch_recent_closed_5m_bars(exchange: ccxt.binanceusdm, symbol: str, since_ms: int | None = None, limit: int = 20) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe="5m", since=since_ms, limit=limit)
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv, columns=cols)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    current_bar_open = utc_now().replace(second=0, microsecond=0)
    current_bar_open = current_bar_open - timedelta(minutes=current_bar_open.minute % 5)
    return df[df["time"] < current_bar_open]


def close_pair_position(
    exchange: ccxt.binanceusdm,
    pair: str,
    current_qty: float,
    execute: bool,
) -> dict[str, Any]:
    symbol = PAIR_TO_MARKET[pair]
    amount = quantize_amount(exchange, symbol, current_qty)
    action = {
        "pair": pair,
        "symbol": symbol,
        "current_qty": current_qty,
        "amount": amount,
        "side": "sell" if current_qty > 0 else "buy",
        "placed": False,
    }
    if amount <= 0.0:
        return action
    if execute:
        order = exchange.create_market_order(symbol, action["side"], amount, params={"reduceOnly": True})
        action["placed"] = True
        action["order_id"] = order.get("id")
    return action


def open_overlay_position(
    exchange: ccxt.binanceusdm,
    equity: float,
    side: str,
    leverage: float,
    execute: bool,
) -> dict[str, Any]:
    symbol = PAIR_TO_MARKET["BTCUSDT"]
    margin_action = ensure_symbol_margin_settings(exchange, symbol)
    price = fetch_last_price(exchange, symbol)
    direction = 1.0 if side == "LONG" else -1.0
    target_notional = float(equity) * float(leverage) * direction
    target_qty = target_notional / price if abs(price) > 1e-12 else 0.0
    amount = quantize_amount(exchange, symbol, target_qty)
    result = {
        "pair": "BTCUSDT",
        "symbol": symbol,
        "side": "buy" if direction > 0 else "sell",
        "price": price,
        "target_qty": target_qty,
        "amount": amount,
        "placed": False,
        "margin_action": margin_action,
    }
    if amount <= 0.0:
        return result
    if execute:
        order = exchange.create_market_order(symbol, result["side"], amount)
        result["placed"] = True
        result["order_id"] = order.get("id")
        if order.get("average") is not None:
            result["price"] = float(order["average"])
    return result


def manage_overlay(
    exchange: ccxt.binanceusdm,
    state: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    overlay = state.get("overlay")
    if not overlay:
        return {"status": "no_overlay"}

    symbol = overlay["symbol"]
    direction = 1.0 if overlay["side"] == "LONG" else -1.0
    entry_price = float(overlay["entry_price"])
    last_bar_ts = overlay.get("last_bar_ts")
    since_ms = int(pd.Timestamp(last_bar_ts).timestamp() * 1000) if last_bar_ts else None
    bars = fetch_recent_closed_5m_bars(exchange, symbol, since_ms=since_ms, limit=50)
    if bars.empty:
        return {"status": "no_new_bars"}

    risk_pct = abs(DAILY_MAX_LOSS_PCT)
    gross_stop = -risk_pct + 2 * COMMISSION_PCT
    gross_target = BEST_OVERLAY_PARAMS.reward_multiple * risk_pct + 2 * COMMISSION_PCT
    gross_trail_activation = BEST_OVERLAY_PARAMS.trail_activation_pct + 2 * COMMISSION_PCT
    gross_trail_floor = BEST_OVERLAY_PARAMS.trail_floor_pct + 2 * COMMISSION_PCT

    best_favorable = float(overlay.get("best_favorable", 0.0))
    trail_active = bool(overlay.get("trail_active", False))
    exit_reason = None
    exit_return = None
    latest_bar_ts = None

    for row in bars.itertuples(index=False):
        latest_bar_ts = row.time.isoformat()
        favorable = max(
            direction * (float(row.high) / entry_price - 1.0),
            direction * (float(row.low) / entry_price - 1.0),
        )
        adverse = min(
            direction * (float(row.high) / entry_price - 1.0),
            direction * (float(row.low) / entry_price - 1.0),
        )
        best_favorable = max(best_favorable, favorable)

        dynamic_stop = gross_stop
        if best_favorable >= gross_trail_activation:
            trail_active = True
            dynamic_stop = max(
                gross_stop,
                gross_trail_floor,
                best_favorable - BEST_OVERLAY_PARAMS.trail_distance_pct,
            )

        stop_hit = adverse <= dynamic_stop
        target_hit = favorable >= gross_target

        if stop_hit and target_hit:
            exit_reason = "trail_stop_and_target_same_bar" if trail_active else "stop_and_target_same_bar"
            exit_return = dynamic_stop
            break
        if stop_hit:
            exit_reason = "trail_stop" if trail_active and dynamic_stop > gross_stop else "stop"
            exit_return = dynamic_stop
            break
        if target_hit:
            exit_reason = "target"
            exit_return = gross_target
            break

    overlay["best_favorable"] = best_favorable
    overlay["trail_active"] = trail_active
    if latest_bar_ts is not None:
        overlay["last_bar_ts"] = latest_bar_ts

    if exit_reason is None:
        state["overlay"] = overlay
        return {"status": "open", "best_favorable": best_favorable, "trail_active": trail_active}

    current_qty = float(overlay["qty"])
    close_action = close_pair_position(exchange, "BTCUSDT", current_qty, execute)
    if execute or close_action["amount"] <= 0.0:
        state["overlay"] = None
        state["overlay_completed_day"] = overlay.get("effective_day")
        status = "closed"
    else:
        state["overlay"] = overlay
        status = "would_close"
    return {
        "status": status,
        "exit_reason": exit_reason,
        "exit_return": exit_return,
        "overlay_side": overlay["side"],
        "entry_price": entry_price,
        "effective_day": overlay.get("effective_day"),
        "close_action": close_action,
    }


def maybe_force_overlay_eod_close(
    exchange: ccxt.binanceusdm,
    state: dict[str, Any],
    current_effective_day: str,
    execute: bool,
) -> dict[str, Any] | None:
    overlay = state.get("overlay")
    if not overlay:
        return None
    if overlay.get("effective_day") == current_effective_day:
        return None
    close_action = close_pair_position(exchange, "BTCUSDT", float(overlay["qty"]), execute)
    if execute or close_action["amount"] <= 0.0:
        state["overlay"] = None
        state["overlay_completed_day"] = overlay.get("effective_day")
        status = "forced_eod_close"
    else:
        status = "would_force_eod_close"
    return {
        "status": status,
        "close_action": close_action,
        "overlay_side": overlay.get("side"),
        "entry_price": overlay.get("entry_price"),
        "effective_day": overlay.get("effective_day"),
    }


def sync_state_command(args: argparse.Namespace) -> None:
    state = load_state(args.state_path)
    plan = build_strategy_plan(args.effective_date, args.leverage)
    exchange = get_exchange(args.mode)
    register_runtime_context(args, state, exchange)
    try:
        sync_report = sync_exchange_state(exchange, state, plan, args.execute)
        save_state(args.state_path, state)
        print(json.dumps(sync_report, indent=2))
        print(f"\nState saved: {args.state_path}")
    finally:
        clear_runtime_context()


def shutdown_protect_command(args: argparse.Namespace) -> None:
    state = load_state(args.state_path)
    exchange = get_exchange(args.mode)
    register_runtime_context(args, state, exchange)
    try:
        report = install_shutdown_protection(exchange, state, args.execute)
        save_state(args.state_path, state)
        print(json.dumps(report, indent=2))
        print(f"\nState saved: {args.state_path}")
    finally:
        clear_runtime_context()


def close_all_positions_command(args: argparse.Namespace) -> None:
    state = load_state(args.state_path)
    exchange = get_exchange(args.mode)
    register_runtime_context(args, state, exchange)
    try:
        positions_before = fetch_open_position_map(exchange)
        protection_orders = fetch_strategy_protection_orders(exchange)
        cancel_actions = cancel_strategy_protection_orders(exchange, protection_orders, args.execute)
        flatten_actions = flatten_pairs(exchange, PAIRS, args.execute)

        report = {
            "status": "flattened" if args.execute else "planned",
            "warning": (
                "If the live loop is still running, strategy logic may reopen positions on the next cycle."
            ),
            "positions_before": [
                {
                    "pair": pair,
                    "symbol": pos["symbol"],
                    "qty": pos["qty"],
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "mark_price": pos["mark_price"],
                    "margin_mode": pos["margin_mode"],
                }
                for pair, pos in positions_before.items()
            ],
            "managed_protection_orders": [
                {
                    "id": order.get("id"),
                    "client_order_id": extract_client_order_id(order),
                    "symbol": order.get("symbol"),
                    "type": order.get("type"),
                    "side": order.get("side"),
                }
                for order in protection_orders
            ],
            "cancel_actions": cancel_actions,
            "flatten_actions": flatten_actions,
        }

        if args.execute:
            clear_strategy_state(state)
            save_state(args.state_path, state)

        print(json.dumps(report, indent=2))
        if args.execute:
            print(f"\nState saved: {args.state_path}")
    finally:
        clear_runtime_context()


def run_once(args: argparse.Namespace) -> None:
    state = load_state(args.state_path)
    exchange = get_exchange(args.mode)
    register_runtime_context(args, state, exchange)
    try:
        plan = build_strategy_plan(args.effective_date, args.leverage)
        notifications: list[str] = []
        notification_state = state.setdefault("notification_state", {})

        equity = args.equity
        api_key, secret = get_demo_credentials()
        authenticated = bool(api_key and secret)
        if args.execute and not authenticated:
            raise ValueError("Demo API credentials are missing. Set BINANCE_DEMO_API_KEY and BINANCE_DEMO_API_SECRET.")
        if equity is None:
            if not authenticated:
                raise ValueError("Equity is required for dry-run without demo API credentials.")
            equity = fetch_equity(exchange)
        equity = float(equity)

        if args.execute:
            previous_briefing_day = notification_state.get("last_briefing_effective_day")
            previous_briefing_equity = notification_state.get("last_briefing_equity")
            last_summary_day = notification_state.get("last_summary_effective_day")
            if (
                previous_briefing_day
                and previous_briefing_day != plan["effective_day"]
                and previous_briefing_equity is not None
                and last_summary_day != previous_briefing_day
            ):
                notifications.append(build_daily_summary(str(previous_briefing_day), float(previous_briefing_equity), equity))
                notification_state["last_summary_effective_day"] = previous_briefing_day

            if notification_state.get("last_briefing_effective_day") != plan["effective_day"]:
                notifications.append(build_daily_briefing(plan, equity))
                notification_state["last_briefing_effective_day"] = plan["effective_day"]
                notification_state["last_briefing_equity"] = equity

            previous_session_day = notification_state.get("last_session_effective_day")
            previous_session_type = notification_state.get("last_session_type")
            if (
                previous_session_day == plan["effective_day"]
                and previous_session_type
                and previous_session_type != plan["session_type"]
            ):
                notifications.append(
                    build_session_change_notification(str(previous_session_type), str(plan["session_type"]), plan["effective_day"])
                )
            notification_state["last_session_effective_day"] = plan["effective_day"]
            notification_state["last_session_type"] = plan["session_type"]

        print_plan(plan, equity)

        sync_report = sync_exchange_state(exchange, state, plan, args.execute)
        print("\nExchange Sync")
        print(json.dumps(sync_report, indent=2))

        forced_close = maybe_force_overlay_eod_close(exchange, state, plan["effective_day"], args.execute)
        if forced_close is not None:
            print("\nForced previous-day overlay close")
            print(json.dumps(forced_close, indent=2))
            if args.execute:
                note = build_overlay_exit_notification(forced_close)
                if note:
                    notifications.append(note)

        core_kill_report = evaluate_core_emergency_kill_switch(exchange, state, plan, equity, args.execute)
        if core_kill_report["status"] != "inactive":
            print("\nCore Kill Switch")
            print(json.dumps(core_kill_report, indent=2))
            if args.execute:
                note = build_kill_switch_notification(core_kill_report)
                if note:
                    notifications.append(note)

        if plan["session_type"] == "core":
            if core_kill_report["status"] == "triggered":
                print("\nAction: core kill-switch triggered, stay flat")
                print(json.dumps(core_kill_report.get("flatten_actions", []), indent=2))
            elif core_kill_report["status"] == "locked":
                print("\nAction: core kill-switch locked, stay flat")
                flat_actions = reconcile_target_positions(exchange, equity, {pair: 0.0 for pair in PAIRS}, args.execute)
                print(json.dumps(flat_actions, indent=2))
                if args.execute:
                    note = build_core_rebalance_notification(flat_actions, title="코어 포지션 정리")
                    if note:
                        notifications.append(note)
            else:
                print("\nAction: rebalance core positions")
                if state.get("overlay"):
                    overlay_qty = float(state["overlay"]["qty"])
                    close_action = close_pair_position(exchange, "BTCUSDT", overlay_qty, args.execute)
                    state["overlay"] = None
                    print("Closed stale overlay:")
                    print(json.dumps(close_action, indent=2))
                    if args.execute and close_action.get("placed"):
                        notifications.append(
                            "\n".join(
                                [
                                    "오버레이 강제 정리",
                                    f"- 수량: {abs(float(close_action.get('current_qty', 0.0))):.6f}",
                                    "- 사유: 코어 세션 전환",
                                ]
                            )
                        )
                actions = reconcile_target_positions(exchange, equity, plan["core_weights"], args.execute)
                print(json.dumps(actions, indent=2))
                if args.execute:
                    note = build_core_rebalance_notification(actions)
                    if note:
                        notifications.append(note)
        elif plan["session_type"] == "overlay":
            print("\nAction: flatten core, then run overlay")

            overlay_state = state.get("overlay")
            completed_day = state.get("overlay_completed_day")
            if overlay_state and overlay_state.get("effective_day") == plan["effective_day"]:
                flat_actions = flatten_pairs(exchange, ["ETHUSDT", "SOLUSDT", "XRPUSDT"], args.execute)
                print(json.dumps(flat_actions, indent=2))
                if args.execute:
                    note = build_core_rebalance_notification(flat_actions, title="코어 포지션 정리")
                    if note:
                        notifications.append(note)
                managed = manage_overlay(exchange, state, args.execute)
                print(json.dumps(managed, indent=2))
                if args.execute:
                    note = build_overlay_exit_notification(managed)
                    if note:
                        notifications.append(note)
            elif completed_day == plan["effective_day"]:
                flat_actions = reconcile_target_positions(exchange, equity, {pair: 0.0 for pair in PAIRS}, args.execute)
                print(json.dumps(flat_actions, indent=2))
                print(json.dumps({"status": "overlay_complete_for_day", "effective_day": plan["effective_day"]}, indent=2))
                if args.execute:
                    note = build_core_rebalance_notification(flat_actions, title="코어 포지션 정리")
                    if note:
                        notifications.append(note)
            else:
                flat_actions = reconcile_target_positions(exchange, equity, {pair: 0.0 for pair in PAIRS}, args.execute)
                print(json.dumps(flat_actions, indent=2))
                if args.execute:
                    note = build_core_rebalance_notification(flat_actions, title="코어 포지션 정리")
                    if note:
                        notifications.append(note)
                opened = open_overlay_position(exchange, equity, plan["overlay"]["side"], plan["leverage"], args.execute)
                qty = float(opened["amount"])
                if args.execute and opened["placed"] and qty > 0.0:
                    state["overlay"] = {
                        "effective_day": plan["effective_day"],
                        "symbol": opened["symbol"],
                        "side": plan["overlay"]["side"],
                        "qty": qty if plan["overlay"]["side"] == "LONG" else -qty,
                        "entry_price": float(opened["price"]),
                        "best_favorable": 0.0,
                        "trail_active": False,
                        "last_bar_ts": None,
                        "opened_at": utc_now().isoformat(),
                    }
                    state["overlay_completed_day"] = None
                print(json.dumps(opened, indent=2))
                if args.execute:
                    note = build_overlay_entry_notification(opened, plan["effective_day"], plan["leverage"])
                    if note:
                        notifications.append(note)
        else:
            print("\nAction: stay flat")
            flat_actions = reconcile_target_positions(exchange, equity, {pair: 0.0 for pair in PAIRS}, args.execute)
            print(json.dumps(flat_actions, indent=2))
            if state.get("overlay") and args.execute:
                state["overlay_completed_day"] = state["overlay"].get("effective_day")
                state["overlay"] = None
            if args.execute:
                note = build_core_rebalance_notification(flat_actions, title="포지션 정리")
                if note:
                    notifications.append(note)

        save_state(args.state_path, state)
        print(f"\nState saved: {args.state_path}")
        if args.execute and notifications:
            dispatch_notifications(notifications)
    finally:
        clear_runtime_context()


def atexit_shutdown_handler() -> None:
    if not RUNTIME_CONTEXT.get("shutdown_requested"):
        return
    args = RUNTIME_CONTEXT.get("args")
    state = RUNTIME_CONTEXT.get("state")
    exchange = RUNTIME_CONTEXT.get("exchange")
    if args is None:
        args = argparse.Namespace(
            execute=True,
            mode=os.getenv("BINANCE_MODE", "demo"),
            state_path=str(STATE_PATH),
        )
    if getattr(args, "execute", True) is not True:
        return
    try:
        if state is None:
            state = load_state(getattr(args, "state_path", str(STATE_PATH)))
        if exchange is None:
            exchange = get_exchange(getattr(args, "mode", os.getenv("BINANCE_MODE", "demo")))
        report = install_shutdown_protection(exchange, state, True)
        save_state(getattr(args, "state_path", str(STATE_PATH)), state)
        print("\nShutdown Protection")
        print(json.dumps({"reason": RUNTIME_CONTEXT.get("shutdown_reason"), "report": report}, indent=2))
    except Exception as exc:
        print(f"[WARN] Failed to install shutdown protection: {exc}")


def main() -> None:
    args = parse_args()

    if args.command == "status":
        plan = build_strategy_plan(args.effective_date, resolve_default_leverage())
        print_plan(plan, args.equity)
        return

    if args.command == "run-once":
        run_once(args)
        return

    if args.command == "sync-state":
        sync_state_command(args)
        return

    if args.command == "shutdown-protect":
        shutdown_protect_command(args)
        return

    if args.command == "close-all":
        close_all_positions_command(args)
        return

    if args.command == "loop":
        while True:
            try:
                ns = argparse.Namespace(
                    command="run-once",
                    execute=args.execute,
                    equity=args.equity,
                    effective_date=None,
                    mode=args.mode,
                    leverage=args.leverage,
                    state_path=args.state_path,
                )
                run_once(ns)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[ERROR] {exc}")
                send_telegram_notification(
                    "\n".join(
                        [
                            "트레이더 루프 오류",
                            f"- 시각: {utc_now().isoformat()}",
                            f"- 내용: {exc}",
                        ]
                    )
                )
            time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    atexit.register(atexit_shutdown_handler)
    main()
