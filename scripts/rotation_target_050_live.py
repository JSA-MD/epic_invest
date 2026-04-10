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
import hashlib
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

from backtest_cash_filtered_rotation import build_daily_close
from backtest_rotation_target_050 import (
    BEST_CORE_PARAMS,
    BEST_OVERLAY_PARAMS,
    MEAN_TARGET_LEVERAGE,
)
from core_strategy_registry import (
    DEFAULT_CORE_CHAMPION_PATH,
    LONG_ONLY_FAMILY,
    LONG_SHORT_FAMILY,
    ResolvedCoreStrategy,
    build_core_target_weights,
    load_core_artifact,
    resolve_core_strategy,
)
from core_market_profile import (
    build_context_corr_snapshot,
    build_core_market_profile,
    build_route_state_snapshot,
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
from market_context import load_market_context_dataset

load_dotenv()

PAIR_TO_MARKET = {
    "BTCUSDT": "BTC/USDT:USDT",
    "ETHUSDT": "ETH/USDT:USDT",
    "SOLUSDT": "SOL/USDT:USDT",
    "XRPUSDT": "XRP/USDT:USDT",
}
MARKET_TO_PAIR = {v: k for k, v in PAIR_TO_MARKET.items()}
STATE_PATH = MODELS_DIR / "rotation_target_050_live_state.json"
DECISION_LOG_PATH = Path(
    os.getenv(
        "TRADER_DECISION_LOG_FILE",
        str(MODELS_DIR.parent / "logs" / "rotation_target_050_decisions.jsonl"),
    )
)
DEFAULT_EXCHANGE_LEVERAGE = 5
REBALANCE_NOTIONAL_BAND_USD = float(os.getenv("REBALANCE_NOTIONAL_BAND_USD", "25"))
CORE_KILL_SWITCH_SIGMA_MULTIPLE = 1.5
CORE_KILL_SWITCH_MIN_PCT = 0.06
CORE_KILL_SWITCH_MAX_PCT = 0.10
SHUTDOWN_PROTECTION_RISK_PCT = abs(DAILY_MAX_LOSS_PCT)
SHUTDOWN_PROTECTION_REWARD_MULTIPLE = 2.5
SHUTDOWN_PROTECTION_WORKING_TYPE = "MARK_PRICE"
PROTECTION_CLIENT_PREFIX = "epiP"
PROTECTION_PRICE_REPLACE_BAND = float(os.getenv("PROTECTION_PRICE_REPLACE_BAND", "0.0005"))
PROTECTION_AMOUNT_REPLACE_BAND = float(os.getenv("PROTECTION_AMOUNT_REPLACE_BAND", "0.0005"))
UTC = timezone.utc
EPSILON = 1e-12
RUNTIME_CONTEXT: dict[str, Any] = {
    "args": None,
    "state": None,
    "exchange": None,
    "shutdown_requested": False,
    "shutdown_reason": None,
}
LIVE_CORE_STRATEGY: ResolvedCoreStrategy | None = None

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


def resolve_live_core_strategy() -> ResolvedCoreStrategy:
    global LIVE_CORE_STRATEGY
    if LIVE_CORE_STRATEGY is not None:
        return LIVE_CORE_STRATEGY

    resolved: ResolvedCoreStrategy | None = None
    artifact_path = DEFAULT_CORE_CHAMPION_PATH
    if artifact_path.exists():
        try:
            resolved = load_core_artifact(artifact_path)
        except Exception as exc:
            print(f"[WARN] Failed to load core champion artifact {artifact_path}: {exc}")
            resolved = None

    if resolved is None or resolved.family not in {LONG_ONLY_FAMILY, LONG_SHORT_FAMILY}:
        if resolved is not None and resolved.family not in {LONG_ONLY_FAMILY, LONG_SHORT_FAMILY}:
            print(
                f"[WARN] Unsupported core family {resolved.family!r} in {artifact_path}; "
                "falling back to legacy long-only core params."
            )
        resolved = resolve_core_strategy(
            LONG_ONLY_FAMILY,
            BEST_CORE_PARAMS,
            key="legacy_best_core_params",
            source="backtest_rotation_target_050.BEST_CORE_PARAMS",
        )

    LIVE_CORE_STRATEGY = resolved
    return resolved


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


def protection_tag_from_order(order: dict[str, Any]) -> str | None:
    client_order_id = extract_client_order_id(order)
    if client_order_id.startswith(f"{PROTECTION_CLIENT_PREFIX}SL"):
        return "SL"
    if client_order_id.startswith(f"{PROTECTION_CLIENT_PREFIX}TP"):
        return "TP"

    info = order.get("info", {}) if isinstance(order.get("info"), dict) else {}
    text = " ".join(
        str(value)
        for value in (
            order.get("type"),
            info.get("type"),
            info.get("origType"),
            info.get("strategyType"),
        )
        if value is not None
    ).upper()
    if "TAKE" in text:
        return "TP"
    if "STOP" in text:
        return "SL"
    return None


def protection_order_float(order: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    info = order.get("info", {}) if isinstance(order.get("info"), dict) else {}
    for key in keys:
        value = order.get(key)
        if value not in (None, ""):
            try:
                parsed = float(value)
                if parsed > 0.0:
                    return parsed
            except (TypeError, ValueError):
                pass
        value = info.get(key)
        if value not in (None, ""):
            try:
                parsed = float(value)
                if parsed > 0.0:
                    return parsed
            except (TypeError, ValueError):
                pass
    return None


def values_materially_same(actual: float | None, target: float, rel_band: float) -> bool:
    if actual is None:
        return False
    tolerance = max(EPSILON, abs(float(target)) * float(rel_band))
    return abs(float(actual) - float(target)) <= tolerance


def protection_order_matches_desired(order: dict[str, Any], desired: dict[str, Any]) -> bool:
    if str(order.get("symbol")) != str(desired["symbol"]):
        return False
    if str(order.get("side", "")).lower() != str(desired["side"]).lower():
        return False
    if protection_tag_from_order(order) != str(desired["tag"]):
        return False

    amount = protection_order_float(order, ("amount", "origQty", "quantity", "qty"))
    trigger_price = protection_order_float(
        order,
        (
            "stopPrice",
            "triggerPrice",
            "stopLossPrice",
            "takeProfitPrice",
            "activationPrice",
            "price",
        ),
    )
    return values_materially_same(amount, float(desired["amount"]), PROTECTION_AMOUNT_REPLACE_BAND) and values_materially_same(
        trigger_price,
        float(desired["stop_price"]),
        PROTECTION_PRICE_REPLACE_BAND,
    )


def load_state(path: str | Path) -> dict[str, Any]:
    state_file = Path(path)
    if not state_file.exists():
        return {
            "overlay": None,
            "overlay_completed_day": None,
            "core_day_state": None,
            "shutdown_protection": [],
            "latest_decision_snapshot": None,
            "decision_journal": {
                "last_fingerprint": None,
                "last_recorded_at": None,
                "log_path": str(DECISION_LOG_PATH),
            },
            "latest_runtime_snapshot": {
                "captured_at": None,
                "equity": None,
                "positions": [],
                "protections": [],
                "exchange_error": None,
                "plan": None,
            },
            "notification_state": {
                "last_briefing_effective_day": None,
                "last_briefing_equity": None,
                "last_summary_effective_day": None,
                "last_session_effective_day": None,
                "last_session_type": None,
            },
            "runtime_health": {
                "process_started_at": None,
                "last_loop_started_at": None,
                "last_loop_completed_at": None,
                "last_success_at": None,
                "last_error_at": None,
                "last_error_message": None,
                "consecutive_errors": 0,
                "loop_success_count": 0,
                "loop_error_count": 0,
                "last_duration_seconds": None,
                "pid": None,
                "poll_seconds": None,
            },
            "updated_at": None,
        }
    with open(state_file, "r") as f:
        state = json.load(f)
    state.setdefault("overlay", None)
    state.setdefault("overlay_completed_day", None)
    state.setdefault("core_day_state", None)
    state.setdefault("shutdown_protection", [])
    state.setdefault("latest_decision_snapshot", None)
    state.setdefault("decision_journal", {})
    state["decision_journal"].setdefault("last_fingerprint", None)
    state["decision_journal"].setdefault("last_recorded_at", None)
    state["decision_journal"].setdefault("log_path", str(DECISION_LOG_PATH))
    state.setdefault("latest_runtime_snapshot", {})
    state["latest_runtime_snapshot"].setdefault("captured_at", None)
    state["latest_runtime_snapshot"].setdefault("equity", None)
    state["latest_runtime_snapshot"].setdefault("positions", [])
    state["latest_runtime_snapshot"].setdefault("protections", [])
    state["latest_runtime_snapshot"].setdefault("exchange_error", None)
    state["latest_runtime_snapshot"].setdefault("plan", None)
    state.setdefault("notification_state", {})
    state["notification_state"].setdefault("last_briefing_effective_day", None)
    state["notification_state"].setdefault("last_briefing_equity", None)
    state["notification_state"].setdefault("last_summary_effective_day", None)
    state["notification_state"].setdefault("last_session_effective_day", None)
    state["notification_state"].setdefault("last_session_type", None)
    state.setdefault("runtime_health", {})
    state["runtime_health"].setdefault("process_started_at", None)
    state["runtime_health"].setdefault("last_loop_started_at", None)
    state["runtime_health"].setdefault("last_loop_completed_at", None)
    state["runtime_health"].setdefault("last_success_at", None)
    state["runtime_health"].setdefault("last_error_at", None)
    state["runtime_health"].setdefault("last_error_message", None)
    state["runtime_health"].setdefault("consecutive_errors", 0)
    state["runtime_health"].setdefault("loop_success_count", 0)
    state["runtime_health"].setdefault("loop_error_count", 0)
    state["runtime_health"].setdefault("last_duration_seconds", None)
    state["runtime_health"].setdefault("pid", None)
    state["runtime_health"].setdefault("poll_seconds", None)
    state.setdefault("updated_at", None)
    return state


def update_runtime_health(state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    runtime_health = state.setdefault("runtime_health", {})
    for key, value in updates.items():
        runtime_health[key] = value
    return runtime_health


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def append_jsonl(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(json_ready(payload), ensure_ascii=False) + "\n")


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


def pair_rows(series: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, value in series.items():
        if not np.isfinite(value):
            continue
        rows.append({"pair": str(pair), "value": float(value)})
    return rows


def selected_position_rows(weights: pd.Series) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair, weight in weights.items():
        weight_value = float(weight)
        if abs(weight_value) <= EPSILON:
            continue
        rows.append(
            {
                "pair": str(pair),
                "side": side_label_from_qty(weight_value),
                "weight": weight_value,
                "abs_weight": abs(weight_value),
            }
        )
    rows.sort(key=lambda item: (-abs(float(item["weight"])), str(item["pair"])))
    return rows


def build_core_selection_diagnostics(
    close: pd.DataFrame,
    market_day: pd.Timestamp,
    leverage: float,
    strategy: ResolvedCoreStrategy,
) -> dict[str, Any]:
    params = strategy.params
    market_context_close, market_context_status = load_market_context_dataset(
        refresh=False,
        allow_fetch_on_miss=False,
        target_index=close.index,
        min_context_days=20,
    )
    long_regime_threshold = float(getattr(params, "long_regime_threshold", getattr(params, "regime_threshold", 0.0)))
    long_breadth_threshold = float(getattr(params, "long_breadth_threshold", getattr(params, "breadth_threshold", 0.50)))
    market_profile = build_core_market_profile(
        close,
        market_context_close,
        fast_lookback=int(params.lookback_fast),
        slow_lookback=int(params.lookback_slow),
        vol_window=int(params.vol_window),
        corr_window=20,
        regime_threshold=long_regime_threshold,
        breadth_threshold=long_breadth_threshold,
    )
    feature_frame = market_profile["feature_frame"]
    momentum = market_profile["momentum"]
    realized_vol = market_profile["realized_vol"]
    btc_regime = market_profile["regime_score"]
    breadth = market_profile["breadth"]

    target_weights = build_core_target_weights(close, strategy)
    base_core = target_weights.loc[market_day].fillna(0.0).astype("float64")
    levered_core = base_core * float(leverage)

    ranked = momentum.loc[market_day].dropna().sort_values(ascending=False)
    positive_ranked = ranked[ranked > 0.0]
    negative_ranked = ranked[ranked < 0.0].sort_values(ascending=True)
    if base_core.abs().sum() <= EPSILON:
        selected_mode = "flat"
        selected = pd.Series(dtype="float64")
    elif float(base_core.sum()) >= 0.0:
        selected_mode = "long"
        selected = positive_ranked.head(params.top_n)
    else:
        selected_mode = "short"
        selected = negative_ranked.head(params.top_n)
    selected_vol = realized_vol.loc[market_day, selected.index].replace(0.0, np.nan).dropna()
    selected = selected.loc[selected_vol.index]

    raw_score = pd.Series(dtype="float64")
    scale = None
    if not selected.empty:
        raw_score = (selected / selected_vol).abs()
        raw_sum = float(raw_score.sum())
        if raw_sum > 0.0:
            weights = raw_score / raw_sum
            port_vol = float(np.sqrt(np.sum(np.square(weights.to_numpy() * selected_vol.to_numpy()))))
            if np.isfinite(port_vol) and port_vol > 1e-8:
                scale = min(
                    float(params.target_vol_ann) / port_vol,
                    float(params.gross_cap) / max(float(weights.sum()), 1e-8),
                )

    btc_regime_value = float(btc_regime.loc[market_day])
    breadth_value = float(breadth.loc[market_day])
    route_state = build_route_state_snapshot(feature_frame, market_day)
    context_corr_snapshot = build_context_corr_snapshot(market_profile["corr_state_profiles"], market_day)
    selected_positions = selected_position_rows(base_core)
    selected_pairs = [item["pair"] for item in selected_positions]
    long_pairs = [item["pair"] for item in selected_positions if item["side"] == "롱"]
    short_pairs = [item["pair"] for item in selected_positions if item["side"] == "숏"]
    long_gate = bool(np.isfinite(btc_regime_value) and btc_regime_value > long_regime_threshold)
    long_breadth_gate = bool(np.isfinite(breadth_value) and breadth_value >= long_breadth_threshold)
    short_regime_threshold = float(getattr(params, "short_regime_threshold", getattr(params, "regime_threshold", 0.0)))
    short_breadth_threshold = float(getattr(params, "short_breadth_threshold", getattr(params, "breadth_threshold", 0.50)))
    short_gate = bool(np.isfinite(btc_regime_value) and btc_regime_value <= -short_regime_threshold)
    short_breadth_gate = bool(np.isfinite(breadth_value) and breadth_value <= short_breadth_threshold)
    return {
        "family": strategy.family,
        "strategy_key": strategy.key,
        "strategy_source": strategy.source,
        "market_day": market_day.date().isoformat(),
        "selected_mode": selected_mode,
        "lookback_fast": int(params.lookback_fast),
        "lookback_slow": int(params.lookback_slow),
        "top_n": int(params.top_n),
        "regime_threshold": float(getattr(params, "regime_threshold", getattr(params, "long_regime_threshold", 0.0))),
        "breadth_threshold": float(getattr(params, "breadth_threshold", getattr(params, "long_breadth_threshold", 0.50))),
        "long_regime_threshold": float(getattr(params, "long_regime_threshold", getattr(params, "regime_threshold", 0.0))),
        "short_regime_threshold": short_regime_threshold,
        "long_breadth_threshold": float(getattr(params, "long_breadth_threshold", getattr(params, "breadth_threshold", 0.50))),
        "short_breadth_threshold": short_breadth_threshold,
        "passed_regime": long_gate,
        "passed_breadth": long_breadth_gate,
        "passed_short_regime": short_gate,
        "passed_short_breadth": short_breadth_gate,
        "btc_regime": btc_regime_value,
        "breadth": breadth_value,
        "route_state": route_state,
        "context_corr_snapshot": context_corr_snapshot,
        "market_context_status": market_context_status,
        "selected_positions": selected_positions,
        "momentum_ranked": pair_rows(ranked.sort_values(ascending=False)),
        "positive_ranked": pair_rows(positive_ranked),
        "negative_ranked": pair_rows(negative_ranked),
        "selected_pairs": selected_pairs,
        "long_pairs": long_pairs,
        "short_pairs": short_pairs,
        "selected_vol_ann": {str(pair): float(value) for pair, value in selected_vol.items()},
        "raw_score": {str(pair): float(value) for pair, value in raw_score.items()},
        "scale": float(scale) if scale is not None else None,
        "unlevered_weights": {str(pair): float(value) for pair, value in base_core.items()},
        "levered_weights": {str(pair): float(value) for pair, value in levered_core.items()},
    }


def build_overlay_exit_template() -> dict[str, Any]:
    risk_pct = abs(DAILY_MAX_LOSS_PCT)
    return {
        "reward_multiple": float(BEST_OVERLAY_PARAMS.reward_multiple),
        "stop_return": float(-risk_pct + 2 * COMMISSION_PCT),
        "target_return": float(BEST_OVERLAY_PARAMS.reward_multiple * risk_pct + 2 * COMMISSION_PCT),
        "trail_activation_return": float(BEST_OVERLAY_PARAMS.trail_activation_pct + 2 * COMMISSION_PCT),
        "trail_distance_return": float(BEST_OVERLAY_PARAMS.trail_distance_pct),
        "trail_floor_return": float(BEST_OVERLAY_PARAMS.trail_floor_pct + 2 * COMMISSION_PCT),
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
    core_strategy = resolve_live_core_strategy()
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

    target_weights = build_core_target_weights(close, core_strategy)
    if market_day not in target_weights.index:
        raise ValueError(f"Missing core weights for {market_day.date()}")

    base_core = target_weights.loc[market_day].fillna(0.0).astype("float64")
    core_weights = base_core * float(leverage)
    core_active = bool(core_weights.abs().sum() > EPSILON)
    overlay = compute_overlay_signal(close, market_day)
    core_diagnostics = build_core_selection_diagnostics(close, market_day, leverage, core_strategy)
    session_type = "core" if core_active else "overlay" if overlay["signal_pct"] != 0.0 else "flat"

    return {
        "strategy_class": "rotation_target_050_live",
        "strategy_profile": resolve_strategy_profile(),
        "market_day": market_day.date().isoformat(),
        "effective_day": effective_day.date().isoformat(),
        "leverage": float(leverage),
        "core_strategy_family": core_strategy.family,
        "core_strategy_key": core_strategy.key,
        "core_strategy_source": core_strategy.source,
        "core_strategy_metadata": core_strategy.metadata,
        "core_validation_context": {
            "promotion_gate": core_strategy.metadata.get("promotion_gate"),
            "validation_profile": core_strategy.metadata.get("validation_profile"),
            "cpcv_pbo": core_strategy.metadata.get("cpcv_pbo"),
            "regime_breakdown_summary": core_strategy.metadata.get("regime_breakdown_summary"),
            "corr_state_summary": core_strategy.metadata.get("corr_state_summary"),
            "candidate_selection_pbo": core_strategy.metadata.get("candidate_selection_pbo"),
        },
        "core_params": asdict(core_strategy.params),
        "overlay_params": asdict(BEST_OVERLAY_PARAMS),
        "session_type": session_type,
        "core_active": core_active,
        "core_weights": {pair: float(core_weights[pair]) for pair in core_weights.index},
        "core_gross_leverage": float(core_weights.abs().sum()),
        "core_diagnostics": core_diagnostics,
        "overlay": overlay,
        "overlay_exit_template": build_overlay_exit_template(),
        "latest_prices": latest_prices,
    }


def print_plan(plan: dict[str, Any], equity: float) -> None:
    print("=" * 72)
    print("  Rotation Target 0.5% Live Plan")
    print("=" * 72)
    print(f"Market Day       : {plan['market_day']}")
    print(f"Effective Day    : {plan['effective_day']}")
    print(f"Strategy Leverage: {plan['leverage']:.2f}x")
    print(f"Core Family      : {plan.get('core_strategy_family', 'long_only')}")
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


def build_shutdown_protection_plan(
    exchange: ccxt.binanceusdm,
    position: dict[str, Any],
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
    result["desired_orders"] = [
        {
            "tag": "SL",
            "pair": pair,
            "symbol": symbol,
            "side": close_side,
            "amount": amount,
            "stop_price": stop_price,
            "param_key": "stopLossPrice",
        },
        {
            "tag": "TP",
            "pair": pair,
            "symbol": symbol,
            "side": close_side,
            "amount": amount,
            "stop_price": take_price,
            "param_key": "takeProfitPrice",
        },
    ]
    return result


def place_shutdown_protection_order(
    exchange: ccxt.binanceusdm,
    desired: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    order_report = {
        "tag": desired["tag"],
        "pair": desired["pair"],
        "symbol": desired["symbol"],
        "side": desired["side"],
        "amount": float(desired["amount"]),
        "stop_price": float(desired["stop_price"]),
        "status": "planned",
    }
    if not execute:
        return order_report

    params = {
        desired["param_key"]: float(desired["stop_price"]),
        "reduceOnly": True,
        "workingType": SHUTDOWN_PROTECTION_WORKING_TYPE,
        "clientOrderId": build_exit_client_id(str(desired["tag"]), str(desired["pair"])),
    }
    order = exchange.create_order(
        str(desired["symbol"]),
        "market",
        str(desired["side"]),
        float(desired["amount"]),
        None,
        params,
    )
    order_report.update(
        {
            "id": order.get("id"),
            "client_order_id": extract_client_order_id(order),
            "type": order.get("type"),
            "status": "placed",
        }
    )
    return order_report


def place_shutdown_protection_for_position(
    exchange: ccxt.binanceusdm,
    position: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    result = build_shutdown_protection_plan(exchange, position)
    if result.get("status") == "skipped":
        return result
    if not execute:
        return result

    placed_orders = [
        place_shutdown_protection_order(exchange, desired, execute=True)
        for desired in result.get("desired_orders", [])
    ]
    result["status"] = "placed"
    result["orders"] = placed_orders
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


def update_latest_runtime_snapshot(
    exchange: ccxt.binanceusdm,
    state: dict[str, Any],
    plan: dict[str, Any] | None,
    equity_hint: float | None = None,
) -> dict[str, Any]:
    snapshot = state.setdefault("latest_runtime_snapshot", {})
    captured_at = utc_now().isoformat()
    try:
        equity = float(equity_hint) if equity_hint is not None else fetch_equity(exchange)
        positions = fetch_open_position_map(exchange)
        protections = fetch_strategy_protection_orders(exchange)
        snapshot.update(
            {
                "captured_at": captured_at,
                "equity": equity,
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
                "protections": [
                    {
                        "id": order.get("id"),
                        "client_order_id": extract_client_order_id(order),
                        "symbol": order.get("symbol"),
                        "type": order.get("type"),
                        "side": order.get("side"),
                    }
                    for order in protections
                ],
                "exchange_error": None,
                "plan": plan,
            }
        )
    except Exception as exc:
        snapshot.update(
            {
                "captured_at": captured_at,
                "exchange_error": str(exc),
                "plan": plan,
            }
        )
    return snapshot


def prime_latest_runtime_snapshot(
    state: dict[str, Any],
    plan: dict[str, Any] | None,
    equity: float | None = None,
) -> dict[str, Any]:
    snapshot = state.setdefault("latest_runtime_snapshot", {})
    snapshot["captured_at"] = utc_now().isoformat()
    snapshot["plan"] = plan
    if equity is not None:
        snapshot["equity"] = float(equity)
    snapshot.setdefault("positions", [])
    snapshot.setdefault("protections", [])
    snapshot.setdefault("exchange_error", None)
    return snapshot


def update_snapshot_from_sync_report(
    state: dict[str, Any],
    plan: dict[str, Any] | None,
    equity: float | None,
    sync_report: dict[str, Any],
) -> dict[str, Any]:
    snapshot = prime_latest_runtime_snapshot(state, plan, equity)
    snapshot["positions"] = sync_report.get("positions") or []
    snapshot["protections"] = sync_report.get("managed_protection_orders") or []
    return snapshot


def price_for_signed_return(entry_price: float, side: str, signed_return: float) -> float:
    direction = 1.0 if str(side).upper() == "LONG" else -1.0
    return float(entry_price) * (1.0 + direction * float(signed_return))


def normalized_positions_for_journal(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for pos in positions:
        normalized.append(
            {
                "pair": pos.get("pair"),
                "side": pos.get("side"),
                "qty": round(float(pos.get("qty", 0.0)), 8),
                "entry_price": round(float(pos.get("entry_price", 0.0)), 8),
            }
        )
    return sorted(normalized, key=lambda item: (str(item.get("pair")), str(item.get("side"))))


def build_core_exit_plan(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    day_state = state.get("core_day_state") or {}
    kill_switch_pct = float(day_state.get("kill_switch_pct", compute_core_kill_switch_pct(plan)))
    baseline_equity = day_state.get("baseline_equity")
    trigger_equity = None
    if baseline_equity not in (None, 0):
        trigger_equity = float(baseline_equity) * (1.0 - kill_switch_pct)
    active_pairs = [pair for pair, weight in (plan.get("core_weights") or {}).items() if abs(float(weight)) > EPSILON]
    return {
        "mode": "core",
        "active_pairs": active_pairs,
        "rebalance_notional_band_usd": float(REBALANCE_NOTIONAL_BAND_USD),
        "kill_switch": {
            "effective_day": day_state.get("effective_day") or plan.get("effective_day"),
            "baseline_equity": float(baseline_equity) if baseline_equity not in (None, 0) else None,
            "threshold_pct": kill_switch_pct,
            "trigger_equity": trigger_equity,
            "triggered": bool(day_state.get("kill_triggered", False)),
        },
        "shutdown_protection": state.get("shutdown_protection") or [],
        "rules": [
            {
                "code": "core_target_change",
                "description": "다음 리밸런싱에서 목표 비중이 줄거나 0이 되면 감축 또는 청산합니다.",
            },
            {
                "code": "session_change",
                "description": "세션이 overlay 또는 flat 으로 바뀌면 코어 포지션을 정리합니다.",
            },
            {
                "code": "kill_switch",
                "description": "당일 자산이 킬 스위치 임계값 아래로 내려가면 코어 포지션을 전량 정리합니다.",
            },
        ],
    }


def build_overlay_exit_plan(plan: dict[str, Any], state: dict[str, Any], positions: list[dict[str, Any]]) -> dict[str, Any]:
    overlay_state = state.get("overlay") or {}
    side = str(overlay_state.get("side") or (plan.get("overlay") or {}).get("side") or "FLAT")
    entry_price = overlay_state.get("entry_price")
    if entry_price in (None, 0, 0.0):
        for pos in positions:
            if pos.get("pair") == "BTCUSDT":
                entry_price = pos.get("entry_price")
                break
    template = dict(plan.get("overlay_exit_template") or build_overlay_exit_template())
    price_levels = None
    if entry_price not in (None, 0, 0.0) and side in {"LONG", "SHORT"}:
        entry_price_value = float(entry_price)
        price_levels = {
            "entry_price": entry_price_value,
            "stop_price": price_for_signed_return(entry_price_value, side, template["stop_return"]),
            "target_price": price_for_signed_return(entry_price_value, side, template["target_return"]),
            "trail_activation_price": price_for_signed_return(entry_price_value, side, template["trail_activation_return"]),
            "trail_floor_price": price_for_signed_return(entry_price_value, side, template["trail_floor_return"]),
        }
    return {
        "mode": "overlay",
        "side": side,
        "thresholds": template,
        "price_levels": price_levels,
        "rules": [
            {
                "code": "overlay_stop",
                "description": "기본 손절선에 닿으면 오버레이를 종료합니다.",
            },
            {
                "code": "overlay_target",
                "description": "목표 수익률에 닿으면 오버레이를 종료합니다.",
            },
            {
                "code": "overlay_trailing",
                "description": "트레일링이 활성화된 뒤 되밀리면 트레일링 손절로 종료합니다.",
            },
            {
                "code": "overlay_eod",
                "description": "적용일이 바뀌면 일자 종료 규칙으로 강제 청산합니다.",
            },
        ],
    }


def build_flat_exit_plan() -> dict[str, Any]:
    return {
        "mode": "flat",
        "rules": [
            {
                "code": "flat",
                "description": "현재는 포지션이 없으므로 종료 규칙도 비활성입니다.",
            }
        ],
    }


def build_entry_rationale(plan: dict[str, Any]) -> dict[str, Any]:
    if plan["session_type"] == "core":
        diagnostics = plan.get("core_diagnostics") or {}
        return {
            "mode": "core",
            "family": plan.get("core_strategy_family"),
            "strategy_key": plan.get("core_strategy_key"),
            "selected_mode": diagnostics.get("selected_mode"),
            "selected_pairs": [pair for pair, weight in plan["core_weights"].items() if abs(float(weight)) > EPSILON],
            "selected_positions": diagnostics.get("selected_positions"),
            "core_diagnostics": diagnostics,
            "core_validation_context": plan.get("core_validation_context"),
        }
    if plan["session_type"] == "overlay":
        return {
            "mode": "overlay",
            "overlay_signal": plan.get("overlay"),
            "core_active": bool(plan.get("core_active")),
        }
    return {
        "mode": "flat",
        "overlay_signal": plan.get("overlay"),
    }


def build_decision_snapshot(state: dict[str, Any]) -> dict[str, Any] | None:
    runtime_snapshot = state.get("latest_runtime_snapshot") or {}
    plan = runtime_snapshot.get("plan")
    if not isinstance(plan, dict):
        state["latest_decision_snapshot"] = None
        return None

    positions = list(runtime_snapshot.get("positions") or [])
    protections = list(runtime_snapshot.get("protections") or [])
    if plan["session_type"] == "core":
        exit_plan = build_core_exit_plan(plan, state)
    elif plan["session_type"] == "overlay":
        exit_plan = build_overlay_exit_plan(plan, state, positions)
    else:
        exit_plan = build_flat_exit_plan()

    decision = {
        "captured_at": runtime_snapshot.get("captured_at") or utc_now().isoformat(),
        "effective_day": plan.get("effective_day"),
        "market_day": plan.get("market_day"),
        "session_type": plan.get("session_type"),
        "equity": runtime_snapshot.get("equity"),
        "positions": positions,
        "protections": protections,
        "entry_rationale": build_entry_rationale(plan),
        "exit_plan": exit_plan,
    }
    state["latest_decision_snapshot"] = decision
    return decision


def decision_fingerprint(decision: dict[str, Any]) -> str:
    stable_payload = {
        "effective_day": decision.get("effective_day"),
        "market_day": decision.get("market_day"),
        "session_type": decision.get("session_type"),
        "positions": normalized_positions_for_journal(decision.get("positions") or []),
        "entry_rationale": decision.get("entry_rationale"),
        "exit_plan": decision.get("exit_plan"),
    }
    encoded = json.dumps(json_ready(stable_payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def refresh_decision_snapshot_and_journal(state: dict[str, Any]) -> dict[str, Any] | None:
    decision = build_decision_snapshot(state)
    journal_state = state.setdefault("decision_journal", {})
    journal_state["log_path"] = str(DECISION_LOG_PATH)
    if decision is None:
        return None

    fingerprint = decision_fingerprint(decision)
    decision["fingerprint"] = fingerprint
    if fingerprint != journal_state.get("last_fingerprint"):
        recorded_at = utc_now().isoformat()
        append_jsonl(
            DECISION_LOG_PATH,
            {
                "recorded_at": recorded_at,
                **decision,
            },
        )
        journal_state["last_fingerprint"] = fingerprint
        journal_state["last_recorded_at"] = recorded_at
    return decision


def install_shutdown_protection(
    exchange: ccxt.binanceusdm,
    state: dict[str, Any],
    execute: bool,
) -> dict[str, Any]:
    positions = fetch_open_position_map(exchange)
    existing_orders = fetch_strategy_protection_orders(exchange)
    installed: list[dict[str, Any]] = []
    desired_orders: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for _, position in positions.items():
        plan = build_shutdown_protection_plan(exchange, position)
        installed.append(plan)
        if plan.get("status") == "skipped":
            continue
        for desired in plan.get("desired_orders", []):
            desired_orders.append((plan, desired))

    retained_order_ids: set[int] = set()
    retained_reports_by_plan: dict[int, list[dict[str, Any]]] = {}
    orders_to_place: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for plan, desired in desired_orders:
        match: dict[str, Any] | None = None
        for order in existing_orders:
            order_identity = id(order)
            if order_identity in retained_order_ids:
                continue
            if protection_order_matches_desired(order, desired):
                match = order
                retained_order_ids.add(order_identity)
                break
        if match is None:
            orders_to_place.append((plan, desired))
            continue
        retained_reports_by_plan.setdefault(id(plan), []).append(
            {
                "tag": desired["tag"],
                "pair": desired["pair"],
                "symbol": desired["symbol"],
                "side": desired["side"],
                "amount": float(desired["amount"]),
                "stop_price": float(desired["stop_price"]),
                "id": match.get("id"),
                "client_order_id": extract_client_order_id(match),
                "type": match.get("type"),
                "status": "retained",
            }
        )

    orders_to_cancel = [order for order in existing_orders if id(order) not in retained_order_ids]
    cancel_actions = cancel_strategy_protection_orders(exchange, orders_to_cancel, execute)

    for plan in installed:
        if plan.get("status") == "skipped":
            continue
        retained_orders = retained_reports_by_plan.get(id(plan), [])
        planned_orders = [
            {
                "tag": desired["tag"],
                "pair": desired["pair"],
                "symbol": desired["symbol"],
                "side": desired["side"],
                "amount": float(desired["amount"]),
                "stop_price": float(desired["stop_price"]),
                "status": "planned",
            }
            for candidate_plan, desired in orders_to_place
            if candidate_plan is plan and not execute
        ]
        placed_orders = [
            place_shutdown_protection_order(exchange, desired, execute=True)
            for candidate_plan, desired in orders_to_place
            if candidate_plan is plan and execute
        ]
        plan["orders"] = [*retained_orders, *placed_orders, *planned_orders]
        if execute:
            plan["status"] = "retained" if retained_orders and not placed_orders else "placed"
        else:
            plan["status"] = "planned"

    if execute:
        state["shutdown_protection"] = installed
    status = "planned"
    if execute:
        if orders_to_place:
            status = "placed"
        elif cancel_actions:
            status = "cancelled"
        else:
            status = "retained"
    return {
        "status": status,
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
        "retained_count": int(len(retained_order_ids)),
        "placed_count": int(len(orders_to_place)) if execute else 0,
        "planned_place_count": int(len(orders_to_place)) if not execute else 0,
        "cancelled_count": int(sum(1 for action in cancel_actions if action.get("cancelled"))),
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
        update_snapshot_from_sync_report(state, plan, None, sync_report)
        refresh_decision_snapshot_and_journal(state)
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
        refresh_decision_snapshot_and_journal(state)
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
            runtime_snapshot = state.setdefault("latest_runtime_snapshot", {})
            runtime_snapshot["captured_at"] = utc_now().isoformat()
            runtime_snapshot["positions"] = []
            runtime_snapshot["protections"] = []
            runtime_snapshot["plan"] = None
            state["latest_decision_snapshot"] = None
            state.setdefault("decision_journal", {})["log_path"] = str(DECISION_LOG_PATH)
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
        prime_latest_runtime_snapshot(state, plan, equity)
        save_state(args.state_path, state)

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
        update_snapshot_from_sync_report(state, plan, equity, sync_report)
        save_state(args.state_path, state)
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

        update_latest_runtime_snapshot(exchange, state, plan, equity_hint=equity)
        refresh_decision_snapshot_and_journal(state)
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
        loop_state = load_state(args.state_path)
        update_runtime_health(
            loop_state,
            process_started_at=utc_now().isoformat(),
            pid=os.getpid(),
            poll_seconds=int(args.poll_seconds),
        )
        save_state(args.state_path, loop_state)
        while True:
            loop_started_at = utc_now()
            loop_state = load_state(args.state_path)
            update_runtime_health(
                loop_state,
                last_loop_started_at=loop_started_at.isoformat(),
                pid=os.getpid(),
                poll_seconds=int(args.poll_seconds),
            )
            save_state(args.state_path, loop_state)
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
                loop_completed_at = utc_now()
                loop_state = load_state(args.state_path)
                runtime_health = update_runtime_health(
                    loop_state,
                    last_loop_completed_at=loop_completed_at.isoformat(),
                    last_success_at=loop_completed_at.isoformat(),
                    last_error_message=None,
                    consecutive_errors=0,
                    last_duration_seconds=round((loop_completed_at - loop_started_at).total_seconds(), 3),
                    pid=os.getpid(),
                    poll_seconds=int(args.poll_seconds),
                )
                runtime_health["loop_success_count"] = int(runtime_health.get("loop_success_count", 0)) + 1
                save_state(args.state_path, loop_state)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                loop_failed_at = utc_now()
                loop_state = load_state(args.state_path)
                runtime_health = update_runtime_health(
                    loop_state,
                    last_loop_completed_at=loop_failed_at.isoformat(),
                    last_error_at=loop_failed_at.isoformat(),
                    last_error_message=str(exc),
                    last_duration_seconds=round((loop_failed_at - loop_started_at).total_seconds(), 3),
                    pid=os.getpid(),
                    poll_seconds=int(args.poll_seconds),
                )
                runtime_health["consecutive_errors"] = int(runtime_health.get("consecutive_errors", 0)) + 1
                runtime_health["loop_error_count"] = int(runtime_health.get("loop_error_count", 0)) + 1
                save_state(args.state_path, loop_state)
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
