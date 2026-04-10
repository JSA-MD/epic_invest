#!/usr/bin/env python3
"""Telegram control bot for Epic Invest trader."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from rotation_target_050_live import (
    PAIRS,
    STATE_PATH,
    UTC,
    build_strategy_plan,
    fetch_equity,
    fetch_open_position_map,
    fetch_strategy_protection_orders,
    get_exchange,
    load_state,
    resolve_default_leverage,
)

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"
TRADER_SCRIPT = ROOT_DIR / "scripts" / "rotation_target_050_live.py"
START_SCRIPT = ROOT_DIR / "start.sh"
STOP_SCRIPT = ROOT_DIR / "stop.sh"
RESTART_SCRIPT = ROOT_DIR / "restart.sh"

BOT_STATE_PATH = Path(os.getenv("TELEGRAM_BOT_STATE_PATH", "/tmp/epic-invest-telegram-bot-state.json"))
AUDIT_LOG_PATH = Path(os.getenv("TELEGRAM_BOT_AUDIT_LOG", "/tmp/epic-invest-telegram-audit.jsonl"))
TRADER_LOG_PATH = Path(os.getenv("TRADER_LOG_FILE", "/tmp/epic-invest-trader.log"))
TRADER_PID_PATH = Path(os.getenv("TRADER_PID_FILE", "/tmp/epic-invest-trader.pid"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_ALLOWED_CHAT_IDS = {
    int(value.strip())
    for value in os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "").split(",")
    if value.strip()
}
DEFAULT_MODE = os.getenv("BINANCE_MODE", "demo")
POLL_TIMEOUT_SECONDS = int(os.getenv("TELEGRAM_POLL_TIMEOUT_SECONDS", "30"))
PENDING_CONFIRM_TTL_SECONDS = int(os.getenv("TELEGRAM_CONFIRM_TTL_SECONDS", "120"))
MESSAGE_LIMIT = 3500
TRADER_STATE_STALE_SECONDS = int(os.getenv("TRADER_STATE_STALE_SECONDS", "180"))
BOT_MAX_WORKERS = max(1, int(os.getenv("TELEGRAM_BOT_MAX_WORKERS", "4")))
ASYNC_READ_COMMANDS = {
    "status",
    "plan",
    "positions",
    "protection",
    "killswitch",
    "logs",
    "recent",
    "why",
    "rationale",
    "reason",
    "exitplan",
}
SNAPSHOT_CACHE_STALE_SECONDS = int(os.getenv("TELEGRAM_SNAPSHOT_CACHE_STALE_SECONDS", "180"))
STATE_LOCK = threading.Lock()
COMMAND_EXECUTOR = ThreadPoolExecutor(max_workers=BOT_MAX_WORKERS, thread_name_prefix="telegram-bot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram control bot for Epic Invest trader.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("loop", help="Run Telegram bot long polling loop.")
    sub.add_parser("print-help", help="Print supported Telegram commands.")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(UTC)


def default_bot_runtime() -> dict[str, Any]:
    return {
        "last_started_at": None,
        "last_poll_started_at": None,
        "last_poll_ok_at": None,
        "last_reply_at": None,
        "last_error_at": None,
        "last_error_message": None,
        "consecutive_poll_errors": 0,
        "last_update_id": None,
        "last_command": None,
        "pid": None,
    }


def ensure_bot_runtime(state: dict[str, Any]) -> dict[str, Any]:
    runtime = state.setdefault("runtime", {})
    defaults = default_bot_runtime()
    for key, value in defaults.items():
        runtime.setdefault(key, value)
    return runtime


def update_bot_runtime(state: dict[str, Any], **updates: Any) -> dict[str, Any]:
    runtime = ensure_bot_runtime(state)
    for key, value in updates.items():
        runtime[key] = value
    return runtime


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def age_seconds(value: Any) -> float | None:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return None
    return max(0.0, (utc_now() - parsed).total_seconds())


def latest_signal(values: list[Any]) -> str | None:
    parsed: list[tuple[datetime, str]] = []
    for value in values:
        dt = parse_iso_datetime(value)
        if dt is None:
            continue
        parsed.append((dt, str(value)))
    if not parsed:
        return None
    parsed.sort(key=lambda item: item[0])
    return parsed[-1][1]


def load_bot_state() -> dict[str, Any]:
    if not BOT_STATE_PATH.exists():
        return {"offset": None, "pending": {}, "runtime": default_bot_runtime()}
    with open(BOT_STATE_PATH, "r") as f:
        state = json.load(f)
    state.setdefault("offset", None)
    state.setdefault("pending", {})
    ensure_bot_runtime(state)
    return state


def save_bot_state(state: dict[str, Any]) -> None:
    BOT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BOT_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def audit(event_type: str, payload: dict[str, Any]) -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "time": utc_now().isoformat(),
        "event": event_type,
        "payload": payload,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def telegram_api(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")

    payload = urlencode(params or {}).encode()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    request = Request(url, data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urlopen(request, timeout=POLL_TIMEOUT_SECONDS + 10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Telegram API error: {exc}") from exc
    if not data.get("ok"):
        raise RuntimeError(f"Telegram API failure: {data}")
    return data


def get_updates(offset: int | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"timeout": POLL_TIMEOUT_SECONDS}
    if offset is not None:
        params["offset"] = offset
    return telegram_api("getUpdates", params).get("result", [])


def chunk_text(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks


def send_message(chat_id: int, text: str) -> None:
    for chunk in chunk_text(text):
        telegram_api(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": "true",
            },
        )


def send_chat_action(chat_id: int, action: str = "typing") -> None:
    telegram_api(
        "sendChatAction",
        {
            "chat_id": chat_id,
            "action": action,
        },
    )


def processing_notice(state: dict[str, Any], chat_id: int, command: str) -> str | None:
    if command in ASYNC_READ_COMMANDS:
        return (
            f"/{command} 요청을 받았습니다.\n"
            "지금 상태를 확인하는 중입니다. 결과를 곧 보내드리겠습니다."
        )
    if command == "confirm":
        pending = state.get("pending", {}).get(str(chat_id))
        summary = str((pending or {}).get("summary") or "요청한 작업")
        return (
            f"{summary} 요청을 실행 중입니다.\n"
            "먼저 진행 중임을 알려드립니다. 완료되면 결과를 바로 보내드리겠습니다."
        )
    return None


def normalize_command(text: str) -> tuple[str, list[str]]:
    tokens = text.strip().split()
    if not tokens:
        return "", []
    command = tokens[0]
    if not command.startswith("/"):
        return "", tokens
    command = command[1:]
    if "@" in command:
        command = command.split("@", 1)[0]
    return command.lower(), tokens[1:]


def trader_process_rows() -> list[dict[str, str]]:
    entry = str(TRADER_SCRIPT)
    try:
        output = subprocess.run(
            ["ps", "-ax", "-o", "pid=,ppid=,stat=,etime=,command="],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for line in output:
        if entry not in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "pid": parts[0],
                "ppid": parts[1],
                "stat": parts[2],
                "etime": parts[3],
                "command": parts[4],
            }
        )
    return rows


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def is_pid_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def trader_runtime_pid() -> int | None:
    state = load_state(STATE_PATH)
    runtime_health = state.get("runtime_health") or {}
    raw_pid = runtime_health.get("pid")
    try:
        return int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        return None


def live_state_fresh() -> bool:
    state = load_state(STATE_PATH)
    runtime_health = state.get("runtime_health") or {}
    last_progress_at = latest_signal(
        [
            runtime_health.get("last_success_at"),
            runtime_health.get("last_loop_started_at"),
            runtime_health.get("last_loop_completed_at"),
            state.get("updated_at"),
        ]
    )
    last_age = age_seconds(last_progress_at)
    return last_age is not None and last_age <= TRADER_STATE_STALE_SECONDS


def snapshot_cache_age(snapshot: dict[str, Any] | None) -> float | None:
    if not snapshot:
        return None
    return age_seconds(snapshot.get("captured_at"))


def snapshot_cache_fresh(snapshot: dict[str, Any] | None) -> bool:
    age = snapshot_cache_age(snapshot)
    return age is not None and age <= SNAPSHOT_CACHE_STALE_SECONDS


def cached_positions_map(snapshot: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    positions = (snapshot or {}).get("positions") or []
    result: dict[str, dict[str, Any]] = {}
    for pos in positions:
        pair = str(pos.get("pair") or "")
        if not pair:
            continue
        result[pair] = pos
    return result


def is_trader_running() -> bool:
    runtime_pid = trader_runtime_pid()
    if is_pid_running(runtime_pid):
        return True
    pid = read_pid(TRADER_PID_PATH)
    if is_pid_running(pid):
        return True
    rows = trader_process_rows()
    if rows:
        return True
    return live_state_fresh()


def read_recent_lines(path: Path, max_lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return list(deque((line.rstrip() for line in f), maxlen=max_lines))


def read_recent_audit(max_items: int = 10) -> list[dict[str, Any]]:
    if not AUDIT_LOG_PATH.exists():
        return []
    rows = deque(maxlen=max_items)
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(rows)


def get_runtime_snapshot() -> dict[str, Any]:
    bot_state = load_bot_state()
    bot_runtime = ensure_bot_runtime(bot_state)
    trader_state = load_state(STATE_PATH)
    trader_runtime = trader_state.get("runtime_health") or {}
    trader_pid = trader_runtime_pid() or read_pid(TRADER_PID_PATH)
    trader_rows = trader_process_rows()
    cached_snapshot = trader_state.get("latest_runtime_snapshot") or {}
    snapshot: dict[str, Any] = {
        "mode": DEFAULT_MODE,
        "leverage": resolve_default_leverage(),
        "processes": trader_rows,
        "state": trader_state,
        "trader_running": bool(trader_rows) or is_pid_running(trader_pid) or live_state_fresh(),
        "trader_pid": trader_pid,
        "trader_runtime": trader_runtime,
        "bot_runtime": bot_runtime,
        "snapshot_age_seconds": snapshot_cache_age(cached_snapshot),
        "snapshot_captured_at": cached_snapshot.get("captured_at"),
        "snapshot_ready": bool(cached_snapshot.get("captured_at")),
        "decision": trader_state.get("latest_decision_snapshot"),
    }
    if cached_snapshot:
        snapshot["equity"] = cached_snapshot.get("equity")
        snapshot["positions"] = cached_positions_map(cached_snapshot)
        snapshot["protections"] = cached_snapshot.get("protections") or []
        snapshot["exchange_error"] = cached_snapshot.get("exchange_error")
        snapshot["plan"] = cached_snapshot.get("plan")
        if not snapshot["plan"] and snapshot_cache_fresh(cached_snapshot):
            snapshot["plan_error"] = "트레이더 초기화 중입니다."
    else:
        snapshot["equity"] = None
        snapshot["positions"] = {}
        snapshot["protections"] = []
        snapshot["exchange_error"] = "트레이더 캐시가 아직 준비되지 않았습니다."
        snapshot["plan"] = None
        snapshot["plan_error"] = "트레이더 캐시가 아직 준비되지 않았습니다."

    return snapshot


def format_pct(value: Any, signed: bool = True) -> str:
    if value is None:
        return "-"
    number = float(value) * 100.0
    return f"{number:+.2f}%" if signed else f"{number:.2f}%"


def format_price(value: Any) -> str:
    if value is None:
        return "-"
    return f"${float(value):,.4f}"


def format_qty(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.6f}"


def side_label(side: str) -> str:
    return {"LONG": "롱", "SHORT": "숏"}.get(str(side).upper(), str(side))


def protection_summary_lines(protections: list[dict[str, Any]]) -> list[str]:
    if not protections:
        return ["  서버측 종료 보호주문은 현재 없습니다."]
    lines = ["  서버측 종료 보호주문:"]
    for item in protections[:4]:
        pair = item.get("pair") or item.get("symbol") or "알 수 없음"
        stop_price = item.get("stop_price")
        take_price = item.get("take_price")
        status = item.get("status")
        if stop_price is not None or take_price is not None:
            lines.append(
                f"    {pair}: 상태={status} | 손절 {format_price(stop_price)} | 익절 {format_price(take_price)}"
            )
        else:
            lines.append(f"    {pair}: 상태={status}")
    return lines


def format_decision(snapshot: dict[str, Any]) -> str:
    decision = snapshot.get("decision") or {}
    if not decision:
        if not snapshot.get("snapshot_ready"):
            return "진입근거 스냅샷을 준비 중입니다. 첫 동기화가 끝나면 /why 로 확인할 수 있습니다."
        return "아직 진입근거 스냅샷이 없습니다. 트레이더가 한 번 더 루프를 완료한 뒤 다시 시도하세요."

    lines = ["현재 포지션 설명"]
    if decision.get("captured_at"):
        lines.append(f"- 데이터 기준시각: {decision['captured_at']}")
    lines.append(f"- 세션: {str(decision.get('session_type', '')).upper()}")
    lines.append(f"- 시장 기준일: {decision.get('market_day')}")
    lines.append(f"- 적용일: {decision.get('effective_day')}")

    positions = decision.get("positions") or []
    if positions:
        lines.append("- 현재 포지션:")
        for pos in positions:
            lines.append(
                f"  {pos.get('pair')}: {side_label(str(pos.get('side')))} {format_qty(pos.get('qty'))} | "
                f"진입가 {format_price(pos.get('entry_price'))} | 현재가 {format_price(pos.get('mark_price'))}"
            )
    else:
        lines.append("- 현재 포지션: 없음")

    rationale = decision.get("entry_rationale") or {}
    rationale_mode = rationale.get("mode")
    exit_plan = decision.get("exit_plan") or {}
    if rationale_mode == "core":
        core = rationale.get("core_diagnostics") or {}
        selected_pairs = rationale.get("selected_pairs") or []
        lines.append("- 진입근거:")
        lines.append(
            f"  코어 세션이며 선택 종목은 {', '.join(selected_pairs) if selected_pairs else '없음'} 입니다."
        )
        if core:
            lines.append(
                f"  BTC 레짐 {format_pct(core.get('btc_regime'))} / 기준 {format_pct(core.get('regime_threshold'), signed=False)}"
            )
            lines.append(
                f"  breadth {format_pct(core.get('breadth'), signed=False)} / 기준 {format_pct(core.get('breadth_threshold'), signed=False)}"
            )
            ranked = core.get("positive_ranked") or []
            if ranked:
                ranked_preview = ", ".join(
                    f"{row.get('pair')} {format_pct(row.get('value'))}"
                    for row in ranked[:4]
                )
                lines.append(f"  양의 모멘텀 순위: {ranked_preview}")
            if selected_pairs:
                first_pair = selected_pairs[0]
                unlevered = ((core.get("unlevered_weights") or {}).get(first_pair))
                levered = ((core.get("levered_weights") or {}).get(first_pair))
                selected_vol = ((core.get("selected_vol_ann") or {}).get(first_pair))
                if unlevered is not None and levered is not None:
                    lines.append(
                        f"  목표 비중: 비레버리지 {format_pct(unlevered)} -> 레버리지 적용 {format_pct(levered)}"
                    )
                if selected_vol is not None:
                    lines.append(f"  선택 종목 연환산 변동성: {format_pct(selected_vol, signed=False)}")
        lines.append("- 종료 조건:")
        kill_switch = exit_plan.get("kill_switch") or {}
        if kill_switch.get("trigger_equity") is not None:
            lines.append(
                "  코어 킬 스위치: "
                f"기준 자산 {format_price(kill_switch.get('baseline_equity'))}, "
                f"임계값 {format_pct(kill_switch.get('threshold_pct'), signed=False)}, "
                f"트리거 {format_price(kill_switch.get('trigger_equity'))}"
            )
        lines.append("  다음 리밸런싱에서 목표 비중이 줄거나 0이 되면 감축 또는 청산합니다.")
        lines.append("  세션이 OVERLAY 또는 FLAT 으로 바뀌면 코어 포지션을 정리합니다.")
        lines.extend(protection_summary_lines(exit_plan.get("shutdown_protection") or []))
        return "\n".join(lines)

    if rationale_mode == "overlay":
        overlay = rationale.get("overlay_signal") or {}
        lines.append("- 진입근거:")
        lines.append("  코어가 비활성이어서 BTC 오버레이 세션을 사용합니다.")
        lines.append(
            f"  오버레이 방향 {overlay.get('side')} | 신호 {format_pct(float(overlay.get('signal_pct', 0.0)) / 100.0)}"
        )
        lines.append(
            f"  BTC 1일 수익률 {format_pct(overlay.get('btc_ret1'))} | breadth3 {format_pct(overlay.get('breadth3'), signed=False)}"
        )
        lines.append("- 종료 조건:")
        thresholds = exit_plan.get("thresholds") or {}
        lines.append(
            f"  기본 손절 {format_pct(thresholds.get('stop_return'))} | 기본 익절 {format_pct(thresholds.get('target_return'))}"
        )
        lines.append(
            "  트레일링: "
            f"활성화 {format_pct(thresholds.get('trail_activation_return'))}, "
            f"바닥 {format_pct(thresholds.get('trail_floor_return'))}, "
            f"거리 {format_pct(thresholds.get('trail_distance_return'), signed=False)}"
        )
        price_levels = exit_plan.get("price_levels") or {}
        if price_levels:
            lines.append(
                "  가격 레벨: "
                f"손절 {format_price(price_levels.get('stop_price'))}, "
                f"익절 {format_price(price_levels.get('target_price'))}, "
                f"트레일 활성화 {format_price(price_levels.get('trail_activation_price'))}"
            )
        lines.append("  적용일이 바뀌면 오버레이를 강제 청산합니다.")
        return "\n".join(lines)

    lines.append("- 진입근거: 현재 세션이 FLAT 이라 새 포지션을 잡지 않습니다.")
    lines.append("- 종료 조건: 현재 열린 포지션이 없어서 적용되지 않습니다.")
    return "\n".join(lines)


def format_status(snapshot: dict[str, Any]) -> str:
    processes = snapshot["processes"]
    state = snapshot["state"]
    plan = snapshot.get("plan")
    positions = snapshot.get("positions", {})
    protections = snapshot.get("protections", [])
    trader_runtime = snapshot.get("trader_runtime") or {}
    bot_runtime = snapshot.get("bot_runtime") or {}
    trader_running = bool(snapshot.get("trader_running"))
    snapshot_ready = bool(snapshot.get("snapshot_ready"))

    lines = ["에픽 인베스트 상태"]
    lines.append(f"- 트레이더: {'실행 중' if trader_running else '중지됨'}")
    if processes:
        p = processes[0]
        lines.append(f"- PID: {p['pid']} ({p['etime']})")
    elif snapshot.get("trader_pid"):
        lines.append(f"- PID: {snapshot['trader_pid']} (PID 파일 기준)")
    lines.append(f"- 모드: {snapshot['mode']}")
    lines.append(f"- 전략 레버리지: {snapshot['leverage']:.2f}x")

    equity = snapshot.get("equity")
    if equity is not None:
        lines.append(f"- 자산: ${float(equity):,.2f}")
    if snapshot.get("exchange_error"):
        lines.append(f"- 거래소 오류: {snapshot['exchange_error']}")
    if snapshot.get("snapshot_captured_at"):
        lines.append(f"- 데이터 기준시각: {snapshot['snapshot_captured_at']}")
    elif trader_running:
        lines.append("- 상세 데이터: 초기 동기화 중")

    if plan:
        lines.append(f"- 세션: {plan['session_type'].upper()}")
        lines.append(f"- 시장 기준일: {plan['market_day']}")
        lines.append(f"- 적용일: {plan['effective_day']}")
        lines.append(f"- 코어 총 레버리지: {plan['core_gross_leverage']:.3f}x")

    if snapshot_ready:
        lines.append(f"- 열린 포지션 수: {len(positions)}")
        lines.append(f"- 관리 중 보호주문 수: {len(protections)}")
    else:
        lines.append("- 열린 포지션 수: 초기 동기화 중")
        lines.append("- 관리 중 보호주문 수: 초기 동기화 중")
    if trader_runtime.get("last_success_at"):
        lines.append(f"- 최근 트레이더 정상 시각: {trader_runtime['last_success_at']}")
    if bot_runtime.get("last_reply_at"):
        lines.append(f"- 최근 봇 응답 시각: {bot_runtime['last_reply_at']}")
    if trader_runtime.get("last_error_message"):
        lines.append(f"- 최근 트레이더 오류: {trader_runtime['last_error_message']}")

    core_day_state = state.get("core_day_state") or {}
    if core_day_state:
        lines.append(
            "- 킬 스위치: "
            f"발동={bool(core_day_state.get('kill_triggered', False))}, "
            f"최근 수익률={float(core_day_state.get('last_return', 0.0))*100:+.2f}%, "
            f"임계값={float(core_day_state.get('kill_switch_pct', 0.0))*100:.2f}%"
        )

    return "\n".join(lines)


def format_plan(snapshot: dict[str, Any]) -> str:
    plan = snapshot.get("plan")
    if not plan:
        if not snapshot.get("snapshot_ready"):
            return "전략 계획 캐시를 준비 중입니다. 첫 동기화가 끝나면 바로 조회됩니다."
        return f"전략 계획을 불러오지 못했습니다: {snapshot.get('plan_error', '알 수 없는 오류')}"

    lines = ["오늘의 전략 계획"]
    if snapshot.get("snapshot_captured_at"):
        lines.append(f"- 데이터 기준시각: {snapshot['snapshot_captured_at']}")
    lines.append(f"- 세션: {plan['session_type'].upper()}")
    lines.append(f"- 시장 기준일: {plan['market_day']}")
    lines.append(f"- 적용일: {plan['effective_day']}")
    lines.append(f"- 전략 레버리지: {plan['leverage']:.2f}x")
    lines.append(f"- 코어 총 레버리지: {plan['core_gross_leverage']:.3f}x")
    lines.append("- 코어 비중:")
    for pair in PAIRS:
        lines.append(f"  {pair}: {float(plan['core_weights'][pair]):+.3%}")
    overlay = plan["overlay"]
    lines.append("- 오버레이:")
    lines.append(f"  방향={overlay['side']}, 신호={overlay['signal_pct']:+.1f}%")
    lines.append(f"  BTC 1일 수익률={overlay['btc_ret1']:+.4%}, breadth3={overlay['breadth3']:.2%}")
    return "\n".join(lines)


def format_positions(snapshot: dict[str, Any]) -> str:
    positions = snapshot.get("positions", {})
    if not snapshot.get("snapshot_ready"):
        return "포지션 캐시를 준비 중입니다. 첫 동기화가 끝나면 바로 조회됩니다."
    if not positions:
        return "열린 포지션이 없습니다."
    lines = ["열린 포지션"]
    if snapshot.get("snapshot_captured_at"):
        lines.append(f"- 데이터 기준시각: {snapshot['snapshot_captured_at']}")
    for pair, pos in positions.items():
        side = {"LONG": "롱", "SHORT": "숏"}.get(str(pos["side"]), str(pos["side"]))
        margin_mode = {"isolated": "아이솔레이티드", "cross": "크로스"}.get(
            str(pos["margin_mode"] or "").lower(),
            pos["margin_mode"] or "알 수 없음",
        )
        lines.append(
            f"- {pair}: {side} {pos['qty']:+.6f} | "
            f"진입가 ${pos['entry_price']:,.4f} | 현재가 ${pos['mark_price']:,.4f} | "
            f"마진 {margin_mode}"
        )
    return "\n".join(lines)


def format_protections(snapshot: dict[str, Any]) -> str:
    protections = snapshot.get("protections", [])
    if not snapshot.get("snapshot_ready"):
        return "보호주문 캐시를 준비 중입니다. 첫 동기화가 끝나면 바로 조회됩니다."
    if not protections:
        return "관리 중인 보호주문이 없습니다."
    lines = ["관리 중인 보호주문"]
    if snapshot.get("snapshot_captured_at"):
        lines.append(f"- 데이터 기준시각: {snapshot['snapshot_captured_at']}")
    for order in protections:
        lines.append(
            f"- {order.get('symbol')}: 유형={order.get('type')} 방향={order.get('side')} "
            f"id={order.get('id')} client={order.get('clientOrderId') or order.get('client_order_id') or '없음'}"
        )
    return "\n".join(lines)


def format_killswitch(snapshot: dict[str, Any]) -> str:
    core_day_state = (snapshot.get("state") or {}).get("core_day_state")
    if not core_day_state:
        return "현재 세션에서는 킬 스위치 상태가 비활성입니다."
    return "\n".join(
        [
            "코어 킬 스위치",
            f"- 적용일: {core_day_state.get('effective_day')}",
            f"- 기준 자산: ${float(core_day_state.get('baseline_equity', 0.0)):,.2f}",
            f"- 최근 자산: ${float(core_day_state.get('last_equity', 0.0)):,.2f}",
            f"- 최근 수익률: {float(core_day_state.get('last_return', 0.0))*100:+.2f}%",
            f"- 임계값: {float(core_day_state.get('kill_switch_pct', 0.0))*100:.2f}%",
            f"- 발동 여부: {bool(core_day_state.get('kill_triggered', False))}",
        ]
    )


def format_logs() -> str:
    lines = read_recent_lines(TRADER_LOG_PATH, max_lines=25)
    if not lines:
        return "트레이더 로그가 비어 있습니다."
    return "최근 트레이더 로그\n" + "\n".join(lines)


def format_recent_audit() -> str:
    rows = read_recent_audit(10)
    if not rows:
        return "최근 봇 감사 기록이 없습니다."
    lines = ["최근 봇 감사 기록"]
    for row in rows:
        payload = row.get("payload", {})
        lines.append(f"- {row.get('time')} | {row.get('event')} | {json.dumps(payload, ensure_ascii=False)}")
    return "\n".join(lines)


def build_help_text() -> str:
    return "\n".join(
        [
            "에픽 인베스트 텔레그램 명령어",
            "",
            "조회",
            "/help - 도움말 보기",
            "/start - 도움말 보기",
            "/status - 트레이더/자산/세션 요약",
            "/plan - 오늘의 코어 비중과 오버레이 신호",
            "/positions - 열린 포지션 조회",
            "/why - 현재 포지션의 진입근거와 종료 조건",
            "/protection - 관리 중인 보호주문 조회",
            "/killswitch - 코어 킬 스위치 상태 조회",
            "/logs - 최근 트레이더 로그",
            "/recent - 최근 봇 감사 로그",
            "/ping - 봇 상태 확인",
            "",
            "제어",
            "/starttrader - 트레이더 루프 시작",
            "/stoptrader - 트레이더 루프 중지",
            "/restarttrader - 트레이더 루프 재시작",
            "/sync - 상태 동기화 및 관리 중 보호주문 정리",
            "/protect - fallback reduceOnly 보호주문 설치",
            "/closeall - 트레이더 중지 후 전체 포지션 종료",
            "/flatten - /closeall 과 동일",
            "",
            "안전장치",
            "/confirm <토큰> - 대기 중인 제어 명령 확인",
            "/cancel - 대기 중인 제어 명령 취소",
        ]
    )


def run_local_command(command: list[str], timeout: int = 120) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def summarize_command_result(title: str, result: dict[str, Any]) -> str:
    lines = [title, f"- 종료 코드: {result['returncode']}"]
    if result["stdout"]:
        stdout_lines = result["stdout"].splitlines()
        lines.append("- 출력:")
        lines.extend(f"  {line}" for line in stdout_lines[-20:])
    if result["stderr"]:
        stderr_lines = result["stderr"].splitlines()
        lines.append("- 오류:")
        lines.extend(f"  {line}" for line in stderr_lines[-20:])
    return "\n".join(lines)


def make_pending_action(state: dict[str, Any], chat_id: int, action: str, summary: str) -> str:
    token = secrets.token_hex(3)
    state["pending"][str(chat_id)] = {
        "action": action,
        "summary": summary,
        "token": token,
        "expires_at": (utc_now() + timedelta(seconds=PENDING_CONFIRM_TTL_SECONDS)).isoformat(),
    }
    save_bot_state(state)
    return token


def clear_pending_action(state: dict[str, Any], chat_id: int) -> None:
    state["pending"].pop(str(chat_id), None)
    save_bot_state(state)


def execute_control_action(action: str) -> str:
    if action == "starttrader":
        result = run_local_command([str(START_SCRIPT)])
        return summarize_command_result("트레이더 시작", result)
    if action == "stoptrader":
        result = run_local_command([str(STOP_SCRIPT)])
        return summarize_command_result("트레이더 종료", result)
    if action == "restarttrader":
        result = run_local_command([str(RESTART_SCRIPT)])
        return summarize_command_result("트레이더 재시작", result)
    if action == "sync":
        result = run_local_command([str(PYTHON_BIN), str(TRADER_SCRIPT), "sync-state", "--execute"])
        return summarize_command_result("상태 동기화", result)
    if action == "protect":
        result = run_local_command([str(PYTHON_BIN), str(TRADER_SCRIPT), "shutdown-protect", "--execute"])
        return summarize_command_result("보호주문 설치", result)
    if action in {"closeall", "flatten"}:
        steps: list[str] = []
        if is_trader_running():
            stop_result = run_local_command([str(STOP_SCRIPT)])
            steps.append(summarize_command_result("트레이더 종료", stop_result))
        close_result = run_local_command([str(PYTHON_BIN), str(TRADER_SCRIPT), "close-all", "--execute"])
        steps.append(summarize_command_result("전체 포지션 종료", close_result))
        return "\n\n".join(steps)
    raise ValueError(f"Unsupported action: {action}")


def handle_read_command(command: str) -> str:
    if command in {"start", "help"}:
        return build_help_text()
    if command == "ping":
        return "정상입니다."

    snapshot = get_runtime_snapshot()
    if command == "status":
        return format_status(snapshot)
    if command == "plan":
        return format_plan(snapshot)
    if command == "positions":
        return format_positions(snapshot)
    if command in {"why", "rationale", "reason", "exitplan"}:
        return format_decision(snapshot)
    if command == "protection":
        return format_protections(snapshot)
    if command == "killswitch":
        return format_killswitch(snapshot)
    if command == "logs":
        return format_logs()
    if command == "recent":
        return format_recent_audit()
    return ""


def handle_command(state: dict[str, Any], chat_id: int, text: str) -> str:
    command, args = normalize_command(text)
    if not command:
        return "지원하지 않는 입력입니다. /help 를 사용하세요."

    if command in {
        "start",
        "help",
        "ping",
        "status",
        "plan",
        "positions",
        "why",
        "rationale",
        "reason",
        "exitplan",
        "protection",
        "killswitch",
        "logs",
        "recent",
    }:
        return handle_read_command(command)

    if command == "cancel":
        clear_pending_action(state, chat_id)
        return "대기 중인 제어 명령을 취소했습니다."

    if command == "confirm":
        pending = state["pending"].get(str(chat_id))
        if pending is None:
            return "확인 대기 중인 명령이 없습니다."
        if not args:
            return "토큰이 필요합니다. 예: /confirm abc123"
        if args[0] != pending.get("token"):
            return "토큰이 일치하지 않습니다."
        expires_at = datetime.fromisoformat(pending["expires_at"])
        if utc_now() > expires_at:
            clear_pending_action(state, chat_id)
            return "확인 시간이 만료되었습니다. 명령을 다시 요청하세요."

        action = str(pending["action"])
        clear_pending_action(state, chat_id)
        audit("control_execute", {"chat_id": chat_id, "action": action})
        return execute_control_action(action)

    control_commands = {
        "starttrader": "트레이더 시작",
        "stoptrader": "트레이더 종료",
        "restarttrader": "트레이더 재시작",
        "sync": "상태 동기화",
        "protect": "보호주문 설치",
        "closeall": "트레이더 중지 후 모든 포지션 종료",
        "flatten": "트레이더 중지 후 모든 포지션 종료",
    }
    if command in control_commands:
        token = make_pending_action(state, chat_id, command, control_commands[command])
        audit("control_requested", {"chat_id": chat_id, "action": command, "token": token})
        return (
            f"{control_commands[command]} 요청이 접수됐습니다.\n"
            f"확인하려면 {PENDING_CONFIRM_TTL_SECONDS}초 안에 /confirm {token}\n"
            f"취소하려면 /cancel"
        )

    return "알 수 없는 명령입니다. /help 를 사용하세요."


def record_bot_reply(state: dict[str, Any], command: str) -> None:
    with STATE_LOCK:
        update_bot_runtime(
            state,
            last_reply_at=utc_now().isoformat(),
            last_command=command,
            last_error_message=None,
            pid=os.getpid(),
        )
        save_bot_state(state)


def record_bot_error(state: dict[str, Any], command: str, exc: Exception) -> None:
    with STATE_LOCK:
        update_bot_runtime(
            state,
            last_error_at=utc_now().isoformat(),
            last_error_message=f"{command}: {exc}",
            last_command=command,
            pid=os.getpid(),
        )
        save_bot_state(state)


def send_response_and_audit(state: dict[str, Any], chat_id: int, text: str, response: str, command: str) -> None:
    send_message(int(chat_id), response)
    record_bot_reply(state, command)
    audit("outgoing_message", {"chat_id": chat_id, "response_preview": response[:500]})


def handle_async_read_command(state: dict[str, Any], chat_id: int, text: str, command: str) -> None:
    try:
        send_chat_action(int(chat_id), "typing")
    except Exception:
        pass
    try:
        response = handle_command(state, int(chat_id), text)
    except Exception as exc:
        response = f"명령 처리 중 오류가 발생했습니다: {exc}"
        audit("command_error", {"chat_id": chat_id, "text": text, "error": str(exc)})
        record_bot_error(state, command, exc)
    try:
        send_response_and_audit(state, int(chat_id), text, response, command)
    except Exception as exc:
        audit(
            "send_error",
            {
                "chat_id": chat_id,
                "text": text,
                "response_preview": response[:500],
                "error": str(exc),
            },
        )
        record_bot_error(state, command, exc)


def process_update(state: dict[str, Any], update: dict[str, Any]) -> None:
    message = update.get("message") or update.get("edited_message")
    if not message:
        return
    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = (message.get("text") or "").strip()
    if not text or chat_id is None:
        return

    audit("incoming_message", {"chat_id": chat_id, "text": text})

    if TELEGRAM_ALLOWED_CHAT_IDS and int(chat_id) not in TELEGRAM_ALLOWED_CHAT_IDS:
        audit("unauthorized_chat", {"chat_id": chat_id, "text": text})
        try:
            send_message(int(chat_id), "허용되지 않은 chat_id 입니다.")
        except Exception:
            pass
        return

    command, _ = normalize_command(text)
    notice = processing_notice(state, int(chat_id), command)
    if notice:
        try:
            send_message(int(chat_id), notice)
            audit("processing_notice", {"chat_id": chat_id, "command": command, "preview": notice[:500]})
        except Exception as exc:
            audit(
                "processing_notice_error",
                {"chat_id": chat_id, "command": command, "error": str(exc)},
            )

    if command in ASYNC_READ_COMMANDS:
        COMMAND_EXECUTOR.submit(handle_async_read_command, state, int(chat_id), text, command)
        audit("async_command_queued", {"chat_id": chat_id, "command": command})
        return

    try:
        response = handle_command(state, int(chat_id), text)
    except Exception as exc:
        response = f"명령 처리 중 오류가 발생했습니다: {exc}"
        audit("command_error", {"chat_id": chat_id, "text": text, "error": str(exc)})
        record_bot_error(state, command or "unknown", exc)

    try:
        send_response_and_audit(state, int(chat_id), text, response, command or "unknown")
    except Exception as exc:
        audit(
            "send_error",
            {
                "chat_id": chat_id,
                "text": text,
                "response_preview": response[:500],
                "error": str(exc),
            },
        )
        record_bot_error(state, command or "unknown", exc)
        return


def run_loop() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured")
    if not TELEGRAM_ALLOWED_CHAT_IDS:
        raise RuntimeError("TELEGRAM_ALLOWED_CHAT_IDS is not configured")
    state = load_bot_state()
    with STATE_LOCK:
        update_bot_runtime(
            state,
            last_started_at=utc_now().isoformat(),
            last_error_message=None,
            consecutive_poll_errors=0,
            pid=os.getpid(),
        )
        save_bot_state(state)
    print("Telegram bot ready")
    audit("bot_start", {"allowed_chat_ids": sorted(TELEGRAM_ALLOWED_CHAT_IDS)})

    while True:
        try:
            with STATE_LOCK:
                update_bot_runtime(state, last_poll_started_at=utc_now().isoformat(), pid=os.getpid())
                save_bot_state(state)
            updates = get_updates(state.get("offset"))
            with STATE_LOCK:
                update_bot_runtime(
                    state,
                    last_poll_ok_at=utc_now().isoformat(),
                    consecutive_poll_errors=0,
                    last_error_message=None,
                    pid=os.getpid(),
                )
                save_bot_state(state)
            for update in updates:
                update_id = int(update["update_id"])
                state["offset"] = update_id + 1
                with STATE_LOCK:
                    update_bot_runtime(state, last_update_id=update_id, pid=os.getpid())
                    save_bot_state(state)
                try:
                    process_update(state, update)
                except Exception as exc:
                    audit("update_error", {"update_id": update_id, "error": str(exc)})
                    record_bot_error(state, "update", exc)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            audit("poll_error", {"error": str(exc)})
            with STATE_LOCK:
                runtime = update_bot_runtime(
                    state,
                    last_error_at=utc_now().isoformat(),
                    last_error_message=str(exc),
                    pid=os.getpid(),
                )
                runtime["consecutive_poll_errors"] = int(runtime.get("consecutive_poll_errors", 0)) + 1
                save_bot_state(state)
            time.sleep(5)


def main() -> None:
    args = parse_args()
    if args.command == "print-help":
        print(build_help_text())
        return
    if args.command == "loop":
        run_loop()
        return


if __name__ == "__main__":
    main()
