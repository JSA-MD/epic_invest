#!/usr/bin/env python3
"""Operational watchdog for trader and Telegram bot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
MODELS_DIR = ROOT_DIR / "models"
STATE_PATH = MODELS_DIR / "rotation_target_050_live_state.json"
BOT_STATE_PATH = Path(os.getenv("TELEGRAM_BOT_STATE_PATH", "/tmp/epic-invest-telegram-bot-state.json"))
BOT_PID_PATH = Path(os.getenv("TELEGRAM_BOT_PID_FILE", "/tmp/epic-invest-telegram-bot.pid"))
TRADER_PID_PATH = Path(os.getenv("TRADER_PID_FILE", "/tmp/epic-invest-trader.pid"))
WATCHDOG_REPORT_PATH = MODELS_DIR / "operation_health_report.json"
WATCHDOG_HISTORY_PATH = MODELS_DIR / "operation_health_history.jsonl"
WATCHDOG_STATE_PATH = Path(os.getenv("WATCHDOG_STATE_PATH", "/tmp/epic-invest-watchdog-state.json"))
TRADER_LOG_PATH = Path(os.getenv("TRADER_LOG_FILE", "/tmp/epic-invest-trader.log"))
WATCHDOG_LOG_PATH = Path(os.getenv("WATCHDOG_LOG_FILE", "/tmp/epic-invest-watchdog.log"))

TRADER_LABEL = "com.epicinvest.trader"
BOT_LABEL = "com.epicinvest.telegram-bot"
WATCHDOG_LABEL = "com.epicinvest.watchdog"
TRADER_PLIST = ROOT_DIR / "scripts" / "com.epicinvest.trader.plist"
BOT_PLIST = ROOT_DIR / "scripts" / "com.epicinvest.telegram-bot.plist"
WATCHDOG_PLIST = ROOT_DIR / "scripts" / "com.epicinvest.watchdog.plist"
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"
TRADER_SCRIPT = ROOT_DIR / "scripts" / "rotation_target_050_live.py"

UTC = timezone.utc
DOMAIN = f"gui/{os.getuid()}"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_ALLOWED_CHAT_IDS = [
    item.strip()
    for raw in [os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""), os.getenv("TELEGRAM_CHAT_ID", "")]
    for item in raw.split(",")
    if item.strip()
]
WATCHDOG_INTERVAL_SECONDS = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "60"))
WATCHDOG_TRADER_STALE_SECONDS = int(os.getenv("WATCHDOG_TRADER_STALE_SECONDS", "180"))
WATCHDOG_BOT_STALE_SECONDS = int(os.getenv("WATCHDOG_BOT_STALE_SECONDS", "180"))
WATCHDOG_PROTECT_STALE_SECONDS = int(os.getenv("WATCHDOG_PROTECT_STALE_SECONDS", "300"))
WATCHDOG_ERROR_ESCALATION_COUNT = int(os.getenv("WATCHDOG_ERROR_ESCALATION_COUNT", "3"))
WATCHDOG_ALERT_COOLDOWN_SECONDS = int(os.getenv("WATCHDOG_ALERT_COOLDOWN_SECONDS", "300"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operational watchdog for Epic Invest.")
    sub = parser.add_subparsers(dest="command", required=True)

    check_once = sub.add_parser("check-once", help="Run one health check and optional recovery.")
    check_once.add_argument("--recover", action="store_true", help="Attempt automatic recovery when needed.")

    loop = sub.add_parser("loop", help="Run health checks continuously.")
    loop.add_argument("--interval-seconds", type=int, default=WATCHDOG_INTERVAL_SECONDS)
    loop.add_argument("--recover", action="store_true", help="Attempt automatic recovery when needed.")

    drill = sub.add_parser("drill", help="Intentionally stop components and verify recovery.")
    drill.add_argument("--target", choices=["trader", "bot", "both"], default="trader")
    drill.add_argument("--timeout-seconds", type=int, default=180)
    drill.add_argument("--recover", action="store_true", help="Use automatic recovery during the drill.")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def file_age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None
    return max(0.0, (utc_now() - modified).total_seconds())


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
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def resolve_live_pid(*candidates: int | None) -> int | None:
    ordered: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
    for candidate in ordered:
        if is_pid_running(candidate):
            return candidate
    return ordered[0] if ordered else None


def run_command(command: list[str], timeout: int = 120) -> dict[str, Any]:
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


def launchctl_print(label: str) -> dict[str, Any]:
    return run_command(["launchctl", "print", f"{DOMAIN}/{label}"], timeout=30)


def kickstart_launchd(label: str, plist_path: Path) -> dict[str, Any]:
    print_result = launchctl_print(label)
    actions: list[dict[str, Any]] = []
    if print_result["returncode"] != 0:
        actions.append(run_command(["launchctl", "bootstrap", DOMAIN, str(plist_path)], timeout=30))
    actions.append(run_command(["launchctl", "enable", f"{DOMAIN}/{label}"], timeout=30))
    actions.append(run_command(["launchctl", "kickstart", "-k", f"{DOMAIN}/{label}"], timeout=30))
    return {
        "label": label,
        "actions": actions,
        "ok": all(action["returncode"] == 0 for action in actions),
    }


def send_telegram_notification(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ALLOWED_CHAT_IDS:
        return False
    delivered = False
    for chat_id in TELEGRAM_ALLOWED_CHAT_IDS:
        params = urlencode(
            {
                "chat_id": str(chat_id),
                "text": text,
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
            delivered = delivered or bool(data.get("ok"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
    return delivered


def maybe_send_alert(report: dict[str, Any]) -> None:
    issues = []
    if report["trader"]["status"] != "ok":
        issues.append(
            f"트레이더 {report['trader']['status']} | stale={report['trader'].get('stale_seconds')}초 | errors={report['trader'].get('consecutive_errors')}"
        )
    if report["bot"]["status"] != "ok":
        issues.append(
            f"텔레그램 봇 {report['bot']['status']} | stale={report['bot'].get('stale_seconds')}초 | poll_errors={report['bot'].get('consecutive_poll_errors')}"
        )
    if not issues and not report.get("recovery_actions"):
        return

    message_lines = [
        "운영 감시 알림",
        f"- 시각: {report['generated_at']}",
        *[f"- {issue}" for issue in issues],
    ]
    if report.get("recovery_actions"):
        message_lines.append(f"- 자동 조치 수: {len(report['recovery_actions'])}")
    message = "\n".join(message_lines)
    fingerprint = hashlib.sha1(message.encode("utf-8")).hexdigest()

    state = read_json(WATCHDOG_STATE_PATH, {"last_alert_at": None, "last_alert_fingerprint": None})
    last_alert_at = age_seconds(state.get("last_alert_at"))
    if state.get("last_alert_fingerprint") == fingerprint and last_alert_at is not None and last_alert_at < WATCHDOG_ALERT_COOLDOWN_SECONDS:
        return

    if send_telegram_notification(message):
        write_json(
            WATCHDOG_STATE_PATH,
            {
                "last_alert_at": utc_now().isoformat(),
                "last_alert_fingerprint": fingerprint,
            },
        )


def evaluate_trader() -> dict[str, Any]:
    state = read_json(STATE_PATH, {})
    runtime = state.get("runtime_health") or {}
    runtime_pid = int(runtime.get("pid")) if str(runtime.get("pid") or "").isdigit() else None
    pid = resolve_live_pid(runtime_pid, read_pid(TRADER_PID_PATH))
    pid_verified = is_pid_running(pid)
    last_progress_at = latest_signal(
        [
            runtime.get("last_success_at"),
            runtime.get("last_loop_started_at"),
            runtime.get("last_loop_completed_at"),
            state.get("updated_at"),
        ]
    )
    stale_seconds = age_seconds(last_progress_at)
    consecutive_errors = int(runtime.get("consecutive_errors", 0) or 0)
    last_error_message = runtime.get("last_error_message")
    reasons: list[str] = []

    if stale_seconds is None:
        reasons.append("no_recent_success_signal")
    elif stale_seconds > WATCHDOG_TRADER_STALE_SECONDS:
        reasons.append("state_stale")
    if consecutive_errors >= WATCHDOG_ERROR_ESCALATION_COUNT:
        reasons.append("consecutive_errors")
    if not pid_verified:
        reasons.append("pid_unverified")

    status = "ok"
    if "state_stale" in reasons or "consecutive_errors" in reasons or "no_recent_success_signal" in reasons:
        status = "critical"

    return {
        "status": status,
        "pid": pid,
        "running": status == "ok" or status == "warning",
        "pid_verified": pid_verified,
        "state_updated_at": state.get("updated_at"),
        "last_success_at": runtime.get("last_success_at"),
        "last_loop_started_at": runtime.get("last_loop_started_at"),
        "last_loop_completed_at": runtime.get("last_loop_completed_at"),
        "stale_seconds": None if stale_seconds is None else round(stale_seconds, 1),
        "consecutive_errors": consecutive_errors,
        "last_error_at": runtime.get("last_error_at"),
        "last_error_message": last_error_message,
        "log_age_seconds": file_age_seconds(TRADER_LOG_PATH),
        "reasons": reasons,
    }


def evaluate_bot() -> dict[str, Any]:
    state = read_json(BOT_STATE_PATH, {})
    runtime = state.get("runtime") or {}
    runtime_pid = int(runtime.get("pid")) if str(runtime.get("pid") or "").isdigit() else None
    pid = resolve_live_pid(runtime_pid, read_pid(BOT_PID_PATH))
    pid_verified = is_pid_running(pid)
    last_progress_at = latest_signal(
        [
            runtime.get("last_poll_ok_at"),
            runtime.get("last_poll_started_at"),
            runtime.get("last_started_at"),
            runtime.get("last_reply_at"),
        ]
    )
    stale_seconds = age_seconds(last_progress_at)
    if stale_seconds is None:
        stale_seconds = file_age_seconds(BOT_STATE_PATH)
    consecutive_poll_errors = int(runtime.get("consecutive_poll_errors", 0) or 0)
    reasons: list[str] = []

    if stale_seconds is None:
        reasons.append("no_poll_signal")
    elif stale_seconds > WATCHDOG_BOT_STALE_SECONDS:
        reasons.append("bot_stale")
    if consecutive_poll_errors >= WATCHDOG_ERROR_ESCALATION_COUNT:
        reasons.append("poll_errors")
    if not pid_verified:
        reasons.append("pid_unverified")

    status = "ok"
    if "bot_stale" in reasons or "poll_errors" in reasons or "no_poll_signal" in reasons:
        status = "critical"

    return {
        "status": status,
        "pid": pid,
        "running": status == "ok" or status == "warning",
        "pid_verified": pid_verified,
        "last_started_at": runtime.get("last_started_at"),
        "last_poll_ok_at": runtime.get("last_poll_ok_at"),
        "last_reply_at": runtime.get("last_reply_at"),
        "stale_seconds": None if stale_seconds is None else round(stale_seconds, 1),
        "consecutive_poll_errors": consecutive_poll_errors,
        "last_error_at": runtime.get("last_error_at"),
        "last_error_message": runtime.get("last_error_message"),
        "reasons": reasons,
    }


def build_report() -> dict[str, Any]:
    return {
        "generated_at": utc_now().isoformat(),
        "trader": evaluate_trader(),
        "bot": evaluate_bot(),
        "recovery_actions": [],
    }


def protect_positions() -> dict[str, Any]:
    return run_command([str(PYTHON_BIN), str(TRADER_SCRIPT), "shutdown-protect", "--execute"], timeout=180)


def maybe_recover(report: dict[str, Any]) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    trader = report["trader"]
    bot = report["bot"]

    if trader["status"] == "critical":
        if trader.get("stale_seconds") is not None and float(trader["stale_seconds"]) >= WATCHDOG_PROTECT_STALE_SECONDS:
            actions.append({"type": "protect_positions", "result": protect_positions()})
        elif int(trader.get("consecutive_errors", 0)) >= WATCHDOG_ERROR_ESCALATION_COUNT:
            actions.append({"type": "protect_positions", "result": protect_positions()})
        actions.append({"type": "restart_trader", "result": kickstart_launchd(TRADER_LABEL, TRADER_PLIST)})

    if bot["status"] == "critical":
        actions.append({"type": "restart_bot", "result": kickstart_launchd(BOT_LABEL, BOT_PLIST)})

    report["recovery_actions"] = actions
    return report


def write_report(report: dict[str, Any]) -> None:
    write_json(WATCHDOG_REPORT_PATH, report)
    append_jsonl(
        WATCHDOG_HISTORY_PATH,
        {
            "generated_at": report["generated_at"],
            "trader": {
                "status": report["trader"]["status"],
                "stale_seconds": report["trader"].get("stale_seconds"),
                "consecutive_errors": report["trader"].get("consecutive_errors"),
            },
            "bot": {
                "status": report["bot"]["status"],
                "stale_seconds": report["bot"].get("stale_seconds"),
                "consecutive_poll_errors": report["bot"].get("consecutive_poll_errors"),
            },
            "recovery_count": len(report.get("recovery_actions", [])),
        },
    )


def check_once(recover: bool) -> dict[str, Any]:
    report = build_report()
    if recover:
        report = maybe_recover(report)
    write_report(report)
    maybe_send_alert(report)
    print(json.dumps(report, indent=2))
    return report


def kill_target_processes(target: str) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    trader_state = read_json(STATE_PATH, {})
    trader_runtime = trader_state.get("runtime_health") or {}
    trader_runtime_pid = int(trader_runtime.get("pid")) if str(trader_runtime.get("pid") or "").isdigit() else None
    trader_pid = resolve_live_pid(trader_runtime_pid, read_pid(TRADER_PID_PATH))
    bot_state = read_json(BOT_STATE_PATH, {})
    bot_runtime = (bot_state.get("runtime") or {})
    bot_runtime_pid = int(bot_runtime.get("pid")) if str(bot_runtime.get("pid") or "").isdigit() else None
    bot_pid = resolve_live_pid(bot_runtime_pid, read_pid(BOT_PID_PATH))

    targets: list[tuple[str, int | None]] = []
    if target in {"trader", "both"}:
        targets.append(("trader", trader_pid))
    if target in {"bot", "both"}:
        targets.append(("bot", bot_pid))

    for name, pid in targets:
        if not is_pid_running(pid):
            actions.append({"type": f"kill_{name}", "result": "pid_not_running"})
            continue
        try:
            os.kill(int(pid), signal.SIGTERM)
            actions.append({"type": f"kill_{name}", "result": f"sent_sigterm:{pid}"})
        except OSError as exc:
            actions.append({"type": f"kill_{name}", "result": f"error:{exc}"})
    return actions


def drill(target: str, timeout_seconds: int, recover: bool) -> None:
    started_at = utc_now()
    actions = kill_target_processes(target)
    deadline = time.time() + max(10, timeout_seconds)
    recovered_report: dict[str, Any] | None = None

    while time.time() < deadline:
        report = check_once(recover=recover)
        trader_ok = target not in {"trader", "both"} or report["trader"]["status"] == "ok"
        bot_ok = target not in {"bot", "both"} or report["bot"]["status"] == "ok"
        if trader_ok and bot_ok:
            recovered_report = report
            break
        time.sleep(5)

    payload = {
        "started_at": started_at.isoformat(),
        "target": target,
        "initial_actions": actions,
        "recovered": recovered_report is not None,
        "final_report": recovered_report,
    }
    print(json.dumps(payload, indent=2))
    if recovered_report is None:
        raise SystemExit(1)


def run_loop(interval_seconds: int, recover: bool) -> None:
    while True:
        try:
            check_once(recover=recover)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            payload = {
                "generated_at": utc_now().isoformat(),
                "error": str(exc),
            }
            append_jsonl(WATCHDOG_HISTORY_PATH, payload)
            print(json.dumps(payload, indent=2))
        time.sleep(max(10, interval_seconds))


def main() -> None:
    args = parse_args()
    if args.command == "check-once":
        check_once(recover=bool(args.recover))
        return
    if args.command == "loop":
        run_loop(interval_seconds=int(args.interval_seconds), recover=bool(args.recover))
        return
    if args.command == "drill":
        drill(args.target, int(args.timeout_seconds), recover=bool(args.recover))
        return


if __name__ == "__main__":
    main()
