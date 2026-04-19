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
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from data_quality_monitor import build_data_quality_snapshot

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
MODELS_DIR = ROOT_DIR / "models"
RUNTIME_PROFILE_PATH = MODELS_DIR / "active_runtime_profile.json"
CORE_STATE_PATH = MODELS_DIR / "rotation_target_050_live_state.json"
PAIRWISE_STATE_PATH = Path(os.getenv("PAIRWISE_LIVE_STATE_PATH", str(MODELS_DIR / "pairwise_regime_live_state.json")))
PAIRWISE_SHADOW_STATE_PATH = Path(os.getenv("PAIRWISE_SHADOW_STATE_PATH", str(MODELS_DIR / "pairwise_regime_shadow_state.json")))
PAIRWISE_DECISION_LOG_PATH = Path(
    os.getenv("PAIRWISE_LIVE_DECISION_LOG_PATH", str(ROOT_DIR / "logs" / "pairwise_regime_decisions.jsonl"))
)
BOT_STATE_PATH = Path(os.getenv("TELEGRAM_BOT_STATE_PATH", "/tmp/epic-invest-telegram-bot-state.json"))
BOT_PID_PATH = Path(os.getenv("TELEGRAM_BOT_PID_FILE", "/tmp/epic-invest-telegram-bot.pid"))
CORE_PID_PATH = Path(os.getenv("TRADER_PID_FILE", "/tmp/epic-invest-trader.pid"))
PAIRWISE_PID_PATH = Path(os.getenv("PAIRWISE_LIVE_PID_FILE", "/tmp/epic_pairwise_live.pid"))
WATCHDOG_REPORT_PATH = MODELS_DIR / "operation_health_report.json"
WATCHDOG_HISTORY_PATH = MODELS_DIR / "operation_health_history.jsonl"
DATA_QUALITY_REPORT_PATH = MODELS_DIR / "data_quality_report.json"
DECISION_QUALITY_REPORT_PATH = MODELS_DIR / "decision_quality_report.json"
WATCHDOG_STATE_PATH = Path(os.getenv("WATCHDOG_STATE_PATH", "/tmp/epic-invest-watchdog-state.json"))
CORE_LOG_PATH = Path(os.getenv("TRADER_LOG_FILE", "/tmp/epic-invest-trader.log"))
PAIRWISE_LOG_PATH = Path(os.getenv("PAIRWISE_LIVE_LOG_FILE", str(ROOT_DIR / "logs" / "pairwise_live_service.log")))
WATCHDOG_LOG_PATH = Path(os.getenv("WATCHDOG_LOG_FILE", "/tmp/epic-invest-watchdog.log"))

TRADER_LABEL = "com.epicinvest.trader"
PAIRWISE_LABEL = "com.epicinvest.pairwise-trader"
BOT_LABEL = "com.epicinvest.telegram-bot"
WATCHDOG_LABEL = "com.epicinvest.watchdog"
TRADER_PLIST = ROOT_DIR / "scripts" / "com.epicinvest.trader.plist"
PAIRWISE_PLIST = ROOT_DIR / "scripts" / "com.epicinvest.pairwise-trader.plist"
BOT_PLIST = ROOT_DIR / "scripts" / "com.epicinvest.telegram-bot.plist"
WATCHDOG_PLIST = ROOT_DIR / "scripts" / "com.epicinvest.watchdog.plist"
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"
CORE_TRADER_SCRIPT = ROOT_DIR / "scripts" / "rotation_target_050_live.py"
PAIRWISE_TRADER_SCRIPT = ROOT_DIR / "scripts" / "pairwise_regime_live.py"
PAIRWISE_SERVICE_SCRIPT = ROOT_DIR / "scripts" / "pairwise_live_service.sh"
PAIRWISE_SHADOW_LOAD_SCRIPT = ROOT_DIR / "pairwise_shadow_launchd_load.sh"
PAIRWISE_SHADOW_UNLOAD_SCRIPT = ROOT_DIR / "pairwise_shadow_launchd_unload.sh"

UTC = timezone.utc
DOMAIN = f"gui/{os.getuid()}"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def resolve_telegram_chat_ids() -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for raw in [os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""), os.getenv("TELEGRAM_CHAT_ID", "")]:
        for item in raw.split(","):
            item = item.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            values.append(item)
    return values


TELEGRAM_ALLOWED_CHAT_IDS = resolve_telegram_chat_ids()
WATCHDOG_INTERVAL_SECONDS = int(os.getenv("WATCHDOG_INTERVAL_SECONDS", "60"))
WATCHDOG_TRADER_STALE_SECONDS = int(os.getenv("WATCHDOG_TRADER_STALE_SECONDS", "180"))
WATCHDOG_BOT_STALE_SECONDS = int(os.getenv("WATCHDOG_BOT_STALE_SECONDS", "180"))
WATCHDOG_PROTECT_STALE_SECONDS = int(os.getenv("WATCHDOG_PROTECT_STALE_SECONDS", "300"))
WATCHDOG_ERROR_ESCALATION_COUNT = int(os.getenv("WATCHDOG_ERROR_ESCALATION_COUNT", "3"))
WATCHDOG_ALERT_COOLDOWN_SECONDS = int(os.getenv("WATCHDOG_ALERT_COOLDOWN_SECONDS", "300"))
WATCHDOG_TRADER_GRACE_SECONDS = int(os.getenv("WATCHDOG_TRADER_GRACE_SECONDS", "90"))
PAIRWISE_POLL_SECONDS = int(os.getenv("PAIRWISE_LIVE_POLL_SECONDS", "300"))
WATCHDOG_PAIRWISE_SIGNAL_STALE_SECONDS = int(os.getenv("WATCHDOG_PAIRWISE_SIGNAL_STALE_SECONDS", str(20 * 60)))
WATCHDOG_PAIRWISE_SIGNAL_DRIFT_SECONDS = int(
    os.getenv("WATCHDOG_PAIRWISE_SIGNAL_DRIFT_SECONDS", str(PAIRWISE_POLL_SECONDS + WATCHDOG_TRADER_GRACE_SECONDS))
)
WATCHDOG_WEIGHT_DRIFT_THRESHOLD = float(os.getenv("WATCHDOG_WEIGHT_DRIFT_THRESHOLD", "0.25"))
WATCHDOG_TELEGRAM_ALERTS_ENABLED = os.getenv("WATCHDOG_TELEGRAM_ALERTS_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


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


def run_command(command: list[str], timeout: int = 120, env_updates: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    completed = subprocess.run(
        command,
        cwd=str(ROOT_DIR),
        text=True,
        capture_output=True,
        timeout=timeout,
        env=env,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def launchctl_print(label: str) -> dict[str, Any]:
    return run_command(["launchctl", "print", f"{DOMAIN}/{label}"], timeout=30)


def launchd_service_pid(label: str | None) -> int | None:
    if not label:
        return None
    result = launchctl_print(str(label))
    if result["returncode"] != 0:
        return None
    for line in str(result.get("stdout") or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("pid = "):
            continue
        raw_pid = stripped.split("=", 1)[1].strip()
        if raw_pid.isdigit():
            return int(raw_pid)
    return None


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


def read_runtime_profile() -> dict[str, Any]:
    payload = read_json(RUNTIME_PROFILE_PATH, {})
    return payload if isinstance(payload, dict) else {}


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def write_runtime_profile(active_trader: str, mode: str, force_execute: bool, **extra: Any) -> dict[str, Any]:
    payload = {
        "active_trader": str(active_trader),
        "mode": str(mode),
        "force_execute": bool(force_execute),
        "updated_at": utc_now().isoformat(),
    }
    payload.update(extra)
    write_json(RUNTIME_PROFILE_PATH, payload)
    return payload


def resolve_active_trader_key() -> str:
    profile = read_runtime_profile()
    key = str(profile.get("active_trader") or "").strip().lower()
    if key in {"core", "pairwise"}:
        return key
    pairwise_pid = read_pid(PAIRWISE_PID_PATH)
    core_pid = read_pid(CORE_PID_PATH)
    pairwise_launchd_pid = launchd_service_pid(PAIRWISE_LABEL)
    core_launchd_pid = launchd_service_pid(TRADER_LABEL)
    if is_pid_running(resolve_live_pid(pairwise_pid, pairwise_launchd_pid)) and not is_pid_running(
        resolve_live_pid(core_pid, core_launchd_pid)
    ):
        return "pairwise"
    return "core"


def active_trader_profile() -> dict[str, Any]:
    runtime_profile = read_runtime_profile()
    key = resolve_active_trader_key()
    if key == "pairwise":
        runtime_force_requested = truthy(runtime_profile.get("force_execute"))
        force_execute = runtime_force_requested and truthy(os.getenv("PAIRWISE_FORCE_EXECUTE"))
        stale_threshold = max(WATCHDOG_TRADER_STALE_SECONDS, PAIRWISE_POLL_SECONDS + WATCHDOG_TRADER_GRACE_SECONDS)
        protect_threshold = max(WATCHDOG_PROTECT_STALE_SECONDS, stale_threshold + WATCHDOG_TRADER_GRACE_SECONDS)
        return {
            "key": "pairwise",
            "state_path": PAIRWISE_STATE_PATH,
            "pid_path": PAIRWISE_PID_PATH,
            "launchd_label": PAIRWISE_LABEL,
            "launchd_plist": PAIRWISE_PLIST,
            "log_path": PAIRWISE_LOG_PATH,
            "decision_log_path": PAIRWISE_DECISION_LOG_PATH,
            "mode": str(runtime_profile.get("mode") or os.getenv("PAIRWISE_LIVE_MODE", os.getenv("BINANCE_MODE", "demo"))),
            "force_execute": force_execute,
            "runtime_force_execute_requested": runtime_force_requested,
            "stale_threshold_seconds": stale_threshold,
            "protect_threshold_seconds": protect_threshold,
        }
    return {
        "key": "core",
        "state_path": CORE_STATE_PATH,
        "pid_path": CORE_PID_PATH,
        "log_path": CORE_LOG_PATH,
        "mode": str(runtime_profile.get("mode") or os.getenv("BINANCE_MODE", "demo")),
        "force_execute": False,
        "stale_threshold_seconds": WATCHDOG_TRADER_STALE_SECONDS,
        "protect_threshold_seconds": WATCHDOG_PROTECT_STALE_SECONDS,
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


def build_alert_fingerprint(report: dict[str, Any]) -> str:
    payload = {
        "active_trader": report.get("trader", {}).get("active_profile"),
        "trader_status": report.get("trader", {}).get("status"),
        "trader_reasons": report.get("trader", {}).get("reasons") or [],
        "bot_status": report.get("bot", {}).get("status"),
        "bot_reasons": report.get("bot", {}).get("reasons") or [],
        "recovery_types": [item.get("type") for item in report.get("recovery_actions", [])],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def weight_map(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get("target_weights", payload)
    if not isinstance(raw, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, value in raw.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def requested_weight_map(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    pair_plans = normalize_pair_plans(payload.get("pair_plans"))
    if not pair_plans:
        return {}
    result: dict[str, float] = {}
    for pair, pair_plan in pair_plans.items():
        try:
            result[str(pair)] = float(pair_plan.get("requested_weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return result


def signal_timestamp_from_snapshot(snapshot: Any) -> str | None:
    if not isinstance(snapshot, Mapping):
        return None
    rationale = snapshot.get("rationale")
    if isinstance(rationale, Mapping) and rationale.get("signal_timestamp"):
        return str(rationale.get("signal_timestamp"))
    if snapshot.get("signal_timestamp"):
        return str(snapshot.get("signal_timestamp"))
    return None


def max_weight_drift(lhs: Mapping[str, float], rhs: Mapping[str, float]) -> float:
    keys = set(lhs) | set(rhs)
    if not keys:
        return 0.0
    return max(abs(float(lhs.get(key, 0.0)) - float(rhs.get(key, 0.0))) for key in keys)


def active_position_count(execution_snapshot: Mapping[str, Any] | None) -> int:
    if not isinstance(execution_snapshot, Mapping):
        return 0
    positions = execution_snapshot.get("positions")
    if not isinstance(positions, list):
        return 0
    count = 0
    for item in positions:
        if not isinstance(item, Mapping):
            continue
        try:
            qty = abs(float(item.get("qty", 0.0) or 0.0))
        except (TypeError, ValueError):
            qty = 0.0
        if qty > 0.0:
            count += 1
    return count


def protection_order_health(protection_snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    default = {"expected": 0, "active": 0, "status": None}
    if not isinstance(protection_snapshot, Mapping):
        return default
    positions = protection_snapshot.get("positions")
    expected = len(positions) * 2 if isinstance(positions, list) else 0
    active = 0
    for item in protection_snapshot.get("protections", []):
        if not isinstance(item, Mapping):
            continue
        for order in item.get("orders", []):
            if not isinstance(order, Mapping):
                continue
            if str(order.get("status") or "").lower() in {"placed", "retained"}:
                active += 1
    return {
        "expected": expected,
        "active": active,
        "status": protection_snapshot.get("status"),
    }


def normalize_pair_plans(payload: Any) -> dict[str, Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        return {str(key): value for key, value in payload.items() if isinstance(value, Mapping)}
    if isinstance(payload, list):
        result: dict[str, Mapping[str, Any]] = {}
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            pair = item.get("pair")
            if pair:
                result[str(pair)] = item
        return result
    return {}


def active_pairs_from_state(latest_runtime_snapshot: Mapping[str, Any], latest_decision_snapshot: Mapping[str, Any]) -> dict[str, float]:
    weights = weight_map(latest_decision_snapshot)
    if not weights:
        weights = weight_map((latest_runtime_snapshot.get("plan") or {}).get("target_weights") if isinstance(latest_runtime_snapshot, Mapping) else {})
    active: dict[str, float] = {pair: weight for pair, weight in weights.items() if abs(weight) > 1e-9}

    execution = ((latest_runtime_snapshot.get("extra") or {}).get("execution") or {}) if isinstance(latest_runtime_snapshot, Mapping) else {}
    protection = execution.get("shutdown_protection") if isinstance(execution, Mapping) else {}
    positions = protection.get("positions") if isinstance(protection, Mapping) else None
    if isinstance(positions, list):
        for item in positions:
            if not isinstance(item, Mapping):
                continue
            pair = item.get("pair")
            if not pair:
                continue
            try:
                qty = float(item.get("qty", 0.0) or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if abs(qty) > 0.0:
                active.setdefault(str(pair), qty)
    return active


def summarize_pair_feed_status(group_snapshot: Mapping[str, Any] | None, pair: str) -> dict[str, Any]:
    feeds = ((group_snapshot or {}).get("per_pair") or {}).get(pair) if isinstance(group_snapshot, Mapping) else None
    if not isinstance(feeds, Mapping):
        return {"status": "missing", "feeds": []}
    feed_statuses: list[str] = []
    feed_names: list[str] = []
    for name, payload in feeds.items():
        if not isinstance(payload, Mapping):
            continue
        freshness = str(payload.get("freshness") or "missing")
        feed_statuses.append(freshness)
        if freshness != "fresh":
            feed_names.append(str(name))
    return {
        "status": rollup_feed_status(feed_statuses),
        "feeds": feed_names,
    }


def rollup_feed_status(values: list[str]) -> str:
    if not values:
        return "missing"
    if any(value in {"critical", "stale"} for value in values):
        return "critical"
    if any(value in {"warning", "aging", "missing"} for value in values):
        return "warning"
    return "ok"


def build_decision_quality_snapshot(trader_report: Mapping[str, Any], data_quality_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    profile = active_trader_profile()
    state = read_json(profile["state_path"], {})
    latest_runtime_snapshot = state.get("latest_runtime_snapshot") or {}
    latest_decision_snapshot = state.get("latest_decision_snapshot") or {}
    plan = (latest_runtime_snapshot.get("plan") or {}) if isinstance(latest_runtime_snapshot, Mapping) else {}
    pair_plans = normalize_pair_plans(plan.get("pair_plans") or latest_decision_snapshot.get("pair_plans"))
    active_pairs = active_pairs_from_state(latest_runtime_snapshot, latest_decision_snapshot)

    runtime_risks: list[dict[str, Any]] = []
    upgrade_risks: list[dict[str, Any]] = []
    research_risks: list[dict[str, Any]] = []
    recommendations: list[str] = []

    market_context_status = str((data_quality_snapshot.get("market_context") or {}).get("status") or "missing")
    if active_pairs and market_context_status != "ok":
        impacted = []
        for pair, pair_plan in pair_plans.items():
            if pair not in active_pairs:
                continue
            if str(pair_plan.get("equity_corr_source_mode") or "") == "market_context":
                impacted.append(pair)
        if impacted:
            runtime_risks.append(
                {
                    "scope": "runtime_regime_gating",
                    "severity": "warning" if market_context_status == "warning" else "critical",
                    "pairs": impacted,
                    "reason": "상관/레짐 라우팅이 market context에 의존하는데 입력 캐시가 최신이 아닙니다.",
                }
            )
            recommendations.append("QQQ/SPY/GLD/DXY 문맥 캐시를 자동 갱신해 corr-state 라우팅 품질 저하를 막아야 합니다.")

    for pair in sorted(active_pairs):
        lob_status = summarize_pair_feed_status(data_quality_snapshot.get("lob"), pair)
        if lob_status["status"] != "ok":
            upgrade_risks.append(
                {
                    "scope": "entry_quality_upgrade_loop",
                    "severity": "warning" if lob_status["status"] == "warning" else "critical",
                    "pair": pair,
                    "stale_feeds": lob_status["feeds"],
                    "reason": "LOB 미시구조가 stale이라 spread/depth 기반 진입 품질 개선 루프가 멈춰 있습니다.",
                }
            )
            recommendations.append(f"{pair} LOB 수집을 상시화해 spread/depth imbalance 기반 진입 필터 학습을 복구해야 합니다.")

        derivative_status = summarize_pair_feed_status(data_quality_snapshot.get("derivatives"), pair)
        if derivative_status["status"] != "ok":
            research_risks.append(
                {
                    "scope": "positioning_research",
                    "severity": "warning" if derivative_status["status"] == "warning" else "critical",
                    "pair": pair,
                    "stale_feeds": derivative_status["feeds"],
                    "reason": "파생 포지셔닝 캐시가 stale이라 OI/basis/long-short 기반 보강 연구 신뢰도가 떨어집니다.",
                }
            )

        ohlcv_snapshot = ((data_quality_snapshot.get("ohlcv") or {}).get("per_pair") or {}).get(pair, {})
        daily_freshness = str(((ohlcv_snapshot.get("1d") or {}).get("freshness")) or "missing") if isinstance(ohlcv_snapshot, Mapping) else "missing"
        if daily_freshness != "fresh":
            research_risks.append(
                {
                    "scope": "validation_research",
                    "severity": "warning",
                    "pair": pair,
                    "reason": "일봉 캐시가 누락 또는 노후화돼 장기 검증/교차자산 해석 신뢰도가 낮습니다.",
                }
            )

    status = "ok"
    if any(item["severity"] == "critical" for item in runtime_risks):
        status = "critical"
    elif runtime_risks or any(item["severity"] == "critical" for item in upgrade_risks):
        status = "warning"
    elif upgrade_risks or research_risks:
        status = "warning"

    insights: list[str] = []
    if active_pairs:
        pair_text = ", ".join(sorted(active_pairs))
        insights.append(f"현재 활성 노출은 {pair_text}이며, stale 데이터가 있는 경우 해당 자산의 진입 품질 개선 속도가 즉시 떨어집니다.")
    if any(item.get("pair") == "BNBUSDT" for item in upgrade_risks):
        insights.append("BNBUSDT는 현재 실제 포지션이 있고, LOB stale이면 BNB 약세 구간 전용 진입 필터를 더 이상 개선할 수 없습니다.")
    if runtime_risks:
        insights.append("현재 전략은 market context 기반 corr-state 라우팅을 쓰고 있어, 문맥 캐시 노후화가 직접적인 판단 품질 저하로 이어질 수 있습니다.")

    return {
        "generated_at": utc_now().isoformat(),
        "status": status,
        "active_pairs": sorted(active_pairs),
        "runtime_risks": runtime_risks,
        "upgrade_risks": upgrade_risks,
        "research_risks": research_risks,
        "insights": insights,
        "recommendations": recommendations,
        "trader_status": trader_report.get("status"),
    }


def safe_build_data_quality_snapshot() -> dict[str, Any]:
    try:
        return build_data_quality_snapshot()
    except Exception as exc:
        return {
            "generated_at": utc_now().isoformat(),
            "status": "critical",
            "error": f"data_quality_snapshot_failed: {exc}",
            "recommendations": ["데이터 품질 스냅샷 생성이 실패했습니다. 모니터링 경로를 즉시 점검해야 합니다."],
        }


def safe_build_decision_quality_snapshot(trader_report: Mapping[str, Any], data_quality_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return build_decision_quality_snapshot(trader_report, data_quality_snapshot)
    except Exception as exc:
        return {
            "generated_at": utc_now().isoformat(),
            "status": "critical",
            "error": f"decision_quality_snapshot_failed: {exc}",
            "runtime_risks": [],
            "upgrade_risks": [],
            "research_risks": [],
            "insights": ["의사결정 품질 스냅샷 생성이 실패해 stale 데이터 영향 추적이 중단됐습니다."],
            "recommendations": ["decision_quality 보고 경로를 즉시 복구해야 합니다."],
        }


def maybe_send_alert(report: dict[str, Any]) -> None:
    if not WATCHDOG_TELEGRAM_ALERTS_ENABLED:
        return
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
        f"- 대상: {report['trader'].get('active_profile', 'core')}",
        *[f"- {issue}" for issue in issues],
    ]
    if report.get("recovery_actions"):
        message_lines.append(f"- 자동 조치 수: {len(report['recovery_actions'])}")
    message = "\n".join(message_lines)
    fingerprint = build_alert_fingerprint(report)

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
    profile = active_trader_profile()
    state = read_json(profile["state_path"], {})
    runtime = state.get("runtime_health") or {}
    latest_runtime_snapshot = state.get("latest_runtime_snapshot") or {}
    latest_decision_snapshot = state.get("latest_decision_snapshot") or {}
    runtime_pid = int(runtime.get("pid")) if str(runtime.get("pid") or "").isdigit() else None
    pid = resolve_live_pid(
        runtime_pid,
        read_pid(profile["pid_path"]),
        launchd_service_pid(profile.get("launchd_label")),
    )
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
    shadow_last_progress_at = None
    shadow_stale_seconds = None
    shadow_signal_timestamp = None
    shadow_signal_stale_seconds = None
    shadow_signal_stale_threshold_seconds = None
    shadow_signal_drift_seconds = None
    decision_log_age_seconds = None
    decision_log_path = profile.get("decision_log_path")
    if decision_log_path is not None:
        decision_log_age_seconds = file_age_seconds(decision_log_path)
    live_target_weights: dict[str, float] = {}
    shadow_target_weights: dict[str, float] = {}
    live_requested_weights: dict[str, float] = {}
    shadow_requested_weights: dict[str, float] = {}
    live_shadow_weight_drift = 0.0
    live_shadow_requested_drift = 0.0
    execution_enabled = None
    execution_blocked = None
    protection_health = {"expected": 0, "active": 0, "status": None}
    shadow_state = {}

    shadow_state_path = profile.get("shadow_state_path")
    if profile["key"] == "pairwise" and shadow_state_path is not None:
        shadow_state = read_json(shadow_state_path, {})
        shadow_runtime = shadow_state.get("runtime_health") or {}
        shadow_paper = shadow_state.get("shadow_paper") or {}
        shadow_last_progress_at = latest_signal(
            [
                shadow_runtime.get("last_success_at"),
                shadow_runtime.get("last_loop_started_at"),
                shadow_runtime.get("last_loop_completed_at"),
                shadow_state.get("updated_at"),
            ]
        )
        shadow_stale_seconds = age_seconds(shadow_last_progress_at)
        shadow_signal_timestamp = shadow_paper.get("last_signal_timestamp")
        shadow_signal_stale_seconds = age_seconds(shadow_signal_timestamp)
        shadow_signal_stale_threshold_seconds = max(
            float(WATCHDOG_PAIRWISE_SIGNAL_STALE_SECONDS),
            float(PAIRWISE_POLL_SECONDS + WATCHDOG_TRADER_GRACE_SECONDS),
        )
        live_signal_timestamp = signal_timestamp_from_snapshot(latest_decision_snapshot)
        shadow_decision = shadow_state.get("latest_decision_snapshot") or {}
        shadow_signal_timestamp = signal_timestamp_from_snapshot(shadow_decision) or shadow_signal_timestamp
        if live_signal_timestamp and shadow_signal_timestamp:
            live_dt = parse_iso_datetime(live_signal_timestamp)
            shadow_dt = parse_iso_datetime(shadow_signal_timestamp)
            if live_dt is not None and shadow_dt is not None:
                shadow_signal_drift_seconds = abs((live_dt - shadow_dt).total_seconds())
        live_target_weights = weight_map(latest_decision_snapshot)
        shadow_target_weights = weight_map(shadow_state.get("latest_decision_snapshot") or shadow_paper.get("current_weights") or {})
        live_requested_weights = requested_weight_map(latest_decision_snapshot) or requested_weight_map(
            (latest_runtime_snapshot.get("plan") or {}) if isinstance(latest_runtime_snapshot, Mapping) else {}
        )
        shadow_requested_weights = requested_weight_map(shadow_state.get("latest_decision_snapshot") or {})
        live_shadow_weight_drift = max_weight_drift(live_target_weights, shadow_target_weights)
        live_shadow_requested_drift = max_weight_drift(live_requested_weights, shadow_requested_weights)

    if profile["key"] == "pairwise":
        execution = (latest_runtime_snapshot.get("extra") or {}).get("execution")
        execution_mode = "demo"
        if isinstance(execution, Mapping):
            execution_enabled = bool(execution.get("enabled"))
            execution_blocked = bool(execution.get("blocked"))
            execution_mode = str(execution.get("mode") or execution_mode).lower()
            protection_health = protection_order_health(execution.get("shutdown_protection"))
        plan_target_weights = weight_map((latest_runtime_snapshot.get("plan") or {}).get("target_weights") or {})
        if live_target_weights and plan_target_weights:
            if max_weight_drift(live_target_weights, plan_target_weights) > WATCHDOG_WEIGHT_DRIFT_THRESHOLD:
                reasons.append("plan_snapshot_mismatch")

    if stale_seconds is None:
        reasons.append("no_recent_success_signal")
    elif stale_seconds > float(profile["stale_threshold_seconds"]):
        reasons.append("state_stale")
    if profile["key"] == "pairwise":
        promotion_gate = state.get("promotion_gate") or {}
        demo_gate_ready = bool(
            promotion_gate.get(
                "ready_for_demo",
                promotion_gate.get(
                    "ready_for_shadow_live",
                    promotion_gate.get("ready_for_live", promotion_gate.get("ready_for_merge")),
                ),
            )
        )
        live_gate_ready = bool(promotion_gate.get("ready_for_live", promotion_gate.get("ready_for_merge")))
        gate_ready = demo_gate_ready if execution_mode == "demo" else live_gate_ready
        if execution_blocked and gate_ready:
            reasons.append("execution_blocked_with_open_gate")
        if execution_enabled and active_position_count(latest_runtime_snapshot) > 0:
            if protection_health["expected"] > 0 and protection_health["active"] < protection_health["expected"]:
                reasons.append("shutdown_protection_missing")
    if profile["key"] == "pairwise" and shadow_state_path is not None:
        promotion_gate = state.get("promotion_gate") or {}
        shadow_required = bool(promotion_gate.get("shadow_required", True))
        if shadow_stale_seconds is None:
            if shadow_required:
                reasons.append("shadow_state_missing")
        elif shadow_stale_seconds > float(profile["stale_threshold_seconds"]):
            if shadow_required:
                reasons.append("shadow_state_stale")
        if shadow_signal_stale_seconds is None:
            if shadow_required:
                reasons.append("shadow_signal_missing")
        elif shadow_signal_stale_seconds > float(shadow_signal_stale_threshold_seconds or profile["stale_threshold_seconds"]):
            if shadow_required:
                reasons.append("shadow_signal_stale")
        if (
            shadow_required
            and shadow_signal_drift_seconds is not None
            and shadow_signal_drift_seconds > float(WATCHDOG_PAIRWISE_SIGNAL_DRIFT_SECONDS)
        ):
            reasons.append("live_shadow_signal_divergence")
        elif (
            shadow_required
            and
            live_shadow_weight_drift > WATCHDOG_WEIGHT_DRIFT_THRESHOLD
            and live_shadow_requested_drift > WATCHDOG_WEIGHT_DRIFT_THRESHOLD
        ):
            reasons.append("live_shadow_target_divergence")
    if consecutive_errors >= WATCHDOG_ERROR_ESCALATION_COUNT:
        reasons.append("consecutive_errors")
    if not pid_verified:
        reasons.append("pid_unverified")
    if profile["key"] == "pairwise":
        if decision_log_age_seconds is None:
            reasons.append("decision_log_missing")
        elif decision_log_age_seconds > float(profile["protect_threshold_seconds"]):
            reasons.append("decision_log_stale")

    status = "ok"
    critical_reasons = {
        "state_stale",
        "consecutive_errors",
        "no_recent_success_signal",
        "shadow_state_stale",
        "shadow_state_missing",
        "shadow_signal_stale",
        "shadow_signal_missing",
        "live_shadow_signal_divergence",
        "live_shadow_target_divergence",
        "execution_blocked_with_open_gate",
        "shutdown_protection_missing",
        "plan_snapshot_mismatch",
    }
    warning_reasons = {
        "pid_unverified",
        "decision_log_missing",
        "decision_log_stale",
    }
    if any(reason in critical_reasons for reason in reasons):
        status = "critical"
    elif any(reason in warning_reasons for reason in reasons):
        status = "warning"

    return {
        "status": status,
        "active_profile": profile["key"],
        "force_execute": bool(profile.get("force_execute", False)),
        "pid": pid,
        "running": status == "ok" or status == "warning",
        "pid_verified": pid_verified,
        "state_updated_at": state.get("updated_at"),
        "last_success_at": runtime.get("last_success_at"),
        "last_loop_started_at": runtime.get("last_loop_started_at"),
        "last_loop_completed_at": runtime.get("last_loop_completed_at"),
        "stale_seconds": None if stale_seconds is None else round(stale_seconds, 1),
        "stale_threshold_seconds": profile["stale_threshold_seconds"],
        "protect_threshold_seconds": profile["protect_threshold_seconds"],
        "consecutive_errors": consecutive_errors,
        "last_error_at": runtime.get("last_error_at"),
        "last_error_message": last_error_message,
        "shadow_state_updated_at": shadow_last_progress_at,
        "shadow_stale_seconds": None if shadow_stale_seconds is None else round(shadow_stale_seconds, 1),
        "shadow_signal_timestamp": shadow_signal_timestamp,
        "shadow_signal_stale_seconds": None if shadow_signal_stale_seconds is None else round(shadow_signal_stale_seconds, 1),
        "shadow_signal_stale_threshold_seconds": shadow_signal_stale_threshold_seconds,
        "shadow_signal_drift_seconds": None if shadow_signal_drift_seconds is None else round(shadow_signal_drift_seconds, 1),
        "live_shadow_weight_drift": round(live_shadow_weight_drift, 4),
        "live_shadow_requested_drift": round(live_shadow_requested_drift, 4),
        "execution_enabled": execution_enabled,
        "execution_blocked": execution_blocked,
        "shutdown_protection_expected": protection_health["expected"],
        "shutdown_protection_active": protection_health["active"],
        "shutdown_protection_status": protection_health["status"],
        "log_age_seconds": None if decision_log_age_seconds is None else round(decision_log_age_seconds, 1),
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
    trader = evaluate_trader()
    bot = evaluate_bot()
    data_quality = safe_build_data_quality_snapshot()
    decision_quality = safe_build_decision_quality_snapshot(trader, data_quality)
    return {
        "generated_at": utc_now().isoformat(),
        "trader": trader,
        "bot": bot,
        "data_quality": data_quality,
        "decision_quality": decision_quality,
        "recovery_actions": [],
    }


def protect_positions(profile: dict[str, Any]) -> dict[str, Any]:
    if profile["key"] == "pairwise":
        return run_command(
            [
                str(PYTHON_BIN),
                str(PAIRWISE_TRADER_SCRIPT),
                "shutdown-protect",
                "--execute",
                "--mode",
                str(profile["mode"]),
                "--state-path",
                str(profile["state_path"]),
            ],
            timeout=180,
        )
    return run_command([str(PYTHON_BIN), str(CORE_TRADER_SCRIPT), "shutdown-protect", "--execute"], timeout=180)


def restart_active_trader(profile: dict[str, Any]) -> dict[str, Any]:
    if profile["key"] == "pairwise":
        actions = [
            run_command([str(PAIRWISE_SERVICE_SCRIPT), "stop"], timeout=60),
            run_command(
                [str(PAIRWISE_SERVICE_SCRIPT), "start"],
                timeout=60,
                env_updates={
                    "PAIRWISE_FORCE_EXECUTE": "1" if profile.get("force_execute") else "0",
                    "PAIRWISE_FORCE_NOTE": "watchdog_recovery",
                    "PAIRWISE_LIVE_MODE": str(profile.get("mode") or "demo"),
                },
            ),
        ]
        return {
            "profile": profile["key"],
            "actions": actions,
            "ok": all(action["returncode"] == 0 for action in actions),
        }
    return kickstart_launchd(TRADER_LABEL, TRADER_PLIST)


def degrade_pairwise_force_execute(profile: dict[str, Any], reason: str) -> dict[str, Any]:
    updated = write_runtime_profile(
        "pairwise",
        str(profile.get("mode") or "demo"),
        False,
        degraded_by="watchdog",
        degraded_reason=str(reason),
        degraded_at=utc_now().isoformat(),
    )
    return {
        "ok": True,
        "updated_profile": updated,
    }


def maybe_recover(report: dict[str, Any]) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    trader = report["trader"]
    bot = report["bot"]
    profile = active_trader_profile()

    if trader["status"] == "critical":
        if trader.get("stale_seconds") is not None and float(trader["stale_seconds"]) >= float(profile["protect_threshold_seconds"]):
            actions.append({"type": "protect_positions", "result": protect_positions(profile)})
        elif int(trader.get("consecutive_errors", 0)) >= WATCHDOG_ERROR_ESCALATION_COUNT:
            actions.append({"type": "protect_positions", "result": protect_positions(profile)})
        if profile["key"] == "pairwise" and bool(profile.get("force_execute", False)):
            actions.append(
                {
                    "type": "degrade_pairwise_force_execute",
                    "result": degrade_pairwise_force_execute(profile, "watchdog_critical_recovery"),
                }
            )
            profile = {**profile, "force_execute": False}
        actions.append({"type": "restart_trader", "result": restart_active_trader(profile)})

    if bot["status"] == "critical":
        actions.append({"type": "restart_bot", "result": kickstart_launchd(BOT_LABEL, BOT_PLIST)})

    report["recovery_actions"] = actions
    return report


def write_report(report: dict[str, Any]) -> None:
    write_json(WATCHDOG_REPORT_PATH, report)
    write_json(DATA_QUALITY_REPORT_PATH, report.get("data_quality", {}))
    write_json(DECISION_QUALITY_REPORT_PATH, report.get("decision_quality", {}))
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
            "data_quality": {
                "status": report.get("data_quality", {}).get("status"),
            },
            "decision_quality": {
                "status": report.get("decision_quality", {}).get("status"),
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
    profile = active_trader_profile()
    trader_state = read_json(profile["state_path"], {})
    trader_runtime = trader_state.get("runtime_health") or {}
    trader_runtime_pid = int(trader_runtime.get("pid")) if str(trader_runtime.get("pid") or "").isdigit() else None
    trader_pid = resolve_live_pid(trader_runtime_pid, read_pid(profile["pid_path"]))
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
