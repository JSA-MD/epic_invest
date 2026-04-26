#!/usr/bin/env python3
"""Hourly badge sender for Epic Invest pairwise demo.

Runs every hour 09:00-21:00 KST (00:00-12:00 UTC via launchd).
Sends a silent one-line status badge. Skips silently if state
has not changed since last send.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telegram_format import (  # noqa: E402
    AlertLevel,
    format_alert,
    format_kst,
    now_kst,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

LIVE_PNL_PATH = ROOT_DIR / "models" / "live_actual_pnl_30d.json"
LIVE_STATE_PATH = ROOT_DIR / "models" / "pairwise_regime_live_state.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

KST = timezone(timedelta(hours=9))
LAST_SEND_PATH = Path("/tmp/epic-invest-tg-hourly-last.json")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _daily_total_today(rows: list[dict], today_str: str) -> float:
    total = 0.0
    for row in rows:
        if str(row.get("date", "")) == today_str:
            total += float(row.get("total", 0.0))
    return total


def _active_alerts(live_state: dict) -> list[str]:
    """Return list of active alert descriptions."""
    alerts: list[str] = []
    now_ts = datetime.now(tz=timezone.utc)

    # CVaR cut active pairs
    cvar_cut = live_state.get("cvar_cut_until_ts") or {}
    for pair, ts in cvar_cut.items():
        if ts is None:
            continue
        try:
            until = datetime.fromisoformat(str(ts))
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            if now_ts < until:
                alerts.append(f"{pair} CVaR cut")
        except ValueError:
            pass

    # Runtime errors
    runtime_health = live_state.get("runtime_health") or {}
    last_err = runtime_health.get("last_error")
    if last_err:
        # Only count as active if error is recent (< 30 min)
        last_err_at = runtime_health.get("last_error_at")
        if last_err_at:
            try:
                err_dt = datetime.fromisoformat(str(last_err_at))
                if err_dt.tzinfo is None:
                    err_dt = err_dt.replace(tzinfo=timezone.utc)
                if (now_ts - err_dt).total_seconds() < 1800:
                    alerts.append("런타임 오류")
            except ValueError:
                pass

    return alerts


def _build_state_fingerprint(pnl: float, alerts: list[str]) -> str:
    """Build a string fingerprint to detect change since last hour."""
    return json.dumps(
        {"pnl_rounded": round(pnl, 4), "alerts": sorted(alerts)},
        ensure_ascii=False,
        sort_keys=True,
    )


def _load_last() -> dict:
    return _load_json(LAST_SEND_PATH)


def _save_last(fingerprint: str, kst_hour: str) -> None:
    try:
        LAST_SEND_PATH.write_text(
            json.dumps({"fingerprint": fingerprint, "kst_hour": kst_hour})
        )
    except OSError:
        pass


def build_badge(kst_now: datetime) -> tuple[dict, str]:
    """Return (payload, fingerprint)."""
    live_pnl = _load_json(LIVE_PNL_PATH)
    live_state = _load_json(LIVE_STATE_PATH)

    today_kst = kst_now.date().isoformat()
    rows = live_pnl.get("daily_pnl_live") or []
    today_pnl = _daily_total_today(rows, today_kst)

    alerts = _active_alerts(live_state)
    n_alerts = len(alerts)
    fingerprint = _build_state_fingerprint(today_pnl, alerts)

    hour_str = kst_now.strftime("%H:%M")

    if n_alerts == 0:
        icon = "✅"
        status_str = "정상"
        alert_part = "0 active alerts"
    else:
        icon = "⚠️"
        status_str = "주의"
        first_alert = alerts[0]
        alert_part = f"{n_alerts} alert: {first_alert}"

    pnl_str = f"{today_pnl:+.4f}%"  # shown as running total (unitless ratio display)
    body = f"{icon} {hour_str} KST {status_str} / {today_pnl:+.4f} USDT / {alert_part}"

    payload = format_alert(
        AlertLevel.DIGEST,
        title=f"시간별 현황 — {hour_str} KST",
        body=body,
        context=None,
        buttons=None,
        timestamp_kst=kst_now,
    )
    # Force silent (DIGEST already sets disable_notification=True, but be explicit)
    payload["disable_notification"] = True

    return payload, fingerprint


def send_payload(payload: dict) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID is not set")

    params: dict = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": payload["text"],
        "parse_mode": payload.get("parse_mode", "Markdown"),
        "disable_notification": "true",
    }
    data = urlencode(params).encode()
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    req = Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("ok"):
                raise RuntimeError(f"Telegram API error: {result}")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Telegram request failed: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Epic Invest hourly badge sender.")
    parser.add_argument("--dry-run", action="store_true", help="Print payload, do not send.")
    parser.add_argument(
        "--force", action="store_true", help="Send even if no change since last hour."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kst_now = now_kst()
    hour = kst_now.hour

    # Quiet hours: 22:00-08:00 KST — skip silently
    if hour < 9 or hour > 21:
        if args.dry_run:
            print(f"[건너뜀] 정숙 시간 ({hour:02d}:00 KST)")
        return

    payload, fingerprint = build_badge(kst_now)
    kst_hour_key = kst_now.strftime("%Y-%m-%dT%H")

    # Skip if no change since last hour
    if not args.force and not args.dry_run:
        last = _load_last()
        if last.get("fingerprint") == fingerprint:
            # No change — skip silently
            return

    if args.dry_run:
        print("=== DRY RUN: Hourly Badge Payload ===")
        print(f"text:\n{payload['text']}")
        print(f"parse_mode: {payload['parse_mode']}")
        print(f"disable_notification: {payload.get('disable_notification')}")
        print(f"fingerprint: {fingerprint}")
        return

    send_payload(payload)
    _save_last(fingerprint, kst_hour_key)
    print(f"Hourly badge sent at {format_kst(kst_now)}")


if __name__ == "__main__":
    main()
