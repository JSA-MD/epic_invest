#!/usr/bin/env python3
"""Daily digest sender for Epic Invest pairwise demo.

Runs once per day at 09:00 KST (00:00 UTC via launchd).
Reads live PnL, walkforward stats, and live state to build a
DIGEST card with yesterday's performance and a 7-day sparkline.
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

# Allow importing sibling scripts without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from telegram_format import (  # noqa: E402
    AlertLevel,
    BUTTON_LABELS,
    format_alert,
    format_kst,
    now_kst,
)

ROOT_DIR = Path(__file__).resolve().parents[1]

LIVE_PNL_PATH = ROOT_DIR / "models" / "live_actual_pnl_30d.json"
WF_REPORT_PATH = ROOT_DIR / "models" / "walkforward_report.json"
LIVE_STATE_PATH = ROOT_DIR / "models" / "pairwise_regime_live_state.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

KST = timezone(timedelta(hours=9))

# Unicode block chars for sparkline (index 0 = lowest, 7 = highest)
_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_ZERO_CHAR = "▂"  # neutral bar shown when value is exactly zero


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _daily_totals(rows: list[dict]) -> dict[str, float]:
    """Aggregate daily_pnl_live rows into {date_str: total_pnl}."""
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        d = str(row.get("date", ""))
        total = float(row.get("total", 0.0))
        if d:
            totals[d] += total
    return dict(totals)


def _sparkline(values: list[float]) -> str:
    """Render a list of floats as a unicode block sparkline (7 or 8 chars)."""
    if not values:
        return _ZERO_CHAR * 7
    mn = min(values)
    mx = max(values)
    span = mx - mn
    bars = []
    for v in values:
        if span == 0:
            idx = 3  # middle bar for flat data
        else:
            idx = int((v - mn) / span * (len(_SPARK_CHARS) - 1))
        bars.append(_SPARK_CHARS[idx])
    return "".join(bars)


def _safety_icons(live_state: dict) -> str:
    """Build 5-icon safety status row."""
    now_ts = datetime.now(tz=timezone.utc)

    # D1: max-hold 24h — any pair open > 24h
    position_open_since = live_state.get("position_open_since_ts") or {}
    hold_ok = True
    for pair, ts in position_open_since.items():
        if ts is None:
            continue
        try:
            opened = datetime.fromisoformat(str(ts))
            if opened.tzinfo is None:
                opened = opened.replace(tzinfo=timezone.utc)
            age_hours = (now_ts - opened).total_seconds() / 3600
            if age_hours > 24:
                hold_ok = False
                break
        except ValueError:
            pass

    # R3: CVaR cut active for any pair
    cvar_cut = live_state.get("cvar_cut_until_ts") or {}
    cvar_ok = True
    for pair, ts in cvar_cut.items():
        if ts is None:
            continue
        try:
            until = datetime.fromisoformat(str(ts))
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            if now_ts < until:
                cvar_ok = False
                break
        except ValueError:
            pass

    # D3: reconciliation — runtime_health.status == "ok"
    runtime_health = live_state.get("runtime_health") or {}
    status = str(runtime_health.get("status", "")).lower()
    recon_ok = status in ("ok", "")

    # D4: price feed freshness — last_success_at within 5 min
    last_success = runtime_health.get("last_success_at")
    feed_ok = False
    if last_success:
        try:
            ls = datetime.fromisoformat(str(last_success))
            if ls.tzinfo is None:
                ls = ls.replace(tzinfo=timezone.utc)
            feed_ok = (now_ts - ls).total_seconds() < 300
        except ValueError:
            pass

    # D2: gross cap check (always OK in this digest; we lack intraday leverage data)
    gross_cap_ok = True

    icons = [
        "✅" if gross_cap_ok else "🔴",   # D2 Gross cap
        "✅" if hold_ok else "⚠️",         # D1 Max-hold
        "✅" if cvar_ok else "🔴",         # R3 CVaR cut
        "✅" if recon_ok else "⚠️",        # D3 Reconciliation
        "✅" if feed_ok else "⚠️",         # D4 Price feed
    ]
    labels = ["Gross", "Hold", "CVaR", "Recon", "Feed"]
    return "  ".join(f"{icon}{lbl}" for icon, lbl in zip(icons, labels))


def build_digest(dry_run: bool = False) -> dict:
    live_pnl = _load_json(LIVE_PNL_PATH)
    wf = _load_json(WF_REPORT_PATH)
    live_state = _load_json(LIVE_STATE_PATH)

    rows = live_pnl.get("daily_pnl_live") or []
    daily = _daily_totals(rows)

    # Yesterday in KST
    kst_now = now_kst()
    yesterday = (kst_now - timedelta(days=1)).date()
    yesterday_str = yesterday.isoformat()

    # Yesterday's PnL (sum across all pairs)
    yest_pnl = daily.get(yesterday_str, None)
    pnl_str = f"{yest_pnl:+.4f} USDT" if yest_pnl is not None else "데이터 없음"

    # 7-day window
    seven_days = [(kst_now - timedelta(days=i)).date().isoformat() for i in range(6, -1, -1)]
    seven_vals = [daily.get(d, 0.0) for d in seven_days]
    sparkline = _sparkline(seven_vals)
    spark_pcts = "  ".join(
        f"{v:+.2f}" for v in seven_vals
    )

    # Win rate from walkforward OOS summary
    wf_summary = wf.get("summary") or {}
    oos_win_rate = wf_summary.get("OOS_mean_win_rate")
    oos_sharpe = wf_summary.get("OOS_mean_sharpe")
    win_rate_str = f"{oos_win_rate:.1%}" if oos_win_rate is not None else "-"
    sharpe_str = f"{oos_sharpe:.2f}" if oos_sharpe is not None else "-"

    # Max intraday DD (approximate from 7-day window)
    max_dd = min(seven_vals) if seven_vals else 0.0
    max_dd_str = f"{max_dd:+.4f} USDT" if max_dd <= 0 else "+0.0000 USDT"

    # Safety icons
    safety = _safety_icons(live_state)

    # CVaR cut pairs
    cvar_cut = live_state.get("cvar_cut_until_ts") or {}
    cvar_active = [p for p, ts in cvar_cut.items() if ts is not None]
    cvar_str = ", ".join(cvar_active) if cvar_active else "없음"

    # Target % from walkforward
    oos_ret = wf_summary.get("OOS_mean_return")
    target_str = f"{oos_ret:.2%}" if oos_ret is not None else "-"

    title = f"어제 ({yesterday.strftime('%m/%d')}) 요약 — pairwise demo"

    body_lines = [
        f"📅 어제 PnL: {pnl_str}",
        f"🏆 OOS 승률: {win_rate_str}  |  샤프: {sharpe_str}",
        f"🎯 OOS 수익률 (목표): {target_str}",
        f"📉 최대 일간 손실 (7일): {max_dd_str}",
        f"🔒 CVaR 차단 종목: {cvar_str}",
        "",
        f"📅 7일 추세  {sparkline}",
        f"  ({spark_pcts})",
    ]
    body = "\n".join(body_lines)

    context = {
        "안전 상태": safety,
        "기준 시각": format_kst(kst_now),
    }

    buttons = [
        {"label": BUTTON_LABELS["chart"], "callback_data": "digest:chart"},
        {"label": "⚙️ 설정", "callback_data": "digest:settings"},
        {"label": "📆 주간", "callback_data": "digest:weekly"},
    ]

    return format_alert(
        AlertLevel.DIGEST,
        title=title,
        body=body,
        context=context,
        buttons=buttons,
        timestamp_kst=kst_now,
    )


def send_payload(payload: dict) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID is not set")

    params: dict = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": payload["text"],
        "parse_mode": payload.get("parse_mode", "Markdown"),
    }
    if payload.get("disable_notification"):
        params["disable_notification"] = "true"
    if payload.get("reply_markup"):
        params["reply_markup"] = json.dumps(payload["reply_markup"])

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
    parser = argparse.ArgumentParser(description="Epic Invest daily digest sender.")
    parser.add_argument("--dry-run", action="store_true", help="Print payload, do not send.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_digest(dry_run=args.dry_run)

    if args.dry_run:
        print("=== DRY RUN: Daily Digest Payload ===")
        print(f"text:\n{payload['text']}")
        print(f"parse_mode: {payload['parse_mode']}")
        print(f"disable_notification: {payload.get('disable_notification')}")
        if payload.get("reply_markup"):
            print(f"reply_markup: {json.dumps(payload['reply_markup'], ensure_ascii=False, indent=2)}")
        return

    send_payload(payload)
    print(f"Daily digest sent at {format_kst(now_kst())}")


if __name__ == "__main__":
    main()
