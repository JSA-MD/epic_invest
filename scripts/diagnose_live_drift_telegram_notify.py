#!/usr/bin/env python3
"""
diagnose_live_drift_telegram_notify.py — Telegram alert for live-drift diagnostic results.

Reads the JSON report produced by diagnose_live_drift.py and sends a Telegram
message when the mean daily drift exceeds the alert threshold.

Usage:
    python scripts/diagnose_live_drift_telegram_notify.py \
        --report models/live_drift_diagnostic_20260426.json

Exit codes:
    0  normal (sent or skipped below threshold)
    1  configuration error (missing token/chat_id)
    2  report file not found or malformed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.environ.get("TELEGRAM_CHAT_ID", "")

# Alert threshold: absolute daily drift exceeding this value (in bps) triggers a message.
# Fires in BOTH directions — a +200 bps "free money" gap is also a calibration error.
# Override via env: ATTRIBUTION_ALERT_THRESHOLD_BPS=50 (default 50 bps).
# Legacy env DRIFT_ALERT_THRESHOLD_BPS still accepted for backward compatibility.
ATTRIBUTION_ALERT_THRESHOLD_BPS: float = float(
    os.environ.get(
        "ATTRIBUTION_ALERT_THRESHOLD_BPS",
        os.environ.get("DRIFT_ALERT_THRESHOLD_BPS", "50"),
    )
)
# Keep old name as alias so any external callers don't break.
DRIFT_ALERT_THRESHOLD_BPS: float = ATTRIBUTION_ALERT_THRESHOLD_BPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_kst() -> datetime:
    from datetime import timedelta
    KST = timezone(timedelta(hours=9))
    return datetime.now(KST)


def _send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_notification": "false",
    }
    data = urlencode(params).encode("utf-8")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    req = Request(url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Telegram API 오류: {exc}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API 실패: {result}")


def _extract_drift_bps(report: dict) -> float | None:
    """Return mean daily drift (bps) from the diagnostic report, or None.

    Supports two schemas:

    1. Legacy live_drift_diagnostic JSON (diagnose_live_drift.py --json)::

        {
          "attribution": {"avg_daily_gap_bps": ..., ...},
          ...
        }

    2. New attribution_daily JSON (Stage 2.3)::

        {
          "rows": [{"drift_bps": ..., ...}, ...],
          "per_pair": {"BNBUSDT": {"drift_bps": ...}, ...},
          ...
        }

    If extraction fails, return None so the caller can exit non-zero loudly.
    """
    # Schema 2: attribution_daily — average drift_bps across all rows
    rows = report.get("rows")
    if isinstance(rows, list) and rows:
        drifts = [r["drift_bps"] for r in rows if isinstance(r.get("drift_bps"), (int, float))]
        if drifts:
            return float(sum(drifts) / len(drifts))

    # Schema 2 alt: per_pair dict
    per_pair = report.get("per_pair")
    if isinstance(per_pair, dict) and per_pair:
        drifts = [v["drift_bps"] for v in per_pair.values()
                  if isinstance(v.get("drift_bps"), (int, float))]
        if drifts:
            return float(sum(drifts) / len(drifts))

    # Schema 1: legacy key paths
    for path in (
        ("attribution", "avg_daily_gap_bps"),
        ("attribution", "mean_daily_drift_bps"),
        ("summary", "mean_daily_drift_bps"),
        ("summary", "avg_daily_gap_bps"),
        ("summary", "drift_bps"),
        ("drift_bps",),
        ("mean_daily_drift_bps",),
        ("avg_daily_gap_bps",),
    ):
        node = report
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                node = None
                break
        if isinstance(node, (int, float)):
            return float(node)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send Telegram alert if live-vs-backtest drift exceeds threshold."
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to JSON report produced by diagnose_live_drift.py",
    )
    parser.add_argument(
        "--threshold-bps",
        type=float,
        default=ATTRIBUTION_ALERT_THRESHOLD_BPS,
        help=f"Alert fires when |drift| > this value in bps (default: {ATTRIBUTION_ALERT_THRESHOLD_BPS})",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"[skip] 보고서 파일 없음: {report_path}", file=sys.stderr)
        sys.exit(2)

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[error] 보고서 파싱 실패: {exc}", file=sys.stderr)
        sys.exit(2)

    drift_bps = _extract_drift_bps(report)
    ts_kst = _now_kst().strftime("%Y-%m-%d %H:%M KST")

    if drift_bps is None:
        # Schema mismatch is a real wiring failure, not a benign skip — exit 2
        # so the launchd entry sees it (entry uses `|| true` so it won't kill
        # the run, but stderr will surface the problem in launchd logs).
        print(
            f"[error] 보고서에서 drift_bps를 찾지 못함 (schema mismatch?) "
            f"({report_path.name}). diagnose_live_drift.py 의 JSON 키와 "
            f"_extract_drift_bps 의 경로 목록을 점검하세요.",
            file=sys.stderr,
        )
        sys.exit(2)

    direction = "over" if drift_bps > 0 else "under"
    print(f"[info] mean_daily_drift_bps = {drift_bps:.1f} bps  |abs| = {abs(drift_bps):.1f}  (임계: ±{args.threshold_bps:.0f} bps)")

    if abs(drift_bps) <= args.threshold_bps:
        print(f"[ok] |drift| {abs(drift_bps):.1f} bps ≤ {args.threshold_bps:.0f} bps 임계 → 알림 불필요.")
        sys.exit(0)

    # Build message — fires for both overperformance (+) and underperformance (-)
    direction_label = "초과 (+, 보정 오류 의심)" if drift_bps > 0 else "부족 (-, 손실 의심)"
    text = (
        f"*[EpicInvest] Live Drift 경보* — {ts_kst}\n\n"
        f"일평균 drift: *{drift_bps:+.1f} bps* ({direction_label})\n"
        f"임계: ±{args.threshold_bps:.0f} bps (양방향)\n"
        f"보고서: `{report_path.name}`\n\n"
        f"즉시 확인: `bash scripts/diagnose_live_drift_launchd_entry.sh`"
    )

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"[warn] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정 — 알림 스킵.", file=sys.stderr)
        sys.exit(0)

    try:
        _send_telegram(text)
        print(f"[ok] Telegram 알림 전송 완료 (drift={drift_bps:.1f} bps)")
    except RuntimeError as exc:
        print(f"[error] Telegram 전송 실패: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
