#!/usr/bin/env python3
"""Component 4: Anomaly detection alarm — Sharpe / win-rate drop.

Runs daily at 09:30 KST (via launchd). Compares last-7-day OOS Sharpe
and live win-rate against the prior 30-day baseline. Sends AlertLevel.HIGH
via format_alert if Sharpe drops >30% or win-rate drops >15pp.

Usage:
    python telegram_anomaly_alarm.py              # normal run
    python telegram_anomaly_alarm.py --dry-run    # print payload, do not send
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

# Insert scripts dir so telegram_format is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from telegram_format import AlertLevel, format_alert, format_kst, now_kst

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
WALKFORWARD_PATH = ROOT_DIR / "models" / "walkforward_report.json"
PNL_JSON_PATH = ROOT_DIR / "models" / "live_actual_pnl_30d.json"
ANOMALY_STATE_PATH = Path("/tmp/epic-anomaly-last.json")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

KST = timezone(timedelta(hours=9))

# Thresholds
SHARPE_DROP_THRESHOLD = 0.30   # >30% relative drop triggers alarm
WINRATE_DROP_THRESHOLD = 0.15  # >15 percentage-point drop triggers alarm

# Debounce: do not re-alarm within 23 hours
DEBOUNCE_HOURS = 23


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


def _extract_oos_metrics(folds: list[dict], recent_n: int) -> tuple[float, float]:
    """Return (mean_sharpe, mean_win_rate) for the last *recent_n* folds.

    Averages across all pairs within each selected fold.
    """
    selected = folds[-recent_n:] if len(folds) >= recent_n else folds
    sharpes: list[float] = []
    win_rates: list[float] = []
    for fold in selected:
        oos = fold.get("OOS", {})
        for pair_metrics in oos.values():
            if isinstance(pair_metrics, dict):
                s = pair_metrics.get("sharpe")
                w = pair_metrics.get("roundtrip_win_rate")
                if s is not None:
                    sharpes.append(float(s))
                if w is not None:
                    win_rates.append(float(w))
    mean_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
    mean_winrate = sum(win_rates) / len(win_rates) if win_rates else 0.0
    return mean_sharpe, mean_winrate


def load_walkforward_metrics() -> dict:
    """Load walkforward report and compute recent vs. baseline metrics."""
    data = json.loads(WALKFORWARD_PATH.read_text())
    folds: list[dict] = data.get("folds", [])

    if len(folds) < 2:
        return {
            "recent_sharpe": 0.0,
            "baseline_sharpe": 0.0,
            "recent_winrate": 0.0,
            "baseline_winrate": 0.0,
            "n_folds": len(folds),
        }

    # Recent = last ~7 days worth of folds; each fold is ~30 days
    # Use last 1 fold as "recent" and folds[-4:-1] as "prior 30d baseline"
    # More precisely: folds cover test_days=30, so last fold ~ last 30d.
    # "Recent 7 days" = last fold's last week. Since we only have fold-level
    # granularity, we use last fold vs prior 3 folds as proxy.
    recent_sharpe, recent_winrate = _extract_oos_metrics(folds, recent_n=1)
    baseline_sharpe, baseline_winrate = _extract_oos_metrics(folds[:-1], recent_n=3)

    return {
        "recent_sharpe": recent_sharpe,
        "baseline_sharpe": baseline_sharpe,
        "recent_winrate": recent_winrate,
        "baseline_winrate": baseline_winrate,
        "n_folds": len(folds),
    }


def load_live_win_rate(window_days: int = 7) -> float | None:
    """Compute live win-rate from the last *window_days* daily PnL rows.

    Returns fraction of positive-PnL days, or None if no data.
    """
    if not PNL_JSON_PATH.exists():
        return None
    raw = json.loads(PNL_JSON_PATH.read_text())
    rows: list[dict] = raw.get("daily_pnl_live", [])

    # Aggregate per date
    by_date: dict[str, float] = {}
    for row in rows:
        d = row["date"]
        by_date[d] = by_date.get(d, 0.0) + float(row.get("total", 0.0))

    sorted_dates = sorted(by_date.keys())
    recent = sorted_dates[-window_days:]
    if not recent:
        return None

    wins = sum(1 for d in recent if by_date[d] > 0)
    return wins / len(recent)


# ---------------------------------------------------------------------------
# Debounce
# ---------------------------------------------------------------------------


def _should_alarm() -> bool:
    """Return True if enough time has passed since the last alarm."""
    if not ANOMALY_STATE_PATH.exists():
        return True
    try:
        state = json.loads(ANOMALY_STATE_PATH.read_text())
        last_ts = float(state.get("last_alarm_ts", 0))
    except (json.JSONDecodeError, ValueError, OSError):
        return True
    import time
    return (time.time() - last_ts) >= DEBOUNCE_HOURS * 3600


def _record_alarm() -> None:
    import time
    try:
        ANOMALY_STATE_PATH.write_text(json.dumps({
            "last_alarm_ts": time.time(),
            "last_alarm_kst": now_kst().isoformat(),
        }))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Telegram send
# ---------------------------------------------------------------------------


def _send_telegram(payload: dict, chat_id: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
    params = {
        "chat_id": chat_id,
        "text": payload["text"],
        "parse_mode": payload.get("parse_mode", "Markdown"),
        "disable_notification": str(payload.get("disable_notification", False)).lower(),
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


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def run(dry_run: bool = False) -> bool:
    """Execute anomaly check. Returns True if an anomaly was detected."""
    wf = load_walkforward_metrics()
    live_winrate_7d = load_live_win_rate(window_days=7)
    live_winrate_30d = load_live_win_rate(window_days=30)

    anomalies: list[str] = []
    context: dict[str, str] = {}

    # Sharpe drop check (relative)
    recent_sharpe = wf["recent_sharpe"]
    baseline_sharpe = wf["baseline_sharpe"]
    if baseline_sharpe > 0:
        sharpe_drop_rel = (baseline_sharpe - recent_sharpe) / abs(baseline_sharpe)
        context["Sharpe (최근)"] = f"{recent_sharpe:.3f}"
        context["Sharpe (기준선)"] = f"{baseline_sharpe:.3f}"
        context["Sharpe 하락률"] = f"{sharpe_drop_rel * 100:.1f}%"
        if sharpe_drop_rel > SHARPE_DROP_THRESHOLD:
            anomalies.append(
                f"OOS Sharpe {sharpe_drop_rel * 100:.1f}% 하락 "
                f"({baseline_sharpe:.3f} → {recent_sharpe:.3f})"
            )
    elif baseline_sharpe <= 0:
        context["Sharpe (최근)"] = f"{recent_sharpe:.3f}"
        context["Sharpe (기준선)"] = f"{baseline_sharpe:.3f} (음수 기준선, 스킵)"

    # Win-rate drop check (absolute pp)
    effective_recent_wr = live_winrate_7d if live_winrate_7d is not None else wf["recent_winrate"]
    effective_base_wr = live_winrate_30d if live_winrate_30d is not None else wf["baseline_winrate"]
    wr_drop = effective_base_wr - effective_recent_wr
    context["승률 (최근 7일)"] = f"{effective_recent_wr * 100:.1f}%"
    context["승률 (30일 기준선)"] = f"{effective_base_wr * 100:.1f}%"
    context["승률 하락폭"] = f"{wr_drop * 100:.1f}pp"
    if wr_drop > WINRATE_DROP_THRESHOLD:
        anomalies.append(
            f"승률 {wr_drop * 100:.1f}pp 하락 "
            f"({effective_base_wr * 100:.1f}% → {effective_recent_wr * 100:.1f}%)"
        )

    if not anomalies:
        print("이상 없음: Sharpe / 승률 정상 범위.")
        return False

    # Build alert
    body = "전략 성과 이상 감지됨:\n" + "\n".join(f"  • {a}" for a in anomalies)
    payload = format_alert(
        level=AlertLevel.HIGH,
        title="전략 성과 이상 경보",
        body=body,
        context=context,
        timestamp_kst=now_kst(),
    )

    if dry_run:
        print("=== [DRY-RUN] Telegram 페이로드 ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("=== 이상 항목 ===")
        for a in anomalies:
            print(f"  • {a}")
        return True

    if not _should_alarm():
        print(f"디바운스 활성: 마지막 알람 후 {DEBOUNCE_HOURS}시간 미경과. 전송 스킵.")
        return True

    chat_id = TELEGRAM_CHAT_ID
    if not chat_id:
        print("오류: TELEGRAM_CHAT_ID가 설정되지 않았습니다.", file=sys.stderr)
        sys.exit(1)

    _send_telegram(payload, chat_id)
    _record_alarm()
    print(f"이상 경보 전송 완료: {len(anomalies)}건")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="전략 성과 이상 감지 알람")
    parser.add_argument("--dry-run", action="store_true", help="전송 없이 페이로드 출력")
    args = parser.parse_args()
    detected = run(dry_run=args.dry_run)
    sys.exit(0 if not detected else 2)


if __name__ == "__main__":
    main()
