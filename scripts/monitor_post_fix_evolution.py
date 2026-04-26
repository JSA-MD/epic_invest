#!/usr/bin/env python3
"""Post-fix evolution monitor: daily snapshot of cooldown / breadth / trades / drift.

Runs once per KST day (self-gating). Captures:
  1) cooldown_bars_left per pair from shadow_paper state
  2) breadth_score daily aggregates from decisions log
  3) trade count per pair from slippage log
  4) avg_daily_gap_bps from latest drift diagnostic

Saves:
  models/post_fix_monitoring_<YYYY-MM-DD>.json   (daily snapshot)
  docs/post_fix_monitoring_summary.md            (append-only rolling table)

Sends Telegram alert on threshold violations.

Usage:
    python monitor_post_fix_evolution.py              # normal (KST self-gate)
    python monitor_post_fix_evolution.py --force      # bypass date gate
    python monitor_post_fix_evolution.py --dry-run    # no writes/sends
    python monitor_post_fix_evolution.py --baseline   # tag snapshot as baseline
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telegram_format import AlertLevel, format_alert, format_kst, now_kst

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
DOCS_DIR = ROOT_DIR / "docs"

# Read from the actively running pairwise live trader's state, not the
# shadow-mode trader's separate state file. active_runtime_profile.json
# determines which trader is "live"; for the pairwise variant the file is
# pairwise_regime_live_state.json (same shadow_paper structure, different
# file). Reading the wrong file silently reports stale or zero values.
LIVE_STATE_PATH = MODELS_DIR / "pairwise_regime_live_state.json"
DECISIONS_LOG_PATH = LOGS_DIR / "pairwise_regime_decisions.jsonl"
SLIPPAGE_LOG_PATH = LOGS_DIR / "pairwise_slippage.jsonl"

MONITORING_STATE_PATH = Path("/tmp/epic-invest-post-fix-monitor-last.json")

# 7-day window: auto-stop after this
MONITORING_WINDOW_DAYS = 7

# Thresholds for alerts
ALERT_COOLDOWN_NEVER_ZERO_DAYS = 7   # BNB cooldown never hit 0 in N days
ALERT_NO_TRADE_CONSECUTIVE_DAYS = 3   # daily trade=0 for N consecutive days
ALERT_DRIFT_BPS_THRESHOLD = -100      # drift worse than this bps

KST = timezone(timedelta(hours=9))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

PAIRS = ("BTCUSDT", "BNBUSDT")

# ---------------------------------------------------------------------------
# Data collectors
# ---------------------------------------------------------------------------


def _read_json_safe(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def collect_cooldown_snapshot() -> dict[str, int]:
    """Return current cooldown_bars_left per pair from the live trader state.

    pairwise_regime_live writes its shadow-paper accounting (which is what
    feeds final_decision_cooldown_override) into LIVE_STATE_PATH. Reading
    the standalone shadow trader's file would report a different process.
    """
    state = _read_json_safe(LIVE_STATE_PATH)
    cd = state.get("shadow_paper", {}).get("cooldown_bars_left", {})
    return {pair: int(cd.get(pair, -1)) for pair in PAIRS}


def collect_breadth_stats(date_str: str) -> dict[str, dict]:
    """Aggregate breadth_score from today's decisions entries.

    Returns per-pair: {mean, p50, max, count}.
    """
    if not DECISIONS_LOG_PATH.exists():
        return {pair: {"mean": None, "p50": None, "max": None, "count": 0} for pair in PAIRS}

    scores: dict[str, list[float]] = {pair: [] for pair in PAIRS}
    try:
        for line in DECISIONS_LOG_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Filter to today KST
            at = entry.get("at", "")
            if date_str not in at and date_str not in _utc_to_kst_date(at):
                continue
            pp = entry.get("plan", {}).get("pair_plans", {})
            for pair in PAIRS:
                v = pp.get(pair, {})
                bs = v.get("breadth_score")
                if bs is not None:
                    scores[pair].append(float(bs))
    except Exception:
        pass

    result = {}
    for pair in PAIRS:
        vals = scores[pair]
        if vals:
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            result[pair] = {
                "mean": round(sum(vals) / n, 4),
                "p50": round(vals_sorted[n // 2], 4),
                "max": round(max(vals), 4),
                "count": n,
            }
        else:
            result[pair] = {"mean": None, "p50": None, "max": None, "count": 0}
    return result


def collect_trade_counts(date_str: str) -> dict[str, int]:
    """Count actual trades per pair from slippage log for date_str (KST)."""
    counts: dict[str, int] = {pair: 0 for pair in PAIRS}
    if not SLIPPAGE_LOG_PATH.exists():
        return counts
    try:
        for line in SLIPPAGE_LOG_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Slippage records written by _log_slippage use the "ts" key
            # (iso_now() output). Fall back to "at"/"timestamp" for
            # forward compatibility if the schema changes.
            at = entry.get("ts") or entry.get("at") or entry.get("timestamp") or ""
            if not at:
                continue
            if date_str not in at and date_str not in _utc_to_kst_date(at):
                continue
            sym = entry.get("symbol", "")
            if sym in counts:
                counts[sym] += 1
    except Exception:
        pass
    return counts


def collect_drift_bps() -> float | None:
    """Return avg_daily_gap_bps from the most recent drift diagnostic."""
    try:
        files = sorted(MODELS_DIR.glob("live_drift_diagnostic_*.json"))
        if not files:
            return None
        data = json.loads(files[-1].read_text())
        attr = data.get("attribution", {})
        val = attr.get("avg_daily_gap_bps")
        return float(val) if val is not None else None
    except Exception:
        return None


def _utc_to_kst_date(iso_str: str) -> str:
    """Convert UTC ISO string to KST date string YYYY-MM-DD."""
    try:
        if iso_str.endswith("Z"):
            iso_str = iso_str[:-1] + "+00:00"
        dt = datetime.fromisoformat(iso_str)
        kst_dt = dt.astimezone(KST)
        return kst_dt.strftime("%Y-%m-%d")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _load_monitor_state() -> dict:
    try:
        if MONITORING_STATE_PATH.exists():
            return json.loads(MONITORING_STATE_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_monitor_state(state: dict) -> None:
    try:
        MONITORING_STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def _load_last_run_date() -> str | None:
    return _load_monitor_state().get("last_run_kst_date")


def _save_last_run_date(date_str: str) -> None:
    state = _load_monitor_state()
    state["last_run_kst_date"] = date_str
    _save_monitor_state(state)


def _load_history() -> list[dict]:
    """Load all saved daily snapshot JSON files, sorted by date."""
    snapshots = []
    for f in sorted(MODELS_DIR.glob("post_fix_monitoring_*.json")):
        try:
            snapshots.append(json.loads(f.read_text()))
        except Exception:
            pass
    return snapshots


def _is_past_window(history: list[dict]) -> bool:
    """Return True if we have >= MONITORING_WINDOW_DAYS snapshots."""
    return len(history) >= MONITORING_WINDOW_DAYS


# ---------------------------------------------------------------------------
# Alert evaluation
# ---------------------------------------------------------------------------


def evaluate_alerts(
    today_snapshot: dict,
    history: list[dict],
) -> list[str]:
    """Return list of alert messages (empty = no violations)."""
    alerts: list[str] = []

    # 1) BNB cooldown never reached 0 over the full window
    if len(history) >= ALERT_COOLDOWN_NEVER_ZERO_DAYS:
        bnb_min_cooldowns = [
            s.get("cooldown", {}).get("BNBUSDT", 999)
            for s in history[-ALERT_COOLDOWN_NEVER_ZERO_DAYS:]
        ]
        if all(v > 0 for v in bnb_min_cooldowns):
            alerts.append(
                f"BNB cooldown이 {ALERT_COOLDOWN_NEVER_ZERO_DAYS}일 동안 한 번도 0에 도달하지 않음 "
                f"(최솟값: {min(bnb_min_cooldowns)})"
            )

    # 2) No trades for N consecutive days
    # Check both pairs combined
    recent = history[-(ALERT_NO_TRADE_CONSECUTIVE_DAYS):]
    if len(recent) >= ALERT_NO_TRADE_CONSECUTIVE_DAYS:
        zero_days = [
            s for s in recent
            if sum(s.get("trade_counts", {}).values()) == 0
        ]
        if len(zero_days) >= ALERT_NO_TRADE_CONSECUTIVE_DAYS:
            alerts.append(
                f"전체 거래 건수가 {ALERT_NO_TRADE_CONSECUTIVE_DAYS}일 연속 0건"
            )

    # 3) Drift threshold
    drift = today_snapshot.get("drift_bps")
    if drift is not None and drift < ALERT_DRIFT_BPS_THRESHOLD:
        alerts.append(
            f"Drift {drift:.1f} bps < 임계 {ALERT_DRIFT_BPS_THRESHOLD} bps"
        )

    return alerts


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def _send_telegram(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN:
        print("[monitor] TELEGRAM_BOT_TOKEN not set, skipping send.", file=sys.stderr)
        return
    chat_id = TELEGRAM_CHAT_ID
    if not chat_id:
        print("[monitor] TELEGRAM_CHAT_ID not set, skipping send.", file=sys.stderr)
        return
    params = {
        "chat_id": chat_id,
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
        if not result.get("ok"):
            print(f"[monitor] Telegram send failed: {result}", file=sys.stderr)
    except (HTTPError, URLError) as exc:
        print(f"[monitor] Telegram error: {exc}", file=sys.stderr)


def build_daily_message(today: dict, prev: dict | None, alerts: list[str], is_final: bool) -> str:
    """Format daily Telegram message."""
    date_str = today["date"]
    header = f"*[모니터] Cooldown Fix 추적 {'(7일차 최종)' if is_final else ''}*\n{date_str}"

    lines = [header, ""]

    # Cooldown
    cd = today.get("cooldown", {})
    prev_cd = prev.get("cooldown", {}) if prev else {}
    lines.append("*Cooldown (bars_left)*")
    for pair in PAIRS:
        cur = cd.get(pair, "?")
        prv = prev_cd.get(pair)
        delta = f" (Δ{cur - prv:+d})" if prv is not None and isinstance(cur, int) else ""
        lines.append(f"  {pair}: {cur}{delta}")

    # Breadth
    lines.append("")
    lines.append("*Breadth Score (오늘)*")
    br = today.get("breadth", {})
    for pair in PAIRS:
        b = br.get(pair, {})
        mean = b.get("mean")
        mx = b.get("max")
        n = b.get("count", 0)
        if mean is not None:
            lines.append(f"  {pair}: mean={mean:.3f} max={mx:.3f} (n={n})")
        else:
            lines.append(f"  {pair}: 데이터 없음")

    # Trades
    lines.append("")
    lines.append("*거래 건수 (오늘)*")
    tc = today.get("trade_counts", {})
    prev_tc = prev.get("trade_counts", {}) if prev else {}
    for pair in PAIRS:
        cur = tc.get(pair, 0)
        prv = prev_tc.get(pair, 0)
        delta = f" (Δ{cur - prv:+d})" if prev else ""
        lines.append(f"  {pair}: {cur}건{delta}")

    # Drift
    drift = today.get("drift_bps")
    lines.append("")
    if drift is not None:
        flag = " ⚠️" if drift < ALERT_DRIFT_BPS_THRESHOLD else ""
        lines.append(f"*Drift*: {drift:.1f} bps{flag}")
    else:
        lines.append("*Drift*: 데이터 없음")

    # Alerts
    if alerts:
        lines.append("")
        lines.append("*임계 위반*")
        for a in alerts:
            lines.append(f"  • {a}")

    if is_final:
        lines.append("")
        lines.append("_7일 모니터링 완료. 자동 종료됩니다._")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _append_summary_md(snapshot: dict, is_final: bool) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = DOCS_DIR / "post_fix_monitoring_summary.md"

    date = snapshot["date"]
    cd = snapshot.get("cooldown", {})
    br = snapshot.get("breadth", {})
    tc = snapshot.get("trade_counts", {})
    drift = snapshot.get("drift_bps")
    drift_str = f"{drift:.1f}" if drift is not None else "N/A"

    # Header if file is new
    if not md_path.exists():
        header = (
            "# Post-Fix Evolution Monitor\n\n"
            "Tracks cooldown / breadth / trades / drift after cooldown-override bug fix.\n\n"
            "| Date | BTC cd | BNB cd | BTC breadth | BNB breadth | BTC trades | BNB trades | drift_bps | alerts |\n"
            "|------|--------|--------|-------------|-------------|------------|------------|-----------|--------|\n"
        )
        md_path.write_text(header)

    alerts = snapshot.get("alerts", [])
    alert_str = "; ".join(alerts) if alerts else "-"

    btc_b = br.get("BTCUSDT", {})
    bnb_b = br.get("BNBUSDT", {})
    btc_mean = f"{btc_b.get('mean', 0):.3f}" if btc_b.get("mean") is not None else "N/A"
    bnb_mean = f"{bnb_b.get('mean', 0):.3f}" if bnb_b.get("mean") is not None else "N/A"

    row = (
        f"| {date} "
        f"| {cd.get('BTCUSDT', '?')} "
        f"| {cd.get('BNBUSDT', '?')} "
        f"| {btc_mean} "
        f"| {bnb_mean} "
        f"| {tc.get('BTCUSDT', 0)} "
        f"| {tc.get('BNBUSDT', 0)} "
        f"| {drift_str} "
        f"| {alert_str} |\n"
    )

    with md_path.open("a") as f:
        f.write(row)
        if is_final:
            f.write("\n_7일 모니터링 완료._\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(dry_run: bool = False, is_baseline: bool = False) -> None:
    now = now_kst()
    today_str = now.strftime("%Y-%m-%d")

    # Collect metrics
    try:
        cooldown = collect_cooldown_snapshot()
    except Exception as exc:
        print(f"[monitor] cooldown collect failed: {exc}", file=sys.stderr)
        cooldown = {pair: -1 for pair in PAIRS}

    try:
        breadth = collect_breadth_stats(today_str)
    except Exception as exc:
        print(f"[monitor] breadth collect failed: {exc}", file=sys.stderr)
        breadth = {}

    try:
        trade_counts = collect_trade_counts(today_str)
    except Exception as exc:
        print(f"[monitor] trade_count collect failed: {exc}", file=sys.stderr)
        trade_counts = {pair: 0 for pair in PAIRS}

    try:
        drift_bps = collect_drift_bps()
    except Exception as exc:
        print(f"[monitor] drift collect failed: {exc}", file=sys.stderr)
        drift_bps = None

    snapshot: dict = {
        "date": today_str,
        "generated_at": now.isoformat(),
        "is_baseline": is_baseline,
        "cooldown": cooldown,
        "breadth": breadth,
        "trade_counts": trade_counts,
        "drift_bps": drift_bps,
    }

    # Load history for alert evaluation and diff
    history = _load_history()
    prev = history[-1] if history else None
    is_final = len(history) + 1 >= MONITORING_WINDOW_DAYS

    # Evaluate alerts
    all_history = history + [snapshot]
    alerts = evaluate_alerts(snapshot, all_history)
    snapshot["alerts"] = alerts

    # Print diff table
    print(f"[monitor] {today_str} {'(BASELINE)' if is_baseline else ''}")
    print(f"  cooldown: BTC={cooldown.get('BTCUSDT')} BNB={cooldown.get('BNBUSDT')}")
    if prev:
        prev_cd = prev.get("cooldown", {})
        print(f"  cooldown delta vs prev: BTC Δ{cooldown.get('BTCUSDT',0) - prev_cd.get('BTCUSDT',0):+d}  BNB Δ{cooldown.get('BNBUSDT',0) - prev_cd.get('BNBUSDT',0):+d}")
    for pair in PAIRS:
        b = breadth.get(pair, {})
        print(f"  breadth {pair}: mean={b.get('mean')} p50={b.get('p50')} max={b.get('max')} n={b.get('count',0)}")
    print(f"  trades: BTC={trade_counts.get('BTCUSDT',0)} BNB={trade_counts.get('BNBUSDT',0)}")
    print(f"  drift_bps: {drift_bps}")
    if alerts:
        print(f"  ALERTS: {alerts}")
    if is_final:
        print("  *** 7일 모니터링 완료 ***")

    if dry_run:
        print("[monitor] dry-run: no writes or sends.")
        return

    # Save JSON snapshot
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"post_fix_monitoring_{today_str}.json"
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False))
    print(f"[monitor] saved {out_path}")

    # Append to summary MD
    try:
        _append_summary_md(snapshot, is_final)
    except Exception as exc:
        print(f"[monitor] summary MD write failed: {exc}", file=sys.stderr)

    # Send Telegram
    try:
        msg = build_daily_message(snapshot, prev, alerts, is_final)
        _send_telegram(msg)
    except Exception as exc:
        print(f"[monitor] Telegram send failed: {exc}", file=sys.stderr)

    # Record run date (dedup)
    _save_last_run_date(today_str)
    print(f"[monitor] done (day {len(history) + 1}/{MONITORING_WINDOW_DAYS})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-fix evolution monitor")
    parser.add_argument("--dry-run", action="store_true", help="No writes or Telegram sends")
    parser.add_argument("--force", action="store_true", help="Bypass KST hour/date gate")
    parser.add_argument("--baseline", action="store_true", help="Tag snapshot as baseline")
    parser.add_argument(
        "--target-kst-hour",
        type=int,
        default=int(os.environ.get("MONITOR_TARGET_KST_HOUR", "9")),
        help="KST hour gate (default 9)",
    )
    args = parser.parse_args()

    if not args.force and not args.dry_run:
        now = now_kst()
        today_str = now.strftime("%Y-%m-%d")

        # Catch-up: run if past target hour and not yet run today
        if now.hour < args.target_kst_hour:
            sys.exit(0)

        if _load_last_run_date() == today_str:
            sys.exit(0)

        # Auto-stop after 7-day window
        history = _load_history()
        if _is_past_window(history):
            print("[monitor] 7일 윈도우 완료. 추가 실행 스킵.")
            sys.exit(0)

    run(dry_run=args.dry_run, is_baseline=args.baseline)


if __name__ == "__main__":
    main()
