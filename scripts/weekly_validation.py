#!/usr/bin/env python3
"""Weekly self-validation wrapper.

Runs walk-forward OOS validation, live-vs-backtest drift detection, and
live daily win-rate computation, then sends a Telegram digest.

Usage:
    python scripts/weekly_validation.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()

UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPTS = ROOT / "scripts"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

WF_REPORT = MODELS / "weekly_validation_walkforward.json"
RECON_REPORT = MODELS / "weekly_validation_reconciliation.json"
DECISION_LOG = LOGS / "pairwise_regime_decisions.jsonl"

# Thresholds
SHARPE_DEFLATION_WARN = 0.50       # 50% drop in OOS Sharpe latest-4 vs prior-4
SIGN_MISMATCH_WARN = 25.0          # % per pair
ROUTE_MISMATCH_WARN = 30.0         # % across all pairs (route_state_mismatch_count / n_matched)
WIN_RATE_WARN = 40.0               # daily win-rate below this is WARN


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def _telegram_post(token: str, method: str, payload: dict) -> None:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urlencode(payload).encode()
    req = Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urlopen(req, timeout=15) as resp:
            resp.read()
    except (HTTPError, URLError) as exc:
        print(f"[weekly_validation] Telegram post failed: {exc}", file=sys.stderr)


def send_telegram(token: str, chat_id: str, text: str) -> None:
    # Split at 4000 chars to stay under Telegram's 4096 limit.
    chunk_size = 4000
    for i in range(0, len(text), chunk_size):
        _telegram_post(token, "sendMessage", {
            "chat_id": chat_id,
            "text": text[i:i + chunk_size],
            "disable_web_page_preview": "true",
        })


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> tuple[int, str]:
    """Run a subprocess, stream stdout, return (returncode, combined output)."""
    print(f"\n[weekly_validation] Running {label} ...", flush=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    return result.returncode, result.stdout + result.stderr


def run_walkforward(start: str, end: str) -> int:
    rc, _ = _run(
        [
            str(PYTHON), str(SCRIPTS / "walkforward_pairwise.py"),
            "--start", start,
            "--end", end,
            "--report-out", str(WF_REPORT),
        ],
        "walkforward_pairwise",
    )
    return rc


def run_reconciliation(days: int) -> int:
    rc, _ = _run(
        [
            str(PYTHON), str(SCRIPTS / "live_vs_backtest_reconciliation.py"),
            "--days", str(days),
            "--report-out", str(RECON_REPORT),
        ],
        "live_vs_backtest_reconciliation",
    )
    return rc


LIVE_PNL_REPORT = ROOT / "models" / "live_actual_pnl_30d.json"
LIVE_PNL_MAX_AGE_SECONDS = 7200  # 2 hours; weekly run produces fresh data each invocation


def run_live_pnl_refresh(days: int = 30) -> int:
    """Regenerate models/live_actual_pnl_30d.json so the win-rate gate is current.

    Codex flagged that reading a stale snapshot makes the win-rate check
    a constant; we now invoke live_actual_pnl_30d.py freshly each run.
    """
    script = SCRIPTS / "live_actual_pnl_30d.py"
    if not script.exists():
        print(f"[weekly_validation] live_actual_pnl_30d.py missing", file=sys.stderr)
        return 127
    rc, _ = _run(
        [str(PYTHON), str(script), "--days", str(days), "--report-out", str(LIVE_PNL_REPORT)],
        "live_actual_pnl_30d",
    )
    return rc


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        print(f"[weekly_validation] JSON parse error {path}: {exc}", file=sys.stderr)
        return None


def analyse_walkforward(report: dict) -> dict:
    """Return fold-level Sharpe analysis and flags."""
    folds = report.get("folds", [])
    pairs = report.get("params", {}).get("pairs", [])

    # Collect per-fold mean OOS Sharpe across pairs.
    fold_sharpes: list[float] = []
    for fold in folds:
        oos = fold.get("OOS", {})
        sharpes = [oos[p]["sharpe"] for p in pairs if p in oos]
        if sharpes:
            fold_sharpes.append(sum(sharpes) / len(sharpes))

    n = len(fold_sharpes)
    prior_4 = fold_sharpes[max(0, n - 8): max(0, n - 4)] if n >= 5 else fold_sharpes[:max(0, n - 4)]
    latest_4 = fold_sharpes[max(0, n - 4):]

    latest_mean = sum(latest_4) / len(latest_4) if latest_4 else 0.0
    prior_mean = sum(prior_4) / len(prior_4) if prior_4 else 0.0
    delta = latest_mean - prior_mean

    # Deflation: relative drop when prior is positive
    deflation_flag = False
    if prior_mean > 0 and latest_mean < prior_mean:
        deflation_pct = (prior_mean - latest_mean) / abs(prior_mean)
        if deflation_pct > SHARPE_DEFLATION_WARN:
            deflation_flag = True
    elif prior_mean <= 0 and latest_mean < prior_mean:
        # Prior was already non-positive; any further drop is a flag
        deflation_flag = True

    return {
        "n_folds": n,
        "latest_4_mean_sharpe": round(latest_mean, 4),
        "prior_4_mean_sharpe": round(prior_mean, 4),
        "delta": round(delta, 4),
        "deflation_flag": deflation_flag,
        "fold_sharpes": [round(s, 4) for s in fold_sharpes],
    }


def analyse_reconciliation(report: dict) -> dict:
    """Return per-pair mismatch numbers and flags."""
    if report.get("status") == "no_live_data":
        return {"no_data": True, "btc_sign_mismatch_pct": 0.0, "bnb_sign_mismatch_pct": 0.0,
                "route_mismatch_pct": 0.0, "sign_flag": False, "route_flag": False}

    per_pair = report.get("per_pair", {})

    def _pct(pair: str) -> float:
        d = per_pair.get(pair, {})
        return float(d.get("sign_mismatch_pct", 0.0))

    def _route_mismatch_pct() -> float:
        total_matched = sum(d.get("n_live_matched_to_bar", 0) for d in per_pair.values())
        total_route_mis = sum(d.get("route_state_mismatch_count", 0) for d in per_pair.values())
        if total_matched == 0:
            return 0.0
        return total_route_mis / total_matched * 100.0

    btc = _pct("BTCUSDT")
    bnb = _pct("BNBUSDT")
    route = _route_mismatch_pct()

    sign_flag = btc > SIGN_MISMATCH_WARN or bnb > SIGN_MISMATCH_WARN
    route_flag = route > ROUTE_MISMATCH_WARN

    return {
        "no_data": False,
        "btc_sign_mismatch_pct": round(btc, 1),
        "bnb_sign_mismatch_pct": round(bnb, 1),
        "route_mismatch_pct": round(route, 1),
        "sign_flag": sign_flag,
        "route_flag": route_flag,
    }


def compute_live_win_rate(days: int = 30) -> dict:
    """
    Compute daily win-rate from the last `days` calendar days of trades in
    logs/pairwise_regime_decisions.jsonl.

    A 'trade' here is a bar where |signal_pct| > 0 for at least one pair.
    A 'win' is a bar where the mean signal_pct across active pairs > 0
    AND the subsequent close-to-close return for those pairs is positive —
    approximated by checking the 'pnl_pct' field if present, otherwise
    treating non-zero as unknown (skip).
    """
    if not DECISION_LOG.exists():
        return {"win_rate": None, "n_trades": 0, "flag": False, "note": "no decision log"}

    cutoff = datetime.now(UTC) - timedelta(days=days)
    records = []
    try:
        with DECISION_LOG.open() as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                at_str = rec.get("at", "")
                if not at_str:
                    continue
                try:
                    at = datetime.fromisoformat(at_str)
                except ValueError:
                    continue
                if at.tzinfo is None:
                    at = at.replace(tzinfo=UTC)
                if at < cutoff:
                    continue
                records.append(rec)
    except OSError as exc:
        return {"win_rate": None, "n_trades": 0, "flag": False, "note": str(exc)}

    if not records:
        return {"win_rate": None, "n_trades": 0, "flag": False, "note": "no records in window"}

    # Group by calendar day and check if day was net-positive.
    # Use pnl_pct if available; fall back to signal direction proxy.
    daily_wins = 0
    daily_losses = 0

    from collections import defaultdict
    by_day: dict[str, list] = defaultdict(list)
    for rec in records:
        at_str = rec.get("at", "")
        try:
            at = datetime.fromisoformat(at_str)
        except ValueError:
            continue
        if at.tzinfo is None:
            at = at.replace(tzinfo=UTC)
        day_key = at.strftime("%Y-%m-%d")
        by_day[day_key].append(rec)

    for _day, day_recs in by_day.items():
        pnls = []
        for rec in day_recs:
            pnl = rec.get("pnl_pct")
            if pnl is not None:
                pnls.append(float(pnl))
        if not pnls:
            # No pnl_pct field — skip day (can't evaluate)
            continue
        if sum(pnls) > 0:
            daily_wins += 1
        else:
            daily_losses += 1

    n_evaluated = daily_wins + daily_losses
    if n_evaluated == 0:
        # pnl_pct not in decision log — fall back to the freshly-regenerated
        # live_actual_pnl_30d report. main() invokes run_live_pnl_refresh()
        # before this function so the file should be < LIVE_PNL_MAX_AGE_SECONDS old.
        live_pnl_path = LIVE_PNL_REPORT
        if live_pnl_path.exists():
            age_seconds = time.time() - live_pnl_path.stat().st_mtime
            if age_seconds > LIVE_PNL_MAX_AGE_SECONDS:
                return {
                    "win_rate": None,
                    "n_trades": len(records),
                    "n_days_evaluated": 0,
                    "flag": False,
                    "unevaluated": True,
                    "note": f"live_actual_pnl_30d.json stale ({age_seconds/3600:.1f}h old)",
                    "error": f"win-rate fallback report is stale ({age_seconds/3600:.1f}h > {LIVE_PNL_MAX_AGE_SECONDS/3600:.1f}h max)",
                }
            try:
                live_pnl = json.loads(live_pnl_path.read_text())
                daily_rows = live_pnl.get("daily_pnl_live") or []
                # Rows are per (date, pair); aggregate to per-date totals so we count
                # calendar days, not pair-days. Codex 13th-round fix.
                from collections import defaultdict
                per_date_total: dict[str, float] = defaultdict(float)
                for r in daily_rows:
                    date_key = r.get("date")
                    if not date_key:
                        continue
                    per_date_total[date_key] += float(r.get("total", 0.0))
                wins = sum(1 for total in per_date_total.values() if total > 0.0)
                losses = sum(1 for total in per_date_total.values() if total < 0.0)
                fallback_evaluated = wins + losses
                if fallback_evaluated > 0:
                    fb_win_rate = wins / fallback_evaluated * 100.0
                    return {
                        "win_rate": round(fb_win_rate, 1),
                        "n_trades": len(records),
                        "n_days_evaluated": fallback_evaluated,
                        "flag": fb_win_rate < WIN_RATE_WARN,
                        "note": (
                            f"pnl_pct missing in decisions; live_actual_pnl_30d fallback "
                            f"({fallback_evaluated} dates aggregated from {len(daily_rows)} pair-rows, "
                            f"age={age_seconds/60:.0f}min)"
                        ),
                    }
            except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
                return {
                    "win_rate": None,
                    "n_trades": len(records),
                    "n_days_evaluated": 0,
                    "flag": False,
                    "unevaluated": True,
                    "note": f"fallback parse error: {exc}",
                    "error": f"win-rate fallback parse error: {exc}",
                }
        # No usable pnl source -> mark unevaluated so the verdict gate flags it.
        return {
            "win_rate": None,
            "n_trades": len(records),
            "n_days_evaluated": 0,
            "flag": False,
            "unevaluated": True,
            "note": "pnl_pct not in log AND fallback report not present",
            "error": "win_rate could not be computed: pnl_pct absent and live_actual_pnl_30d.json missing",
        }

    win_rate = daily_wins / n_evaluated * 100.0
    return {
        "win_rate": round(win_rate, 1),
        "n_trades": len(records),
        "n_days_evaluated": n_evaluated,
        "flag": win_rate < WIN_RATE_WARN,
        "note": "",
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def determine_verdict(
    wf: dict,
    recon: dict,
    wr: dict,
    wf_rc: int = 0,
    recon_rc: int = 0,
) -> tuple[str, list[str]]:
    flags: list[str] = []
    error_flags: list[str] = []

    if wf_rc != 0:
        error_flags.append(f"walkforward subprocess failed (exit={wf_rc})")
    elif not wf:
        error_flags.append("walkforward report missing or empty")
    elif wf.get("error"):
        error_flags.append(f"walkforward error: {wf['error']}")

    if recon_rc != 0:
        error_flags.append(f"reconciliation subprocess failed (exit={recon_rc})")
    elif not recon:
        error_flags.append("reconciliation report missing or empty")
    elif recon.get("error"):
        error_flags.append(f"reconciliation error: {recon['error']}")

    if wr.get("error"):
        error_flags.append(f"win-rate computation error: {wr['error']}")

    if wf.get("deflation_flag"):
        flags.append(
            f"WF Sharpe deflation >50%: latest={wf['latest_4_mean_sharpe']:.2f}"
            f" prior={wf['prior_4_mean_sharpe']:.2f}"
        )

    if recon.get("sign_flag"):
        flags.append(
            f"Sign mismatch: BTC={recon['btc_sign_mismatch_pct']:.1f}%"
            f" BNB={recon['bnb_sign_mismatch_pct']:.1f}% (warn>{SIGN_MISMATCH_WARN}%)"
        )

    if recon.get("route_flag"):
        flags.append(
            f"Route mismatch: {recon['route_mismatch_pct']:.1f}%"
            f" (warn>{ROUTE_MISMATCH_WARN}%)"
        )

    if wr.get("flag"):
        flags.append(
            f"Daily win-rate low: {wr.get('win_rate', 'N/A')}%"
            f" (warn<{WIN_RATE_WARN}%)"
        )

    flags = [*error_flags, *flags]

    if error_flags:
        verdict = "CRITICAL"
    elif not flags:
        verdict = "OK"
    elif len(flags) >= 2 or wf.get("deflation_flag"):
        verdict = "CRITICAL"
    else:
        verdict = "WARN"

    return verdict, flags


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_message(
    today: str,
    verdict: str,
    wf: dict,
    recon: dict,
    wr: dict,
    flags: list[str],
    wf_rc: int,
    recon_rc: int,
) -> str:
    lines = [f"[Weekly Validation {today}]"]
    lines.append(f"- Verdict: {verdict}")

    # Walk-forward
    if wf_rc != 0:
        lines.append("- WF Sharpe: ERROR (walkforward failed)")
    else:
        delta_sign = "+" if wf["delta"] >= 0 else ""
        lines.append(
            f"- WF Sharpe latest 4 folds: {wf['latest_4_mean_sharpe']:.2f}"
            f" (delta vs prior: {delta_sign}{wf['delta']:.2f},"
            f" {wf['n_folds']} total folds)"
        )

    # Drift
    if recon_rc != 0:
        lines.append("- Live drift 7d: ERROR (reconciliation failed)")
    elif recon.get("no_data"):
        lines.append("- Live drift 7d: no live data")
    else:
        lines.append(
            f"- Live drift 7d:"
            f" BTC sign mismatch {recon['btc_sign_mismatch_pct']:.1f}%,"
            f" BNB {recon['bnb_sign_mismatch_pct']:.1f}%,"
            f" route mismatch {recon['route_mismatch_pct']:.1f}%"
        )

    # Win rate
    if wr.get("win_rate") is not None:
        lines.append(
            f"- Last 30d live daily WR: {wr['win_rate']:.1f}%"
            f" ({wr['n_days_evaluated']} days evaluated, {wr['n_trades']} decisions)"
        )
    else:
        note = wr.get("note", "")
        lines.append(f"- Last 30d live daily WR: N/A ({note}, {wr.get('n_trades', 0)} decisions)")

    # Flags
    if flags:
        lines.append("- Flags:")
        for f in flags:
            lines.append(f"  * {f}")
    else:
        lines.append("- Flags: none")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Weekly self-validation: walk-forward + drift + win-rate + Telegram alert."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Telegram message to stdout instead of sending it.",
    )
    parser.add_argument(
        "--days-lookback",
        type=int,
        default=365,
        help="Walk-forward start = today minus this many days (default 365).",
    )
    parser.add_argument(
        "--recon-days",
        type=int,
        default=7,
        help="Reconciliation lookback window in days (default 7).",
    )
    parser.add_argument(
        "--win-rate-days",
        type=int,
        default=30,
        help="Live win-rate lookback in days (default 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    wf_start = (datetime.now(UTC) - timedelta(days=args.days_lookback)).strftime("%Y-%m-%d")

    # Validate env vars before doing any work.
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not args.dry_run:
        missing = []
        if not token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not chat_id:
            missing.append("TELEGRAM_CHAT_ID")
        if missing:
            print(
                f"[weekly_validation] ERROR: missing env vars: {', '.join(missing)}. "
                "Set them in .env or environment before running.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Walk-forward ---
    wf_rc = run_walkforward(wf_start, today)
    wf_report = _load_json(WF_REPORT) or {}
    wf = analyse_walkforward(wf_report)

    # --- Reconciliation ---
    recon_rc = run_reconciliation(args.recon_days)
    recon_report = _load_json(RECON_REPORT) or {}
    recon = analyse_reconciliation(recon_report)

    # --- Live actual P&L refresh (must precede win-rate compute) ---
    live_pnl_rc = run_live_pnl_refresh(args.win_rate_days)

    # --- Live win-rate ---
    wr = compute_live_win_rate(args.win_rate_days)
    if live_pnl_rc != 0 and wr.get("error") is None:
        # Surface refresh failure even if win-rate happens to be parseable from the stale file.
        wr = {**wr, "error": f"live_actual_pnl_30d refresh failed (exit={live_pnl_rc}); win-rate may be stale"}

    # --- Verdict ---
    verdict, flags = determine_verdict(wf, recon, wr, wf_rc=wf_rc, recon_rc=recon_rc)

    # --- Message ---
    message = build_message(today, verdict, wf, recon, wr, flags, wf_rc, recon_rc)

    print("\n" + "=" * 60)
    print(message)
    print("=" * 60)

    if args.dry_run:
        print("\n[dry-run] Message above would be sent to Telegram.")
    else:
        print(f"\n[weekly_validation] Sending Telegram message (verdict={verdict}) ...")
        send_telegram(token, chat_id, message)
        print("[weekly_validation] Done.")


if __name__ == "__main__":
    main()
