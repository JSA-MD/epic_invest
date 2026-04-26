"""Slippage calibration: compare live-measured slippage against backtest assumptions.

Usage
-----
    python scripts/calibrate_slippage.py [--days 7] [--log PATH] [--alert] [--output PATH]

Reads logs/pairwise_slippage.jsonl (written by P1-8), computes per-symbol
percentile statistics, compares against the flat backtest assumption (2 bp),
and emits a human-readable report plus a JSON artefact for downstream use.

Alert threshold: P75 > 1.5 × backtest_assumption triggers ALERT.
Optional --alert flag sends the report via Telegram.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOG = _ROOT / "logs" / "pairwise_slippage.jsonl"
_DEFAULT_OUTPUT_DIR = _ROOT / "models"

# Backtest flat-slippage assumption in basis points (2 bp = 0.0002 fractional)
BACKTEST_ASSUMPTION_BPS: float = 2.0

# P75 / assumption ratio that triggers an ALERT
ALERT_RATIO_THRESHOLD: float = 1.5

# ---------------------------------------------------------------------------
# Telegram helpers (self-contained; no import from other scripts)
# ---------------------------------------------------------------------------
_TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
_TELEGRAM_NOTIFICATIONS_ENABLED: bool = os.getenv(
    "TELEGRAM_NOTIFICATIONS_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}


def _resolve_telegram_chat_ids() -> list[int]:
    values: list[int] = []
    for raw in [
        os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", ""),
        os.getenv("TELEGRAM_CHAT_ID", ""),
    ]:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(int(item))
            except ValueError:
                continue
    seen: set[int] = set()
    deduped: list[int] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


_TELEGRAM_CHAT_IDS: list[int] = _resolve_telegram_chat_ids()


def _telegram_ready() -> bool:
    return (
        _TELEGRAM_NOTIFICATIONS_ENABLED
        and bool(_TELEGRAM_BOT_TOKEN)
        and bool(_TELEGRAM_CHAT_IDS)
    )


def send_telegram_notification(text: str) -> bool:
    """Send *text* to all configured Telegram chat IDs.

    Returns True if at least one delivery succeeded.
    """
    if not _telegram_ready():
        return False
    payload_text = text.strip()
    if not payload_text:
        return False
    delivered = False
    for chat_id in _TELEGRAM_CHAT_IDS:
        params = urlencode(
            {
                "chat_id": str(chat_id),
                "text": payload_text,
                "disable_web_page_preview": "true",
            }
        ).encode()
        request = Request(
            f"https://api.telegram.org/bot{_TELEGRAM_BOT_TOKEN}/sendMessage",
            data=params,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            if not data.get("ok"):
                print(
                    f"[WARN] Telegram send failed for chat_id={chat_id}: {data}",
                    file=sys.stderr,
                )
                continue
            delivered = True
        except Exception as exc:  # noqa: BLE001
            print(
                f"[WARN] Telegram exception for chat_id={chat_id}: {exc}",
                file=sys.stderr,
            )
    return delivered


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_slippage_records(
    log_path: Path,
    cutoff_ts: datetime,
) -> list[dict[str, Any]]:
    """Return all JSONL records whose ``ts`` field is >= *cutoff_ts*."""
    records: list[dict[str, Any]] = []
    if not log_path.exists():
        return records
    with log_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_raw = rec.get("ts", "")
            if not ts_raw:
                continue
            try:
                # Parse ISO-8601 with or without timezone offset
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if ts >= cutoff_ts:
                records.append(rec)
    return records


def compute_symbol_stats(
    records: list[dict[str, Any]],
    mode_filter: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate per-symbol percentile stats.

    Parameters
    ----------
    records:
        Filtered slippage records (already within the time window).
    mode_filter:
        If provided, only include records with ``mode == mode_filter``.

    Returns
    -------
    Dict keyed by canonical symbol (e.g. ``"BTCUSDT"``) with sub-keys:
    n, mean, std, p50, p75, p90, p99.
    """
    by_symbol: dict[str, list[float]] = {}
    for rec in records:
        if mode_filter and rec.get("mode") != mode_filter:
            continue
        bps = rec.get("slippage_bps")
        if bps is None:
            continue
        # Normalise "BTC/USDT:USDT" -> "BTCUSDT"
        sym_raw: str = str(rec.get("symbol", "UNKNOWN"))
        sym = sym_raw.replace("/", "").replace(":", "").replace("USDT", "USDT", 1)
        # Simpler: strip non-alphanum except already clean
        sym = sym_raw.split("/")[0].replace(":", "") + "USDT" if "/" in sym_raw else sym_raw
        by_symbol.setdefault(sym, []).append(float(bps))

    stats: dict[str, dict[str, Any]] = {}
    for sym, values in by_symbol.items():
        arr = np.array(values, dtype=float)
        stats[sym] = {
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }
    return stats


def verdict_for_symbol(
    sym: str,
    stats: dict[str, Any],
    assumption_bps: float,
    alert_ratio: float,
) -> tuple[str, bool]:
    """Return (verdict_line, is_alert) for a single symbol."""
    p75 = stats["p75"]
    ratio = p75 / assumption_bps if assumption_bps > 0 else float("inf")
    pct_diff = (ratio - 1.0) * 100.0

    if ratio < 0.9:
        status = "OVERESTIMATED"
        is_alert = False
        note = f"backtest {assumption_bps:.1f}bp > live P75 ({p75:.1f}bp) — {status} by {abs(pct_diff):.0f}%"
    elif ratio <= 1.2:
        status = "REASONABLE"
        is_alert = False
        note = f"backtest {assumption_bps:.1f}bp ≈ live P75 ({p75:.1f}bp) — {status}"
    elif ratio < alert_ratio:
        status = "UNDERESTIMATED"
        is_alert = False
        note = f"backtest {assumption_bps:.1f}bp < live P75 ({p75:.1f}bp) — {status} by {pct_diff:.0f}%"
    else:
        status = "ALERT"
        is_alert = True
        note = (
            f"backtest {assumption_bps:.1f}bp << live P75 ({p75:.1f}bp) — {status}"
            f" (ratio {ratio:.2f}x > threshold {alert_ratio:.1f}x)\n"
            f"            consider raising to {p75:.1f}bp or volatility-scaled"
        )
    return f"  {sym:<10} {note}", is_alert


def build_report(
    days: int,
    cutoff_ts: datetime,
    now_ts: datetime,
    live_stats: dict[str, dict[str, Any]],
    demo_stats: dict[str, dict[str, Any]],
    live_count: int,
    demo_count: int,
    assumption_bps: float,
    alert_ratio: float,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Return (full_report_text, alert_symbols, json_payload_list)."""
    date_from = cutoff_ts.strftime("%Y-%m-%d")
    date_to = now_ts.strftime("%Y-%m-%d")
    lines: list[str] = [
        f"=== Slippage Calibration Report ({days}d, {date_from} → {date_to}) ===",
        "",
        f"Live trades: {live_count}  |  Demo trades: {demo_count}",
        "",
    ]

    json_rows: list[dict[str, Any]] = []
    alert_syms: list[str] = []

    if live_stats:
        lines.append("Per-symbol (live):")
        for sym in sorted(live_stats):
            s = live_stats[sym]
            lines.append(
                f"  {sym:<10} n={s['n']:<5} "
                f"P50={s['p50']:.1f}bp  "
                f"P75={s['p75']:.1f}bp  "
                f"P90={s['p90']:.1f}bp  "
                f"P99={s['p99']:.1f}bp"
            )
        lines.append("")
    else:
        lines.append("Per-symbol (live): (no data)")
        lines.append("")

    lines.append(f"Backtest assumption: {assumption_bps:.1f}bp (flat)")
    lines.append("")
    lines.append("Verdict:")
    for sym in sorted(live_stats):
        vline, is_alert = verdict_for_symbol(sym, live_stats[sym], assumption_bps, alert_ratio)
        lines.append(vline)
        if is_alert:
            alert_syms.append(sym)
        json_rows.append(
            {
                "symbol": sym,
                "mode": "live",
                **live_stats[sym],
                "backtest_assumption_bps": assumption_bps,
                "ratio_p75_vs_assumption": round(live_stats[sym]["p75"] / assumption_bps, 4)
                if assumption_bps > 0
                else None,
                "alert": is_alert,
            }
        )

    if alert_syms:
        lines.append("")
        lines.append("Recommendation:")
        for sym in alert_syms:
            s = live_stats[sym]
            pct = round((s["p75"] / assumption_bps - 1.0) * 100)
            lines.append(
                f"  - {sym} slippage assumption needs +{pct}% adjustment"
            )
        lines.append(
            "  - Or switch to volatility-scaled model"
            " (compute_trade_slippage already exists in"
            " backtest_pairwise_regime_stop_loss_compare.py)"
        )

    report_text = "\n".join(lines)
    return report_text, alert_syms, json_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    days: int,
    log_path: Path,
    alert: bool,
    output_path: Path | None,
    assumption_bps: float = BACKTEST_ASSUMPTION_BPS,
    alert_ratio: float = ALERT_RATIO_THRESHOLD,
    _now: datetime | None = None,
) -> dict[str, Any]:
    """Execute calibration and return the result dict."""
    now_ts = _now or datetime.now(timezone.utc)
    cutoff_ts = now_ts - timedelta(days=days)

    records = load_slippage_records(log_path, cutoff_ts)

    live_records = [r for r in records if r.get("mode") != "demo"]
    demo_records = [r for r in records if r.get("mode") == "demo"]

    live_stats = compute_symbol_stats(live_records)
    demo_stats = compute_symbol_stats(demo_records)

    report_text, alert_syms, json_rows = build_report(
        days=days,
        cutoff_ts=cutoff_ts,
        now_ts=now_ts,
        live_stats=live_stats,
        demo_stats=demo_stats,
        live_count=len(live_records),
        demo_count=len(demo_records),
        assumption_bps=assumption_bps,
        alert_ratio=alert_ratio,
    )

    print(report_text)

    result: dict[str, Any] = {
        "generated_at": iso_now(),
        "days": days,
        "date_from": cutoff_ts.strftime("%Y-%m-%d"),
        "date_to": now_ts.strftime("%Y-%m-%d"),
        "live_count": len(live_records),
        "demo_count": len(demo_records),
        "backtest_assumption_bps": assumption_bps,
        "alert_ratio_threshold": alert_ratio,
        "symbols": json_rows,
        "alert_symbols": alert_syms,
        "has_alert": bool(alert_syms),
    }

    # JSON output
    if output_path is None:
        date_tag = now_ts.strftime("%Y%m%d")
        output_path = _DEFAULT_OUTPUT_DIR / f"slippage_calibration_{date_tag}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[INFO] JSON saved → {output_path}")

    # Telegram alert
    if alert and alert_syms:
        summary = (
            f"[SLIPPAGE ALERT] {', '.join(alert_syms)} P75 > {alert_ratio}x backtest assumption.\n"
            + report_text[:1500]
        )
        sent = send_telegram_notification(summary)
        if sent:
            print("[INFO] Telegram alert sent.")
        else:
            print("[WARN] Telegram alert not delivered (check env vars).", file=sys.stderr)
    elif alert and not alert_syms:
        print("[INFO] --alert: no symbols exceeded threshold; no Telegram notification sent.")

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate live vs backtest slippage assumptions."
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Look-back window in days (default: 7)"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=_DEFAULT_LOG,
        help=f"Path to pairwise_slippage.jsonl (default: {_DEFAULT_LOG})",
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Send Telegram alert when P75 > 1.5x backtest assumption",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: models/slippage_calibration_<date>.json)",
    )
    parser.add_argument(
        "--assumption-bps",
        type=float,
        default=BACKTEST_ASSUMPTION_BPS,
        help=f"Backtest flat slippage assumption in bp (default: {BACKTEST_ASSUMPTION_BPS})",
    )
    parser.add_argument(
        "--alert-ratio",
        type=float,
        default=ALERT_RATIO_THRESHOLD,
        help=f"Alert when P75 / assumption > this ratio (default: {ALERT_RATIO_THRESHOLD})",
    )
    args = parser.parse_args(argv)
    run(
        days=args.days,
        log_path=args.log,
        alert=args.alert,
        output_path=args.output,
        assumption_bps=args.assumption_bps,
        alert_ratio=args.alert_ratio,
    )


if __name__ == "__main__":
    main()
