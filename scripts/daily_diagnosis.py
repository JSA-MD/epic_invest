#!/usr/bin/env python3
"""Daily diagnosis — Prompt 1 of the 1%-target prompt suite.

Decomposes yesterday's live trader behaviour into the actionable signals
the operator needs to decide between hold / tighten_cap / loosen_cap /
recertify / escalate. Designed to run unattended via launchd at KST 06:00
on every trading day.

The decomposition follows the 1%-day-target gap analysis:

    gap_to_target_bps = 100 (1% target) - live_pnl_pct_of_base * 100

Each suspicious bar is classified into one of:

    suppressed       — signal != 0 but final target_weight ≈ 0 (gate killed it)
    binding_overlay  — an overlay (CVaR_CUT, MAX_HOLD, SIGN_INSTABILITY,
                       STALE_PRICE, PROMOTION_FREEZE) forced flat
    sign_flip        — sign changed from previous bar within same hour
    sized_off        — |live target| differs from prior backtest expectation
                       by more than 10× (apples-to-oranges)

The output JSON is the canonical input to Prompt 6 (postmortem_drift)
and Prompt 7 (weekly_iteration).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

log = logging.getLogger("daily_diagnosis")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)


DEFAULT_DECISION_LOG = ROOT / "logs" / "pairwise_regime_decisions.jsonl"
DEFAULT_LIVE_PNL_PATH = ROOT / "models" / "live_actual_pnl_30d.json"
DEFAULT_DRIFT_PATH = ROOT / "models" / "live_vs_backtest_same_window.json"
DEFAULT_OUTPUT_DIR = ROOT / "models"

DAILY_TARGET_PCT = 1.0  # 1% per day


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def yesterday_utc_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    now = now or utc_now()
    end = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)
    return start, end


def _load_jsonl(path: Path, since: datetime, until: datetime) -> list[dict]:
    """Read JSONL entries whose `at` field falls in [since, until)."""
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_raw = row.get("at") or row.get("timestamp")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if since <= ts < until:
                row["_ts"] = ts
                out.append(row)
    return out


def _classify_suppression(row: dict) -> str | None:
    """Identify which overlay (if any) zeroed an otherwise-non-flat signal."""
    plan = row.get("plan") or {}
    pair_plans = plan.get("pair_plans") or {}
    journal_overrides = {}
    for entry in row.get("decision_journal") or []:
        pair = entry.get("pair")
        if pair:
            journal_overrides.setdefault(pair, []).append(entry.get("override_reason"))
    suppressed_pairs: list[str] = []
    for pair, pp in pair_plans.items():
        signal_pct = float(pp.get("signal_pct") or 0.0)
        target = float(pp.get("target_weight") or 0.0)
        if abs(signal_pct) > 10.0 and abs(target) < 1e-6:
            reasons = journal_overrides.get(pair, [])
            tag = ",".join(filter(None, reasons)) if reasons else "gate"
            suppressed_pairs.append(f"{pair}:{tag}")
    return ";".join(suppressed_pairs) if suppressed_pairs else None


def _sign(x: float, eps: float = 1e-6) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _signal_decomposition(rows: list[dict]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n_bars": 0,
            "n_flat": 0,
            "n_suppressed": 0,
            "binding_overlay_counts": {},
            "sign_flip_count": 0,
            "silently_dropped_signals": 0,
        }
    n_flat = 0
    n_suppressed = 0
    overlays: Counter = Counter()
    sign_flips = 0
    prev_sign: dict[str, int] = {}
    silently_dropped = 0
    for row in rows:
        plan = row.get("plan") or {}
        if (plan.get("session_type") or "").lower() == "flat":
            n_flat += 1
        suppression = _classify_suppression(row)
        if suppression:
            n_suppressed += 1
            for tok in suppression.split(";"):
                _, _, tag = tok.partition(":")
                overlays[tag or "gate"] += 1
        for pair, pp in (plan.get("pair_plans") or {}).items():
            sgn = _sign(float(pp.get("target_weight") or 0.0))
            prev = prev_sign.get(pair, 0)
            if prev != 0 and sgn != 0 and prev != sgn:
                sign_flips += 1
            if sgn != 0:
                prev_sign[pair] = sgn
            sig_pct = float(pp.get("signal_pct") or 0.0)
            if abs(sig_pct) > 10.0 and sgn == 0:
                silently_dropped += 1
    return {
        "n_bars": n,
        "n_flat": n_flat,
        "n_suppressed": n_suppressed,
        "binding_overlay_counts": dict(overlays.most_common()),
        "sign_flip_count": sign_flips,
        "silently_dropped_signals": silently_dropped,
    }


def _live_pnl_for_window(start: datetime, end: datetime, source: dict) -> dict[str, Any]:
    """Pull total live PnL between [start, end) from live_actual_pnl_30d.json."""
    daily = source.get("daily_pnl_live") or []
    target_date = start.date().isoformat()
    pair_pnls = [r for r in daily if r.get("date") == target_date]
    if not pair_pnls:
        return {"date": target_date, "total_usd": 0.0, "data_missing": True}
    total = sum(float(r.get("total") or 0.0) for r in pair_pnls)
    base = float(source.get("initial_equity_estimate") or 0.0) or 1.0
    return {
        "date": target_date,
        "total_usd": total,
        "base_notional_usd": base,
        "pct_of_base": (total / base) * 100.0,
        "data_missing": all(r.get("data_missing") for r in pair_pnls),
        "attribution_method": pair_pnls[0].get("attribution_method"),
    }


def _drift_for_window(start: datetime, source: dict) -> dict[str, Any]:
    target_date = start.date().isoformat()
    rows = source.get("per_date_per_pair") or []
    same_day = [r for r in rows if r.get("date") == target_date]
    if not same_day:
        return {"date": target_date, "n_pair_rows": 0}
    diffs = [float(r.get("diff_bps_of_base") or 0.0) for r in same_day]
    return {
        "date": target_date,
        "n_pair_rows": len(same_day),
        "max_abs_diff_bps": max(abs(d) for d in diffs) if diffs else 0.0,
        "sum_diff_bps": sum(diffs),
        "rows": same_day,
    }


def _hypothesis(signal_decomp: dict, drift: dict, pnl: dict) -> str:
    pct = pnl.get("pct_of_base", 0.0) or 0.0
    n_flat = signal_decomp.get("n_flat", 0)
    n_bars = signal_decomp.get("n_bars", 0) or 1
    flat_rate = n_flat / n_bars
    suppressed = signal_decomp.get("silently_dropped_signals", 0)
    overlays = signal_decomp.get("binding_overlay_counts", {})
    if pct < -0.5 and drift.get("max_abs_diff_bps", 0.0) > 200.0:
        return (
            f"Live drift dominated by sizing/timing mismatch "
            f"(max_drift={drift['max_abs_diff_bps']:.0f}bps); investigate magnitude ratio."
        )
    if flat_rate > 0.6 and suppressed > 5:
        top = next(iter(overlays.keys()), "gate")
        return (
            f"Signal alive but execution suppressed on {suppressed} bars "
            f"({flat_rate:.0%} flat session); binding overlay={top}."
        )
    if abs(pct) < 0.05 and flat_rate > 0.9:
        return "All bars flat — promotion gate likely closed (recertification or freeze)."
    if pct >= DAILY_TARGET_PCT:
        return "Hit 1% target today."
    return f"Below target by {DAILY_TARGET_PCT - pct:.2f}%; no single dominant cause."


def _action(signal_decomp: dict, drift: dict, pnl: dict, days_negative: int) -> str:
    if days_negative >= 5:
        return "escalate"
    pct = pnl.get("pct_of_base", 0.0) or 0.0
    n_flat = signal_decomp.get("n_flat", 0)
    n_bars = signal_decomp.get("n_bars", 0) or 1
    if drift.get("max_abs_diff_bps", 0.0) > 300.0:
        return "tighten_cap"
    if signal_decomp.get("silently_dropped_signals", 0) > 10 and (n_flat / n_bars) > 0.6:
        return "recertify"
    if pct < -0.5 and -drift.get("sum_diff_bps", 0.0) > 100:
        return "tighten_cap"
    if pct < -2.0:
        return "escalate"
    return "hold"


def _consecutive_negative_days(daily: list[dict]) -> int:
    streak = 0
    for r in reversed(daily):
        if (r.get("total") or 0.0) < 0:
            streak += 1
        elif (r.get("total") or 0.0) > 0:
            break
        else:
            continue  # zero PnL = ignore (no trade)
    return streak


def diagnose(
    *,
    decision_log: Path,
    live_pnl_path: Path,
    drift_path: Path,
    target_pct: float = DAILY_TARGET_PCT,
    now: datetime | None = None,
) -> dict[str, Any]:
    start, end = yesterday_utc_window(now=now)
    log.info("Window: %s -> %s", start.isoformat(), end.isoformat())

    rows = _load_jsonl(decision_log, start, end)
    log.info("Decision-log rows in window: %d", len(rows))

    pnl_src = json.loads(live_pnl_path.read_text()) if live_pnl_path.exists() else {}
    drift_src = json.loads(drift_path.read_text()) if drift_path.exists() else {}

    signal_decomp = _signal_decomposition(rows)
    pnl = _live_pnl_for_window(start, end, pnl_src)
    drift = _drift_for_window(start, drift_src)
    days_neg = _consecutive_negative_days(pnl_src.get("daily_pnl_live") or [])

    pct = pnl.get("pct_of_base", 0.0) or 0.0
    gap_to_target_bps = (target_pct - pct) * 100.0

    report = {
        "generated_at": utc_now().isoformat(),
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "target_pct": target_pct,
        "pnl": pnl,
        "gap_to_target_bps": gap_to_target_bps,
        "drift": drift,
        "signal_decomposition": signal_decomp,
        "consecutive_negative_days": days_neg,
        "root_cause_hypothesis": _hypothesis(signal_decomp, drift, pnl),
        "recommended_action": _action(signal_decomp, drift, pnl, days_neg),
    }
    return report


def telegram_summary(report: dict) -> str:
    pnl = report.get("pnl") or {}
    decomp = report.get("signal_decomposition") or {}
    overlays = decomp.get("binding_overlay_counts") or {}
    overlays_str = ", ".join(f"{k}:{v}" for k, v in list(overlays.items())[:3]) or "none"
    return (
        f"[daily-diagnosis] {pnl.get('date','?')}\n"
        f"PnL: {pnl.get('pct_of_base', 0.0):+.2f}% | "
        f"Gap to target: {report.get('gap_to_target_bps', 0.0):.0f}bps\n"
        f"Bars: {decomp.get('n_bars',0)} (flat {decomp.get('n_flat',0)}, "
        f"suppressed {decomp.get('n_suppressed',0)}, sign-flip {decomp.get('sign_flip_count',0)})\n"
        f"Overlays: {overlays_str}\n"
        f"Hypothesis: {report.get('root_cause_hypothesis','-')}\n"
        f"Action: {report.get('recommended_action','-')}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Daily diagnosis — Prompt 1 of the 1%-suite")
    p.add_argument("--decision-log", type=Path, default=DEFAULT_DECISION_LOG)
    p.add_argument("--live-pnl", type=Path, default=DEFAULT_LIVE_PNL_PATH)
    p.add_argument("--drift", type=Path, default=DEFAULT_DRIFT_PATH)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-pct", type=float, default=DAILY_TARGET_PCT)
    p.add_argument("--telegram", action="store_true",
                   default=os.getenv("DAILY_DIAGNOSIS_TELEGRAM", "1").strip().lower()
                   in {"1", "true", "yes", "on"})
    return p


def main() -> int:
    args = build_parser().parse_args()
    report = diagnose(
        decision_log=args.decision_log,
        live_pnl_path=args.live_pnl,
        drift_path=args.drift,
        target_pct=args.target_pct,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"daily_diagnosis_{report['pnl'].get('date', 'unknown')}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", out_path)
    print(json.dumps(report, indent=2, default=str))
    if args.telegram:
        try:
            from notification_bridge import load_notification_bridge
            bridge = load_notification_bridge()
            bridge.send_telegram_notification(telegram_summary(report))
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram dispatch skipped: %s", exc)
    return 0 if report["recommended_action"] in {"hold", "promote"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
