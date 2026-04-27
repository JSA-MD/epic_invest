"""Execution-trigger monitor.

Detects bars where the signal *would* take a non-flat position but no
matching exchange fill landed. This is the canonical "live trade
silently dropped" failure mode the operator reported.

A bar is classified as a *silently dropped signal* when ALL of:

    abs(signal_pct) > MIN_SIGNAL_PCT   (default 10 — same as the
                                        Stage 0 daily_diagnosis cutoff)
    abs(target_weight) <= TARGET_WEIGHT_EPS  (≈ flat)
    overlay_force_flat is None         (no overlay claimed credit)

The third condition is what makes this monitor different from the
existing flat-session counter: it catches *unattributed* drops — the
ones that no D-overlay or Stage 2 overlay accounts for. Those are the
ones the operator cannot find in the journal and that look like "the
exchange just didn't take my order".

Output: writes `models/execution_trigger_report_<UTC>.json` and, when
the rate exceeds `EXECUTION_DROP_ALERT_THRESHOLD` (default 5 bars in
24 h), fires a HIGH Telegram alert via `notification_bridge`.
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

log = logging.getLogger("execution_trigger_monitor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)


DEFAULT_DECISION_LOG = ROOT / "logs" / "pairwise_regime_decisions.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "models"
MIN_SIGNAL_PCT = 10.0
TARGET_WEIGHT_EPS = 1e-6
DEFAULT_ALERT_THRESHOLD = 5
DEFAULT_WINDOW_HOURS = 24


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _is_silently_dropped(
    pp: dict[str, Any],
    journal_for_pair: list[dict[str, Any]] | None = None,
) -> bool:
    """Return True only when nothing accounts for the zeroed target weight.

    A bar is *not* silently dropped when any of the following hold:
    - the signal itself was tiny (gate respected an honest noise signal),
    - the realised target_weight is non-flat (no drop happened),
    - the new live_overlay_runner stamped pp["overlay_force_flat"]
      with the binding overlay reason, or
    - the legacy overlays (D1 max_hold, R3 CVaR cut, stale-price guard)
      appended an entry to the same bar's decision_journal — those
      overlays mutate plan target_weights but only record their reason
      in the journal, not on the pair_plan itself.

    Without the journal check the monitor would mis-classify every
    legitimate Stage 0 overlay firing as a silent drop and burn down
    the operator's alert budget.
    """
    sig = pp.get("signal_pct")
    tw = pp.get("target_weight")
    if sig is None or tw is None:
        return False
    try:
        sig_v = float(sig)
        tw_v = float(tw)
    except (TypeError, ValueError):
        return False
    if abs(sig_v) <= MIN_SIGNAL_PCT:
        return False
    if abs(tw_v) > TARGET_WEIGHT_EPS:
        return False
    if pp.get("overlay_force_flat"):  # new live_overlay_runner stamp
        return False
    if journal_for_pair:
        for entry in journal_for_pair:
            if entry.get("override_reason"):
                return False
    return True


def scan(
    decision_log: Path,
    *,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or _utc_now()
    since = now - timedelta(hours=window_hours)
    log.info("Scanning %s for window [%s, %s)", decision_log, since.isoformat(), now.isoformat())

    if not decision_log.exists():
        return {
            "generated_at": now.isoformat(),
            "window_hours": window_hours,
            "decision_log_exists": False,
            "n_bars": 0,
            "n_silent_drops": 0,
            "drops_by_pair": {},
            "samples": [],
        }

    n_bars = 0
    silent_drops_by_pair: Counter = Counter()
    samples: list[dict] = []

    with decision_log.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = _parse_ts(row.get("at") or row.get("timestamp"))
            if ts is None or ts < since or ts > now:
                continue
            n_bars += 1
            plan = row.get("plan") or {}
            # Group decision_journal entries by pair so the silent-drop
            # classifier can attribute legacy-overlay-fired flats correctly.
            journal_by_pair: dict[str, list[dict[str, Any]]] = {}
            for entry in row.get("decision_journal") or []:
                p = entry.get("pair")
                if p is not None:
                    journal_by_pair.setdefault(str(p), []).append(entry)
            for pair, pp in (plan.get("pair_plans") or {}).items():
                if _is_silently_dropped(pp, journal_by_pair.get(pair)):
                    silent_drops_by_pair[pair] += 1
                    if len(samples) < 20:
                        samples.append(
                            {
                                "at": ts.isoformat(),
                                "pair": pair,
                                "signal_pct": pp.get("signal_pct"),
                                "target_weight": pp.get("target_weight"),
                                "regime_score": pp.get("regime_score"),
                                "route_state_name": pp.get("route_state_name"),
                            }
                        )

    total_drops = sum(silent_drops_by_pair.values())
    return {
        "generated_at": now.isoformat(),
        "window_hours": window_hours,
        "decision_log_exists": True,
        "n_bars": n_bars,
        "n_silent_drops": total_drops,
        "drops_by_pair": dict(silent_drops_by_pair),
        "samples": samples,
    }


def alert_text(report: dict[str, Any]) -> str:
    by_pair = ", ".join(f"{p}:{n}" for p, n in (report.get("drops_by_pair") or {}).items()) or "-"
    return (
        f"[execution-trigger] {report['generated_at']}\n"
        f"Window: {report['window_hours']}h | Bars: {report['n_bars']} | "
        f"Silently dropped: {report['n_silent_drops']}\n"
        f"By pair: {by_pair}\n"
        f"Sample: {json.dumps(report['samples'][:1], default=str) if report.get('samples') else 'none'}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Execution-trigger monitor")
    p.add_argument("--decision-log", type=Path, default=DEFAULT_DECISION_LOG)
    p.add_argument("--window-hours", type=int, default=DEFAULT_WINDOW_HOURS)
    p.add_argument(
        "--alert-threshold",
        type=int,
        default=int(os.getenv("EXECUTION_DROP_ALERT_THRESHOLD", DEFAULT_ALERT_THRESHOLD)),
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--telegram",
        action="store_true",
        default=os.getenv("EXECUTION_TRIGGER_TELEGRAM", "1").strip().lower()
        in {"1", "true", "yes", "on"},
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    report = scan(args.decision_log, window_hours=args.window_hours)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_dir / f"execution_trigger_report_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log.info("Wrote %s", out_path)
    print(json.dumps(report, indent=2, default=str))
    if report["n_silent_drops"] >= args.alert_threshold and args.telegram:
        try:
            from notification_bridge import load_notification_bridge

            bridge = load_notification_bridge()
            bridge.send_telegram_notification(alert_text(report))
        except Exception as exc:  # noqa: BLE001
            log.warning("telegram dispatch skipped: %s", exc)
    return 0 if report["n_silent_drops"] < args.alert_threshold else 2


if __name__ == "__main__":
    raise SystemExit(main())
