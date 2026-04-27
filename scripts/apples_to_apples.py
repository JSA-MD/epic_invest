"""Apples-to-apples live↔backtest parity checker (Stage 2 — Chan).

The 31-day live drift report (`docs/live_drift_diagnostic_20260426.md`)
attributes 97.6% of the −$574 gap to position-sizing and holding-period
mismatch between live and backtest. The root cause is that backtest and
live read different runtime knobs:

| knob                          | live had | backtest had |
|-------------------------------|----------|--------------|
| `PAIRWISE_GROSS_CAP`          | 1.5      | implicit ~0.01–0.03 |
| `PAIRWISE_LIVE_MAX_GROSS_CAP` | 1.0      | not enforced |
| `PAIRWISE_MAX_HOLD_BARS`      | 0        | 288 |
| slippage / fill model         | demo exchange | none (immediate) |

This module hard-codes the parity contract: any backtest report that
will be used for promotion *must* declare its runtime environment, and
that environment *must* match the live env on every key listed in
`PARITY_REQUIRED_KEYS`. A divergence aborts promotion the same way a
failed CPCV test does.

Use the CLI to verify a backtest report against the running live env:

    python scripts/apples_to_apples.py \\
        --backtest models/live_vs_backtest_same_window.json \\
        --output models/apples_to_apples_report.json

The exit code is 0 when parity holds, 2 otherwise. The Stage 1 promotion
gate guard already refuses promotion when the recertification report is
not green; downstream tooling can call `check_runtime_parity()` to add
parity to that gate without code in the hot live loop.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger("apples_to_apples")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


# Keys whose value must match between live and backtest.
PARITY_REQUIRED_KEYS: tuple[str, ...] = (
    "PAIRWISE_GROSS_CAP",
    "PAIRWISE_LIVE_MAX_GROSS_CAP",
    "PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP",
    "PAIRWISE_NO_TRADE_BAND_PCT",
    "REBALANCE_NOTIONAL_BAND_USD",
    "PAIRWISE_MAX_HOLD_BARS",
    "PAIRWISE_CVAR_CUT",
    "PAIRWISE_RUNTIME_BLEND",
)

# Keys we expect to see and warn-only if absent (not hard fail).
PARITY_ADVISORY_KEYS: tuple[str, ...] = (
    "EPIC_MARKET_DATA_SOURCE",
    "PAIRWISE_BREADTH_NOISE_EPSILON",
    "PAIRWISE_EQUITY_CORR_RISK",
)


@dataclass(frozen=True)
class ParityIssue:
    key: str
    live_value: str
    backtest_value: str
    severity: str  # "error" or "warning"


def _normalise(value: Any) -> str:
    """Canonical string form for comparison: trims whitespace, lowercases booleans."""
    if value is None:
        return "<missing>"
    s = str(value).strip()
    lower = s.lower()
    if lower in {"true", "yes", "on"}:
        return "1"
    if lower in {"false", "no", "off"}:
        return "0"
    # Numeric trim (avoid "1.0" vs "1" mismatch)
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.10g}"
    except ValueError:
        return s


def check_runtime_parity(
    live_env: Mapping[str, Any],
    backtest_env: Mapping[str, Any],
    *,
    required_keys: tuple[str, ...] = PARITY_REQUIRED_KEYS,
    advisory_keys: tuple[str, ...] = PARITY_ADVISORY_KEYS,
) -> list[ParityIssue]:
    """Compare two env dicts and return a list of parity issues."""
    issues: list[ParityIssue] = []
    for key in required_keys:
        live_v = _normalise(live_env.get(key))
        bt_v = _normalise(backtest_env.get(key))
        if live_v != bt_v:
            issues.append(
                ParityIssue(
                    key=key,
                    live_value=live_v,
                    backtest_value=bt_v,
                    severity="error",
                )
            )
    for key in advisory_keys:
        live_v = _normalise(live_env.get(key))
        bt_v = _normalise(backtest_env.get(key))
        if live_v != bt_v:
            issues.append(
                ParityIssue(
                    key=key,
                    live_value=live_v,
                    backtest_value=bt_v,
                    severity="warning",
                )
            )
    return issues


def parity_holds(issues: list[ParityIssue]) -> bool:
    return not any(i.severity == "error" for i in issues)


def _load_backtest_env(report_path: Path) -> Mapping[str, Any]:
    data = json.loads(Path(report_path).read_text())
    # Two known schemas:
    # 1) live_vs_backtest_same_window.json → top-level key "runtime_env"
    # 2) some legacy reports nest it under "config" or "settings"
    env = (
        data.get("runtime_env")
        or data.get("config", {}).get("runtime_env")
        or data.get("settings", {}).get("runtime_env")
    )
    if not isinstance(env, Mapping):
        raise ValueError(
            f"{report_path} does not declare runtime_env — refuse to compare apples to oranges"
        )
    return env


def _live_env_snapshot() -> Mapping[str, Any]:
    return {key: os.environ.get(key) for key in (*PARITY_REQUIRED_KEYS, *PARITY_ADVISORY_KEYS)}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--backtest",
        type=Path,
        required=True,
        help="Path to a backtest report whose runtime_env will be compared against the running live env.",
    )
    p.add_argument(
        "--live-snapshot",
        type=Path,
        default=None,
        help="Optional JSON file with a captured live env to compare against; if omitted, os.environ is used.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/apples_to_apples_report.json"),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    backtest_env = _load_backtest_env(args.backtest)
    if args.live_snapshot is not None:
        live_env = json.loads(args.live_snapshot.read_text())
    else:
        live_env = _live_env_snapshot()
    issues = check_runtime_parity(live_env, backtest_env)
    holds = parity_holds(issues)
    report = {
        "backtest_path": str(args.backtest),
        "parity_holds": holds,
        "n_errors": sum(1 for i in issues if i.severity == "error"),
        "n_warnings": sum(1 for i in issues if i.severity == "warning"),
        "issues": [issue.__dict__ for issue in issues],
        "live_env": dict(live_env),
        "backtest_env": dict(backtest_env),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    log.info("Wrote %s — parity_holds=%s errors=%d warnings=%d",
             args.output, holds, report["n_errors"], report["n_warnings"])
    if not holds:
        for issue in issues:
            if issue.severity == "error":
                log.error("PARITY VIOLATION %s: live=%s backtest=%s",
                          issue.key, issue.live_value, issue.backtest_value)
    return 0 if holds else 2


if __name__ == "__main__":
    raise SystemExit(main())
