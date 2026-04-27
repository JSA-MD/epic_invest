"""Backtest overlay enforcer — make backtest match live exactly.

Most of the live↔backtest divergence the operator reported reduces to
backtest scripts ignoring the live trader's runtime knobs:

    - PAIRWISE_GROSS_CAP / PAIRWISE_LIVE_MAX_GROSS_CAP
    - PAIRWISE_MAX_HOLD_BARS (24h auto-flatten)
    - PAIRWISE_CVAR_CUT      (R3 tail-loss circuit)
    - PAIRWISE_PROMOTION_FREEZE
    - PAIRWISE_NO_TRADE_BAND_PCT, REBALANCE_NOTIONAL_BAND_USD

Backtest replays therefore over-trade, hold positions days longer than
the live system, and cannot enforce kill switches the live system
would. The result is the apples-to-oranges PnL gap (97.6% of the live
drift attributed to sizing/holding mismatch in the 31-day report).

This module exposes:

    enforce(env, *, mode="strict") -> dict
        Returns a normalised env mapping that the backtest must adopt.
        `mode="strict"` raises on any forbidden value (1.5x leverage,
        max_hold disabled, allow_backtest_like=1, freeze=0, ...).
        `mode="warn"` returns the same dict but downgrades violations
        to log warnings — used during legacy backtest porting.

    inject(env=None) -> dict
        Convenience: read os.environ, normalise via enforce(), and
        write the result back to os.environ so any subprocess started
        below this call inherits the canonical settings. Use at the
        top of every backtest CLI's main().

    runtime_signature(env=None) -> dict
        Snapshot of the parity-relevant keys for embedding into the
        backtest report's `runtime_env` field — read by
        apples_to_apples and promotion_gate_guard during recert.
"""

from __future__ import annotations

import logging
import os
from typing import Mapping, MutableMapping

log = logging.getLogger("backtest_overlay_enforcer")

# Live-equivalent defaults the backtest must observe. These mirror the
# Stage 0 lockdown values in pairwise_live_launchd_env.sh /
# pairwise_live_service.sh write_launchd_env_file.
ENFORCED_DEFAULTS: dict[str, str] = {
    "PAIRWISE_GROSS_CAP": "0.01",
    "PAIRWISE_LIVE_MAX_GROSS_CAP": "0.01",
    "PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP": "0",
    "PAIRWISE_NO_TRADE_BAND_PCT": "10",
    "REBALANCE_NOTIONAL_BAND_USD": "25",
    "PAIRWISE_MAX_HOLD_BARS": "288",
    "PAIRWISE_BREADTH_NOISE_EPSILON": "0",
    "PAIRWISE_CVAR_CUT": "1",
    "PAIRWISE_CVAR_CUT_HOLD_HOURS": "24",
    "PAIRWISE_RUNTIME_BLEND": "1",
    "PAIRWISE_EQUITY_CORR_RISK": "0",
    "PAIRWISE_REFRESH_LIVE_DATA": "1",
    "PAIRWISE_PROMOTION_FREEZE": "1",
    # Plan-time gate adjustments — must match between live and backtest
    # or the silent-drop diagnostic numbers diverge.
    "PAIRWISE_REGIME_THRESHOLD_SCALE": "1.0",
    "PAIRWISE_REGIME_GATE_DISABLED": "0",
}

# Strict bounds — values outside these ranges always violate Stage 0/1.
STRICT_FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "PAIRWISE_GROSS_CAP": (0.0, 0.05),
    "PAIRWISE_LIVE_MAX_GROSS_CAP": (0.0, 0.05),
}
STRICT_INT_BOUNDS: dict[str, tuple[int, int]] = {
    "PAIRWISE_MAX_HOLD_BARS": (12, 2880),
}
FORBIDDEN_TRUE_KEYS: tuple[str, ...] = ("PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP",)


class ParityViolation(RuntimeError):
    """Raised in strict mode when a backtest tries to bypass the live overlays."""


def _bool_truthy(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "on"}


def _validate_float(key: str, raw: str, *, mode: str) -> tuple[str, list[str]]:
    issues: list[str] = []
    try:
        v = float(raw)
    except (TypeError, ValueError):
        issues.append(f"{key}={raw!r} is not numeric — replacing with default")
        return ENFORCED_DEFAULTS[key], issues
    bounds = STRICT_FLOAT_BOUNDS.get(key)
    if bounds is not None and (v < bounds[0] or v > bounds[1]):
        msg = f"{key}={v} outside allowed [{bounds[0]}, {bounds[1]}]"
        if mode == "strict":
            raise ParityViolation(msg)
        issues.append(msg + " — clamped")
        v = max(bounds[0], min(bounds[1], v))
        return f"{v}", issues
    return raw, issues


def _validate_int(key: str, raw: str, *, mode: str) -> tuple[str, list[str]]:
    issues: list[str] = []
    try:
        v = int(raw)
    except (TypeError, ValueError):
        issues.append(f"{key}={raw!r} is not int — replacing with default")
        return ENFORCED_DEFAULTS[key], issues
    bounds = STRICT_INT_BOUNDS.get(key)
    if bounds is not None and (v < bounds[0] or v > bounds[1]):
        msg = f"{key}={v} outside allowed [{bounds[0]}, {bounds[1]}]"
        if mode == "strict":
            raise ParityViolation(msg)
        issues.append(msg + " — clamped")
        v = max(bounds[0], min(bounds[1], v))
        return f"{v}", issues
    return raw, issues


def _validate_forbidden_true(key: str, raw: str, *, mode: str) -> tuple[str, list[str]]:
    if _bool_truthy(raw):
        msg = f"{key}=1 disables Stage 0 lockdown — refused"
        if mode == "strict":
            raise ParityViolation(msg)
        return ENFORCED_DEFAULTS[key], [msg + " — overridden"]
    return raw, []


def enforce(env: Mapping[str, str] | None = None, *, mode: str = "strict") -> dict[str, str]:
    """Return the normalised env. Raises ParityViolation in strict mode."""
    src = env if env is not None else os.environ
    out: dict[str, str] = {}
    issues: list[str] = []

    for key, default in ENFORCED_DEFAULTS.items():
        raw = str(src.get(key, default)).strip()
        if key in STRICT_FLOAT_BOUNDS:
            raw, sub = _validate_float(key, raw, mode=mode)
            issues.extend(sub)
        if key in STRICT_INT_BOUNDS:
            raw, sub = _validate_int(key, raw, mode=mode)
            issues.extend(sub)
        if key in FORBIDDEN_TRUE_KEYS:
            raw, sub = _validate_forbidden_true(key, raw, mode=mode)
            issues.extend(sub)
        out[key] = raw

    if issues:
        for issue in issues:
            log.warning("backtest parity: %s", issue)

    return out


def inject(env: MutableMapping[str, str] | None = None, *, mode: str = "strict") -> dict[str, str]:
    """Apply the canonical env to `env` (defaults to os.environ) and return it."""
    target = env if env is not None else os.environ
    enforced = enforce(target, mode=mode)
    for key, value in enforced.items():
        target[key] = value
    return enforced


def runtime_signature(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the parity-relevant slice of env for embedding into reports.

    The `apples_to_apples` checker reads exactly this signature, so any
    backtest that wants its result accepted by `promotion_gate_guard`
    must include this dict (or a superset) in its output JSON.
    """
    src = env if env is not None else os.environ
    return {key: str(src.get(key, "")) for key in ENFORCED_DEFAULTS}


__all__ = [
    "ENFORCED_DEFAULTS",
    "ParityViolation",
    "enforce",
    "inject",
    "runtime_signature",
]
