from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional


def enforce_runtime_gross_cap_ceiling(
    env: Mapping[str, str] | None = None,
) -> tuple[float, Optional[str]]:
    """Return (effective_gross_cap, warning_message_or_None).

    Reads PAIRWISE_GROSS_CAP (default "1.0") and PAIRWISE_LIVE_MAX_GROSS_CAP
    (default "0.05") from *env* (os.environ when env is None).  Rejects
    negative, NaN, infinite, or unreasonably large (>1.0) values.

    Invalid inputs are excluded from the effective minimum so a corrupted env
    var can never widen exposure beyond the user's stricter valid setting.
    The fallback is the minimum of (SAFE_DEFAULT, all valid inputs).
    """
    import math

    if env is None:
        env = os.environ

    _safe_default_raw = env.get("PAIRWISE_SAFETY_DEFAULT_GROSS_CAP", "0.01")
    try:
        SAFE_DEFAULT = float(_safe_default_raw)
        if not (0 < SAFE_DEFAULT <= 1.0):
            SAFE_DEFAULT = 0.01
            _safe_default_warn: Optional[str] = (
                f"PAIRWISE_SAFETY_DEFAULT_GROSS_CAP={_safe_default_raw!r} outside (0, 1.0] "
                f"— falling back to 0.01"
            )
        else:
            _safe_default_warn = None
    except (TypeError, ValueError):
        SAFE_DEFAULT = 0.01
        _safe_default_warn = (
            f"PAIRWISE_SAFETY_DEFAULT_GROSS_CAP={_safe_default_raw!r} is not a valid number "
            f"— falling back to 0.01"
        )
    HARD_MAX = 1.0
    promotion_freeze = str(env.get("PAIRWISE_PROMOTION_FREEZE", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    allow_backtest_like_raw = str(env.get("PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # Stage 0 lockdown: PROMOTION_FREEZE=1 disables backtest-like bypass entirely.
    allow_backtest_like = allow_backtest_like_raw and not promotion_freeze

    def _parse_safe(name: str, default: str) -> tuple[Optional[float], Optional[str]]:
        raw = env.get(name, default)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None, f"{name}={raw!r} is not a valid number"
        if math.isnan(v) or math.isinf(v):
            return None, f"{name}={raw!r} is NaN/inf"
        if v < 0.0:
            return None, f"{name}={v} is negative"
        if v > HARD_MAX:
            return None, f"{name}={v} exceeds HARD_MAX={HARD_MAX}"
        return v, None

    runtime, runtime_err = _parse_safe("PAIRWISE_GROSS_CAP", str(SAFE_DEFAULT))
    ceiling, ceiling_err = _parse_safe("PAIRWISE_LIVE_MAX_GROSS_CAP", str(SAFE_DEFAULT))
    errors = [e for e in (_safe_default_warn, runtime_err, ceiling_err) if e]

    # Conservative fallback: take min of SAFE_DEFAULT and any valid input.
    # Invalid inputs are excluded so a corrupt env var can never widen exposure
    # beyond the user's stricter valid setting.
    candidates = [SAFE_DEFAULT]
    if runtime is not None:
        candidates.append(runtime)
    if ceiling is not None:
        candidates.append(ceiling)
    effective = min(candidates)

    if errors:
        warning = "; ".join(errors) + f" — falling back to most conservative valid value {effective}"
        return effective, warning

    if allow_backtest_like and runtime is not None and ceiling is not None:
        # Backtest-like mode bypasses SAFE_DEFAULT but still respects ceiling
        # (only enabled when PROMOTION_FREEZE=0).
        if runtime > ceiling:
            warning = (
                f"PAIRWISE_GROSS_CAP={runtime} exceeds PAIRWISE_LIVE_MAX_GROSS_CAP={ceiling} — "
                f"clipping to {ceiling}"
            )
            return ceiling, warning
        return runtime, None

    # Live path: SAFE_DEFAULT acts as an absolute hard ceiling. Any valid input
    # exceeding it is clipped down. This is what makes Stage 0 lockdown durable
    # across watchdog restarts and accidental env var overrides.
    if runtime > effective:
        if ceiling is not None and ceiling < SAFE_DEFAULT and ceiling == effective:
            binding = f"PAIRWISE_LIVE_MAX_GROSS_CAP={ceiling}"
        else:
            binding = f"SAFE_DEFAULT={SAFE_DEFAULT} (Stage 0 lockdown)"
        warning = (
            f"PAIRWISE_GROSS_CAP={runtime} exceeds {binding} — clipping to {effective}"
        )
        return effective, warning
    if ceiling < runtime:
        # Defensive: shouldn't trigger after the runtime > effective branch above,
        # but kept for parity with prior contract when runtime <= SAFE_DEFAULT
        # but ceiling is tighter than runtime.
        warning = (
            f"PAIRWISE_GROSS_CAP={runtime} exceeds PAIRWISE_LIVE_MAX_GROSS_CAP={ceiling} — "
            f"clipping to {ceiling}"
        )
        return ceiling, warning
    return effective, None


def clip_candidate_gross_cap(candidate: dict, equity: float | None = None) -> dict:
    """DEPRECATED — this function is a no-op retained for backward compatibility.

    The candidate JSON does not carry gross_cap keys under pair_configs or
    pair_convex_blends, so this function never clipped anything in production.
    Use enforce_runtime_gross_cap_ceiling() instead, which reads the runtime
    env vars directly and is wired into run_live_once (guard C).
    """
    return candidate


def check_position_divergence(
    state: dict,
    current_positions: Mapping[str, Mapping[str, Any]],
    equity: float | None,
) -> Optional[str]:
    def _pair_notional(pos: Any) -> tuple[float, float]:
        if not isinstance(pos, Mapping):
            return 0.0, 0.0
        try:
            qty = float(pos.get("qty") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        price = 0.0
        for key in ("mark_price", "entry_price", "notional"):
            raw = pos.get(key)
            if raw:
                try:
                    price = float(raw)
                    break
                except (TypeError, ValueError):
                    continue
        return qty, price

    try:
        if equity is None or equity == 0.0:
            return None
        latest_sync = state.get("latest_live_sync") or {}
        if latest_sync.get("source") != "run_live_once_post_trade":
            return None
        prev_positions = latest_sync.get("positions") or {}
        threshold = abs(equity) * 0.01
        msgs: list[str] = []
        all_pairs = set(prev_positions.keys()) | set(current_positions.keys())
        for pair in all_pairs:
            prev_pos = prev_positions.get(pair) or {}
            curr_pos = current_positions.get(pair) or {}
            prev_qty, prev_price = _pair_notional(prev_pos)
            curr_qty, curr_price = _pair_notional(curr_pos)
            price = curr_price or prev_price
            if price <= 0.0:
                continue
            diff = abs(curr_qty - prev_qty) * abs(price)
            if diff > threshold:
                msgs.append(
                    f"{pair}: qty {prev_qty:.4f}->{curr_qty:.4f} "
                    f"notional_diff={diff:.2f} threshold={threshold:.2f}"
                )
        if msgs:
            return "; ".join(msgs)
        return None
    except Exception:
        return None


def refresh_latest_prices_from_rest(
    exchange: Any,
    pairs: Iterable[str],
    *,
    prev_prices: Mapping[str, float] | None = None,
    timeout_seconds: float = 5.0,
) -> tuple[dict[str, float], dict[str, str]]:
    """Best-effort REST ticker refresh — bypasses LOB cache staleness.

    Returns (refreshed, errors). Each pair is fetched independently;
    failures fall back to prev_prices (if provided) or are omitted.
    """
    import time

    refreshed: dict[str, float] = dict(prev_prices) if prev_prices else {}
    errors: dict[str, str] = {}
    if exchange is None:
        return refreshed, errors
    fetch_ticker = getattr(exchange, "fetch_ticker", None)
    if not callable(fetch_ticker):
        return refreshed, errors
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    for pair in pairs:
        if time.monotonic() > deadline:
            errors[pair] = "deadline_exceeded"
            continue
        try:
            tick = fetch_ticker(pair) or {}
            price_raw = (
                tick.get("last")
                or tick.get("close")
                or tick.get("mark")
                or tick.get("info", {}).get("lastPrice")
            )
            if price_raw is None:
                errors[pair] = "no_price_in_ticker"
                continue
            price = float(price_raw)
            if price > 0.0:
                refreshed[pair] = price
            else:
                errors[pair] = f"non_positive_price={price}"
        except Exception as e:  # noqa: BLE001 — ticker fetch is best-effort
            errors[pair] = f"{type(e).__name__}: {e}"
    return refreshed, errors


def update_stale_price_tracker(
    state: dict,
    latest_prices: Mapping[str, float],
    max_stale_cycles: int = 12,
) -> list[str]:
    try:
        if "stale_price_counts" not in state or state["stale_price_counts"] is None:
            state["stale_price_counts"] = {}
        if "last_known_prices" not in state or state["last_known_prices"] is None:
            state["last_known_prices"] = {}

        counts: dict[str, int] = state["stale_price_counts"]
        last_known: dict[str, float] = state["last_known_prices"]
        stale_pairs: list[str] = []

        for pair, price in latest_prices.items():
            if price is None:
                continue
            try:
                price = float(price)
            except (TypeError, ValueError):
                continue
            prev = last_known.get(pair)
            if prev is not None and price == prev:
                counts[pair] = counts.get(pair, 0) + 1
            else:
                counts[pair] = 0
            last_known[pair] = price
            if counts[pair] > max_stale_cycles:
                stale_pairs.append(pair)

        state["stale_price_counts"] = counts
        state["last_known_prices"] = last_known
        return stale_pairs
    except Exception:
        return []


def validate_safety_switches(env: Mapping[str, str] | None = None) -> list[str]:
    if env is None:
        env = os.environ

    warnings: list[str] = []

    raw_max_hold = env.get("PAIRWISE_MAX_HOLD_BARS")
    if raw_max_hold is not None:
        try:
            val = int(raw_max_hold)
            if val <= 0:
                warnings.append(
                    f"PAIRWISE_MAX_HOLD_BARS={val} — max-hold auto-flatten is disabled"
                )
            elif val < 12:
                warnings.append(
                    f"PAIRWISE_MAX_HOLD_BARS={val} is suspiciously low (<12); expected 12-2880"
                )
            elif val > 2880:
                warnings.append(
                    f"PAIRWISE_MAX_HOLD_BARS={val} exceeds 10-day window (>2880); expected 12-2880"
                )
        except (TypeError, ValueError):
            warnings.append(f"PAIRWISE_MAX_HOLD_BARS={raw_max_hold!r} is not a valid integer")

    _BOOL_VALUES = {"0", "1", "true", "false", "yes", "no"}

    raw_cvar = env.get("PAIRWISE_CVAR_CUT")
    if raw_cvar is not None:
        normalized = raw_cvar.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            warnings.append(
                f"PAIRWISE_CVAR_CUT={raw_cvar!r} — CVaR-99 cut overlay is disabled"
            )
        elif normalized not in _BOOL_VALUES and normalized != "off":
            warnings.append(f"PAIRWISE_CVAR_CUT={raw_cvar!r} is not a valid boolean")

    raw_cvar_hold = env.get("PAIRWISE_CVAR_CUT_HOLD_HOURS")
    if raw_cvar_hold is not None:
        try:
            cvar_hold = int(raw_cvar_hold)
            # Sane range: 1 hour minimum (just-fired cut) to 168 hours (one week)
            if cvar_hold < 1 or cvar_hold > 168:
                warnings.append(
                    f"PAIRWISE_CVAR_CUT_HOLD_HOURS={cvar_hold} outside safe range [1, 168]"
                )
        except (TypeError, ValueError):
            warnings.append(
                f"PAIRWISE_CVAR_CUT_HOLD_HOURS={raw_cvar_hold!r} is not a valid integer"
            )

    raw_force = env.get("PAIRWISE_FORCE_EXECUTE")
    if raw_force is not None:
        normalized = raw_force.strip().lower()
        if normalized == "1" or raw_force.strip() == "1":
            warnings.append(
                "PAIRWISE_FORCE_EXECUTE=1 — force-execute is active (promotion gate bypass)"
            )
        elif normalized not in _BOOL_VALUES:
            warnings.append(f"PAIRWISE_FORCE_EXECUTE={raw_force!r} is not a valid boolean")

    raw_cap = env.get("PAIRWISE_LIVE_MAX_GROSS_CAP")
    if raw_cap is not None:
        try:
            val = float(raw_cap)
            if val > 0.05:
                warnings.append(
                    f"PAIRWISE_LIVE_MAX_GROSS_CAP={val} exceeds Stage A ceiling 0.05 "
                    f"(Stage 0 lockdown — CPCV revalidation required before raising)"
                )
            if val > 0.20:
                warnings.append(
                    f"PAIRWISE_LIVE_MAX_GROSS_CAP={val} exceeds defensive ceiling 0.20 "
                    f"(Stage C sizing — ensure staged validation is complete)"
                )
        except (TypeError, ValueError):
            warnings.append(f"PAIRWISE_LIVE_MAX_GROSS_CAP={raw_cap!r} is not a valid number")

    raw_allow_backtest_cap = env.get("PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP")
    if raw_allow_backtest_cap is not None:
        normalized = raw_allow_backtest_cap.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            warnings.append(
                "PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP=1 — defensive 5% gross-cap floor is bypassed"
            )
        elif normalized not in _BOOL_VALUES and normalized != "off":
            warnings.append(
                f"PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP={raw_allow_backtest_cap!r} is not a valid boolean"
            )

    return warnings


def validate_state_alphas_coverage(
    candidate: dict,
    observed_route_states: Iterable[str],
) -> list[str]:
    try:
        sc = candidate.get("selected_candidate") or {}
        pair_convex_blends = sc.get("pair_convex_blends") or {}
        observed = set(observed_route_states)
        if not observed:
            return []
        msgs: list[str] = []
        for pair, blend in pair_convex_blends.items():
            if not isinstance(blend, dict):
                continue
            if blend.get("mode") != "state_alphas":
                continue
            state_alphas = blend.get("state_alphas") or {}
            if not isinstance(state_alphas, dict):
                continue
            for rs in observed:
                if rs not in state_alphas:
                    msgs.append(
                        f"{pair}: route '{rs}' missing from state_alphas"
                    )
        return msgs
    except Exception:
        return []
