#!/usr/bin/env python3
"""Component 3: Smart digest indicators — delta vs. rolling averages.

Import and call compute_smart_indicators() from the daily digest script
or from telegram_chart_attach.py to enrich the digest context block.

Example:
    from telegram_smart_indicators import compute_smart_indicators
    rows = load_daily_pnl(window_days=30)
    ctx = compute_smart_indicators(rows)
    # Merge ctx into format_alert(..., context=ctx)
"""

from __future__ import annotations

import statistics
from typing import Any


def compute_smart_indicators(daily_pnl_rows: list[dict]) -> dict[str, Any]:
    """Compute smart performance indicators from daily PnL rows.

    Args:
        daily_pnl_rows: List of dicts with at minimum {"date": str, "total": float}.
                        Must be sorted oldest-first (as returned by load_daily_pnl).

    Returns:
        dict with:
            today_vs_7d_avg_pct   — today's PnL vs last-7-day mean, in percent
            today_vs_30d_avg_pct  — today's PnL vs last-30-day mean, in percent
            day_streak            — consecutive positive (>0) or negative (<=0) days
                                    positive int = up-streak, negative int = down-streak
            color_indicator       — "🟢" if today > 7d avg, "🔴" if < 7d avg - 1σ,
                                    "🟡" otherwise (within 1σ of 7d avg)
    """
    if not daily_pnl_rows:
        return {
            "today_vs_7d_avg_pct": 0.0,
            "today_vs_30d_avg_pct": 0.0,
            "day_streak": 0,
            "color_indicator": "🟡",
        }

    vals = [float(r.get("total", 0.0)) for r in daily_pnl_rows]
    today = vals[-1]

    # Rolling averages
    last_7 = vals[-7:] if len(vals) >= 7 else vals
    last_30 = vals[-30:] if len(vals) >= 30 else vals

    avg_7d = statistics.mean(last_7)
    avg_30d = statistics.mean(last_30)

    def _pct_change(current: float, baseline: float) -> float:
        """Percentage change from baseline to current, avoiding div-by-zero."""
        if baseline == 0.0:
            return 0.0 if current == 0.0 else (100.0 if current > 0 else -100.0)
        return (current - baseline) / abs(baseline) * 100.0

    vs_7d = _pct_change(today, avg_7d)
    vs_30d = _pct_change(today, avg_30d)

    # 1-sigma of 7-day window
    if len(last_7) >= 2:
        sigma_7d = statistics.stdev(last_7)
    else:
        sigma_7d = 0.0

    # Color indicator
    if today > avg_7d:
        color = "🟢"
    elif today < avg_7d - sigma_7d:
        color = "🔴"
    else:
        color = "🟡"

    # Day streak: walk backwards from today
    streak = 0
    if vals:
        direction = 1 if today > 0 else -1
        for v in reversed(vals):
            if direction == 1 and v > 0:
                streak += 1
            elif direction == -1 and v <= 0:
                streak -= 1
            else:
                break

    return {
        "today_vs_7d_avg_pct": round(vs_7d, 2),
        "today_vs_30d_avg_pct": round(vs_30d, 2),
        "day_streak": streak,
        "color_indicator": color,
    }
