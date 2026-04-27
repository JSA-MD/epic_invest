"""Shared signal→weight transformation kernel.

Both backtest and live runtime call from here. No hardcoded numerics —
all thresholds come from kwargs or shared_strategy_config.
"""
from __future__ import annotations

from typing import Mapping

from btc_convex_blend import blend_runtime_weight, resolve_blend_alpha


def compute_target_weight(
    *,
    baseline_weight: float,
    specialist_weight: float,
    state_alphas: Mapping[str, float],
    route_state_name: str,
    blend_mode: str = "state_alphas",
    default_blend_alpha: float = 0.0,
    effective_gross_cap: float | None = None,
    target_weight_eps: float = 1e-6,
) -> dict:
    """Single source of truth for signal→target_weight transformation.

    Returns a dict:
      {
        "target_weight": float,           # final weight after blend + clip
        "blend_alpha": float,             # the alpha actually applied
        "pre_clip_weight": float,         # weight before D2 gross-cap clip
        "clipped": bool,                  # True if D2 clamp engaged
      }
    """
    blend_alpha = resolve_blend_alpha(
        baseline_weight=float(baseline_weight),
        specialist_weight=float(specialist_weight),
        route_state_name=str(route_state_name),
        alpha=float(default_blend_alpha),
        mode=str(blend_mode),
        state_alphas=dict(state_alphas) if state_alphas is not None else None,
    )

    blended = blend_runtime_weight(
        baseline_weight=float(baseline_weight),
        specialist_weight=float(specialist_weight),
        route_state_name=str(route_state_name),
        alpha=float(default_blend_alpha),
        mode=str(blend_mode),
        state_alphas=dict(state_alphas) if state_alphas is not None else None,
    )

    pre_clip_weight = float(blended)
    clipped = False

    if effective_gross_cap is not None and float(effective_gross_cap) >= 0.0:
        capped = max(-float(effective_gross_cap), min(float(effective_gross_cap), pre_clip_weight))
        if abs(capped - pre_clip_weight) > target_weight_eps:
            clipped = True
        blended = capped

    if abs(float(blended)) < target_weight_eps:
        blended = 0.0

    return {
        "target_weight": float(blended),
        "blend_alpha": float(blend_alpha),
        "pre_clip_weight": pre_clip_weight,
        "clipped": clipped,
    }
