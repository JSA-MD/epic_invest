from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from btc_convex_blend import get_btc_convex_blend, replay_btc_convex_blend_candidate, replay_target_trace
from search_pair_subset_regime_mixture import realistic_overlay_replay_from_context
from search_pair_subset_regime_mixture import route_state_names


BLEND_PAIR = "BTCUSDT"


def get_btc_online_blend(candidate: Mapping[str, Any] | None, pair: str | None = None) -> dict[str, Any] | None:
    if not isinstance(candidate, Mapping):
        return None
    raw = candidate.get("btc_online_blend")
    if not isinstance(raw, Mapping):
        return None
    payload = copy.deepcopy(dict(raw))
    blend_pair = str(payload.get("pair") or BLEND_PAIR)
    if pair is not None and str(pair) != blend_pair:
        return None
    payload["pair"] = blend_pair
    payload["alpha_cap"] = float(payload.get("alpha_cap", 0.0))
    payload["eta"] = float(payload.get("eta", 0.0))
    payload["decay"] = float(payload.get("decay", 1.0))
    payload["activation_mode"] = str(payload.get("activation_mode") or "disagree_only")
    payload["reward_scale"] = float(payload.get("reward_scale", 10000.0))
    payload["base_expert"] = str(payload.get("base_expert") or "current_main")
    return payload


def _resolve_active_mask(
    *,
    context: Mapping[str, Any],
    route_breadth_threshold: float,
    baseline_target: np.ndarray,
    specialist_target: np.ndarray,
    activation_mode: str,
) -> np.ndarray:
    if activation_mode == "always":
        return np.ones_like(baseline_target, dtype=bool)

    route_names = route_state_names(str(context.get("route_state_mode") or "equity_corr"))
    bucket_codes = np.asarray(context["bucket_codes"][float(route_breadth_threshold)], dtype="int64")[: len(baseline_target)]
    narrow_mask = np.asarray([str(route_names[int(idx)]).endswith("narrow") for idx in bucket_codes], dtype=bool)
    disagree_mask = np.sign(baseline_target) != np.sign(specialist_target)

    if activation_mode == "narrow_only":
        return narrow_mask
    if activation_mode == "disagree_only":
        return disagree_mask
    if activation_mode == "disagree_narrow_only":
        return narrow_mask & disagree_mask
    raise ValueError(f"unsupported activation mode: {activation_mode}")


def _resolve_runtime_active(
    *,
    route_state_name: str,
    baseline_weight: float,
    specialist_weight: float,
    activation_mode: str,
) -> bool:
    if activation_mode == "always":
        return True
    narrow = str(route_state_name).endswith("narrow")
    disagree = np.sign(float(baseline_weight)) != np.sign(float(specialist_weight))
    if activation_mode == "narrow_only":
        return bool(narrow)
    if activation_mode == "disagree_only":
        return bool(disagree)
    if activation_mode == "disagree_narrow_only":
        return bool(narrow and disagree)
    raise ValueError(f"unsupported activation mode: {activation_mode}")


def runtime_online_blend_alpha(
    *,
    previous_score: float,
    alpha_cap: float,
    eta: float,
    activation_mode: str,
    route_state_name: str,
    baseline_weight: float,
    specialist_weight: float,
) -> float:
    if not _resolve_runtime_active(
        route_state_name=route_state_name,
        baseline_weight=baseline_weight,
        specialist_weight=specialist_weight,
        activation_mode=activation_mode,
    ):
        return 0.0
    alpha = float(alpha_cap) * max(0.0, np.tanh(float(eta) * float(previous_score)))
    return float(np.clip(alpha, 0.0, float(alpha_cap)))


def update_runtime_online_score(
    *,
    previous_score: float,
    baseline_weight: float,
    specialist_weight: float,
    previous_price: float | None,
    current_price: float | None,
    decay: float,
    reward_scale: float,
) -> float:
    if previous_price is None or current_price is None:
        return float(previous_score)
    if not np.isfinite(previous_price) or not np.isfinite(current_price) or abs(float(previous_price)) <= 1e-12:
        return float(previous_score)
    price_ret = float(current_price) / float(previous_price) - 1.0
    baseline_bar = float(baseline_weight) * price_ret
    specialist_bar = float(specialist_weight) * price_ret
    advantage = (specialist_bar - baseline_bar) * float(reward_scale)
    return float(decay) * float(previous_score) + float(advantage)


def build_online_expert_blend_trace(
    *,
    context: Mapping[str, Any],
    route_breadth_threshold: float,
    baseline_trace: Mapping[str, Any],
    specialist_trace: Mapping[str, Any],
    alpha_cap: float,
    eta: float,
    decay: float,
    activation_mode: str = "disagree_narrow_only",
    reward_scale: float = 10000.0,
) -> dict[str, np.ndarray]:
    baseline_target = np.asarray(baseline_trace["target_weight"], dtype="float64")
    specialist_target = np.asarray(specialist_trace["target_weight"], dtype="float64")
    baseline_bar = np.asarray(baseline_trace["bar_net"], dtype="float64")
    specialist_bar = np.asarray(specialist_trace["bar_net"], dtype="float64")

    active_mask = _resolve_active_mask(
        context=context,
        route_breadth_threshold=route_breadth_threshold,
        baseline_target=baseline_target,
        specialist_target=specialist_target,
        activation_mode=activation_mode,
    )

    target = baseline_target.copy()
    alpha_trace = np.zeros_like(baseline_target)
    score_trace = np.zeros_like(baseline_target)

    score = 0.0
    for i in range(len(baseline_target)):
        score_trace[i] = score
        if active_mask[i]:
            alpha = float(alpha_cap) * max(0.0, np.tanh(float(eta) * score))
            alpha = float(np.clip(alpha, 0.0, float(alpha_cap)))
            target[i] = (1.0 - alpha) * baseline_target[i] + alpha * specialist_target[i]
            alpha_trace[i] = alpha
        if i < len(baseline_bar):
            advantage = float(specialist_bar[i] - baseline_bar[i]) * float(reward_scale)
            score = float(decay) * score + advantage

    return {
        "target_weight": target,
        "alpha": alpha_trace,
        "score": score_trace,
        "active": active_mask.astype(bool),
    }


def replay_btc_online_blend_candidate(
    *,
    candidate: Mapping[str, Any],
    pair: str,
    context: Mapping[str, Any],
    library_lookup: Mapping[str, Any],
    return_trace: bool = False,
) -> dict[str, Any]:
    online = get_btc_online_blend(candidate, pair)
    if online is None:
        raise RuntimeError("Candidate has no btc_online_blend payload.")
    base_expert = str(online.get("base_expert") or "current_main")
    if base_expert != "current_main":
        raise RuntimeError(f"Unsupported btc_online_blend base_expert: {base_expert}")

    baseline = replay_btc_convex_blend_candidate(
        candidate=candidate,
        pair=pair,
        context=context,
        library_lookup=library_lookup,
        return_trace=True,
    )
    base_blend = get_btc_convex_blend(candidate, pair)
    if base_blend is None:
        raise RuntimeError("btc_online_blend requires btc_convex_blend specialist source.")
    specialist_pair_config = dict(base_blend["specialist_pair_config"])
    specialist = realistic_overlay_replay_from_context(
        context,
        library_lookup,
        tuple(int(v) for v in specialist_pair_config["mapping_indices"]),
        float(specialist_pair_config["route_breadth_threshold"]),
        execution_gene=specialist_pair_config.get("execution_gene"),
        engine="python",
        return_trace=True,
    )
    blended = build_online_expert_blend_trace(
        context=context,
        route_breadth_threshold=float(specialist_pair_config["route_breadth_threshold"]),
        baseline_trace=baseline["trace"],
        specialist_trace=specialist["trace"],
        alpha_cap=float(online["alpha_cap"]),
        eta=float(online["eta"]),
        decay=float(online["decay"]),
        activation_mode=str(online["activation_mode"]),
        reward_scale=float(online["reward_scale"]),
    )
    result = replay_target_trace(
        context=context,
        target_trace=np.asarray(blended["target_weight"], dtype="float64"),
        execution_gene=None,
        trace_template=baseline["trace"],
        return_trace=return_trace,
    )
    if return_trace:
        result["trace"]["online_alpha"] = np.asarray(blended["alpha"], dtype="float64")
        result["trace"]["online_score"] = np.asarray(blended["score"], dtype="float64")
        result["trace"]["online_active"] = np.asarray(blended["active"], dtype=bool)
    return result
