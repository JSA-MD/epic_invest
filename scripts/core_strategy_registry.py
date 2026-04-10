#!/usr/bin/env python3
"""Shared core-strategy artifact helpers for search and live execution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_cash_filtered_rotation import StrategyParams, build_target_weights, json_ready
from ga_long_short_rotation import LongShortParams, build_long_short_target_weights
from gp_crypto_evolution import MODELS_DIR

DEFAULT_CORE_CHAMPION_PATH = MODELS_DIR / "core_champion.json"

LONG_ONLY_FAMILY = "long_only"
LONG_SHORT_FAMILY = "long_short"


@dataclass(frozen=True)
class ResolvedCoreStrategy:
    family: str
    params: StrategyParams | LongShortParams
    key: str
    source: str
    metadata: dict[str, Any]


def normalize_family(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"long_only", "long-only", "cash_rotation"}:
        return LONG_ONLY_FAMILY
    if text in {"long_short", "long-short", "market_regime_long_short"}:
        return LONG_SHORT_FAMILY
    raise ValueError(f"Unsupported core strategy family: {value!r}")


def coerce_long_only_params(value: StrategyParams | dict[str, Any]) -> StrategyParams:
    if isinstance(value, StrategyParams):
        return value
    payload = dict(value)
    return StrategyParams(
        timeframe=str(payload.get("timeframe", "1d")),
        lookback_fast=int(payload["lookback_fast"]),
        lookback_slow=int(payload["lookback_slow"]),
        top_n=int(payload["top_n"]),
        vol_window=int(payload["vol_window"]),
        target_vol_ann=float(payload["target_vol_ann"]),
        regime_threshold=float(payload.get("regime_threshold", 0.0)),
        breadth_threshold=float(payload.get("breadth_threshold", 0.50)),
        gross_cap=float(payload.get("gross_cap", 1.5)),
        fee_rate=float(payload.get("fee_rate", payload.get("commission_pct", 0.0004))),
    )


def coerce_long_short_params(value: LongShortParams | dict[str, Any]) -> LongShortParams:
    if isinstance(value, LongShortParams):
        return value
    payload = dict(value)
    return LongShortParams(
        lookback_fast=int(payload["lookback_fast"]),
        lookback_slow=int(payload["lookback_slow"]),
        top_n=int(payload["top_n"]),
        vol_window=int(payload["vol_window"]),
        target_vol_ann=float(payload["target_vol_ann"]),
        long_regime_threshold=float(payload.get("long_regime_threshold", 0.0)),
        short_regime_threshold=float(payload.get("short_regime_threshold", 0.0)),
        long_breadth_threshold=float(payload.get("long_breadth_threshold", 0.50)),
        short_breadth_threshold=float(payload.get("short_breadth_threshold", 0.35)),
        gross_cap=float(payload.get("gross_cap", 1.5)),
        short_vol_mult=float(payload.get("short_vol_mult", 1.0)),
        fee_rate=float(payload.get("fee_rate", payload.get("commission_pct", 0.0004))),
    )


def params_key(family: str, params: StrategyParams | LongShortParams) -> str:
    normalized = normalize_family(family)
    if normalized == LONG_ONLY_FAMILY:
        payload = asdict(coerce_long_only_params(params))
        return (
            f"long_only:"
            f"f{payload['lookback_fast']}_s{payload['lookback_slow']}_n{payload['top_n']}_"
            f"vw{payload['vol_window']}_tv{payload['target_vol_ann']:.2f}_"
            f"rt{payload['regime_threshold']:.2f}_bt{payload['breadth_threshold']:.2f}_"
            f"gc{payload['gross_cap']:.2f}"
        )
    resolved = coerce_long_short_params(params)
    return f"long_short:{resolved.key()}"


def resolve_core_strategy(
    family: str,
    params: StrategyParams | LongShortParams | dict[str, Any],
    *,
    key: str | None = None,
    source: str = "inline",
    metadata: dict[str, Any] | None = None,
) -> ResolvedCoreStrategy:
    normalized = normalize_family(family)
    resolved_params: StrategyParams | LongShortParams
    if normalized == LONG_ONLY_FAMILY:
        resolved_params = coerce_long_only_params(params)
    else:
        resolved_params = coerce_long_short_params(params)
    return ResolvedCoreStrategy(
        family=normalized,
        params=resolved_params,
        key=str(key or params_key(normalized, resolved_params)),
        source=str(source),
        metadata=dict(metadata or {}),
    )


def build_core_target_weights(close: pd.DataFrame, strategy: ResolvedCoreStrategy) -> pd.DataFrame:
    if strategy.family == LONG_ONLY_FAMILY:
        assert isinstance(strategy.params, StrategyParams)
        return build_target_weights(close, strategy.params)
    assert isinstance(strategy.params, LongShortParams)
    return build_long_short_target_weights(close, strategy.params)


def build_artifact_payload(
    strategy: ResolvedCoreStrategy,
    *,
    selected_score: float | None = None,
    summary_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "selected_family": strategy.family,
        "selected_key": strategy.key,
        "params": asdict(strategy.params),
        "source": strategy.source,
        "metadata": dict(strategy.metadata),
    }
    if selected_score is not None:
        payload["selected_score"] = float(selected_score)
    if summary_path is not None:
        payload["summary_path"] = str(summary_path)
    if extra:
        payload.update(extra)
    return payload


def save_core_artifact(
    path: str | Path,
    strategy: ResolvedCoreStrategy,
    *,
    selected_score: float | None = None,
    summary_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_artifact_payload(
        strategy,
        selected_score=selected_score,
        summary_path=summary_path,
        extra=extra,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_ready(payload), f, indent=2, allow_nan=False)
    return payload


def load_core_artifact(path: str | Path = DEFAULT_CORE_CHAMPION_PATH) -> ResolvedCoreStrategy:
    artifact_path = Path(path)
    with open(artifact_path, "r") as f:
        payload = json.load(f)
    family = payload.get("selected_family") or payload.get("family")
    key = payload.get("selected_key") or payload.get("key")
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("artifact_path", str(artifact_path))
    for passthrough in ("summary_path", "selected_score", "created_at", "search_config"):
        if passthrough in payload:
            metadata[passthrough] = payload[passthrough]
    return resolve_core_strategy(
        str(family),
        payload["params"],
        key=key,
        source=str(payload.get("source") or artifact_path),
        metadata=metadata,
    )
