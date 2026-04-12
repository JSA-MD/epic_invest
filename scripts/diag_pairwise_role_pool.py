#!/usr/bin/env python3
"""Stage-2 diagnostic for Event/DC-aware role pools on the MOO router experiment."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import gp_crypto_evolution as gp
from replay_regime_mixture_realistic import load_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_pairwise_moo_router import (
    SEARCH_WINDOWS,
    SPECIALIST_ROLE_NAMES,
    build_baseline_pair_configs,
    build_role_specific_specialist_pools,
)
from search_pair_subset_regime_mixture import (
    ROUTE_STATE_MODE_EQUITY_CORR,
    aggregate_metrics,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    load_or_fetch_funding,
    realistic_overlay_replay_from_context,
)


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = Path("/tmp/role_pool_stage2_diag.json")
EXP_PATH = ROOT / "models/gp_regime_mixture_btc_bnb_pairwise_moo_router_exp_summary_locked.json"


def _window_alias(label: str) -> str:
    return {
        "recent_2m": "2m",
        "recent_6m": "6m",
        "full_4y": "4y",
    }[label]


def _collect_compare_metrics(
    candidate: dict[str, object],
    *,
    pairs: tuple[str, ...],
    full_lookup: dict[str, object],
    full_window_cache: dict[str, dict[str, object]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in ("recent_2m", "recent_6m", "full_4y"):
        per_pair = {}
        for pair in pairs:
            cfg = candidate["pair_configs"][pair]
            per_pair[pair] = realistic_overlay_replay_from_context(
                full_window_cache[label]["pairs"][pair]["fast_context"],
                full_lookup,
                tuple(int(v) for v in cfg["mapping_indices"]),
                float(cfg["route_breadth_threshold"]),
                execution_gene=cfg.get("execution_gene"),
                state_specialists=tuple(int(v) for v in cfg.get("state_specialists") or ()),
                engine="python",
            )
        agg = aggregate_metrics(per_pair)
        out[_window_alias(label)] = {
            "mean_total_return": float(agg["mean_total_return"]),
            "worst_pair_total_return": float(agg["worst_pair_total_return"]),
            "worst_max_drawdown": float(agg["worst_max_drawdown"]),
            "mean_daily_win_rate": float(agg.get("mean_daily_win_rate", 0.0)),
            "worst_pair_daily_win_rate": float(agg.get("worst_pair_daily_win_rate", 0.0)),
        }
    return out


def main() -> None:
    exp = json.loads(EXP_PATH.read_text())
    main_summary = json.loads(Path(exp["baseline_summary_path"]).read_text())
    main_backtest = json.loads((ROOT / "models/main_current_backtest_locked.json").read_text())

    pairs = tuple(exp.get("pairs") or ("BTCUSDT", "BNBUSDT"))
    route_thresholds = tuple(
        float(v) for v in ((exp.get("search") or {}).get("route_thresholds") or (0.35, 0.50, 0.65, 0.80))
    )
    baseline_pair_configs = build_baseline_pair_configs(pairs, main_summary)

    full_library = list(iter_params())
    selected = exp.get("selected_candidate") or (exp.get("shadow_challenger_candidates") or [None])[0]

    subset_orig: set[int] = set()
    for source in [selected, *((exp.get("shadow_challenger_candidates") or [])[:3]), main_summary.get("selected_candidate")]:
        if not isinstance(source, dict):
            continue
        for pair in pairs:
            cfg = (source.get("pair_configs") or {}).get(pair) or {}
            subset_orig.update(int(v) for v in (cfg.get("specialist_indices") or []))
            subset_orig.update(int(v) for v in (cfg.get("mapping_indices") or []))
    subset_indices = tuple(sorted(subset_orig))

    subset_library = [full_library[idx] for idx in subset_indices]
    subset_lookup = build_library_lookup(subset_library)
    full_lookup = build_library_lookup(full_library)
    subset_pos_by_orig = {int(orig): int(pos) for pos, orig in enumerate(subset_indices)}

    local_baseline_pair_configs = copy.deepcopy(baseline_pair_configs)
    for pair in pairs:
        cfg = local_baseline_pair_configs[pair]
        cfg["specialist_indices"] = [
            int(subset_pos_by_orig[idx])
            for idx in (cfg.get("specialist_indices") or [])
            if int(idx) in subset_pos_by_orig
        ]
        cfg["mapping_indices"] = [
            int(subset_pos_by_orig[idx])
            for idx in (cfg.get("mapping_indices") or [])
            if int(idx) in subset_pos_by_orig
        ]

    model, _ = load_model(Path(exp["model_path"]))
    compiled = gp.toolbox.compile(expr=model)
    start_all = SEARCH_WINDOWS[-1][1]
    end_all = SEARCH_WINDOWS[0][2]
    df_all = gp.load_all_pairs(pairs=list(pairs), start=start_all, end=end_all, refresh_cache=False)
    raw_signal_all = {
        pair: pd.Series(
            compiled(*gp.get_feature_arrays(df_all, pair)),
            index=df_all.index,
            dtype="float64",
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        for pair in pairs
    }
    funding_all = {pair: load_or_fetch_funding(pair, start_all, end_all) for pair in pairs}

    def build_cache(library_lookup: dict[str, object], labels: tuple[str, ...]) -> dict[str, dict[str, object]]:
        cache: dict[str, dict[str, object]] = {}
        for label, start, end in SEARCH_WINDOWS:
            if label not in labels:
                continue
            df = df_all.loc[start:end].copy()
            pair_cache = {}
            for pair in pairs:
                overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)
                signal_slice = raw_signal_all[pair].loc[start:end].copy()
                funding_slice = funding_all[pair]
                if not funding_slice.empty:
                    funding_slice = funding_slice[
                        (funding_slice["fundingTime"] >= pd.Timestamp(start, tz="UTC"))
                        & (funding_slice["fundingTime"] <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
                    ].copy()
                pair_cache[pair] = {
                    "fast_context": build_fast_context(
                        df=df,
                        pair=pair,
                        raw_signal=signal_slice,
                        overlay_inputs=overlay_inputs,
                        route_thresholds=route_thresholds,
                        library_lookup=library_lookup,
                        funding_df=funding_slice,
                        route_state_mode=ROUTE_STATE_MODE_EQUITY_CORR,
                    )
                }
            cache[label] = {"pairs": pair_cache}
        return cache

    subset_window_cache = build_cache(
        subset_lookup,
        ("recent_2m", "recent_4m", "recent_6m", "recent_1y", "full_4y"),
    )
    full_window_cache = build_cache(full_lookup, ("recent_2m", "recent_6m", "full_4y"))

    started = time.perf_counter()
    role_pools_local = build_role_specific_specialist_pools(
        pairs=pairs,
        library=subset_library,
        library_lookup=subset_lookup,
        window_cache=subset_window_cache,
        baseline_pair_configs=local_baseline_pair_configs,
        fast_engine="numba",
        pool_size=12,
    )
    role_pool_seconds = round(time.perf_counter() - started, 2)

    updated = copy.deepcopy(selected)
    role_pools_global: dict[str, dict[str, list[int]]] = {}
    for pair in pairs:
        role_pools_global[pair] = {}
        for role in SPECIALIST_ROLE_NAMES:
            translated = [int(subset_indices[idx]) for idx in role_pools_local[pair][role]]
            role_pools_global[pair][role] = translated
        top_global = [role_pools_global[pair][role][0] for role in SPECIALIST_ROLE_NAMES]
        state_specialists = [int(v) for v in updated["pair_configs"][pair].get("state_specialists") or []]
        updated["pair_configs"][pair]["specialist_indices"] = top_global
        updated["pair_configs"][pair]["mapping_indices"] = [int(top_global[idx]) for idx in state_specialists]

    main_periods: dict[str, dict[str, float]] = {}
    for item in main_backtest.get("periods") or []:
        if item.get("label") not in {"2m", "6m", "4y"}:
            continue
        agg = item.get("aggregate") or {}
        per_pair = item.get("per_pair") or {}
        daily_wins = [float(metrics.get("daily_win_rate", 0.0)) for metrics in per_pair.values()] or [0.0]
        main_periods[str(item["label"])] = {
            "mean_total_return": float(agg.get("mean_total_return", 0.0)),
            "worst_pair_total_return": float(agg.get("worst_pair_total_return", 0.0)),
            "worst_max_drawdown": float(agg.get("worst_max_drawdown", 0.0)),
            "mean_daily_win_rate": float(np.mean(daily_wins)),
            "worst_pair_daily_win_rate": float(np.min(daily_wins)),
        }

    payload = {
        "subset_size": int(len(subset_indices)),
        "subset_indices": [int(v) for v in subset_indices],
        "role_pool_seconds": float(role_pool_seconds),
        "role_pool_top3_global": {
            pair: {role: values[:3] for role, values in role_pools_global[pair].items()}
            for pair in pairs
        },
        "selected_periods": _collect_compare_metrics(
            selected,
            pairs=pairs,
            full_lookup=full_lookup,
            full_window_cache=full_window_cache,
        ),
        "updated_periods": _collect_compare_metrics(
            updated,
            pairs=pairs,
            full_lookup=full_lookup,
            full_window_cache=full_window_cache,
        ),
        "main_periods": main_periods,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
