#!/usr/bin/env python3
"""Sweep no_trade_band_pct to find the Sharpe-maximising trade frequency.

Multipliers relative to gp.NO_TRADE_BAND (default = 10):
    0.25×, 0.50×, 1.0× (default), 2.0×, 4.0×

Uses realistic_overlay_replay_from_context (full_4y window) for BTC + BNB.
Output: models/trade_frequency_optimization_report.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import gp_crypto_evolution as gp
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    ROUTE_STATE_MODE_BASE,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    load_or_fetch_funding,
    normalize_mapping_indices,
    normalize_route_state_mode,
    realistic_overlay_replay_from_context,
    resolve_candidate,
)
from replay_regime_mixture_realistic import load_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAIRS = ("BTCUSDT", "BNBUSDT")
MULTIPLIERS = (0.25, 0.50, 1.0, 2.0, 4.0)
SUMMARY_PATH = gp.MODELS_DIR / "gp_regime_mixture_search_summary.json"
MODEL_PATH = gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill"
REPORT_OUT = gp.MODELS_DIR / "trade_frequency_optimization_report.json"

# Use the full_4y window that the task specifies
FULL_4Y_LABEL = "full_4y"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def build_pair_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the per-pair metrics required by the output schema."""
    # Payoff / kelly may be nested under "trade_stats" or at top level
    trade_stats = result.get("trade_stats", result)
    payoff_ratio = float(trade_stats.get("payoff_ratio", result.get("payoff_ratio", 0.0)))
    kelly = float(trade_stats.get("kelly_fraction", result.get("kelly_fraction", 0.0)))
    daily_win_rate = float(
        result.get("daily_win_rate")
        or result.get("daily_metrics", {}).get("daily_win_rate", 0.0)
    )
    return {
        "n_trades": int(result.get("n_trades", 0)),
        "roi": float(result.get("total_return", 0.0)),
        "sharpe": float(result.get("sharpe", 0.0)),
        "mdd": float(result.get("max_drawdown", 0.0)),
        "fee_paid": float(result.get("fee_paid", 0.0)),
        "daily_win_rate": daily_win_rate,
        "kelly": kelly,
        "payoff_ratio": payoff_ratio,
    }


def run_sweep() -> dict[str, Any]:
    # ------------------------------------------------------------------
    # 1. Load candidate / library / model
    # ------------------------------------------------------------------
    candidate, library, _ = resolve_candidate(SUMMARY_PATH, None, None)
    route_state_mode = normalize_route_state_mode(ROUTE_STATE_MODE_BASE)
    mapping = normalize_mapping_indices(candidate.mapping_indices, route_state_mode)
    route_breadth_threshold = float(candidate.route_breadth_threshold)
    library_lookup = build_library_lookup(library)

    model, _ = load_model(MODEL_PATH)
    compiled = gp.toolbox.compile(expr=model)

    # ------------------------------------------------------------------
    # 2. Load data (full_4y span covers all windows)
    # ------------------------------------------------------------------
    # Find the full_4y window dates from DEFAULT_WINDOWS
    full_4y_window = next(w for w in DEFAULT_WINDOWS if w[0] == FULL_4Y_LABEL)
    _, start_all, end_all = full_4y_window

    print(f"Loading OHLCV data {start_all} → {end_all} …")
    df_all = gp.load_all_pairs(
        pairs=list(PAIRS), start=start_all, end=end_all, refresh_cache=False
    )

    # Raw signals per pair
    raw_signal_all: dict[str, pd.Series] = {}
    for pair in PAIRS:
        raw_signal_all[pair] = (
            pd.Series(
                compiled(*gp.get_feature_arrays(df_all, pair)),
                index=df_all.index,
                dtype="float64",
            )
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )

    # Funding rates per pair
    print("Loading funding rates …")
    funding_all: dict[str, pd.DataFrame] = {}
    for pair in PAIRS:
        funding_all[pair] = load_or_fetch_funding(pair, start_all, end_all)

    # Overlay inputs per pair (build once for the full window)
    print("Building overlay inputs …")
    from derivative_market_data import load_derivative_metric_cache

    def _load_derivative_bundle(symbol: str) -> dict[str, pd.DataFrame]:
        metrics = (
            "open_interest",
            "basis_perpetual",
            "top_trader_position_ratio",
            "taker_buy_sell_ratio",
        )
        return {metric: load_derivative_metric_cache(symbol, metric) for metric in metrics}

    derivatives_all = {pair: _load_derivative_bundle(pair) for pair in PAIRS}

    # route_thresholds must include the candidate's threshold for bucket_codes cache
    route_thresholds = (route_breadth_threshold,)

    # Build fast_context once per pair (expensive; reused across all band variants)
    print("Building fast contexts …")
    fast_contexts: dict[str, Any] = {}
    for pair in PAIRS:
        overlay_inputs = build_overlay_inputs(df_all, PAIRS, regime_pair=pair)
        fast_contexts[pair] = build_fast_context(
            df=df_all,
            pair=pair,
            raw_signal=raw_signal_all[pair],
            overlay_inputs=overlay_inputs,
            route_thresholds=route_thresholds,
            library_lookup=library_lookup,
            funding_df=funding_all[pair],
            derivative_bundle=derivatives_all[pair],
            route_state_mode=route_state_mode,
        )

    # ------------------------------------------------------------------
    # 3. Sweep no_trade_band_pct via monkeypatching gp.NO_TRADE_BAND
    #    (realistic_overlay_replay_from_context reads gp.NO_TRADE_BAND
    #    at call time when execution_gene is None)
    # ------------------------------------------------------------------
    default_band = float(gp.NO_TRADE_BAND)
    print(f"Default no_trade_band: {default_band}")
    print(f"Sweeping multipliers: {MULTIPLIERS}\n")

    variants: list[dict[str, Any]] = []

    for mult in MULTIPLIERS:
        band_pct = default_band * mult
        print(f"  multiplier={mult:.2f}  no_trade_band_pct={band_pct:.2f}")
        per_pair: dict[str, dict[str, Any]] = {}

        # Temporarily patch gp.NO_TRADE_BAND so the legacy path reads our value
        original_band = gp.NO_TRADE_BAND
        gp.NO_TRADE_BAND = band_pct  # type: ignore[assignment]
        try:
            for pair in PAIRS:
                result = realistic_overlay_replay_from_context(
                    fast_contexts[pair],
                    library_lookup,
                    mapping,
                    route_breadth_threshold,
                    use_equity_corr_risk=False,
                    execution_gene=None,
                )
                per_pair[pair] = build_pair_metrics(result)
                print(
                    f"    {pair}: n_trades={per_pair[pair]['n_trades']}  "
                    f"sharpe={per_pair[pair]['sharpe']:.3f}  "
                    f"roi={per_pair[pair]['roi']:.3f}"
                )
        finally:
            gp.NO_TRADE_BAND = original_band  # type: ignore[assignment]

        variants.append(
            {
                "multiplier": float(mult),
                "no_trade_band_pct": float(band_pct),
                "per_pair": per_pair,
            }
        )

    # ------------------------------------------------------------------
    # 4. Identify optimal (Sharpe-max) and knee per pair
    # ------------------------------------------------------------------
    optimal: dict[str, dict[str, Any]] = {}
    for pair in PAIRS:
        sharpes = [v["per_pair"][pair]["sharpe"] for v in variants]
        trades = [v["per_pair"][pair]["n_trades"] for v in variants]
        best_idx = int(np.argmax(sharpes))

        # Knee: largest relative Sharpe gain per unit of trade-count increase
        # (first-derivative drop-off in Sharpe as trades increase, i.e. tightest
        # band that keeps Sharpe within 5% of maximum)
        max_sharpe = sharpes[best_idx]
        knee_idx = best_idx
        for i, sh in enumerate(sharpes):
            if sh >= 0.95 * max_sharpe and trades[i] < trades[knee_idx]:
                knee_idx = i

        optimal[pair] = {
            "multiplier": float(variants[best_idx]["multiplier"]),
            "no_trade_band_pct": float(variants[best_idx]["no_trade_band_pct"]),
            "sharpe": float(sharpes[best_idx]),
            "n_trades": int(trades[best_idx]),
            "sharpe_vs_default": float(sharpes[best_idx] - sharpes[MULTIPLIERS.index(1.0)]),
            "knee_multiplier": float(variants[knee_idx]["multiplier"]),
            "knee_sharpe": float(sharpes[knee_idx]),
            "knee_n_trades": int(trades[knee_idx]),
        }

    # ------------------------------------------------------------------
    # 5. Build trade-count vs Sharpe line for JSON (x: trades, y: sharpe)
    # ------------------------------------------------------------------
    curve: dict[str, list[dict[str, float]]] = {pair: [] for pair in PAIRS}
    for v in variants:
        for pair in PAIRS:
            curve[pair].append(
                {
                    "x": float(v["per_pair"][pair]["n_trades"]),
                    "y": float(v["per_pair"][pair]["sharpe"]),
                    "multiplier": float(v["multiplier"]),
                }
            )

    # ------------------------------------------------------------------
    # 6. Assemble report
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "no_trade_band_default": default_band,
        "summary_path": str(SUMMARY_PATH),
        "model_path": str(MODEL_PATH),
        "window": FULL_4Y_LABEL,
        "pairs": list(PAIRS),
        "variants": variants,
        "optimal": optimal,
        "curve": curve,
    }

    REPORT_OUT.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False))
    print(f"\nReport written to {REPORT_OUT}")
    return report


def main() -> None:
    report = run_sweep()

    print("\n=== OPTIMAL TRADE FREQUENCY ===")
    for pair, opt in report["optimal"].items():
        print(
            f"  {pair}: best_mult={opt['multiplier']:.2f}x "
            f"(band={opt['no_trade_band_pct']:.1f})  "
            f"sharpe={opt['sharpe']:.4f}  "
            f"n_trades={opt['n_trades']}  "
            f"sharpe_lift_vs_default={opt['sharpe_vs_default']:+.4f}  "
            f"knee_mult={opt['knee_multiplier']:.2f}x"
        )


if __name__ == "__main__":
    main()
