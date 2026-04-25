#!/usr/bin/env python3
"""López de Prado Probability of Backtest Overfitting (PBO) and Deflated Sharpe Ratio (DSR).

References:
  Bailey & López de Prado (2014) – "The Deflated Sharpe Ratio"
  López de Prado (2018) – "Advances in Financial Machine Learning", Ch. 11 (CSCV)

Method:
  A. DSR – penalises the observed Sharpe for higher-order moments and for the
     expected maximum Sharpe among N independent trials.
  B. PBO via Combinatorial Symmetric Cross-Validation (CSCV) – split bar_net
     returns into S blocks, enumerate all C(S, S/2) IS/OOS splits, measure what
     fraction of splits yield an IS-best strategy that ranks in the lower half OOS.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap project path so imports work when invoked directly
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import gp_crypto_evolution as gp
from pairwise_regime_live import DEFAULT_MODEL_PATH, DEFAULT_SUMMARY_PATH, PAIRS
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    DEFAULT_WINDOWS,
    ROUTE_STATE_MODE_EQUITY_CORR,
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    load_or_fetch_funding,
    realistic_overlay_replay_from_context,
)


UTC = timezone.utc
_NORM = NormalDist()

# Number of independent trials to assume when computing the DSR threshold
# (represents the search space: corr_risk variants, stop_loss variants,
# regime_mixture mapping variants seen in the repo).
DEFAULT_N_TRIALS = 20


# ---------------------------------------------------------------------------
# DSR helpers
# ---------------------------------------------------------------------------

def _euler_gamma() -> float:
    return 0.5772156649015329


def expected_max_sharpe(n_trials: int) -> float:
    """E[max Sharpe | N i.i.d. trials] using the Gumbel approximation (Bailey & LdP 2014)."""
    n = max(2, int(n_trials))
    a = _NORM.inv_cdf(1.0 - 1.0 / n)
    b = _NORM.inv_cdf(1.0 - 1.0 / (n * 2.718281828))
    return float((1.0 - _euler_gamma()) * a + _euler_gamma() * b)


def deflated_sharpe_ratio(
    bar_net: np.ndarray,
    *,
    n_trials: int,
    annualisation: float,
) -> dict[str, float]:
    """Compute DSR and supporting statistics for a bar-net return series.

    Parameters
    ----------
    bar_net:
        Per-bar net returns (not annualised).
    n_trials:
        Number of independent strategy configurations evaluated during search.
    annualisation:
        sqrt(bars_per_year) used for Sharpe annualisation.
    """
    returns = np.asarray(bar_net, dtype="float64")
    T = len(returns)
    if T < 4:
        return {"SR_obs": 0.0, "T": T, "skew": 0.0, "kurt_excess": 0.0,
                "DSR": 0.0, "SR0_threshold": 0.0, "threshold_p": 0.0}

    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    if sigma < 1e-14:
        return {"SR_obs": 0.0, "T": T, "skew": 0.0, "kurt_excess": 0.0,
                "DSR": 0.0, "SR0_threshold": 0.0, "threshold_p": 0.0}

    standardised = (returns - mu) / sigma
    skew = float(np.mean(standardised ** 3))
    kurt_raw = float(np.mean(standardised ** 4))
    kurt_excess = kurt_raw - 3.0

    SR_obs_period = mu / sigma          # per-bar Sharpe (not annualised yet)
    SR_obs = SR_obs_period * annualisation

    SR0 = expected_max_sharpe(n_trials)  # annualised benchmark

    # Variance of the Sharpe ratio estimator (Mertens 2002 / Bailey & LdP 2014)
    # Var(SR) ≈ (1/T) * (1 - γ3*SR + (γ4-1)/4 * SR²)
    # We compute the z-score in per-bar units (divide SR0 by annualisation to get period units)
    SR0_period = SR0 / annualisation
    variance_term = max(1.0 - skew * SR_obs_period + (kurt_excess / 4.0) * SR_obs_period ** 2, 1e-8)
    SR_std = float(np.sqrt(variance_term / max(T - 1, 1)))

    z = (SR_obs_period - SR0_period) / SR_std
    DSR = float(_NORM.cdf(z))

    return {
        "SR_obs": round(SR_obs, 6),
        "T": T,
        "skew": round(skew, 6),
        "kurt_excess": round(kurt_excess, 6),
        "DSR": round(DSR, 6),
        "SR0_threshold": round(SR0, 6),
        "threshold_p": round(_NORM.cdf(0.0), 6),  # 0.5 – DSR > 0.5 means SR_obs > SR0
    }


# ---------------------------------------------------------------------------
# CSCV / PBO helpers
# ---------------------------------------------------------------------------

def _split_into_blocks(
    series_dict: dict[str, np.ndarray], n_blocks: int
) -> list[np.ndarray]:
    """Return index ranges (as lists) for each block, aligned to the first series."""
    first = next(iter(series_dict.values()))
    T = len(first)
    block_size = T // n_blocks
    blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else T
        blocks.append(np.arange(start, end))
    return blocks


def _annualised_sharpe(returns: np.ndarray, annualisation: float) -> float:
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std < 1e-14:
        return 0.0
    return float(np.mean(returns) / std * annualisation)


def compute_pbo_cscv(
    strategy_returns: dict[str, np.ndarray],
    n_blocks: int,
    annualisation: float,
) -> dict[str, Any]:
    """Compute PBO via Combinatorial Symmetric Cross-Validation.

    For each C(S, S/2) split of S blocks into IS and OOS:
    1. Rank strategies by IS Sharpe; identify IS-best.
    2. Rank strategies by OOS Sharpe; identify OOS rank of the IS-best.
    3. PBO = fraction of splits where IS-best is in the lower OOS half.
    """
    keys = list(strategy_returns.keys())
    M = len(keys)
    if M < 2:
        return {
            "M_strategies": M,
            "S_blocks": n_blocks,
            "n_combinations_evaluated": 0,
            "PBO": 1.0,
            "avg_is_best_oos_percentile": 0.0,
            "note": "Need at least 2 strategies",
        }

    blocks = _split_into_blocks(strategy_returns, n_blocks)
    n_blocks_actual = len(blocks)
    half = n_blocks_actual // 2

    combos = list(itertools.combinations(range(n_blocks_actual), half))
    below_median_count = 0
    oos_percentiles: list[float] = []
    split_details: list[dict[str, Any]] = []

    for is_block_indices in combos:
        oos_block_indices = tuple(i for i in range(n_blocks_actual) if i not in is_block_indices)
        is_idx = np.concatenate([blocks[i] for i in is_block_indices])
        oos_idx = np.concatenate([blocks[i] for i in oos_block_indices])

        is_sharpes = {k: _annualised_sharpe(strategy_returns[k][is_idx], annualisation) for k in keys}
        oos_sharpes = {k: _annualised_sharpe(strategy_returns[k][oos_idx], annualisation) for k in keys}

        best_is_key = max(is_sharpes, key=lambda k: is_sharpes[k])

        oos_sorted = sorted(keys, key=lambda k: oos_sharpes[k])  # ascending
        oos_rank = oos_sorted.index(best_is_key)  # 0 = worst OOS
        oos_percentile = oos_rank / max(M - 1, 1)   # 0 = worst, 1 = best
        oos_percentiles.append(oos_percentile)
        if oos_percentile < 0.5:
            below_median_count += 1

        split_details.append({
            "is_blocks": list(is_block_indices),
            "oos_blocks": list(oos_block_indices),
            "best_is_strategy": best_is_key,
            "best_is_sharpe": round(is_sharpes[best_is_key], 4),
            "oos_sharpe_of_best_is": round(oos_sharpes[best_is_key], 4),
            "oos_rank": int(oos_rank),
            "oos_percentile": round(oos_percentile, 4),
        })

    n_combos = len(combos)
    pbo = below_median_count / max(n_combos, 1)
    avg_percentile = float(np.mean(oos_percentiles)) if oos_percentiles else 0.0

    return {
        "M_strategies": M,
        "S_blocks": n_blocks_actual,
        "n_combinations_evaluated": n_combos,
        "PBO": round(pbo, 6),
        "avg_is_best_oos_percentile": round(avg_percentile, 6),
        "strategy_keys": keys,
        "splits": split_details,
    }


# ---------------------------------------------------------------------------
# Strategy configuration helpers
# ---------------------------------------------------------------------------

def _build_strategy_variants(
    base_mapping_btc: tuple[int, ...],
    base_mapping_bnb: tuple[int, ...],
    route_breadth_threshold: float,
    route_state_mode: str,
) -> dict[str, dict[str, Any]]:
    """Build a set of M strategy configurations by perturbing mapping indices.

    Variants:
      baseline         – the selected candidate
      corr_risk        – baseline with use_equity_corr_risk=True
      map_shift_+1     – each mapping index +1 (mod library size) on BTC
      map_shift_-1     – each mapping index -1 (mod library size) on BTC
      stop2pct         – tighter kill switch: not a mapping variant but a distinct
                         parameter regime (encoded as corr_risk=True + shifted map)
      bnb_shift_+1     – BNB mapping shift +1
      bnb_shift_-1     – BNB mapping shift -1

    Each dict has keys: pair -> {mapping, route_breadth_threshold, use_equity_corr_risk, route_state_mode}
    """
    library = list(iter_params())
    lib_size = len(library)

    def shift(m: tuple[int, ...], delta: int) -> tuple[int, ...]:
        return tuple((v + delta) % lib_size for v in m)

    variants: dict[str, dict[str, Any]] = {}

    def add(name: str, btc_map: tuple[int, ...], bnb_map: tuple[int, ...], corr: bool) -> None:
        variants[name] = {
            "BTCUSDT": {
                "mapping": btc_map,
                "route_breadth_threshold": route_breadth_threshold,
                "use_equity_corr_risk": corr,
                "route_state_mode": route_state_mode,
            },
            "BNBUSDT": {
                "mapping": bnb_map,
                "route_breadth_threshold": route_breadth_threshold,
                "use_equity_corr_risk": corr,
                "route_state_mode": route_state_mode,
            },
        }

    add("baseline",         base_mapping_btc, base_mapping_bnb, False)
    add("corr_risk",        base_mapping_btc, base_mapping_bnb, True)
    add("map_btc_shift_+1", shift(base_mapping_btc, +1), base_mapping_bnb, False)
    add("map_btc_shift_-1", shift(base_mapping_btc, -1), base_mapping_bnb, False)
    add("map_btc_shift_+2", shift(base_mapping_btc, +2), base_mapping_bnb, False)
    add("map_bnb_shift_+1", base_mapping_btc, shift(base_mapping_bnb, +1), False)
    add("map_bnb_shift_-1", base_mapping_btc, shift(base_mapping_bnb, -1), False)
    add("corr_map_shift_+1", shift(base_mapping_btc, +1), shift(base_mapping_bnb, +1), True)

    return variants


# ---------------------------------------------------------------------------
# Per-pair trace runner
# ---------------------------------------------------------------------------

def _run_pair_traces(
    df_all: pd.DataFrame,
    pair: str,
    compiled_model,
    pairs: tuple[str, ...],
    funding_df: pd.DataFrame,
    strategy_variants: dict[str, dict[str, Any]],
    window_label: str,
    window_start: str,
    window_end: str,
) -> dict[str, np.ndarray]:
    """Run all strategy variants for one pair/window, return bar_net traces."""
    library = list(iter_params())
    library_lookup = build_library_lookup(library)

    # Slice the window
    start_ts = pd.Timestamp(window_start, tz="UTC")
    end_ts = pd.Timestamp(window_end, tz="UTC") + pd.Timedelta(days=1)
    df = df_all.loc[(df_all.index >= start_ts) & (df_all.index < end_ts)].copy()
    if df.empty:
        return {}

    raw_signal = pd.Series(
        compiled_model(*gp.get_feature_arrays(df, pair)),
        index=df.index,
        dtype="float64",
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    overlay_inputs = build_overlay_inputs(df, pairs, regime_pair=pair)

    funding_slice = funding_df.loc[
        (funding_df["fundingTime"] >= start_ts) & (funding_df["fundingTime"] < end_ts)
    ].copy() if not funding_df.empty else funding_df

    # Collect unique (route_state_mode, route_breadth_threshold) contexts
    route_state_modes = {v[pair]["route_state_mode"] for v in strategy_variants.values()}
    route_thresholds = {v[pair]["route_breadth_threshold"] for v in strategy_variants.values()}

    # Build one context per route_state_mode (thresholds bundled inside)
    contexts: dict[str, dict[str, Any]] = {}
    for rsm in route_state_modes:
        ctx = build_fast_context(
            df=df,
            pair=pair,
            raw_signal=raw_signal,
            overlay_inputs=overlay_inputs,
            route_thresholds=tuple(sorted(route_thresholds)),
            library_lookup=library_lookup,
            funding_df=funding_slice,
            route_state_mode=rsm,
        )
        contexts[rsm] = ctx

    traces: dict[str, np.ndarray] = {}
    for variant_name, pair_configs in strategy_variants.items():
        cfg = pair_configs[pair]
        rsm = cfg["route_state_mode"]
        ctx = contexts[rsm]
        result = realistic_overlay_replay_from_context(
            context=ctx,
            library_lookup=library_lookup,
            mapping=tuple(cfg["mapping"]),
            route_breadth_threshold=float(cfg["route_breadth_threshold"]),
            use_equity_corr_risk=bool(cfg["use_equity_corr_risk"]),
            engine="python",
            return_trace=True,
        )
        traces[variant_name] = result["trace"]["bar_net"]

    return traces


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute LdP PBO (CSCV) and Deflated Sharpe Ratio for pairwise strategy.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to pairwise validated summary JSON.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to GP model .dill file.",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=8,
        help="Number of equal-length blocks for CSCV (default 8 → C(8,4)=70 splits).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Number of independent trials assumed for DSR threshold (default 20).",
    )
    parser.add_argument(
        "--window",
        default="full_4y",
        help="Which window label to use for PBO+DSR (default full_4y).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path(_REPO_ROOT / "models" / "pbo_deflated_sharpe_report.json"),
        help="Output JSON report path.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load summary and model
    summary = json.loads(args.summary_path.read_text())
    pair_configs = summary["selected_candidate"]["pair_configs"]
    model_tree, _ = load_signal_model(args.model_path)
    compiled_model = gp.toolbox.compile(expr=model_tree)

    # Extract baseline mapping indices
    pairs = PAIRS  # ("BTCUSDT", "BNBUSDT")
    base_mapping_btc = tuple(int(v) for v in pair_configs["BTCUSDT"]["mapping_indices"])
    base_mapping_bnb = tuple(int(v) for v in pair_configs["BNBUSDT"]["mapping_indices"])
    route_breadth_threshold = float(pair_configs["BTCUSDT"]["route_breadth_threshold"])
    route_state_mode = str(pair_configs["BTCUSDT"].get("route_state_mode", ROUTE_STATE_MODE_EQUITY_CORR))

    strategy_variants = _build_strategy_variants(
        base_mapping_btc, base_mapping_bnb, route_breadth_threshold, route_state_mode
    )

    # Identify target window
    target_window = next(
        (w for w in DEFAULT_WINDOWS if w[0] == args.window), DEFAULT_WINDOWS[-1]
    )
    window_label, window_start, window_end = target_window

    print(f"Loading OHLCV data [{window_start} .. {window_end}] …", flush=True)
    df_all = gp.load_all_pairs(
        pairs=list(pairs),
        start=window_start,
        end=window_end,
        refresh_cache=False,
    )

    # Load funding rates for both pairs
    funding_cache: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        try:
            funding_cache[pair] = load_or_fetch_funding(
                pair, window_start, window_end, require_coverage=False
            )
        except Exception:
            funding_cache[pair] = pd.DataFrame(columns=["fundingTime", "fundingRate"])

    # Bars-per-year annualisation factor
    annualisation = float(np.sqrt(365.25 * 24.0 * 60.0 / 5.0))

    print(f"Running {len(strategy_variants)} strategy variants for {len(pairs)} pairs …", flush=True)

    # Collect bar_net traces per pair across all variants
    pair_traces: dict[str, dict[str, np.ndarray]] = {}
    for pair in pairs:
        print(f"  Pair {pair} …", flush=True)
        pair_traces[pair] = _run_pair_traces(
            df_all=df_all,
            pair=pair,
            compiled_model=compiled_model,
            pairs=pairs,
            funding_df=funding_cache[pair],
            strategy_variants=strategy_variants,
            window_label=window_label,
            window_start=window_start,
            window_end=window_end,
        )

    # ---------------------------------------------------------------------------
    # DSR: computed per (pair, variant) on the full window
    # ---------------------------------------------------------------------------
    print("Computing DSR …", flush=True)
    dsr_section: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        dsr_section[pair] = {}
        for variant_name, bar_net in pair_traces[pair].items():
            dsr_section[pair][f"{variant_name}_{window_label}"] = deflated_sharpe_ratio(
                bar_net,
                n_trials=args.n_trials,
                annualisation=annualisation,
            )

    # ---------------------------------------------------------------------------
    # PBO via CSCV: pool strategies across pairs (average bar_net across pairs
    # so we get a single per-variant return series for the portfolio, then CSCV)
    # ---------------------------------------------------------------------------
    print(f"Computing PBO via CSCV (S={args.n_blocks}, C(S,S/2)={len(list(itertools.combinations(range(args.n_blocks), args.n_blocks//2)))}) …", flush=True)

    variant_names = list(strategy_variants.keys())
    # Align lengths across pairs (they should be equal since same df_all)
    min_len = min(
        min(len(pair_traces[pair][v]) for v in variant_names if v in pair_traces[pair])
        for pair in pairs
        if pair_traces[pair]
    )
    pooled_returns: dict[str, np.ndarray] = {}
    for v in variant_names:
        per_pair = []
        for pair in pairs:
            arr = pair_traces[pair].get(v)
            if arr is not None and len(arr) > 0:
                per_pair.append(arr[:min_len])
        if per_pair:
            pooled_returns[v] = np.mean(np.stack(per_pair, axis=1), axis=1)

    pbo_section = compute_pbo_cscv(
        strategy_returns=pooled_returns,
        n_blocks=args.n_blocks,
        annualisation=annualisation,
    )
    # Remove verbose split details to keep report compact
    split_details = pbo_section.pop("splits", [])
    pbo_section["splits_omitted"] = len(split_details)

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    pbo_val = float(pbo_section.get("PBO", 1.0))
    # Use baseline DSR for both BTC and BNB as key signal
    btc_dsr_baseline = dsr_section.get("BTCUSDT", {}).get(f"baseline_{window_label}", {}).get("DSR", 0.0)
    bnb_dsr_baseline = dsr_section.get("BNBUSDT", {}).get(f"baseline_{window_label}", {}).get("DSR", 0.0)
    avg_dsr = (btc_dsr_baseline + bnb_dsr_baseline) / 2.0

    if pbo_val < 0.3 and avg_dsr > 0.5:
        verdict = "robust"
    elif pbo_val > 0.5 or avg_dsr < 0.1:
        verdict = "likely_overfit"
    else:
        verdict = "moderate"

    interpretation = (
        f"PBO={pbo_val:.3f} (lower is better; <0.30 robust, >0.50 overfit). "
        f"Avg DSR (BTC+BNB baseline)={avg_dsr:.3f} (>0.50 = SR survives trial-count penalty). "
        f"Verdict: {verdict}."
    )
    pbo_section["interpretation"] = interpretation

    # ---------------------------------------------------------------------------
    # Assemble and write report
    # ---------------------------------------------------------------------------
    report: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "method": {
            "dsr": "Bailey & López de Prado (2014) Deflated Sharpe Ratio",
            "pbo": "López de Prado CSCV – Combinatorial Symmetric Cross-Validation",
            "n_trials_assumed": args.n_trials,
            "window": window_label,
            "window_start": window_start,
            "window_end": window_end,
            "n_blocks": args.n_blocks,
            "annualisation_factor": round(annualisation, 4),
        },
        "deflated_sharpe": dsr_section,
        "pbo": pbo_section,
        "verdict": verdict,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False))
    print(f"\nReport written to {args.report_out}", flush=True)

    # ---------------------------------------------------------------------------
    # Summary print
    # ---------------------------------------------------------------------------
    print("\n=== PBO / DSR SUMMARY ===")
    print(f"Window   : {window_label}  ({window_start} → {window_end})")
    print(f"Variants : {len(pooled_returns)}  blocks={args.n_blocks}  combos={pbo_section['n_combinations_evaluated']}")
    print(f"\nDSR (Deflated Sharpe Ratio):")
    for pair in pairs:
        key = f"baseline_{window_label}"
        d = dsr_section.get(pair, {}).get(key, {})
        print(f"  {pair:10s} baseline  SR_obs={d.get('SR_obs', 0):.3f}  "
              f"SR0_threshold={d.get('SR0_threshold', 0):.3f}  "
              f"DSR={d.get('DSR', 0):.3f}  "
              f"T={d.get('T', 0)}  skew={d.get('skew', 0):.3f}  kurt_excess={d.get('kurt_excess', 0):.3f}")
    print(f"\nPBO via CSCV:")
    print(f"  PBO  = {pbo_val:.4f}  (fraction of splits where IS-best ranked bottom half OOS)")
    print(f"  Avg IS-best OOS percentile = {pbo_section.get('avg_is_best_oos_percentile', 0):.4f}")
    print(f"\nVerdict: {verdict.upper()}")
    print(f"  {interpretation}")


if __name__ == "__main__":
    main()
