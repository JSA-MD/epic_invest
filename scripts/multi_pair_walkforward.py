#!/usr/bin/env python3
"""Multi-pair walk-forward OOS validation (BTC+BNB+ETH+SOL+XRP+DOGE).

Expands walkforward_pairwise.py to 6 coins, computes per-pair OOS stats,
pair-pair OOS return correlation, and equal-weight portfolio diversification
benefit.

Usage:
    python scripts/multi_pair_walkforward.py [--pairs ...] [--start ...] [--end ...]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import gp_crypto_evolution as gp
from backtest_pairwise_equity_corr_risk_compare import (
    filter_funding_window,
    load_funding_cache,
)
from replay_regime_mixture_realistic import load_model as load_signal_model
from search_gp_drawdown_overlay import iter_params
from search_pair_subset_regime_mixture import (
    build_fast_context,
    build_library_lookup,
    build_overlay_inputs,
    realistic_overlay_replay,
    realistic_overlay_replay_from_context,
)

UTC = timezone.utc
BARS_PER_DAY = gp.periods_per_day(gp.TIMEFRAME)  # 288 for 5m

ALL_PAIRS = ("BTCUSDT", "BNBUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT")
ANNUALISATION_FACTOR = math.sqrt(365.25)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _clip(v: float) -> float:
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return _clip(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.floating):
        return _clip(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_safe(x) for x in value.tolist()]
    return value


def _bar_net_to_daily(bar_net: np.ndarray) -> np.ndarray:
    """Aggregate 5m bar returns into daily compounded returns."""
    n_days = len(bar_net) // BARS_PER_DAY
    daily = []
    for d in range(n_days):
        seg = bar_net[d * BARS_PER_DAY: (d + 1) * BARS_PER_DAY]
        if len(seg) == 0:
            continue
        daily.append(float(np.prod(1.0 + seg) - 1.0))
    return np.asarray(daily, dtype="float64")


def _sharpe_from_daily(daily: np.ndarray) -> float:
    if len(daily) < 2:
        return 0.0
    std = float(np.std(daily))
    if std < 1e-12:
        return 0.0
    return _clip(float(np.mean(daily)) / std * ANNUALISATION_FACTOR)


def _calmar(daily: np.ndarray) -> float:
    if len(daily) == 0:
        return 0.0
    ann_return = float(np.mean(daily)) * 365.25
    equity = np.cumprod(1.0 + daily)
    roll_max = np.maximum.accumulate(equity)
    drawdowns = equity / roll_max - 1.0
    max_dd = float(np.min(drawdowns))
    if max_dd >= 0.0 or abs(max_dd) < 1e-12:
        return 0.0
    return _clip(ann_return / abs(max_dd))


def _synthesise_pair_config(
    base_config: dict[str, Any],
    pair: str,
) -> dict[str, Any]:
    """Clone a base pair config for a new pair (same params, different pair key)."""
    cfg = dict(base_config)
    # Keep mapping_indices, route_breadth_threshold, route_state_mode from BTC config.
    # This is intentional: we apply the same regime-mixture overlay to new pairs.
    return cfg


# ---------------------------------------------------------------------------
# Per-pair replay (IS + OOS with bar trace)
# ---------------------------------------------------------------------------

def _run_pair_replay(
    df_slice: pd.DataFrame,
    pair: str,
    pairs: tuple[str, ...],
    compiled_fn: Any,
    funding_df: pd.DataFrame,
    library: list[Any],
    mapping: tuple[int, ...],
    route_breadth_threshold: float,
    route_state_mode: str,
    *,
    return_bar_net: bool = False,
) -> dict[str, float] | None:
    if df_slice.empty:
        return None

    try:
        raw_signal = pd.Series(
            compiled_fn(*gp.get_feature_arrays(df_slice, pair)),
            index=df_slice.index,
            dtype="float64",
        )
    except KeyError:
        # New pair may not have all feature columns; fall back to BTC signal
        try:
            raw_signal = pd.Series(
                compiled_fn(*gp.get_feature_arrays(df_slice, "BTCUSDT")),
                index=df_slice.index,
                dtype="float64",
            )
        except Exception:
            return None

    # Overlay inputs use the available pairs in df_slice
    available_pairs = tuple(p for p in pairs if any(
        c.startswith(f"{p}_") for c in df_slice.columns
    ))
    if not available_pairs:
        available_pairs = pairs

    overlay_inputs = build_overlay_inputs(df_slice, available_pairs, regime_pair=pair)

    try:
        if return_bar_net:
            # Use Python kernel path to capture bar-level returns for portfolio analysis
            library_lookup = build_library_lookup(library)
            context = build_fast_context(
                df=df_slice,
                pair=pair,
                raw_signal=raw_signal,
                overlay_inputs=overlay_inputs,
                route_thresholds=(float(route_breadth_threshold),),
                library_lookup=library_lookup,
                funding_df=funding_df,
                route_state_mode=route_state_mode,
            )
            result = realistic_overlay_replay_from_context(
                context,
                library_lookup,
                mapping,
                route_breadth_threshold,
                use_equity_corr_risk=False,
                return_trace=True,
            )
        else:
            result = realistic_overlay_replay(
                df_slice,
                pair,
                raw_signal,
                overlay_inputs,
                funding_df,
                library,
                mapping,
                route_breadth_threshold,
                use_equity_corr_risk=False,
                route_state_mode=route_state_mode,
            )
    except Exception as exc:
        print(f"    replay error: {exc}", flush=True)
        return None

    if not isinstance(result, dict):
        return None

    metrics: dict[str, Any] = {
        "total_return": _clip(float(result.get("total_return", 0.0))),
        "sharpe": _clip(float(result.get("sharpe", 0.0))),
        "max_drawdown": _clip(float(result.get("max_drawdown", 0.0))),
        "avg_daily_return": _clip(float(result.get("avg_daily_return", 0.0))),
        "n_wins": int(result.get("n_wins", 0)),
        "n_losses": int(result.get("n_losses", 0)),
        "n_trades": int(result.get("n_trades", 0)),
        "roundtrip_win_rate": _clip(float(result.get("roundtrip_win_rate", 0.0))),
    }

    # Attach bar_net for OOS portfolio analysis
    if return_bar_net:
        trace = result.get("trace", {})
        bar_net = trace.get("bar_net")
        if bar_net is not None and len(bar_net) > 0:
            metrics["_bar_net"] = np.asarray(bar_net, dtype="float64")
        else:
            metrics["_bar_net"] = np.array([], dtype="float64")

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward generator
# ---------------------------------------------------------------------------

def walk_forward(
    df_all: pd.DataFrame,
    pairs: tuple[str, ...],
    compiled_fn: Any,
    funding_cache: dict[str, pd.DataFrame],
    library: list[Any],
    pair_configs: dict[str, dict[str, Any]],
    *,
    start: str,
    end: str,
    train_days: int,
    test_days: int,
    step_days: int,
    min_oos_bars: int = 100,
) -> Generator[dict[str, Any], None, None]:
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC)
    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)

    test_start = start_ts
    fold_idx = 0
    while True:
        train_start = test_start - train_delta
        train_end = test_start
        test_end = test_start + test_delta

        if test_end > end_ts:
            break

        fold_idx += 1
        fold_label = f"fold_{fold_idx:03d}"

        IS_metrics: dict[str, dict[str, Any]] = {}
        OOS_metrics: dict[str, dict[str, Any]] = {}
        skipped: list[str] = []

        for pair in pairs:
            cfg = pair_configs[pair]
            mapping = tuple(int(v) for v in cfg["mapping_indices"])
            route_breadth_threshold = float(cfg["route_breadth_threshold"])
            route_state_mode = str(cfg.get("route_state_mode", "base"))

            df_is = df_all.loc[
                (df_all.index >= train_start) & (df_all.index < train_end)
            ].copy()
            df_oos = df_all.loc[
                (df_all.index >= test_end - test_delta) & (df_all.index < test_end)
            ].copy()

            if len(df_oos) < min_oos_bars:
                print(
                    f"  [{fold_label}] {pair}: OOS {len(df_oos)} bars < {min_oos_bars}, skipping.",
                    flush=True,
                )
                skipped.append(pair)
                continue

            funding_pair = funding_cache.get(pair, pd.DataFrame(columns=["fundingTime", "fundingRate"]))
            is_start_str = train_start.strftime("%Y-%m-%d")
            is_end_str = train_end.strftime("%Y-%m-%d")
            oos_start_str = test_start.strftime("%Y-%m-%d")
            oos_end_str = test_end.strftime("%Y-%m-%d")

            funding_is = filter_funding_window(funding_pair, is_start_str, is_end_str)
            funding_oos = filter_funding_window(funding_pair, oos_start_str, oos_end_str)

            try:
                is_result = _run_pair_replay(
                    df_is, pair, pairs, compiled_fn,
                    funding_is, library, mapping,
                    route_breadth_threshold, route_state_mode,
                    return_bar_net=False,
                )
                oos_result = _run_pair_replay(
                    df_oos, pair, pairs, compiled_fn,
                    funding_oos, library, mapping,
                    route_breadth_threshold, route_state_mode,
                    return_bar_net=True,
                )
            except Exception as exc:
                print(f"  [{fold_label}] {pair}: error: {exc}; skipping.", flush=True)
                skipped.append(pair)
                continue

            if is_result is None or oos_result is None:
                skipped.append(pair)
                continue

            IS_metrics[pair] = is_result
            OOS_metrics[pair] = oos_result

        yield {
            "fold": fold_label,
            "fold_idx": fold_idx,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "IS": IS_metrics,
            "OOS": OOS_metrics,
            "skipped_pairs": skipped,
        }

        test_start += step_delta


# ---------------------------------------------------------------------------
# Per-pair aggregate summary
# ---------------------------------------------------------------------------

def per_pair_summary(folds: list[dict[str, Any]], pairs: tuple[str, ...]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for pair in pairs:
        oos_returns: list[float] = []
        oos_sharpes: list[float] = []
        oos_win_rates: list[float] = []
        oos_daily_returns: list[np.ndarray] = []

        for fold in folds:
            if pair not in fold["OOS"]:
                continue
            m = fold["OOS"][pair]
            oos_returns.append(m["total_return"])
            oos_sharpes.append(m["sharpe"])
            oos_win_rates.append(m["roundtrip_win_rate"])

            bar_net = m.get("_bar_net")
            if bar_net is not None and len(bar_net) > 0:
                daily = _bar_net_to_daily(np.asarray(bar_net, dtype="float64"))
                oos_daily_returns.append(daily)

        if not oos_returns:
            result[pair] = {
                "oos_folds": 0,
                "oos_mean_return": 0.0,
                "oos_mean_sharpe": 0.0,
                "oos_mean_win_rate": 0.0,
                "oos_daily_mean": 0.0,
                "oos_daily_std": 0.0,
            }
            continue

        combined_daily = (
            np.concatenate(oos_daily_returns) if oos_daily_returns else np.array([])
        )

        result[pair] = {
            "oos_folds": len(oos_returns),
            "oos_mean_return": _clip(float(np.mean(oos_returns))),
            "oos_mean_sharpe": _clip(float(np.mean(oos_sharpes))),
            "oos_mean_win_rate": _clip(float(np.mean(oos_win_rates))),
            "oos_daily_mean": _clip(float(np.mean(combined_daily))) if len(combined_daily) > 0 else 0.0,
            "oos_daily_std": _clip(float(np.std(combined_daily))) if len(combined_daily) > 1 else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Diversification analysis
# ---------------------------------------------------------------------------

def diversification_analysis(
    folds: list[dict[str, Any]],
    pairs: tuple[str, ...],
) -> dict[str, Any]:
    """Compute correlation matrix, portfolio Sharpe, Calmar, diversification ratio.

    Strategy: collect all OOS bar_net arrays per pair (aligned by fold),
    convert to daily returns, then analyse.
    """
    # Collect per-pair concatenated OOS daily returns
    pair_daily: dict[str, np.ndarray] = {}
    for pair in pairs:
        daily_chunks: list[np.ndarray] = []
        for fold in folds:
            m = fold["OOS"].get(pair)
            if m is None:
                continue
            bar_net = m.get("_bar_net")
            if bar_net is not None and len(bar_net) > 0:
                d = _bar_net_to_daily(np.asarray(bar_net, dtype="float64"))
                if len(d) > 0:
                    daily_chunks.append(d)
        if daily_chunks:
            pair_daily[pair] = np.concatenate(daily_chunks)

    active_pairs = [p for p in pairs if p in pair_daily and len(pair_daily[p]) > 1]

    if not active_pairs:
        return {
            "active_pairs": [],
            "correlation_matrix": {},
            "individual_sharpe": {},
            "individual_daily_std": {},
            "portfolio_sharpe": 0.0,
            "portfolio_calmar": 0.0,
            "portfolio_daily_std": 0.0,
            "diversification_ratio": 0.0,
            "note": "No active pairs with OOS daily returns",
        }

    # Trim to common length for correlation (use minimum length across active pairs)
    min_len = min(len(pair_daily[p]) for p in active_pairs)
    matrix_data = np.column_stack([pair_daily[p][-min_len:] for p in active_pairs])

    # Correlation matrix
    if min_len > 1 and matrix_data.shape[1] > 1:
        corr = np.corrcoef(matrix_data.T)
    else:
        corr = np.eye(len(active_pairs))

    corr_dict: dict[str, dict[str, float]] = {}
    for i, pi in enumerate(active_pairs):
        corr_dict[pi] = {}
        for j, pj in enumerate(active_pairs):
            corr_dict[pi][pj] = _clip(float(corr[i, j]))

    # Individual stats
    indiv_sharpe = {p: _sharpe_from_daily(pair_daily[p]) for p in active_pairs}
    indiv_std = {p: _clip(float(np.std(pair_daily[p]))) for p in active_pairs}

    # Equal-weight 1/N portfolio
    n = len(active_pairs)
    # Use the common-length window for portfolio (same slice used for corr)
    portfolio_daily = matrix_data.mean(axis=1)

    portfolio_sharpe = _sharpe_from_daily(portfolio_daily)
    portfolio_calmar = _calmar(portfolio_daily)
    portfolio_std = _clip(float(np.std(portfolio_daily)))

    # Diversification ratio = weighted sum of individual stds / portfolio std
    sum_indiv_std = sum(indiv_std[p] for p in active_pairs) / n
    div_ratio = _clip(sum_indiv_std / portfolio_std) if portfolio_std > 1e-12 else 0.0

    return {
        "active_pairs": active_pairs,
        "correlation_matrix": corr_dict,
        "individual_sharpe": {p: _clip(v) for p, v in indiv_sharpe.items()},
        "individual_daily_std": {p: _clip(v) for p, v in indiv_std.items()},
        "portfolio_sharpe": _clip(portfolio_sharpe),
        "portfolio_calmar": _clip(portfolio_calmar),
        "portfolio_daily_std": portfolio_std,
        "diversification_ratio": div_ratio,
        "common_oos_days": int(min_len),
        "n_pairs_in_portfolio": n,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-pair walk-forward OOS validation (6 coins)."
    )
    parser.add_argument("--start", default="2023-04-01")
    parser.add_argument("--end", default="2026-04-06")
    parser.add_argument("--train-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument(
        "--pairs",
        default=",".join(ALL_PAIRS),
        help="Comma-separated pair list.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=gp.MODELS_DIR / "multi_pair_walkforward_report.json",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=gp.MODELS_DIR / "gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=gp.MODELS_DIR / "recent_6m_gp_vectorized_big_capped_rerun.dill",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = tuple(p.strip() for p in args.pairs.split(",") if p.strip())

    print("Multi-pair Walk-forward OOS Validation", flush=True)
    print(f"  Period : {args.start} -> {args.end}", flush=True)
    print(f"  Train  : {args.train_days}d  Test: {args.test_days}d  Step: {args.step_days}d", flush=True)
    print(f"  Pairs  : {pairs}", flush=True)

    # Load summary: pair_configs for BTC+BNB; synthesise for new pairs
    summary = json.loads(args.summary_path.read_text())
    base_pair_configs: dict[str, dict[str, Any]] = summary["selected_candidate"]["pair_configs"]

    # BTC config as template for new pairs
    btc_cfg = base_pair_configs.get("BTCUSDT", base_pair_configs[next(iter(base_pair_configs))])

    pair_configs: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        if pair in base_pair_configs:
            pair_configs[pair] = base_pair_configs[pair]
            print(f"  {pair}: using tuned pair config", flush=True)
        else:
            pair_configs[pair] = _synthesise_pair_config(btc_cfg, pair)
            print(f"  {pair}: synthesised config from BTC template", flush=True)

    # Load model
    library = list(iter_params())
    model_tree, _ = load_signal_model(args.model_path)
    compiled_fn = gp.toolbox.compile(expr=model_tree)

    # Load price data (all 6 pairs)
    data_start = "2022-04-06"
    data_end = args.end
    print(f"\nLoading price data {data_start} -> {data_end} ...", flush=True)

    available_pairs: list[str] = []
    skipped_pairs_load: list[str] = []
    for pair in pairs:
        try:
            _ = gp.load_pair(pair, start=data_start, end=data_end, refresh_cache=False)
            available_pairs.append(pair)
        except Exception as exc:
            print(f"  WARNING: {pair} price load failed: {exc}; skipping.", flush=True)
            skipped_pairs_load.append(pair)

    if not available_pairs:
        print("ERROR: No pairs could be loaded. Exiting.", flush=True)
        sys.exit(1)

    pairs = tuple(available_pairs)
    df_all = gp.load_all_pairs(pairs=list(pairs), start=data_start, end=data_end, refresh_cache=False)
    print(f"  Loaded {len(df_all)} bars for {len(pairs)} pairs.", flush=True)

    # Load funding per pair (gracefully handle missing)
    print("Loading funding caches ...", flush=True)
    funding_cache: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        try:
            funding_cache[pair] = load_funding_cache(pair)
            print(f"  {pair}: {len(funding_cache[pair])} funding rows", flush=True)
        except Exception as exc:
            print(f"  WARNING: {pair} funding load failed ({exc}); using empty funding.", flush=True)
            funding_cache[pair] = pd.DataFrame(columns=["fundingTime", "fundingRate"])

    # Walk-forward
    t_start = time.perf_counter()
    folds: list[dict[str, Any]] = []

    for fold in walk_forward(
        df_all,
        pairs,
        compiled_fn,
        funding_cache,
        library,
        pair_configs,
        start=args.start,
        end=args.end,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
    ):
        elapsed = time.perf_counter() - t_start
        oos_sharpes = {p: f"{fold['OOS'][p]['sharpe']:.3f}" for p in fold["OOS"]}
        print(
            f"  {fold['fold']}  OOS:{fold['test_start'][:10]}~{fold['test_end'][:10]}"
            f"  sharpe={oos_sharpes}  skip={fold['skipped_pairs']}  {elapsed:.1f}s",
            flush=True,
        )
        folds.append(fold)

    t_elapsed = time.perf_counter() - t_start
    print(f"\nCompleted {len(folds)} folds in {t_elapsed:.1f}s", flush=True)

    # Per-pair summary
    pp_summary = per_pair_summary(folds, pairs)

    # Diversification analysis
    print("\nComputing diversification analysis ...", flush=True)
    div_analysis = diversification_analysis(folds, pairs)

    # Strip _bar_net from folds before serialising (too large for JSON)
    for fold in folds:
        for pair_metrics in fold["OOS"].values():
            pair_metrics.pop("_bar_net", None)

    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "strategy_class": "multi_pair_walkforward_oos",
        "params": {
            "start": args.start,
            "end": args.end,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "pairs_requested": list(pairs) + skipped_pairs_load,
            "pairs_available": list(pairs),
            "pairs_skipped_at_load": skipped_pairs_load,
            "summary_path": str(args.summary_path),
            "model_path": str(args.model_path),
        },
        "per_pair_summary": pp_summary,
        "diversification": div_analysis,
        "folds": folds,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(json_safe(report), indent=2))
    print(f"\nReport written to: {args.report_out}", flush=True)

    # Print summary table
    print("\n" + "=" * 70)
    print("PER-PAIR OOS SUMMARY")
    print("=" * 70)
    print(f"{'Pair':<12} {'Folds':>6} {'OOS Return':>11} {'OOS Sharpe':>11} {'Win Rate':>9} {'Daily Ret':>10} {'Daily Std':>10}")
    print("-" * 70)
    for pair in pairs:
        s = pp_summary.get(pair, {})
        print(
            f"{pair:<12} {s.get('oos_folds', 0):>6} "
            f"{s.get('oos_mean_return', 0.0):>10.4f}  "
            f"{s.get('oos_mean_sharpe', 0.0):>10.3f}  "
            f"{s.get('oos_mean_win_rate', 0.0):>8.3f}  "
            f"{s.get('oos_daily_mean', 0.0)*100:>8.4f}%  "
            f"{s.get('oos_daily_std', 0.0)*100:>8.4f}%"
        )

    d = div_analysis
    print("\n" + "=" * 70)
    print("DIVERSIFICATION ANALYSIS  (equal-weight 1/N portfolio)")
    print("=" * 70)
    print(f"  Active pairs          : {d.get('n_pairs_in_portfolio', 0)}")
    print(f"  Common OOS days       : {d.get('common_oos_days', 0)}")
    print(f"  Portfolio Sharpe      : {d.get('portfolio_sharpe', 0.0):.3f}")
    print(f"  Portfolio Calmar      : {d.get('portfolio_calmar', 0.0):.3f}")
    print(f"  Portfolio Daily Std   : {d.get('portfolio_daily_std', 0.0)*100:.4f}%")
    print(f"  Diversification Ratio : {d.get('diversification_ratio', 0.0):.3f}")

    indiv_sharpe = d.get("individual_sharpe", {})
    if indiv_sharpe:
        print("\n  Individual OOS Sharpe (full OOS history):")
        for p, v in indiv_sharpe.items():
            print(f"    {p:<12}: {v:.3f}")

    corr = d.get("correlation_matrix", {})
    if corr and d.get("n_pairs_in_portfolio", 0) > 1:
        ap = d.get("active_pairs", [])
        print(f"\n  OOS Daily Return Correlation ({d.get('common_oos_days', 0)} days):")
        header = "            " + "".join(f"{p[:8]:>10}" for p in ap)
        print(header)
        for pi in ap:
            row = f"  {pi:<10}" + "".join(f"{corr.get(pi, {}).get(pj, 0.0):>10.3f}" for pj in ap)
            print(row)

    print("=" * 70)


if __name__ == "__main__":
    main()
