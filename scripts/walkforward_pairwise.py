#!/usr/bin/env python3
"""Rolling-window walk-forward (OOS) validation for the pairwise equity-corr-risk strategy.

Addresses the López de Prado critique that fixed train-end windows mask IS/OOS gap.
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

# Ensure scripts/ is on path so relative imports work when invoked from repo root.
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
    build_overlay_inputs,
    realistic_overlay_replay,
)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _clip(v: float) -> float:
    """Replace NaN/Inf with 0 so JSON stays valid."""
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
    return value


def _extract_metrics(result: dict[str, Any]) -> dict[str, float]:
    """Pull the scalar metrics we care about from a replay result dict."""
    n_wins = int(result.get("n_wins", 0))
    n_losses = int(result.get("n_losses", 0))
    return {
        "total_return": _clip(float(result.get("total_return", 0.0))),
        "sharpe": _clip(float(result.get("sharpe", 0.0))),
        "max_drawdown": _clip(float(result.get("max_drawdown", 0.0))),
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_trades": int(result.get("n_trades", 0)),
        "roundtrip_win_rate": _clip(float(result.get("roundtrip_win_rate", 0.0))),
    }


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
    initial_cooldown_bars: int = 0,
    return_final_cooldown: bool = False,
) -> dict[str, float] | None:
    """Run realistic_overlay_replay for one pair on one df slice.

    Returns None if the slice is too thin to produce a meaningful result.

    When return_final_cooldown=True the returned dict gains an extra key
    ``final_cooldown_bars`` with the cooldown state at the last bar, suitable
    for seeding the next walk-forward window.
    """
    if df_slice.empty:
        return None

    raw_signal = pd.Series(
        compiled_fn(*gp.get_feature_arrays(df_slice, pair)),
        index=df_slice.index,
        dtype="float64",
    )
    overlay_inputs = build_overlay_inputs(df_slice, pairs, regime_pair=pair)
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
        initial_cooldown_bars=initial_cooldown_bars,
        return_trace=return_final_cooldown,
    )
    metrics = _extract_metrics(result)
    if return_final_cooldown:
        trace = result.get("trace") or {}
        cd_arr = trace.get("cooldown_bars_left")
        if cd_arr is not None and len(cd_arr) > 0:
            metrics["final_cooldown_bars"] = int(cd_arr[-1])
        else:
            metrics["final_cooldown_bars"] = 0
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
    carry_cooldown: bool = False,
) -> Generator[dict[str, Any], None, None]:
    """Yield one fold dict per OOS window.

    Each fold dict has keys:
        train_start, train_end, test_start, test_end,
        IS (per-pair metrics), OOS (per-pair metrics), skipped_pairs,
        carry_cooldown_in (per-pair cooldown bars seeded at window start),
        carry_cooldown_out (per-pair cooldown bars at window end).

    When carry_cooldown=True the OOS cooldown state at the end of window N is
    carried into window N+1 as the initial cooldown seed.  This prevents the
    unrealistic reset to zero that inflates trade counts in backtest.
    """
    start_ts = pd.Timestamp(start, tz=UTC)
    end_ts = pd.Timestamp(end, tz=UTC)
    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)

    # Per-pair cooldown carry state (only used when carry_cooldown=True).
    # Keyed by pair name, value is the cooldown_bars_left at end of last OOS window.
    cooldown_carry: dict[str, int] = {p: 0 for p in pairs}

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

        IS_metrics: dict[str, dict[str, float]] = {}
        OOS_metrics: dict[str, dict[str, float]] = {}
        skipped: list[str] = []
        carry_in: dict[str, int] = {}
        carry_out: dict[str, int] = {}

        for pair in pairs:
            cfg = pair_configs[pair]
            mapping = tuple(int(v) for v in cfg["mapping_indices"])
            route_breadth_threshold = float(cfg["route_breadth_threshold"])
            route_state_mode = str(cfg.get("route_state_mode", "base"))

            # Determine OOS initial cooldown seed for this window.
            oos_initial_cooldown = cooldown_carry[pair] if carry_cooldown else 0
            carry_in[pair] = oos_initial_cooldown

            # Slice price data
            df_is = df_all.loc[
                (df_all.index >= train_start) & (df_all.index < train_end)
            ].copy()
            df_oos = df_all.loc[
                (df_all.index >= test_end - test_delta) & (df_all.index < test_end)
            ].copy()

            if len(df_oos) < min_oos_bars:
                print(
                    f"  [{fold_label}] {pair}: OOS has only {len(df_oos)} bars "
                    f"(< {min_oos_bars}), skipping.",
                    flush=True,
                )
                skipped.append(pair)
                continue

            # Slice funding (already pre-loaded for full range)
            funding_pair = funding_cache[pair]
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
                )
                oos_result = _run_pair_replay(
                    df_oos, pair, pairs, compiled_fn,
                    funding_oos, library, mapping,
                    route_breadth_threshold, route_state_mode,
                    initial_cooldown_bars=oos_initial_cooldown,
                    return_final_cooldown=carry_cooldown,
                )
            except Exception as exc:
                print(
                    f"  [{fold_label}] {pair}: replay error: {exc}; skipping.",
                    flush=True,
                )
                skipped.append(pair)
                continue

            if is_result is None or oos_result is None:
                skipped.append(pair)
                continue

            # Extract and persist final cooldown for next window before stripping.
            if carry_cooldown:
                final_cd = int(oos_result.pop("final_cooldown_bars", 0))
                cooldown_carry[pair] = final_cd
                carry_out[pair] = final_cd
            else:
                carry_out[pair] = 0

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
            "carry_cooldown_in": carry_in,
            "carry_cooldown_out": carry_out,
        }

        test_start += step_delta


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def build_summary(folds: list[dict[str, Any]], pairs: tuple[str, ...]) -> dict[str, Any]:
    """Compute IS-vs-OOS gap statistics across all folds and pairs."""
    oos_returns: list[float] = []
    is_returns: list[float] = []
    oos_sharpes: list[float] = []
    is_sharpes: list[float] = []
    oos_win_rates: list[float] = []
    is_win_rates: list[float] = []
    negative_oos_folds: int = 0
    total_oos_obs: int = 0

    for fold in folds:
        fold_oos_returns: list[float] = []
        fold_is_returns: list[float] = []
        for pair in pairs:
            if pair not in fold["OOS"] or pair not in fold["IS"]:
                continue
            oos_ret = fold["OOS"][pair]["total_return"]
            is_ret = fold["IS"][pair]["total_return"]
            oos_returns.append(oos_ret)
            is_returns.append(is_ret)
            fold_oos_returns.append(oos_ret)
            fold_is_returns.append(is_ret)
            oos_sharpes.append(fold["OOS"][pair]["sharpe"])
            is_sharpes.append(fold["IS"][pair]["sharpe"])
            oos_win_rates.append(fold["OOS"][pair]["roundtrip_win_rate"])
            is_win_rates.append(fold["IS"][pair]["roundtrip_win_rate"])
            total_oos_obs += 1

        if fold_oos_returns:
            mean_fold_oos = sum(fold_oos_returns) / len(fold_oos_returns)
            if mean_fold_oos < 0:
                negative_oos_folds += 1

    def _safe_mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    def _safe_median(lst: list[float]) -> float:
        return float(np.median(lst)) if lst else 0.0

    def _safe_std(lst: list[float]) -> float:
        return float(np.std(lst)) if lst else 0.0

    oos_mean_return = _safe_mean(oos_returns)
    oos_median_return = _safe_median(oos_returns)
    oos_std_return = _safe_std(oos_returns)
    is_mean_return = _safe_mean(is_returns)

    is_mean_sharpe = _safe_mean(is_sharpes)
    oos_mean_sharpe = _safe_mean(oos_sharpes)
    sharpe_deflation_pct = (
        (is_mean_sharpe - oos_mean_sharpe) / abs(is_mean_sharpe) * 100.0
        if is_mean_sharpe != 0.0 else 0.0
    )

    is_mean_win_rate = _safe_mean(is_win_rates)
    oos_mean_win_rate = _safe_mean(oos_win_rates)
    win_rate_decay = is_mean_win_rate - oos_mean_win_rate

    fold_count = len(folds)
    negative_oos_pct = (
        negative_oos_folds / fold_count * 100.0 if fold_count > 0 else 0.0
    )

    return {
        "fold_count": fold_count,
        "total_oos_pair_obs": total_oos_obs,
        "OOS_mean_return": _clip(oos_mean_return),
        "OOS_median_return": _clip(oos_median_return),
        "OOS_std_return": _clip(oos_std_return),
        "IS_mean_return": _clip(is_mean_return),
        "IS_OOS_return_gap": _clip(is_mean_return - oos_mean_return),
        "IS_mean_sharpe": _clip(is_mean_sharpe),
        "OOS_mean_sharpe": _clip(oos_mean_sharpe),
        "IS_OOS_sharpe_decay_pct": _clip(sharpe_deflation_pct),
        "IS_mean_win_rate": _clip(is_mean_win_rate),
        "OOS_mean_win_rate": _clip(oos_mean_win_rate),
        "OOS_win_rate_decay": _clip(win_rate_decay),
        "negative_oos_fold_count": negative_oos_folds,
        "OOS_negative_fold_pct": _clip(negative_oos_pct),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling walk-forward OOS validation for pairwise regime-mixture strategy."
    )
    parser.add_argument("--start", default="2023-04-01", help="Walk-forward period start (YYYY-MM-DD).")
    parser.add_argument("--end", default="2026-04-06", help="Walk-forward period end (YYYY-MM-DD).")
    parser.add_argument("--train-days", type=int, default=90, help="IS training window in days.")
    parser.add_argument("--test-days", type=int, default=30, help="OOS test window in days.")
    parser.add_argument("--step-days", type=int, default=30, help="Step size in days between folds.")
    parser.add_argument("--pairs", default="BTCUSDT,BNBUSDT", help="Comma-separated pair list.")
    parser.add_argument(
        "--report-out",
        type=Path,
        default=gp.MODELS_DIR / "walkforward_report.json",
        help="Output path for JSON report.",
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
    parser.add_argument(
        "--carry-cooldown",
        action="store_true",
        default=False,
        help=(
            "Carry OOS cooldown state from window N to window N+1. "
            "Prevents the per-window cooldown reset that inflates trade counts. "
            "Default off to preserve backward compatibility with existing search results."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = tuple(p.strip() for p in args.pairs.split(",") if p.strip())

    print(f"Walk-forward OOS validation", flush=True)
    print(f"  Period : {args.start} → {args.end}", flush=True)
    print(f"  Train  : {args.train_days}d  Test: {args.test_days}d  Step: {args.step_days}d", flush=True)
    print(f"  Pairs  : {pairs}", flush=True)
    print(f"  carry-cooldown: {args.carry_cooldown}", flush=True)

    # Load summary and model once.
    summary = json.loads(args.summary_path.read_text())
    pair_configs = summary["selected_candidate"]["pair_configs"]
    library = list(iter_params())
    model_tree, _ = load_signal_model(args.model_path)
    compiled_fn = gp.toolbox.compile(expr=model_tree)

    # Load full price history once (refresh_cache=False — DB is current).
    data_start = "2022-04-06"
    data_end = args.end
    print(f"\nLoading price data {data_start} → {data_end} ...", flush=True)
    df_all = gp.load_all_pairs(pairs=list(pairs), start=data_start, end=data_end, refresh_cache=False)
    print(f"  Loaded {len(df_all)} bars.", flush=True)

    # Load funding once per pair (covers the full range needed).
    print("Loading funding caches ...", flush=True)
    funding_cache: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        try:
            funding_cache[pair] = load_funding_cache(pair)
            print(f"  {pair}: {len(funding_cache[pair])} funding rows", flush=True)
        except RuntimeError as exc:
            print(f"  WARNING: {pair} funding load failed: {exc}", flush=True)
            funding_cache[pair] = pd.DataFrame(columns=["fundingTime", "fundingRate"])

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
        carry_cooldown=args.carry_cooldown,
    ):
        elapsed = time.perf_counter() - t_start
        oos_rets = {p: fold["OOS"][p]["total_return"] for p in fold["OOS"]}
        oos_trades = {p: fold["OOS"][p]["n_trades"] for p in fold["OOS"]}
        carry_suffix = ""
        if args.carry_cooldown:
            carry_suffix = (
                f"  cd_in={json_safe(fold['carry_cooldown_in'])}"
                f"  cd_out={json_safe(fold['carry_cooldown_out'])}"
            )
        print(
            f"  {fold['fold']}  IS:{fold['train_start'][:10]}→{fold['train_end'][:10]}"
            f"  OOS:{fold['test_start'][:10]}→{fold['test_end'][:10]}"
            f"  OOS_ret={json_safe(oos_rets)}  n_trades={json_safe(oos_trades)}"
            f"  elapsed={elapsed:.1f}s{carry_suffix}",
            flush=True,
        )
        folds.append(fold)

    t_elapsed = time.perf_counter() - t_start
    print(f"\nCompleted {len(folds)} folds in {t_elapsed:.1f}s", flush=True)

    summary_stats = build_summary(folds, pairs)

    # Compute carry-cooldown statistics across folds.
    carry_stats: dict[str, Any] = {"enabled": args.carry_cooldown}
    if args.carry_cooldown:
        windows_with_nonzero_carry: dict[str, int] = {p: 0 for p in pairs}
        total_carry_bars: dict[str, int] = {p: 0 for p in pairs}
        for fold in folds:
            for p in pairs:
                cd_in = fold.get("carry_cooldown_in", {}).get(p, 0)
                if cd_in > 0:
                    windows_with_nonzero_carry[p] += 1
                    total_carry_bars[p] += cd_in
        carry_stats["windows_with_nonzero_carry_per_pair"] = windows_with_nonzero_carry
        carry_stats["total_carry_bars_per_pair"] = total_carry_bars

    report: dict[str, Any] = {
        "generated_at": iso_now(),
        "strategy_class": "pairwise_walkforward_oos",
        "params": {
            "start": args.start,
            "end": args.end,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "pairs": list(pairs),
            "summary_path": str(args.summary_path),
            "model_path": str(args.model_path),
            "carry_cooldown": args.carry_cooldown,
        },
        "folds": folds,
        "summary": summary_stats,
        "carry_cooldown_stats": carry_stats,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(json_safe(report), indent=2))
    print(f"\nReport written to: {args.report_out}", flush=True)

    # Pretty summary print.
    s = summary_stats
    print("\n" + "=" * 60)
    print("WALK-FORWARD OOS SUMMARY")
    print("=" * 60)
    print(f"  Folds executed          : {s['fold_count']}")
    print(f"  Total pair-fold obs     : {s['total_oos_pair_obs']}")
    print(f"  OOS mean return         : {s['OOS_mean_return']:.4f}  ({s['OOS_mean_return']*100:.2f}%)")
    print(f"  OOS median return       : {s['OOS_median_return']:.4f}  ({s['OOS_median_return']*100:.2f}%)")
    print(f"  OOS std return          : {s['OOS_std_return']:.4f}")
    print(f"  IS mean return          : {s['IS_mean_return']:.4f}  ({s['IS_mean_return']*100:.2f}%)")
    print(f"  IS→OOS return gap       : {s['IS_OOS_return_gap']:.4f}  ({s['IS_OOS_return_gap']*100:.2f}%)")
    print(f"  IS Sharpe (mean)        : {s['IS_mean_sharpe']:.3f}")
    print(f"  OOS Sharpe (mean)       : {s['OOS_mean_sharpe']:.3f}")
    print(f"  Sharpe deflation        : {s['IS_OOS_sharpe_decay_pct']:.1f}%")
    print(f"  IS win-rate (mean)      : {s['IS_mean_win_rate']:.3f}")
    print(f"  OOS win-rate (mean)     : {s['OOS_mean_win_rate']:.3f}")
    print(f"  Win-rate decay          : {s['OOS_win_rate_decay']:.3f}")
    print(f"  Negative OOS folds      : {s['negative_oos_fold_count']} / {s['fold_count']}"
          f"  ({s['OOS_negative_fold_pct']:.1f}%)")
    if args.carry_cooldown and carry_stats.get("windows_with_nonzero_carry_per_pair"):
        print()
        print("CARRY-COOLDOWN STATISTICS")
        print("-" * 60)
        for p in pairs:
            nz = carry_stats["windows_with_nonzero_carry_per_pair"].get(p, 0)
            tb = carry_stats["total_carry_bars_per_pair"].get(p, 0)
            print(
                f"  {p}: windows with nonzero carry-in = {nz}/{s['fold_count']}"
                f"  total_carry_bars = {tb}"
            )
    print("=" * 60)


if __name__ == "__main__":
    main()
