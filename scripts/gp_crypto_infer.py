"""
GP Crypto Strategy - Inference & Analysis
==========================================
Load trained GP models, generate signals, backtest on arbitrary periods,
and perform walk-forward analysis.

Usage:
    python scripts/gp_crypto_infer.py backtest --start 2024-07-01 --end 2025-03-01
    python scripts/gp_crypto_infer.py walkforward --window 3 --step 1
    python scripts/gp_crypto_infer.py signal
"""

import argparse
import json
from pathlib import Path
from typing import Dict
from datetime import datetime, timedelta, timezone

import dill
import numpy as np
import pandas as pd

from gp_crypto_evolution import (
    PAIRS, PRIMARY_PAIR, ARG_NAMES,
    INITIAL_CASH, COMMISSION_PCT, NO_TRADE_BAND, TIMEFRAME,
    DAILY_TARGET_PCT,
    pset, toolbox,
    load_all_pairs, fetch_klines, vectorized_backtest, backtest_on_slice,
    daily_session_backtest,
    get_feature_arrays, get_feature_values,
    MODELS_DIR,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Model Manager
# ─────────────────────────────────────────────────────────────────────────────
class GPModelManager:
    """Loads and wraps a trained GP model for inference."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = str(MODELS_DIR / "best_crypto_gp.dill")
        self.model_path = model_path
        self.model = None
        self.compiled_func = None

    def load(self) -> bool:
        try:
            with open(self.model_path, "rb") as f:
                self.model = dill.load(f)
            self.compiled_func = toolbox.compile(expr=self.model)
            fitness_values = tuple(getattr(self.model.fitness, "values", ()))
            if len(fitness_values) == 1:
                fitness_text = f"{fitness_values[0]:.6f}"
            else:
                fitness_text = ", ".join(f"{v:.6f}" for v in fitness_values)
            print(f"Model loaded: {self.model_path}")
            print(f"  Tree size : {len(self.model)} nodes")
            print(f"  Fitness   : {fitness_text}")
            return True
        except FileNotFoundError:
            print(f"Model not found: {self.model_path}")
            print("Run gp_crypto_evolution.py first.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_signal(self, market_data: Dict[str, float]) -> float:
        """Scalar signal from a single row of market data."""
        if self.compiled_func is None:
            raise ValueError("Model not loaded")
        inputs = get_feature_values(market_data, PRIMARY_PAIR)
        return float(self.compiled_func(*inputs))

    def get_signals_vectorized(
        self,
        df: pd.DataFrame,
        pair: str = PRIMARY_PAIR,
    ) -> np.ndarray:
        """Vectorized signals for an entire DataFrame."""
        if self.compiled_func is None:
            raise ValueError("Model not loaded")
        cols = get_feature_arrays(df, pair)
        signals = self.compiled_func(*cols)
        signals = np.where(np.isfinite(signals), signals, 0.0)
        return np.clip(signals, -100.0, 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Backtest a saved model
# ─────────────────────────────────────────────────────────────────────────────
def cmd_backtest(model_path: str = None,
                 start_date: str = None,
                 end_date: str = None) -> Dict:
    mgr = GPModelManager(model_path)
    if not mgr.load():
        return {}

    print("\nLoading market data...")
    df = load_all_pairs()

    if start_date and end_date:
        df = df.loc[start_date:end_date]
        print(f"Period: {start_date} ~ {end_date} ({len(df)} bars)")

    return backtest_on_slice(mgr.model, df, "BACKTEST")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Walk-forward analysis
# ─────────────────────────────────────────────────────────────────────────────
def cmd_walkforward(model_path: str = None,
                    window_months: int = 3,
                    step_months: int = 1) -> pd.DataFrame:
    mgr = GPModelManager(model_path)
    if not mgr.load():
        return pd.DataFrame()

    df = load_all_pairs()
    results = []
    end = df.index[-1]
    current = df.index[0]

    print(f"\nWalk-forward: window={window_months}mo, step={step_months}mo")
    print(f"{'Window':^25} | {'Return':>8} | {'Sharpe':>7} | "
          f"{'MaxDD':>8} | {'Hit':>6} | {'Trades':>6}")
    print("-" * 77)

    while current + timedelta(days=window_months * 30) <= end:
        w_end = current + timedelta(days=window_months * 30)
        window_df = df.loc[current:w_end]

        if len(window_df) >= 100:
            signals = mgr.get_signals_vectorized(window_df)
            r = daily_session_backtest(window_df, signals, PRIMARY_PAIR)
            daily = r["daily_metrics"]

            label = (f"{current.strftime('%Y-%m-%d')} ~ "
                     f"{w_end.strftime('%Y-%m-%d')}")
            print(f"{label} | {r['total_return']*100:+7.2f}% | "
                  f"{r['sharpe']:6.2f} | {r['max_drawdown']*100:7.2f}% | "
                  f"{daily['daily_target_hit_rate']*100:5.1f}% | "
                  f"{r['n_trades']:5d}")

            results.append({"start": current, "end": w_end, **r})

        current += timedelta(days=step_months * 30)

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        print(f"\nSummary:")
        print(f"  Avg Return : {results_df['total_return'].mean()*100:+.2f}%")
        print(f"  Avg Sharpe : {results_df['sharpe'].mean():.2f}")
        print(f"  Win Rate   : "
              f"{(results_df['total_return'] > 0).mean()*100:.1f}%")
        if "daily_metrics" in results_df:
            hit_rates = results_df["daily_metrics"].map(
                lambda m: m.get("daily_target_hit_rate", 0.0)
            )
            print(f"  Target Hit : {hit_rates.mean()*100:.1f}% "
                  f"(>= {DAILY_TARGET_PCT*100:.2f}%/day)")
        print(f"  Worst DD   : {results_df['max_drawdown'].min()*100:.2f}%")

        out_path = MODELS_DIR / "walkforward_results.csv"
        results_df.to_csv(out_path, index=False)
        print(f"  Saved      : {out_path}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Real-time signal generation
# ─────────────────────────────────────────────────────────────────────────────
def cmd_signal(model_path: str = None) -> Dict:
    mgr = GPModelManager(model_path)
    if not mgr.load():
        return {}

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=48)

    print("\nFetching latest market data...")
    dfs = []
    for pair in PAIRS:
        pair_df = fetch_klines(pair, TIMEFRAME, start_dt, end_dt)
        pair_df.columns = [f"{pair}_{c}" for c in pair_df.columns]
        dfs.append(pair_df)

    combined = pd.concat(dfs, axis=1).dropna()
    if combined.empty:
        print("No data available")
        return {}

    latest = combined.iloc[-1]
    signal = mgr.get_signal(latest)
    btc_price = float(latest[f"{PRIMARY_PAIR}_close"])

    if signal > 0:
        direction = "LONG"
    elif signal < 0:
        direction = "SHORT"
    else:
        direction = "FLAT"

    result = {
        "timestamp": combined.index[-1].isoformat(),
        "signal_pct": round(signal, 2),
        "direction": direction,
        "strength": round(abs(signal), 1),
        "btc_price": btc_price,
    }

    print(f"\nLatest Signal ({result['timestamp']})")
    print(f"  BTC Price : ${btc_price:,.2f}")
    print(f"  Signal    : {signal:+.2f}% ({direction})")
    print(f"  Strength  : {abs(signal):.1f} / 100")

    # Recent signal history
    signals = mgr.get_signals_vectorized(combined.tail(24))
    close_24h = combined[f"{PRIMARY_PAIR}_close"].tail(24)
    print(f"\n  Last 24h signal range: "
          f"[{signals.min():+.1f}% ~ {signals.max():+.1f}%]")
    print(f"  Last 24h price range: "
          f"[${close_24h.min():,.2f} ~ ${close_24h.max():,.2f}]")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="GP Crypto Strategy - Inference & Analysis",
    )
    parser.add_argument(
        "--model", default=None, help="Path to .dill model file",
    )

    sub = parser.add_subparsers(dest="command")

    bt = sub.add_parser("backtest", help="Backtest saved model")
    bt.add_argument("--start", default="2024-07-01")
    bt.add_argument("--end", default="2025-03-01")

    wf = sub.add_parser("walkforward", help="Walk-forward analysis")
    wf.add_argument("--window", type=int, default=3)
    wf.add_argument("--step", type=int, default=1)

    sub.add_parser("signal", help="Generate latest trading signal")

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args.model, args.start, args.end)
    elif args.command == "walkforward":
        cmd_walkforward(args.model, args.window, args.step)
    elif args.command == "signal":
        cmd_signal(args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
