"""
Train and evaluate a daily-session GP model without overwriting the default model.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np

from gp_crypto_evolution import (
    DAILY_MAX_LOSS_PCT,
    DAILY_TARGET_PCT,
    DEFAULT_REWARD_MULTIPLE,
    MODELS_DIR,
    N_GEN,
    POP_SIZE,
    TEST_END,
    TEST_START,
    TIMEFRAME,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    PAIRS,
    PRIMARY_PAIR,
    backtest_on_slice,
    load_all_pairs,
    run_evolution,
    select_best_on_validation,
    split_dataset,
)
from gp_crypto_infer import cmd_walkforward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a daily-session GP model and save dedicated outputs.",
    )
    parser.add_argument("--pop-size", type=int, default=POP_SIZE)
    parser.add_argument("--n-gen", type=int, default=N_GEN)
    parser.add_argument(
        "--model-out",
        default=str(MODELS_DIR / "best_crypto_gp_daily_session.dill"),
    )
    parser.add_argument(
        "--reuse-model",
        action="store_true",
        help="Skip training and reuse --model-out for evaluation/summary.",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "best_crypto_gp_daily_session_summary.json"),
    )
    parser.add_argument("--walkforward-window", type=int, default=3)
    parser.add_argument("--walkforward-step", type=int, default=1)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()

    print("=" * 68)
    print("  Daily-Session GP Training")
    print("=" * 68)

    print("\n[Phase 1] Data Loading")
    df_all = load_all_pairs()
    train_df, val_df, test_df = split_dataset(df_all)

    model_path = Path(args.model_out)
    if args.reuse_model:
        print("\n[Phase 2] Reuse Existing Model")
        with open(model_path, "rb") as f:
            best = dill.load(f)
        print(f"Model loaded: {model_path}")
    else:
        print("\n[Phase 2] GP Evolution")
        hof = run_evolution(train_df, pop_size=args.pop_size, n_gen=args.n_gen)

        print("\n[Phase 3] Validation")
        best = select_best_on_validation(hof, val_df)

    print("\n[Phase 4] Out-of-Sample Test")
    test_result = backtest_on_slice(best, test_df, "TEST (Daily Session)")

    print("\n[Phase 5] Full-Period Backtest")
    full_result = backtest_on_slice(best, df_all, "FULL PERIOD (Daily Session)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.reuse_model:
        with open(model_path, "wb") as f:
            dill.dump(best, f)
        print(f"\nModel saved: {model_path}")

    print("\n[Phase 6] Walk-Forward")
    walkforward_df = cmd_walkforward(
        model_path=str(model_path),
        window_months=args.walkforward_window,
        step_months=args.walkforward_step,
    )

    summary = {
        "pairs": PAIRS,
        "primary_pair": PRIMARY_PAIR,
        "timeframe": TIMEFRAME,
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "val_period": f"{VAL_START} ~ {VAL_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "pop_size": args.pop_size,
        "n_gen": args.n_gen,
        "tree_size": len(best),
        "fitness": float(best.fitness.values[0]),
        "daily_target_pct": DAILY_TARGET_PCT,
        "daily_stop_pct": DAILY_MAX_LOSS_PCT,
        "reward_multiple": DEFAULT_REWARD_MULTIPLE,
        "test": {
            "total_return": test_result["total_return"],
            "sharpe": test_result["sharpe"],
            "max_drawdown": test_result["max_drawdown"],
            "n_trades": test_result["n_trades"],
            "win_rate": test_result["win_rate"],
            "target_hit_rate": test_result["target_hit_rate"],
            "daily_metrics": test_result["daily_metrics"],
            "monthly_metrics": test_result["monthly_metrics"],
        },
        "full_period": {
            "total_return": full_result["total_return"],
            "sharpe": full_result["sharpe"],
            "max_drawdown": full_result["max_drawdown"],
            "n_trades": full_result["n_trades"],
            "win_rate": full_result["win_rate"],
            "target_hit_rate": full_result["target_hit_rate"],
            "daily_metrics": full_result["daily_metrics"],
            "monthly_metrics": full_result["monthly_metrics"],
        },
        "walkforward": {
            "n_windows": int(len(walkforward_df)),
            "avg_return": (
                float(walkforward_df["total_return"].mean())
                if not walkforward_df.empty
                else 0.0
            ),
            "avg_sharpe": (
                float(walkforward_df["sharpe"].mean())
                if not walkforward_df.empty
                else 0.0
            ),
            "win_rate": (
                float((walkforward_df["total_return"] > 0).mean())
                if not walkforward_df.empty
                else 0.0
            ),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = Path(args.summary_out)
    with open(summary_path, "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    print(f"Summary saved: {summary_path}")

    print(f"\nBest tree:\n  {best}")


if __name__ == "__main__":
    main()
