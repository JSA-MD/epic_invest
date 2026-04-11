"""
Train a day-level participation filter on top of a base GP RR model.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import pandas as pd

from gp_crypto_evolution import (
    DEFAULT_REWARD_MULTIPLE,
    MODELS_DIR,
    PAIRS,
    PRIMARY_PAIR,
    ROBUST_REWARD_MULTIPLES,
    daily_session_backtest,
    get_feature_arrays,
    load_all_pairs,
    split_dataset,
    toolbox,
)


@dataclass
class DailyParticipationFilter:
    feature_names: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    reward_multiple: float
    base_model_path: str

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_norm = np.nan_to_num((x - self.mean) / self.scale, nan=0.0, posinf=0.0, neginf=0.0)
        x_norm = np.clip(x_norm, -12.0, 12.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = x_norm @ self.weights + self.bias
        z = np.clip(z, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_keep(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x) >= self.threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a daily participation filter for a GP RR model.",
    )
    parser.add_argument(
        "--base-model",
        default=str(MODELS_DIR / "best_crypto_gp_rr_daily.dill"),
    )
    parser.add_argument(
        "--filter-out",
        default=str(MODELS_DIR / "best_crypto_gp_rr_daily_gate.dill"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(MODELS_DIR / "best_crypto_gp_rr_daily_gate_summary.json"),
    )
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


def build_feature_vector(row: pd.Series, signal: float) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []

    for pair in PAIRS:
        close = float(row[f"{pair}_close"])
        close_scale = max(abs(close), 1e-8)
        raw_volume_scale = float(row.get(f"{pair}_vol_sma", 0.0))
        volume_rel = 1.0
        if np.isfinite(raw_volume_scale) and abs(raw_volume_scale) >= 1e-8:
            volume_rel = float(row.get(f"{pair}_volume", 0.0)) / abs(raw_volume_scale)
        pair_feats = {
            f"{pair}_rsi_14": float(row[f"{pair}_rsi_14"]) / 100.0,
            f"{pair}_atr_rel": float(row[f"{pair}_atr_14"]) / close_scale,
            f"{pair}_macd_rel": float(row[f"{pair}_macd_h"]) / close_scale,
            f"{pair}_bb_p": float(row[f"{pair}_bb_p"]),
            f"{pair}_cci_14": float(np.tanh(float(row[f"{pair}_cci_14"]) / 100.0)),
            f"{pair}_mfi_14": float(row[f"{pair}_mfi_14"]) / 100.0,
            f"{pair}_volume_rel": float(volume_rel),
            f"{pair}_dc_trend_05": float(row[f"{pair}_dc_trend_05"]),
            f"{pair}_dc_event_05": float(row[f"{pair}_dc_event_05"]),
            f"{pair}_dc_overshoot_05": float(row[f"{pair}_dc_overshoot_05"]) / 0.01,
            f"{pair}_dc_run_05": float(row[f"{pair}_dc_run_05"]) / 0.01,
        }
        for name, value in pair_feats.items():
            names.append(name)
            values.append(value)

    primary_open = max(abs(float(row[f"{PRIMARY_PAIR}_open"])), 1e-8)
    names.extend(
        [
            "signal_signed",
            "signal_abs",
            "primary_range_rel",
            "primary_open_close_rel",
        ]
    )
    values.extend(
        [
            float(signal) / 100.0,
            abs(float(signal)) / 100.0,
            float(row[f"{PRIMARY_PAIR}_high"] - row[f"{PRIMARY_PAIR}_low"]) / primary_open,
            float(row[f"{PRIMARY_PAIR}_close"] - row[f"{PRIMARY_PAIR}_open"]) / primary_open,
        ]
    )
    values = np.nan_to_num(np.asarray(values, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, -50.0, 50.0)
    return names, values.tolist()


def simulate_day_labels(day_df: pd.DataFrame, day_signals: np.ndarray) -> tuple[int, dict]:
    rr_returns = {}
    for rr in ROBUST_REWARD_MULTIPLES:
        result = daily_session_backtest(
            day_df,
            day_signals,
            PRIMARY_PAIR,
            reward_multiple=float(rr),
        )
        rr_returns[f"rr_{int(rr)}"] = float(result["net_ret"][0]) if len(result["net_ret"]) else 0.0

    avg_rr_return = float(np.mean(list(rr_returns.values())))
    label = int(avg_rr_return > 0.0)
    rr_returns["avg_rr_return"] = avg_rr_return
    return label, rr_returns


def build_day_dataset(df_slice: pd.DataFrame, desired_pcts: np.ndarray) -> dict:
    idx = pd.DatetimeIndex(df_slice.index)
    feature_names = None
    x_rows = []
    y_rows = []
    day_dates = []
    day_stats = []

    for day in pd.Index(idx.normalize().unique()):
        pos = np.where(idx.normalize() == day)[0]
        if len(pos) < 2:
            continue
        start = pos[0]
        row = df_slice.iloc[start]
        names, values = build_feature_vector(row, float(desired_pcts[start]))
        if feature_names is None:
            feature_names = names
        label, stats = simulate_day_labels(df_slice.iloc[pos], desired_pcts[pos])
        x_rows.append(values)
        y_rows.append(label)
        day_dates.append(day)
        day_stats.append(stats)

    x = np.asarray(x_rows, dtype="float64")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -50.0, 50.0)

    return {
        "x": x,
        "y": np.asarray(y_rows, dtype="float64"),
        "day_dates": pd.DatetimeIndex(day_dates),
        "feature_names": feature_names or [],
        "day_stats": day_stats,
    }


def fit_balanced_logistic(
    x: np.ndarray,
    y: np.ndarray,
    reg: float,
    epochs: int = 2000,
    lr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    x_norm = np.clip(x_norm, -12.0, 12.0)

    weights = np.zeros(x.shape[1], dtype="float64")
    bias = float(np.log((y.mean() + 1e-6) / (1.0 - y.mean() + 1e-6)))

    pos_weight = len(y) / max(1.0, 2.0 * y.sum())
    neg_weight = len(y) / max(1.0, 2.0 * (len(y) - y.sum()))
    sample_weight = np.where(y > 0.5, pos_weight, neg_weight)

    for _ in range(epochs):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            z = np.clip(x_norm @ weights + bias, -40.0, 40.0)
        prob = 1.0 / (1.0 + np.exp(-z))
        error = (prob - y) * sample_weight
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            grad_w = (x_norm.T @ error) / len(y) + reg * weights
        grad_b = float(np.mean(error))
        weights -= lr * grad_w
        bias -= lr * grad_b
        weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        bias = float(np.clip(np.nan_to_num(bias, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0))

    return mean, scale, weights, bias


def gate_signals(desired_pcts: np.ndarray, day_dates, keep_mask, index) -> np.ndarray:
    gated = np.array(desired_pcts, copy=True)
    idx = pd.DatetimeIndex(index)
    normalized = idx.normalize()
    for day, keep in zip(day_dates, keep_mask):
        if not keep:
            gated[normalized == day] = 0.0
    return gated


def evaluate_gate(
    df_slice: pd.DataFrame,
    desired_pcts: np.ndarray,
    day_dates,
    probs: np.ndarray,
    threshold: float,
    reward_multiple: float,
) -> dict:
    keep_mask = probs >= threshold
    gated = gate_signals(desired_pcts, day_dates, keep_mask, df_slice.index)
    result = daily_session_backtest(
        df_slice,
        gated,
        PRIMARY_PAIR,
        reward_multiple=reward_multiple,
    )
    result["filter_stats"] = {
        "threshold": float(threshold),
        "trade_days": int(np.sum(keep_mask)),
        "day_count": int(len(keep_mask)),
        "coverage": float(np.mean(keep_mask)) if len(keep_mask) else 0.0,
        "avg_probability": float(np.mean(probs)) if len(probs) else 0.0,
    }
    return result


def validation_score(result: dict) -> float:
    coverage = result["filter_stats"]["coverage"]
    if result["n_trades"] < 5 or coverage < 0.08:
        return -1e9
    return (
        result["total_return"] * 100.0
        - abs(result["max_drawdown"]) * 20.0
        - result["monthly_metrics"]["monthly_shortfall_sum"] * 5.0
    )


def summarize_result(result: dict) -> dict:
    return {
        "total_return": result["total_return"],
        "sharpe": result["sharpe"],
        "max_drawdown": result["max_drawdown"],
        "n_trades": result["n_trades"],
        "win_rate": result["win_rate"],
        "reward_multiple": result["reward_multiple"],
        "daily_metrics": result["daily_metrics"],
        "monthly_metrics": result["monthly_metrics"],
        "filter_stats": result.get("filter_stats", {}),
    }


def main() -> None:
    args = parse_args()

    print("=" * 72)
    print("  Daily Participation Filter Training")
    print("=" * 72)

    with open(args.base_model, "rb") as f:
        base_model = dill.load(f)
    compiled = toolbox.compile(expr=base_model)
    print(f"Base model: {args.base_model}")
    print(f"  Tree size : {len(base_model)} nodes")

    print("\n[Phase 1] Data Loading")
    df_all = load_all_pairs()
    train_df, val_df, test_df = split_dataset(df_all)

    train_signal = compiled(*get_feature_arrays(train_df, PRIMARY_PAIR))
    val_signal = compiled(*get_feature_arrays(val_df, PRIMARY_PAIR))
    test_signal = compiled(*get_feature_arrays(test_df, PRIMARY_PAIR))
    full_signal = compiled(*get_feature_arrays(df_all, PRIMARY_PAIR))

    train_ds = build_day_dataset(train_df, train_signal)
    val_ds = build_day_dataset(val_df, val_signal)
    test_ds = build_day_dataset(test_df, test_signal)

    print("\n[Phase 2] Filter Fit")
    print(f"  Train days : {len(train_ds['y'])}")
    print(f"  Val days   : {len(val_ds['y'])}")
    print(f"  Test days  : {len(test_ds['y'])}")
    print(f"  Train pos  : {train_ds['y'].mean()*100:.1f}%")
    print(f"  Val pos    : {val_ds['y'].mean()*100:.1f}%")
    print(f"  Test pos   : {test_ds['y'].mean()*100:.1f}%")

    best = None
    best_score = -1e18
    for reg in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 5.0):
        mean, scale, weights, bias = fit_balanced_logistic(train_ds["x"], train_ds["y"], reg=reg)
        tmp_model = DailyParticipationFilter(
            feature_names=train_ds["feature_names"],
            mean=mean,
            scale=scale,
            weights=weights,
            bias=bias,
            threshold=0.5,
            reward_multiple=DEFAULT_REWARD_MULTIPLE,
            base_model_path=args.base_model,
        )
        val_probs = tmp_model.predict_proba(val_ds["x"])

        for threshold in np.linspace(0.40, 0.85, 10):
            for rr in ROBUST_REWARD_MULTIPLES:
                val_result = evaluate_gate(
                    val_df,
                    val_signal,
                    val_ds["day_dates"],
                    val_probs,
                    float(threshold),
                    float(rr),
                )
                score = validation_score(val_result)
                if score > best_score:
                    best_score = score
                    best = {
                        "reg": reg,
                        "threshold": float(threshold),
                        "reward_multiple": float(rr),
                        "model": tmp_model,
                        "val_result": val_result,
                    }

    gate_model: DailyParticipationFilter = best["model"]
    gate_model.threshold = best["threshold"]
    gate_model.reward_multiple = best["reward_multiple"]

    print("\n[Phase 3] Validation")
    print(
        f"  reg={best['reg']:.4g}, threshold={best['threshold']:.2f}, "
        f"RR=1:{best['reward_multiple']:.0f}, score={best_score:.4f}"
    )

    val_probs = gate_model.predict_proba(val_ds["x"])
    test_probs = gate_model.predict_proba(test_ds["x"])
    full_ds = build_day_dataset(df_all, full_signal)
    full_probs = gate_model.predict_proba(full_ds["x"])

    baseline_val = daily_session_backtest(val_df, val_signal, PRIMARY_PAIR, reward_multiple=gate_model.reward_multiple)
    baseline_test = daily_session_backtest(test_df, test_signal, PRIMARY_PAIR, reward_multiple=gate_model.reward_multiple)
    baseline_full = daily_session_backtest(df_all, full_signal, PRIMARY_PAIR, reward_multiple=gate_model.reward_multiple)

    filtered_val = evaluate_gate(
        val_df, val_signal, val_ds["day_dates"], val_probs,
        gate_model.threshold, gate_model.reward_multiple,
    )
    filtered_test = evaluate_gate(
        test_df, test_signal, test_ds["day_dates"], test_probs,
        gate_model.threshold, gate_model.reward_multiple,
    )
    filtered_full = evaluate_gate(
        df_all, full_signal, full_ds["day_dates"], full_probs,
        gate_model.threshold, gate_model.reward_multiple,
    )

    print("\n[Phase 4] Test")
    print(
        f"  Baseline OOS : {baseline_test['total_return']*100:+.2f}% "
        f"(DD {baseline_test['max_drawdown']*100:.2f}%, trades {baseline_test['n_trades']})"
    )
    print(
        f"  Filtered OOS : {filtered_test['total_return']*100:+.2f}% "
        f"(DD {filtered_test['max_drawdown']*100:.2f}%, trades {filtered_test['n_trades']}, "
        f"coverage {filtered_test['filter_stats']['coverage']*100:.1f}%)"
    )

    print("\n[Phase 5] Save")
    filter_path = Path(args.filter_out)
    summary_path = Path(args.summary_out)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(filter_path, "wb") as f:
        dill.dump(gate_model, f)

    summary = {
        "base_model_path": args.base_model,
        "filter_path": str(filter_path),
        "feature_count": len(gate_model.feature_names),
        "label_definition": "avg daily return across RR 1:2/1:3/1:4 > 0",
        "train_days": int(len(train_ds["y"])),
        "val_days": int(len(val_ds["y"])),
        "test_days": int(len(test_ds["y"])),
        "train_positive_rate": float(train_ds["y"].mean()),
        "val_positive_rate": float(val_ds["y"].mean()),
        "test_positive_rate": float(test_ds["y"].mean()),
        "best_reg": best["reg"],
        "threshold": gate_model.threshold,
        "reward_multiple": gate_model.reward_multiple,
        "validation_score": best_score,
        "baseline_validation": summarize_result(baseline_val),
        "baseline_test": summarize_result(baseline_test),
        "baseline_full": summarize_result(baseline_full),
        "filtered_validation": summarize_result(filtered_val),
        "filtered_test": summarize_result(filtered_test),
        "filtered_full": summarize_result(filtered_full),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path, "w") as f:
        json.dump(json_safe(summary), f, indent=2)

    print(f"  Filter saved : {filter_path}")
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
