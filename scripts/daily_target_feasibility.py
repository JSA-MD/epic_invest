"""
daily_target_feasibility.py

Estimates P(daily_return >= 1%) and feasibility of a 1%/day target via
Monte Carlo bootstrap from OOS walkforward fold returns.

Inputs:
  models/walkforward_report.json       - 36 OOS folds, BTC + BNB
  models/pairwise_equity_corr_risk_compare.json - full_4y IS baseline
  models/tail_risk_report.json         - distributional params (mean, std, skew)

Output:
  models/daily_target_feasibility_report.json
"""

import json
import math
import random
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WF_PATH = ROOT / "models" / "walkforward_report.json"
PAIRWISE_PATH = ROOT / "models" / "pairwise_equity_corr_risk_compare.json"
TAIL_PATH = ROOT / "models" / "tail_risk_report.json"
OUT_PATH = ROOT / "models" / "daily_target_feasibility_report.json"

# ---------------------------------------------------------------------------
# Monte Carlo parameters
# ---------------------------------------------------------------------------
N_TRIALS = 1000
DAYS_PER_TRIAL = 252
RANDOM_SEED = 42

SIZING_SCENARIOS = {
    "0.25x": 0.25,   # Quarter Kelly
    "0.5x": 0.50,    # Half Kelly
    "1.0x": 1.00,    # Current
    "2.0x": 2.00,    # Leveraged
}

PAIR_KEYS = ["BTCUSDT", "BNBUSDT"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(v, fallback=0.0):
    """Return float, replacing NaN/Inf with fallback."""
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return fallback
        return f
    except (TypeError, ValueError):
        return fallback


def implied_daily_return(fold_total_return: float, n_days: int = 30) -> float:
    """Back-calculate geometric daily return from a fold total return."""
    # total_return here is the fractional return (e.g. 0.134 = 13.4%)
    # Compound: (1 + fold_total)^(1/n_days) - 1
    compound = 1.0 + fold_total_return
    if compound <= 0.0:
        compound = 1e-6
    return compound ** (1.0 / n_days) - 1.0


def compute_max_drawdown(equity_curve: list) -> float:
    """Compute max drawdown from an equity curve (list of cumulative values)."""
    peak = equity_curve[0]
    mdd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < mdd:
            mdd = dd
    return mdd


def run_bootstrap(daily_returns_pool: list, sizing: float, rng: random.Random) -> dict:
    """
    Run N_TRIALS bootstrap trials of DAYS_PER_TRIAL days.
    Returns aggregated statistics.
    """
    p_daily_1pct_counts = 0
    annual_returns = []
    annual_dd_list = []
    annual_sharpes = []
    all_daily_returns = []

    for _ in range(N_TRIALS):
        # Sample with replacement
        sampled = rng.choices(daily_returns_pool, k=DAYS_PER_TRIAL)
        # Apply sizing scalar (linear scale of daily returns)
        scaled = [r * sizing for r in sampled]

        # Fraction of days >= 1%
        days_above_1pct = sum(1 for r in scaled if r >= 0.01)
        p_daily_1pct_counts += days_above_1pct / DAYS_PER_TRIAL

        # Annual cumulative return via compounding
        equity = 1.0
        equity_curve = [equity]
        for r in scaled:
            equity *= (1.0 + r)
            equity_curve.append(equity)
        annual_return = equity - 1.0
        annual_returns.append(annual_return)
        all_daily_returns.extend(scaled)

        # Max drawdown this trial
        mdd = compute_max_drawdown(equity_curve)
        annual_dd_list.append(mdd)

        # Sharpe (annualised, 0% risk-free)
        n = len(scaled)
        mean_r = sum(scaled) / n
        var_r = sum((r - mean_r) ** 2 for r in scaled) / n
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
        sharpe = (mean_r / std_r) * math.sqrt(252)
        annual_sharpes.append(sharpe)

    # Aggregate
    p_daily_1pct = p_daily_1pct_counts / N_TRIALS

    annual_returns_sorted = sorted(annual_returns)
    n = len(annual_returns_sorted)

    def percentile(sorted_lst, p):
        idx = max(0, min(int(p / 100.0 * len(sorted_lst)), len(sorted_lst) - 1))
        return sorted_lst[idx]

    median_annual = percentile(annual_returns_sorted, 50)
    p5_annual = percentile(annual_returns_sorted, 5)
    p95_annual = percentile(annual_returns_sorted, 95)
    expected_annual = sum(annual_returns) / len(annual_returns)
    p_annual_100 = sum(1 for r in annual_returns if r >= 1.0) / len(annual_returns)
    p_annual_dd_25 = sum(1 for d in annual_dd_list if d < -0.25) / len(annual_dd_list)
    median_sharpe = percentile(sorted(annual_sharpes), 50)

    # Daily distribution
    all_sorted = sorted(all_daily_returns)
    median_daily = percentile(all_sorted, 50)
    p5_daily = percentile(all_sorted, 5)
    p95_daily = percentile(all_sorted, 95)

    return {
        "p_daily_1pct": safe_float(p_daily_1pct),
        "median_daily_return": safe_float(median_daily),
        "p5_daily": safe_float(p5_daily),
        "p95_daily": safe_float(p95_daily),
        "median_annual_return": safe_float(median_annual),
        "p5_annual_return": safe_float(p5_annual),
        "p95_annual_return": safe_float(p95_annual),
        "expected_annual": safe_float(expected_annual),
        "p_annual_100": safe_float(p_annual_100),
        "p_annual_dd_25": safe_float(p_annual_dd_25),
        "median_sharpe": safe_float(median_sharpe),
    }


def determine_verdict(per_pair: dict) -> str:
    """
    Determine verdict from combined_50_50 scenario 1.0x stats.
    REALISTIC if P(daily>=1%) >= 30% AND P(annual>=100%) >= 80%
    STRETCH   if P(daily>=1%) in [15%, 30%)
    FANTASY   if P(daily>=1%) < 15% OR P(annual_dd>25%) > 50%
    """
    combined = per_pair.get("combined_50_50", {})
    scenarios = combined.get("scenarios", {})
    base = scenarios.get("1.0x", {})

    p_daily = base.get("p_daily_1pct", 0.0)
    p_annual_100 = base.get("p_annual_100", 0.0)
    p_dd_25 = base.get("p_annual_dd_25", 0.0)

    if p_daily < 0.15 or p_dd_25 > 0.50:
        return "FANTASY"
    if p_daily >= 0.30 and p_annual_100 >= 0.80:
        return "REALISTIC"
    if 0.15 <= p_daily < 0.30:
        return "STRETCH"
    return "FANTASY"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(RANDOM_SEED)

    # Load inputs
    with open(WF_PATH) as f:
        wf = json.load(f)
    with open(PAIRWISE_PATH) as f:
        pairwise = json.load(f)
    with open(TAIL_PATH) as f:
        tail = json.load(f)

    folds = wf["folds"]

    # -----------------------------------------------------------------------
    # 1. Build OOS empirical daily-return pool per pair
    #    Each fold covers ~30 days. Back-calculate implied daily return
    #    for each fold, then replicate 30 copies to form the pool.
    # -----------------------------------------------------------------------
    oos_daily: dict[str, list[float]] = {p: [] for p in PAIR_KEYS}
    fold_days = 30  # each test window is 30 days

    for fold in folds:
        oos = fold.get("OOS", {})
        for pair in PAIR_KEYS:
            pdata = oos.get(pair, {})
            total_ret = safe_float(pdata.get("total_return", 0.0))
            daily_r = implied_daily_return(total_ret, fold_days)
            # Replicate daily_r for each day in the fold to form empirical pool
            oos_daily[pair].extend([daily_r] * fold_days)

    # -----------------------------------------------------------------------
    # 2. Build IS daily-return pool per pair from tail_risk distributional params
    #    Use mean + std from full_4y; generate N representative daily returns
    #    by replicating the mean for n_days (conservative: mean only, since
    #    we only have summary stats, not per-bar series).
    #    We approximate IS distribution using mean_daily replicated n_days times.
    # -----------------------------------------------------------------------
    is_daily: dict[str, list[float]] = {}
    tail_pp = tail.get("per_pair", {})

    for pair in PAIR_KEYS:
        tp = tail_pp.get(pair, {})
        mean_d = safe_float(tp.get("mean_daily", 0.0))
        std_d = safe_float(tp.get("std_daily", 0.01))
        n_days = int(tp.get("n_days", 1129))
        # Generate a synthetic IS pool: use a simple two-point distribution
        # matching mean and std (Rademacher-like), replicated to n_days points.
        # point+ = mean + std, point- = mean - std, weight 0.5 each
        pool = []
        for i in range(n_days):
            if i % 2 == 0:
                pool.append(mean_d + std_d)
            else:
                pool.append(mean_d - std_d)
        is_daily[pair] = pool

    # -----------------------------------------------------------------------
    # 3. Run bootstrap for each pair and IS pool under all sizing scenarios
    # -----------------------------------------------------------------------
    per_pair_results: dict = {}

    for pair in PAIR_KEYS:
        oos_pool = oos_daily[pair]
        is_pool = is_daily[pair]

        oos_scenarios = {}
        is_scenarios = {}

        for label, scale in SIZING_SCENARIOS.items():
            oos_scenarios[label] = run_bootstrap(oos_pool, scale, rng)
            is_scenarios[label] = run_bootstrap(is_pool, scale, rng)

        per_pair_results[pair] = {
            "oos_n_fold_days": len(oos_pool),
            "is_n_days": len(is_pool),
            "scenarios": oos_scenarios,
            "is_scenarios": is_scenarios,
        }

    # -----------------------------------------------------------------------
    # 4. Combined 50/50 portfolio: average OOS daily returns from both pairs
    # -----------------------------------------------------------------------
    btc_pool = oos_daily["BTCUSDT"]
    bnb_pool = oos_daily["BNBUSDT"]
    min_len = min(len(btc_pool), len(bnb_pool))
    combined_pool = [
        0.5 * btc_pool[i] + 0.5 * bnb_pool[i]
        for i in range(min_len)
    ]

    combined_scenarios = {}
    for label, scale in SIZING_SCENARIOS.items():
        combined_scenarios[label] = run_bootstrap(combined_pool, scale, rng)

    per_pair_results["combined_50_50"] = {
        "oos_n_fold_days": len(combined_pool),
        "scenarios": combined_scenarios,
    }

    # -----------------------------------------------------------------------
    # 5. Verdict
    # -----------------------------------------------------------------------
    verdict = determine_verdict(per_pair_results)

    # -----------------------------------------------------------------------
    # 6. Assemble and write output
    # -----------------------------------------------------------------------
    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "monte_carlo_n_trials": N_TRIALS,
        "days_per_trial": DAYS_PER_TRIAL,
        "random_seed": RANDOM_SEED,
        "n_oos_folds": len(folds),
        "fold_days": fold_days,
        "sizing_scenarios": list(SIZING_SCENARIOS.keys()),
        "per_pair": per_pair_results,
        "verdict": verdict,
        "verdict_criteria": {
            "REALISTIC": "P(daily>=1%) >= 30% AND P(annual>=100%) >= 80%",
            "STRETCH": "P(daily>=1%) in [15%, 30%)",
            "FANTASY": "P(daily>=1%) < 15% OR P(annual_dd>25%) > 50%",
        },
    }

    # Sanitize: replace any residual NaN/Inf (shouldn't occur after safe_float)
    report_str = json.dumps(report, indent=2)
    # Quick sanity: load back
    report_check = json.loads(report_str)

    with open(OUT_PATH, "w") as f:
        json.dump(report_check, f, indent=2)

    print(f"Report written to {OUT_PATH}")

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n=== DAILY TARGET FEASIBILITY SUMMARY ===")
    print(f"Verdict: {verdict}")
    print(f"Trials: {N_TRIALS} x {DAYS_PER_TRIAL} days")
    print()

    for pair in PAIR_KEYS + ["combined_50_50"]:
        data = per_pair_results[pair]
        scen_1x = data["scenarios"]["1.0x"]
        print(f"{pair} @ 1x sizing:")
        print(f"  P(daily >= 1%):      {scen_1x['p_daily_1pct']:.1%}")
        print(f"  Median daily return: {scen_1x['median_daily_return']:.4%}")
        print(f"  5th pct daily:       {scen_1x['p5_daily']:.4%}")
        print(f"  P(annual >= 100%):   {scen_1x['p_annual_100']:.1%}")
        print(f"  P(annual DD > 25%):  {scen_1x['p_annual_dd_25']:.1%}")
        print(f"  Expected annual ret: {scen_1x['expected_annual']:.1%}")
        print()


if __name__ == "__main__":
    main()
