#!/usr/bin/env python3
"""Re-certify candidate strategies via CPCV / DSR / PBO (Stage 1 governance).

Usage
-----
1. Build a returns matrix CSV with rows = bars, columns = candidates:

       date,v2,v3,v4,v5,v6,v7,v8
       2026-01-01,0.0012,-0.0003,...
       ...

   The first column may be a date or any non-numeric label; it is dropped.

2. Run the recertifier:

       python scripts/recertify_candidates.py \
         --returns-csv path/to/returns_matrix.csv \
         --n-trials 7 \
         --s-subgroups 8 \
         --periods-per-year 252 \
         --output models/candidate_recertification_report.json

3. Decision rule (Stage 1 governance):

   * `pbo < 0.5`   — IS-best selection generalises better than coin flip.
   * `dsr >= 0.95` — at least one candidate's Sharpe is statistically
     significant after multiple-testing correction.
   * `cpcv mean OOS Sharpe > 0.5` — best candidate's true edge survives
     combinatorial purged CV.

   All three conditions must hold before a candidate may be promoted.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cpcv_validation import CPCVConfig, cpcv_select_and_measure
from deflated_sharpe import deflated_sharpe
from pbo_estimate import estimate_pbo

log = logging.getLogger("recertify_candidates")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)


# --- IO helpers ----------------------------------------------------------
def load_returns_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (column_labels, returns_matrix(T, N))."""
    with path.open() as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no data rows")
    # Detect leading non-numeric column (date label).
    def _is_numeric_token(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    drop_first = not _is_numeric_token(rows[0][0])
    label_start = 1 if drop_first else 0
    labels = header[label_start:]
    data = []
    for r in rows:
        cells = r[label_start:]
        if len(cells) != len(labels):
            raise ValueError(
                f"row width mismatch in {path}: expected {len(labels)}, got {len(cells)}"
            )
        data.append([float(c) for c in cells])
    arr = np.asarray(data, dtype=np.float64)
    return labels, arr


# --- Recertification core ------------------------------------------------
def recertify_candidates(
    returns_matrix: np.ndarray,
    *,
    candidate_labels: Sequence[str],
    n_trials: int,
    s_subgroups: int = 8,
    periods_per_year: float = 252.0,
    cpcv_groups: int = 6,
    cpcv_k_test: int = 2,
    embargo_pct: float = 0.01,
) -> dict:
    """Run all three López de Prado tests on a (T, N) matrix."""
    t, n = returns_matrix.shape
    if n != len(candidate_labels):
        raise ValueError(
            f"label count {len(candidate_labels)} != matrix columns {n}"
        )
    if n_trials < n:
        log.warning(
            "n_trials=%s is less than candidate count %s; the multiple-testing "
            "penalty will be understated. Set n_trials to the *total* number "
            "of strategies considered (including ones not in this matrix).",
            n_trials,
            n,
        )

    # 1) Per-candidate Deflated Sharpe ----------------------------------------
    dsr_results = {}
    for j, label in enumerate(candidate_labels):
        col = returns_matrix[:, j]
        try:
            dsr = deflated_sharpe(col, n_trials=n_trials, periods_per_year=periods_per_year)
            dsr_results[label] = {
                "sharpe": dsr.sharpe,
                "expected_max_sharpe": dsr.expected_max_sharpe,
                "dsr": dsr.dsr,
                "skew": dsr.skew,
                "kurtosis_excess": dsr.kurtosis_excess,
                "n_observations": dsr.n_observations,
            }
        except Exception as exc:  # noqa: BLE001
            dsr_results[label] = {"error": f"{type(exc).__name__}: {exc}"}

    # 2) PBO via CSCV ---------------------------------------------------------
    try:
        pbo_res = estimate_pbo(
            returns_matrix, s_subgroups=s_subgroups, metric="sharpe"
        )
        pbo_block = {
            "pbo": pbo_res.pbo,
            "n_splits": pbo_res.n_splits,
            "median_oos_rank_pct": pbo_res.median_oos_rank_pct,
            "mean_logit": pbo_res.mean_logit,
            "fraction_oos_top_quartile": pbo_res.fraction_oos_top_quartile,
        }
    except Exception as exc:  # noqa: BLE001
        pbo_block = {"error": f"{type(exc).__name__}: {exc}"}

    # 3) CPCV under the candidate-selection rule ----------------------------
    # Honest CPCV: for every split we (a) pick the IS-best candidate on the
    # train fold, (b) measure that picked candidate's Sharpe on the held-out
    # test fold. This is the OOS distribution under the same selection rule
    # promotion would apply, so its mean is a real generalisation estimate
    # rather than a stub-driven artefact.
    cpcv_block: dict
    try:
        cfg = CPCVConfig(
            n_groups=cpcv_groups,
            k_test_groups=cpcv_k_test,
            embargo_pct=embargo_pct,
            label_horizon=0,
        )
        stats = cpcv_select_and_measure(
            returns_matrix,
            cfg,
            annualization=periods_per_year,
            selection="sharpe",
        )
        cpcv_block = {
            "n_groups": cpcv_groups,
            "k_test_groups": cpcv_k_test,
            "embargo_pct": embargo_pct,
            **stats,
        }
    except Exception as exc:  # noqa: BLE001
        cpcv_block = {"error": f"{type(exc).__name__}: {exc}"}

    # 4) Decision rule --------------------------------------------------------
    pass_pbo = isinstance(pbo_block, dict) and pbo_block.get("pbo", 1.0) < 0.5
    finite_dsr = {
        k: v for k, v in dsr_results.items()
        if isinstance(v, dict) and "dsr" in v and np.isfinite(v.get("dsr", float("nan")))
    }
    best_dsr = max(
        (v.get("dsr", 0.0) for v in finite_dsr.values()),
        default=0.0,
    )
    pass_dsr = best_dsr >= 0.95
    cpcv_n = int(cpcv_block.get("n_paths_evaluated", 0)) if isinstance(cpcv_block, dict) else 0
    cpcv_mean = (
        float(cpcv_block.get("mean_sharpe", float("nan")))
        if isinstance(cpcv_block, dict) else float("nan")
    )
    pass_cpcv = (
        cpcv_n > 0
        and np.isfinite(cpcv_mean)
        and cpcv_mean > 0.5
    )
    decision = {
        "pass_pbo_lt_0_5": bool(pass_pbo),
        "pass_dsr_ge_0_95": bool(pass_dsr),
        "pass_cpcv_oos_sharpe_gt_0_5": bool(pass_cpcv),
        "ready_for_promotion": bool(pass_pbo and pass_dsr and pass_cpcv),
    }

    # `generated_at` is consumed by promotion_gate_guard._report_age_days
    # to enforce the freshness window. Reports without it are correctly
    # rejected as "undated" — never silently accepted — so the recertifier
    # must always emit it.
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_periods": t,
        "n_candidates": n,
        "n_trials_assumed": n_trials,
        "deflated_sharpe": dsr_results,
        "pbo": pbo_block,
        "cpcv": cpcv_block,
        "decision": decision,
    }


# --- CLI -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--returns-csv",
        type=Path,
        required=True,
        help="CSV with bar-level candidate returns (rows=bars, cols=candidates).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        required=True,
        help="Total number of strategy variants tried (e.g. multitree v1..v8 → 8).",
    )
    p.add_argument(
        "--s-subgroups",
        type=int,
        default=8,
        help="Even number of CSCV subgroups (default 8 → C(8,4)=70 splits).",
    )
    p.add_argument(
        "--periods-per-year",
        type=float,
        default=252.0,
        help="Sharpe annualisation factor (252 daily, 8760 hourly).",
    )
    p.add_argument(
        "--cpcv-groups",
        type=int,
        default=6,
    )
    p.add_argument(
        "--cpcv-k-test",
        type=int,
        default=2,
    )
    p.add_argument(
        "--embargo-pct",
        type=float,
        default=0.01,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("models/candidate_recertification_report.json"),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    labels, returns_matrix = load_returns_csv(args.returns_csv)
    log.info("Loaded %d periods × %d candidates from %s", *returns_matrix.shape, args.returns_csv)
    report = recertify_candidates(
        returns_matrix,
        candidate_labels=labels,
        n_trials=args.n_trials,
        s_subgroups=args.s_subgroups,
        periods_per_year=args.periods_per_year,
        cpcv_groups=args.cpcv_groups,
        cpcv_k_test=args.cpcv_k_test,
        embargo_pct=args.embargo_pct,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    log.info("Wrote %s", args.output)
    print(json.dumps(report["decision"], indent=2))
    return 0 if report["decision"]["ready_for_promotion"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
