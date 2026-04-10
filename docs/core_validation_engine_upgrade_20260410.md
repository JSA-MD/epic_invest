# Core Validation Engine Upgrade (2026-04-10)

## Decision

- Objective changed from `find more good-looking candidates` to `kill false positives faster`.
- Core champion promotion is now blocked unless validation hard gates pass.
- Current champion was **not re-promoted** under the upgraded engine.

## What Changed

- Added hard thresholds for:
  - OOS DSR proxy
  - CPCV pass rate / positive rate / tail loss
  - CPCV split overfit rate
  - PBO-style train-winner test-rank degradation
  - generalization gap / return stability gap
  - fold robustness
  - regime / correlation robustness
- Changed artifact behavior:
  - if no candidate passes, keep the existing champion artifact instead of overwriting it with the top-ranked loser

## Validation Result

- Report: [models/core_competition_validation_recheck_pbo_20260410.json](/Users/jsa/work/epic_invest/models/core_competition_validation_recheck_pbo_20260410.json)
- CPCV candidate-selection PBO:
  - `n_splits = 15`
  - `pbo = 0.4667`
  - `avg_selected_test_percentile = 0.5797`
  - `worst_selected_test_percentile = 0.0875`

## Executive Meaning

- The search engine can still find attractive candidates.
- The upgraded validation engine shows those candidates are not robust enough to be promoted.
- The immediate bottleneck is validation quality, not strategy breadth.

## Current Status

- Existing live champion artifact remains active.
- New promotion is frozen until a candidate clears the upgraded validation engine.
