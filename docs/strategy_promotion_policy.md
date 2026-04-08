# Strategy Promotion Policy

## Locked Baseline

Current promotion baseline:
- summary: [/Users/jsa/work/epic_invest/models/rotation_target_050_summary.json](/Users/jsa/work/epic_invest/models/rotation_target_050_summary.json)
- manifest: [/Users/jsa/work/epic_invest/models/rotation_target_050_baseline.json](/Users/jsa/work/epic_invest/models/rotation_target_050_baseline.json)
- source commit: `b415696`

The accepted stage group is `stages.levered_hybrid`.

Tracked metrics per stage:
- `total_return`
- `avg_daily_return`
- `daily_win_rate`
- `daily_target_hit_rate`
- `max_drawdown`

Compared stages:
- `validation`
- `test`
- `oos`

Required target checks:
- `validation_pass`
- `test_pass`
- `oos_pass`

## Promotion Rule

A candidate is promotable only when all of the following are true:

1. `target_check.validation_pass == true`
2. `target_check.test_pass == true`
3. `target_check.oos_pass == true`
4. None of the tracked metrics regress versus the baseline in `validation`, `test`, or `oos`
5. At least one tracked metric improves versus the baseline

If any tracked metric is worse, the candidate is rejected and should be discarded instead of committed.

## Workflow

1. Generate a candidate summary JSON from the new experiment.
2. Run:

```bash
python3 scripts/check_strategy_promotion.py \
  --candidate-summary /absolute/path/to/candidate_summary.json \
  --require-improvement
```

3. Only if the command exits with `0`, commit and push the candidate.
4. If the command exits with non-zero, discard the candidate as a regression or non-improvement.
