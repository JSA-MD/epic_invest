# Artifact Review 2026-04-10

## Keep

- `models/core_stop_loss_2pct_compare.json`
  - Core stop-loss idea was rejected, but this file is the direct decision record for that rejection.
- `models/gp_regime_mixture_btc_bnb_pairwise_stress_report_rerun.json`
  - Pairwise strategy를 실전 승격하지 않고 shadow에 유지하는 핵심 근거다.
- `scripts/pairwise_regime_mixture_shadow_live.py`
  - Source code, not runtime output. It should be tracked once reviewed.
- `scripts/run_smoke_checks.py`
  - Operational smoke gate for critical scripts.
- `tests/`
  - New regression guardrail for watchdog, Telegram, and core live loading.
- `docs/artifact_policy.md`
  - Operating rule for what belongs in Git.
- `models/rotation_target_050_live_state.example.json`
  - Example schema replaces live state tracking.

## Hold

- 없음

## Drop From Git Review Queue

- `models/ga_long_short_rotation_curve.csv`
- `models/ga_long_short_rotation_daily_returns.csv`
- `models/ga_long_short_rotation_selection.csv`
- `models/ga_long_short_rotation_weights.csv`
  - These are detailed traces for a losing candidate and are redundant with tracked summary artifacts.
- `models/pairwise_regime_stop_loss_2pct_compare.json`
  - Pairwise stop-loss is not needed while pairwise remains shadow-only and stop-loss itself has already been rejected.

## Current Decision

- Pairwise strategy stays in shadow mode.
- Stop-loss branch remains rejected.
- Durable evidence should stay concise: summary, report, champion, and approved examples.
