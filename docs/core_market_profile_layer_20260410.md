# Core Market Profile Layer (2026-04-10)

## Decision

- Search and live now read the same market-state representation.
- The shared layer calculates:
  - BTC regime score and regime label
  - cross-sectional breadth
  - route bucket
  - market-context correlation states

## Why It Matters

- Validation and execution no longer depend on different regime math.
- Search robustness findings can now flow into the live decision path.
- This reduces the risk that a strategy passes research checks but is explained or gated differently in production.

## What Was Added

- Shared module: [core_market_profile.py](/Users/jsa/work/epic_invest/scripts/core_market_profile.py)
- Search integration: [search_core_champion.py](/Users/jsa/work/epic_invest/scripts/search_core_champion.py)
- Live integration: [rotation_target_050_live.py](/Users/jsa/work/epic_invest/scripts/rotation_target_050_live.py)
- Telegram rationale exposure: [telegram_bot.py](/Users/jsa/work/epic_invest/scripts/telegram_bot.py)

## Current Outcome

- Champion artifact now stores:
  - promotion gate
  - validation profile
  - candidate-level PBO profile
  - regime robustness summary
  - correlation robustness summary
- Live plan now exposes:
  - validation context
  - route state
  - correlation context snapshot
