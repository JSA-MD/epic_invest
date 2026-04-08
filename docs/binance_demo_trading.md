# Binance Demo Trading Setup

This project now uses Binance USD-M Demo Trading instead of the deprecated USD-M futures sandbox path.

## Required Environment

Add these values to `.env`:

```dotenv
BINANCE_MODE=demo
BINANCE_DEMO_API_KEY=your_demo_api_key
BINANCE_DEMO_API_SECRET=your_demo_api_secret
# default leverage is 1.08x for the validated target-0.5% path
# STRATEGY_LEVERAGE=1.08
REBALANCE_NOTIONAL_BAND_USD=25
```

`BINANCE_DEMO_API_SECRET` is required for actual order placement.

## Commands

Recommended process control:

```bash
./start.sh
./stop.sh
./restart.sh
```

These scripts:

- check existing `rotation_target_050_live.py` and legacy `live_trader.py` processes
- remove stale PID files
- stop duplicate processes before start
- install exchange-native reduce-only shutdown protection on `stop`/`restart`
- write logs to `/tmp/epic-invest-trader.log`
- track the active PID in `/tmp/epic-invest-trader.pid`
- let the running strategy cancel those managed protection orders again on next sync/start

Show the current daily plan without orders:

```bash
.venv/bin/python scripts/rotation_target_050_live.py status --equity 100000
```

Run one dry-run reconciliation against Binance Demo public endpoints:

```bash
.venv/bin/python scripts/rotation_target_050_live.py run-once --equity 100000
```

Place demo orders:

```bash
.venv/bin/python scripts/rotation_target_050_live.py run-once --execute
```

Sync local state with exchange positions and cancel this strategy's managed conditional orders:

```bash
.venv/bin/python scripts/rotation_target_050_live.py sync-state --execute
```

Install exchange-native shutdown protection without starting the loop:

```bash
.venv/bin/python scripts/rotation_target_050_live.py shutdown-protect --execute
```

Close all open positions immediately and clear the local strategy state:

```bash
.venv/bin/python scripts/rotation_target_050_live.py close-all --execute
```

Run continuously:

```bash
.venv/bin/python scripts/rotation_target_050_live.py loop --execute --poll-seconds 60
```

## Strategy Shape

- Core: aggressive walk-forward long-only rotation across BTC, ETH, SOL, XRP
- Overlay: BTC intraday session only when core is flat
- Overlay signal:
  - BTC 1-day momentum
  - 3-day cross-asset breadth
  - `trend_both` mode
- Core emergency kill switch:
  - portfolio-level only, not per-position TP/SL
  - monitored from portfolio equity while the loop is running
  - locks the strategy flat for the rest of the day after a catastrophic intraday drawdown
  - threshold defaults to `1.5 x target daily vol`, clipped to `6% ~ 10%`
## Notes

- State is stored in `models/rotation_target_050_live_state.json`
- Default leverage is `1.08x`; `STRATEGY_LEVERAGE` still wins if set explicitly
- New opening/rebalance orders attempt to force `isolated` margin mode before leverage/order placement
- While the loop is running, exits still follow the local algorithm
- While the loop is running on a core day, a portfolio-level emergency kill switch can flatten all core positions and prevent same-day re-entry
- On stop/restart or caught termination signals, the runner installs fallback `reduceOnly` SL/TP orders at `1:2.5`
- On the next start/sync, those managed fallback orders are cancelled and positions are re-synced into the strategy state
- `close-all --execute` uses `reduceOnly` market orders, but if the live loop is still running it may reopen positions on the next cycle
- The older `scripts/live_trader.py` path still points at the deprecated sandbox-style futures setup and should not be used for this strategy
