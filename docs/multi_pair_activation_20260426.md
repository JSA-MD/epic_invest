# Multi-Pair (6-Pair) Live Trader Activation

Date: 2026-04-26

## Architecture

Multi-instance approach (option B): one independent launchd job per pair, each running
`scripts/pairwise_regime_live.py` with `PAIRWISE_PAIR_OVERRIDE` env var set to the target
symbol. Failure of one pair does not affect others; each has its own kill switch.

Existing BTC+BNB trader (`com.epicinvest.pairwise-trader`) is unchanged.

## New Pairs

| Pair     | Label                        | Entry script                     | State file                        | Decision log                  |
|----------|------------------------------|----------------------------------|-----------------------------------|-------------------------------|
| ETHUSDT  | com.epicinvest.eth-trader    | scripts/eth_live_launchd_entry.sh  | models/eth_live_state.json        | logs/eth_decisions.jsonl      |
| SOLUSDT  | com.epicinvest.sol-trader    | scripts/sol_live_launchd_entry.sh  | models/sol_live_state.json        | logs/sol_decisions.jsonl      |
| XRPUSDT  | com.epicinvest.xrp-trader    | scripts/xrp_live_launchd_entry.sh  | models/xrp_live_state.json        | logs/xrp_decisions.jsonl      |
| DOGEUSDT | com.epicinvest.doge-trader   | scripts/doge_live_launchd_entry.sh | models/doge_live_state.json       | logs/doge_decisions.jsonl     |

## Files Created

- `scripts/pairwise_regime_live.py` — added `PAIRWISE_PAIR_OVERRIDE` env var support (line ~46)
- `scripts/{eth,sol,xrp,doge}_live_launchd_entry.sh` — per-pair entry scripts
- `scripts/com.epicinvest.{eth,sol,xrp,doge}-trader.plist` — launchd plists
- `scripts/install_{eth,sol,xrp,doge}_trader.sh` — load/unload helpers

## Pre-requisites (BLOCKERS — do not activate without these)

**BLOCKER: Each new pair requires its own validated candidate summary JSON.**
Without it the trader crashes at startup when loading `selected_candidate`.

Required files (do not exist yet — must be generated):

| Pair     | Required summary file                              |
|----------|----------------------------------------------------|
| ETHUSDT  | `models/eth_pairwise_candidate_summary.json`       |
| SOLUSDT  | `models/sol_pairwise_candidate_summary.json`       |
| XRPUSDT  | `models/xrp_pairwise_candidate_summary.json`       |
| DOGEUSDT | `models/doge_pairwise_candidate_summary.json`      |

Generation command (one per pair, adjust symbol/output path):

```bash
python scripts/search_pair_subset_regime_mixture.py \
  --pairs ETHUSDT \
  --output models/eth_pairwise_candidate_summary.json
```

After generation, run walk-forward validation:

```bash
python scripts/pairwise_validation_engine.py \
  --summary models/eth_pairwise_candidate_summary.json
```

The summary is ready for live trading only when the validation report shows
`promotion_ready: true` (same gate as BTC/BNB).

Also update `PAIRWISE_LIVE_PROMOTION_REPORT_PATH` in each entry script to point to the
per-pair pipeline report once one is generated, rather than the BTC/BNB default.

## Staged Roll-out Order

Activate pairs one at a time, separated by at least 48 h of demo-mode observation.

1. **ETH** (most liquid alt, tightest spreads — lowest risk)
2. **SOL**
3. **XRP**
4. **DOGE** (highest volatility / widest spread — activate last)

## Activation Steps (per pair, e.g. ETH)

```bash
# 1. Verify summary exists and is valid
ls -lh models/eth_pairwise_candidate_summary.json

# 2. Set MODE to demo in models/pairwise_live_launchd_env.sh first
#    PAIRWISE_LIVE_MODE=demo

# 3. Install job (copies plist, loads with launchd)
bash scripts/install_eth_trader.sh load

# 4. Tail logs and confirm no startup crash
tail -f logs/eth_live_service.log

# 5. After >=48h demo-mode validation, switch to live:
#    PAIRWISE_LIVE_MODE=live  (in pairwise_live_launchd_env.sh)
#    Then reload: install_eth_trader.sh unload && install_eth_trader.sh load
```

## Kill Switch (per pair)

```bash
bash scripts/install_eth_trader.sh unload   # stops job immediately, removes plist
```

## Status Check

```bash
launchctl list | grep com.epicinvest
```
