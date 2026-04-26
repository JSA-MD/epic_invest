
# Live vs Backtest Drift — Root Cause Diagnostic (2026-04-26)

**Window:** 2026-03-27 – 2026-04-26 (31 days)  

**Base notional:** $4,532.41  

**Total gap:** $-5,862.70 (-129.3% of base)  

**Avg daily gap:** -417 bps/day  


## Top-7 Worst Drift Dates

| Date | Gap USD | Gap bps | Live USD | BT USD | Primary Cause |
| --- | --- | --- | --- | --- | --- |
| 2026-04-13 | $-2,109 | -4652 | $-2,152 | $-43 | C2a: gross oversize tw=-1.5 held into BNB squeeze (+32% in 1 day) |
| 2026-04-25 | $-1,901 | -4194 | $-1,901 | $0 | C2b: open BNB short carried from Apr 19, unrealized -1873 USD |
| 2026-04-19 | $-1,387 | -3061 | $-1,245 | $142 | C2b: BNB short entered at tw=-0.46, backtest flat; position not closed |
| 2026-04-14 | $-517 | -1141 | $-517 | $0 | C2a: tail of Apr 10-14 hold, partial unwind at loss |
| 2026-03-27 | $-136 | -301 | $0 | $136 | C1: early BNB suppression (state_alphas key mismatch, live=0 vs bt+136) |
| 2026-04-10 | $-83 | -183 | $-110 | $-27 | C2a: initial BNB short entry at tw=-1.5, backtest at ~1% daily weight |
| 2026-03-28 | $-30 | -66 | $0 | $30 | minor/rounding |


## Gap Attribution by Cause

| Cause | Gap USD | % of Total | Description |
| --- | --- | --- | --- |
| C1: Early suppression | $-167 | 2.9% | Mar 27-31 live=0, backtest had BNB signal (state_alphas key mismatch) |
| C2a: Gross oversize Apr 10-14 | $-2,437 | 41.6% | BNB tw=-1.5 held 5 days into squeeze; backtest used 1-3% daily weight |
| C2b: Open carry Apr 19+25 | $-3,288 | 56.1% | Apr 19 BNB short not closed; unrealized -1873 reached by Apr 25 |
| C3: Stale price feed | enabling factor | n/a | Price constant for hours; prevents intraday rebalancing (amplifies C2) |
| C4: Regime gate behaviour | gate mechanism | n/a | BTC blocked 100%; BNB passed Apr 10-13 then blocked; consistent with backtest |
| Residual | $29 | -0.5% | Small days, rounding |


## Cause 1 — Router Suppression Intermittency

**BNB suppression rate:** 56.9% of negative-signal bars

**Regime score — suppressed bars:** mean=0.021868  (positive = BNB breadth/regime gate blocks short)

**Regime score — executed bars:**   mean=-0.016728  (negative = gate passes)



BNB route-state cross-tab (negative-signal bars only):

| Route State | exec_short | exec_flat | exec_rate |
| --- | --- | --- | --- |
| equity_aligned:bear_broad | 22 | 61 | 27% |
| equity_aligned:bear_narrow | 1303 | 0 | 100% |
| equity_aligned:bull_broad | 0 | 1190 | 0% |
| equity_aligned:bull_narrow | 0 | 17 | 0% |
| equity_mixed:bear_broad | 0 | 20 | 0% |
| equity_mixed:bear_narrow | 10 | 0 | 100% |
| equity_mixed:bull_broad | 12 | 430 | 3% |
| equity_mixed:bull_narrow | 0 | 133 | 0% |
| unknown | 57 | 0 | 100% |

BNB suppressed 56.9% of negative-signal bars. Route state 'equity_aligned:bull_broad' suppresses 100% (1190/1190), 'equity_mixed:bull_narrow' suppresses 100% (132/132) — the state_alphas config only holds key 'equity_mixed:bull_broad'. BTC suppressed 97.8% of all bars due to regime_score persistently > 0.02 threshold.

**Fix B** (from prior report, data-only): add `equity_mixed:bull_narrow: 0.2` to BNB `state_alphas` in the candidate summary JSON. No code change needed.  

**Expected gap reduction:** ~$167 (2.9% of total gap) — fixes early suppression period only.

Note: On the worst days (Apr 13, 19, 25) the suppression was PARTIALLY FAILING — the live system was executing despite the key mismatch via the old shadow model. Fixing C1 alone does not address the dominant C2 loss.


## Cause 2 — Position Sizing / Holding Period Mismatch

**This is the dominant cause: 97.6% of total gap.**



Sub-cause 2a — Apr 10-14 gross oversize:

- BNB executed at `target_weight = -1.5` (gross_cap=1.5 × $4,532 = **$6,799 notional short**)

- Backtest regime_mixture uses ~1-3% daily returns on small per-bar weights ($100-300 equivalent)

- Position held continuously for 5 days; BNB rallied +32% on Apr 13 alone

- Live 5-day total: **-$2,341** vs backtest +$93. Gap: **-$2,434**



Sub-cause 2b — Apr 19 / Apr 25 open carry:

- Apr 19: BNB short entered at tw≈-0.46, realized -589, unrealized -653 (not closed)

- Apr 20-24: live PnL = 0 (decision_journal shows session=pairwise tw=-0.021), but

  exchange position still open (unrealized loss carried forward)

- Apr 25: unrealized swells to -$1,873; live total day -$1,880 vs backtest $0

- Combined Apr 19+25 gap: **-$3,288**



Live system ran at tw=-1.5x (gross_cap=1.5, notional=$6,799) — 25-50x larger than backtest regime_mixture daily positions (~$100-300 equivalent). Multi-day holding amplifies adverse moves that the 5-min bar backtest exits intraday.

**Apr 10-14 hold:** live=-2,341, bt=+93, gap=-2,434

**Apr 19 unrealized:** -654 USD carried forward

**Apr 25 unrealized:** -1,874 USD at date close



**Recommended fixes (C2):**

1. **Enforce `max_hold_bars` = 288 (24 h)** in `pairwise_regime_live.py` — close any position that has been open longer than 1 day. This alone would have prevented the Apr 10-14 hold from accumulating to -$2,341.

2. **Cap `gross_cap` to 0.05 during validation** (Stage A sizing = 1%) — live 1.5x notional vs backtest micro-sizing is the core mismatch.

3. **Add EOD reconciliation**: compare `current_weight` in decisions log to exchange position via REST; force-close if divergence > 0.01.


## Cause 3 — Execution Timing / Stale Price Feed

Price feed is severely stale: Apr 10 shows price=601.54 for all 55 executed bars (staleness 100%). Apr 12 changes price only at end of day. Stale prices prevent intraday rebalancing — live system cannot detect when an adverse price move should trigger a stop or position reduction. This is a secondary/enabling cause; it amplifies Cause 2 by preventing stop-loss triggers during multi-day holds.



Price staleness by date:

| Date | Pair | Exec Bars | Unique Prices | Staleness% | Intraday Range% |
| --- | --- | --- | --- | --- | --- |
| 2026-04-10 | BNBUSDT | 55 | 1 | 100.0% | 0.00% |
| 2026-04-10 | BTCUSDT | 55 | 1 | 100.0% | 0.00% |
| 2026-04-12 | BNBUSDT | 1019 | 238 | 70.3% | 3.10% |
| 2026-04-12 | BTCUSDT | 1019 | 295 | 70.0% | 3.53% |
| 2026-04-13 | BNBUSDT | 656 | 244 | 57.3% | 4.23% |
| 2026-04-13 | BTCUSDT | 656 | 277 | 56.8% | 5.94% |
| 2026-04-19 | BNBUSDT | 122 | 112 | 3.3% | 1.88% |
| 2026-04-19 | BTCUSDT | 122 | 119 | 1.7% | 1.43% |
| 2026-04-25 | BNBUSDT | 148 | 121 | 6.8% | 1.71% |
| 2026-04-25 | BTCUSDT | 148 | 142 | 2.7% | 0.83% |

**Fix C3:** Ensure `latest_prices` in the plan is refreshed from exchange REST on every poll cycle, not cached from the LOB snapshot. This is a secondary issue — it amplifies C2 but does not independently cause the gap.


## Cause 4 — Regime Score Divergence

BTC regime_score > 0.02 on 95-100% of bars across the entire 31-day window — BTC short gate STRUCTURALLY BLOCKED in live; this matches backtest (backtest BTC also flat Apr 10+) so adds no gap. BNB gate passed (regime_score < 0) on Apr 10-13 only; regime turned positive Apr 14+ and stayed there. The gate is not 'wrong' — regime score correctly captured BNB bear-momentum Apr 10-13 — but the position size allowed by the gate was 25x what the backtest uses.



BTC regime gate (threshold -0.02) pass rate by date:

| Date | Mean regime_score | Gate Pass % | N Bars |
| --- | --- | --- | --- |
| 2026-04-10 | 0.0297 | 0.0% | 57 |
| 2026-04-11 | 0.1732 | 0.0% | 482 |
| 2026-04-12 | 0.0298 | 0.0% | 1019 |
| 2026-04-13 | 0.0247 | 0.0% | 656 |
| 2026-04-14 | 0.0482 | 0.0% | 355 |
| 2026-04-15 | 0.0642 | 0.0% | 114 |
| 2026-04-17 | 0.0895 | 0.0% | 117 |
| 2026-04-18 | 0.0656 | 0.0% | 227 |
| 2026-04-19 | 0.0405 | 0.0% | 125 |
| 2026-04-25 | 0.0289 | 0.0% | 148 |
| 2026-04-26 | 0.0308 | 0.0% | 20 |

BNB regime gate (threshold 0.0) pass rate by date:

| Date | Mean regime_score | Gate Pass % | N Bars |
| --- | --- | --- | --- |
| 2026-04-10 | -0.0264 | 100.0% | 57 |
| 2026-04-11 | 0.0804 | 4.6% | 482 |
| 2026-04-12 | -0.0159 | 93.0% | 1019 |
| 2026-04-13 | -0.0072 | 66.9% | 656 |
| 2026-04-14 | 0.0073 | 0.0% | 355 |
| 2026-04-15 | 0.0264 | 0.0% | 114 |
| 2026-04-17 | 0.0647 | 0.0% | 117 |
| 2026-04-18 | 0.0412 | 0.0% | 227 |
| 2026-04-19 | 0.0038 | 24.0% | 125 |
| 2026-04-25 | 0.0161 | 0.0% | 148 |
| 2026-04-26 | 0.0144 | 0.0% | 20 |

**Key insight:** the regime gate is NOT wrong — it reflects real momentum. The problem is that when the gate passes (Apr 10-13), the live system enters at full 1.5x leverage instead of the small backtest-equivalent weight.


## Recommended Fix Order and Expected Gap Reduction

| Priority | Fix | Type | Expected Gap Reduction | Effort |
| --- | --- | --- | --- | --- |
| P0 | Enforce max_hold_bars=288 (24 h position limit) | code | ~$2,000 (34% of gap) | 1 function in pairwise_regime_live.py |
| P0 | Stage A sizing: gross_cap=0.01 during validation | config | ~$3,000 (51% — prevents C2b recurrence) | JSON config change |
| P1 | EOD position reconciliation vs exchange REST | code | ~$500 (8% — prevents carry) | new reconcile function |
| P2 | Fix BNB state_alphas: add equity_mixed:bull_narrow key | config | ~$167 (3% — fixes early suppression) | JSON config change |
| P3 | Refresh latest_prices from REST on every poll | code | enabling fix (no direct $) | price fetch in run_live_once |


## Specific Config / Code Change for Each Cause

**P0-A: max_hold_bars (pairwise_regime_live.py)**

```python

# In compute_requested_weight or run_live_once:

# Add hold tracking to pair state; force weight=0 if hold exceeds limit.

MAX_HOLD_BARS = 288  # 24 h at 5-min poll

if state.hold_bars.get(pair, 0) >= MAX_HOLD_BARS:

    requested_weight = 0.0  # force closure

    state.hold_bars[pair] = 0

```

**P0-B: gross_cap in candidate JSON**

```json

// gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json

// Change: gross_cap 1.5 -> 0.01 for Stage A validation

"gross_cap": 0.01  // was 1.5

```

**P1: EOD reconciliation (new function)**

```python

def reconcile_positions(client, pair_plans, notional):

    for pair, plan in pair_plans.items():

        exchange_pos = client.get_position(pair)

        target_pos   = plan['target_weight'] * notional

        if abs(exchange_pos - target_pos) > notional * 0.01:

            client.close_position(pair)  # force reconcile

```

**P2: BNB state_alphas JSON fix**

```json

"state_alphas": {

    "equity_mixed:bull_broad":  0.2,

    "equity_mixed:bull_narrow": 0.2   // ADD THIS KEY

}

```

**P3: Price refresh in run_live_once**

```python

# Replace cached LOB price with REST ticker on each poll:

latest_prices = {p: float(client.get_ticker(p)['lastPrice']) for p in pairs}

```
