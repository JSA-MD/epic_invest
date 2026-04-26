# Live Drift Root Cause Analysis — 2026-04-26

**Signal computed:** BTC signal_pct ≈ −197%, BNB signal_pct ≈ −167%  
**Specialist target_weight:** −0.21 (BNB), −0.18 (BTC)  
**Live output:** target_weight ≈ 1e-14 (both pairs), session_type = flat  
**Promotion gate:** ready_for_live = True (validated_stress_report)  

---

## 1. Exact Suppression Path

### 1a. BTC — Regime Short-Gate Blocks Baseline

File: `scripts/pairwise_regime_live.py`, function `compute_requested_weight`, lines 556–563:

```python
short_ok = regime_score <= -effective_regime_threshold and \
           breadth_score <= (1.0 - float(params.breadth_threshold))
...
elif requested_weight < 0.0 and not short_ok:
    signal_pct = 0.0
    requested_weight = 0.0          # ← BTC baseline zeroed here
```

BTC `pair_config` has `regime_threshold=0.02`. The condition for `short_ok` requires
`regime_score <= −0.02`. But the live regime_score is **+0.019** (positive — weakly bullish
momentum), so `short_ok = False` and the −197% signal collapses to 0 before blending.

| Timestamp | session_type | regime_score | threshold | short_ok | BTC requested_weight |
|-----------|-------------|-------------|-----------|----------|---------------------|
| 2026-04-25T21:55 | pairwise | +0.01936 | ±0.02 | **True*** | −0.0684 |
| 2026-04-25T22:02 | flat | +0.01938 | ±0.02 | **False** | 0.0000 |

*At 21:55 the short was allowed because `regime_score` was read from the trace one bar earlier
(the synthetic planning bar shifts the signal index), crossing the threshold sign between bars.

The BTC blend is `mode=always, alpha=0.4`, so:

```
final_target_weight = 0.6 × baseline(0.0) + 0.4 × specialist(−0.18) = −0.072
```

…but only when `baseline_weight != 0`. With `baseline_weight = 0`, `blend_runtime_weight`
correctly computes −0.072. **The blend IS applied** — yet BTC still shows `target_weight ≈ −6e-14`
(effectively zero). The reason is that `specialist_plan` itself uses the same gate-blocked
`requested_weight` via `_build_trace_driven_pair_plan` → `_trace_scalar(trace, "target_weight")`,
which reflects the end-of-trace value from `realistic_overlay_replay_from_context`. The specialist
config also has short-gate parameters (`regime_threshold`, `breadth_threshold`) that suppress the
trace at the same bar. So **both baseline and specialist traces are zeroed by the short gate**, and
blending 0 with 0 yields 0.

### 1b. BNB — `state_alphas` Key Mismatch Suppresses Specialist

File: `scripts/btc_convex_blend.py`, function `resolve_blend_alpha`, lines 94–97:

```python
def resolve_blend_alpha(*, ..., state_alphas, ...):
    if state_alphas is not None and route_state_name in state_alphas:
        return float(state_alphas[route_state_name])   # only match
    if mode == "state_alphas":
        return 0.0                                      # ← fallthrough: alpha=0
```

BNB blend config (from `gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json`):

```json
"mode": "state_alphas",
"state_alphas": { "equity_mixed:bull_broad": 0.2 }
```

Live route state: **`equity_mixed:bull_narrow`**

`"equity_mixed:bull_narrow"` is not in `state_alphas`, so `resolve_blend_alpha` returns `0.0`.
With `blend_alpha = 0`, `blend_runtime_weight` returns `baseline_weight` unchanged:

```python
if blend_alpha <= 0.0:
    return float(baseline_weight)     # baseline = 0.0 (short gate blocked)
```

BNB's baseline `requested_weight = 0.0` (breadth gate: `breadth_score=0.0 <= (1−0.65)=0.35`,
short gate passes, but `regime_score=+0.006 > −0.0` so short_ok also requires regime_score ≤ 0,
fails). The specialist weight −0.21 is computed but **never blended in** because the route state
does not match the single key in `state_alphas`.

The `target_weight ≈ −2.3e-8` seen in BNB is floating-point residual from a prior trace bar,
not a real position — it falls far below `TARGET_WEIGHT_EPS = 1e-6`.

### 1c. Summary Signal Flow

```
signal_pct (BTC: −197%, BNB: −167%)
    │
    ▼ realistic_overlay_replay_from_context
    │  (baseline trace, specialist trace)
    │
    ├─► compute_requested_weight  ←─ short_ok=False (regime_score > 0)
    │       └─► baseline_weight = 0.0  [pairwise_regime_live.py:558-563]
    │
    ├─► blend_runtime_weight (BTC, mode=always)
    │       └─► 0.6×0 + 0.4×specialist_tw(0.0) = 0.0  [btc_convex_blend.py:127]
    │            (specialist trace also zeroed by same regime gate)
    │
    ├─► blend_runtime_weight (BNB, mode=state_alphas)
    │       resolve_blend_alpha: route="bull_narrow" ∉ state_alphas → alpha=0.0
    │       └─► returns baseline_weight = 0.0  [btc_convex_blend.py:126]
    │
    ▼
target_weights: {BTC: ~1e-14, BNB: ~1e-8}
gross = sum(|weights|) = ~1e-8  <  TARGET_WEIGHT_EPS (1e-6)
    │
    ▼
session_type = "flat"   [pairwise_regime_live.py:913]
```

---

## 2. What PAIRWISE_FORCE_EXECUTE=0 Actually Gates

`PAIRWISE_FORCE_EXECUTE` is read at `run_live_once`, lines 1463–1507:

```python
force_execute = bool(getattr(args, "force_execute", False))   # False (env=0)
gate_ready = promotion_gate_allows_execution(promotion_gate, args.mode)
if not gate_ready and not force_execute:
    # ... log and return 2 (blocked)
```

`PAIRWISE_FORCE_EXECUTE=0` gates only the **exchange execution block** — it prevents orders from
being submitted when `promotion_gate.ready_for_demo = False`. It does **not** affect signal
computation, blending, or `session_type`. In the current live run:

- `promotion_gate.ready_for_demo = True` → `gate_ready = True`  
- The execution block is entered (orders are attempted)  
- `force_execute=False` is irrelevant because the gate already passes  

The `force_execute` flag only matters when `gate_ready=False`. Since the gate is `True`,
`PAIRWISE_FORCE_EXECUTE=0` has no effect on the current flat behavior.

**The flat output is entirely a signal/blending problem upstream of execution gating.**

---

## 3. State-Specialists / route_state_mode Interaction

### Which Mapping Indices Are Selected for `equity_mixed:bull_narrow`?

`route_state_mode = "equity_corr"` produces 9 named states:

```
equity_aligned:bull_broad, equity_aligned:bull_narrow,
equity_aligned:bear_broad, equity_aligned:bear_narrow,
equity_mixed:bull_broad, equity_mixed:bull_narrow,   ← index 5
equity_mixed:bear_broad, equity_mixed:bear_narrow,
equity_unknown
```

`bucket_code` for `equity_mixed:bull_narrow` is index 5. The `mapping_indices` array maps each
bucket code to a library parameter set index. From the live decision log:

- BTC: `route_bucket=6`, `route_mapping_index=3769` — the active param set
- BNB: `route_bucket=6`, `route_mapping_index=2043`

`bucket_code=6` corresponds to `equity_mixed:bull_narrow` (0-indexed from the equity_corr list).
The mapped library entry (index 3769 / 2043) carries the `regime_threshold` and
`breadth_threshold` that define the short gate. Neither pair has a configured `state_specialists`
override in the current candidate, so no abstain specialist is injected — the standard regime gate
applies uniformly.

### Why Backtest Differs (12.5% / 15.9% Sign Mismatch)

The backtest uses historical bar data where `regime_score` fluctuates across the threshold
continuously. In the live replay, the synthetic planning bar appended at line 729-730 uses the
last completed bar's values for the current period, creating a stable positive regime_score that
persistently blocks the short gate. The live `regime_score ≈ +0.019` (vs. threshold −0.02) means
the signal is structurally gated in this market regime regardless of signal magnitude.

---

## 4. Key JSON Field That Flips

In `pairwise_regime_decisions.jsonl`, compare:

**Non-flat (2026-04-25T21:55:41):**
```json
"BTCUSDT": {
  "requested_weight": -0.0684,
  "target_weight": -0.0684,
  "regime_score": 0.019359,
  "blend": { "mode": "always", "alpha": 0.4,
              "specialist_target_weight": -0.1801 }
}
```

**Flat (2026-04-25T22:02:01):**
```json
"BTCUSDT": {
  "requested_weight": 0.0,          ← ← ← flipped to 0
  "target_weight": -6.4e-14,
  "regime_score": 0.019376,         ← still positive (short gate blocks)
  "blend": { "mode": "always", "alpha": 0.4,
              "specialist_target_weight": -6.9e-14 }  ← specialist also 0
}
```

The single field that drives the flip: **`requested_weight`** in `BTCUSDT.pair_plans`, toggling
from −0.0684 to 0.0 when `short_ok` in `compute_requested_weight` evaluates False (regime_score
positive, not ≤ −threshold). The specialist trace collapses simultaneously because it uses the
same regime gate logic internally.

For BNB the discriminating field is **`blend.state_alphas`** — the route key
`"equity_mixed:bull_narrow"` is permanently absent, so `specialist_target_weight` is computed
but never applied.

---

## 5. Concrete Patch Design (NOT Applied)

Two independent fixes are needed; either alone partially restores execution.

### Fix A — BTC: Decouple Specialist Execution from Regime Short-Gate

The specialist config should run without the regime short-gate suppressing it, or the blend should
use specialist weight unconditionally when the baseline is zeroed by a regime gate (not by signal).

**Smallest change** — in `btc_convex_blend.py`, `blend_runtime_weight`: when `mode="always"` and
`baseline_weight == 0` but the gate log indicates regime suppression, use specialist weight
directly at the configured alpha:

```python
# btc_convex_blend.py — blend_runtime_weight  (patch, not applied)
def blend_runtime_weight(
    *,
    baseline_weight: float,
    specialist_weight: float,
    route_state_name: str,
    alpha: float,
    mode: str,
    state_alphas: Mapping[str, float] | None = None,
    baseline_gate_suppressed: bool = False,   # NEW param
) -> float:
    blend_alpha = resolve_blend_alpha(...)
    if blend_alpha <= 0.0:
        return float(baseline_weight)
    if baseline_gate_suppressed and mode == "always":
        # Use specialist weight at full alpha when baseline is regime-gated (not signal-zero)
        return float(alpha * specialist_weight)
    return float((1.0 - blend_alpha) * baseline_weight + blend_alpha * specialist_weight)
```

Caller in `pairwise_regime_live.py` line 785-792 passes
`baseline_gate_suppressed = (baseline_plan["requested_weight"] == 0.0 and abs(baseline_plan["signal_pct"]) > 10.0)`.

**Alternative smallest change** — give the specialist config a separate `regime_threshold=0.0`
(or negative) in `specialist_pair_config`, so its trace is not zeroed by the bullish-regime gate.
This requires updating the saved candidate JSON, not the code.

### Fix B — BNB: Add `equity_mixed:bull_narrow` to `state_alphas`

```python
# In the candidate summary JSON (gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json)
# BNBUSDT.pair_convex_blends entry:
"state_alphas": {
    "equity_mixed:bull_broad": 0.2,
    "equity_mixed:bull_narrow": 0.2    # ← add this key
}
```

This is a data fix (no code change). The route has been `equity_mixed:bull_narrow` for the
entire recent observation window. The `bull_broad` key was never matched.

### Fix C — Promotion Report Path Resolution (Structural)

`resolve_runtime_summary_path` (line 277) switches to `DEFAULT_MARKET_OS_SUMMARY_PATH` only when
the pipeline report is `_promotion_report_runtime_ready`. The pipeline report's
`decision.ready_for_live = False`, so the system falls back to the validated_stress_report
summary. Ensure the promotion report used for summary resolution matches the one used for gate
evaluation (currently divergent: gate uses validated_stress_report, summary resolves from
pipeline_report).

---

## 6. Three-Stage Activation Plan

### Stage A — 1% Notional Sizing (Observation)

**Change:** Scale `gross_cap` to `0.01` in both `pair_configs` for the candidate in use.  
**Gate criteria to advance:**
- ≥ 288 bars observed (24 h) with non-flat session_type
- Max drawdown < 2% on shadow paper
- Signal sign matches backtest for ≥ 80% of bars in the window
- BTC `requested_weight != 0` on ≥ 50% of bars

**Rollback:** Revert `gross_cap` to 0. Trigger if max_drawdown > 2% or ≥ 3 consecutive execution
errors.

### Stage B — 5% Notional Sizing (Validation)

**Change:** Scale `gross_cap` to `0.05`. Apply Fix B (add `bull_narrow` to BNB state_alphas).
Set `PAIRWISE_LIVE_MODE=demo` and confirm orders execute in demo exchange.  
**Gate criteria to advance:**
- ≥ 864 bars (72 h) with session_type=pairwise
- Shadow return > −1% over the window
- Kill-switch not triggered (`kill_switch_pct=0.08` for BNB, `0.16` for BTC)
- Backtest/live sign mismatch < 10%

**Rollback:** Drop to Stage A sizing. Trigger if shadow return < −3% or any single session
exceeds kill-switch threshold.

### Stage C — 20% Notional Sizing (Live)

**Change:** Apply Fix A (specialist alpha gate). Scale `gross_cap` to configured values
(`1.0` BTC, `1.5` BNB). Set `PAIRWISE_FORCE_EXECUTE=1` if needed for manual override path.
Monitor `gross_leverage` against `gross_cap` bounds.  
**Gate criteria:**
- ≥ 2016 bars (7 d) at Stage B with positive return
- Market OS gate re-validated against recent 2-month window
- `promotion_gate.ready_for_live = True` on validated_stress_report with no `target_060_stress` failures
- Manual sign-off on reconciliation report

**Rollback criteria:** Shadow max_drawdown > 18%, or live drawdown > 8% (BNB kill-switch),
or reconciliation error rate > 5%. Rollback resets to Stage A; do not skip to flat without
reviewing signal gate configuration.

---

## Summary

The live system produces `session_type=flat` due to two simultaneous suppressions:

1. **BTC**: `regime_score = +0.019 > −0.02 threshold` → `short_ok=False` → baseline and
   specialist traces both zeroed at `compute_requested_weight` (line 558-563). Signal −197%
   never reaches blending.

2. **BNB**: Blend `mode=state_alphas` with only key `"equity_mixed:bull_broad"` but live route
   is permanently `"equity_mixed:bull_narrow"` → `resolve_blend_alpha` returns 0.0 → specialist
   weight −0.21 discarded, baseline 0.0 returned (line 126 of `btc_convex_blend.py`).

`PAIRWISE_FORCE_EXECUTE=0` does not contribute — the promotion gate already passes for demo mode.

**Recommended next step:** Apply Fix B (data-only: add `equity_mixed:bull_narrow: 0.2` to BNB
`state_alphas`) and separately investigate whether the BTC `specialist_pair_config` should carry
a lower `regime_threshold` to allow short execution in the current `equity_mixed` regime. Stage A
sizing (1% gross_cap) should be used to validate sign agreement before scaling.
