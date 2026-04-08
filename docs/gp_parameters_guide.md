# GP Crypto Trading Strategy - Parameter Guide

## 1. GP Evolution Parameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `POP_SIZE` | 2,000 | Population size per generation | Search breadth. Larger = more strategies explored, slower. Paper uses 7,500 |
| `N_GEN` | 15 | Number of generations | Convergence depth. More = deeper optimization, overfitting risk |
| `P_CX` | 0.90 | Crossover probability | Chance of combining two parent trees. Higher = more exploration |
| `P_MUT` | 0.15 | Mutation probability | Chance of random tree modification. Higher = escape local optima, too high = instability |
| `MAX_DEPTH` | 8 | Max tree depth | Strategy complexity cap. Deeper = more complex conditions, overfitting risk |
| `MAX_LEN` | 60 | Max tree nodes | Limits tree size with MAX_DEPTH. Larger = more complex strategies |

### Speed vs Quality Presets

| Preset | POP_SIZE | N_GEN | Time Est. | Use Case |
|--------|----------|-------|-----------|----------|
| Quick Test | 500 | 5 | ~1 min | Fast validation |
| Balanced | 2,000 | 15 | ~10 min | Default, reasonable results |
| Deep Search | 5,000 | 25 | ~1 hour | Better strategies |
| Paper-grade | 7,500 | 30 | ~2-3 hours | Maximum quality |

## 2. Data Parameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `PAIRS` | BTC,ETH,SOL,XRP | Input coins for GP | More pairs = more input dimensions (4 OHLC per pair) |
| `PRIMARY_PAIR` | BTCUSDT | Traded instrument | PnL calculated on this coin's price |
| `TIMEFRAME` | 1h | Candle interval | 5m=precise but slow, 1h=balanced, 4h=less noise |
| `TRAIN period` | 2022-07 ~ 2024-01 | Training period (18mo) | GP optimizes on this. Include bull+bear for robustness |
| `VAL period` | 2024-01 ~ 2024-07 | Validation period (6mo) | Overfitting prevention. Min 3 months recommended |
| `TEST period` | 2024-07 ~ 2025-03 | Test period (8mo) | Unseen data for real performance evaluation |

## 3. Backtest / Trading Parameters

| Parameter | Default | Description | Impact |
|-----------|---------|-------------|--------|
| `INITIAL_CASH` | $100,000 | Starting capital | Affects absolute PnL, not returns |
| `COMMISSION_PCT` | 0.0004 (0.04%) | Fee per side | Higher = penalizes frequent trading, GP converges to longer holds |
| `NO_TRADE_BAND` | 10 | Signal change threshold (pp) | Higher = fewer trades (less cost, less opportunity) |

### Commission Scenarios

| Setting | Value | Scenario |
|---------|-------|----------|
| 0.0002 | Maker fee (Binance VIP0) | Optimistic |
| 0.0004 | Taker fee (current default) | Conservative |
| 0.0006 | Taker + slippage | Very conservative |

## 4. GP Primitives (Building Blocks)

| Function | Inputs | Role |
|----------|--------|------|
| add, sub, mul | 2 | Price relationship calculation |
| pdiv | 2 | Protected division (safe from div-by-zero) |
| sin, cos, tanh | 1 | Periodic pattern detection, nonlinear transform |
| gt | 2 | Comparison: +100 or -100 (direction decision) |
| neg, abs | 1 | Sign flip, absolute value |
| rand | 0 | Random constant (-1 to 1) |

## 5. Robustness-focused Settings

For strategies that are more likely to work in live trading:

```python
POP_SIZE = 3000
N_GEN = 20
MAX_DEPTH = 6      # simpler trees
MAX_LEN = 40       # smaller trees
COMMISSION_PCT = 0.0006  # include slippage
NO_TRADE_BAND = 15       # reduce trading frequency
```

Simpler trees + higher costs = strategies that survive real market conditions.
