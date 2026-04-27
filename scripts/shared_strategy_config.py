"""Shared strategy constants used by both backtest (gp_crypto_evolution.py)
and live runtime (pairwise_regime_live.py).

Single source of truth — no duplicated constants in either file. All values
are pure data; runtime overrides come through env vars resolved at the call
sites using _safe_env_float / _safe_env_int helpers in each consumer.

Per user mandate 2026-04-27: 백테스트와 실매매는 모든 로직과 사항을 똑같이
맞추고 하드코딩도 하지 말것 설정값도 공유하고 백테스트가 그대로 실매매로
작동하게 변경.
"""
from __future__ import annotations

# Bar / timeframe
BAR_SECONDS: int = 300                      # 5-minute bars (default poll/refresh)
BAR_MINUTES: float = BAR_SECONDS / 60.0     # 5.0 — used in annualisation formulas

# Cost model — exchange round-trip fee rate applied per fill side.
# Backtest and shadow tracker MUST share this value; otherwise PnL
# attribution between them carries a mechanical bias.
FEE_RATE: float = 0.0004

# Sizing defaults
INITIAL_CASH_USD: float = 100_000.0
MIN_NOTIONAL_USD: float = 25.0          # micro-rebalance dead-zone
MAX_HOLD_BARS: int = 288                # 24h auto-flatten ceiling

# Calibration reference for tail-risk scaling
BACKTEST_GROSS_CAP: float = 0.75
