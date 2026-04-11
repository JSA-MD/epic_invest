#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/pairwise_regime_mixture_shadow_live.py"
POLL_SECONDS="${PAIRWISE_SHADOW_POLL_SECONDS:-300}"
STATE_PATH="${PAIRWISE_SHADOW_STATE_PATH:-$ROOT_DIR/models/pairwise_regime_shadow_state.json}"
DECISION_LOG_FILE="${PAIRWISE_SHADOW_DECISION_LOG_FILE:-${PAIRWISE_SHADOW_DECISION_LOG_PATH:-$ROOT_DIR/logs/pairwise_regime_shadow_decisions.jsonl}}"
SUMMARY_PATH="${PAIRWISE_SHADOW_SUMMARY_PATH:-$ROOT_DIR/models/gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json}"
MODEL_PATH="${PAIRWISE_SHADOW_MODEL_PATH:-$ROOT_DIR/models/recent_6m_gp_vectorized_big_capped_rerun.dill}"
BASE_SUMMARY_PATH="${PAIRWISE_SHADOW_BASE_SUMMARY_PATH:-$ROOT_DIR/models/gp_regime_mixture_search_summary.json}"
PROMOTION_REPORT_PATH="${PAIRWISE_SHADOW_PROMOTION_REPORT_PATH:-$ROOT_DIR/models/gp_regime_mixture_btc_bnb_pairwise_market_os_pipeline_report.json}"
MODE="${PAIRWISE_SHADOW_MODE:-${BINANCE_MODE:-demo}}"

exec /bin/bash -lc "cd \"$ROOT_DIR\" && \"$PYTHON_BIN\" -u \"$ENTRY_SCRIPT\" loop \
  --summary \"$SUMMARY_PATH\" \
  --base-summary \"$BASE_SUMMARY_PATH\" \
  --model \"$MODEL_PATH\" \
  --promotion-report \"$PROMOTION_REPORT_PATH\" \
  --state-path \"$STATE_PATH\" \
  --decision-log \"$DECISION_LOG_FILE\" \
  --mode \"$MODE\" \
  --poll-seconds \"$POLL_SECONDS\""
