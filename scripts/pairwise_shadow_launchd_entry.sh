#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/pairwise_regime_live.py"
POLL_SECONDS="${PAIRWISE_SHADOW_POLL_SECONDS:-300}"
STATE_PATH="${PAIRWISE_SHADOW_STATE_PATH:-$ROOT_DIR/models/pairwise_regime_shadow_state.json}"
DECISION_LOG_FILE="${PAIRWISE_SHADOW_DECISION_LOG_FILE:-$ROOT_DIR/logs/pairwise_regime_shadow_decisions.jsonl}"
SUMMARY_PATH="${PAIRWISE_SHADOW_SUMMARY_PATH:-$ROOT_DIR/models/gp_regime_mixture_btc_bnb_pairwise_repair_equity_corr_validated_summary.json}"
MODEL_PATH="${PAIRWISE_SHADOW_MODEL_PATH:-$ROOT_DIR/models/recent_6m_gp_vectorized_big_capped_rerun.dill}"

exec /bin/bash -lc "cd \"$ROOT_DIR\" && \"$PYTHON_BIN\" -u \"$ENTRY_SCRIPT\" shadow-loop \
  --summary-path \"$SUMMARY_PATH\" \
  --model-path \"$MODEL_PATH\" \
  --state-path \"$STATE_PATH\" \
  --decision-log-path \"$DECISION_LOG_FILE\" \
  --poll-seconds \"$POLL_SECONDS\""
