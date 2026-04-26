#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/position_reconciliation.py"
RUNTIME_ENV_PATH="${PAIRWISE_LAUNCHD_ENV_PATH:-$ROOT_DIR/models/pairwise_live_launchd_env.sh}"
ENV_FILE="$ROOT_DIR/.env"

if [[ -f "$RUNTIME_ENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_PATH"
fi

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "$ENV_FILE"
  set +a
fi

MODE="${PAIRWISE_LIVE_MODE:-demo}"
STATE_PATH="${PAIRWISE_LIVE_STATE_PATH:-$ROOT_DIR/models/pairwise_regime_live_state.json}"
TOLERANCE="${RECON_TOLERANCE:-0.001}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" -u "$ENTRY_SCRIPT" \
  --mode "$MODE" \
  --state-path "$STATE_PATH" \
  --tolerance "$TOLERANCE"
