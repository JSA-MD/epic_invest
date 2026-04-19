#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/lob_market_data.py"
RUNTIME_ENV_PATH="${LOB_LAUNCHD_ENV_PATH:-$ROOT_DIR/models/lob_collection_launchd_env.sh}"

if [[ -f "$RUNTIME_ENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_PATH"
fi

INTERVAL_SECONDS="${LOB_COLLECTION_INTERVAL_SECONDS:-60}"
DEPTH_LIMIT="${LOB_COLLECTION_DEPTH_LIMIT:-20}"
AGG_LIMIT="${LOB_COLLECTION_AGG_LIMIT:-200}"
SYMBOLS_RAW="${LOB_COLLECTION_SYMBOLS:-BTCUSDT,BNBUSDT}"

IFS=',' read -r -a symbols <<< "$SYMBOLS_RAW"

cmd=(
  "$PYTHON_BIN"
  -u
  "$ENTRY_SCRIPT"
  loop
  --symbols
)
cmd+=("${symbols[@]}")
cmd+=(
  --depth-limit
  "$DEPTH_LIMIT"
  --agg-limit
  "$AGG_LIMIT"
  --interval-seconds
  "$INTERVAL_SECONDS"
)

cd "$ROOT_DIR"
exec "${cmd[@]}"
