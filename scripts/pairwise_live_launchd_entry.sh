#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/pairwise_regime_live.py"
RUNTIME_ENV_PATH="${PAIRWISE_LAUNCHD_ENV_PATH:-$ROOT_DIR/models/pairwise_live_launchd_env.sh}"

if [[ -f "$RUNTIME_ENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_PATH"
fi

# Bar-close alignment is handled inside the loop subcommand of pairwise_regime_live.py.
# The plist uses KeepAlive=true for process resurrection only; no StartCalendarInterval needed.
MODE="${PAIRWISE_LIVE_MODE:-demo}"
POLL_SECONDS="${PAIRWISE_LIVE_POLL_SECONDS:-300}"
PROMOTION_REPORT_PATH="${PAIRWISE_LIVE_PROMOTION_REPORT_PATH:-$ROOT_DIR/models/gp_regime_mixture_btc_bnb_pairwise_market_os_pipeline_report.json}"
STATE_PATH="${PAIRWISE_LIVE_STATE_PATH:-$ROOT_DIR/models/pairwise_regime_live_state.json}"
DECISION_LOG_PATH="${PAIRWISE_LIVE_DECISION_LOG_PATH:-$ROOT_DIR/logs/pairwise_regime_decisions.jsonl}"
FORCE_EXECUTE="${PAIRWISE_FORCE_EXECUTE:-0}"
FORCE_NOTE="${PAIRWISE_FORCE_NOTE:-manual_primary_switch}"
REFRESH_LIVE_DATA="${PAIRWISE_REFRESH_LIVE_DATA:-0}"

cmd=(
  "$PYTHON_BIN"
  -u
  "$ENTRY_SCRIPT"
  loop
  --execute
  --mode
  "$MODE"
  --poll-seconds
  "$POLL_SECONDS"
  --promotion-report
  "$PROMOTION_REPORT_PATH"
  --state-path
  "$STATE_PATH"
  --decision-log-path
  "$DECISION_LOG_PATH"
)

case "$(printf '%s' "$REFRESH_LIVE_DATA" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    cmd+=(--refresh-live-data)
    ;;
  *)
    cmd+=(--no-refresh-live-data)
    ;;
esac

case "$(printf '%s' "$FORCE_EXECUTE" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    cmd+=(--force-execute --force-note "$FORCE_NOTE")
    ;;
esac

cd "$ROOT_DIR"
exec "${cmd[@]}"
