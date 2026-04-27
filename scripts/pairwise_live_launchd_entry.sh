#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/pairwise_regime_live.py"
RUNTIME_ENV_PATH="${PAIRWISE_LAUNCHD_ENV_PATH:-$ROOT_DIR/models/pairwise_live_launchd_env.sh}"

# Capture plist-injected values BEFORE sourcing the env file so the env file
# cannot override risk-critical variables set by the plist EnvironmentVariables.
# Uses ${VAR-} (dash, not colon-dash) to tolerate set -u when var is unset.
_PLIST_PAIRWISE_GROSS_CAP="${PAIRWISE_GROSS_CAP-}"
_PLIST_PAIRWISE_LIVE_MAX_GROSS_CAP="${PAIRWISE_LIVE_MAX_GROSS_CAP-}"
_PLIST_PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP="${PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP-}"
_PLIST_PAIRWISE_NO_TRADE_BAND_PCT="${PAIRWISE_NO_TRADE_BAND_PCT-}"
_PLIST_REBALANCE_NOTIONAL_BAND_USD="${REBALANCE_NOTIONAL_BAND_USD-}"
_PLIST_PAIRWISE_CVAR_CUT="${PAIRWISE_CVAR_CUT-}"
_PLIST_PAIRWISE_CVAR_SCALE_BY_GROSS_CAP="${PAIRWISE_CVAR_SCALE_BY_GROSS_CAP-}"
_PLIST_PAIRWISE_BACKTEST_GROSS_CAP="${PAIRWISE_BACKTEST_GROSS_CAP-}"
_PLIST_PAIRWISE_PROMOTION_FREEZE="${PAIRWISE_PROMOTION_FREEZE-}"
_PLIST_PAIRWISE_EQUITY_CORR_RISK="${PAIRWISE_EQUITY_CORR_RISK-}"
_PLIST_PAIRWISE_REGIME_GATE_DISABLED="${PAIRWISE_REGIME_GATE_DISABLED-}"
_PLIST_PAIRWISE_FORCE_EXECUTE="${PAIRWISE_FORCE_EXECUTE-}"
_PLIST_PAIRWISE_SAFETY_DEFAULT_GROSS_CAP="${PAIRWISE_SAFETY_DEFAULT_GROSS_CAP-}"

if [[ -f "$RUNTIME_ENV_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_PATH"
fi

# Restore plist-injected values so the env file cannot override them.
# Only restores vars that the plist actually set (non-empty shadow).
if [[ -n "$_PLIST_PAIRWISE_GROSS_CAP" ]]; then
  export PAIRWISE_GROSS_CAP="$_PLIST_PAIRWISE_GROSS_CAP"
  echo "[entry] plist override restored: PAIRWISE_GROSS_CAP=$PAIRWISE_GROSS_CAP" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_LIVE_MAX_GROSS_CAP" ]]; then
  export PAIRWISE_LIVE_MAX_GROSS_CAP="$_PLIST_PAIRWISE_LIVE_MAX_GROSS_CAP"
  echo "[entry] plist override restored: PAIRWISE_LIVE_MAX_GROSS_CAP=$PAIRWISE_LIVE_MAX_GROSS_CAP" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP" ]]; then
  export PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP="$_PLIST_PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP"
  echo "[entry] plist override restored: PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP=$PAIRWISE_ALLOW_BACKTEST_LIKE_GROSS_CAP" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_NO_TRADE_BAND_PCT" ]]; then
  export PAIRWISE_NO_TRADE_BAND_PCT="$_PLIST_PAIRWISE_NO_TRADE_BAND_PCT"
  echo "[entry] plist override restored: PAIRWISE_NO_TRADE_BAND_PCT=$PAIRWISE_NO_TRADE_BAND_PCT" >&2
fi
if [[ -n "$_PLIST_REBALANCE_NOTIONAL_BAND_USD" ]]; then
  export REBALANCE_NOTIONAL_BAND_USD="$_PLIST_REBALANCE_NOTIONAL_BAND_USD"
  echo "[entry] plist override restored: REBALANCE_NOTIONAL_BAND_USD=$REBALANCE_NOTIONAL_BAND_USD" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_CVAR_CUT" ]]; then
  export PAIRWISE_CVAR_CUT="$_PLIST_PAIRWISE_CVAR_CUT"
  echo "[entry] plist override restored: PAIRWISE_CVAR_CUT=$PAIRWISE_CVAR_CUT" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_CVAR_SCALE_BY_GROSS_CAP" ]]; then
  export PAIRWISE_CVAR_SCALE_BY_GROSS_CAP="$_PLIST_PAIRWISE_CVAR_SCALE_BY_GROSS_CAP"
  echo "[entry] plist override restored: PAIRWISE_CVAR_SCALE_BY_GROSS_CAP=$PAIRWISE_CVAR_SCALE_BY_GROSS_CAP" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_BACKTEST_GROSS_CAP" ]]; then
  export PAIRWISE_BACKTEST_GROSS_CAP="$_PLIST_PAIRWISE_BACKTEST_GROSS_CAP"
  echo "[entry] plist override restored: PAIRWISE_BACKTEST_GROSS_CAP=$PAIRWISE_BACKTEST_GROSS_CAP" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_PROMOTION_FREEZE" ]]; then
  export PAIRWISE_PROMOTION_FREEZE="$_PLIST_PAIRWISE_PROMOTION_FREEZE"
  echo "[entry] plist override restored: PAIRWISE_PROMOTION_FREEZE=$PAIRWISE_PROMOTION_FREEZE" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_EQUITY_CORR_RISK" ]]; then
  export PAIRWISE_EQUITY_CORR_RISK="$_PLIST_PAIRWISE_EQUITY_CORR_RISK"
  echo "[entry] plist override restored: PAIRWISE_EQUITY_CORR_RISK=$PAIRWISE_EQUITY_CORR_RISK" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_REGIME_GATE_DISABLED" ]]; then
  export PAIRWISE_REGIME_GATE_DISABLED="$_PLIST_PAIRWISE_REGIME_GATE_DISABLED"
  echo "[entry] plist override restored: PAIRWISE_REGIME_GATE_DISABLED=$PAIRWISE_REGIME_GATE_DISABLED" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_FORCE_EXECUTE" ]]; then
  export PAIRWISE_FORCE_EXECUTE="$_PLIST_PAIRWISE_FORCE_EXECUTE"
  echo "[entry] plist override restored: PAIRWISE_FORCE_EXECUTE=$PAIRWISE_FORCE_EXECUTE" >&2
fi
if [[ -n "$_PLIST_PAIRWISE_SAFETY_DEFAULT_GROSS_CAP" ]]; then
  export PAIRWISE_SAFETY_DEFAULT_GROSS_CAP="$_PLIST_PAIRWISE_SAFETY_DEFAULT_GROSS_CAP"
  echo "[entry] plist override restored: PAIRWISE_SAFETY_DEFAULT_GROSS_CAP=$PAIRWISE_SAFETY_DEFAULT_GROSS_CAP" >&2
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
