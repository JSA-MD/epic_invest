#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Host-TZ-independent self-gate WITH catch-up. launchd fires every hour; this
# script must (a) only run each stage once per KST day and (b) still run after
# the target hour even if the system was asleep at 09:30 KST.
#
# State is split into two stages so a Telegram notify failure does NOT mark
# diagnostic as "done": the next hour will re-run notify only (cheap),
# without regenerating the diagnostic report.
#
#   {"diagnostic_kst_date": "YYYY-MM-DD", "notify_kst_date": "YYYY-MM-DD"}
#
_STATE_FILE="/tmp/epic-invest-diagnose-live-drift-last.json"
_KST_TODAY=$(TZ=Asia/Seoul date +%Y-%m-%d)
_KST_HOUR=$(TZ=Asia/Seoul date +%H)

# Force base-10: bash treats leading-zero numerics (08, 09) as octal which is
# invalid and errors out, making the [[ -lt ]] gate silently fall through and
# run the diagnostic at 08:xx KST.
if (( 10#$_KST_HOUR < 9 )); then
    exit 0
fi

# --- Read existing state (best-effort) ---
_diag_done=""
_notify_done=""
if [[ -f "$_STATE_FILE" ]]; then
    _diag_done=$(python3 -c "import json; d=json.load(open('$_STATE_FILE')); print(d.get('diagnostic_kst_date',''))" 2>/dev/null || echo "")
    _notify_done=$(python3 -c "import json; d=json.load(open('$_STATE_FILE')); print(d.get('notify_kst_date',''))" 2>/dev/null || echo "")
fi

# Both stages already done today → skip.
if [[ "$_diag_done" == "$_KST_TODAY" && "$_notify_done" == "$_KST_TODAY" ]]; then
    exit 0
fi

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENTRY_SCRIPT="$ROOT_DIR/scripts/diagnose_live_drift.py"
NOTIFY_SCRIPT="$ROOT_DIR/scripts/diagnose_live_drift_telegram_notify.py"
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

OUT_DIR="$ROOT_DIR/models"
DOCS_DIR="$ROOT_DIR/docs"
mkdir -p "$OUT_DIR" "$DOCS_DIR"

TS=$(date -u +%Y%m%d)
JSON_OUT="$OUT_DIR/live_drift_diagnostic_${TS}.json"
MD_OUT="$DOCS_DIR/live_drift_diagnostic_${TS}.md"

cd "$ROOT_DIR"

# Helper: merge a single key into the state JSON, atomically (best-effort).
_write_state() {
    local key="$1"
    local value="$2"
    python3 - "$_STATE_FILE" "$key" "$value" <<'PY' || true
import json, os, sys, tempfile
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
data = {}
try:
    with open(path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    pass
data[key] = value
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".state-", suffix=".json")
with os.fdopen(fd, "w") as f:
    json.dump(data, f)
os.replace(tmp, path)
PY
}

# --- Stage 1: diagnostic (skip if already done today) ---
diag_ok=0
if [[ "$_diag_done" == "$_KST_TODAY" ]]; then
    diag_ok=1
elif "$PYTHON_BIN" -u "$ENTRY_SCRIPT" --json "$JSON_OUT" --out "$MD_OUT"; then
    diag_ok=1
    _write_state "diagnostic_kst_date" "$_KST_TODAY"
fi

# --- Stage 2: notify (only if diagnostic ok and notify not yet done today) ---
if [[ "$diag_ok" == "1" && "$_notify_done" != "$_KST_TODAY" && -f "$NOTIFY_SCRIPT" ]]; then
    if "$PYTHON_BIN" -u "$NOTIFY_SCRIPT" --report "$JSON_OUT"; then
        _write_state "notify_kst_date" "$_KST_TODAY"
    else
        # Notify failed (network, schema, missing token). Leave notify_kst_date
        # unset so the next hour retries — diagnostic will be skipped via state.
        echo "[diagnose-live-drift] notify failed (exit $?); will retry next hour" >&2
    fi
fi
