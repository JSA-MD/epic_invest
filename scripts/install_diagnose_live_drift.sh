#!/usr/bin/env bash
# install_diagnose_live_drift.sh
# Copy the diagnose-live-drift launchd plist to ~/Library/LaunchAgents
# and optionally load/unload it.
#
# Usage:
#   ./scripts/install_diagnose_live_drift.sh           # copy only (safe default)
#   ./scripts/install_diagnose_live_drift.sh load      # copy + launchctl load
#   ./scripts/install_diagnose_live_drift.sh unload    # launchctl unload + remove
#
# NOTE: This script does NOT load the plist by default.
#       Pass "load" explicitly when you are ready to activate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
LABEL="com.epicinvest.diagnose-live-drift"

PLIST_SRC="$SCRIPT_DIR/${LABEL}.plist"
PLIST_DST="$LAUNCH_AGENTS_DIR/${LABEL}.plist"

ACTION="${1:-copy}"

copy_plist() {
    mkdir -p "$LAUNCH_AGENTS_DIR"
    cp -f "$PLIST_SRC" "$PLIST_DST"
    echo "[ok] plist copied to $PLIST_DST"
}

load_agent() {
    copy_plist
    launchctl load "$PLIST_DST"
    echo "[ok] launchd agent loaded: $LABEL"
    echo "     schedule: daily 09:30 KST (hourly :30 fire + KST self-gate in entry.sh)"
    echo "     logs:     /Users/jsa/work/epic_invest/logs/diagnose_drift.log"
}

unload_agent() {
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "[ok] launchd agent unloaded and plist removed: $LABEL"
}

case "$ACTION" in
    load)
        load_agent
        ;;
    unload)
        unload_agent
        ;;
    copy)
        copy_plist
        echo ""
        echo "Plist copied but NOT loaded. Run with 'load' to activate:"
        echo "  $SCRIPT_DIR/install_diagnose_live_drift.sh load"
        echo ""
        echo "To run immediately (manual):"
        echo "  bash $SCRIPT_DIR/diagnose_live_drift_launchd_entry.sh"
        ;;
    *)
        echo "Usage: $0 [copy|load|unload]"
        exit 1
        ;;
esac
