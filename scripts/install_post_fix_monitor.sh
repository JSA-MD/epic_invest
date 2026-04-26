#!/usr/bin/env bash
# install_post_fix_monitor.sh
# Copy launchd plist to ~/Library/LaunchAgents and optionally load it.
#
# Usage:
#   ./scripts/install_post_fix_monitor.sh           # copy only (safe default)
#   ./scripts/install_post_fix_monitor.sh load      # copy + launchctl load
#   ./scripts/install_post_fix_monitor.sh unload    # launchctl unload + remove
#
# NOTE: This script does NOT load the plist by default.
#       Pass "load" explicitly when you are ready to activate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

PLIST_SRC="$SCRIPT_DIR/com.epicinvest.post-fix-monitor.plist"
PLIST_DST="$LAUNCH_AGENTS_DIR/com.epicinvest.post-fix-monitor.plist"
LABEL="com.epicinvest.post-fix-monitor"

ACTION="${1:-copy}"

copy_plist() {
    mkdir -p "$LAUNCH_AGENTS_DIR"
    cp -f "$PLIST_SRC" "$PLIST_DST"
    chmod 644 "$PLIST_DST"
    echo "[ok] plist copied to $PLIST_DST"
}

load_agent() {
    copy_plist
    launchctl load "$PLIST_DST"
    echo "[ok] launchd agent loaded: $LABEL"
    echo "     fires at :15 every hour, self-gates to KST 09:xx"
    echo "     logs: /tmp/epic-invest-post-fix-monitor.log"
    echo "     state: /tmp/epic-invest-post-fix-monitor-last.json"
    echo ""
    echo "     To capture baseline now (no KST gate):"
    echo "       .venv/bin/python scripts/monitor_post_fix_evolution.py --force --baseline"
}

unload_agent() {
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "[ok] launchd agent unloaded and plist removed"
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
        echo "  $SCRIPT_DIR/install_post_fix_monitor.sh load"
        ;;
    *)
        echo "Usage: $0 [copy|load|unload]"
        exit 1
        ;;
esac
