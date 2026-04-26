#!/usr/bin/env bash
# install_telegram_digests.sh
# Copy launchd plists to ~/Library/LaunchAgents and optionally load them.
#
# Usage:
#   ./scripts/install_telegram_digests.sh           # copy only (safe default)
#   ./scripts/install_telegram_digests.sh load      # copy + launchctl load
#   ./scripts/install_telegram_digests.sh unload    # launchctl unload + remove
#
# NOTE: This script does NOT load plists by default.
#       Pass "load" explicitly when you are ready to activate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

DAILY_PLIST_SRC="$SCRIPT_DIR/com.epicinvest.telegram-daily-digest.plist"
HOURLY_PLIST_SRC="$SCRIPT_DIR/com.epicinvest.telegram-hourly-badge.plist"

DAILY_PLIST_DST="$LAUNCH_AGENTS_DIR/com.epicinvest.telegram-daily-digest.plist"
HOURLY_PLIST_DST="$LAUNCH_AGENTS_DIR/com.epicinvest.telegram-hourly-badge.plist"

DAILY_LABEL="com.epicinvest.telegram-daily-digest"
HOURLY_LABEL="com.epicinvest.telegram-hourly-badge"

ACTION="${1:-copy}"

copy_plists() {
    mkdir -p "$LAUNCH_AGENTS_DIR"
    cp -f "$DAILY_PLIST_SRC"  "$DAILY_PLIST_DST"
    cp -f "$HOURLY_PLIST_SRC" "$HOURLY_PLIST_DST"
    echo "[ok] plists copied to $LAUNCH_AGENTS_DIR"
}

load_agents() {
    copy_plists
    launchctl load "$DAILY_PLIST_DST"
    launchctl load "$HOURLY_PLIST_DST"
    echo "[ok] launchd agents loaded"
    echo "     daily  -> 09:00 KST (00:00 UTC)"
    echo "     hourly -> 09:00-21:00 KST (00:00-12:00 UTC), silent, skip-if-no-change"
}

unload_agents() {
    launchctl unload "$DAILY_PLIST_DST"  2>/dev/null || true
    launchctl unload "$HOURLY_PLIST_DST" 2>/dev/null || true
    rm -f "$DAILY_PLIST_DST" "$HOURLY_PLIST_DST"
    echo "[ok] launchd agents unloaded and plists removed"
}

case "$ACTION" in
    load)
        load_agents
        ;;
    unload)
        unload_agents
        ;;
    copy)
        copy_plists
        echo ""
        echo "Plists copied but NOT loaded. Run with 'load' to activate:"
        echo "  $SCRIPT_DIR/install_telegram_digests.sh load"
        ;;
    *)
        echo "Usage: $0 [copy|load|unload]"
        exit 1
        ;;
esac
