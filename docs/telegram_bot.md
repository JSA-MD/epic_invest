# Telegram Bot Control

This project can now be controlled through Telegram without a UI.

## Required Environment

Add these values to `.env`:

```dotenv
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_CHAT_IDS=123456789
```

`TELEGRAM_ALLOWED_CHAT_IDS` is a comma-separated allowlist. Only those chats can control or read the bot.

## Bot Service

Start the Telegram bot:

```bash
./telegram_start.sh
```

Stop it:

```bash
./telegram_stop.sh
```

Restart it:

```bash
./telegram_restart.sh
```

Default runtime files:

- bot log: `/tmp/epic-invest-telegram-bot.log`
- bot pid: `/tmp/epic-invest-telegram-bot.pid`
- bot state: `/tmp/epic-invest-telegram-bot-state.json`
- bot audit: `/tmp/epic-invest-telegram-audit.jsonl`

## Telegram Commands

### Read

- `/help`
- `/start`
- `/ping`
- `/status`
- `/plan`
- `/positions`
- `/protection`
- `/killswitch`
- `/logs`
- `/recent`

### Control

- `/starttrader`
- `/stoptrader`
- `/restarttrader`
- `/sync`
- `/protect`
- `/closeall`
- `/flatten`

### Safety

Every control command requires confirmation:

```text
/closeall
/confirm abc123
```

Cancel pending control commands:

```text
/cancel
```

## Behavior Notes

- `/closeall` stops the trader first if it is running, then cancels managed fallback protections and closes all open positions.
- `/flatten` is an alias of `/closeall`.
- `close_pair_position()` uses `reduceOnly` market orders for flattening.
- `/logs` reads `/tmp/epic-invest-trader.log`.
- `/recent` reads the bot audit file, not the trader log.
