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
- trader decision journal: `logs/rotation_target_050_decisions.jsonl`

## Telegram Commands

### Read

- `/help`
- `/start`
- `/ping`
- `/status`
- `/plan`
- `/positions`
- `/why`
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
- `/why` explains the current position thesis and the conditions that would close or reduce it.
- The trader keeps the latest thesis in `rotation_target_050_live_state.json` and appends decision changes to `logs/rotation_target_050_decisions.jsonl`.

## 자동 알림

텔레그램 봇은 사용자가 먼저 말을 걸지 않아도 중요한 이벤트가 발생하면 자동으로 메시지를 보냅니다.

### 프로그램 생명주기

- 트레이더 시작 완료
- 트레이더 종료 완료
- 트레이더 재시작 완료

### 포지션/주문 변화

- 코어 리밸런싱으로 신규 진입
- 코어 리밸런싱으로 전량 청산
- 코어 포지션 증액/감축/전환
- 오버레이 진입
- `closeall` / `flatten` 으로 전체 포지션 정리

### 손절/익절/리스크

- 오버레이 익절
- 오버레이 손절
- 오버레이 트레일링 손절
- 코어 킬 스위치 발동
- 종료 보호주문 설치

### 일일 상태 변화

- 일일 시작 브리핑
  - 적용일
  - 세션(`core` / `overlay` / `flat`)
  - 자산
  - 레버리지
  - 코어 비중 또는 오버레이 방향
- 일일 마감 요약
  - 시작 자산
  - 종료 자산
  - 손익
  - 수익률
- 같은 날 세션이 바뀌면 세션 변경 알림

### 오류/예외

- 트레이더 루프 예외 발생 시 오류 알림

## 관련 환경 변수

```dotenv
TELEGRAM_NOTIFICATIONS_ENABLED=1
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_CHAT_IDS=123456789
```

- `TELEGRAM_NOTIFICATIONS_ENABLED=0` 이면 자동 알림을 끌 수 있습니다.
- `TELEGRAM_ALLOWED_CHAT_IDS` 가 없으면 `TELEGRAM_CHAT_ID` 도 보조로 사용합니다.
