import asyncio
import websockets
import json
from datetime import datetime
import sys

async def monitor_liquidations():
    # 바이낸스 선물 모든 심볼의 강제 청산 스트림
    uri = "wss://fstream.binance.com/ws/!forceOrder@arr"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 바이낸스 선물 실시간 청산(Liquidation) 관측 레이더 가동...")
    print("-------------------------------------------------------------------------")
    print(f"{'시간':^12} | {'심볼':^10} | {'청산 포지션':^15} | {'피해액(USD 규모)':^15} | {'체결 가격'}")
    print("-------------------------------------------------------------------------")
    
    try:
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                # 데이터 파싱: 'o' 키 안쪽에 실제 청산 주문 상세 내역이 들어있음
                order_data = data.get('o', {})
                if not order_data:
                    continue
                    
                symbol = order_data.get('s')
                side = order_data.get('S') # SELL = 시장가 매도 발생 (기존 롱 포지션 강제청산) / BUY = 시장가 매수 발생 (기존 숏 포지션 강제청산)
                price = float(order_data.get('p', 0)) # 주문 가격 (보통 파산 가격 근처)
                qty = float(order_data.get('q', 0))   # 청산 수량
                
                usd_value = price * qty # 달러 가치로 환산
                
                # 테스트를 위해 모든 금액($0 이상)의 필터링 해제
                if usd_value >= 0:
                    time_str = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    
                    if side == "SELL":
                        position_type = "🩸 LONG 붕괴"
                        color = "\033[91m" # Red
                    else:
                        position_type = "🚀 SHORT 붕괴"
                        color = "\033[92m" # Green
                    reset = "\033[0m"
                    
                    print(f"{color}[{time_str}] | {symbol:<10} | {position_type:<14} | ${usd_value:>13,.2f} | {price:,.4f}{reset}")
                    sys.stdout.flush()
                    
    except asyncio.CancelledError:
        print("\n[관측 레이더] 스트리밍이 사용자에 의해 종료되었습니다.")
    except Exception as e:
        print(f"\n[관측 레이더] 연결 오류 또는 예외 발생: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(monitor_liquidations())
    except KeyboardInterrupt:
        pass
