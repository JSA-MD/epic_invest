import os
import time
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import ccxt
import dill
import numpy as np
import pandas as pd

from gp_crypto_evolution import (
    PAIRS, ARG_NAMES, TIMEFRAME, MODELS_DIR, fetch_klines, toolbox
)

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_SECRET", "")
IS_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"

def get_exchange():
    exchange = ccxt.binanceusdm({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })
    
    if IS_TESTNET:
        testnet_url = 'https://testnet.binancefuture.com'
        exchange.urls['api'] = {
            'public': f'{testnet_url}/fapi/v1',
            'private': f'{testnet_url}/fapi/v1',
            'fapiPublic': f'{testnet_url}/fapi/v1',
            'fapiPrivate': f'{testnet_url}/fapi/v1',
            'fapiPrivateV2': f'{testnet_url}/fapi/v2',
        }
    return exchange

def execute_multicoin_trades():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Waking up for {TIMEFRAME} Multi-Coin Scalping...")
    
    # Load Model
    model_path = MODELS_DIR / "best_crypto_gp.dill"
    try:
        with open(model_path, "rb") as f:
            model = dill.load(f)
        compiled_func = toolbox.compile(expr=model)
    except Exception as e:
        print(f"[ERROR] Failed to load Multi-Coin GP Model: {e}")
        return

    # Fetch Latest Data
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=24)
    
    dfs = []
    for pair in PAIRS:
        df = fetch_klines(pair, TIMEFRAME, start_dt, end_dt)
        df.columns = [f"{pair}_{c}" for c in df.columns]
        dfs.append(df)
        
    combined = pd.concat(dfs, axis=1).dropna()
    if combined.empty:
        print("No market data available.")
        return
        
    latest = combined.iloc[-1]
    
    exchange = get_exchange()

    try:
        balance = exchange.fetch_balance()
        total_usdt = float(balance['USDT']['total'])
    except Exception as e:
        print(f"Failed to fetch balance: {e}")
        return
        
    # Allocate 100% of the account total equity evenly across the 4 pairs
    allocated_usd_per_pair = total_usdt / len(PAIRS)
    print(f"\n[ACCOUNT] Total Balance: ${total_usdt:,.2f} | Allocated per pair: ${allocated_usd_per_pair:,.2f}")

    print(f"------------ MULTI-COIN SIGNALS ------------")
    for pair in PAIRS:
        # Extract features for this specific pair
        try:
            inputs = [float(latest[f"{pair}_{c}"]) for c in ARG_NAMES]
            signal_raw = float(compiled_func(*inputs))
            signal_pct = np.clip(np.where(np.isfinite(signal_raw), signal_raw, 0.0), -500.0, 500.0)
        except Exception as e:
            continue
            
        coin_price = float(latest[f"{pair}_close"])
        direction = "LONG" if signal_pct > 0 else "SHORT" if signal_pct < 0 else "FLAT"
        
        leverage = signal_pct / 100.0
        target_pos_usd = allocated_usd_per_pair * leverage
        target_qty = target_pos_usd / coin_price
        
        print(f"[{pair}] Price: ${coin_price:,.4f} | Signal: {signal_pct:+.2f}% | Target Notion: ${target_pos_usd:+.2f}")
        
        # CCXT Execution
        if not API_KEY: continue
            
        try:
            # Try setting leverage
            try:
                exchange.fapiprivate_post_leverage({
                    'symbol': pair,
                    'leverage': 5
                })
            except: pass
            
            # Get current position
            if not hasattr(exchange, '_positions_cache'):
                exchange._positions_cache = exchange.fetch_positions([p for p in PAIRS])
                
            current_qty = 0.0
            for pos in exchange._positions_cache:
                if pair in pos['symbol']:
                    current_qty = float(pos['info'].get('positionAmt', 0))
                    break
                    
            diff_qty = target_qty - current_qty
            
            # Simple dead band to avoid microscopic updates
            amount_precision = exchange.markets[pair]['precision']['amount'] if pair in exchange.markets else 0.001
            if abs(diff_qty) < amount_precision:
                continue

            side = 'buy' if diff_qty > 0 else 'sell'
            order = exchange.create_market_order(pair, side, abs(diff_qty))
            print(f"  -> Executed {side.upper()} {abs(diff_qty):.5f} {pair} (Order: {order['id']})")
        except Exception as e:
            print(f"  -> Execution Failed for {pair}: {e}")

def wait_for_next_5m_sync():
    now = datetime.now()
    minutes_to_next = 5 - (now.minute % 5)
    
    if minutes_to_next == 5 and now.second < 2:
        minutes_to_next = 0
        
    target_time = now.replace(minute=(now.minute + minutes_to_next) % 60, second=2, microsecond=0)
    
    if target_time < now:
        target_time = target_time.replace(hour=(target_time.hour + 1) % 24)

    sleep_seconds = (target_time - now).total_seconds()
    print(f"\n[SYNC] Sleeping for {sleep_seconds:.1f}s until next 5m trigger: {target_time.strftime('%H:%M:%S')}")
    time.sleep(sleep_seconds)

def main():
    print("=========================================")
    print(" 🚀 UNIVERSAL MULTI-COIN SCALPER LIVE")
    print("=========================================")
    print(f"Pairs Traded   : {PAIRS}")
    print(f"Target Leverge : Max 5x per coin")
    print(f"Safe Mode Seed : ALL 100% ACCOUNT MAXIMIZED")
    print(f"Testnet Mode   : {IS_TESTNET}")
    print("-----------------------------------------")

    while True:
        wait_for_next_5m_sync()
        execute_multicoin_trades()

if __name__ == "__main__":
    main()
