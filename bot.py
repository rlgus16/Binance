import ccxt
import os
import time
import json
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Configuration
SYMBOL = 'LTC/USDT:USDT'
TIMEFRAME = '4h'
LEVERAGE = 5
MAX_LONG_SIZE_USDT = 2000
LOOP_INTERVAL_MINUTES = 30
DRY_RUN = False  # 실전 가동 (False)

class AutoTrader:
    def __init__(self):
        binance_api_key = os.getenv("BINANCE_API_KEY")
        binance_secret_key = os.getenv("BINANCE_SECRET_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not binance_api_key or not binance_secret_key:
            raise ValueError("Missing Binance API keys in .env")
        if not gemini_api_key:
            raise ValueError("Missing Gemini API key in .env")

        print("Initializing Binance USDS-M Futures...")
        self.exchange = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'positionMode': True
            }
        })

        self.client = genai.Client(api_key=gemini_api_key)
        self.setup_exchange()

    def setup_exchange(self):
        try:
            self.exchange.load_markets()
            self.exchange.set_position_mode(True)
            print("Hedge Mode (Dual Side Position) enabled/verified.")
            
            print(f"Setting leverage for {SYMBOL} to {LEVERAGE}x...")
            self.exchange.set_leverage(LEVERAGE, SYMBOL)
            
            print(f"Setting margin mode to CROSS for {SYMBOL}...")
            self.exchange.set_margin_mode('cross', SYMBOL)
        except Exception as e:
            print(f"Setup info/warning: {e}")

    def fetch_data(self):
        print(f"Fetching {TIMEFRAME} candles for {SYMBOL}...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            df.ta.macd(append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.bbands(length=20, append=True)
            
            df.bfill(inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def get_account_state(self):
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = float(balance.get('USDT', {}).get('free', 0.0))
            usdt_total = float(balance.get('USDT', {}).get('total', 0.0))
            
            positions = self.exchange.fetch_positions([SYMBOL])
            long_pos = next((p for p in positions if p.get('side') == 'long'), None)
            short_pos = next((p for p in positions if p.get('side') == 'short'), None)
            
            open_orders = self.exchange.fetch_open_orders(SYMBOL)
            
            return {
                'usdt_free': usdt_free,
                'usdt_total': usdt_total,
                'long_position': {
                    'notional': float(long_pos['info'].get('notional', 0)) if long_pos else 0.0,
                    'entryPrice': float(long_pos.get('entryPrice', 0)) if long_pos else 0.0,
                    'unrealizedPnl': float(long_pos.get('unrealizedPnl', 0)) if long_pos else 0.0,
                },
                'short_position': {
                    'notional': float(short_pos['info'].get('notional', 0)) if short_pos else 0.0,
                    'entryPrice': float(short_pos.get('entryPrice', 0)) if short_pos else 0.0,
                    'unrealizedPnl': float(short_pos.get('unrealizedPnl', 0)) if short_pos else 0.0,
                },
                'open_orders': [
                    {'id': o['id'], 'side': o['side'], 'type': o['type'], 'price': o['price'], 'amount': o['amount'], 'positionSide': o.get('info', {}).get('positionSide')}
                    for o in open_orders
                ]
            }
        except Exception as e:
            print(f"Error fetching account state: {e}")
            return None

    def get_gemini_signal(self, df, account_state):
        print("Analyzing data with Gemini 3.1 Pro...")
        recent_data = df.tail(10).to_dict(orient='records')
        
        system_instruction = f"""You are an advanced quantitative trading AI for Binance USD-M Futures.
You are trading {SYMBOL} on {TIMEFRAME} candles.

RULES AND CONSTRAINTS:
1. Cross Margin with {LEVERAGE}x Leverage.
2. Hedge Mode is ON (You can hold both LONG and SHORT positions simultaneously).
3. The MAXIMUM total LONG position size must NEVER exceed {MAX_LONG_SIZE_USDT} USDT (notional).
4. The TOTAL SHORT position size must NEVER exceed the CURRENT LONG position size at all times.
5. NEVER execute or place a STOP_LOSS order.
6. You can average down (place lower limits for long, higher limits for short) to maximize profit.
7. Do NOT hedge if free balance is abundant. Focus on maximizing profit.
8. Opens LONG and SHORT positions to maximize profit according to technical analysis.
9. Exits MUST rely on take_profit orders hitting their targets.
10. Predict and provide limit_order entry prices and take_profit prices.
11. Instead of HOLDING, place take_profit_orders to maximize profit.
12. If you are holding an existing LONG or SHORT position, you MUST provide a take profit price for it in the 'existing_position_tp' field.

Respond ONLY with a valid JSON format (without markdown code blocks) representing your trading decision.
Format:
{{
    "reasoning": "Explain your analysis...",
    "cancel_all_open_orders": true/false,
    "existing_position_tp": {{
        "LONG": <take profit limit for existing LONG position, or 0 if none>,
        "SHORT": <take profit limit for existing SHORT position, or 0 if none>
    }},
    "orders": [
        {{
            "side": "buy",
            "positionSide": "LONG",
            "type": "limit",
            "amount_usdt": <amount in USDT>,
            "price": <entry price limit>,
            "take_profit_price": <take profit target limit>
        }}
    ]
}}
"""
        prompt = f"""
Current Account State:
USDT Free: {account_state['usdt_free']}
USDT Total: {account_state['usdt_total']}
Long Position: Notional {account_state['long_position']['notional']} USDT at Avg Price {account_state['long_position']['entryPrice']}
Short Position: Notional {account_state['short_position']['notional']} USDT at Avg Price {account_state['short_position']['entryPrice']}
Open Orders count: {len(account_state['open_orders'])}
Last prices from 4H Candles:
{recent_data}

Based on this, what are your next orders?
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                )
            )
            return response.text
        except Exception as e:
            print(f"Error calling Gemini: {e}")
            return None

    def execute_orders(self, signal_text):
        if not signal_text: return

        try:
            signal_text = signal_text.replace("```json\n", "").replace("```\n", "").replace("```", "")
            decision = json.loads(signal_text)
            print(f"AI Reasoning: {decision.get('reasoning')}")

            if decision.get('cancel_all_open_orders'):
                print("Canceling all open orders as instructed by AI...")
                if not DRY_RUN: 
                    self.exchange.cancel_all_orders(SYMBOL)
                    time.sleep(1) # API 동기화 대기
                    
                    positions = self.exchange.fetch_positions([SYMBOL])
                    long_pos = next((p for p in positions if p.get('side') == 'long'), None)
                    short_pos = next((p for p in positions if p.get('side') == 'short'), None)
                    
                    long_contracts = float(long_pos.get('contracts', 0)) if long_pos else 0.0
                    short_contracts = float(short_pos.get('contracts', 0)) if short_pos else 0.0
                    
                    existing_tp = decision.get('existing_position_tp') or {}
                    l_tp = float(existing_tp.get('LONG', 0))
                    s_tp = float(existing_tp.get('SHORT', 0))
                    latest_price = self.exchange.fetch_ticker(SYMBOL)['last']
                    
                    if long_contracts > 0 and l_tp > 0:
                        tp_str = self.exchange.price_to_precision(SYMBOL, l_tp)
                        if l_tp > latest_price:
                            self.exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side='sell', amount=None, price=None, params={'positionSide': 'LONG', 'stopPrice': float(tp_str), 'closePosition': True})
                            print(f"🛡️ 기존 롱 포지션 익절(TP) 복구 완료: {tp_str}")
                        else:
                            # [핵심 수정 1] 시장가 즉시 청산 시 closePosition 옵션을 빼고, 정확한 수량(long_contracts)을 넣어 해결
                            print(f"🚨 현재가({latest_price})가 롱 목표가({tp_str}) 돌파! 즉시 시장가 익절합니다.")
                            self.exchange.create_order(symbol=SYMBOL, type='market', side='sell', amount=long_contracts, params={'positionSide': 'LONG'})
                    
                    if short_contracts > 0 and s_tp > 0:
                        tp_str = self.exchange.price_to_precision(SYMBOL, s_tp)
                        if s_tp < latest_price:
                            self.exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side='buy', amount=None, price=None, params={'positionSide': 'SHORT', 'stopPrice': float(tp_str), 'closePosition': True})
                            print(f"🛡️ 기존 숏 포지션 익절(TP) 복구 완료: {tp_str}")
                        else:
                            # [핵심 수정 2] 시장가 즉시 청산 시 closePosition 옵션을 빼고, 정확한 수량(short_contracts)을 넣어 해결
                            print(f"🚨 현재가({latest_price})가 숏 목표가({tp_str}) 돌파! 즉시 시장가 익절합니다.")
                            self.exchange.create_order(symbol=SYMBOL, type='market', side='buy', amount=short_contracts, params={'positionSide': 'SHORT'})

            orders = decision.get('orders') or []
            if not orders:
                print("No new orders to execute.")
                return

            for order in orders:
                side = order.get('side') 
                pos_side = order.get('positionSide') 
                amount_usdt = float(order.get('amount_usdt', 0))
                price = float(order.get('price', 0))
                tp_price = float(order.get('take_profit_price', 0))

                if amount_usdt <= 0 or price <= 0: continue
                
                amount_coin_str = self.exchange.amount_to_precision(SYMBOL, amount_usdt / price)
                amount_coin = float(amount_coin_str)
                if amount_coin <= 0:
                    print(f"⚠️ 주문 수량이 거래소 최소 단위보다 작습니다. (USDT: {amount_usdt})")
                    continue
                
                price_str = self.exchange.price_to_precision(SYMBOL, price)
                print(f"Action: {side.upper()} {amount_coin} {SYMBOL} at {price_str} mapping to {pos_side}")
                
                if not DRY_RUN:
                    try:
                        entry_val = self.exchange.create_order(
                            symbol=SYMBOL, type='limit', side=side,
                            amount=amount_coin, price=float(price_str), params={'positionSide': pos_side}
                        )
                        print(f"✅ 진입 주문 접수 완료: {entry_val['id']}")
                        
                        if tp_price > 0:
                            tp_price_str = self.exchange.price_to_precision(SYMBOL, tp_price)
                            tp_side = 'sell' if side == 'buy' else 'buy'
                            tp_val = self.exchange.create_order(
                                symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side=tp_side,
                                amount=amount_coin, price=None,
                                params={'positionSide': pos_side, 'stopPrice': float(tp_price_str)} 
                            )
                            print(f"🎯 익절(TP) 예약 주문 동시 접수 완료: {tp_val['id']}")
                            
                    except Exception as e:
                        print(f"Error executing trade: {e}")
                else:
                    print(f"[DRY RUN] Limit Entry: {side.upper()} {amount_coin} at {price_str} ({pos_side})")
                    if tp_price > 0:
                        tp_side = 'sell' if side == 'buy' else 'buy'
                        print(f"[DRY RUN] Would place TP {tp_side.upper()} Trigger at {tp_price} ({pos_side})")

        except Exception as e:
            print(f"Unexpected error in execute_orders: {e}")

    def check_short_position_constraint(self, account_state):
        long_notional = float(account_state['long_position']['notional'])
        short_notional = float(account_state['short_position']['notional'])

        if short_notional > long_notional:
            print(f"🚨 Constraint Violation: Short({short_notional}) exceeds Long({long_notional}).")
            excess_short_notional = short_notional - long_notional
            
            if not DRY_RUN:
                try:
                    ticker = self.exchange.fetch_ticker(SYMBOL)
                    current_price = ticker['last']
                    amount_coin_to_reduce = float(self.exchange.amount_to_precision(SYMBOL, excess_short_notional / current_price))
                    
                    if amount_coin_to_reduce > 0:
                        reduce_val = self.exchange.create_order(
                            symbol=SYMBOL, type='market', side='buy',
                            amount=amount_coin_to_reduce, params={'positionSide': 'SHORT'}
                        )
                        print(f"✅ 숏 포지션 강제 축소 완료: {reduce_val['id']}")
                except Exception as e:
                    print(f"Error reducing short position: {e}")
            else:
                print(f"[DRY RUN] Would MARKET BUY to reduce SHORT by {excess_short_notional} USDT.")

    def run(self):
        print("🚀 AutoTrader Bot initialized. Starting main loop.")
        while True:
            try:
                print(f"\n--- Waking up ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
                
                df = self.fetch_data()
                if df is None or df.empty:
                    time.sleep(60); continue
                
                account_state = self.get_account_state()
                if account_state is None:
                    time.sleep(60); continue
                
                self.check_short_position_constraint(account_state)
                if not DRY_RUN: account_state = self.get_account_state()
                
                signal = self.get_gemini_signal(df, account_state)
                self.execute_orders(signal)
                
                print(f"💤 Sleeping for {LOOP_INTERVAL_MINUTES} minutes...")
                time.sleep(LOOP_INTERVAL_MINUTES * 60)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    try:
        trader = AutoTrader()
        trader.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot safely stopped by user (Ctrl+C).")
    except Exception as e:
        print(f"Fatal error: {e}")