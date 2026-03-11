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
DRY_RUN = True  # Set to False to enable actual trading

class AutoTrader:
    def __init__(self):
        # Initialize Binance
        # Make sure your API keys in .env have Futures trading enabled
        binance_api_key = os.getenv("BINANCE_API_KEY")
        binance_secret_key = os.getenv("BINANCE_SECRET_KEY")
        
        # Initialize Gemini
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not binance_api_key or not binance_secret_key:
            raise ValueError("Missing Binance API keys in .env")
        if not gemini_api_key:
            raise ValueError("Missing Gemini API key in .env")

        print("Initializing Binance USDS-M Futures...")
        self.exchange = ccxt.binanceusdm({
            'apiKey': binance_api_key,
            'secret': binance_secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })

        self.client = genai.Client(api_key=gemini_api_key)
        self.setup_exchange()

    def setup_exchange(self):
        """Configure hedge mode, leverage, and margin type."""
        try:
            # Set Hedge Mode (Position Mode)
            position_mode = self.exchange.fapiPrivateGetPositionSideDual()
            if position_mode.get('dualSidePosition') == False or position_mode.get('dualSidePosition') == 'false':
                print("Enabling Hedge Mode (Dual Side Position)...")
                self.exchange.fapiPrivatePostPositionSideDual({'dualSidePosition': 'true'})
            else:
                print("Hedge Mode already enabled.")
                
            # Note: setting cross margin using ccxt can sometimes be tricky for specific symbols, 
            # we'll try to set leverage first.
            try:
                print(f"Setting leverage for {SYMBOL} to {LEVERAGE}x...")
                self.exchange.set_leverage(LEVERAGE, SYMBOL)
            except Exception as e:
                print(f"Leverage setting info: {e}")
            
            try:
                # Set multi-asset or cross margin if necessary. Note: set_margin_mode requires symbol
                print(f"Setting margin mode to CROSS for {SYMBOL}...")
                self.exchange.set_margin_mode('cross', SYMBOL)
            except Exception as e:
                print(f"Margin mode info/warning: {e} (Usually means it's already set or not applicable like this)")
        except Exception as e:
            print(f"Error during exchange setup: {e}")

    def fetch_data(self):
        """Fetch 4H candles and calculate technical indicators."""
        print(f"Fetching {TIMEFRAME} candles for {SYMBOL}...")
        try:
            # Fetch 100 recent candles to calculate indicators (like moving averages requiring history)
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Calculate Technical Indicators using pandas_ta
            df.ta.macd(append=True)
            df.ta.rsi(length=14, append=True)
            df.ta.sma(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.bbands(length=20, append=True)
            
            # Fill NaN values to avoid issues with LLM interpretation
            df.bfill(inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def get_account_state(self):
        """Fetch current balance, positions, and open orders."""
        try:
            balance = self.exchange.fetch_balance()
            usdt_free = balance.get('USDT', {}).get('free', 0.0)
            usdt_total = balance.get('USDT', {}).get('total', 0.0)
            
            positions = self.exchange.fetch_positions([SYMBOL])
            long_pos = next((p for p in positions if p['side'] == 'long'), None)
            short_pos = next((p for p in positions if p['side'] == 'short'), None)
            
            open_orders = self.exchange.fetch_open_orders(SYMBOL)
            
            return {
                'usdt_free': usdt_free,
                'usdt_total': usdt_total,
                'long_position': {
                    'notional': long_pos['notional'] if long_pos else 0,
                    'entryPrice': long_pos['entryPrice'] if long_pos else 0,
                    'unrealizedPnl': long_pos['unrealizedPnl'] if long_pos else 0,
                },
                'short_position': {
                    'notional': short_pos['notional'] if short_pos else 0,
                    'entryPrice': short_pos['entryPrice'] if short_pos else 0,
                    'unrealizedPnl': short_pos['unrealizedPnl'] if short_pos else 0,
                },
                'open_orders': [
                    {'id': o['id'], 'side': o['side'], 'type': o['type'], 'price': o['price'], 'amount': o['amount'], 'positionSide': o['info'].get('positionSide')}
                    for o in open_orders
                ]
            }
        except Exception as e:
            print(f"Error fetching account state: {e}")
            return None

    def get_gemini_signal(self, df, account_state):
        """Analyze data using Gemini 3.1 Pro and get trading signals."""
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

Respond ONLY with a valid JSON format (without markdown code blocks) representing your trading decision.
Format:
{{
    "reasoning": "Explain your technical analysis of the 4H trends, indicators, and why you are placing these orders...",
    "orders": [
        {{
            "side": "buy",
            "positionSide": "LONG",
            "type": "limit",
            "amount_usdt": <amount in USDT to add to position>,
            "price": <entry price limit>,
            "take_profit_price": <take profit target limit>
        }},
        {{
            "side": "sell",
            "positionSide": "SHORT",
            "type": "limit",
            "amount_usdt": <amount in USDT to add to position>,
            "price": <entry price limit>,
            "take_profit_price": <take profit target limit>
        }}
    ],
    "cancel_all_open_orders": true/false
}}
If no new entries are recommended based on your constraints, or if doing so would exceed the constraints, return an empty "orders" array.
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
                model='gemini-3.1-pro',
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
        """Parse the LLM signal and execute actual trades."""
        if not signal_text:
            return

        try:
            # Strip markdown if present
            signal_text = signal_text.replace("```json\n", "").replace("```\n", "").replace("```", "")
            decision = json.loads(signal_text)
            print(f"AI Reasoning: {decision.get('reasoning')}")

            if decision.get('cancel_all_open_orders'):
                print("Canceling all open orders as instructed by AI...")
                if not DRY_RUN:
                    self.exchange.cancel_all_orders(SYMBOL)
                else:
                    print(f"[DRY RUN] Would cancel all open orders for {SYMBOL}")

            orders = decision.get('orders', [])
            if not orders:
                print("No new orders to execute.")
                return

            for order in orders:
                side = order.get('side') # 'buy' or 'sell'
                pos_side = order.get('positionSide') # 'LONG' or 'SHORT'
                amount_usdt = float(order.get('amount_usdt', 0))
                price = float(order.get('price', 0))
                tp_price = order.get('take_profit_price')

                if amount_usdt <= 0 or price <= 0:
                    continue
                
                # Calculate coin amount
                amount_coin = amount_usdt / price
                amount_coin = self.exchange.amount_to_precision(SYMBOL, amount_coin)
                
                print(f"Action: {side.upper()} {amount_coin} {SYMBOL} at {price} mapping to {pos_side}")
                
                if not DRY_RUN:
                    try:
                        # Place Entry Order
                        entry_val = self.exchange.create_order(
                            symbol=SYMBOL,
                            type='limit',
                            side=side,
                            amount=float(amount_coin),
                            price=price,
                            params={'positionSide': pos_side}
                        )
                        print(f"Successfully placed entry order: {entry_val['id']}")
                        
                        # Place TP Order
                        if tp_price and float(tp_price) > 0:
                            tp_side = 'sell' if side == 'buy' else 'buy'
                            tp_val = self.exchange.create_order(
                                symbol=SYMBOL,
                                type='limit',
                                side=tp_side,
                                amount=float(amount_coin),
                                price=float(tp_price),
                                params={'positionSide': pos_side} # TP must match the SAME positionSide!
                            )
                            print(f"Successfully placed take profit order: {tp_val['id']}")

                    except Exception as e:
                        print(f"Error executing trade: {e}")
                else:
                    print(f"[DRY RUN] Would place {side.upper()} limit for {amount_coin} coins at {price} (PositionSide: {pos_side})")
                    if tp_price and float(tp_price) > 0:
                        tp_side = 'sell' if side == 'buy' else 'buy'
                        print(f"[DRY RUN] Would place TP {tp_side.upper()} limit for {amount_coin} coins at {tp_price} (PositionSide: {pos_side})")

        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini JSON output: {e}")
            print(f"Raw output was:\n{signal_text}")
        except Exception as e:
            print(f"Unexpected error in execute_orders: {e}")

    def check_short_position_constraint(self, account_state):
        """
        Verify that the total short_size does not exceed the current long_size.
        If it does (e.g., due to a LONG take-profit hitting), reduce the short position.
        """
        long_notional = float(account_state['long_position']['notional'])
        short_notional = float(account_state['short_position']['notional'])

        if short_notional > long_notional:
            print(f"Constraint Violation: Short size ({short_notional}) exceeds Long size ({long_notional}).")
            excess_short_notional = short_notional - long_notional
            
            # We need to buy to reduce the short position.
            # Get the current price from the last candle or just use a market order to reduce immediately.
            # To be safe and ensure the constraint is met immediately, a market order is best.
            print(f"Reducing short position by {excess_short_notional} USDT.")
            
            if not DRY_RUN:
                try:
                    # In ccxt for Binance USD-M futures, we can place a market buy to close the short.
                    # We need the amount in coins. We can fetch the current ticker or use the entry price as a rough estimate for amount calculation,
                    # but it's better to fetch the ticker to get the exact current amount needed.
                    ticker = self.exchange.fetch_ticker(SYMBOL)
                    current_price = ticker['last']
                    amount_coin_to_reduce = excess_short_notional / current_price
                    amount_coin_to_reduce = self.exchange.amount_to_precision(SYMBOL, amount_coin_to_reduce)
                    
                    if float(amount_coin_to_reduce) > 0:
                        reduce_val = self.exchange.create_order(
                            symbol=SYMBOL,
                            type='market',
                            side='buy',
                            amount=float(amount_coin_to_reduce),
                            params={'positionSide': 'SHORT'} # Important: buy on the SHORT side to reduce it
                        )
                        print(f"Successfully reduced short position: {reduce_val['id']}")
                    else:
                        print("Amount to reduce is too small.")
                except Exception as e:
                    print(f"Error reducing short position: {e}")
            else:
                print(f"[DRY RUN] Would place MARKET BUY to reduce SHORT position by roughly {excess_short_notional} USDT.")
        else:
            print("Short position constraint check passed.")

    def run(self):
        print("Bot initialized successfully. Starting main loop.")
        while True:
            try:
                print(f"\n--- Waking up ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
                
                # 1. Fetch 4H candles and Indicators
                df = self.fetch_data()
                if df is None or df.empty:
                    print("Failed to fetch data, retrying in 1 minute...")
                    time.sleep(60)
                    continue
                
                # 2. Fetch Account State
                account_state = self.get_account_state()
                if account_state is None:
                    print("Failed to fetch account state, retrying in 1 minute...")
                    time.sleep(60)
                    continue
                
                # 2.5 Ensure constraints are met before asking Gemini
                self.check_short_position_constraint(account_state)
                # Re-fetch state if we adjusted it
                account_state = self.get_account_state()
                
                # 3. Get LLM Prediction and JSON signal
                signal = self.get_gemini_signal(df, account_state)
                
                # 4. Execute orders based on parsed signal
                self.execute_orders(signal)
                
                print(f"Sleeping for {LOOP_INTERVAL_MINUTES} minutes...")
                time.sleep(LOOP_INTERVAL_MINUTES * 60)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    try:
        trader = AutoTrader()
        trader.run()
    except Exception as e:
        print(f"Fatal error: {e}")
