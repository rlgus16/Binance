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
LOOP_INTERVAL_MINUTES = 15

class AutoTrader:
    def __init__(self):
        binance_api_key = os.getenv("BINANCE_API_KEY")
        binance_secret_key = os.getenv("BINANCE_SECRET_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not binance_api_key or not binance_secret_key:
            raise ValueError("🚨 .env 파일에 바이낸스 API 키가 누락되었습니다.")
        if not gemini_api_key:
            raise ValueError("🚨 .env 파일에 Gemini API 키가 누락되었습니다.")

        print("🔄 바이낸스 USDT-M 선물 거래소 초기화 중...")
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
            print("✅ 헤지 모드 (양방향 포지션) 활성화 및 확인 완료.")
            
            print(f"⚙️ {SYMBOL} 레버리지를 {LEVERAGE}배로 설정 중...")
            self.exchange.set_leverage(LEVERAGE, SYMBOL)
            
            print(f"⚙️ {SYMBOL} 마진 모드를 교차(CROSS)로 설정 중...")
            self.exchange.set_margin_mode('cross', SYMBOL)
        except Exception as e:
            print(f"⚠️ 설정 정보/경고: {e}")

    def fetch_data(self):
        print(f"📊 {SYMBOL}의 {TIMEFRAME} 캔들 데이터 가져오는 중...")
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
            print(f"❌ 데이터 가져오기 오류: {e}")
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
            
            long_notional = float(long_pos.get('contracts', 0)) * float(long_pos.get('entryPrice', 0)) if long_pos else 0.0
            short_notional = float(short_pos.get('contracts', 0)) * float(short_pos.get('entryPrice', 0)) if short_pos else 0.0
            
            return {
                'usdt_free': usdt_free,
                'usdt_total': usdt_total,
                'long_position': {
                    'notional': long_notional,
                    'entryPrice': float(long_pos.get('entryPrice', 0)) if long_pos else 0.0,
                    'unrealizedPnl': float(long_pos.get('unrealizedPnl', 0)) if long_pos else 0.0,
                },
                'short_position': {
                    'notional': short_notional,
                    'entryPrice': float(short_pos.get('entryPrice', 0)) if short_pos else 0.0,
                    'unrealizedPnl': float(short_pos.get('unrealizedPnl', 0)) if short_pos else 0.0,
                },
                'open_orders': [
                    {'id': o['id'], 'side': o['side'], 'type': o['type'], 'price': o['price'], 'amount': o['amount'], 'positionSide': o.get('info', {}).get('positionSide')}
                    for o in open_orders
                ]
            }
        except Exception as e:
            print(f"❌ 계좌 상태 가져오기 오류: {e}")
            return None

    def get_gemini_signal(self, df, account_state):
        print("🤖 Gemini 3.1 Pro 모델로 시장 데이터 분석 중...")
        recent_data = df.tail(100).to_dict(orient='records')
        
        system_instruction = f"""You are an advanced quantitative trading AI for Binance USD-M Futures.
You are trading {SYMBOL} on {TIMEFRAME} candles.

RULES AND CONSTRAINTS:
1. Cross Margin with {LEVERAGE}x Leverage.
2. Hedge Mode is ON. Open LONG and SHORT positions to maximize profit.
3. The MAXIMUM total LONG position size must NEVER exceed {MAX_LONG_SIZE_USDT} USDT (notional).
4. The TOTAL SHORT position size must NEVER exceed the CURRENT LONG NOTIONAL size at all times.
5. LONG doesn't need hedging. Free_balance is abundant for LONG.
6. SHORT must use LONG as a shield.
7. NEVER execute a STOP_LOSS order.
8. You can average down to maximize profit.
9. Exits MUST rely on take_profit orders hitting their targets.
10. Predict and provide take_profit for open positions.
11. Provide only one take_profit for each position.
12. Predict and place limit_order for entries.

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
            "price": <entry price limit>
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
            print(f"❌ Gemini AI 호출 오류: {e}")
            return None

    def execute_orders(self, signal_text, account_state):
        if not signal_text: return

        try:
            signal_text = signal_text.replace("```json\n", "").replace("```\n", "").replace("```", "")
            decision = json.loads(signal_text)
            print(f"💡 AI 분석 결과 및 전략: {decision.get('reasoning')}")

            # ==========================================
            # 🧮 [핵심 수학] 대기 중인(예약된) 숏 물량 계산
            # ==========================================
            pending_short_amount_coin = 0.0
            
            # 1. 취소하지 않고 유지하는 기존 숏 대기 주문 수량 합산
            if not decision.get('cancel_all_open_orders'):
                for o in account_state.get('open_orders', []):
                    if o.get('positionSide') == 'SHORT' and o.get('side') == 'sell':
                        pending_short_amount_coin += float(o.get('amount', 0))
            
            # 2. 이번 턴에 AI가 새롭게 진입하려는 신규 숏 대기 주문 수량 합산
            new_orders = decision.get('orders') or []
            for order in new_orders:
                if order.get('positionSide', '').upper() == 'SHORT' and order.get('side', '').lower() == 'sell':
                    a_usdt = float(order.get('amount_usdt') or 0)
                    p = float(order.get('price') or 0)
                    if p > 0:
                        amount_coin_str = self.exchange.amount_to_precision(SYMBOL, a_usdt / p)
                        pending_short_amount_coin += float(amount_coin_str)


            if True: # 무조건 기존 주문 전체 취소!
                print("🗑️ 턴 시작: 모든 미체결 주문을 싹쓸이합니다...")
                
                cancel_success = False
                for _ in range(3):
                    try:
                        self.exchange.cancel_all_orders(SYMBOL)
                        cancel_success = True
                        time.sleep(1)
                        break
                    except Exception as e:
                        print(f"⚠️ 기존 주문 취소 실패 (1초 후 재시도...): {e}")
                        time.sleep(1)
                        
                if not cancel_success:
                    print("🚨 기존 TP 취소에 3회 연속 실패했습니다! 중복 주문 꼬임 대참사를 막기 위해 이번 턴을 포기합니다.")
                    return
                
                positions = self.exchange.fetch_positions([SYMBOL])
                long_pos = next((p for p in positions if p.get('side') == 'long'), None)
                short_pos = next((p for p in positions if p.get('side') == 'short'), None)
                
                long_contracts = float(long_pos.get('contracts', 0)) if long_pos else 0.0
                short_contracts = float(short_pos.get('contracts', 0)) if short_pos else 0.0
                
                existing_tp = decision.get('existing_position_tp') or {}
                l_tp = float(existing_tp.get('LONG') or 0)
                s_tp = float(existing_tp.get('SHORT') or 0)
                
                # ==========================================
                # 🛡️ 롱 수학적 익절 (대기 숏 물량만큼 롱 익절 보류)
                # ==========================================
                amount_to_close_long = long_contracts - short_contracts - pending_short_amount_coin
                amount_to_close_long_str = self.exchange.amount_to_precision(SYMBOL, amount_to_close_long) if amount_to_close_long > 0 else "0"
                amount_to_close_long_clean = float(amount_to_close_long_str)

                if long_contracts > 0 and l_tp > 0:
                    # 새로운 TP를 걸기 전에 기존 롱 매도(TP) 주문을 찾아 모조리 취소합니다.
                    for o in account_state.get('open_orders', []):
                        if o.get('positionSide') == 'LONG' and o.get('side') == 'sell':
                            try:
                                self.exchange.cancel_order(o['id'], SYMBOL)
                                print(f"🔄 롱 익절가 갱신을 위해 기존 주문({o['id']})을 취소했습니다.")
                            except:
                                pass # 취소 실패해도 그냥 넘어감

                    if amount_to_close_long_clean > 0:
                        tp_str = self.exchange.price_to_precision(SYMBOL, l_tp)
                        latest_price = self.exchange.fetch_ticker(SYMBOL)['last']
                        if l_tp > latest_price:
                            try:
                                # closePosition 옵션을 빼고 안전 수량만큼만 부분 익절
                                self.exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side='sell', amount=amount_to_close_long_clean, price=None, params={'positionSide': 'LONG', 'stopPrice': float(tp_str)})
                                print(f"🛡️ 롱 익절 장전: {tp_str} (총 {long_contracts}개 중 익절 {amount_to_close_long_clean}개 / 방어용 예비군 {short_contracts + pending_short_amount_coin}개 유지)")
                            except Exception as e:
                                print(f"⚠️ 롱 부분 익절 예약 실패: {e}")
                        else:
                            try:
                                print(f"🚨 현재가({latest_price})가 롱 목표가({tp_str}) 돌파! 즉시 시장가 부분 익절합니다.")
                                self.exchange.create_order(symbol=SYMBOL, type='market', side='sell', amount=amount_to_close_long_clean, params={'positionSide': 'LONG'})
                            except Exception as e:
                                print(f"⚠️ 롱 시장가 부분 익절 실패: {e}")
                    else:
                        print(f"🛡️ 롱 수량({long_contracts}개)이 현재 숏+대기 숏({short_contracts + pending_short_amount_coin}개)과 같거나 적습니다! 숏 방어를 위해 롱 익절(TP)을 보류합니다.")
                
                # ==========================================
                # 🛡️ 숏 100% 전량 익절
                # ==========================================
                if short_contracts > 0 and s_tp > 0:
                    
                    # 새로운 숏 TP를 걸기 전에 기존 숏 매수(TP) 주문을 모조리 취소합니다.
                    for o in account_state.get('open_orders', []):
                        if o.get('positionSide') == 'SHORT' and o.get('side') == 'buy':
                            try:
                                self.exchange.cancel_order(o['id'], SYMBOL)
                                print(f"🔄 숏 익절가 갱신을 위해 기존 주문({o['id']})을 취소했습니다.")
                            except:
                                pass # 취소에 실패하더라도 다음 코드로 자연스럽게 넘어감

                    tp_str = self.exchange.price_to_precision(SYMBOL, s_tp)
                    latest_price = self.exchange.fetch_ticker(SYMBOL)['last']
                    
                    if s_tp < latest_price:
                        try:
                            self.exchange.create_order(symbol=SYMBOL, type='TAKE_PROFIT_MARKET', side='buy', amount=None, price=None, params={'positionSide': 'SHORT', 'stopPrice': float(tp_str), 'closePosition': True})
                            print(f"💰 숏 포지션 100% 전량 익절(TP) 장전 완료: {tp_str}")
                        except Exception as e:
                            print(f"⚠️ 숏 포지션 TP 복구 실패: {e}")
                    else:
                        try:
                            print(f"🚨 현재가({latest_price})가 숏 목표가({tp_str}) 돌파! 즉시 시장가로 100% 익절합니다.")
                            self.exchange.create_order(symbol=SYMBOL, type='market', side='buy', amount=short_contracts, params={'positionSide': 'SHORT'})
                        except Exception as e:
                            print(f"⚠️ 숏 시장가 익절 실패: {e}")

            orders = decision.get('orders') or []
            if not orders:
                print("🛑 실행할 새로운 진입 주문이 없습니다.")
                return

            fresh_state = self.get_account_state()
            if fresh_state:
                account_state = fresh_state

            tracked_long = float(account_state['long_position']['notional'])
            tracked_short = float(account_state['short_position']['notional'])

            for order in orders:
                side = order.get('side', '').lower() 
                pos_side = order.get('positionSide', '').upper()
                
                if not side or not pos_side: continue
                amount_usdt = float(order.get('amount_usdt') or 0)
                price = float(order.get('price') or 0)

                if amount_usdt <= 0 or price <= 0: continue
                
                if pos_side == 'LONG' and side == 'buy':
                    if tracked_long + amount_usdt > MAX_LONG_SIZE_USDT:
                        amount_usdt = MAX_LONG_SIZE_USDT - tracked_long
                        if amount_usdt < 5.0:
                            print(f"⚠️ 롱 포지션 최대 한도({MAX_LONG_SIZE_USDT} USDT)에 도달했습니다. 추가 진입을 강제 차단합니다.")
                            continue
                    tracked_long += amount_usdt 
                    
                # 숏 진입 (롱 방어막 크기 안에서만)
                elif pos_side == 'SHORT' and side == 'sell':
                    if tracked_short + amount_usdt > tracked_long:
                        amount_usdt = tracked_long - tracked_short
                        if amount_usdt < 5.0:
                            print(f"⚠️ 숏 포지션이 롱 포지션(방패) 크기를 초과하려 합니다! 진입을 강제 차단합니다.")
                            continue
                    tracked_short += amount_usdt 

                amount_coin_str = self.exchange.amount_to_precision(SYMBOL, amount_usdt / price)
                amount_coin = float(amount_coin_str)
                if amount_coin <= 0:
                    continue
                
                price_str = self.exchange.price_to_precision(SYMBOL, price)
                print(f"🎯 신규 액션: {side.upper()} {amount_coin} {SYMBOL} 진입가 {price_str} / 포지션: {pos_side}")
                
                try:
                    entry_val = self.exchange.create_order(
                        symbol=SYMBOL, type='limit', side=side,
                        amount=amount_coin, price=float(price_str), params={'positionSide': pos_side}
                    )
                    print(f"✅ 진입(Limit) 전송 완료 (주문번호: {entry_val['id']})")
                except Exception as e:
                    print(f"⚠️ 진입 주문 에러: {e}")

        except Exception as e:
            print(f"❌ 주문 실행 중 예기치 않은 오류 발생: {e}")

    def run(self):
        print("🚀 자동매매 봇 초기화 완료. [수학적 절대 방어 모드] 메인 루프를 시작합니다.")
        while True:
            try:
                print(f"\n--- ☀️ AI 기상 및 시장 분석 시작 ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
                
                df = self.fetch_data()
                if df is None or df.empty:
                    time.sleep(60); continue
                
                account_state = self.get_account_state()
                if account_state is None:
                    time.sleep(60); continue
                
                signal = self.get_gemini_signal(df, account_state)
                self.execute_orders(signal, account_state)
                
                print(f"💤 {LOOP_INTERVAL_MINUTES}분 동안 대기(수면) 모드 진입...")
                time.sleep(LOOP_INTERVAL_MINUTES * 60)
                
            except Exception as e:
                print(f"🚨 메인 루프 실행 중 치명적 오류 발생: {e}")
                print("⏳ 안전을 위해 5분 대기 후 재시도합니다...")
                time.sleep(300)

if __name__ == "__main__":
    try:
        trader = AutoTrader()
        trader.run()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 봇이 안전하게 종료되었습니다 (Ctrl+C).")
    except Exception as e:
        print(f"❌ 봇 전체 강제 종료 오류: {e}")