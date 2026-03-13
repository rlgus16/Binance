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
            
            long_notional = abs(float(long_pos.get('notional', 0))) if long_pos else 0.0
            short_notional = abs(float(short_pos.get('notional', 0))) if short_pos else 0.0
            
            return {
                'usdt_free': usdt_free,
                'usdt_total': usdt_total,
                'long_position': {
                    'notional': long_notional,
                    'contracts': float(long_pos.get('contracts', 0)) if long_pos else 0.0,
                    'entryPrice': float(long_pos.get('entryPrice', 0)) if long_pos else 0.0,
                    'unrealizedPnl': float(long_pos.get('unrealizedPnl', 0)) if long_pos else 0.0,
                },
                'short_position': {
                    'notional': short_notional,
                    'contracts': float(short_pos.get('contracts', 0)) if short_pos else 0.0,
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

            if True: # 무조건 기존 주문 전체 취소!
                print("🗑️ 턴 시작: 모든 미체결 주문을 싹쓸이합니다...")
                
                cancel_success = False
                for _ in range(3):
                    try:
                        self.exchange.cancel_all_orders(SYMBOL)
                        cancel_success = True
                        time.sleep(2)
                        break
                    except Exception as e:
                        print(f"⚠️ 기존 주문 취소 실패 (2초 후 재시도...): {e}")
                        time.sleep(2)
                        
                if not cancel_success:
                    print("🚨 기존 TP 취소에 3회 연속 실패했습니다! 중복 주문 꼬임 대참사를 막기 위해 이번 턴을 포기합니다.")
                    return
                
                positions = self.exchange.fetch_positions([SYMBOL])
                long_pos = next((p for p in positions if p.get('side') == 'long'), None)
                short_pos = next((p for p in positions if p.get('side') == 'short'), None)
                
                long_contracts = float(long_pos.get('contracts', 0)) if long_pos else 0.0
                short_contracts = float(short_pos.get('contracts', 0)) if short_pos else 0.0
                
                # 여기로 이사 옵니다! 과거 잔고가 아닌, 방금 불러온 "최신 잔고"로 시뮬레이션을 돌립니다!
                pending_short_amount_coin = 0.0
                simulated_long_shield_coin = long_contracts
                simulated_tracked_short_coin = short_contracts
                
                new_orders = decision.get('orders') or []
                for order in new_orders:
                    if order.get('positionSide', '').upper() == 'SHORT' and order.get('side', '').lower() == 'sell':
                        a_usdt = float(order.get('amount_usdt') or 0)
                        p = float(order.get('price') or 0)
                        if p > 0:
                            raw_coin_str = self.exchange.amount_to_precision(SYMBOL, a_usdt / p)
                            order_coin = float(raw_coin_str)
                            if order_coin <= 0: continue
                            
                            if simulated_tracked_short_coin + order_coin > simulated_long_shield_coin:
                                order_coin = simulated_long_shield_coin - simulated_tracked_short_coin
                                capped_coin_str = self.exchange.amount_to_precision(SYMBOL, order_coin)
                                order_coin = float(capped_coin_str)
                            
                            if order_coin <= 0: continue
                                
                            simulated_tracked_short_coin += order_coin
                            pending_short_amount_coin += order_coin
                
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

            fresh_state = None
            for attempt in range(3):
                fresh_state = self.get_account_state()
                if fresh_state:
                    account_state = fresh_state
                    break
                else:
                    print(f"⚠️ 잔고 최신화 실패 ({attempt + 1}/3회). 2초 후 다시 시도합니다...")
                    time.sleep(2)
            
            # 3번 모두 실패했을 경우에만 최종적으로 취소 처리
            if not fresh_state:
                print("🚨 3회 연속 잔고 최신화 실패! 안전을 위해 이번 턴 신규 진입을 취소합니다.")
                return

            tracked_long = float(account_state['long_position']['notional'])
            actual_long_shield_coin = float(account_state['long_position']['contracts']) # 숏 방어막은 무조건 "코인 개수(Contracts)" 기준
            tracked_short_coin = float(account_state['short_position']['contracts'])

            for order in orders:
                side = order.get('side', '').lower() 
                pos_side = order.get('positionSide', '').upper()
                
                if not side or not pos_side: continue
                amount_usdt = float(order.get('amount_usdt') or 0)
                price = float(order.get('price') or 0)

                if amount_usdt <= 0 or price <= 0: continue
                
                # 먼저 정상적인 코인 수량을 계산합니다.
                amount_coin_str = self.exchange.amount_to_precision(SYMBOL, amount_usdt / price)
                amount_coin = float(amount_coin_str)
                if amount_coin <= 0: continue
                
                if pos_side == 'LONG' and side == 'buy':
                    if tracked_long + amount_usdt > MAX_LONG_SIZE_USDT:
                        amount_usdt = MAX_LONG_SIZE_USDT - tracked_long
                        if amount_usdt < 5.0:
                            print(f"⚠️ 롱 포지션 최대 한도({MAX_LONG_SIZE_USDT} USDT)에 도달했습니다. 추가 진입을 강제 차단합니다.")
                            continue
                        # USDT 한도가 깎였으므로 진입 수량(coin) 다시 계산
                        amount_coin_str = self.exchange.amount_to_precision(SYMBOL, amount_usdt / price)
                        amount_coin = float(amount_coin_str)
                        
                    tracked_long += amount_usdt 
                    
                # 숏 진입 (반드시 '이미 체결된' 롱 코인 개수 안에서만)
                elif pos_side == 'SHORT' and side == 'sell':
                    if tracked_short_coin + amount_coin > actual_long_shield_coin: 
                        amount_coin = actual_long_shield_coin - tracked_short_coin
                        if amount_coin <= 0:
                            print(f"⚠️ 숏 포지션이 실제 롱 포지션(방패) 수량을 초과하려 합니다! 진입을 강제 차단합니다.")
                            continue
                        # 수량이 깎였으므로 거래소 규격에 맞게 소수점 정밀도 다시 맞춤
                        amount_coin_str = self.exchange.amount_to_precision(SYMBOL, amount_coin)
                        amount_coin = float(amount_coin_str)
                        
                    tracked_short_coin += amount_coin
                
                # 허락되지 않은 방향의 환각 주문이 들어오면 무조건 차단
                else:
                    print(f"⚠️ AI가 신규 진입 로직에서 허가되지 않은 방향({pos_side} {side.upper()})의 주문을 시도했습니다! (강제 차단)")
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