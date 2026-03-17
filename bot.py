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
TIMEFRAME_EXEC = '8h'  # 매매 진입 타점용 (실행 프레임)
TIMEFRAME_TREND = '1d' # 큰 추세 확인용 (트렌드 프레임)
TIMEFRAME_MACRO = '1w' # 초거시적 추세 확인용 (매크로 프레임 - 주봉)
LEVERAGE = 5
MAX_LONG_SIZE_USDT = 2500
LOOP_INTERVAL_MINUTES = 30

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

    def fetch_data(self, timeframe):
        print(f"📊 {SYMBOL}의 {timeframe} 캔들 데이터 가져오는 중...")
        try:
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=100)
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
            print(f"❌ {timeframe} 데이터 가져오기 오류: {e}")
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

    def get_gemini_signal(self, df_exec, df_trend, df_macro, account_state):
        print("🤖 Gemini 모델로 데이터 분석 중...")
        
        cols_to_keep = ['open', 'high', 'low', 'close', 'volume', 'MACD_12_26_9', 'RSI_14', 'SMA_20', 'EMA_50']
        
        data_exec = df_exec[cols_to_keep].tail(50).round(3).to_dict(orient='records') 
        data_trend = df_trend[cols_to_keep].tail(30).round(3).to_dict(orient='records') 
        data_macro = df_macro[cols_to_keep].tail(25).round(3).to_dict(orient='records')
        
        max_allowed_long = min(MAX_LONG_SIZE_USDT, float(account_state['usdt_total']))
        
        system_instruction = f"""Quant trading AI for {SYMBOL} (Binance Futures).

RULES AND CONSTRAINTS:
1. Mode: Hedge Mode ON, Cross Margin, {LEVERAGE}x Leverage.
2. Risk: Max LONG {max_allowed_long} USDT. SHORT notional MUST <= LONG notional.
3. Use LONG as a shield for SHORT. LONG doesn't need hedging. Free_balance is abundant for LONG.
4. Strategy: NO STOP_LOSS. Use averaging down. Exit via TAKE_PROFIT only.
5. Orders: Use limit orders for entries. Minimum order amount > 20 USDT. Always provide TP for each position.
6. Trend: Follow {TIMEFRAME_MACRO} & {TIMEFRAME_TREND} trends. NEVER counter-trade {TIMEFRAME_MACRO} trend.

Respond ONLY with JSON:
{{
    "reasoning": "Brief analysis...",
    "cancel_all_open_orders": true/false,
    "existing_position_tp": {{"LONG": price, "SHORT": price}},
    "orders": [{{"side": "buy/sell", "positionSide": "LONG/SHORT", "type": "limit", "amount_usdt": val, "price": val}}]
}}"""
        prompt = f"""
Current Account State:
USDT Free: {account_state['usdt_free']}
USDT Total: {account_state['usdt_total']}
Long Position: Notional {account_state['long_position']['notional']} USDT at Avg Price {account_state['long_position']['entryPrice']}
Short Position: Notional {account_state['short_position']['notional']} USDT at Avg Price {account_state['short_position']['entryPrice']}
Open Orders count: {len(account_state['open_orders'])}

[SUPER MACRO TREND - Last prices from {TIMEFRAME_MACRO} Candles]
{data_macro}

[MACRO TREND - Last prices from {TIMEFRAME_TREND} Candles]
{data_trend}

[EXECUTION TIMING - Last prices from {TIMEFRAME_EXEC} Candles]
{data_exec}

Based on this 3-stage multi-timeframe analysis, what are your next orders?
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
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

            if True: 
                print("🗑️ 턴 시작: 모든 미체결 주문을 싹쓸이합니다...")
                
                cancel_success = False
                for _ in range(3):
                    try:
                        self.exchange.cancel_all_orders(SYMBOL)
                        time.sleep(2) 
                        
                        leftovers = self.exchange.fetch_open_orders(SYMBOL)
                        for leftover in leftovers:
                            try:
                                self.exchange.cancel_order(leftover['id'], SYMBOL)
                                print(f"🗑️ 끈질긴 TP 주문 개별 삭제 완료 (ID: {leftover['id']})")
                            except Exception as ex:
                                print(f"⚠️ 개별 취소 에러 발생 (최종 검증으로 넘어감): {ex}")
                                
                        time.sleep(2) 
                        final_check = self.exchange.fetch_open_orders(SYMBOL)
                        if len(final_check) > 0:
                            raise Exception(f"여전히 {len(final_check)}개의 주문이 지워지지 않고 살아있습니다!")
                                
                        cancel_success = True
                        time.sleep(2) 
                        break
                    except Exception as e:
                        print(f"⚠️ 기존 주문 취소/확인 실패 (60초 후 재시도...): {e}")
                        time.sleep(60)
                        
                if not cancel_success:
                    print("🚨 기존 TP 취소에 3회 연속 실패했습니다! 중복 주문 꼬임 대참사를 막기 위해 이번 턴을 포기합니다.")
                    return
                
                positions = self.exchange.fetch_positions([SYMBOL])
                long_pos = next((p for p in positions if p.get('side') == 'long'), None)
                short_pos = next((p for p in positions if p.get('side') == 'short'), None)
                
                long_contracts = float(long_pos.get('contracts', 0)) if long_pos else 0.0
                short_contracts = float(short_pos.get('contracts', 0)) if short_pos else 0.0
                
                # 진입 평단가 추출 (손절 방어용)
                long_entry_price = float(long_pos.get('entryPrice', 0)) if long_pos else 0.0
                short_entry_price = float(short_pos.get('entryPrice', 0)) if short_pos else 0.0
                
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
                # 🛡️ 롱 수학적 익절 (손절 강제 차단 포함)
                # ==========================================
                amount_to_close_long = long_contracts - short_contracts - pending_short_amount_coin
                amount_to_close_long_str = self.exchange.amount_to_precision(SYMBOL, amount_to_close_long) if amount_to_close_long > 0 else "0"
                amount_to_close_long_clean = float(amount_to_close_long_str)

                if long_contracts > 0 and l_tp > 0:
                    # [핵심 방어 로직] 롱 목표가가 내 평단가보다 낮거나 같으면 무조건 거부!
                    if l_tp <= long_entry_price:
                        print(f"🚨 [강제 차단] AI가 롱 진입가({long_entry_price})보다 낮거나 같은 목표가({l_tp})를 제시했습니다!")
                    else:
                        if amount_to_close_long_clean > 0:
                            tp_str = self.exchange.price_to_precision(SYMBOL, l_tp)
                            latest_price = self.exchange.fetch_ticker(SYMBOL)['last']
                            if l_tp > latest_price:
                                try:
                                    self.exchange.create_order(symbol=SYMBOL, type='limit', side='sell', amount=amount_to_close_long_clean, price=float(tp_str), params={'positionSide': 'LONG'})
                                    print(f"🛡️ 롱 부분 익절(Limit) 장전: {tp_str} (총 {long_contracts}개 중 익절 {amount_to_close_long_clean}개 / 방어용 예비군 {short_contracts + pending_short_amount_coin}개 유지)")
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
                # 🛡️ 숏 100% 전량 익절 (손절 강제 차단 포함)
                # ==========================================
                if short_contracts > 0 and s_tp > 0:
                    # [핵심 방어 로직] 숏 목표가가 내 평단가보다 높거나 같으면 무조건 거부!
                    if s_tp >= short_entry_price:
                        print(f"🚨 [강제 차단] AI가 숏 진입가({short_entry_price})보다 높거나 같은 목표가({s_tp})를 제시했습니다!")
                    else:
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
                    print(f"⚠️ 잔고 최신화 실패 ({attempt + 1}/3회). 60초 후 다시 시도합니다...")
                    time.sleep(60)
            
            if not fresh_state:
                print("🚨 3회 연속 잔고 최신화 실패! 안전을 위해 이번 턴 신규 진입을 취소합니다.")
                return

            tracked_long = float(account_state['long_position']['notional'])
            actual_long_shield_coin = float(account_state['long_position']['contracts'])
            tracked_short_coin = float(account_state['short_position']['contracts'])

            for order in orders:
                side = order.get('side', '').lower() 
                pos_side = order.get('positionSide', '').upper()
                
                if not side or not pos_side: continue
                amount_usdt = float(order.get('amount_usdt') or 0)
                price = float(order.get('price') or 0)

                if amount_usdt <= 0 or price <= 0: continue
                
                amount_coin_str = self.exchange.amount_to_precision(SYMBOL, amount_usdt / price)
                amount_coin = float(amount_coin_str)
                if amount_coin <= 0: continue
                
                if pos_side == 'LONG' and side == 'buy':
                    dynamic_max_long = min(MAX_LONG_SIZE_USDT, float(account_state['usdt_total'])) 
                    if tracked_long + amount_usdt > dynamic_max_long:
                        amount_usdt = dynamic_max_long - tracked_long
                        if amount_usdt < 5.0:
                            print(f"⚠️ 롱 포지션 최대 한도(내 잔고: {dynamic_max_long:.2f} USDT)에 도달했습니다. 진입을 차단합니다.")
                            continue
                        amount_coin_str = self.exchange.amount_to_precision(SYMBOL, amount_usdt / price)
                        amount_coin = float(amount_coin_str)
                        
                    tracked_long += amount_usdt
                    
                elif pos_side == 'SHORT' and side == 'sell':
                    if tracked_short_coin + amount_coin > actual_long_shield_coin: 
                        amount_coin = actual_long_shield_coin - tracked_short_coin
                        if amount_coin <= 0:
                            print(f"⚠️ 숏 포지션이 실제 롱 포지션(방패) 수량을 초과하려 합니다! 진입을 강제 차단합니다.")
                            continue
                        amount_coin_str = self.exchange.amount_to_precision(SYMBOL, amount_coin)
                        amount_coin = float(amount_coin_str)
                        
                    tracked_short_coin += amount_coin
                
                else:
                    print(f"⚠️ AI가 신규 진입 로직에서 허가되지 않은 방향({pos_side} {side.upper()})의 주문을 시도했습니다! (강제 차단)")
                    continue

                price_str = self.exchange.price_to_precision(SYMBOL, price)
                if amount_coin * float(price_str) < 20.0:
                    print(f"⚠️ 주문 금액 너무 작음 차단: {amount_coin * float(price_str):.2f} USDT (바이낸스 최소 기준 20 USDT 미만). 주문을 건너뜁니다.")
                    continue
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
        print("🚀 자동매매 봇 초기화 완료. 메인 루프를 시작합니다.")
        while True:
            try:
                print(f"\n--- ☀️ AI 기상 및 3단계 시장 분석 시작 ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
                
                df_exec = self.fetch_data(TIMEFRAME_EXEC)
                df_trend = self.fetch_data(TIMEFRAME_TREND)
                df_macro = self.fetch_data(TIMEFRAME_MACRO)
                
                if df_exec is None or df_exec.empty or df_trend is None or df_trend.empty or df_macro is None or df_macro.empty:
                    time.sleep(60); continue 
                
                account_state = self.get_account_state()
                if account_state is None:
                    time.sleep(60); continue 
                
                signal = self.get_gemini_signal(df_exec, df_trend, df_macro, account_state)
                self.execute_orders(signal, account_state)
                
                print(f"💤 {LOOP_INTERVAL_MINUTES}분 동안 대기(수면) 모드 진입...")
                time.sleep(LOOP_INTERVAL_MINUTES * 60)
                
            except Exception as e:
                print(f"🚨 메인 루프 실행 중 치명적 오류 발생: {e}")
                print("⏳ 안전을 위해 10분 대기 후 재시도합니다...")
                time.sleep(600)

if __name__ == "__main__":
    try:
        trader = AutoTrader()
        trader.run()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 봇이 안전하게 종료되었습니다 (Ctrl+C).")
    except Exception as e:
        print(f"❌ 봇 전체 강제 종료 오류: {e}")