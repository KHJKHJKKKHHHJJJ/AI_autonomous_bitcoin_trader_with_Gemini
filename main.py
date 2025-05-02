print("========= Welcome to GEMINI BTC TRADER =========")
print("initializing...")
import bit_AI
import time
import warnings
warnings.filterwarnings('ignore')
import datetime
import json
import asyncio
import Prud_AI
import logging
from dotenv import load_dotenv
# import sqlite3 as sql # DB 사용 안 함
from bit_AI import (
    get_held_symbols_from_binance, gen_bit_model, get_instructions as get_bit_instructions, get_binance_chart, get_technical_indicators,
    get_binance_balances, get_trade_suggestion, # gem_sug 제거
    execute_binance_market_buy, execute_binance_market_sell, # 바이낸스 거래 함수 임포트
    log_trade_decision, send_telegram_message, # 필요한 함수 명시
    binance_client # binance_client 직접 사용 위해 import
)
import pytz # 한국 시간대 설정

# --- Global Variables & Constants --- #
last_prudence_run_date = None # 마지막 Prudence AI 실행 날짜 저장
todays_symbols = None # 오늘 거래할 심볼 목록 저장
todays_prudence_reason = None # 오늘 심볼 선정 이유 저장
MIN_BUY_AMOUNT_USDT = 10.0 # 최소 매수 주문 금액 (USDT)
MAX_PORTFOLIO_POSITIONS = 5   # 동시에 보유할 최대 자산 종류 수
MAX_USDT_PER_POSITION = 20.0  # 신규 포지션 하나에 투자할 최대 USDT 금액
last_check_times = {} # 각 심볼의 마지막 확인 시간을 저장하는 딕셔너리
DEFAULT_CHECK_INTERVAL_MINUTES = 15 # AI가 유효한 주기를 알려주지 않을 경우 기본 대기 시간 (분)
MINIMUM_CHECK_INTERVAL_MINUTES = 30 # AI 제안과 관계없이 적용할 최소 확인 간격 (분)
KST = pytz.timezone('Asia/Seoul') # 한국 시간대 정의 추가
CONFIDENCE_THRESHOLD = 0.7 # 매수/매도 결정을 실행할 최소 신뢰도

# --- Configuration --- #ㄷㄷ
load_dotenv()

# --- Helper function to get portfolio summary --- #
def get_portfolio_summary():
    """현재 포트폴리오 상태 요약을 계산하여 반환합니다."""
    summary = {
        "held_symbols_list": [],
        "holdings_value_usdt": {}, # 각 자산의 USDT 가치
        "total_portfolio_value_usdt": 0.0,
        "usdt_balance": 0.0,
        "num_positions": 0,
        "max_positions": MAX_PORTFOLIO_POSITIONS
    }
    total_asset_value = 0.0

    if not bit_AI.binance_client:
        logging.error("Binance client not available for portfolio summary.")
        return summary # 기본 빈 요약 반환

    try:
        # 1. 전체 계정 잔고 가져오기
        account_info = bit_AI.binance_client.get_account()
        balances = account_info.get('balances', [])
        held_assets = {} # 보유 자산과 수량 저장 (USDT 제외)
        for balance in balances:
            asset = balance.get('asset')
            free = float(balance.get('free', 0.0))
            locked = float(balance.get('locked', 0.0))
            total_balance = free + locked
            if total_balance > 0.0:
                if asset == 'USDT':
                    summary['usdt_balance'] = free # 사용 가능한 USDT
                else:
                    # 수량이 매우 작은 자산은 제외 (예: 0.00000001) - 필요시 조정
                    if total_balance > 1e-8:
                         held_assets[asset] = total_balance # 총 보유량 (free+locked)

        summary['held_symbols_list'] = [f"{asset}USDT" for asset in held_assets.keys()]
        summary['num_positions'] = len(held_assets)

        # 2. 모든 티커 가격 가져오기 (가치 계산용)
        if held_assets: # 보유 자산이 있을 때만 가격 조회
            all_tickers = bit_AI.binance_client.get_all_tickers()
            ticker_map = {ticker['symbol']: float(ticker['price']) for ticker in all_tickers}

            # 3. 각 보유 자산의 USDT 가치 계산 및 총 자산 가치 합산
            for asset, quantity in held_assets.items():
                symbol = f"{asset}USDT"
                current_price = ticker_map.get(symbol)
                if current_price:
                    asset_value_usdt = quantity * current_price
                    summary['holdings_value_usdt'][asset] = asset_value_usdt
                    total_asset_value += asset_value_usdt
                else:
                    logging.warning(f"Could not find USDT pair price for held asset {asset}. Value calculation might be incomplete.")
                    summary['holdings_value_usdt'][asset] = 0.0 # 가격 없으면 0으로

        # 4. 총 포트폴리오 가치 계산 (보유 자산 가치 + USDT 잔고)
        summary['total_portfolio_value_usdt'] = total_asset_value + summary['usdt_balance']

        logging.info(f"Portfolio Summary Calculated: Positions={summary['num_positions']}/{summary['max_positions']}, Total Value={summary['total_portfolio_value_usdt']:.2f} USDT, Available USDT={summary['usdt_balance']:.2f}")

    except Exception as e:
        logging.error(f"Error calculating portfolio summary: {e}", exc_info=True)
        # 오류 발생 시에도 부분적인 정보라도 반환하거나 기본값 반환

    return summary

# --- Helper function to log portfolio status --- #
def log_portfolio_status(portfolio_summary):
    """주어진 포트폴리오 요약 정보를 로그 파일(PRUD_LOG_FILE)에 기록합니다."""
    log_entry = {
        "log_time": datetime.datetime.now(KST).isoformat(),
        "role": "portfolio_summary", # 역할 구분
        "portfolio_data": portfolio_summary # 요약 정보 전체 저장
    }
    try:
        with open(Prud_AI.PRUD_LOG_FILE, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')
        # logging.info(f"Portfolio status logged to {Prud_AI.PRUD_LOG_FILE}") # 너무 자주 로깅될 수 있으므로 주석 처리
    except Exception as e:
        logging.error(f"Error logging portfolio status to {Prud_AI.PRUD_LOG_FILE}: {e}")

def get_instructions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        print(f"Instruction file not found: {file_path}")
        return None # 파일을 찾지 못하면 None 반환
    except Exception as e:
        print(f"An error occurred while reading the file {file_path}: {e}")
        return None # 다른 오류 발생 시 None 반환

def check_and_run_prudence_ai(pru_chat_session):
    """오늘 거래할 심볼 목록을 가져오거나, 필요 시 Prudence AI를 실행합니다."""
    global last_prudence_run_date, todays_symbols, todays_prudence_reason
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    # --- Default Result (실패 또는 유효 데이터 없을 시, 또는 AI 비활성화 시) ---
    default_result = {
        "date": today_str,
        "symbols_to_trade": ["BTCUSDT", "ETHUSDT"], # 테스트용 기본 심볼
        "reason": "AI Call Disabled for Testing / Prudence AI failed or no valid symbols suggested."
    }

    # 1. 메모리에 저장된 오늘 날짜 및 결과 확인
    if last_prudence_run_date == today_str and todays_symbols is not None: # 심볼 목록 존재 여부 확인
        logging.info(f"Symbol list for today ({today_str}) already in memory. Using cached symbols: {todays_symbols}")
        # 메모리에 있는 데이터를 반환 형식에 맞춰 반환
        return {
            "date": today_str,
            "symbols_to_trade": todays_symbols,
            "reason": todays_prudence_reason or "Reason cached from memory."
        }

    # 2. 로그 파일 확인 (메모리에 없거나 결과가 없는 경우)
    try:
        with open(Prud_AI.PRUD_LOG_FILE, 'r', encoding='utf-8') as f:
            last_line = None
            for line in f:
                last_line = line
            if last_line:
                try:
                    last_log = json.loads(last_line.strip())
                    # prudence_data 안에 결과가 저장됨
                    prudence_data_from_log = last_log.get('prudence_data')
                    if isinstance(prudence_data_from_log, dict):
                        log_date_str = prudence_data_from_log.get('date')
                        symbols_from_log = prudence_data_from_log.get('symbols_to_trade')
                        reason_from_log = prudence_data_from_log.get('reason')

                        # 로그 시간으로 날짜 재확인 (선택적)
                        if not log_date_str and 'log_time' in last_log:
                             try:
                                 log_date_str = datetime.datetime.fromisoformat(last_log['log_time']).strftime("%Y-%m-%d")
                             except ValueError: pass

                        # 오늘 날짜 로그이고, symbols_to_trade가 리스트 형태인지 확인
                        if log_date_str == today_str and isinstance(symbols_from_log, list) and symbols_from_log:
                            logging.info(f"Found valid symbol list for today ({today_str}) in log file: {symbols_from_log}")
                            last_prudence_run_date = today_str # 메모리 업데이트
                            todays_symbols = symbols_from_log # 메모리 업데이트
                            todays_prudence_reason = reason_from_log # 메모리 업데이트
                            return prudence_data_from_log # 로그 파일의 prudence_data 전체 반환
                        elif log_date_str == today_str:
                            logging.warning(f"Found log entry for today ({today_str}), but symbols_to_trade is missing or invalid: {prudence_data_from_log}")
                except json.JSONDecodeError as json_err:
                    logging.error(f"Error decoding JSON from last log line in {Prud_AI.PRUD_LOG_FILE}: {json_err}. Line: '{last_line.strip()}'")
    except FileNotFoundError:
        logging.info(f"{Prud_AI.PRUD_LOG_FILE} not found. Will run Prudence AI.")
    except Exception as e:
        logging.error(f"Error reading {Prud_AI.PRUD_LOG_FILE}: {e}. Will proceed as if no log found.")

    # 3. 오늘 실행/기록된 유효한 결과가 없으면 Prudence AI 실행
    logging.info(f"No valid symbol list found for today ({today_str}) in memory or logs. Calling Prudence AI...")
    # --- AI 호출 다시 활성화 --- #
    try:
        # gem_pru_sug는 이제 get_symbols_to_trade를 호출함
        prudence_result = Prud_AI.gem_pru_sug(pru_chat_session)
        if prudence_result and isinstance(prudence_result.get('symbols_to_trade'), list) and prudence_result['symbols_to_trade']:
            logging.info(f"Prudence AI call completed successfully. Symbols: {prudence_result['symbols_to_trade']}")
            last_prudence_run_date = today_str # 성공 시 메모리 업데이트
            todays_symbols = prudence_result['symbols_to_trade'] # 성공 시 메모리 업데이트
            todays_prudence_reason = prudence_result.get('reason') # 성공 시 메모리 업데이트
            return prudence_result
        else:
            logging.warning(f"Prudence AI call returned invalid or empty result: {prudence_result}")
            # 실패 시 메모리의 오늘 날짜 기록은 유지하되, 심볼 리스트는 None으로 둠
            last_prudence_run_date = today_str
            todays_symbols = None
            todays_prudence_reason = None
            return default_result # 기본값 반환
    except Exception as e:
        logging.error(f"Error occured during Prudence AI processing: {e}")
        last_prudence_run_date = today_str # 실패해도 날짜는 기록
        todays_symbols = None
        todays_prudence_reason = None
        return default_result # 실패 시 기본값 반환
    # --- AI 호출 다시 활성화 끝 --- #
    # last_prudence_run_date = today_str # 날짜는 기록 (try 블록 내부에서 처리됨)
    # todays_symbols = default_result["symbols_to_trade"] # 기본값 사용 (try 블록 내부에서 처리됨)
    # todays_prudence_reason = default_result["reason"] # 기본값 사용 (try 블록 내부에서 처리됨)
    # return default_result # 테스트용 기본값 반환 (try 블록 내부에서 처리됨)

print("Gemini models will be processed.")
time.sleep(0.5)

# Instruction 파일 로드
bit_instruction_path = './Bitcoin Gemini Instruction.md'
pru_instruction_path = './Prudence Gemini Instruction.md'

bit_instruction = get_instructions(bit_instruction_path)
pru_instruction = get_instructions(pru_instruction_path)

# 모델 초기화 (Instruction 로드 성공 시에만)
bit_chat_session = None
pru_chat_session = None

if bit_instruction:
    bit_chat_session = gen_bit_model(bit_instruction)
if pru_instruction:
    pru_chat_session = Prud_AI.gen_pru_model(pru_instruction)

# 모델 초기화 실패 시 종료
if not bit_chat_session or not pru_chat_session:
    print("Failed to initialize Gemini models. Exiting.")
    exit() # 프로그램 종료

# --- 텔레그램 알림: 시스템 시작 --- #
try:
    send_telegram_message("🚀 AI Trader System Initialized 🚀")
except Exception as e:
    logging.error(f"Failed to send initial Telegram message: {e}")

former_dec = 3 # 1 = sell/ 0 = buy/ 3 = initial

u_input = ''
ticker = 0
timeout = 0
# timestamp = datetime.datetime.now() # DB 사용 안 함

while u_input not in ['Y', 'y', 'Yes', 'yes', 'YES']:
    current_time = datetime.datetime.now(KST) # 현재 시간 (타임존 포함)
    if ticker >= timeout:
        # --- Prudence AI 호출: 오늘 거래할 심볼 목록 가져오기 (AI 비활성화됨) --- #
        prudence_run_result = check_and_run_prudence_ai(pru_chat_session)
        recommended_symbols = prudence_run_result.get('symbols_to_trade', [])
        prudence_reasoning = prudence_run_result.get('reason', 'No reason provided.')

        logging.info(f"Prudence AI recommended symbols (testing): {recommended_symbols}")
        # --- 텔레그램 알림: Prudence 결과 --- #
        try:
            send_telegram_message(f"🤔 Prudence Check ({prudence_run_result.get('date')})\nRecommended: {recommended_symbols}\nReason: {prudence_reasoning}")
        except Exception as e:
            logging.error(f"Failed to send Prudence result Telegram message: {e}")

        # --- 바이낸스에서 현재 보유 중인 심볼 목록 가져오기 --- #
        held_symbols = get_held_symbols_from_binance() # bit_AI 임포트 확인
        logging.info(f"Currently held symbols on Binance (excluding USDT): {held_symbols}")
        # --- 텔레그램 알림: 보유 종목 --- #
        try:
            send_telegram_message(f"💼 Held Symbols: {held_symbols if held_symbols else 'None'}")
        except Exception as e:
            logging.error(f"Failed to send Held Symbols Telegram message: {e}")

        # --- 추천 목록과 보유 목록 합치기 (중복 제거) --- #
        symbols_to_process_today = list(set(recommended_symbols) | set(held_symbols))
        logging.info(f"Final list of symbols to process today: {symbols_to_process_today}")
        # --- 텔레그램 알림: 최종 처리 목록 --- #
        try:
             send_telegram_message(f"📋 Symbols to Process: {symbols_to_process_today if symbols_to_process_today else 'None'}")
        except Exception as e:
            logging.error(f"Failed to send Symbols to Process Telegram message: {e}")

        if not symbols_to_process_today: # 처리할 최종 심볼 목록이 없으면 대기
            logging.warning("No symbols recommended or held. Waiting for 1 hour.")
            timeout = 60 * 60 # 심볼 없으면 1시간 후 재시도
            ticker = 0
            continue # 다음 루프 실행

        logging.info(f"=== Symbols to process today ({prudence_run_result.get('date')}): {symbols_to_process_today} ===")
        # logging.info(f"Prudence Reason (applies to recommended symbols): {prudence_reasoning}")

        # --- 포트폴리오 요약 정보 계산 (사이클당 한 번) --- #
        portfolio_summary = get_portfolio_summary()
        log_portfolio_status(portfolio_summary) # 포트폴리오 상태 로깅 추가
        # --- 텔레그램 알림: 포트폴리오 요약 --- #
        try:
            pf_msg = f"📊 Portfolio Summary:\nPositions: {portfolio_summary.get('num_positions', 'N/A')}/{portfolio_summary.get('max_positions', 'N/A')}\nTotal Value: ${portfolio_summary.get('total_portfolio_value_usdt', 0.0):.2f}\nAvailable USDT: ${portfolio_summary.get('usdt_balance', 0.0):.2f}"
            send_telegram_message(pf_msg)
        except Exception as e:
            logging.error(f"Failed to send portfolio summary Telegram message: {e}")
        # --- 포트폴리오 요약 정보 계산 끝 --- #

        logging.info(f"Final symbol list to process: {symbols_to_process_today}")

        # --- 각 심볼에 대해 Trading AI 처리 --- #
        all_decisions = {} # 모든 심볼의 결정을 저장할 딕셔너리
        suggested_wait_times = [] # 각 심볼별 제안된 대기 시간을 저장할 리스트

        for symbol_to_process in symbols_to_process_today:
            logging.info(f"\n--- Processing symbol: {symbol_to_process} --- ")
            decision_data = None
            try:
                # --- 경과 시간 계산 --- #
                last_checked = last_check_times.get(symbol_to_process)
                elapsed_minutes = 0
                if last_checked:
                    time_diff = current_time - last_checked
                    elapsed_minutes = time_diff.total_seconds() / 60
                logging.info(f"Time since last check for {symbol_to_process}: {elapsed_minutes:.1f} minutes")
                # --- 경과 시간 계산 끝 --- #

                # 1. 데이터 가져오기 (차트 및 지표)
                logging.info(f"Fetching chart data for {symbol_to_process}...")
                chart_df = get_binance_chart(symbol=symbol_to_process)
                indicator_df = None
                if chart_df is not None:
                    indicator_df = get_technical_indicators(chart_df)
                else:
                    logging.error(f"Failed to get chart data for {symbol_to_process}. Skipping symbol.")
                    continue # 다음 심볼 처리

                if indicator_df is None:
                    logging.error(f"Failed to get indicator data for {symbol_to_process}. Skipping symbol.")
                    continue # 다음 심볼 처리

                # 2. 잔고 가져오기 (개별 심볼 기준)
                base_asset = symbol_to_process.replace('USDT', '')
                base_balance, quote_balance = get_binance_balances(base_asset=base_asset, quote_asset='USDT')
                if base_balance is None or quote_balance is None:
                     logging.warning(f"Could not retrieve Binance balances for {symbol_to_process}. Skipping...")
                     continue # 잔고 조회 실패 시 해당 심볼 건너뛰기

                # 3. Trading AI 호출 (portfolio_summary 전달)
                logging.info(f"Calling Trading AI for {symbol_to_process}...")
                decision_data = get_trade_suggestion(bit_chat_session,
                                                   prudence_context=prudence_run_result,
                                                   indicator_df=indicator_df,
                                                   base_balance=base_balance,
                                                   quote_balance=quote_balance,
                                                   symbol=symbol_to_process,
                                                   elapsed_minutes=elapsed_minutes,
                                                   portfolio_summary=portfolio_summary) # 포트폴리오 요약 전달

                # --- 마지막 확인 시간 업데이트 --- #
                last_check_times[symbol_to_process] = current_time
                # --- 마지막 확인 시간 업데이트 끝 --- #

                # 4. 결정 로그 저장 및 텔레그램 알림
                if decision_data:
                    log_trade_decision(decision_data, symbol=symbol_to_process)
                    all_decisions[symbol_to_process] = decision_data
                    # --- 결과 출력 및 텔레그램 알림 --- #
                    print(f"======== Gemini Suggestion Recipt ({symbol_to_process}) ========")
                    print(f"Decision:    {decision_data.get('decision', 'N/A')}")
                    print(f"Reason:      {decision_data.get('reason', 'N/A')}")

                    # --- 신뢰도 처리 --- #
                    confidence = decision_data.get('confidence') # 신뢰도 값 가져오기
                    valid_confidence = False
                    conf_str = "N/A"
                    if isinstance(confidence, (float, int)) and 0.0 <= confidence <= 1.0:
                        valid_confidence = True
                        conf_str = f"{confidence:.2f}"
                    else:
                        logging.warning(f"Invalid or missing confidence value ({confidence}) received for {symbol_to_process}.")
                        # 유효하지 않은 신뢰도면 거래 안 함, 로그는 남김
                    # --- 신뢰도 처리 끝 --- #

                    # --- 텔레그램 알림: 개별 심볼 결정 (신뢰도 포함) --- #
                    try:
                        # confidence_str 은 위에서 정의됨
                        send_telegram_message(f"🤖 Decision for {symbol_to_process}: {decision_data.get('decision')} (Conf: {conf_str})\nReason: {decision_data.get('reason')}")
                    except Exception as e:
                         logging.error(f"Failed to send individual decision Telegram message for {symbol_to_process}: {e}")

                    # --- 제안된 대기 시간 추출 --- #
                    next_check = decision_data.get('next_check_minutes')
                    if isinstance(next_check, int) and next_check > 0:
                        suggested_wait_times.append(next_check)
                        logging.info(f"AI suggested next check for {symbol_to_process} in {next_check} minutes.")
                    else:
                        logging.warning(f"Invalid 'next_check_minutes' ({next_check}) from AI for {symbol_to_process}. Using default.")
                    # --- 제안된 대기 시간 추출 끝 --- #

                    # --- 5. 거래 실행 로직 --- #
                    trade_decision = decision_data.get('decision')
                    if trade_decision == "BUY":
                        logging.info(f"BUY decision received for {symbol_to_process}. Confidence: {conf_str}")
                        # --- 신뢰도 임계값 확인 --- #
                        if valid_confidence and confidence >= CONFIDENCE_THRESHOLD:
                            logging.info(f"Confidence ({conf_str}) meets threshold ({CONFIDENCE_THRESHOLD}). Checking portfolio constraints...")

                            # --- 포트폴리오 제약 조건 확인 --- #
                            current_held_symbols = get_held_symbols_from_binance() # 현재 보유 심볼 다시 확인
                            num_held_positions = len(current_held_symbols)

                            if num_held_positions < MAX_PORTFOLIO_POSITIONS:
                                logging.info(f"Portfolio slots available ({num_held_positions}/{MAX_PORTFOLIO_POSITIONS}). Calculating buy amount...")

                                # --- 동적 매수 금액 계산 --- #
                                # 최대 포지션당 금액과 현재 USDT 잔고 중 작은 값으로 최대 투자 가능액 결정
                                max_investment_this_position = min(MAX_USDT_PER_POSITION, quote_balance)
                                logging.info(f"Calculated max investment for this position: {max_investment_this_position:.2f} USDT (Max per pos: {MAX_USDT_PER_POSITION}, Available USDT: {quote_balance:.2f})")

                                # --- 최소 매수 금액 확인 --- #
                                if max_investment_this_position >= MIN_BUY_AMOUNT_USDT:
                                    dynamic_buy_amount = max_investment_this_position
                                    logging.info(f"Dynamic buy amount ({dynamic_buy_amount:.2f} USDT) meets minimum ({MIN_BUY_AMOUNT_USDT}). Executing market buy...")
                                    execute_binance_market_buy(symbol_to_process, dynamic_buy_amount)
                                else:
                                    logging.warning(f"Calculated buy amount ({max_investment_this_position:.2f} USDT) is below minimum ({MIN_BUY_AMOUNT_USDT}). Skipping BUY for {symbol_to_process}.")
                                    send_telegram_message(f"ℹ️ BUY signal for {symbol_to_process} ignored (Calculated amount {max_investment_this_position:.2f} < Min {MIN_BUY_AMOUNT_USDT}).")
                                # --- 최소 매수 금액 확인 끝 --- #

                            else:
                                logging.warning(f"Maximum portfolio positions ({MAX_PORTFOLIO_POSITIONS}) reached. Cannot open new position for {symbol_to_process}. Currently holding: {current_held_symbols}")
                                send_telegram_message(f"ℹ️ BUY signal for {symbol_to_process} ignored (Max portfolio positions {MAX_PORTFOLIO_POSITIONS} reached).")
                            # --- 포트폴리오 제약 조건 확인 끝 --- #

                        elif valid_confidence:
                             logging.info(f"Confidence ({conf_str}) below threshold ({CONFIDENCE_THRESHOLD}). Skipping BUY for {symbol_to_process}.")
                             send_telegram_message(f"ℹ️ BUY signal for {symbol_to_process} ignored (Confidence {conf_str} < {CONFIDENCE_THRESHOLD}).")
                        else: # valid_confidence가 False인 경우 (위에서 이미 경고 로깅됨)
                             logging.warning(f"BUY signal for {symbol_to_process} ignored due to invalid confidence value: {confidence}")
                    elif trade_decision == "SELL":
                        logging.info(f"SELL decision received for {symbol_to_process}. Confidence: {conf_str}")
                        # --- 신뢰도 임계값 확인 (매도에도 적용) --- #
                        if valid_confidence and confidence >= CONFIDENCE_THRESHOLD:
                            logging.info(f"Confidence ({conf_str}) meets threshold ({CONFIDENCE_THRESHOLD}). Checking balance...")
                            if base_balance > 0:
                                logging.info(f"Sufficient {base_asset} balance ({base_balance:.8f}). Executing market sell (all)..." )
                                execute_binance_market_sell(symbol_to_process, base_balance)
                            else:
                                logging.warning(f"No {base_asset} balance ({base_balance:.8f}) to sell for {symbol_to_process}. Skipping sell.")
                        elif valid_confidence:
                            logging.info(f"Confidence ({conf_str}) below threshold ({CONFIDENCE_THRESHOLD}). Skipping SELL for {symbol_to_process}.")
                            send_telegram_message(f"ℹ️ SELL signal for {symbol_to_process} ignored (Confidence {conf_str} < {CONFIDENCE_THRESHOLD}).")
                        else: # valid_confidence가 False인 경우
                            logging.warning(f"SELL signal for {symbol_to_process} ignored due to invalid confidence value: {confidence}")
                    elif trade_decision == "HOLD":
                         logging.info(f"HOLD decision received for {symbol_to_process}. No trade action taken.")
                    else:
                         logging.warning(f"Unknown decision '{trade_decision}' received for {symbol_to_process}. No trade action taken.")
                    # --- 거래 실행 로직 끝 --- #

                else:
                    logging.warning(f"Received None decision from Trading AI for {symbol_to_process}. No trade action taken.")
                    all_decisions[symbol_to_process] = None # 실패 기록

            except Exception as e:
                logging.error(f"Error processing symbol {symbol_to_process}: {e}", exc_info=True) # Include traceback
                all_decisions[symbol_to_process] = {"error": str(e)} # 에러 기록
                # 오류 발생 시 텔레그램 알림 (선택적)
                try:
                    send_telegram_message(f"Trading AI ERROR ({symbol_to_process}): {e}")
                except Exception as tel_e:
                     logging.error(f"Failed to send Telegram error notification: {tel_e}")
                # 특정 에러(예: Rate Limit) 시 전체 루프 중단 또는 긴 대기 고려 가능
                # if '429' in str(e): ...
                continue # 오류 발생 시 다음 심볼 처리

        # --- 모든 심볼 처리 후 다음 timeout 설정 --- #
        next_timeout_minutes = DEFAULT_CHECK_INTERVAL_MINUTES # 기본값으로 초기화

        if suggested_wait_times:
            min_wait_minutes = min(suggested_wait_times)
            logging.info(f"AI suggested minimum next check in {min_wait_minutes} minutes.")
            next_timeout_minutes = min_wait_minutes # AI 제안 사용
        else:
            logging.warning(f"No valid next check times suggested by AI. Using default: {DEFAULT_CHECK_INTERVAL_MINUTES} minutes.")
            # next_timeout_minutes는 이미 기본값으로 설정됨

        # --- 최소 확인 간격 강제 적용 --- #
        final_timeout_minutes = max(next_timeout_minutes, MINIMUM_CHECK_INTERVAL_MINUTES)
        timeout = final_timeout_minutes * 60 # 초 단위로 변환

        if final_timeout_minutes != next_timeout_minutes:
             logging.warning(f"Enforcing minimum check interval. Setting timeout to {final_timeout_minutes} minutes (Minimum: {MINIMUM_CHECK_INTERVAL_MINUTES} minutes).")
        else:
             logging.info(f"Setting next cycle timeout to {final_timeout_minutes} minutes.")

        logging.info(f"Next symbol processing cycle in {timeout} seconds ({final_timeout_minutes:.1f} minutes)")
        ticker = 0 # 타이머 초기화
    else:
        # 1분마다 ticker 증가 및 대기
        ticker += 60
        # print(f"Waiting... {ticker}/{timeout} seconds passed.") # 진행 상황 로그 (선택적)
        time.sleep(60)

logging.info("Exiting main loop.") # 일반적으로는 도달하지 않음
