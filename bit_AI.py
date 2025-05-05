#!/usr/bin/env python
# coding: utf-8

import google.generativeai as genai
import requests
import json
import logging
import time # time 추가
import re # Import regular expression module

import base64
import hashlib
import hmac
import uuid
# import httplib2 # requests로 대체
from pandas_ta import ema, stochrsi
from pandas_ta.candles import ha as hei

# import KEYS
from dotenv import load_dotenv
import os
load_dotenv()

# import sqlite3 as sql # DB 사용 안 함
# from bs4 import BeautifulSoup # 사용 안 됨
import datetime
import pandas as pd

import telegram # python-telegram-bot 사용 가정
import asyncio

from binance.client import Client # Import Binance Client
from binance.enums import * # Import enums for order types, sides etc.
from binance.exceptions import BinanceAPIException # <<< Add this import
import pytz # For timezone conversion

from decimal import Decimal, ROUND_DOWN

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 상수 정의
COINONE_API_URL = "https://api.coinone.co.kr"
REQUEST_TIMEOUT = 10 # API 요청 타임아웃 (초)
BIT_LOG_FILE = "bit_log.jsonl" # Trading AI 전용 로그 파일
TRADE_LOG_FILE = "trade_log.jsonl" # Trading AI 결정 로그 파일

# --- Constants ---
CHART_INTERVAL = '1h' # 1시간 봉
DATA_LIMIT = 300 # 가져올 데이터 개수 (RSI, Stochastic 등 계산에 충분하도록)
SYMBOL = 'BTCUSDT' # 분석할 바이낸스 심볼 (BTC/USDT)
KST = pytz.timezone('Asia/Seoul') # 한국 시간대

# --- Load Environment Variables ---
load_dotenv()
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
# Prud_AI와의 통신을 위한 API 키 (별도 관리)
GEMINI_API_KEY = os.getenv("Gemini") # Prud_AI 와 동일한 키 사용 가정

# --- Initialize Binance Client ---
if BINANCE_API_KEY and BINANCE_API_SECRET:
    binance_client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    logging.info("Binance client initialized.")
else:
    binance_client = None
    logging.warning("Binance API Key or Secret not found. Some functionalities (like balance check, trading) will be disabled.")

# --- Helper function to load instructions (from Prud_AI.py) ---
def get_instructions(file_path):
    """주어진 파일 경로에서 지침 텍스트를 읽어 반환합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Instruction file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading instruction file {file_path}: {e}")
        return None

# --- Gemini 관련 함수 ---
def model_usage(response):
    try:
        # 모델 정보 가져오기 (모델 이름 확인 필요)
        # model_info = genai.get_model("models/gemini-1.5-pro-002") # 실제 모델 이름 확인 필요
        # logging.info(f"Model Info: Output Token Limit={model_info.output_token_limit}")
        logging.info(f"Gemini Usage Metadata: {response.usage_metadata}")
        # 텔레그램 알림은 필요한 경우에만 호출 (비동기 문제 고려)
        # asyncio.run(send_telegram_message(f"Usage: {response.usage_metadata}"))
    except Exception as e:
        logging.error(f"Error getting model usage info: {e}")
    
def gen_bit_model(instruction):
    try:
        genai.configure(api_key=os.getenv("Gemini"))
        generation_config = {
        "temperature": 0.75,
        "top_p": 0.95,
        "top_k": 40,
                "max_output_tokens": 8192, # 필요시 조정
        "response_mime_type": "application/json",
        }
        model = genai.GenerativeModel(
                model_name="gemini-2.5-pro-preview-03-25", # 안정 버전 사용 권장 또는 최신 확인
        generation_config=generation_config,
        system_instruction=instruction,
        )
        logging.info("Bitcoin Trading Gemini model initialized.")
        return model.start_chat()
    except Exception as e:
        logging.error(f"Failed to initialize Bitcoin Trading Gemini model: {e}")
        return None

# --- 데이터 조회 함수 ---
def get_binance_chart(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1HOUR, limit=DATA_LIMIT):
    """Fetches historical klines from Binance and returns a processed DataFrame."""
    if not binance_client:
        logging.error("Binance client is not initialized. Cannot fetch chart data.")
        return None

    logging.info(f"Fetching {limit} historical klines for {symbol} with interval {interval} from Binance...")
    try:
        # Fetch klines
        klines = binance_client.get_historical_klines(symbol, interval, f"{limit + 5} hours ago UTC", limit=limit) # 조금 더 여유롭게 가져옴

        if not klines:
            logging.warning("No klines data received from Binance.")
            return None

        # Create DataFrame
        df = pd.DataFrame(klines, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])

        # Convert timestamp to KST datetime and set as index
        # Binance returns UTC milliseconds
        df['timestamp'] = pd.to_datetime(df['Open time'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(KST)
        df.set_index('timestamp', inplace=True)


        # Convert necessary columns to numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Select and rename columns to match indicator calculation needs
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        # Rename columns for pandas_ta compatibility if needed (usually not necessary for standard OHLCV)
        # df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

        logging.info(f"Successfully fetched and processed {len(df)} klines.")
        return df

    except Exception as e:
        logging.error(f"Error fetching or processing Binance klines: {e}")
        return None

def get_technical_indicators(df):
    """pandas_ta를 사용하여 기술적 지표를 계산합니다."""
    # Check if DataFrame is valid
    if df is None or df.empty:
        logging.error("Cannot calculate indicators: DataFrame is None or empty.")
        return None

    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Missing required columns for indicator calculation. Found: {df.columns.tolist()}")
        return None

    try:
        logging.info("Calculating technical indicators...")
        # Use pandas_ta to calculate indicators
        # MACD
        df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
        # RSI (제거 예정 - 일단 계산은 유지하나 prompt에서 제외)
        df.ta.rsi(close='Close', length=14, append=True)
        # Stochastic Oscillator (%K, %D) (제거)
        # df.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, append=True)
        # Stochastic RSI 추가
        df.ta.stochrsi(close='Close', length=14, rsi_length=14, k=3, d=3, append=True)
        # EMA 200
        df.ta.ema(close='Close', length=200, append=True)
        # Heikin Ashi 추가
        df.ta.ha(append=True)

        # Rename EMA column if needed (pandas_ta might name it EMA_200)
        if 'EMA_200' in df.columns:
            df.rename(columns={'EMA_200': 'ema200'}, inplace=True)
        # Check if the column exists after potential rename or direct calculation
        if 'ema200' not in df.columns:
             logging.warning("EMA 200 column ('ema200') not found after calculation.")

        # Heikin Ashi 컬럼 이름 확인 및 로깅 (HA_open, HA_high, HA_low, HA_close)
        ha_cols = [col for col in df.columns if col.startswith('HA_')]
        if not ha_cols:
            logging.warning("Heikin Ashi columns not found after calculation.")
        else:
            logging.info(f"Heikin Ashi columns added: {ha_cols}")

        # Stoch RSI 컬럼 이름 확인 및 로깅 (STOCHRSIk_14_14_3_3, STOCHRSId_14_14_3_3)
        stochrsi_cols = [col for col in df.columns if col.startswith('STOCHRSI')]
        if not stochrsi_cols:
             logging.warning("Stochastic RSI columns not found after calculation.")
        else:
            logging.info(f"Stochastic RSI columns added: {stochrsi_cols}")

        # Handle potential NaN values - Strategy: Forward fill then backward fill
        # df.fillna(method='ffill', inplace=True)
        # df.fillna(method='bfill', inplace=True)
        # Keep NaNs for now, signal determination logic handles them.

        logging.info("Technical indicators calculated successfully.")
        return df

    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        logging.debug(f"DataFrame info before error:\n{df.info()}") # Log more info
        return None

def get_current_btc_price():
    "Coinone API를 통해 현재 BTC/KRW 가격을 가져옵니다."
    url = f"{COINONE_API_URL}/public/v2/ticker_utc_new/KRW/BTC?additional_data=true"
    headers = {"accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        tickers_data = response.json().get('tickers')
        if tickers_data and isinstance(tickers_data, list) and len(tickers_data) > 0:
            last_price = tickers_data[0].get('last')
            if last_price:
                return float(last_price)
        logging.warning("Could not extract last BTC price from API response.")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching BTC ticker: {e}")
        return None
    except (json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
        logging.error(f"Error parsing BTC ticker data: {e}")
        return None

# --- Coinone Private API 관련 함수 ---

def get_encoded_payload(payload):
    payload['nonce'] = str(uuid.uuid4())
    dumped_json = json.dumps(payload)
    encoded_json = base64.b64encode(dumped_json.encode('utf-8'))
    return encoded_json

def get_signature(encoded_payload):
    api_secret = os.getenv("Secret")
    if not api_secret:
        logging.error("Coinone API Secret Key not found in environment variables.")
        return None
    signature = hmac.new(api_secret.encode('utf-8'), encoded_payload, hashlib.sha512)
    return signature.hexdigest()

def make_coinone_request(action, payload):
    "Coinone Private API에 요청을 보냅니다."
    url = f'{COINONE_API_URL}{action}' # action 앞에 /가 포함되어 있다고 가정
    access_token = os.getenv("Access")
    if not access_token:
        logging.error("Coinone API Access Token not found.")
        return None

    payload['access_token'] = access_token
    encoded_payload = get_encoded_payload(payload)
    signature = get_signature(encoded_payload)

    if not signature:
        return None

    headers = {
        'Content-type': 'application/json',
        'X-COINONE-PAYLOAD': encoded_payload,
        'X-COINONE-SIGNATURE': signature,
    }

    try:
        response = requests.post(url, headers=headers, data=encoded_payload, timeout=REQUEST_TIMEOUT) # POST data로 payload 전달
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Coinone API request failed ({action}): {e}")
        # 응답 내용 로깅 (오류 디버깅 시 유용)
        if e.response is not None:
            logging.error(f"Response body: {e.response.text}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding Coinone API response ({action}): {e}")
        return None

def get_wallet_balances(currencies=['KRW', 'BTC']):
    "주어진 통화들의 지갑 잔액 정보를 가져옵니다."
    payload = {'currencies': currencies}
    response_data = make_coinone_request(action='/v2.1/account/balance', payload=payload)

    if response_data and response_data.get('result') == 'success':
        return response_data.get('balances')
    else:
        error_msg = response_data.get('error_msg', 'Unknown error') if response_data else 'No response'
        logging.error(f"Failed to get wallet balances: {error_msg}")
        return None

# --- Binance Account Balance (수정) ---
def get_binance_balances(base_asset='BTC', quote_asset='USDT'): # 기본값을 BTC/USDT로 유지
    """Fetches balances for the specified base and quote assets from Binance."""
    if not binance_client:
        logging.error("Binance client is not initialized. Cannot fetch balances.")
        return None, None # Return tuple of Nones

    logging.info(f"Fetching Binance balances for {base_asset} and {quote_asset}...")
    try:
        # Fetch balance for base asset (e.g., DOGE)
        base_asset_balance = binance_client.get_asset_balance(asset=base_asset)
        # Fetch balance for quote asset (e.g., USDT)
        quote_asset_balance = binance_client.get_asset_balance(asset=quote_asset)

        # Extract the 'free' (available) balance
        base_avail = float(base_asset_balance['free']) if base_asset_balance and 'free' in base_asset_balance else 0.0
        quote_avail = float(quote_asset_balance['free']) if quote_asset_balance and 'free' in quote_asset_balance else 0.0

        logging.info(f"Binance Balances - {base_asset} Available: {base_avail}, {quote_asset} Available: {quote_avail}")
        return base_avail, quote_avail # Return tuple (base_balance, quote_balance)
    except Exception as e:
        # Log the specific exception from Binance if available
        logging.error(f"Error fetching Binance balances for {base_asset}/{quote_asset}: {e}")
        return None, None # Return tuple of Nones on error

# --- 신규 함수: 보유 중인 알트코인 심볼 목록 가져오기 ---
def get_held_symbols_from_binance():
    """Fetches all assets with a balance > 0 (excluding USDT) from Binance
       and returns them as a list of USDT pair symbols (e.g., ['BTCUSDT', 'ETHUSDT']).
    """
    if not binance_client:
        logging.error("Binance client is not initialized. Cannot fetch held symbols.")
        return [] # Return empty list

    logging.info("Fetching held symbols from Binance account...")
    held_symbols = []
    try:
        account_info = binance_client.get_account()
        balances = account_info.get('balances', [])

        for balance in balances:
            asset = balance.get('asset')
            free_balance = float(balance.get('free', 0.0))

            # Check if balance is positive and asset is not the quote currency (USDT)
            # Also exclude common stablecoins if needed, but start with just USDT
            if free_balance > 0.0 and asset != 'USDT':
                 # Construct the symbol name (assuming USDT pair)
                 symbol = f"{asset}USDT"
                 held_symbols.append(symbol)
                 logging.debug(f"Found held asset: {asset}, Balance: {free_balance}, Added Symbol: {symbol}")

        logging.info(f"Found {len(held_symbols)} held symbols (excluding USDT): {held_symbols}")
        return held_symbols
    except Exception as e:
        logging.error(f"Error fetching or processing account balances for held symbols: {e}")
        return [] # Return empty list on error

# --- Gemini 제안 함수 (수정) ---
def get_trade_suggestion(chat_session, prudence_context, indicator_df, base_balance, quote_balance, symbol, elapsed_minutes, portfolio_summary): # portfolio_summary 인자 추가
    """기술적 지표, 신중함 컨텍스트, 지갑 상태, 경과 시간, 포트폴리오 요약, 심볼 정보를 종합하여 Gemini에게 거래 제안을 요청합니다."""
    if not chat_session:
        logging.error("Gemini chat session is not initialized.")
    if indicator_df is None or indicator_df.empty:
        logging.warning(f"Indicator DataFrame for {symbol} is empty. Cannot generate context.")
        return None
    if prudence_context is None: # 이제 심볼 선정 이유 등을 담은 딕셔너리
        logging.warning("Prudence context is missing. Cannot generate context.")
        return None
    if base_balance is None or quote_balance is None:
         logging.warning(f"Wallet balance data for {symbol} pair is missing.")
         return None
    if elapsed_minutes is None or not isinstance(elapsed_minutes, (int, float)) or elapsed_minutes < 0:
        logging.warning(f"Invalid elapsed_minutes value ({elapsed_minutes}). Using 0.")
        elapsed_minutes = 0 # 유효하지 않으면 0으로 처리
    if portfolio_summary is None: # 포트폴리오 요약 유효성 검사
        logging.warning("Portfolio summary is missing. Cannot generate full context.")
        # portfolio_summary = {} # 또는 기본값 설정
        return None # 필수 정보로 간주하고 None 반환

    try:
        # 최신 데이터 추출
        latest_indicators = indicator_df.iloc[-1].to_dict()

        # Format indicators safely
        close_str = f"{latest_indicators.get('Close', 'N/A'):.2f}" if pd.notna(latest_indicators.get('Close')) else "N/A"
        # rsi_str = f"{latest_indicators.get('RSI_14', 'N/A'):.2f}" if pd.notna(latest_indicators.get('RSI_14')) else "N/A" # RSI 제외
        # stochk_str = f"{latest_indicators.get('STOCHk_14_3_3', 'N/A'):.2f}" if pd.notna(latest_indicators.get('STOCHk_14_3_3')) else "N/A" # Stoch 제외
        # stochd_str = f"{latest_indicators.get('STOCHd_14_3_3', 'N/A'):.2f}" if pd.notna(latest_indicators.get('STOCHd_14_3_3')) else "N/A" # Stoch 제외
        stochrsi_k_str = f"{latest_indicators.get('STOCHRSIk_14_14_3_3', 'N/A'):.2f}" if pd.notna(latest_indicators.get('STOCHRSIk_14_14_3_3')) else "N/A"
        stochrsi_d_str = f"{latest_indicators.get('STOCHRSId_14_14_3_3', 'N/A'):.2f}" if pd.notna(latest_indicators.get('STOCHRSId_14_14_3_3')) else "N/A"
        ha_open_str = f"{latest_indicators.get('HA_open', 'N/A'):.2f}" if pd.notna(latest_indicators.get('HA_open')) else "N/A"
        ha_high_str = f"{latest_indicators.get('HA_high', 'N/A'):.2f}" if pd.notna(latest_indicators.get('HA_high')) else "N/A"
        ha_low_str = f"{latest_indicators.get('HA_low', 'N/A'):.2f}" if pd.notna(latest_indicators.get('HA_low')) else "N/A"
        ha_close_str = f"{latest_indicators.get('HA_close', 'N/A'):.2f}" if pd.notna(latest_indicators.get('HA_close')) else "N/A"
        macd_str = f"{latest_indicators.get('MACD_12_26_9', 'N/A'):.4f}" if pd.notna(latest_indicators.get('MACD_12_26_9')) else "N/A"
        macds_str = f"{latest_indicators.get('MACDs_12_26_9', 'N/A'):.4f}" if pd.notna(latest_indicators.get('MACDs_12_26_9')) else "N/A"
        ema200_str = f"{latest_indicators.get('ema200', 'N/A'):.2f}" if pd.notna(latest_indicators.get('ema200')) else "N/A"
        # Format balances safely
        base_asset = symbol.replace('USDT', '') # Derive base asset from symbol
        base_bal_str = f"{base_balance:.8f}" if pd.notna(base_balance) else "N/A"
        quote_bal_str = f"{quote_balance:.2f}" if pd.notna(quote_balance) else "N/A"

        # Prudence context (Reasoning for selecting this symbol pool)
        # prudence_context는 이제 reason 키를 포함한 dict임
        prudence_reasoning = prudence_context.get('reason', 'N/A') if isinstance(prudence_context, dict) else 'Invalid Prudence Context'

        # 포트폴리오 요약 정보 포맷팅
        held_symbols_str = ", ".join(portfolio_summary.get('held_symbols_list', []))
        total_value_str = f"{portfolio_summary.get('total_portfolio_value_usdt', 0.0):.2f}"
        usdt_balance_str = f"{portfolio_summary.get('usdt_balance', 0.0):.2f}"
        num_pos_str = str(portfolio_summary.get('num_positions', 0))
        max_pos_str = str(portfolio_summary.get('max_positions', 'N/A'))

        context = f"""
        **Analyze Trading Suggestion for: {symbol}**

        *   **Time Since Last Check:** {elapsed_minutes:.0f} minutes
        *   **Prudence Context (Why this symbol group was chosen):** {prudence_reasoning}
        *   **Latest Technical Indicators ({symbol}, 1h):**
            *   Close (Regular Candle): {close_str}
            *   EMA(200): {ema200_str}
            *   StochRSI K(14,14,3,3): {stochrsi_k_str}
            *   StochRSI D(14,14,3,3): {stochrsi_d_str}
            *   Heikin Ashi Open: {ha_open_str}
            *   Heikin Ashi High: {ha_high_str}
            *   Heikin Ashi Low: {ha_low_str}
            *   Heikin Ashi Close: {ha_close_str}
            *   MACD(12,26,9): {macd_str}
            *   MACD Signal(9): {macds_str}
        *   **Current Wallet Status (Binance, specific to {symbol}):**
            *   Available {base_asset}: {base_bal_str}
            *   Available USDT (for this specific pair - may differ from total): {quote_bal_str}
        *   **Overall Portfolio Status:**
            *   Currently Held Symbols: {held_symbols_str if held_symbols_str else 'None'}
            *   Estimated Total Portfolio Value (USDT): {total_value_str}
            *   Total Available USDT: {usdt_balance_str}
            *   Number of Positions Held: {num_pos_str} / {max_pos_str}

        **Request:**
        Based ONLY on the provided context for {symbol} (Time Since Last Check, Prudence Context, Technical Indicators, Wallet Status for this pair, AND Overall Portfolio Status), provide a specific trading suggestion. Your suggestion MUST be in the following JSON format ONLY:
        ```json
        {{
            "symbol": "{symbol}",
            "decision": "BUY" | "SELL" | "HOLD",
            "reason": "Explain the rationale for {symbol} based ONLY on the provided data. Mention key indicators, prudence context relevance, elapsed time, AND portfolio status considerations (e.g., portfolio full, need cash, diversification opportunity).",
            "confidence": scale from 0 (low) to 1 (high),
            "next_check_minutes": <integer>, // Recommend wait time in minutes before next check (e.g., 15, 30, 60)
            "analysis_summary": "<string>" // Brief summary of analysis or focus for next check, potentially including portfolio impact
        }}
        ```
        Be concise. Focus strictly on the provided data for {symbol}. **Crucially, consider the Overall Portfolio Status when determining your decision and reason.** For example, if the portfolio is full (at max positions), avoid BUY suggestions unless the signal is exceptionally strong and potentially suggest SELL for weaker holdings. If USDT is low, prioritize SELL or HOLD. If the portfolio lacks diversification and this symbol offers it, mention it.
        """

        logging.info(f"Sending context for {symbol} (elapsed: {elapsed_minutes:.0f} min, portfolio included) to Bitcoin Trading Gemini...")
        response = chat_session.send_message(context)
        model_usage(response)

        # 응답 파싱 (JSON 형식 가정, symbol 필드 확인)
        suggestion = None
        try:
            json_str_match = re.search(r'```(?:json)?\n(\{.*?\})\n```', response.text, re.DOTALL | re.IGNORECASE)
            if json_str_match:
                json_str = json_str_match.group(1)
            else:
                json_str = response.text

            suggestion = json.loads(json_str)
            # Validate the response includes the correct symbol AND decision
            if isinstance(suggestion, dict) and suggestion.get('symbol') == symbol and suggestion.get('decision') in ["BUY", "SELL", "HOLD"]:
                logging.info(f"Received suggestion for {symbol}: {suggestion.get('decision')}")
                # 추가 필드 유효성 검사 (선택적이지만 권장)
                if not isinstance(suggestion.get('next_check_minutes'), int) or suggestion.get('next_check_minutes') <= 0:
                    logging.warning(f"Invalid or missing 'next_check_minutes' for {symbol}. Using default later.")
            else:
                 logging.warning(f"Received suggestion, but invalid format/content. Expected: {symbol}, Decision: BUY/SELL/HOLD. Got: {suggestion}")
                 suggestion = None # Discard invalid suggestion

        except json.JSONDecodeError as json_e:
            logging.error(f"Failed to decode Gemini response JSON for {symbol}: {json_e}. Response text: {response.text[:500]}")
            suggestion = None
        except Exception as parse_e:
             logging.error(f"Error parsing Gemini response for {symbol}: {parse_e}. Response text: {response.text[:500]}")
             suggestion = None

        return suggestion

    except Exception as e:
        logging.error(f"Error getting trade suggestion for {symbol} from Gemini: {e}")
        return None

# gem_sug = get_trade_suggestion # Keep alias for compatibility (이제 인자 개수가 다르므로 주의 필요)

# --- 거래 실행 함수 (기존 Coinone용) ---
def execute_market_buy(amount_krw):
    """주어진 KRW 금액만큼 BTC 시장가 매수를 실행합니다. (Coinone)"""
    if amount_krw < 5500: # 최소 주문 금액 확인
        logging.warning(f"Buy amount {amount_krw} KRW is below minimum (5500 KRW). Skipping buy.")
        return None

    payload = {
        'type': 'MARKET',
        'quote_currency': "KRW",
        'target_currency': "BTC",
        'side': 'BUY',
        "amount": amount_krw # KRW 금액
    }
    logging.info(f"Executing market buy for {amount_krw} KRW.")
    response_data = make_coinone_request(action="/v2.1/order", payload=payload)

    if response_data and response_data.get('result') == 'success':
        logging.info(f"Market buy successful: Order ID {response_data.get('order_id')}")
        # 성공 시 추가 로직 (텔레그램 알림 등) 가능
        send_telegram_message(f"✅ Market Buy Executed: {amount_krw} KRW")
        return response_data
    else:
        error_msg = response_data.get('error_msg', 'Unknown error') if response_data else 'No response'
        logging.error(f"Market buy failed: {error_msg}")
        send_telegram_message(f"❌ Market Buy FAILED: {amount_krw} KRW. Reason: {error_msg}")
        return None

def execute_market_sell(amount_btc):
    """주어진 BTC 수량만큼 시장가 매도를 실행합니다. (Coinone)"""
    try:
        amount_btc_float = float(amount_btc)
        if amount_btc_float <= 0:
            logging.warning("Sell amount must be positive. Skipping sell.")
            return None
    except ValueError:
        logging.error(f"Invalid BTC amount for selling: {amount_btc}")
        return None

    # 최소 주문 금액 확인 (BTC 수량 * 현재가 >= 5500 KRW)
    current_price = get_current_btc_price()
    if not current_price or (amount_btc_float * current_price < 5500):
        logging.warning(f"Estimated sell value ({amount_btc_float * (current_price or 0):.0f} KRW) is below minimum (5500 KRW). Skipping sell.")
        return None

    payload = {
        'type': 'MARKET',
        'quote_currency': "KRW",
        'target_currency': "BTC",
        'side': 'SELL',
        'qty': amount_btc_float # BTC 수량
    }
    logging.info(f"Executing market sell for {amount_btc_float:.8f} BTC.")
    response_data = make_coinone_request(action="/v2.1/order", payload=payload)

    # --- 매도 시 정보 로깅 및 알림 ---
    current_price = get_current_btc_price() # 현재가 다시 조회 (API 호출 줄이려면 이전 값 재사용 가능)
    balances = get_wallet_balances(['BTC']) # 매도 후 잔고 확인 위해 호출 또는 이전 상태 사용
    btc_avg_price = 0
    # btc_available_before = amount_btc_float # 판매된 수량은 입력값과 동일하다고 가정
    if balances and balances[0]:
        # 평단가는 매도 전 정보를 기준으로 해야 의미 있음 (현재 로직은 매도 후 정보)
        # 정확한 수익 계산을 위해서는 매도 실행 전 지갑 상태를 조회/저장해두어야 함
        # 임시로 매도 후 평단가 사용 (개선 필요)
        btc_avg_price = float(balances[0].get('average_price', 0))

    profit_percent = 0
    profit_krw = 0
    if btc_avg_price > 0 and current_price:
        # 수익률 계산 시점: 매도 시점의 현재가 vs 매수 평단가
        profit_percent = round((current_price - btc_avg_price) / btc_avg_price * 100, 2) if btc_avg_price else 0
        # 수익 금액: (매도 가격 - 평단가) * 매도 수량
        profit_krw = round((current_price - btc_avg_price) * amount_btc_float, 0) if btc_avg_price else 0

    sell_info_msg = f"""--- Market Sell Attempt Result ---
    Amount: {amount_btc_float:.8f} BTC
    Executed Price (Current): {current_price or 'N/A'} KRW
    Avg Buy Price (Post-Sell*): {btc_avg_price or 'N/A'} KRW (*Note: Based on post-sell balance, needs pre-sell data for accuracy)
    Est. Profit: {profit_percent}% ({profit_krw:.0f} KRW)
    """
    logging.info(sell_info_msg)
    # --- 정보 로깅 및 알림 끝 ---

    if response_data and response_data.get('result') == 'success':
        logging.info(f"Market sell successful: Order ID {response_data.get('order_id')}")
        # 성공 메시지에 수익 정보 포함
        send_telegram_message(f"✅ Market Sell Executed: {amount_btc_float:.8f} BTC.\n{sell_info_msg}")
        return response_data
    else:
        error_msg = response_data.get('error_msg', 'Unknown error') if response_data else 'No response'
        logging.error(f"Market sell failed: {error_msg}")
        # 실패 메시지에도 시도 정보 포함
        send_telegram_message(f"❌ Market Sell FAILED: {amount_btc_float:.8f} BTC. Reason: {error_msg}\nAttempt Info:\n{sell_info_msg}")
        return None

# --- 신규 바이낸스 거래 실행 함수 ---
def execute_binance_market_buy(symbol, quote_order_qty):
    """주어진 USDT 금액만큼 지정된 심볼을 시장가로 매수합니다. (Binance)"""
    if not binance_client:
        logging.error("Binance client not initialized. Cannot execute buy order.")
        return None
    if quote_order_qty <= 0:
        logging.warning(f"Buy amount ({quote_order_qty} USDT) must be positive. Skipping buy for {symbol}.")
        return None

    logging.info(f"Attempting Binance market buy for {symbol} with {quote_order_qty:.2f} USDT...")
    try:
        # 바이낸스 시장가 매수 주문 (USDT 사용량 기준)
        order = binance_client.create_order(
            symbol=symbol,
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quoteOrderQty=quote_order_qty
        )
        logging.info(f"✅ Binance Market Buy SUCCESSFUL for {symbol}: {order}")
        send_telegram_message(f"✅ Binance Market Buy: {symbol} with {quote_order_qty:.2f} USDT\nOrder Details: {order}")
        return order
    except Exception as e:
        logging.error(f"❌ Binance Market Buy FAILED for {symbol} ({quote_order_qty:.2f} USDT): {e}")
        send_telegram_message(f"❌ Binance Market Buy FAILED: {symbol} with {quote_order_qty:.2f} USDT\nReason: {e}")
        return None

def execute_binance_market_sell(symbol, quantity):
    """보유 중인 지정된 심볼의 수량 전체를 시장가로 매도합니다. (Binance)"""
    if not binance_client:
        logging.error("Binance client not initialized. Cannot execute sell order.")
        return None
    if quantity <= 0:
        logging.warning(f"Sell quantity ({quantity}) must be positive. Skipping sell for {symbol}.")
        return None

    logging.info(f"Attempting Binance market sell for {quantity:.8f} {symbol}...")
    try:
        # 바이낸스 시장가 매도 주문 (코인 수량 기준)
        # 주의: quantity는 최소 거래 단위를 맞춰야 함 (API 에러 발생 가능)
        # 여기서는 일단 받은 quantity 그대로 사용
        order = binance_client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        logging.info(f"✅ Binance Market Sell SUCCESSFUL for {symbol}: {order}")
        send_telegram_message(f"✅ Binance Market Sell: {quantity:.8f} {symbol}\nOrder Details: {order}")
        return order
    except Exception as e:
        logging.error(f"❌ Binance Market Sell FAILED for {symbol} ({quantity:.8f}): {e}")
        send_telegram_message(f"❌ Binance Market Sell FAILED: {quantity:.8f} {symbol}\nReason: {e}")
        return None

# --- 유틸리티 함수 ---
def write_chat_log(data):
    """주어진 데이터를 로그 파일(bit_log.jsonl)에 JSON Lines 형식으로 추가합니다."""
    if not data:
        logging.warning("No data provided to write_chat_log.")
        return

    try:
        # 로그 항목 생성 (Binance 데이터 포함)
        binance_balances = get_binance_balances() # 현재 잔고 조회
        log_entry = {
            "log_time": datetime.datetime.now().isoformat(), # 로그 기록 시간
            "role": "bit_AI_run", # 역할 명시
            "suggestion_data": data, # Gemini 제안 결과 전체
            "btc_balance": binance_balances.get('BTC', 0.0), # BTC 잔고
            "quote_balance": binance_balances.get('USDT', 0.0), # USDT 잔고
            "data_source": "Binance",
            "symbol": SYMBOL,
            "interval": CHART_INTERVAL
            # 여기에 필요하다면 다른 메타데이터 추가 가능
        }
        # 파일에 쓰기 (jsonl 형식, append 모드)
        with open(BIT_LOG_FILE, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')
        logging.info(f"Successfully wrote log entry to {BIT_LOG_FILE}.")
    except Exception as e:
        logging.error(f"Error writing to log file {BIT_LOG_FILE}: {e}")

async def send_telegram_message_async(text):
    "텔레그램으로 비동기 메시지를 보냅니다."
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logging.warning("Telegram Bot Token or Chat ID not found. Skipping notification.")
        return
    try:
        bot = telegram.Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, text=text)
        logging.info(f"Sent Telegram message (async): {text[:50]}...")
    except Exception as e:
        logging.error(f"Error sending Telegram message (async): {e}")

def send_telegram_message(text):
    "텔레그램으로 동기 메시지를 보냅니다."
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logging.warning("Telegram Bot Token or Chat ID not found. Skipping notification.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown' # 또는 필요에 따라 'HTML'
    }
    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        if response.json().get('ok'):
            logging.info(f"Sent Telegram message: {text[:50]}...")
        else:
            logging.error(f"Telegram API error: {response.json().get('description')}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending Telegram message: {e}")

# --- Signal Determination ---
def determine_signal(df):
    """Determines Buy/Sell signal based on the latest indicator values."""
    if df is None or df.empty:
        logging.warning("DataFrame is empty, cannot determine signal.")
        return None

    latest_data = df.iloc[-1] # Get the most recent row

    # --- Define Strategy Conditions ---
    # Example Strategy (can be much more complex):
    # Buy Signal: RSI < 30 and Stochastic %K < 20 and Close > EMA200 and MACD line > Signal line
    # Sell Signal: RSI > 70 and Stochastic %K > 80 and Close < EMA200 and MACD line < Signal line

    # Check for NaN values in required indicators for the latest data point
    # Ensure column names match the ones generated by pandas_ta (e.g., RSI_14, STOCHk_14_3_3, etc.)
    required_indicators = ['RSI_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3', 'ema200', 'MACD_12_26_9', 'MACDs_12_26_9']
    if not all(indicator in latest_data.index for indicator in required_indicators) or latest_data[required_indicators].isnull().any():
        logging.warning(f"Latest data missing required indicators or contains NaN. Cannot determine signal. Available: {latest_data.index.tolist()} Data: {latest_data.filter(items=required_indicators)}")
        return None # Cannot determine signal if indicators are missing or NaN

    rsi = latest_data['RSI_14']
    stoch_k = latest_data['STOCHk_14_3_3']
    stoch_d = latest_data['STOCHd_14_3_3']
    close_price = latest_data['Close']
    ema200 = latest_data['ema200']
    macd_line = latest_data['MACD_12_26_9'] # MACD line
    signal_line = latest_data['MACDs_12_26_9'] # Signal line

    # Log latest values for debugging
    logging.info(f"Latest Data for Signal: Close={close_price:.2f}, RSI={rsi:.2f}, StochK={stoch_k:.2f}, StochD={stoch_d:.2f}, EMA200={ema200:.2f}, MACD={macd_line:.4f}, Signal={signal_line:.4f}")

    # --- Buy Conditions ---
    # Stochastics: K line crosses D line upwards in oversold territory
    # Simple check: K < 20 (oversold)
    stoch_buy_condition = stoch_k < 20

    # MACD: MACD line crosses Signal line upwards (MACD > Signal)
    macd_buy_condition = macd_line > signal_line

    # RSI: Oversold condition
    rsi_buy_condition = rsi < 30

    # Trend Filter: Price above EMA 200
    trend_buy_condition = close_price > ema200

    # Combine conditions (Example: RSI & Stochastic confirm oversold, trend is up, MACD confirms momentum)
    if rsi_buy_condition and stoch_buy_condition and trend_buy_condition and macd_buy_condition:
        logging.info("BUY SIGNAL DETECTED")
        return "BUY"

    # --- Sell Conditions ---
    # Stochastics: K line crosses D line downwards in overbought territory
    # Simple check: K > 80 (overbought)
    stoch_sell_condition = stoch_k > 80

    # MACD: MACD line crosses Signal line downwards (MACD < Signal)
    macd_sell_condition = macd_line < signal_line

    # RSI: Overbought condition
    rsi_sell_condition = rsi > 70

    # Trend Filter: Price below EMA 200
    trend_sell_condition = close_price < ema200

    # Combine conditions
    if rsi_sell_condition and stoch_sell_condition and trend_sell_condition and macd_sell_condition:
        logging.info("SELL SIGNAL DETECTED")
        return "SELL"

    logging.info("No Buy/Sell signal detected based on the current strategy.")
    return None # No signal

# --- Logging Functions (수정) ---
def log_trade_decision(decision_data, symbol="UNKNOWN", success=None, error_message=None, order_details=None, side=None, quantity=None):
    """Logs the trading decision data (including symbol, side, quantity, success, etc.) to TRADE_LOG_FILE."""
    log_entry = {
        "log_time": datetime.datetime.now(KST).isoformat(),
        "symbol": symbol,
        "side_attempted": side, # 추가: 시도한 거래 방향
        "quantity_attempted_or_adjusted": str(quantity) if quantity is not None else None, # 추가: 시도/조정된 수량 (문자열로)
        "success": success, # 추가: 실행 성공 여부
        "error_message": error_message, # 추가: 오류 메시지
        "order_details": order_details, # 추가: 성공 시 주문 상세
        "trade_decision": decision_data # 기존 AI 결정 데이터
    }
    try:
        with open(TRADE_LOG_FILE, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False, default=str) # Decimal 등 직렬화 위해 default=str 추가
            f.write('\n')
        logging.info(f"Trade execution log for {symbol} saved to {TRADE_LOG_FILE}")
    except Exception as e:
        logging.error(f"Failed to log trade execution for {symbol} to {TRADE_LOG_FILE}: {e}")

# --- Main Execution ---
# if __name__ == '__main__':
#     # 1. Fetch and Analyze Binance Chart Data
#     chart_df = get_binance_chart(symbol=SYMBOL, interval=Client.KLINE_INTERVAL_1HOUR, limit=DATA_LIMIT)
#     latest_indicators = None
#     signal = None
#     latest_rsi = None
#     indicator_df = None # Initialize indicator_df
#
#     if chart_df is not None:
#         indicator_df = get_technical_indicators(chart_df)
#         if indicator_df is not None:
#             print("\n--- Latest Indicator Data (Binance) ---")
#             latest_indicators = indicator_df.iloc[-1].filter(regex='^(?!geom_)')
#             print(latest_indicators)
#
#             signal = determine_signal(indicator_df)
#             print(f"\nTrade Signal based on TA: {signal}") # Clarify signal source
#
#             latest_rsi = latest_indicators.get('RSI_14', None)
#             rsi_str = f"{latest_rsi:.2f}" if latest_rsi is not None else "N/A"
#             print(f"Latest RSI: {rsi_str}")
#
#         else:
#             print("\nFailed to calculate indicators.")
#     else:
#         print("\nFailed to get chart data.")
#
#     # 2. Get Binance Balances (Requires API Key)
#     print("\n--- Binance Wallet Balances ---")
#     btc_balance, quote_balance = get_binance_balances()
#     if btc_balance is not None and quote_balance is not None:
#         print(f"Available BTC: {btc_balance}")
#         print(f"Available Quote (USDT): {quote_balance}")
#     else:
#         print("Could not retrieve Binance wallet balances (API keys might be missing or invalid).")
#
#     # 3. Get Prudence Index (Use the value obtained previously)
#     print("\n--- Prudence Index (Previously Obtained) ---")
#     # Replace with the actual call if needed: prudence_result = get_today_prudence()
#     prudence_result = {'prudence': 60, 'reason': 'Obtained from previous Prud_AI run'} # Using previous result
#     print(json.dumps(prudence_result, indent=2))
#
#     # 4. Get Gemini Trading Suggestion
#     print("\n--- Getting Gemini Trading Suggestion ---")
#     gemini_suggestion = None
#     if prudence_result and indicator_df is not None and btc_balance is not None and quote_balance is not None:
#         bit_instruction_text = get_instructions('./Bitcoin Gemini Instruction.md')
#         if bit_instruction_text:
#             bit_chat_session = gen_bit_model(bit_instruction_text)
#             if bit_chat_session:
#                 # Pass balances to the suggestion function
#                 gemini_suggestion = get_trade_suggestion(bit_chat_session, prudence_result, indicator_df, btc_balance, quote_balance, SYMBOL, 0, {})
#                 print("\n=== Gemini Suggestion ===")
#                 print(json.dumps(gemini_suggestion, indent=2, ensure_ascii=False) if gemini_suggestion else "Failed to get suggestion from Gemini.")
#             else:
#                 print("Failed to initialize Gemini chat session for bit_AI.")
#         else:
#             print("Could not load Bitcoin Gemini instruction file.")
#     else:
#         print("Skipping Gemini suggestion due to missing prudence index, indicator data, or balance data.")
#
#     # 5. Log interaction (Example)
#     log_entry = {
#         "timestamp": datetime.datetime.now(KST).isoformat(),
#         "role": "bit_AI_run",
#         "signal_TA": signal, # Technical Analysis Signal
#         "latest_rsi": rsi_str if 'rsi_str' in locals() else None,
#         "prudence_index": prudence_result.get('prudence') if prudence_result else None,
#         "gemini_suggestion": gemini_suggestion, # Log Gemini's suggestion
#         "btc_balance": btc_balance,
#         "quote_balance": quote_balance,
#         "data_source": "Binance",
#         "symbol": SYMBOL,
#         "interval": CHART_INTERVAL
#     }
#     write_chat_log(log_entry)
#
#     print("\nScript execution finished.")

# --- Trading AI Class --- #
class Bit_AI:
    """Handles Binance API interactions, trade execution, and related logic."""
    def __init__(self):
        """Initializes the Bit_AI class, primarily setting up the Binance client."""
        # Use the module-level binance_client
        self.client = binance_client
        if not self.client:
            logging.error("Bit_AI initialized, but Binance client is not available. Trading functionalities will fail.")
            # raise ConnectionError("Binance client failed to initialize inside Bit_AI.") # Or raise error

    def _adjust_quantity_to_step(self, quantity, step_size_str):
        """Adjusts quantity down to the nearest multiple of step_size using Decimal."""
        # (이전에 추가했던 _adjust_quantity_to_step 함수 내용을 여기에 붙여넣기)
        try:
            quantity_d = Decimal(str(quantity))
            step_size_d = Decimal(step_size_str)

            if step_size_d <= 0:
                logging.warning(f"Invalid step_size '{step_size_str}' for quantity adjustment. Returning original quantity as string.")
                return str(quantity) # Return original as string

            # Calculate adjusted quantity by flooring to the step size
            adjusted_quantity_d = (quantity_d / step_size_d).to_integral_value(rounding=ROUND_DOWN) * step_size_d

            # Determine the number of decimal places from step_size
            step_tuple = step_size_d.as_tuple()
            if step_tuple.exponent >= 0: # Integer step size (e.g., 1, 10)
                decimals = 0
            else:
                # Use max(0, ...) to handle cases like step_size='1.0'
                decimals = abs(step_tuple.exponent)
                # Refinement: Ensure trailing zeros in step_size are considered
                if '.' in step_size_str:
                    decimals = len(step_size_str.split('.')[1])

            # Format the adjusted quantity as a string
            return "{:.{prec}f}".format(adjusted_quantity_d, prec=decimals)
        except Exception as e:
             logging.error(f"Error adjusting quantity {quantity} with step size {step_size_str}: {e}")
             return None # Indicate failure

    def execute_trade(self, symbol, side, quantity, decision_data):
        """Executes a trade based on the AI decision, adjusting for LOT_SIZE."""
        # (이전에 수정했던 execute_trade 함수 내용을 여기에 붙여넣기, self 추가)
        if not self.client:
            logging.error("Binance client not available in Bit_AI instance. Cannot execute trade.")
            self._log_trade_decision(symbol, side, quantity, decision_data, success=False, error_message="Binance client not available")
            return False

        logging.info(f"Attempting to execute {side} trade for {quantity} {symbol}")
        adjusted_quantity_str = None # Initialize for logging

        try:
            # 1. Get Symbol Info for Filters
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                logging.error(f"Could not retrieve symbol info for {symbol}. Aborting trade.")
                self._log_trade_decision(symbol, side, quantity, decision_data, success=False, error_message="Could not get symbol info")
                return False

            filters = {f['filterType']: f for f in symbol_info['filters']}

            # 2. Apply LOT_SIZE filter
            lot_size_filter = filters.get('LOT_SIZE')
            adjusted_quantity_str = str(quantity) # Default to original if filter not found
            min_qty_str = "0"

            if lot_size_filter:
                min_qty_str = lot_size_filter.get('minQty')
                step_size_str = lot_size_filter.get('stepSize')
                logging.info(f"Applying LOT_SIZE filter for {symbol}: minQty={min_qty_str}, stepSize={step_size_str}")

                # Use the instance method for adjustment
                adjusted_quantity_str_maybe = self._adjust_quantity_to_step(quantity, step_size_str)
                if adjusted_quantity_str_maybe is None:
                     logging.error(f"Failed to adjust quantity for {symbol}. Aborting trade.")
                     self._log_trade_decision(symbol, side, quantity, decision_data, success=False, error_message="Quantity adjustment failed")
                     return False # Adjustment failed
                adjusted_quantity_str = adjusted_quantity_str_maybe # Assign if adjustment successful

                logging.info(f"Adjusted quantity for {symbol}: {quantity} -> {adjusted_quantity_str}")
            else:
                 logging.warning(f"LOT_SIZE filter not found for {symbol}. Using original quantity.")

            # Convert to float/Decimal for comparison AFTER adjustment
            adjusted_quantity_val = Decimal(adjusted_quantity_str)
            min_qty_val = Decimal(min_qty_str)

            # Check minQty
            if adjusted_quantity_val < min_qty_val:
                 logging.error(f"Adjusted quantity {adjusted_quantity_str} is less than minQty {min_qty_str} for {symbol}. Aborting trade.")
                 self._log_trade_decision(symbol, side, adjusted_quantity_str, decision_data, success=False, error_message=f"Quantity < minQty ({min_qty_str})")
                 return False

            # Check if adjusted quantity is zero
            if adjusted_quantity_val <= 0:
                logging.error(f"Adjusted quantity is zero or less ({adjusted_quantity_str}) for {symbol}. Aborting trade.")
                self._log_trade_decision(symbol, side, adjusted_quantity_str, decision_data, success=False, error_message="Adjusted quantity is zero or less")
                return False

            # TODO: Apply PRICE_FILTER, MIN_NOTIONAL filters

            # 3. Create Order with adjusted quantity
            order_params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': 'MARKET',
                'quantity': adjusted_quantity_str # Use adjusted string quantity
            }
            logging.info(f"Placing order with parameters: {order_params}")
            order = self.client.create_order(**order_params)

            logging.info(f"Order placed successfully: {order}")
            self._log_trade_decision(symbol, side, adjusted_quantity_str, decision_data, success=True, order_details=order)
            return True

        except BinanceAPIException as e:
            logging.error(f"Binance API Error executing trade for {symbol}: {e}")
            self._log_trade_decision(symbol, side, adjusted_quantity_str or str(quantity), decision_data, success=False, error_message=str(e))
            return False
        except Exception as e:
            logging.exception(f"Unexpected error executing trade for {symbol}:")
            self._log_trade_decision(symbol, side, adjusted_quantity_str or str(quantity), decision_data, success=False, error_message=f"Unexpected error: {e}")
            return False

    def get_min_order_quantity(self, symbol):
        """Retrieves the minimum order quantity (minQty) for a given symbol."""
        # (이전에 추가했던 get_min_order_quantity 함수 내용을 여기에 붙여넣기, self 추가)
        if not self.client:
            logging.error("Binance client not available in Bit_AI instance. Cannot get minQty.")
            return None
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                logging.error(f"Could not retrieve symbol info for {symbol} to get minQty.")
                return None

            filters = {f['filterType']: f for f in symbol_info['filters']}
            lot_size_filter = filters.get('LOT_SIZE')

            if lot_size_filter:
                min_qty_str = lot_size_filter.get('minQty')
                if min_qty_str:
                    logging.debug(f"Retrieved minQty for {symbol}: {min_qty_str}")
                    return Decimal(min_qty_str)
                else:
                    logging.warning(f"minQty not found within LOT_SIZE filter for {symbol}.")
                    return None
            else:
                 logging.warning(f"LOT_SIZE filter not found for {symbol}. Cannot determine minQty.")
                 return None

        except BinanceAPIException as e:
            logging.error(f"Binance API Error retrieving minQty for {symbol}: {e}")
            return None
        except Exception as e:
            logging.exception(f"Unexpected error retrieving minQty for {symbol}:")
            return None

    def _log_trade_decision(self, symbol, side, quantity, decision_data, success, error_message=None, order_details=None):
        """Logs the trade execution attempt and outcome."""
        # This is a placeholder. The actual log_trade_decision is a module-level function.
        # We call the module-level function here.
        # Pass the actual quantity attempted or adjusted.
        log_trade_decision(decision_data=decision_data, symbol=symbol, success=success, error_message=error_message, order_details=order_details, side=side, quantity=quantity)
