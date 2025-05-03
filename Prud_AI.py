#!/usr/bin/env python
# coding: utf-8

import google.generativeai as genai
# import KEYS
import os
from dotenv import load_dotenv
load_dotenv()

import json
# import sqlite3 as sql # DB 사용 안 함
import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
import time # time 추가
import re # Import re for parsing relative time

# bit_AI 모듈의 함수 직접 임포트 대신 로깅 함수 사용 (의존성 감소)
# from bit_AI import model_usage

# 로깅 설정 (bit_AI와 동일한 설정 사용 또는 별도 설정)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 상수 정의
REQUEST_TIMEOUT = 10 # API 요청 타임아웃 (초)
# CHAT_LOG_FILE = "chat_log.jsonl" # bit_AI와 동일한 로그 파일 사용 -> 이제 사용 안 함
PRUD_LOG_FILE = "prud_log.jsonl" # Prudence AI 전용 로그 파일
NEWS_COUNT = 10 # 가져올 뉴스 기사 수

# --- Helper Function for Relative Time Parsing ---
def parse_relative_time(time_str):
    """Converts relative time strings like 'X minutes ago' to datetime objects."""
    if not time_str:
        return None
    
    now = datetime.datetime.now()
    time_str = time_str.lower().strip()

    try:
        if 'minute' in time_str:
            minutes = int(re.search(r'\d+', time_str).group())
            return now - datetime.timedelta(minutes=minutes)
        elif 'hour' in time_str:
            hours = int(re.search(r'\d+', time_str).group())
            return now - datetime.timedelta(hours=hours)
        elif 'day' in time_str:
            days = int(re.search(r'\d+', time_str).group())
            return now - datetime.timedelta(days=days)
        # Add more specific cases if needed (e.g., 'yesterday', specific date formats)
        else:
            # Attempt to parse as a standard date format if possible, otherwise return None
            # This part might need refinement based on actual non-relative date formats observed
            # For now, assume only relative times are used or return None
            logging.warning(f"Could not parse relative time: {time_str}")
            return None
    except Exception as e:
        logging.error(f"Error parsing relative time string '{time_str}': {e}")
        return None

def gen_pru_model(instruction):
    "Prudence Gemini 모델을 초기화합니다."
    try:
        genai.configure(api_key=os.getenv("Gemini"))
        # Generation Config - Prudence 모델에 맞는 설정 조정 필요
        generation_config = {
            "temperature": 0.8, # 약간 더 창의적인 답변 유도?
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro-preview-03-25", # 모델 이름 확인 및 선택
            generation_config=generation_config,
            system_instruction=instruction,
        )
        logging.info("Prudence Gemini model initialized.")
        return model.start_chat()
    except Exception as e:
        logging.error(f"Failed to initialize Prudence Gemini model: {e}")
        return None

def log_gemini_usage(response):
    "Gemini API 사용량 정보를 로깅합니다."
    try:
        logging.info(f"Prudence Gemini Usage Metadata: {response.usage_metadata}")
    except Exception as e:
        logging.error(f"Error logging Prudence Gemini usage info: {e}")

def get_instructions(file_path):
    "지침 파일을 읽어옵니다."
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        logging.error(f"Instruction file not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while reading the instruction file {file_path}: {e}")
        return None

def get_recent_crypto_news(count=NEWS_COUNT):
    """Fetches recent Bitcoin news from crypto.news/tag/bitcoin/."""
    news_list = []
    url = "https://crypto.news/tag/bitcoin/" # New URL
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }
    logging.info(f"Attempting to fetch crypto news from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Updated selector based on provided HTML structure
        article_elements = soup.select("div.post-loop.post-loop--style-horizontal")

        if not article_elements:
             logging.warning("Could not find articles with selector 'div.post-loop.post-loop--style-horizontal'. HTML structure might have changed or content is loaded dynamically.")
             return []

        logging.info(f"Found {len(article_elements)} potential article items using 'div.post-loop.post-loop--style-horizontal'.")

        for article in article_elements[:count]:
            news_item = {"datetime": None, "title": "N/A", "paragraph": "(Content fetching disabled)", "url": None, "provider": "crypto.news"} # Provider set
            try:
                # Updated selectors for title, link, and time
                title_element = article.select_one('p.post-loop__title > a')
                time_element = article.select_one('time.post-loop__date')

                if title_element:
                    news_item['title'] = title_element.text.strip()
                    link = title_element.get('href')
                    # Ensure the link is absolute (crypto.news seems to use absolute links)
                    news_item['url'] = link # Assuming absolute URLs

                if time_element:
                    # Get the datetime attribute directly
                    datetime_str = time_element.get('datetime')
                    news_item['datetime'] = datetime_str # Store as ISO format string
                else:
                    logging.warning(f"Could not find time element for article: {news_item.get('title')}")

            except Exception as e_item:
                logging.warning(f"Error processing a news item: {e_item} for article: {article.prettify()[:200]}...")

            # Only add if we have a title and URL
            if news_item['title'] != "N/A" and news_item['url']:
                news_list.append(news_item)

        logging.info(f"Successfully processed {len(news_list)} news items from crypto.news.")
        return news_list

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch news list from crypto.news: {e}. Returning empty list.")
        return []
    except Exception as e:
        logging.error(f"Error parsing news list page from crypto.news: {e}")
        return []

def get_fear_greed_index():
    """Fetches the Fear & Greed Index from blockstreet.co.kr."""
    url = "https://www.blockstreet.co.kr/fear-greed"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }
    logging.info(f"Fetching Fear & Greed Index from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Updated selector based on the provided HTML structure for "Now" value
        # Find the list item containing the text "Now", then find the span with class starting with "greed-background-" within it.
        fgi_element = soup.select_one('li:has(span:-soup-contains("Now")) span[class*="greed-background-"]')

        if fgi_element and fgi_element.text.isdigit():
            fgi_value = int(fgi_element.text)
            logging.info(f"Successfully fetched Fear & Greed Index: {fgi_value}")
            # Use datetime.datetime.now() here to correctly call the method
            return {"date": datetime.datetime.now().strftime("%Y-%m-%d"), "FGI": fgi_value}
        else:
            # Log the selector used and potentially part of the HTML if failed
            logging.warning(f"Could not find or parse Fear & Greed Index value using selector: 'li:has(span:-soup-contains(\"Now\")) span[class*=\"greed-background-\"]'. Check site structure.")
            # Optional: Log relevant part of soup
            # historical_values_section = soup.select_one('h3:has(span:-soup-contains("Historical Values")) + ul')
            # logging.debug(f"Historical Values HTML: {historical_values_section.prettify()[:500] if historical_values_section else 'Not Found'}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch Fear & Greed Index: {e}")
        return None
    except Exception as e:
        logging.error(f"Error parsing Fear & Greed Index page: {e}")
        return None

def get_recent_chat_logs(days=1):
    """채팅 로그 읽기 기능은 현재 비활성화 상태입니다."""
    logging.warning("get_recent_chat_logs is disabled.")
    return [] # 비활성화 유지

# --- Function to get symbols to trade --- #
def get_symbols_to_trade(chat_session):
    """Fetches market data and asks Prudence AI to suggest symbols to trade."""
    if not chat_session:
        logging.error("Symbol suggestion requested but chat session is not initialized.")
        return None

    # 데이터 수집 (뉴스, FGI)
    news_data = get_recent_crypto_news(count=NEWS_COUNT)
    fgi_data = get_fear_greed_index()
    # chat_log_data = get_recent_chat_logs() # 채팅 로그는 현재 비활성화

    # 프롬프트 구성 (새로운 지침 기반)
    today_date_str = str(datetime.date.today())
    prompt = f"""## 현재 시장 및 컨텍스트 정보:

**1. 최신 암호화폐 뉴스 (최대 {NEWS_COUNT}개):**
```json
{json.dumps(news_data, indent=2, ensure_ascii=False) if news_data else "N/A"}
```

**2. 현재 공포-탐욕 지수(FGI):**
```json
{json.dumps(fgi_data, indent=2) if fgi_data else "N/A"}
```

**요청:**
오늘 날짜({today_date_str})를 기준으로, 위 정보(뉴스, FGI)를 종합적으로 분석하여 오늘 단기 트레이딩에 가장 유망해 보이는 USDT 페어 알트코인 3~5개를 선정해주세요.
- 선정 이유를 간략하게 설명해주세요.
- 이유에는 각 정보(뉴스 동향 요약, FGI 해석)가 어떻게 반영되었는지 포함해주세요.

**응답은 다음 JSON 형식을 따라야 합니다:**
```json
{{
  "date": "{today_date_str}",
  "symbols_to_trade": ["SYMBOL1USDT", "SYMBOL2USDT", "SYMBOL3USDT"],
  "reason": "<종합적인 분석 및 선정 이유>"
}}
```
"""

    try:
        logging.info("Preparing to send data to Prudence Gemini for symbol suggestion...")
        prompt_cleaned = " ".join(prompt.split())
        logging.debug(f"Sending prompt (first 200 chars): {prompt_cleaned[:200]}...")
        response = chat_session.send_message(prompt_cleaned)
        logging.info("Received response from Prudence Gemini.")
        log_gemini_usage(response)
        logging.debug(f"Prudence Gemini raw response text: {response.text}")

        # JSON 응답 파싱 (새로운 형식에 맞게)
        suggestion = None
        try:
            # Try to extract JSON from markdown block first
            json_str_match = re.search(r'```(?:json)?\n(\{.*?\})\n```', response.text, re.DOTALL | re.IGNORECASE)
            if json_str_match:
                json_str = json_str_match.group(1)
                logging.debug("Extracted JSON from markdown block.")
            else:
                json_str = response.text # Assume direct JSON
                logging.debug("Attempting to parse response text directly as JSON.")

            suggestion = json.loads(json_str)
            # Basic validation for the new structure
            if isinstance(suggestion, dict) and 'symbols_to_trade' in suggestion and isinstance(suggestion['symbols_to_trade'], list):
                logging.info(f"Received symbol suggestions from Gemini: {suggestion.get('symbols_to_trade')}")
                # --- Log the symbol suggestion --- #
                log_prudence_data(suggestion) # Log the new data structure
                return suggestion
            else:
                 logging.error(f"Parsed JSON lacks required 'symbols_to_trade' list: {suggestion}")
                 return None

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding Prudence Gemini suggestion JSON: {e}. Response text: {response.text}")
            return None
        except Exception as parse_e:
            logging.error(f"Error processing parsed suggestion: {parse_e}")
            return None

    except Exception as e:
        logging.error(f"Error getting symbol suggestion from Prudence Gemini: {e}")
        return None

# --- Logging Function (Updated) --- #
def log_prudence_data(prudence_output_data):
    """Logs the Prudence AI output (symbol list and reason) to JSONL file."""
    log_entry = {
        "log_time": datetime.datetime.now().isoformat(),
        "role": "Prud_AI_symbol_suggestion", # 역할 명시
        "prudence_data": prudence_output_data # 받은 결과 그대로 저장 (이제 symbols_to_trade 포함)
    }
    try:
        with open(PRUD_LOG_FILE, 'a', encoding='utf-8') as f:
            json.dump(log_entry, f, ensure_ascii=False)
            f.write('\n')
        logging.info(f"Prudence symbol suggestion logged to {PRUD_LOG_FILE}")
    except Exception as e:
        logging.error(f"Error logging prudence suggestion to {PRUD_LOG_FILE}: {e}")

# Rename the main function for clarity, keep the old name for compatibility if needed by main.py initially
gem_pru_sug = get_symbols_to_trade

if __name__ == '__main__':
    # --- Updated Test Code --- #
    print("\n--- Getting Prudence AI Symbol Suggestions ---")
    pru_instruction_text = get_instructions('./Prudence Gemini Instruction.md')
    if pru_instruction_text:
        test_chat_session = gen_pru_model(pru_instruction_text)
        if test_chat_session:
            suggestion = get_symbols_to_trade(test_chat_session) # Call the renamed function
            print("\n=== Prudence Suggestion ===")
            print(json.dumps(suggestion, indent=2, ensure_ascii=False) if suggestion else "Failed to get suggestion.")
        else:
             print("Failed to initialize Prudence chat session.")
    else:
        print("Could not load Prudence instruction file.")

# Remove old/unused functions
# def write_prudence(data): ...
# def get_prudence(): ...
# def bring_fear_greed(): ...
# def write_fear_greed(): ...
# def get_trans_record(): ...
