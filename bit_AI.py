import google.generativeai as genai
import requests as rqs
import json

import base64
import hashlib
import hmac
import uuid
import httplib2
from pandas_ta import ema
from pandas_ta import stochrsi
from pandas_ta.candles import ha as hei

# import KEYS
from dotenv import load_dotenv
import os
load_dotenv()

import sqlite3 as sql
from bs4 import BeautifulSoup
import requests as rqs
import datetime
import pandas as pd

import telegram
import asyncio

# related to Gemini
def model_usage():
    model_info = genai.get_model("models/gemini-1.5-pro-002")
    print(f"{model_info.output_token_limit}")
    print(model_info.output_token_limit,"\n",response.usage_metadata)
    asyncio.run(tel(f"{model_info.output_token_limit} | {response.usage_metadata}"))
    
def gen_bit_model(instruction):
    genai.configure(api_key = os.getenv("Gemini"))

    # Create the model
    generation_config = {
    "temperature": 0.75,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-002",
    generation_config=generation_config,
    system_instruction=instruction,
    )
    return model.start_chat()

def get_btc():
    url = "https://api.coinone.co.kr/public/v2/chart/KRW/BTC?interval=1h&size=300"
    header = {"accept": "application/json"}
    bit_response = rqs.get(url, headers=header)
    return bit_response.text

def get_tech_indi():
    # If you get an error here, You should modify the pandas_ta's py files
    chart = json.loads(get_btc(), strict = False)['chart']
    df = pd.DataFrame(chart).astype('float64')
    ha = hei(
            open_ = df['open'],
            high = df['high'],
            low = df['low'],
            close = df['close']
            )
    ha[['stochestic_k', 'stochestic_d']] = stochrsi(close = df['close'])
    ha['ema200'] = ema(df['close'], length=200)
    ha = pd.concat([df[['timestamp', 'target_volume', 'quote_volume']], ha], axis=1).fillna(0)
    # print(ha['ema200'][-1], ha['HA_close'][-1])
    return json.dumps(ha.to_dict('list'))

def gem_sug(chat_session, prudence):
    chart = get_tech_indi()
    if len(chart) > 1:
        response = chat_session.send_message(f'{chart} {prudence} {get_cur_status()}')
        return json.loads(response.text, strict = False)
    else:
        print("chart error")
        return None
    # print(str(get_tech_indi()) + str(prudence) + str(get_cur_status()))

def get_cur_status():
    Krw = wallet('KRW')[0]
    Btc = wallet('BTC')[0]

    krw_wallet = round(float(Krw['available']) * (1 - 0.2 / 100), 0)
    btc_wallet = float(Btc['available'])
    position = Btc['average_price'] if btc_wallet != 0 else 0
    profit = round((get_btc_index() - float(position)) / float(position) * 100, 2) if float(position) > 0 else 0

    return json.dumps({
        "KRW_wallet" : krw_wallet,
        "BTC_Wallet" : btc_wallet,
        "position" : position,
        "profit" : profit,
    })

def get_encoded_payload(payload):
    payload['nonce'] = str(uuid.uuid4())

    dumped_json = json.dumps(payload)
    encoded_json = base64.b64encode(bytes(dumped_json, 'utf-8'))
    return encoded_json

def get_signature(encoded_payload):
    signature = hmac.new(bytes(os.getenv("Secret"), 'utf-8'), encoded_payload, hashlib.sha512)
    return signature.hexdigest()

def get_response(action, payload):
    url = '{}{}'.format('https://api.coinone.co.kr/', action)

    encoded_payload = get_encoded_payload(payload)

    headers = {
        'Content-type': 'application/json',
        'X-COINONE-PAYLOAD': encoded_payload,
        'X-COINONE-SIGNATURE': get_signature(encoded_payload),
    }
    http = httplib2.Http()
    response, content = http.request(url, 'POST', headers=headers)
    return content

def wallet(currs): 
    """returns balances, available, limit, average_price of given currencies
    type = list of dicts"""
    currs = currs.split(' ')
    from_wallet = json.loads(get_response(action='/v2.1/account/balance', 
                               payload={'access_token': os.getenv("Access"),
                                        'currencies': currs}), strict = False)
    if from_wallet['result'] != "success":
        print("Something went wrong", from_wallet['error_msg'])
    else:
        return from_wallet['balances']

def market_buy(amount):
    """Buy an amount of BTC"""
    return get_response(action="/v2.1/order", payload={'type' : 'MARKET',
                                                       'access_token' : os.getenv("Access"),
                                                       'quote_currency' : "KRW",
                                                       'target_currency' : "BTC",
                                                       'side' : 'BUY',
                                                       "amount" : amount})

def stop_sell(curr, lp_max, amount):
    # For future use
    """Set a stop price to sell.
    curr: currency to sell
    lp_max: maximum of loss / profit percentage (ex. 3.0)
    qty: A ratio amount to sell (ex. 50(%))"""
    perc = wallet(curr)[0]['average_price'] / 100 * lp_max
    return get_response(action="/v2.1/order",payload={'type' : 'STOP_LIMIT',
                                                       'access_token' : os.getenv("Access"),
                                                       'quote_currency' : "KRW",
                                                       'target_currency' : "BTC",
                                                       'side' : 'SELL', 
                                                       'price' : perc,
                                                       'trigger_price' : perc,
                                                       'qty' : wallet(curr)[0]['available'] / 100 * amount})

def market_sell(curr, amount):
    """Sell a ratio amount of currency. And write into transaction record
    curr = currency
    amount = ratio amount to sell (ex. 50(%))"""
    available = float(wallet('BTC')[0]['available'])
    avg_btc = float(wallet('BTC')[0]['average_price']) if available > 0 else 0
    with sql.connect('./Record.db') as dbop:
        dbcs = dbop.cursor()
        if available > 0:
            dbcs.execute("INSERT INTO TRANSRECORD VALUES (?, ?, ?);", (datetime.datetime.now(),          # time
                                                           round((get_btc_index() - avg_btc) / avg_btc * 100, 2),  # profit
                                                           int(round((get_btc_index() - avg_btc) * float(available), 0)))) # amount
        dbop.commit()
    send_tel(f"""==== Sold BTC ====
             Amount:\t{amount}
             Profit:\t{round((get_btc_index() - avg_btc) / avg_btc * 100, 2)}
             In KRW:\t{ round((get_btc_index() - avg_btc) * float(available), 0)}
             Price:\t{get_btc_index()}
             AVG:\t{avg_btc}""")
    
    return get_response(action="/v2.1/order",payload={'access_token' : os.getenv("Access"),
                                                       'quote_currency' : "KRW",
                                                       'target_currency' : "BTC",
                                                       'type' : 'MARKET', 
                                                       'side' : 'SELL', 
                                                       'qty' : amount})
 
def get_btc_index():
    url = "https://api.coinone.co.kr/public/v2/ticker_utc_new/KRW/BTC?additional_data=true"
    headers = {"accept": "application/json"}
    response = rqs.get(url, headers=headers)
    return float(json.loads(response.text, strict = False)['tickers'][0]['last'])

# related to DB
def get_today_prudence():
    print("getting prudence record:", end='\t')
    dbop = sql.connect("./Record.db")
    dbcs = dbop.cursor()
    for i in range(2):
        prudence = pd.DataFrame(dbcs.execute(f"SELECT * FROM PRUDENCERECORD WHERE DATE = '{str(datetime.datetime.now() - datetime.timedelta(i))[:10] }';"))
        try:
            prudict = {
                'date' : list(prudence[0]),
                'prudence' : list(prudence[1]),
                'prureason' : list(prudence[2])
            }
        except Exception:
            print("No prudence recorded, finding yesterday's prudence index.")
            continue
        break
    print("completed.")
    return json.dumps(prudict)

def write_chat(data):
    """Please give in JSON format"""
    chat = json.loads(data, strict = False)
    date = str(datetime.datetime.now())
    if chat['decision'] == 'buy':
        decision = 0
        curr = 'KRW'
    elif chat['decision'] == 'sell':
        decision = 1
        curr = 'BTC'
    elif chat['decision'] == 'hold' or chat['decision'] == 'wait':
        decision = 2
        curr = 'BTC'
    else:
        decision = 3
        curr = 'BTC'
    
    with sql.connect('./Record.db') as dbop:
        dbcs = dbop.cursor()
        try:
            dbcs.execute("INSERT INTO CHATRECORD VALUES(?, ?, ?, ?, ?, ?);", (date, decision, 
                                                                            chat['profit'],
                                                                            chat['ET'],
                                                                            get_btc_index(), chat['reason']))
        except:
            dbcs.execute("INSERT INTO CHATRECORD VALUES(?, ?, ?, ?, ?, ?);", (date, decision, 
                                                                            chat['loss'],
                                                                            chat['ET'],
                                                                            get_btc_index(), chat['reason']))
        dbop.commit()
# trans_record: time, profit = avgbtc/currbtc * 100 (sell 실행시 입력)

# Sending messages by telegram
async def tel(text):
    bot = telegram.Bot(os.getenv("tel"))
    await bot.send_message(chat_id=os.getenv("chat_id"), text=text) # delete it before publish

def send_tel(text):
    # asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(tel(text))

if __name__ == '__main__':
    # print(f'{get_tech_indi()} {get_today_prudence()} {get_cur_status()}')
    print(gem_sug(gen_bit_model('./Bitcoin Gemini Instruction.md'), get_today_prudence()))
