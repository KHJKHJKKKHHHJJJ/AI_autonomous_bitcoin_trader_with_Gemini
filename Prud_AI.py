import google.generativeai as genai
# import KEYS
import os
from dotenv import load_dotenv
load_dotenv()

import json
import sqlite3 as sql
import datetime
import requests as rqs
from bs4 import BeautifulSoup
import pandas as pd


def gen_pru_model(instruction):
    genai.configure(api_key = os.getenv("Gemini"))

    # Create the model
    generation_config = {
    "temperature": 1.5,
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


def gem_pru_sug(chat_session):
    records = get_chat_record(), bring_fear_greed(), get_news(), get_prudence(), get_trans_record()
    
    chat_record = ["".join(json.dumps(i)) for i in records]

    response = chat_session.send_message(chat_record[0] + "today's date: " + str(datetime.datetime.now())[:10])
    return json.loads(response.text, strict = False)
    # print(chat_record[0] + "today's date: " + str(datetime.datetime.now())[:10])


def get_instructions(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            instructions = file.read()
        return instructions
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("An error occurred while reading the file:", e)


def write_prudence(data):
    """Please give dict format data."""
    with sql.connect("./Record.db") as dbop:
        dbcs = dbop.cursor()
        dbcs.execute("INSERT INTO PRUDENCERECORD VALUES(?,?,?)", [data[i][1] for i in range(3)])
        dbcs.commit()


def get_news():
    '''Brings the news articles about bitcoin or crypto from investing.com 
    returns json data type'''

    news_dict = {
        "datetime": [],
        "title" : [],
        "paragraph" : []
    }

    url = "https://www.investing.com/news/cryptocurrency-news"

    response = rqs.get(url)
    response.status_code
    soup = BeautifulSoup(response.text, 'html.parser')
    news = soup.find_all(name="a", 
                attrs='text-inv-blue-500 hover:text-inv-blue-500 hover:underline focus:text-inv-blue-500 focus:underline whitespace-normal text-sm font-bold leading-5 !text-[#181C21] sm:text-base sm:leading-6 lg:text-lg lg:leading-7')
    new = news[-1]
    news_dict['title'] = [i.text for i in news]
    pub_dates = soup.find_all(name = 'div', attrs= 'flex flex-wrap items-center text-2xs')
    news_dict['datetime'] = [i.find('time')['datetime'][:-3] for i in pub_dates]
    for new in news[:10]:
        response = rqs.get(new['href'])
        if response.status_code != 200:
            print("Error Occured: ", response.status_code)
            news_dict['paragraph'].append(None)
        else:
            soup = BeautifulSoup(response.text, 'html.parser')
            div = soup.find("div", "article_WYSIWYG__O0uhw article_articlePage__UMz3q text-[18px] leading-8")
            news_dict['paragraph'].append("".join([i.text for i in div.find_all("p")]))
    return json.dumps(news_dict)


def get_prudence():
    print("getting prudence record:", end='\t')
    dbop = sql.connect("./Record.db")
    dbcs = dbop.cursor()
    prudence = pd.DataFrame(dbcs.execute("SELECT * FROM PRUDENCERECORD;"))
    try:
        prudict = {
            'date' : list(prudence[0]),
            'prudence' : list(prudence[1]),
            'prureason' : list(prudence[2])
        }
    except Exception:
        print("No prudence recorded")
        return '{}'
    print("completed.")
    return json.dumps(prudict)


def bring_fear_greed():
    """brings Fear Greed Index from Record DB as JSON format."""
    print('getting FGIndex:', end='\t')
    with sql.connect('./Record.db') as dbop:
        dbcs = dbop.cursor()
        fgi = pd.DataFrame(dbcs.execute('SELECT * FROM FGRECORD;'), columns = ['date', 'FGindex'])
    if str(datetime.datetime.now())[:10] not in list(fgi['date']):
        print("Today FGI not found. \nExtracting...")
        write_fear_greed()
        bring_fear_greed()
    else:
        print("completed")
        return json.dumps({'date': list(fgi['date']),
                        'FGI' : list(fgi['FGindex'])})

def write_fear_greed():
    '''Records Fear_Greed Index of its date.
    It's void function'''
    headers = {
        "ACCEPT" : "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "USER-AGENT" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    }

    url = "https://www.blockstreet.co.kr/fear-greed"
    response = rqs.get(url, headers=headers)
    if response.status_code != 200:
        print("response error occured", response.status_code)
        return 
    soup = BeautifulSoup(response.text, 'html.parser')
    fear_greed = int(soup.find('span', 'greed-background-3').text)
    date = str(datetime.datetime.today())[:10]
    
    with sql.connect('Record.db') as dbop:
        dbcs = dbop.cursor()
        dbcs.execute('INSERT INTO FGRECORD VALUES (?, ?);', (date, fear_greed))
        dbop.commit()


def get_chat_record():
    print("getting chat record:", end='\t')
    dbop = sql.connect('Record.db')
    dbcs = dbop.cursor()
    chat = pd.DataFrame(dbcs.execute(f"SELECT * FROM CHATRECORD WHERE DATE LIKE '{str(datetime.datetime.now() - datetime.timedelta(1))[:10]}%';"))
    try:
        chat_dict = {
            "date"              :list(chat[0]),
            'buy_or_sell'       :list(chat[1]),
            'ratio'             :list(chat[2]),
            'estimated'         :list(chat[3]),
            'price'             :list(chat[4]),
            'reason'            :list(chat[5])
        }
    except Exception:
        print("No chatting recorded")
        return '{}'
    print("completed.")
    return json.dumps(chat_dict)

def get_trans_record():
    """returns transaction records in JSON format"""
    date = str(datetime.datetime.today() - datetime.timedelta(1))[:10]
    with sql.connect('./Record.db') as dbop:
        dbcs = dbop.cursor()
        transaction = pd.DataFrame(dbcs.execute(f"SELECT * FROM TRANSRECORD WHERE TIME LIKE '{date}%';"))
    return json.dumps(transaction.to_dict('list'))


if __name__ == '__main__':
    print(get_news(), get_prudence(), bring_fear_greed())
