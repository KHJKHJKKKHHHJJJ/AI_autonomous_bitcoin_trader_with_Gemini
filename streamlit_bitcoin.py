import streamlit as st
import sqlite3 as sql
import pandas as pd
import time
import datetime
from bit_AI import get_btc_index, wallet 

dbop = sql.connect("Record.db")
dbcs = dbop.cursor()

prudence_table = pd.DataFrame(list(dbcs.execute("SELECT * FROM PRUDENCERECORD;")), columns=['Date', "PI", "Reason"])
fgtable = pd.DataFrame(list(dbcs.execute("SELECT * FROM FGRECORD;")), columns=['date', 'FGI'])

st.header("Welcome To Gemini Bitcoin Assistant!")

status = st.container(border = True)
col1, col2 = status.columns(2)
if fgtable.count() > 2:
  tfg, yfg = fgtable.iloc[-1:-3:-1, 1]
else:
  tfg = fgtable.iloc[-1, 1]
  yfg = tfg
  
col1.metric("Today's Fear Greed Index", f"{tfg}", f"{tfg - yfg}")
tpi, ypi = prudence_table.iloc[-1:-3:-1, 1]
col2.metric("Today's Prudence Index", f"{tpi}", f"{tpi - ypi}")

prudence_reason = st.container(border = True)
prudence_reason.header("Prudence Reason")
prudence_reason.markdown(prudence_table.iloc[-1, -1])

trans_record = pd.DataFrame(list(dbcs.execute("SELECT * FROM CHATRECORD;")), 
                            columns="Date,Decision,ProLoss,EstimatedTime,Price,Reason".split(','))

transaction = st.container(border = True)
transaction.header("Result")
decision = trans_record.iloc[-1, 1]
# decision switcher
if decision == 0:
    decision = ['Buy', 'green']
elif decision == 1:
    decision = ['Sell', 'red']
else:
    decision = ['Hold', 'grey']

transaction.subheader(f"Last Decision: :{decision[1]}[{decision[0]}]")
transaction.subheader(f"Estimated Profit/Loss: :{decision[1]}[{int(trans_record.iloc[-1, 2])}]", 
                      divider = 'grey')
transaction.header("Status")

# Status Calculation
currency = get_btc_index()
avg = float(wallet("BTC")[0]['average_price'])
available = float(wallet("BTC")[0]['available'])

if available:
    curr_profit = (currency - avg) / avg * 100
else:
    curr_profit = 0

status_color = []

if curr_profit > 0:
    status_color.append("green")
elif curr_profit < 0:
    status_color.append("red")
else:
    status_color.append("grey")

inKRW = round((currency - avg) * float(wallet('BTC')[0]['available']), 0)

transaction.subheader(f"Current Profit(%): :{status_color[0]}[{round(curr_profit, 2)}]")
transaction.subheader(f"Current Profit(KRW): :{status_color[0]}[{inKRW}]")

sum_proloss = round(float(list(dbcs.execute("SELECT SUM(PROFIT) FROM TRANSRECORD;"))[0][0]), 2)
if sum_proloss > 0:
    cum_col = ['green']
elif sum_proloss == 0:
    cum_col = ['grey']
else:
    cum_col = ['red']

transaction.subheader(f"Cumulative Profit(%): :{cum_col[0]}[{sum_proloss}]")

dbop.close()
