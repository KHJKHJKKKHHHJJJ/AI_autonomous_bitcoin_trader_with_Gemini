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
if len(fgtable) > 2:
  tfg, yfg = fgtable.iloc[-1:-3:-1, 1]
elif len(fgtable) == 1:
  tfg = fgtable.iloc[-1, 1]
  yfg = tfg
else:
  tfg = "Fear Greed Not Found"
  yfg = ""
  
col1.metric("Today's Fear Greed Index", f"{tfg}", f"{tfg - yfg}")
if len(prudence_table) > 2:
  tpi, ypi = prudence_table.iloc[-1:-3:-1, 1]
elif len(prudence_table) == 1:
  tpi = prudence_table.iloc[-1, 1]
  ypi = tpi
else:
  tpi = "Prudence Index Not Found"
  ypi = ""

col2.metric("Today's Prudence Index", f"{tpi}", f"{tpi - ypi}")

prudence_reason = st.container(border = True)
prudence_reason.header("Prudence Reason")
if len(prudence_table) > 0:
  prudence_reason.markdown(prudence_table.iloc[-1, -1])
else:
  prudence_reason.write("Prudence Not Found.")

trans_record = pd.DataFrame(list(dbcs.execute("SELECT * FROM CHATRECORD;")), 
                            columns="Date,Decision,ProLoss,EstimatedTime,Price,Reason".split(','))

transaction = st.container(border = True)
transaction.header("Result")
if len(trans_record) > 0:
  decision = trans_record.iloc[-1, 1]
else:
  decision = 3
# decision switcher
if decision == 0:
    decision = ['Buy', 'green']
elif decision == 1:
    decision = ['Sell', 'red']
elif decision == 2:
    decision = ['Hold', 'grey']
else:
  decision = ['Transaction Not Found', 'black']

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
