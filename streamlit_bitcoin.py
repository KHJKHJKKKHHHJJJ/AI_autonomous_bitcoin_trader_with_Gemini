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
if len(fgtable) >= 2:
  tfg, yfg = fgtable.iloc[-1:-3:-1, 1]
  yfg = -yfg
elif len(fgtable) == 1:
  tfg = fgtable.iloc[-1, 1]
  yfg = -tfg
else:
  tfg = "Fear Greed Not Found"
  yfg = ""
  
col1.metric("Today's Fear Greed Index", f"{tfg}", f"{tfg + yfg}")
if len(prudence_table) >= 2:
  tpi, ypi = prudence_table.iloc[-1:-3:-1, 1]
  ypi = -ypi
elif len(prudence_table) == 1:
  tpi = prudence_table.iloc[-1, 1]
  ypi = -tpi
else:
  tpi = "Prudence Index Not Found"
  ypi = ""

col2.metric("Today's Prudence Index", f"{tpi}", f"{tpi + ypi}")

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
  Ep = round(float(trans_record.iloc[-1, 2]), 2)
else:
  decision = 3
  Ep = "Decision Not Found."

# decision switcher
if decision == 0:
    decision = ['Buy', 'green']
elif decision == 1:
    decision = ['Sell', 'red']
elif decision == 2:
    decision = ['Hold', 'grey']
else:
  decision = ['Transaction Not Found', 'grey']

transaction.subheader(f"Last Decision:\t:{decision[1]}[{decision[0]}]")
transaction.subheader(f"Estimated Profit/Loss:\t:{decision[1]}[{Ep}]", 
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

transaction.subheader(f"Current Profit(%):\t:{status_color[0]}[{round(curr_profit, 2)}]")
transaction.subheader(f"Current Profit(KRW):\t:{status_color[0]}[{inKRW}]")

profit = list(dbcs.execute("SELECT SUM(PROFIT) FROM TRANSRECORD;"))
pro_am = list(dbcs.execute("SELECT SUM(AMOUNT) FROM TRANSRECORD;"))

if len(profit) > 0:
    sum_proloss = round(float(profit[0][0]), 2)
else:
   sum_proloss = 0
  
if len(pro_am) > 0:
   sum_pro_am = round(pro_am[0][0], 2)
else:
   sum_pro_am = 0

if sum_proloss > 0:
    cum_col = ['green']
elif sum_proloss == 0:
    cum_col = ['grey']
else:
    cum_col = ['red']

transaction.subheader(f"Cumulative Profit(%):\t:{cum_col[0]}[{sum_proloss}]")
transaction.subheader(f"Cumulative Profit(KRW):\t:{cum_col[0]}[{sum_pro_am}]")

dbop.close()
