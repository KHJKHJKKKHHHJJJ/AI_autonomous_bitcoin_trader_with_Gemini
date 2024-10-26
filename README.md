# AI_autonomous_bitcoin_trader_with_Gemini
Bitcoin trading assistant Gemini will help you to make money.

# How To Use
## Initializing
- Websites to enable APIs
  - [Gemini](https://console.cloud.google.com/apis/) **requires google account**
  - [Coinone](https://docs.coinone.co.kr/docs/about-public-api) **requires Coinone account**
  - [Telegram](https://core.telegram.org/) **I recommend you to watch a video to generage a bot**
- to install the packages, run
  - `pip install -r requirements.txt`
  - `pip3 install -r requirements.txt`
 
## Prud_AI.py
Prudence Index (0~100): By moderating prudence index, Gemini will make a decision more conservatively or Progressively. 

This bitcoin trader is basically designed as a short-term bit-coin trader, so factors that might be necessary to long-term trend used to set the Prudence Index. 

Prudence AI setting a Prudence Index also gives a feedback from yesterday's decision made by Bit_AI. 

## bit_AI.py
Make a decision based on chart data, technical indicators, and Prudence Index. Once it makes a decision, it will automatically proceed a transaction, and send a transaction record to user via Telegram.

# Caution
- Before using this, please change the one of the file of Pandas_ta named `pandas_ta/overlap/ema.py`, first line `from numpy import npNaN` to `from numpy import nan as npNaN`.
