import streamlit as st
import pandas as pd
import json
from datetime import datetime
import pytz # For timezone handling
from pandas import json_normalize # Import json_normalize
import logging # 로깅 추가
import plotly.graph_objects as go # Plotly 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PRUD_LOG_FILE = "prud_log.jsonl"
BIT_LOG_FILE = "trade_log.jsonl"
KST = pytz.timezone('Asia/Seoul')
MAX_LOGS_DISPLAY = 10 # 채팅 로그 표시 개수 제한

# --- Data Loading Function ---
@st.cache_data(ttl=60)
def load_log_data(log_file_path):
    """Loads and processes log data from a specified JSON Lines file."""
    processed_records = []
    logging.info(f"Attempting to load log data from: {log_file_path}")

    # Determine potential nested keys based on the file path
    potential_nested_keys = []
    if log_file_path == PRUD_LOG_FILE:
        potential_nested_keys = ['prudence_data', 'portfolio_data']
    elif log_file_path == BIT_LOG_FILE:
        potential_nested_keys = ['trade_decision']

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    final_record = record.copy() # Start with the original record

                    # Try flattening known nested keys for this file type
                    for key_to_flatten in potential_nested_keys:
                         if key_to_flatten in record and isinstance(record.get(key_to_flatten), dict):
                             original_nested_data = record[key_to_flatten]
                             try:
                                 flat_nested = json_normalize(original_nested_data).to_dict(orient='records')[0]
                                 # Update final_record: remove original nested, add flattened
                                 if key_to_flatten in final_record: # Make sure key exists before deleting
                                     del final_record[key_to_flatten]
                                 final_record.update(flat_nested)
                                 logging.debug(f"Successfully flattened key '{key_to_flatten}' in line {line_num} of {log_file_path}")
                             except IndexError:
                                 logging.warning(f"json_normalize produced empty result for key '{key_to_flatten}' in line {line_num} of {log_file_path}. Nested data: {original_nested_data}")
                             except Exception as e_flat:
                                 logging.warning(f"Error flattening key '{key_to_flatten}' in line {line_num} of {log_file_path}: {e_flat}")

                    processed_records.append(final_record) # Append the processed record for this line

                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON line #{line_num} in {log_file_path}: {e}")
                except Exception as e_proc:
                    logging.warning(f"Error processing line #{line_num} in {log_file_path}: {e_proc}. Line: {line.strip()[:100]}...")

        if not processed_records:
            logging.warning(f"{log_file_path} is empty or contains no valid/processable JSON.")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(processed_records)
        logging.info(f"Successfully created DataFrame from {log_file_path} with {len(df)} records.")
        logging.debug(f"Initial DataFrame columns for {log_file_path}: {df.columns.tolist()}") # Debug: Show columns

        # --- Timestamp Standardization ---
        timestamp_col = None
        # Prioritize 'log_time' as it seems more consistent across logs
        if 'log_time' in df.columns:
            df.rename(columns={'log_time': 'timestamp'}, inplace=True)
            timestamp_col = 'timestamp'
            if 'timestamp' in df.columns and 'log_time' in df.columns and df['timestamp'].equals(df['log_time']):
                 try: df.drop(columns=['log_time'], inplace=True, errors='ignore')
                 except KeyError: pass
        elif 'timestamp' in df.columns:
             timestamp_col = 'timestamp'
        else:
            logging.error(f"Could not find a suitable timestamp column ('timestamp' or 'log_time') in {log_file_path}.")
            # Attempt to add a dummy timestamp if none exists? Or return empty.
            # For now, return empty to avoid downstream errors.
            return pd.DataFrame()

        # Convert timestamp column
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
             # Try parsing with timezone awareness if possible
             try:
                 df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', utc=True).dt.tz_convert(KST)
             except Exception: # Fallback to naive conversion
                  df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')


        df.dropna(subset=[timestamp_col], inplace=True)

        if df.empty:
             logging.warning(f"No valid timestamp data found in {log_file_path} after cleaning.")
             return pd.DataFrame()

        df = df.sort_values(by=timestamp_col, ascending=False)
        # --- End Timestamp Standardization ---

        # --- Column Renaming (REMOVED specific logic for nested keys) ---
        # rename_map = {}
        # if nested_key == 'suggestion_data': # Now 'trade_decision'
        #     # No automatic renaming needed if display logic handles keys directly
        #     pass
        # elif nested_key == 'prudence_data':
        #     # No automatic renaming needed if display logic handles keys directly
        #     pass
        # df.rename(columns=rename_map, inplace=True)
        # --- End Column Renaming ---

        # Ensure common numeric types after potential flattening
        numeric_cols_to_check = ['confidence', 'latest_rsi', 'btc_balance', 'quote_balance', 'prudence_index']
        for col in numeric_cols_to_check:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')


        logging.debug(f"Final DataFrame columns for {log_file_path}: {df.columns.tolist()}") # Debug: Show final columns

        # Check unique roles before returning
        if 'role' in df.columns:
            logging.debug(f"Unique roles in final DataFrame for {log_file_path}: {df['role'].unique().tolist()}")
        else:
            logging.warning(f"'role' column missing before returning DataFrame from {log_file_path}")

        return df
    except FileNotFoundError:
        st.error(f"Log file not found: {log_file_path}. Please ensure the AI bots have run and generated logs.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading or processing log data from {log_file_path}: {e}")
        logging.exception(f"Detailed error loading {log_file_path}:") # Log full traceback
        return pd.DataFrame()


# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("📊 AI Bitcoin Trader Dashboard")

# Load data from both log files
prud_df = load_log_data(PRUD_LOG_FILE) # nested_key 인자 제거
# Add logging immediately after loading prud_df
if not prud_df.empty and 'role' in prud_df.columns:
    logging.info(f"Unique roles found in prud_df immediately after loading: {prud_df['role'].unique().tolist()}")
elif prud_df.empty:
    logging.warning("prud_df is empty after loading.")
else: # Not empty, but no 'role' column
    logging.warning("prud_df loaded, but 'role' column is missing.")

bit_df = load_log_data(BIT_LOG_FILE) # nested_key 인자 제거

# --- Visualizations (Moved Up) --- #
st.subheader("📈 Portfolio Overview")

# 최신 포트폴리오 데이터 추출 (Prudence 로그에서)
portfolio_logs = prud_df[prud_df['role'] == 'portfolio_summary']
logging.info(f"Found {len(portfolio_logs)} portfolio log entries.") # 디버깅 로그 추가
latest_portfolio_data = None
if not portfolio_logs.empty:
    logging.info(f"Columns in portfolio_logs: {portfolio_logs.columns.tolist()}") # 디버깅 로그 추가
    if 'portfolio_data' in portfolio_logs.columns:
        latest_portfolio_row = portfolio_logs.iloc[0]
        logging.info(f"Latest portfolio log row raw data: {latest_portfolio_row.to_dict()}") # 디버깅 로그 추가
        portfolio_data_content = latest_portfolio_row['portfolio_data']
        if isinstance(portfolio_data_content, dict):
            latest_portfolio_data = portfolio_data_content
            logging.info(f"Successfully extracted latest portfolio data: {latest_portfolio_data}") # 디버깅 로그 추가
        else:
            logging.warning(f"Latest portfolio log entry's 'portfolio_data' is not a dictionary. Type: {type(portfolio_data_content)}")
    else:
        logging.warning("'portfolio_data' column not found in filtered portfolio logs.")
else:
     logging.warning("No log entries found with role 'portfolio_summary'. Cannot display portfolio overview.")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Total Portfolio Value (USDT)", value=f"${latest_portfolio_data.get('total_portfolio_value_usdt', 0.0):.2f}" if latest_portfolio_data else "N/A")
    st.metric(label="Available USDT", value=f"${latest_portfolio_data.get('usdt_balance', 0.0):.2f}" if latest_portfolio_data else "N/A")
    st.metric(label="Positions Held", value=f"{latest_portfolio_data.get('num_positions', 0)} / {latest_portfolio_data.get('max_positions', 'N/A')}" if latest_portfolio_data else "N/A")

with col2:
    st.write("**Portfolio Composition (USDT Value)**")
    if latest_portfolio_data:
        holdings = latest_portfolio_data.get('holdings_value_usdt', {})
        usdt_balance = latest_portfolio_data.get('usdt_balance', 0.0)
        plot_data = holdings.copy()
        if usdt_balance > 0:
            plot_data['Available USDT'] = usdt_balance
        if plot_data:
            labels = list(plot_data.keys())
            values = list(plot_data.values())
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, textinfo='percent+label', pull=[0.05 if label=='Available USDT' else 0 for label in labels])])
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No holdings or USDT balance to display in chart.")
    else:
        st.info("Portfolio data not available for chart.")

st.divider() # 시각화와 로그 사이에 구분선 추가

# --- 총 포트폴리오 가치 추이 --- #
st.subheader("Total Portfolio Value Over Time")
if not portfolio_logs.empty:
    plot_data = []
    for index, row in portfolio_logs.iterrows():
        if isinstance(row.get('portfolio_data'), dict):
            plot_data.append({
                'timestamp': row['timestamp'],
                'value': row['portfolio_data'].get('total_portfolio_value_usdt')
            })
    if plot_data:
        portfolio_value_df = pd.DataFrame(plot_data)
        portfolio_value_df.dropna(subset=['value'], inplace=True)
        if not portfolio_value_df.empty and pd.api.types.is_datetime64_any_dtype(portfolio_value_df['timestamp']):
            portfolio_value_df.set_index('timestamp', inplace=True)
            st.line_chart(portfolio_value_df[['value']], use_container_width=True)
        else:
            st.info("Not enough valid portfolio value data to plot.")
    else:
        st.info("Could not extract valid portfolio value data from logs.")
else:
    st.info("No portfolio history data found for plotting.")

st.divider() # 시각화와 로그 사이에 구분선 추가

# --- Display Tables (Moved Down) --- #
st.subheader("🤖 Trading AI Log")
st.caption(f"Displaying latest {MAX_LOGS_DISPLAY} entries from {BIT_LOG_FILE}") # 캡션 수정

if not bit_df.empty:
    # Display latest logs (use head because data is already sorted descending)
    for index, row in bit_df.head(MAX_LOGS_DISPLAY).iterrows(): # head() 사용
        # Use 'assistant' role for all AI decisions
        with st.chat_message("assistant"):
            # Extract symbol (should be top-level) and decision data (now flattened)
            symbol = row.get('symbol', 'UNKNOWN') # Get symbol from top level

            # Access flattened decision data directly from the row
            timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S KST') if pd.notna(row.get('timestamp')) else "N/A" # Use get for safety
            decision_val = row.get('decision', 'N/A') # Get directly from flattened row
            confidence_val = row.get('confidence', 'N/A') # Get directly from flattened row
            confidence_str = f"{confidence_val:.2f}" if pd.notna(confidence_val) else "N/A" # Format confidence

            # Include symbol in the decision string
            decision_str = f"**Symbol:** {symbol} | **Decision:** {decision_val} (Confidence: {confidence_str})"

            st.caption(f"*{timestamp_str}*")
            st.write(decision_str) # Display symbol, decision, and confidence

            # Display the main reason using markdown (handles tables)
            reason_text = row.get('reason', '*No reason provided*') # Get directly from flattened row
            if reason_text:
                # Ensure reason_text is a string before passing to markdown
                st.markdown(str(reason_text)) # Render the reason field as markdown

            st.divider() # Add a separator between messages

    # --- Older logs in expander --- # (선택적: 모든 로그를 보고 싶을 경우)
    if len(bit_df) > MAX_LOGS_DISPLAY:
        with st.expander("View Older Trading AI Logs..."):
            for index, row in bit_df.iloc[MAX_LOGS_DISPLAY:].iterrows():
                 with st.chat_message("assistant"):
                    symbol = row.get('symbol', 'UNKNOWN')
                    timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S KST') if pd.notna(row.get('timestamp')) else "N/A"
                    decision_val = row.get('decision', 'N/A')
                    confidence_val = row.get('confidence', 'N/A')
                    confidence_str = f"{confidence_val:.2f}" if pd.notna(confidence_val) else "N/A"
                    decision_str = f"**Symbol:** {symbol} | **Decision:** {decision_val} (Confidence: {confidence_str})"
                    st.caption(f"*{timestamp_str}*")
                    st.write(decision_str)
                    reason_text = row.get('reason', '*No reason provided*')
                    if reason_text:
                        st.markdown(str(reason_text))
                    st.divider()

else:
    st.warning(f"No Trading AI log data found or failed to load from {BIT_LOG_FILE}.")

# --- Prudence AI 로그 표시 수정 --- #
st.subheader("🤔 Prudence AI Log")
st.caption(f"Displaying data from {PRUD_LOG_FILE}")
if not prud_df.empty:
    # --- Prudence 제안 로그 표시 (expander 사용) --- #
    prudence_suggestion_logs = prud_df[prud_df['role'] != 'portfolio_summary']
    if not prudence_suggestion_logs.empty:
         # Display latest few prudence suggestions directly? Or keep all in expanders.
         # For now, keep all in expanders for consistency.
         for index, row in prudence_suggestion_logs.iterrows():
             # Use expander to show details for each log entry
             timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S KST') if pd.notna(row.get('timestamp')) else 'Log Entry'
             expander_title = f"{timestamp_str} - Prudence Suggestion"
             with st.expander(expander_title):
                 # Access flattened data directly from the row
                 st.write(f"**Date:** {row.get('date', 'N/A')}") # Get directly from flattened row
                 st.write(f"**Suggested Symbols:**")
                 symbols = row.get('symbols_to_trade', []) # Get directly from flattened row
                 if isinstance(symbols, list) and symbols: # Ensure it's a list
                     # Display symbols nicely
                     st.write(", ".join(symbols))
                 elif isinstance(symbols, str): # Handle case where it might be a string representation
                      st.write(symbols)
                 else:
                     st.write("*No symbols suggested.*")
                 st.write(f"**Reason:**")
                 st.markdown(str(row.get('reason', '*No reason provided*'))) # Get directly from flattened row
    else:
         st.info("No Prudence suggestion logs found.")

else:
    st.warning(f"No Prudence AI log data found or failed to load from {PRUD_LOG_FILE}.")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun() 