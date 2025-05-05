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

                    # --- Timestamp Pre-processing (Parse before DataFrame creation) ---
                    ts_val = None
                    ts_key = None
                    if 'log_time' in record:
                        ts_key = 'log_time'
                        ts_val = record.get('log_time')
                    elif 'timestamp' in record:
                        ts_key = 'timestamp'
                        ts_val = record.get('timestamp')

                    parsed_dt = pd.NaT # Default to NaT
                    if ts_val:
                        try:
                            # Attempt direct ISO format parsing first (handles +HH:MM and no offset)
                            parsed_dt = datetime.fromisoformat(str(ts_val)) # Ensure it's string
                            logging.debug(f"Line {line_num}: Parsed '{ts_val}' using datetime.fromisoformat.")
                        except ValueError:
                            try:
                                # Fallback to pandas to_datetime for potentially different formats
                                # Apply errors='coerce' here for the fallback
                                parsed_dt = pd.to_datetime(ts_val, errors='coerce')
                                if pd.isna(parsed_dt):
                                     logging.warning(f"Line {line_num}: Fallback pd.to_datetime resulted in NaT for '{ts_val}'.")
                                else:
                                     logging.debug(f"Line {line_num}: Parsed '{ts_val}' using fallback pd.to_datetime.")
                            except Exception as e_parse:
                                logging.warning(f"Line {line_num}: Failed to parse timestamp '{ts_val}' with fallback: {e_parse}. Will be NaT.")
                                # parsed_dt remains pd.NaT

                    # Store the parsed datetime object (or NaT) back into the record
                    # using a consistent key 'timestamp'
                    if ts_key and ts_key != 'timestamp':
                         if ts_key in final_record: del final_record[ts_key] # Remove original key if different
                    final_record['timestamp'] = parsed_dt
                    # --- End Timestamp Pre-processing ---

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

        # --- Timestamp Standardization (Revised) ---
        timestamp_col = 'timestamp'

        # Check if timestamp column exists
        if timestamp_col not in df.columns:
             logging.error(f"Timestamp column '{timestamp_col}' not found in {log_file_path} after processing lines.")
             return pd.DataFrame()

        # 1. Convert to datetime objects with UTC timezone, coercing errors
        logging.info(f"Converting '{timestamp_col}' to UTC datetime objects for {log_file_path}...")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors='coerce')

        # 2. Log and drop rows where conversion failed (NaT)
        nat_timestamps = df[pd.isna(df[timestamp_col])]
        if not nat_timestamps.empty:
            logging.warning(f"Dropping {len(nat_timestamps)} rows due to NaT in timestamp column after UTC conversion in {log_file_path}:")
            for i, row in nat_timestamps.head(3).iterrows():
                 try: logging.warning(f"  - Row index {i}, Original data sample: {row.to_dict()}")
                 except Exception: logging.warning(f"  - Row index {i}, partial data: {row.get('role', 'N/A')}, raw timestamp: {processed_records[i].get('timestamp') if i < len(processed_records) else 'N/A'}") # Try to get original raw timestamp
        df.dropna(subset=[timestamp_col], inplace=True)

        # Check if empty after dropping NaTs
        if df.empty:
             logging.warning(f"No valid timestamp data remains in {log_file_path} after UTC conversion and NaT drop.")
             return pd.DataFrame()

        # 3. Convert from UTC to KST
        logging.info(f"Converting '{timestamp_col}' from UTC to KST for {log_file_path}...")
        try:
             df[timestamp_col] = df[timestamp_col].dt.tz_convert(KST)
        except Exception as e_conv:
              logging.error(f"Error converting timestamps from UTC to KST in {log_file_path}: {e_conv}")
              # Decide how to handle this - return empty or keep UTC? For now, return empty.
              return pd.DataFrame()

        # 4. Sort by timestamp descending
        df = df.sort_values(by=timestamp_col, ascending=False)
        logging.info(f"Successfully processed timestamps for {log_file_path}. Final dtype: {df[timestamp_col].dtype}")
        # --- End Timestamp Standardization ---

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

# --- Check prud_df validity ---
prud_df_valid = False
if not prud_df.empty and 'role' in prud_df.columns:
    logging.info(f"Unique roles found in prud_df immediately after loading: {prud_df['role'].unique().tolist()}")
    prud_df_valid = True # Mark as valid
elif prud_df.empty:
    logging.warning("prud_df is empty after loading.")
    st.warning(f"Could not load or process data from {PRUD_LOG_FILE}. Portfolio section might be unavailable.")
else: # Not empty, but no 'role' column
    logging.warning(f"prud_df loaded, but 'role' column is missing. Columns: {prud_df.columns.tolist()}")
    st.warning(f"Data loaded from {PRUD_LOG_FILE} is missing the 'role' column. Portfolio section might be unavailable.")

bit_df = load_log_data(BIT_LOG_FILE) # nested_key 인자 제거

# --- Visualizations (Moved Up) --- #
st.subheader("📈 Portfolio Overview")

# Only proceed if prud_df is valid
if prud_df_valid:
    # 최신 포트폴리오 데이터 추출 (Prudence 로그에서)
    portfolio_logs = prud_df[prud_df['role'] == 'portfolio_summary'] # This line is now safe
    logging.info(f"Found {len(portfolio_logs)} portfolio log entries.") # 디버깅 로그 추가
    latest_portfolio_row = None # Initialize
    if not portfolio_logs.empty:
        logging.info(f"Columns in portfolio_logs: {portfolio_logs.columns.tolist()}") # 디버깅 로그 추가
        # Use .iloc[0] to get the Series for the latest log
        latest_portfolio_row = portfolio_logs.iloc[0]
        logging.info(f"Latest portfolio log row raw data: {latest_portfolio_row.to_dict()}") # 디버깅 로그 추가
    else:
         logging.warning("No log entries found with role 'portfolio_summary' after filtering.")

    col1, col2 = st.columns(2)

    with col1:
        # Access flattened data directly from the row (Series)
        total_value = latest_portfolio_row.get('total_portfolio_value_usdt', 0.0) if latest_portfolio_row is not None else 0.0
        usdt_balance_val = latest_portfolio_row.get('usdt_balance', 0.0) if latest_portfolio_row is not None else 0.0
        num_positions = latest_portfolio_row.get('num_positions', 0) if latest_portfolio_row is not None else 0
        max_positions = latest_portfolio_row.get('max_positions', 'N/A') if latest_portfolio_row is not None else 'N/A'

        st.metric(label="Total Portfolio Value (USDT)", value=f"${total_value:.2f}" if total_value is not None else "N/A")
        st.metric(label="Available USDT", value=f"${usdt_balance_val:.2f}" if usdt_balance_val is not None else "N/A")
        st.metric(label="Positions Held", value=f"{num_positions} / {max_positions}" if num_positions is not None else "N/A")

    with col2:
        st.write("**Portfolio Composition (USDT Value)**")
        if latest_portfolio_row is not None:
            # Reconstruct holdings from flattened columns
            holdings = {}
            prefix = 'holdings_value_usdt.'
            for col_name in latest_portfolio_row.index:
                if col_name.startswith(prefix):
                    # Ensure the value is numeric before adding
                    value = pd.to_numeric(latest_portfolio_row[col_name], errors='coerce')
                    if pd.notna(value) and value > 0:
                         symbol = col_name[len(prefix):] # Extract symbol (e.g., ADA)
                         holdings[symbol] = value

            usdt_balance_chart = pd.to_numeric(latest_portfolio_row.get('usdt_balance', 0.0), errors='coerce')
            # Ensure usdt_balance_chart is not NaN before comparison
            if pd.isna(usdt_balance_chart):
                 usdt_balance_chart = 0.0
            plot_data = holdings.copy()
            if usdt_balance_chart > 0:
                plot_data['Available USDT'] = usdt_balance_chart

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
else:
    # Display message indicating data is unavailable
    st.info(f"Portfolio overview data from {PRUD_LOG_FILE} is unavailable due to loading issues.")
    # Display placeholder metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Total Portfolio Value (USDT)", value="N/A")
        st.metric(label="Available USDT", value="N/A")
        st.metric(label="Positions Held", value="N/A")
    with col2:
        st.write("**Portfolio Composition (USDT Value)**")
        st.info("Portfolio data not available for chart.")

st.divider() # 시각화와 로그 사이에 구분선 추가

# --- 총 포트폴리오 가치 추이 --- #
st.subheader("Total Portfolio Value Over Time")
if prud_df_valid:
     # Re-filter prud_df here to get logs for the trend chart
     portfolio_logs_for_trend = prud_df[prud_df['role'] == 'portfolio_summary'].copy() # Use .copy()

     if not portfolio_logs_for_trend.empty:
         plot_data_trend = [] # Use different variable name
         for index, row in portfolio_logs_for_trend.iterrows():
             # Access flattened value directly and ensure it's numeric
             value = pd.to_numeric(row.get('total_portfolio_value_usdt'), errors='coerce')
             timestamp = row.get('timestamp') # Get timestamp
             # Ensure both value and timestamp are valid
             if pd.notna(value) and pd.notna(timestamp):
                 plot_data_trend.append({
                     'timestamp': timestamp,
                     'value': value
                 })

         if plot_data_trend:
             portfolio_value_df = pd.DataFrame(plot_data_trend)
             # Check again if DataFrame is valid and has correct types before plotting
             if not portfolio_value_df.empty and \
                'timestamp' in portfolio_value_df.columns and \
                'value' in portfolio_value_df.columns and \
                pd.api.types.is_datetime64_any_dtype(portfolio_value_df['timestamp']):
                 portfolio_value_df.set_index('timestamp', inplace=True)
                 st.line_chart(portfolio_value_df[['value']], use_container_width=True)
             else:
                 st.info("Not enough valid portfolio value data to plot after processing.")
         else:
             st.info("Could not extract valid portfolio value data from logs for plotting.")
     else:
         st.info("No 'portfolio_summary' logs found for plotting.")
else:
     st.info(f"Portfolio history data from {PRUD_LOG_FILE} is unavailable.")

st.divider() # 시각화와 로그 사이에 구분선 추가

# --- Display Tables (Moved Down) --- #
st.subheader("🤖 Trading AI Log")
st.caption(f"Displaying latest {MAX_LOGS_DISPLAY} entries from {BIT_LOG_FILE}") # 캡션 수정

# --- Bit Trader AI 로그 표시 수정 (trade_log.jsonl) --- #
st.subheader("🤖 Bit Trader AI Log (Execution Attempts)")
st.caption(f"Displaying data from {BIT_LOG_FILE}")

# Filter controls for Bit Trader Log
symbol_filter = None
if not bit_df.empty and 'symbol' in bit_df.columns:
    # Get unique symbols from the log, handle potential None/NaN
    unique_symbols = bit_df['symbol'].dropna().unique().tolist()
    if unique_symbols:
        symbol_filter = st.multiselect(
            'Filter by Symbol(s):',
            options=sorted(unique_symbols),
            default=None, # Initially show all
            key='bit_log_symbol_filter' # Unique key for the widget
        )

# Apply filter if symbols are selected
filtered_bit_df = bit_df
if symbol_filter: # If list is not empty (user selected something)
    filtered_bit_df = bit_df[bit_df['symbol'].isin(symbol_filter)]
    st.caption(f"Showing logs for: {', '.join(symbol_filter)}")
elif symbol_filter == []: # If list is empty (user selected then deselected all)
    st.caption("Showing logs for all symbols (no filter selected).")
    # filtered_bit_df remains bit_df (show all)

if not filtered_bit_df.empty:
    # Display filtered logs in expanders
    for index, row in filtered_bit_df.iterrows():
        # Determine if it's an execution log or just a decision log
        is_execution_log = pd.notna(row.get('success')) # success 필드 유무로 판단

        # Determine expander title
        timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S KST') if pd.notna(row.get('timestamp')) else 'Log Entry'
        log_symbol = row.get('symbol', 'Unknown Symbol')
        expander_title = f"{timestamp_str} - {log_symbol}"
        title_prefix = ""
        decision_text = "N/A"

        if is_execution_log:
            # Execution Log Title
            action = row.get('side_attempted')
            decision_text = f"Attempt: {action}" if pd.notna(action) else "Attempt: N/A"
            success = row.get('success')
            if pd.notna(success):
                title_prefix = "✅" if success else "❌"
            else:
                title_prefix = "➡️" # Status unknown
            expander_title = f"{title_prefix} {timestamp_str} - {log_symbol} - {decision_text}"
        else:
            # Decision Log Title (Assume from 'trade_decision' field)
            ai_decision_data = row.get('trade_decision')
            ai_decision = "N/A"
            if pd.notna(ai_decision_data) and isinstance(ai_decision_data, dict):
                 ai_decision = ai_decision_data.get('decision', 'N/A')
            title_prefix = "🤔"
            decision_text = f"AI Decision: {ai_decision}"
            expander_title = f"{title_prefix} {timestamp_str} - {log_symbol} - {decision_text}"

        with st.expander(expander_title):
            st.write(f"**Symbol:** {log_symbol}")

            if is_execution_log:
                # Display Execution Log Details
                action = row.get('side_attempted')
                if pd.notna(action):
                    st.write(f"**Attempted Action:** {action}")
                qty = row.get('quantity_attempted_or_adjusted')
                if pd.notna(qty):
                    st.write(f"**Attempted/Adjusted Qty:** {qty}")
                success = row.get('success')
                if pd.notna(success):
                    st.write(f"**Execution Success:** {success}")
                error_msg = row.get('error_message')
                if pd.notna(error_msg):
                    st.write(f"**Error:** {error_msg}")
                order_details = row.get('order_details')
                if pd.notna(order_details) and isinstance(order_details, dict):
                    st.write("**Order Details:**")
                    st.json(order_details, expanded=False)

                # Show the AI decision that triggered this execution
                ai_decision_data = row.get('trade_decision')
                if pd.notna(ai_decision_data) and isinstance(ai_decision_data, dict):
                    st.write("**Triggering AI Decision:**")
                    ai_decision = ai_decision_data.get('decision')
                    if pd.notna(ai_decision):
                         st.write(f" - AI Decision: {ai_decision}")
                    ai_confidence = ai_decision_data.get('confidence')
                    if pd.notna(ai_confidence):
                         st.write(f" - Confidence: {ai_confidence}")
                    ai_reason = ai_decision_data.get('reason')
                    if pd.notna(ai_reason):
                         st.write(f" - AI Reason: {ai_reason}")
                    ai_next_check = ai_decision_data.get('next_check_minutes')
                    if pd.notna(ai_next_check):
                         st.write(f" - Next Check (min): {ai_next_check}")
                elif pd.notna(ai_decision_data):
                     st.write(f"**Triggering AI Decision Data (raw):** {ai_decision_data}")

            else:
                # Display AI Decision Log Details (from 'trade_decision')
                ai_decision_data = row.get('trade_decision')
                if pd.notna(ai_decision_data) and isinstance(ai_decision_data, dict):
                    ai_decision = ai_decision_data.get('decision')
                    if pd.notna(ai_decision):
                        st.write(f"**AI Decision:** {ai_decision}")
                    ai_confidence = ai_decision_data.get('confidence')
                    if pd.notna(ai_confidence):
                        st.write(f"**Confidence:** {ai_confidence}")
                    ai_reason = ai_decision_data.get('reason')
                    if pd.notna(ai_reason):
                        st.write(f"**AI Reason:** {ai_reason}")
                    ai_next_check = ai_decision_data.get('next_check_minutes')
                    if pd.notna(ai_next_check):
                        st.write(f"**Next Check (min):** {ai_next_check}")
                    ai_summary = ai_decision_data.get('analysis_summary')
                    if pd.notna(ai_summary):
                        st.write(f"**Analysis Summary:** {ai_summary}")
                    # Optionally display full raw decision
                    # with st.expander("Raw Decision Data"):
                    #     st.json(ai_decision_data)
                elif pd.notna(ai_decision_data):
                    st.write(f"**AI Decision Data (raw):** {ai_decision_data}")
                else:
                    st.warning("Could not find AI decision data in this log entry.")
else:
    st.info(f"No Bit Trader AI execution logs found (or none match the filter: {symbol_filter}).")

st.divider()

# --- Prudence AI 로그 표시 수정 --- #
st.subheader("🤔 Prudence AI Log")
st.caption(f"Displaying data from {PRUD_LOG_FILE}")

# Check validity again for this section
if prud_df_valid:
    # --- Prudence 제안 로그 표시 (expander 사용) --- #
    # Filter logs that are NOT portfolio summaries
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
         st.info("No Prudence suggestion logs found (excluding portfolio summaries).")
else:
    st.warning(f"Could not display Prudence AI logs due to loading issues with {PRUD_LOG_FILE}.")

if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun() 