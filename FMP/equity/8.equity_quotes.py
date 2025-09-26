import requests
import pandas as pd
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Import resource-aware components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib import (
    wait_for_resources, csv_buffer_context, create_checkpoint,
    save_checkpoint, load_checkpoint, should_resume_processing,
    wait_for_recovery, cleanup_checkpoint, update_checkpoint_progress,
    get_resource_stats, calculate_optimal_batch_size
)

load_dotenv()

FMP_API_KEY = os.getenv('FMP_API_KEY')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Validate required environment variables
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY must be set in .env file")
if not all([DB_HOST, DB_NAME, DB_USER]):
    raise ValueError("DB_HOST, DB_NAME, and DB_USER must be set in .env file")

def fetch_price_history(symbol, ipo_date, checkpoint=None):
    """Fetch price history with resource-aware processing."""
    if not wait_for_resources(max_wait=60.0):
        print(f"Resource constraints - skipping {symbol}")
        return symbol, []

    start_date = ipo_date if ipo_date else datetime(1900, 1, 1).date()
    today = datetime.now().date()
    data = []
    current_start = start_date
    api_calls = 0

    try:
        while current_start < today:
            # Check resources before each API call
            if not wait_for_resources(max_wait=30.0):
                print(f"Resource constraints during {symbol} processing")
                break

            current_end = current_start + timedelta(days=5 * 365 + 1)  # Approx 5 years, +1 for leap
            current_end = min(current_end, today)
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={current_start}&to={current_end}&apikey={FMP_API_KEY}"

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            json_data = response.json()
            api_calls += 1

            if 'historical' in json_data:
                data.extend(json_data['historical'])

            current_start = current_end + timedelta(days=1)

            # Rate limiting - 25 calls/second
            time.sleep(0.04)

        return symbol, data, api_calls

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return symbol, [], api_calls

if __name__ == "__main__":
    script_name = "equity_quotes"

    # Check for existing checkpoint and wait for recovery if needed
    if should_resume_processing(script_name):
        if not wait_for_recovery(script_name):
            print("Failed to wait for system recovery")
            exit(1)
        checkpoint = load_checkpoint(script_name)
    else:
        checkpoint = None

    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    profile_df = pd.read_sql("SELECT symbol, ipo_date FROM equity_profile;", engine)
    profile_dict = dict(zip(profile_df['symbol'], profile_df['ipo_date']))
    symbols = list(profile_dict.keys())

    # Filter out completed symbols if resuming
    if checkpoint:
        completed_symbols = set(checkpoint.completed_symbols)
        symbols = [s for s in symbols if s not in completed_symbols]
        print(f"Resuming: {len(completed_symbols)} symbols already completed")

    script_checkpoint = checkpoint or create_checkpoint(
        script_name="equity_quotes",
        total_symbols=len(symbols) + (len(checkpoint.completed_symbols) if checkpoint else 0)
    )

    # Define column order for CSV buffer
    columns = ['date', 'open', 'high', 'low', 'close', 'adjClose', 'volume',
              'unadjustedVolume', 'change', 'changePercent', 'vwap', 'label',
              'changeOverTime', 'symbol']

    try:
        # Use reduced thread pool to manage resource consumption
        optimal_batch = calculate_optimal_batch_size()
        max_workers = min(5, max(1, int(optimal_batch['available_memory_gb'] * 2)))

        with csv_buffer_context('equity_quotes', columns, engine):
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch_price_history, symbol, profile_dict[symbol], script_checkpoint): symbol
                          for symbol in symbols}

                for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
                    symbol = futures[future]
                    try:
                        symbol_result, data, api_calls = future.result()

                        if data:
                            # Add rows to CSV buffer
                            from lib.csv_buffer import add_row
                            for record in data:
                                record['symbol'] = symbol_result
                                # Ensure all expected columns are present
                                row_data = {col: record.get(col, None) for col in columns}
                                add_row('equity_quotes', row_data)

                            # Update checkpoint with success
                            script_checkpoint = update_checkpoint_progress(
                                script_checkpoint,
                                new_api_calls=api_calls,
                                new_symbol=symbol_result,
                                completed_symbol=symbol_result
                            )
                        else:
                            print(f"No data for {symbol_result}")
                            # Update checkpoint with failure
                            script_checkpoint = update_checkpoint_progress(
                                script_checkpoint,
                                new_api_calls=api_calls,
                                failed_symbol=symbol_result
                            )

                        save_checkpoint(script_checkpoint)

                    except Exception as e:
                        print(f"Error processing {symbol}: {e}")
                        script_checkpoint = update_checkpoint_progress(
                            script_checkpoint,
                            failed_symbol=symbol
                        )
                        save_checkpoint(script_checkpoint)

        print("Processing completed successfully")
        cleanup_checkpoint(script_name)

    except KeyboardInterrupt:
        print("Processing interrupted - checkpoint saved")
        save_checkpoint(script_checkpoint)
        raise
    except Exception as e:
        print(f"Error in main processing: {e}")
        save_checkpoint(script_checkpoint)
        raise