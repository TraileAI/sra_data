import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os
import gc
import psutil
from datetime import datetime, timedelta

load_dotenv()

FMP_API_KEY = os.getenv('FMP_API_KEY', "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT")
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

DB_CONFIG = {
    'host': DB_HOST,
    'port': DB_PORT,
    'database': DB_NAME,
    'user': DB_USER,
    'password': DB_PASSWORD
}

def fetch_price_history(symbol, ipo_date):
    start_date = ipo_date if ipo_date else datetime(1900, 1, 1).date()
    today = datetime.now().date()
    data = []
    current_start = start_date
    while current_start < today:
        current_end = current_start + timedelta(days=5 * 365 + 1)  # Approx 5 years, +1 for leap
        current_end = min(current_end, today)
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={current_start}&to={current_end}&apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        if 'historical' in json_data:
            data.extend(json_data['historical'])
        current_start = current_end + timedelta(days=1)
    return symbol, data

def create_quotes_table():
    """Create equity_quotes table if it doesn't exist."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS equity_quotes (
                    symbol VARCHAR(50),
                    date DATE,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    adjClose DOUBLE PRECISION,
                    volume BIGINT,
                    unadjustedVolume BIGINT,
                    change DOUBLE PRECISION,
                    changePercent DOUBLE PRECISION,
                    vwap DOUBLE PRECISION,
                    label VARCHAR(50),
                    changeOverTime DOUBLE PRECISION,
                    PRIMARY KEY (symbol, date)
                );
                CREATE INDEX IF NOT EXISTS idx_equity_quotes_symbol ON equity_quotes(symbol);
                CREATE INDEX IF NOT EXISTS idx_equity_quotes_date ON equity_quotes(date);
            """)
            conn.commit()
    finally:
        conn.close()

def bulk_insert_quotes(quotes_data):
    """Use COPY for bulk insert of quotes data."""
    if not quotes_data:
        return

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # Create temporary table
            cur.execute("CREATE TEMP TABLE temp_quotes (LIKE equity_quotes INCLUDING DEFAULTS)")

            # Prepare data for COPY
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO temp_quotes VALUES %s",
                quotes_data,
                template=None,
                page_size=1000
            )

            # Insert with conflict handling
            cur.execute("""
                INSERT INTO equity_quotes SELECT * FROM temp_quotes
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    adjClose = EXCLUDED.adjClose,
                    volume = EXCLUDED.volume,
                    unadjustedVolume = EXCLUDED.unadjustedVolume,
                    change = EXCLUDED.change,
                    changePercent = EXCLUDED.changePercent,
                    vwap = EXCLUDED.vwap,
                    label = EXCLUDED.label,
                    changeOverTime = EXCLUDED.changeOverTime
            """)
            conn.commit()
    finally:
        conn.close()

def process_symbol_batch(symbols_batch, profile_dict):
    """Process a batch of symbols with reduced concurrency."""
    quotes_data = []

    with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced from 10 to 2
        futures = [executor.submit(fetch_price_history, symbol, profile_dict.get(symbol)) for symbol in symbols_batch]

        for future in tqdm(as_completed(futures), total=len(symbols_batch), desc=f"Batch of {len(symbols_batch)}"):
            try:
                symbol, data = future.result()
                if data:
                    for record in data:
                        quotes_data.append((
                            symbol,
                            record.get('date'),
                            record.get('open'),
                            record.get('high'),
                            record.get('low'),
                            record.get('close'),
                            record.get('adjClose'),
                            record.get('volume'),
                            record.get('unadjustedVolume'),
                            record.get('change'),
                            record.get('changePercent'),
                            record.get('vwap'),
                            record.get('label'),
                            record.get('changeOverTime')
                        ))
                else:
                    print(f"No data for {symbol}")
            except Exception as e:
                print(f"Error processing symbol: {e}")

    # Bulk insert the batch
    if quotes_data:
        print(f"Inserting {len(quotes_data)} quote records...")
        bulk_insert_quotes(quotes_data)

    # Memory cleanup
    del quotes_data
    gc.collect()

    # System monitoring
    memory_percent = psutil.virtual_memory().percent
    print(f"📊 Memory usage: {memory_percent:.1f}%")

    if memory_percent > 70:
        print("⚠️ High memory usage - extended cooldown (30 seconds)...")
        import time
        time.sleep(30)

if __name__ == "__main__":
    create_quotes_table()

    # Get symbols from database
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT symbol, ipo_date FROM equity_profile")
            results = cur.fetchall()
            profile_dict = {row[0]: row[1] for row in results}
            symbols = list(profile_dict.keys())
    finally:
        conn.close()

    print(f"Processing {len(symbols)} symbols in batches of 50...")

    # Process symbols in batches to prevent memory overload
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"\n🔄 Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")
        process_symbol_batch(batch, profile_dict)

        # Brief pause between batches
        import time
        time.sleep(5)

    print("✅ Equity quotes processing completed")