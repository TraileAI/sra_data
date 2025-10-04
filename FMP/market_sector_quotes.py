import requests
import psycopg2
import logging
import os
import sys
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"

# Market and sector symbols to track
SYMBOLS = ['XIU.TO', '^GSPC', 'QQQ', 'DIA', 'EURUSD', 'SPY', 'XLK', 'XLV', 'XLF',
           'XLY', 'XLI', 'XLB', 'XLE', 'XLU', 'XLRE', 'XLC', 'XLP']

def fetch_price_history(symbol):
    """Fetch historical price data for a symbol."""
    start_date = (datetime.now() - timedelta(days=800)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={today}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        data = json_data.get('historical', [])
        return symbol, data
    except Exception as e:
        logger.error(f"Error fetching price history for {symbol}: {e}")
        return symbol, []

def create_table(conn):
    """Create market_sector_quotes table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_sector_quotes (
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
                symbol VARCHAR(15),
                PRIMARY KEY (symbol, date)
            )
        """)
    conn.commit()

def save_to_db(conn, symbol, data):
    """Save price history to database."""
    if not data:
        return

    with conn.cursor() as cur:
        for record in data:
            try:
                cur.execute("""
                    INSERT INTO market_sector_quotes (
                        date, open, high, low, close, adjClose, volume,
                        unadjustedVolume, change, changePercent, vwap,
                        label, changeOverTime, symbol
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                """, (
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
                    record.get('changeOverTime'),
                    symbol
                ))
            except Exception as e:
                logger.error(f"Error saving data for {symbol} on {record.get('date')}: {e}")
    conn.commit()

if __name__ == "__main__":
    conn = psycopg2.connect(**config.db_config)
    create_table(conn)

    print(f"Fetching price history for {len(SYMBOLS)} market/sector symbols...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_price_history, symbol) for symbol in SYMBOLS]
        for future in tqdm(as_completed(futures), total=len(SYMBOLS), desc="Processing symbols"):
            symbol, data = future.result()
            if data:
                save_to_db(conn, symbol, data)
                print(f"Saved {len(data)} records for {symbol}")
            else:
                print(f"No data for {symbol}")

    print("✅ Market sector quotes updated successfully")
    conn.close()
