import requests
import psycopg2
import logging
import os
import sys
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"

def fetch_std_dev(symbol):
    """Fetch standard deviation for a given symbol."""
    url = f"https://financialmodelingprep.com/stable/technical-indicators/standarddeviation?symbol={symbol}&periodLength=756&timeframe=1day&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0 and 'standardDeviation' in data[0]:
            return symbol, data[0]['standardDeviation']
        return symbol, None
    except Exception as e:
        logger.error(f"Error fetching std dev for {symbol}: {e}")
        return symbol, None

def create_table(conn):
    """Create equity_std_dev table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_std_dev (
                symbol VARCHAR(15) PRIMARY KEY,
                std_dev DOUBLE PRECISION
            )
        """)
    conn.commit()

def get_symbols(conn) -> List[str]:
    """Get list of symbols from equity_profile table."""
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM equity_profile")
        return [row[0] for row in cur.fetchall()]

def save_to_db(conn, results):
    """Save standard deviation data to database."""
    with conn.cursor() as cur:
        for symbol, std_dev in results:
            if std_dev is not None:
                try:
                    cur.execute("""
                        INSERT INTO equity_std_dev (symbol, std_dev)
                        VALUES (%s, %s)
                        ON CONFLICT (symbol) DO UPDATE SET
                            std_dev = EXCLUDED.std_dev
                    """, (symbol, std_dev))
                except Exception as e:
                    logger.error(f"Error saving data for {symbol}: {e}")
    conn.commit()

if __name__ == "__main__":
    conn = psycopg2.connect(**config.db_config)
    create_table(conn)
    symbols = get_symbols(conn)

    print(f"Fetching standard deviation for {len(symbols)} symbols...")
    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_std_dev, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
            symbol, std_dev = future.result()
            if std_dev is not None:
                results.append((symbol, std_dev))

    if results:
        print(f"Saving {len(results)} records to database...")
        save_to_db(conn, results)
        print("✅ Standard deviation data saved successfully")
    else:
        print("No data to save")

    conn.close()
