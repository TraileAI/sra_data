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

def fetch_financial_scores(symbol):
    """Fetch financial scores for a given symbol."""
    url = f"https://financialmodelingprep.com/api/v4/score?symbol={symbol}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return symbol, response.json()
    except Exception as e:
        logger.error(f"Error fetching financial scores for {symbol}: {e}")
        return symbol, []

def create_table(conn):
    """Create equity_financial_scores table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_financial_scores (
                symbol VARCHAR(15) PRIMARY KEY,
                altmanZScore DOUBLE PRECISION,
                piotroskiScore INTEGER,
                workingCapital BIGINT,
                totalAssets BIGINT,
                retainedEarnings BIGINT,
                ebit BIGINT,
                marketCap BIGINT,
                totalLiabilities BIGINT,
                revenue BIGINT
            )
        """)
    conn.commit()

def get_symbols(conn) -> List[str]:
    """Get list of symbols from equity_profile table."""
    with conn.cursor() as cur:
        cur.execute("SELECT symbol FROM equity_profile")
        return [row[0] for row in cur.fetchall()]

def save_to_db(conn, data_list):
    """Save financial scores to database."""
    with conn.cursor() as cur:
        for data in data_list:
            if not data:
                continue
            try:
                cur.execute("""
                    INSERT INTO equity_financial_scores (
                        symbol, altmanZScore, piotroskiScore, workingCapital,
                        totalAssets, retainedEarnings, ebit, marketCap,
                        totalLiabilities, revenue
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                        altmanZScore = EXCLUDED.altmanZScore,
                        piotroskiScore = EXCLUDED.piotroskiScore,
                        workingCapital = EXCLUDED.workingCapital,
                        totalAssets = EXCLUDED.totalAssets,
                        retainedEarnings = EXCLUDED.retainedEarnings,
                        ebit = EXCLUDED.ebit,
                        marketCap = EXCLUDED.marketCap,
                        totalLiabilities = EXCLUDED.totalLiabilities,
                        revenue = EXCLUDED.revenue
                """, (
                    data.get('symbol'),
                    data.get('altmanZScore'),
                    data.get('piotroskiScore'),
                    data.get('workingCapital'),
                    data.get('totalAssets'),
                    data.get('retainedEarnings'),
                    data.get('ebit'),
                    data.get('marketCap'),
                    data.get('totalLiabilities'),
                    data.get('revenue')
                ))
            except Exception as e:
                logger.error(f"Error saving data for {data.get('symbol')}: {e}")
    conn.commit()

if __name__ == "__main__":
    conn = psycopg2.connect(**config.db_config)
    create_table(conn)
    symbols = get_symbols(conn)

    print(f"Fetching financial scores for {len(symbols)} symbols...")
    all_data = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_financial_scores, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
            symbol, data = future.result()
            if data:
                for record in data:
                    all_data.append(record)

    if all_data:
        print(f"Saving {len(all_data)} records to database...")
        save_to_db(conn, all_data)
        print("✅ Financial scores saved successfully")
    else:
        print("No data to save")

    conn.close()
