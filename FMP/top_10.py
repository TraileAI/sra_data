"""
Calculate market-wide top 10 gainers and losers.
This script uses FMP API to get the current day's top performers across all markets.
"""
import requests
import psycopg2
import logging
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"

def create_table(conn):
    """Create top_10 table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS top_10 (
                symbol VARCHAR(15),
                changesPercentage DOUBLE PRECISION,
                type VARCHAR(10),
                rank INTEGER,
                PRIMARY KEY (type, rank)
            )
        """)
    conn.commit()

def fetch_gainers_losers(conn):
    """Fetch top gainers and losers from FMP API."""
    with conn.cursor() as cur:
        # Get list of symbols from equity_profile
        cur.execute("SELECT symbol FROM equity_profile")
        profile_symbols = set(row[0] for row in cur.fetchall())

    logger.info(f"Found {len(profile_symbols)} symbols in equity_profile")

    # Fetch gainers
    gainers_url = f"https://financialmodelingprep.com/api/v3/stock_market/gainers?apikey={FMP_API_KEY}"
    try:
        response = requests.get(gainers_url, timeout=10)
        response.raise_for_status()
        gainers_data = response.json()
        logger.info(f"Fetched {len(gainers_data)} gainers from API")

        # Filter to only include symbols in our equity_profile
        gainers = [
            {'symbol': item['symbol'], 'changesPercentage': item.get('changesPercentage', 0)}
            for item in gainers_data
            if item['symbol'] in profile_symbols
        ]
        # Sort by changes percentage descending and take top 10
        gainers = sorted(gainers, key=lambda x: x['changesPercentage'], reverse=True)[:10]
        logger.info(f"Filtered to {len(gainers)} gainers in our portfolio")

    except Exception as e:
        logger.error(f"Error fetching gainers: {e}")
        gainers = []

    # Fetch losers
    losers_url = f"https://financialmodelingprep.com/api/v3/stock_market/losers?apikey={FMP_API_KEY}"
    try:
        response = requests.get(losers_url, timeout=10)
        response.raise_for_status()
        losers_data = response.json()
        logger.info(f"Fetched {len(losers_data)} losers from API")

        # Filter to only include symbols in our equity_profile
        losers = [
            {'symbol': item['symbol'], 'changesPercentage': item.get('changesPercentage', 0)}
            for item in losers_data
            if item['symbol'] in profile_symbols
        ]
        # Sort by changes percentage ascending and take top 10
        losers = sorted(losers, key=lambda x: x['changesPercentage'])[:10]
        logger.info(f"Filtered to {len(losers)} losers in our portfolio")

    except Exception as e:
        logger.error(f"Error fetching losers: {e}")
        losers = []

    return gainers, losers

def save_to_db(conn, gainers, losers):
    """Save top gainers and losers to database."""
    with conn.cursor() as cur:
        # Clear existing data
        cur.execute("TRUNCATE TABLE top_10")

        # Insert gainers
        for rank, item in enumerate(gainers, 1):
            cur.execute("""
                INSERT INTO top_10 (symbol, changesPercentage, type, rank)
                VALUES (%s, %s, %s, %s)
            """, (item['symbol'], item['changesPercentage'], 'gainer', rank))

        # Insert losers
        for rank, item in enumerate(losers, 1):
            cur.execute("""
                INSERT INTO top_10 (symbol, changesPercentage, type, rank)
                VALUES (%s, %s, %s, %s)
            """, (item['symbol'], item['changesPercentage'], 'loser', rank))

    conn.commit()
    logger.info(f"✅ Saved {len(gainers)} gainers and {len(losers)} losers")

if __name__ == "__main__":
    conn = psycopg2.connect(**config.db_config)
    create_table(conn)

    gainers, losers = fetch_gainers_losers(conn)
    save_to_db(conn, gainers, losers)

    conn.close()
