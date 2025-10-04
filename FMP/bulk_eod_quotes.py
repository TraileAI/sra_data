import requests
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

load_dotenv()

FMP_API_KEY = os.getenv('FMP_API_KEY')
FMP_BULK_EOD_URL = os.getenv('FMP_BULK_EOD_URL')
DB_CONFIG = config.db_config

def download_bulk_eod(date_str=None):
    """Download bulk EOD data from FMP API.

    Args:
        date_str: Date in YYYY-MM-DD format. If None, uses yesterday's date.

    Returns:
        DataFrame with columns: symbol, date, open, low, high, close, adjClose, volume
    """
    if date_str is None:
        # Default to yesterday's date (market data is typically available next day)
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime('%Y-%m-%d')

    url = f"{FMP_BULK_EOD_URL}?date={date_str}&apikey={FMP_API_KEY}"

    print(f"📥 Downloading bulk EOD data for {date_str}...")
    print(f"   URL: {FMP_BULK_EOD_URL}?date={date_str}&apikey=***")

    response = requests.get(url, timeout=300)
    response.raise_for_status()

    # Parse CSV response
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    print(f"✅ Downloaded {len(df):,} records")

    return df

def load_equity_quotes(df, conn):
    """Load equity quotes to database.

    Args:
        df: DataFrame with quote data
        conn: Database connection
    """
    # Get list of equity symbols from equity_profile
    cur = conn.cursor()
    cur.execute("SELECT symbol FROM equity_profile")
    equity_symbols = set(row[0] for row in cur.fetchall())

    print(f"   Found {len(equity_symbols):,} equity symbols in database")

    # Filter to only equity symbols
    equity_df = df[df['symbol'].isin(equity_symbols)].copy()

    if len(equity_df) == 0:
        print(f"   ⚠️  No equity quotes found in bulk data")
        return 0

    print(f"   📊 Matched {len(equity_df):,} equity quotes")

    # Normalize column names to lowercase
    equity_df.columns = equity_df.columns.str.lower()

    # Delete existing records for this date before inserting
    if len(equity_df) > 0:
        date_val = equity_df['date'].iloc[0]
        cur.execute("DELETE FROM equity_quotes WHERE date = %s", (date_val,))
        deleted_count = cur.rowcount
        print(f"   🗑️  Deleted {deleted_count:,} existing equity_quotes records for {date_val}")

    # Insert new records
    insert_query = """
        INSERT INTO equity_quotes (symbol, date, open, low, high, close, adjclose, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    records = [
        (row['symbol'], row['date'], row['open'], row['low'], row['high'],
         row['close'], row['adjclose'], row['volume'])
        for _, row in equity_df.iterrows()
    ]

    cur.executemany(insert_query, records)
    conn.commit()

    print(f"   ✅ Inserted {len(records):,} equity_quotes records")

    return len(records)

def load_etfs_quotes(df, conn):
    """Load ETF quotes to database.

    Args:
        df: DataFrame with quote data
        conn: Database connection
    """
    # Get list of ETF symbols from etfs_profile
    cur = conn.cursor()
    cur.execute("SELECT symbol FROM etfs_profile")
    etf_symbols = set(row[0] for row in cur.fetchall())

    print(f"   Found {len(etf_symbols):,} ETF symbols in database")

    # Filter to only ETF symbols
    etf_df = df[df['symbol'].isin(etf_symbols)].copy()

    if len(etf_df) == 0:
        print(f"   ⚠️  No ETF quotes found in bulk data")
        return 0

    print(f"   📊 Matched {len(etf_df):,} ETF quotes")

    # Normalize column names to lowercase
    etf_df.columns = etf_df.columns.str.lower()

    # Delete existing records for this date before inserting
    if len(etf_df) > 0:
        date_val = etf_df['date'].iloc[0]
        cur.execute("DELETE FROM etfs_quotes WHERE date = %s", (date_val,))
        deleted_count = cur.rowcount
        print(f"   🗑️  Deleted {deleted_count:,} existing etfs_quotes records for {date_val}")

    # Insert new records
    insert_query = """
        INSERT INTO etfs_quotes (symbol, date, open, low, high, close, adjclose, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    records = [
        (row['symbol'], row['date'], row['open'], row['low'], row['high'],
         row['close'], row['adjclose'], row['volume'])
        for _, row in etf_df.iterrows()
    ]

    cur.executemany(insert_query, records)
    conn.commit()

    print(f"   ✅ Inserted {len(records):,} etfs_quotes records")

    return len(records)

if __name__ == "__main__":
    try:
        # Check for date argument
        date_str = sys.argv[1] if len(sys.argv) > 1 else None

        # Download bulk EOD data
        df = download_bulk_eod(date_str)

        # Connect to database
        conn = psycopg2.connect(**DB_CONFIG)

        # Load equity quotes
        print("")
        print("📈 Loading equity quotes...")
        equity_count = load_equity_quotes(df, conn)

        # Load ETF quotes
        print("")
        print("📊 Loading ETF quotes...")
        etf_count = load_etfs_quotes(df, conn)

        # Close connection
        conn.close()

        print("")
        print(f"✅ Bulk EOD quotes update completed!")
        print(f"   Total records loaded: {equity_count + etf_count:,}")
        print(f"   - Equity quotes: {equity_count:,}")
        print(f"   - ETF quotes: {etf_count:,}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
