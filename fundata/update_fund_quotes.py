#!/usr/bin/env python3
"""
Update fund_quotes table with latest pricing data from Fundata API
"""
import requests
import time
import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, date
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Fundata API configuration
FUNDATA_ACCESS_KEY = os.getenv('FUNDATA_SECRET_ACCESS_KEY') or os.getenv('FUNADTA_ACCESS_KEY_ID')
FUNDATA_SECRET = os.getenv('FUNDATA_SECRET_ACCESS_KEY')

# API endpoints
AUTH_URL = "https://authenticate.fundataapi.com/authenticate"
SYNC_BASE_URL = "https://dbsync.fundataapi.com"

def authenticate():
    """Authenticate with Fundata API and get JWT token."""
    print("🔑 Authenticating with Fundata API...")

    # Try both possible key formats from .env
    key_id = os.getenv('FUNADTA_ACCESS_KEY_ID')  # Typo version
    if not key_id:
        key_id = os.getenv('FUNDATA_ACCESS_KEY_ID')

    secret = os.getenv('FUNDATA_SECRET_ACCESS_KEY')

    if not key_id or not secret:
        raise ValueError("Missing Fundata credentials in .env file")

    payload = {
        "key": key_id,
        "secret": secret
    }

    response = requests.post(AUTH_URL, json=payload, timeout=30)
    response.raise_for_status()

    token = response.json()['token']
    print("✅ Authentication successful")
    return token

def request_daily_prices(token, from_date):
    """Request daily price updates from Fundata API."""
    print(f"📊 Requesting daily prices from {from_date}...")

    url = f"{SYNC_BASE_URL}/fund/getdailynavps"
    headers = {"Authorization": token}
    params = {"fromdate": from_date}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    execution_id = response.json()['execution_id']
    print(f"✅ Request submitted: {execution_id}")
    return execution_id

def get_result(token, execution_id, max_retries=30, wait_seconds=10):
    """Poll for result URL (wait for CSV to be prepared)."""
    print(f"⏳ Waiting for data to be prepared...")

    url = f"{SYNC_BASE_URL}/getresults/{execution_id}"
    headers = {"Authorization": token}

    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        status = result.get('status')

        if status == 'Succeeded':
            download_url = result['url']
            print(f"✅ Data ready for download")
            return download_url
        elif status == 'Running':
            print(f"   Still preparing... ({attempt + 1}/{max_retries})")
            time.sleep(wait_seconds)
        else:
            raise Exception(f"Unexpected status: {status}")

    raise Exception(f"Timeout waiting for data after {max_retries * wait_seconds} seconds")

def download_csv(url):
    """Download the CSV file from Fundata."""
    print(f"⬇️  Downloading CSV data...")

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Parse CSV
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    print(f"✅ Downloaded {len(df)} price records")
    return df

def transform_data(df):
    """Transform Fundata CSV format to fund_quotes table format."""
    print("🔄 Transforming data...")

    # Fundata columns (from docs):
    # RecordId, InstrumentKey, CurrentYield, CurrentYieldPercentChange, Date, NAVPS,
    # NAVPSPennyChange, NAVPSPercentChange, PreviousDate, PreviousNAVPS, Split, RecordState, ChangeDate

    # Map to fund_quotes table schema (matching exact column names from database)
    transformed = pd.DataFrame({
        'instrument_key': df['InstrumentKey'].astype(int),
        'date': df['Date'].astype(str),  # Keep as string (varchar in DB)
        'navps': df['NAVPS'].astype(str),  # Keep as string (varchar in DB)
        'daily_total_distribution': None,  # Not in daily price API response
        'daily_dividend_income_distribution': None,
        'daily_foreign_dividend_income_distribution': None,
        'daily_capital_gains_distribution': None,
        'daily_interest_income_distribution': None,
        'daily_return_of_principle_distribution': None,
        'distribution_pay_date': None,
        'split_factor': df.get('Split', '').astype(str),
        'navps_percent_change': df.get('NAVPSPercentChange', '').astype(str),
        'penny_change_day': df.get('NAVPSPennyChange', '').astype(str),
        'current_yield': df.get('CurrentYield', '').astype(str),
        'current_yield_percent_change': df.get('CurrentYieldPercentChange', '').astype(str),
    })

    print(f"✅ Transformed {len(transformed)} records")
    return transformed

def load_to_database(df):
    """Load data into fund_quotes table."""
    print("💾 Loading data to database...")

    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # Check current max date in fund_quotes (date is stored as varchar)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(date::date) as max_date FROM fund_quotes"))
        current_max = result.fetchone()[0]
        print(f"   Current max date in fund_quotes: {current_max}")

    # Filter to only new records
    if current_max:
        current_max_str = current_max.strftime('%Y-%m-%d')
        df = df[df['date'] > current_max_str]
        print(f"   Filtered to {len(df)} new records after {current_max_str}")

    if len(df) == 0:
        print("✅ No new records to load - fund_quotes is up to date!")
        return

    # Append to table (using upsert to handle duplicates)
    print(f"   Loading {len(df)} records...")

    # Use pandas to_sql with on_conflict handling
    # Since we can't use on_conflict_do_nothing directly with pandas, we'll filter before inserting
    df.to_sql('fund_quotes', engine, if_exists='append', index=False, method='multi')

    print(f"✅ Loaded {len(df)} new records to fund_quotes")

    # Verify
    with engine.connect() as conn:
        result = conn.execute(text("SELECT MAX(date::date) as max_date, COUNT(*) as total FROM fund_quotes"))
        row = result.fetchone()
        print(f"   New max date: {row[0]}, Total records: {row[1]:,}")

def update_fund_quotes():
    """Main function to update fund quotes."""
    print("🚀 Starting fund_quotes update from Fundata API...")

    try:
        # Determine from_date (last update + 1 day, or default to start of this year)
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

        with engine.connect() as conn:
            # date column is varchar, so cast to date
            result = conn.execute(text("SELECT MAX(date::date) as max_date FROM fund_quotes"))
            max_date = result.fetchone()[0]

        if max_date:
            # max_date is already a date object from Postgres
            from_date = (max_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            from_date = '2024-01-01'

        print(f"📅 Fetching updates from: {from_date}")

        # Authenticate
        token = authenticate()

        # Request data
        execution_id = request_daily_prices(token, from_date)

        # Wait for result
        download_url = get_result(token, execution_id)

        # Download CSV
        df = download_csv(download_url)

        # Transform data
        transformed_df = transform_data(df)

        # Load to database
        load_to_database(transformed_df)

        print("✅ Fund quotes update completed successfully!")

    except Exception as e:
        print(f"❌ Error updating fund quotes: {e}")
        raise

if __name__ == "__main__":
    update_fund_quotes()
