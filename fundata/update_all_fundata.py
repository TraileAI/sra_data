#!/usr/bin/env python3
"""
Update all Fundata tables from API
"""
import requests
import time
import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, date
from dotenv import load_dotenv
from io import StringIO
import re

load_dotenv()

def camel_to_snake(name):
    """Convert camelCase/PascalCase to snake_case."""
    # Insert underscore before uppercase letters
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # Handle sequences of capitals
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

# Database configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Fundata API configuration
AUTH_URL = "https://authenticate.fundataapi.com/authenticate"
SYNC_BASE_URL = "https://dbsync.fundataapi.com"

# Table to endpoint mapping with their date columns and transformations
TABLE_ENDPOINTS = {
    'fund_quotes': {
        'endpoint': 'fund/getdailynavps',
        'date_column': 'date',
        'transform': 'fund_quotes'
    },
    'fund_general': {
        'endpoint': 'fund/getgeneral',
        'date_column': 'change_date',
        'params': {'language': 'en'}
    },
    'fund_performance_summary': {
        'endpoint': 'fund/getperformancesummary',
        'date_column': 'change_date'
    },
    'fund_distribution': {
        'endpoint': 'fund/getdistribution',
        'date_column': 'change_date'
    },
    'fund_top_holdings': {
        'endpoint': 'fund/gettopholding',
        'date_column': 'change_date'
    },
    'fund_allocation': {
        'endpoint': 'fund/getallocations',
        'date_column': 'change_date'
    },
    'fund_loads': {
        'endpoint': 'fund/getload',
        'date_column': 'change_date'
    },
    'fund_other_fees': {
        'endpoint': 'fund/getotherfees',
        'date_column': 'change_date'
    },
    'fund_expenses': {
        'endpoint': 'fund/getexpense',
        'date_column': 'change_date'
    },
    'fund_trailer_schedule': {
        'endpoint': 'fund/gettrailerschedule',
        'date_column': 'change_date'
    },
    'fund_assets': {
        'endpoint': 'fund/getassets',
        'date_column': 'change_date'
    },
    'fund_associate_benchmark': {
        'endpoint': 'fund/getassociatedbenchmark',
        'date_column': 'change_date'
    },
    'fund_equity_style': {
        'endpoint': 'fund/getequitystyle',
        'date_column': 'change_date'
    },
    'fund_fixed_income_style': {
        'endpoint': 'fund/getfixedincomestyle',
        'date_column': 'change_date'
    },
    'fund_advanced_performance': {
        'endpoint': 'fund/getadvancedperformance',
        'date_column': 'change_date'
    },
    'fund_yearly_performance': {
        'endpoint': 'fund/getyearlyperformance',
        'date_column': 'change_date'
    },
    'fund_yearly_performance_ranking': {
        'endpoint': 'fund/getyearlyperformanceranking',
        'date_column': 'change_date'
    },
    'fund_risk_yearly_performance': {
        'endpoint': 'fund/getyearlyperformancerisk',
        'date_column': 'change_date'
    },
    'instrument_identifier': {
        'endpoint': 'instrument/getidentifiers',
        'date_column': 'change_date'
    },
    'benchmark_general': {
        'endpoint': 'benchmark/getgeneral',
        'date_column': 'change_date',
        'params': {'language': 'en'}
    },
    'benchmark_yearly_performance': {
        'endpoint': 'benchmark/getyearlyperformance',
        'date_column': 'change_date'
    }
}

def authenticate():
    """Authenticate with Fundata API and get JWT token."""
    print("🔑 Authenticating with Fundata API...")

    key_id = os.getenv('FUNADTA_ACCESS_KEY_ID') or os.getenv('FUNDATA_ACCESS_KEY_ID')
    secret = os.getenv('FUNDATA_SECRET_ACCESS_KEY')

    if not key_id or not secret:
        raise ValueError("Missing Fundata credentials in .env file")

    payload = {"key": key_id, "secret": secret}
    response = requests.post(AUTH_URL, json=payload, timeout=30)
    response.raise_for_status()

    token = response.json()['token']
    print("✅ Authentication successful")
    return token

def request_data(token, endpoint, from_date, extra_params=None):
    """Request data from Fundata API."""
    url = f"{SYNC_BASE_URL}/{endpoint}"
    headers = {"Authorization": token}
    params = {"fromdate": from_date}

    if extra_params:
        params.update(extra_params)

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    execution_id = response.json()['execution_id']
    return execution_id

def get_result(token, execution_id, max_retries=30, wait_seconds=10):
    """Poll for result URL (wait for CSV to be prepared)."""
    url = f"{SYNC_BASE_URL}/getresults/{execution_id}"
    headers = {"Authorization": token}

    for attempt in range(max_retries):
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        result = response.json()
        status = result.get('status')

        if status == 'Succeeded':
            return result['url']
        elif status == 'Running':
            if attempt == 0:
                print(f"   ⏳ Waiting for data preparation...", end="", flush=True)
            else:
                print(".", end="", flush=True)
            time.sleep(wait_seconds)
        else:
            raise Exception(f"Unexpected status: {status}")

    raise Exception(f"Timeout waiting for data after {max_retries * wait_seconds} seconds")

def download_csv(url):
    """Download the CSV file from Fundata."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))
    return df

def transform_fund_quotes(df):
    """Transform fund_quotes data."""
    transformed = pd.DataFrame({
        'instrument_key': df['InstrumentKey'].astype(int),
        'date': df['Date'].astype(str),
        'navps': df['NAVPS'].astype(str),
        'daily_total_distribution': None,
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
    return transformed

def get_max_date(engine, table_name, date_column):
    """Get the maximum date from a table."""
    with engine.connect() as conn:
        # Try as date first, fallback to varchar
        try:
            result = conn.execute(text(f"SELECT MAX({date_column}::date) as max_date FROM {table_name}"))
            max_date = result.fetchone()[0]
        except:
            result = conn.execute(text(f"SELECT {date_column} FROM {table_name} ORDER BY {date_column} DESC LIMIT 1"))
            row = result.fetchone()
            max_date = row[0] if row else None

    return max_date

def load_to_database(engine, table_name, df, date_column):
    """Load data to database with upsert logic."""
    # Get table schema to filter DataFrame columns
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            AND table_schema = 'public'
        """))
        db_columns = {row[0] for row in result.fetchall()}

    # Filter DataFrame to only include columns that exist in database
    df_columns = set(df.columns)
    columns_to_drop = df_columns - db_columns
    if columns_to_drop:
        df = df.drop(columns=list(columns_to_drop))

    # Filter to only new/updated records
    with engine.connect() as conn:
        current_max = get_max_date(engine, table_name, date_column)

        if current_max:
            if isinstance(current_max, date):
                current_max_str = current_max.strftime('%Y-%m-%d')
            else:
                current_max_str = str(current_max)

            # Filter based on date column
            df = df[df[date_column] > current_max_str]
            print(f"   Filtered to {len(df)} new records after {current_max_str}")

    if len(df) == 0:
        print(f"   ✅ No new records - {table_name} is up to date!")
        return 0

    # Delete existing records with matching record_id before inserting
    # This handles updates to existing records
    if 'record_id' in df.columns:
        record_ids = df['record_id'].tolist()
        with engine.begin() as conn:
            # Delete in batches to avoid SQL string size limits
            batch_size = 1000
            for i in range(0, len(record_ids), batch_size):
                batch = record_ids[i:i + batch_size]
                placeholders = ','.join([str(int(x)) for x in batch])
                conn.execute(text(f"DELETE FROM {table_name} WHERE record_id IN ({placeholders})"))

    # Append to table
    df.to_sql(table_name, engine, if_exists='append', index=False, method='multi')

    return len(df)

def update_table(engine, token, table_name, config):
    """Update a single table."""
    print(f"\n📊 Updating {table_name}...")

    # Get from_date
    date_column = config['date_column']
    from_date_obj = get_max_date(engine, table_name, date_column)

    if from_date_obj:
        if isinstance(from_date_obj, date):
            from_date = (from_date_obj + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # Try to parse and add a day
            try:
                parsed = pd.to_datetime(from_date_obj)
                from_date = (parsed + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            except:
                from_date = '2024-01-01'
    else:
        from_date = '2024-01-01'

    print(f"   📅 Fetching updates from: {from_date}")

    # Request data
    endpoint = config['endpoint']
    extra_params = config.get('params')
    print(f"   📡 Requesting: {endpoint}")

    execution_id = request_data(token, endpoint, from_date, extra_params)
    print(f"   ✅ Request submitted: {execution_id}")

    # Wait for result
    download_url = get_result(token, execution_id)
    print(f" ✅")

    # Download CSV
    print(f"   ⬇️  Downloading CSV...", end=" ", flush=True)
    df = download_csv(download_url)
    print(f"✅ ({len(df)} records)")

    # Transform if needed
    if config.get('transform') == 'fund_quotes':
        print(f"   🔄 Transforming data...", end=" ", flush=True)
        df = transform_fund_quotes(df)
        print("✅")
    else:
        # Normalize column names from camelCase to snake_case
        df.columns = [camel_to_snake(col) for col in df.columns]

    # Load to database
    print(f"   💾 Loading to database...", end=" ", flush=True)
    loaded = load_to_database(engine, table_name, df, date_column)
    print(f"✅ ({loaded} new records)")

    return loaded

def main():
    """Main function to update all Fundata tables."""
    print("🚀 Starting comprehensive Fundata update...")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    token = authenticate()

    print(f"\n📋 Will update {len(TABLE_ENDPOINTS)} tables")

    total_records = 0
    successful = 0
    failed = []

    # Update each table
    for i, (table_name, config) in enumerate(TABLE_ENDPOINTS.items(), 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(TABLE_ENDPOINTS)}] {table_name}")
        print(f"{'='*60}")

        try:
            records = update_table(engine, token, table_name, config)
            total_records += records
            successful += 1
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed.append(table_name)

    # Summary
    print(f"\n{'='*60}")
    print("📊 UPDATE SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{len(TABLE_ENDPOINTS)}")
    print(f"📈 Total new records: {total_records:,}")

    if failed:
        print(f"\n❌ Failed tables ({len(failed)}):")
        for table in failed:
            print(f"   - {table}")
    else:
        print(f"\n🎉 All tables updated successfully!")

if __name__ == "__main__":
    main()
