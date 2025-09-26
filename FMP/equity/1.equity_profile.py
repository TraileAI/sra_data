import requests
import psycopg2
import logging
import os
import io
import csv
from typing import List

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

FMP_API_KEY = os.getenv('FMP_API_KEY', "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT")
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def get_preliminary_us_tickers() -> List[str]:
    base_url = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        'exchange': 'nyse,nasdaq,amex',
        'country': 'US',
        'priceMoreThan': 1,
        'volumeMoreThan': 50000,
        'isEtf': 'false',
        'isActivelyTrading': 'true',
        'limit': 10000,
        'apikey': FMP_API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return sorted(set(item['symbol'] for item in response.json()))
    else:
        logger.error(f"Error fetching US tickers: {response.status_code} - {response.text}")
        return []

def get_preliminary_cad_tickers() -> List[str]:
    base_url = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        'exchange': 'tsx,tsxv',
        'country': 'CA',
        'priceMoreThan': 1,
        'volumeMoreThan': 10000,
        'isEtf': 'false',
        'isActivelyTrading': 'true',
        'limit': 10000,
        'apikey': FMP_API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return sorted(set(item['symbol'] for item in response.json()))
    else:
        logger.error(f"Error fetching CAD tickers: {response.status_code} - {response.text}")
        return []

def get_company_profiles(tickers: List[str]) -> List[dict]:
    chunk_size = 20
    profiles = []
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size

    for i, start_idx in enumerate(range(0, len(tickers), chunk_size)):
        chunk = tickers[start_idx:start_idx + chunk_size]
        print(f"📈 Fetching profiles batch {i+1}/{total_chunks} ({len(chunk)} symbols: {', '.join(chunk[:3])}{'...' if len(chunk) > 3 else ''})")

        url = f"https://financialmodelingprep.com/api/v3/profile/{','.join(chunk)}?apikey={FMP_API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            batch_profiles = response.json()
            profiles.extend(batch_profiles)
            print(f"✅ Batch {i+1}: Retrieved {len(batch_profiles)} profiles")
        except Exception as e:
            print(f"❌ Batch {i+1}: Error fetching profiles for {chunk}: {e}")
            logger.error(f"Error fetching profiles for {chunk}: {e}")

    print(f"🎉 Profile fetching completed: {len(profiles)} total profiles retrieved")
    return profiles

def create_equity_profile_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_profile (
                symbol VARCHAR(50) PRIMARY KEY,
                price DOUBLE PRECISION,
                beta DOUBLE PRECISION,
                vol_avg BIGINT,
                mkt_cap BIGINT,
                last_div DOUBLE PRECISION,
                range_str VARCHAR(50),
                changes DOUBLE PRECISION,
                company_name VARCHAR(255),
                currency VARCHAR(10),
                cik VARCHAR(20),
                isin VARCHAR(20),
                cusip VARCHAR(20),
                exchange VARCHAR(100),
                exchange_short_name VARCHAR(50),
                industry VARCHAR(100),
                website VARCHAR(255),
                description TEXT,
                ceo VARCHAR(100),
                sector VARCHAR(100),
                country VARCHAR(50),
                full_time_employees BIGINT,
                phone VARCHAR(50),
                address VARCHAR(255),
                city VARCHAR(100),
                state VARCHAR(50),
                zip_code VARCHAR(20),
                dcf_diff DOUBLE PRECISION,
                dcf DOUBLE PRECISION,
                image VARCHAR(255),
                ipo_date DATE,
                default_image BOOLEAN,
                is_etf BOOLEAN,
                is_actively_trading BOOLEAN,
                is_adr BOOLEAN,
                is_fund BOOLEAN
            )
        """)
        conn.commit()

def to_int(value):
    if value is not None:
        try:
            return int(value)
        except ValueError:
            return None
    return None

def to_float(value):
    if value is not None:
        try:
            return float(value)
        except ValueError:
            return None
    return None

def to_date(value):
    if value and isinstance(value, str) and len(value.strip()) > 0:
        return value
    return None

def to_bool(value):
    if value is not None:
        return bool(value)
    return None

def prepare_profile_row(profile: dict):
    """Convert profile dict to row tuple for COPY operation."""
    if not profile:
        return None

    return (
        profile.get('symbol'),
        to_float(profile.get('price')),
        to_float(profile.get('beta')),
        to_int(profile.get('volAvg')),
        to_int(profile.get('mktCap')),
        to_float(profile.get('lastDiv')),
        profile.get('range'),
        to_float(profile.get('changes')),
        profile.get('companyName'),
        profile.get('currency'),
        profile.get('cik'),
        profile.get('isin'),
        profile.get('cusip'),
        profile.get('exchange'),
        profile.get('exchangeShortName'),
        profile.get('industry'),
        profile.get('website'),
        profile.get('description'),
        profile.get('ceo'),
        profile.get('sector'),
        profile.get('country'),
        to_int(profile.get('fullTimeEmployees')),
        profile.get('phone'),
        profile.get('address'),
        profile.get('city'),
        profile.get('state'),
        profile.get('zip'),
        to_float(profile.get('dcfDiff')),
        to_float(profile.get('dcf')),
        profile.get('image'),
        to_date(profile.get('ipoDate')),
        to_bool(profile.get('defaultImage')),
        to_bool(profile.get('isEtf')),
        to_bool(profile.get('isActivelyTrading')),
        to_bool(profile.get('isAdr')),
        to_bool(profile.get('isFund'))
    )

def bulk_store_profiles_in_db(profiles: List[dict], conn):
    """Use COPY for bulk insert of profiles for better performance."""
    if not profiles:
        return

    # Prepare data as CSV in memory
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, delimiter='\t')

    for profile in profiles:
        row = prepare_profile_row(profile)
        if row and row[0]:  # Ensure symbol exists
            writer.writerow(row)

    csv_buffer.seek(0)

    with conn.cursor() as cur:
        # Create temporary table for COPY
        cur.execute("""
            CREATE TEMP TABLE temp_equity_profile (LIKE equity_profile INCLUDING DEFAULTS)
        """)

        # Use COPY for fast bulk insert
        cur.copy_from(
            csv_buffer,
            'temp_equity_profile',
            sep='\t',
            null='',
            columns=(
                'symbol', 'price', 'beta', 'vol_avg', 'mkt_cap', 'last_div', 'range_str', 'changes',
                'company_name', 'currency', 'cik', 'isin', 'cusip', 'exchange', 'exchange_short_name',
                'industry', 'website', 'description', 'ceo', 'sector', 'country', 'full_time_employees',
                'phone', 'address', 'city', 'state', 'zip_code', 'dcf_diff', 'dcf', 'image', 'ipo_date',
                'default_image', 'is_etf', 'is_actively_trading', 'is_adr', 'is_fund'
            )
        )

        # Insert from temp table with ON CONFLICT handling
        cur.execute("""
            INSERT INTO equity_profile SELECT * FROM temp_equity_profile
            ON CONFLICT (symbol) DO UPDATE SET
                price = EXCLUDED.price,
                beta = EXCLUDED.beta,
                vol_avg = EXCLUDED.vol_avg,
                mkt_cap = EXCLUDED.mkt_cap,
                last_div = EXCLUDED.last_div,
                range_str = EXCLUDED.range_str,
                changes = EXCLUDED.changes,
                company_name = EXCLUDED.company_name,
                currency = EXCLUDED.currency,
                cik = EXCLUDED.cik,
                isin = EXCLUDED.isin,
                cusip = EXCLUDED.cusip,
                exchange = EXCLUDED.exchange,
                exchange_short_name = EXCLUDED.exchange_short_name,
                industry = EXCLUDED.industry,
                website = EXCLUDED.website,
                description = EXCLUDED.description,
                ceo = EXCLUDED.ceo,
                sector = EXCLUDED.sector,
                country = EXCLUDED.country,
                full_time_employees = EXCLUDED.full_time_employees,
                phone = EXCLUDED.phone,
                address = EXCLUDED.address,
                city = EXCLUDED.city,
                state = EXCLUDED.state,
                zip_code = EXCLUDED.zip_code,
                dcf_diff = EXCLUDED.dcf_diff,
                dcf = EXCLUDED.dcf,
                image = EXCLUDED.image,
                ipo_date = EXCLUDED.ipo_date,
                default_image = EXCLUDED.default_image,
                is_etf = EXCLUDED.is_etf,
                is_actively_trading = EXCLUDED.is_actively_trading,
                is_adr = EXCLUDED.is_adr,
                is_fund = EXCLUDED.is_fund
        """)

        conn.commit()

if __name__ == "__main__":
    print("🔍 Fetching US and Canadian tickers...")
    us_tickers = get_preliminary_us_tickers()
    print(f"📊 Found {len(us_tickers)} US tickers")

    cad_tickers = get_preliminary_cad_tickers()
    print(f"📊 Found {len(cad_tickers)} Canadian tickers")

    tickers = us_tickers + cad_tickers
    print(f"📊 Total tickers to process: {len(tickers)}")

    print("🚀 Fetching company profiles...")
    profiles = get_company_profiles(tickers)
    print(f"📊 Successfully fetched {len(profiles)} profiles")

    print("🔗 Connecting to database...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    try:
        print("📋 Creating equity profile table...")
        create_equity_profile_table(conn)
        print(f"💾 Processing {len(profiles)} profiles using bulk COPY...")
        bulk_store_profiles_in_db(profiles, conn)
        print("✅ Bulk profile insert completed successfully")
    finally:
        conn.close()