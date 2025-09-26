import requests
import psycopg2
import logging
import time
import os
from typing import List
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Import new resource-aware components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib import (
    wait_for_resources, csv_buffer_context, create_checkpoint,
    save_checkpoint, load_checkpoint, should_resume_processing,
    wait_for_recovery, cleanup_checkpoint, update_checkpoint_progress
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

FMP_API_KEY = os.getenv('FMP_API_KEY')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT', '5432'))
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Validate required environment variables
if not FMP_API_KEY:
    raise ValueError("FMP_API_KEY must be set in .env file")
if not all([DB_HOST, DB_NAME, DB_USER]):
    raise ValueError("DB_HOST, DB_NAME, and DB_USER must be set in .env file")

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

def get_company_profiles(tickers: List[str], checkpoint=None) -> List[dict]:
    """Fetch company profiles with resource-aware processing and checkpointing."""
    chunk_size = 20
    profiles = []
    start_idx = 0

    # Resume from checkpoint if available
    if checkpoint:
        completed_symbols = set(checkpoint.completed_symbols)
        tickers = [t for t in tickers if t not in completed_symbols]
        print(f"Resuming from checkpoint: {len(completed_symbols)} symbols already completed")

    script_checkpoint = checkpoint or create_checkpoint(
        script_name="equity_profile",
        total_symbols=len(tickers)
    )

    try:
        for i in tqdm(range(start_idx, len(tickers), chunk_size)):
            # Wait for resources before making API call
            if not wait_for_resources(max_wait=300.0):
                print("Resource constraints - stopping processing")
                break

            chunk = tickers[i:i + chunk_size]
            url = f"https://financialmodelingprep.com/api/v3/profile/{','.join(chunk)}?apikey={FMP_API_KEY}"

            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                chunk_profiles = response.json()
                profiles.extend(chunk_profiles)

                # Update checkpoint
                script_checkpoint = update_checkpoint_progress(
                    script_checkpoint,
                    new_api_calls=1,
                    new_symbol=chunk[-1] if chunk else None,
                    completed_symbol=chunk[-1] if chunk_profiles else None
                )
                save_checkpoint(script_checkpoint)

                # Rate limiting - 25 calls/second
                time.sleep(0.04)

            except Exception as e:
                logger.error(f"Error fetching profiles for {chunk}: {e}")
                # Update checkpoint with failed symbols
                for symbol in chunk:
                    script_checkpoint = update_checkpoint_progress(
                        script_checkpoint,
                        failed_symbol=symbol
                    )
                save_checkpoint(script_checkpoint)

        return profiles

    except KeyboardInterrupt:
        print("Processing interrupted - checkpoint saved")
        save_checkpoint(script_checkpoint)
        raise

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

def extract_profile_data(profile: dict) -> dict:
    """Extract and format profile data for CSV buffering."""
    if not profile:
        return None

    # Extract all fields in the same order as column definition
    return {
        'symbol': profile.get('symbol'),
        'price': to_float(profile.get('price')),
        'beta': to_float(profile.get('beta')),
        'vol_avg': to_int(profile.get('volAvg')),
        'mkt_cap': to_int(profile.get('mktCap')),
        'last_div': to_float(profile.get('lastDiv')),
        'range_str': profile.get('range'),
        'changes': to_float(profile.get('changes')),
        'company_name': profile.get('companyName'),
        'currency': profile.get('currency'),
        'cik': profile.get('cik'),
        'isin': profile.get('isin'),
        'cusip': profile.get('cusip'),
        'exchange': profile.get('exchange'),
        'exchange_short_name': profile.get('exchangeShortName'),
        'industry': profile.get('industry'),
        'website': profile.get('website'),
        'description': profile.get('description'),
        'ceo': profile.get('ceo'),
        'sector': profile.get('sector'),
        'country': profile.get('country'),
        'full_time_employees': to_int(profile.get('fullTimeEmployees')),
        'phone': profile.get('phone'),
        'address': profile.get('address'),
        'city': profile.get('city'),
        'state': profile.get('state'),
        'zip_code': profile.get('zip'),
        'dcf_diff': to_float(profile.get('dcfDiff')),
        'dcf': to_float(profile.get('dcf')),
        'image': profile.get('image'),
        'ipo_date': to_date(profile.get('ipoDate')),
        'default_image': to_bool(profile.get('defaultImage')),
        'is_etf': to_bool(profile.get('isEtf')),
        'is_actively_trading': to_bool(profile.get('isActivelyTrading')),
        'is_adr': to_bool(profile.get('isAdr')),
        'is_fund': to_bool(profile.get('isFund'))
    }

def store_profile_in_db(profile: dict, conn):
    if not profile:
        return
    symbol = profile.get('symbol')
    price = to_float(profile.get('price'))
    beta = to_float(profile.get('beta'))
    vol_avg = to_int(profile.get('volAvg'))
    mkt_cap = to_int(profile.get('mktCap'))
    last_div = to_float(profile.get('lastDiv'))
    range_str = profile.get('range')
    changes = to_float(profile.get('changes'))
    company_name = profile.get('companyName')
    currency = profile.get('currency')
    cik = profile.get('cik')
    isin = profile.get('isin')
    cusip = profile.get('cusip')
    exchange = profile.get('exchange')
    exchange_short_name = profile.get('exchangeShortName')
    industry = profile.get('industry')
    website = profile.get('website')
    description = profile.get('description')
    ceo = profile.get('ceo')
    sector = profile.get('sector')
    country = profile.get('country')
    full_time_employees = to_int(profile.get('fullTimeEmployees'))
    phone = profile.get('phone')
    address = profile.get('address')
    city = profile.get('city')
    state = profile.get('state')
    zip_code = profile.get('zip')
    dcf_diff = to_float(profile.get('dcfDiff'))
    dcf = to_float(profile.get('dcf'))
    image = profile.get('image')
    ipo_date = to_date(profile.get('ipoDate'))
    default_image = to_bool(profile.get('defaultImage'))
    is_etf = to_bool(profile.get('isEtf'))
    is_actively_trading = to_bool(profile.get('isActivelyTrading'))
    is_adr = to_bool(profile.get('isAdr'))
    is_fund = to_bool(profile.get('isFund'))
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO equity_profile (
                symbol, price, beta, vol_avg, mkt_cap, last_div, range_str, changes,
                company_name, currency, cik, isin, cusip, exchange, exchange_short_name,
                industry, website, description, ceo, sector, country, full_time_employees,
                phone, address, city, state, zip_code, dcf_diff, dcf, image, ipo_date,
                default_image, is_etf, is_actively_trading, is_adr, is_fund
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
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
        """, (
            symbol, price, beta, vol_avg, mkt_cap, last_div, range_str, changes,
            company_name, currency, cik, isin, cusip, exchange, exchange_short_name,
            industry, website, description, ceo, sector, country, full_time_employees,
            phone, address, city, state, zip_code, dcf_diff, dcf, image, ipo_date,
            default_image, is_etf, is_actively_trading, is_adr, is_fund
        ))
        conn.commit()

if __name__ == "__main__":
    script_name = "equity_profile"

    # Check for existing checkpoint and wait for recovery if needed
    if should_resume_processing(script_name):
        if not wait_for_recovery(script_name):
            print("Failed to wait for system recovery")
            exit(1)
        checkpoint = load_checkpoint(script_name)
    else:
        checkpoint = None

    # Create SQLAlchemy engine for CSV buffer
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # Create connection for table setup
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    try:
        create_equity_profile_table(conn)
        conn.close()

        # Fetch tickers
        tickers = get_preliminary_us_tickers() + get_preliminary_cad_tickers()

        # Get profiles with resource-aware processing
        profiles = get_company_profiles(tickers, checkpoint)

        if not profiles:
            print("No profiles retrieved")
            cleanup_checkpoint(script_name)
            exit(0)

        # Define column order for CSV
        columns = [
            'symbol', 'price', 'beta', 'vol_avg', 'mkt_cap', 'last_div', 'range_str', 'changes',
            'company_name', 'currency', 'cik', 'isin', 'cusip', 'exchange', 'exchange_short_name',
            'industry', 'website', 'description', 'ceo', 'sector', 'country', 'full_time_employees',
            'phone', 'address', 'city', 'state', 'zip_code', 'dcf_diff', 'dcf', 'image', 'ipo_date',
            'default_image', 'is_etf', 'is_actively_trading', 'is_adr', 'is_fund'
        ]

        # Process profiles with CSV buffering
        with csv_buffer_context('equity_profile', columns, engine):
            for profile in tqdm(profiles, desc="Processing profiles"):
                if not wait_for_resources(max_wait=60.0):
                    print("Resource constraints - stopping profile processing")
                    break

                profile_data = extract_profile_data(profile)
                if profile_data:
                    from lib.csv_buffer import add_row
                    add_row('equity_profile', profile_data)

        print("Processing completed successfully")
        cleanup_checkpoint(script_name)

    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise
    finally:
        if 'conn' in locals() and not conn.closed:
            conn.close()