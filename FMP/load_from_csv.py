"""
FMP CSV loader - loads pre-extracted FMP data from CSV files to PostgreSQL.
Fast deployment using PostgreSQL COPY FROM for maximum performance.
"""
import os
import psycopg2
import logging
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

# Validate required environment variables
if not all([DB_HOST, DB_NAME, DB_USER]):
    raise ValueError("DB_HOST, DB_NAME, and DB_USER must be set in .env file")

# FMP CSV file mappings
FMP_CSV_TABLES = {
    'equity_profile.csv': 'equity_profile',
    'equity_income.csv': 'equity_income',
    'equity_balance.csv': 'equity_balance',
    'equity_cashflow.csv': 'equity_cashflow',
    'equity_quotes.csv': 'equity_quotes',
    'equity_peers.csv': 'equity_peers',
    'equity_financial_ratio.csv': 'equity_financial_ratio',
    'etfs_profile.csv': 'etfs_profile',
    'etfs_peers.csv': 'etfs_peers',
}

def get_fmp_csv_directory():
    """Get the FMP CSV data directory path."""
    return os.path.join(os.path.dirname(__file__), '..', 'fmp_data')

def create_tables(conn):
    """Create FMP tables if they don't exist."""
    logger.info("Creating FMP tables if needed...")

    with conn.cursor() as cur:
        # Equity profile table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_profile (
                symbol VARCHAR(15) PRIMARY KEY,
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

        # Equity income table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_income (
                date DATE,
                symbol VARCHAR(15),
                reported_currency VARCHAR(3),
                cik VARCHAR(10),
                filling_date DATE,
                accepted_date TIMESTAMP,
                calendar_year SMALLINT,
                period VARCHAR(3),
                revenue BIGINT,
                cost_of_revenue BIGINT,
                gross_profit BIGINT,
                gross_profit_ratio DOUBLE PRECISION,
                research_and_development_expenses BIGINT,
                general_and_administrative_expenses BIGINT,
                selling_and_marketing_expenses BIGINT,
                selling_general_and_administrative_expenses BIGINT,
                other_expenses BIGINT,
                operating_expenses BIGINT,
                cost_and_expenses BIGINT,
                interest_income BIGINT,
                interest_expense BIGINT,
                depreciation_and_amortization BIGINT,
                ebitda BIGINT,
                ebitdaratio DOUBLE PRECISION,
                operating_income BIGINT,
                operating_income_ratio DOUBLE PRECISION,
                total_other_income_expenses_net BIGINT,
                income_before_tax BIGINT,
                income_before_tax_ratio DOUBLE PRECISION,
                income_tax_expense BIGINT,
                net_income BIGINT,
                net_income_ratio DOUBLE PRECISION,
                eps DOUBLE PRECISION,
                epsdiluted DOUBLE PRECISION,
                weighted_average_shs_out BIGINT,
                weighted_average_shs_out_dil BIGINT,
                link TEXT,
                final_link TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity quotes table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_quotes (
                date DATE,
                symbol VARCHAR(15),
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                adj_close DOUBLE PRECISION,
                volume BIGINT,
                unadjusted_volume BIGINT,
                change DOUBLE PRECISION,
                change_percent DOUBLE PRECISION,
                vwap DOUBLE PRECISION,
                label VARCHAR(50),
                change_over_time DOUBLE PRECISION,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity peers table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_peers (
                symbol VARCHAR(15) NOT NULL,
                peer_symbol VARCHAR(15) NOT NULL,
                PRIMARY KEY (symbol, peer_symbol)
            )
        """)

        # ETFs profile table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS etfs_profile (
                symbol VARCHAR(15) PRIMARY KEY,
                company_name VARCHAR(255),
                sector VARCHAR(100),
                industry VARCHAR(100),
                country VARCHAR(50),
                website VARCHAR(255),
                description TEXT,
                exchange VARCHAR(100),
                exchange_short_name VARCHAR(50),
                currency VARCHAR(10),
                price DOUBLE PRECISION,
                beta DOUBLE PRECISION,
                vol_avg BIGINT,
                mkt_cap BIGINT,
                last_div DOUBLE PRECISION,
                range_str VARCHAR(50),
                changes DOUBLE PRECISION,
                is_etf BOOLEAN,
                is_actively_trading BOOLEAN
            )
        """)

        # ETFs peers table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS etfs_peers (
                symbol VARCHAR(15) NOT NULL,
                peer_symbol VARCHAR(15) NOT NULL,
                PRIMARY KEY (symbol, peer_symbol)
            )
        """)

        conn.commit()
        logger.info("Tables created successfully")

def load_csv_to_table(conn, csv_file: str, table_name: str) -> bool:
    """Load a CSV file to PostgreSQL table using COPY FROM."""
    csv_path = os.path.join(get_fmp_csv_directory(), csv_file)

    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return False

    logger.info(f"Loading {csv_file} to {table_name}...")

    try:
        with conn.cursor() as cur:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Skip header line
                next(f)

                # Use COPY FROM for fast loading
                cur.copy_from(
                    f,
                    table_name,
                    sep=',',
                    null='',
                    columns=None  # Use all columns in order
                )

            conn.commit()

            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
            logger.info(f"Successfully loaded {row_count} rows into {table_name}")

        return True

    except Exception as e:
        logger.error(f"Error loading {csv_file} to {table_name}: {e}")
        conn.rollback()
        return False

def clear_existing_data(conn):
    """Clear existing FMP data before loading."""
    logger.info("Clearing existing FMP data...")

    tables = list(FMP_CSV_TABLES.values())

    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                logger.info(f"Cleared table: {table}")
            except Exception as e:
                logger.warning(f"Could not clear table {table}: {e}")

        conn.commit()

def load_all_fmp_csvs() -> bool:
    """Load all FMP CSV files to PostgreSQL."""
    logger.info("Starting FMP CSV loading process...")

    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Create tables
        create_tables(conn)

        # Clear existing data
        clear_existing_data(conn)

        # Load each CSV file
        success_count = 0
        for csv_file, table_name in FMP_CSV_TABLES.items():
            if load_csv_to_table(conn, csv_file, table_name):
                success_count += 1

        conn.close()

        logger.info(f"FMP CSV loading completed: {success_count}/{len(FMP_CSV_TABLES)} files loaded successfully")
        return success_count == len(FMP_CSV_TABLES)

    except Exception as e:
        logger.error(f"Error in FMP CSV loading process: {e}")
        return False

def get_loading_status() -> Dict[str, int]:
    """Get row counts for all FMP tables."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        status = {}
        with conn.cursor() as cur:
            for table_name in FMP_CSV_TABLES.values():
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cur.fetchone()[0]
                    status[table_name] = count
                except Exception as e:
                    status[table_name] = f"Error: {e}"

        conn.close()
        return status

    except Exception as e:
        logger.error(f"Error getting loading status: {e}")
        return {}

if __name__ == "__main__":
    success = load_all_fmp_csvs()

    if success:
        print("\n=== FMP CSV Loading Complete ===")
        status = get_loading_status()
        for table, count in status.items():
            print(f"{table}: {count} rows")
    else:
        print("FMP CSV loading failed - check logs for details")
        exit(1)