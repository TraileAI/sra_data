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
    'equity_cash_flow.csv': 'equity_cashflow',
    'equity_peers.csv': 'equity_peers',
    'equity_ratios.csv': 'equity_financial_ratio',
    'equity_key_metrics.csv': 'equity_key_metrics',
    'equity_balance_growth.csv': 'equity_balance_growth',
    'equity_cashflow_growth.csv': 'equity_cashflow_growth',
    'equity_financial_growth.csv': 'equity_financial_growth',
    'equity_income_growth.csv': 'equity_income_growth',
    'etfs_profile.csv': 'etfs_profile',
    'etfs_peers.csv': 'etfs_peers',
    'etfs_data.csv': 'etfs_data',
}

def get_fmp_csv_directory():
    """Get the FMP CSV data directory path."""
    # Get the project root directory (parent of this script's directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'fmp_data')

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

        # ETFs data table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS etfs_data (
                symbol VARCHAR(15) PRIMARY KEY,
                nav DOUBLE PRECISION,
                nav_currency VARCHAR(10),
                expense_ratio DOUBLE PRECISION,
                category VARCHAR(100),
                last_annual_dividend DOUBLE PRECISION,
                three_year_avg_dividend_yield DOUBLE PRECISION,
                dividend_rate DOUBLE PRECISION,
                dividend_yield DOUBLE PRECISION,
                five_year_avg_dividend_yield DOUBLE PRECISION,
                trailing_pe DOUBLE PRECISION,
                trailing_eps DOUBLE PRECISION,
                last_split_factor VARCHAR(20),
                last_split_date DATE,
                last_capital_gain DOUBLE PRECISION,
                annual_holdings_turnover DOUBLE PRECISION,
                total_net_assets BIGINT,
                avg_volume BIGINT,
                market_cap BIGINT,
                holdings_count INT,
                price_book_ratio DOUBLE PRECISION,
                price_sales_ratio DOUBLE PRECISION,
                price_earnings_ratio DOUBLE PRECISION,
                price_cash_flow_ratio DOUBLE PRECISION,
                price_earnings_to_growth_ratio DOUBLE PRECISION
            )
        """)

        # ETFs quotes table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS etfs_quotes (
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

        # Additional growth tables
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_key_metrics (
                symbol VARCHAR(15),
                date DATE,
                period VARCHAR(3),
                revenue_per_share DOUBLE PRECISION,
                net_income_per_share DOUBLE PRECISION,
                operating_cash_flow_per_share DOUBLE PRECISION,
                free_cash_flow_per_share DOUBLE PRECISION,
                cash_per_share DOUBLE PRECISION,
                book_value_per_share DOUBLE PRECISION,
                tangible_book_value_per_share DOUBLE PRECISION,
                shareholders_equity_per_share DOUBLE PRECISION,
                interest_debt_per_share DOUBLE PRECISION,
                market_cap BIGINT,
                enterprise_value BIGINT,
                pe_ratio DOUBLE PRECISION,
                price_to_sales_ratio DOUBLE PRECISION,
                pocfratio DOUBLE PRECISION,
                pfcfRatio DOUBLE PRECISION,
                pb_ratio DOUBLE PRECISION,
                ptb_ratio DOUBLE PRECISION,
                ev_to_sales DOUBLE PRECISION,
                enterprise_value_over_ebitda DOUBLE PRECISION,
                ev_to_operating_cash_flow DOUBLE PRECISION,
                ev_to_free_cash_flow DOUBLE PRECISION,
                earnings_yield DOUBLE PRECISION,
                free_cash_flow_yield DOUBLE PRECISION,
                debt_to_equity DOUBLE PRECISION,
                debt_to_assets DOUBLE PRECISION,
                net_debt_to_ebitda DOUBLE PRECISION,
                current_ratio DOUBLE PRECISION,
                interest_coverage DOUBLE PRECISION,
                income_quality DOUBLE PRECISION,
                dividend_yield DOUBLE PRECISION,
                payout_ratio DOUBLE PRECISION,
                sales_general_and_administrative_to_revenue DOUBLE PRECISION,
                research_and_development_to_revenue DOUBLE PRECISION,
                intangibles_to_total_assets DOUBLE PRECISION,
                capex_to_operating_cash_flow DOUBLE PRECISION,
                capex_to_revenue DOUBLE PRECISION,
                capex_to_depreciation DOUBLE PRECISION,
                stock_based_compensation_to_revenue DOUBLE PRECISION,
                graham_number DOUBLE PRECISION,
                roic DOUBLE PRECISION,
                return_on_tangible_assets DOUBLE PRECISION,
                graham_net_net DOUBLE PRECISION,
                working_capital BIGINT,
                tangible_asset_value BIGINT,
                net_current_asset_value BIGINT,
                invested_capital BIGINT,
                average_receivables BIGINT,
                average_payables BIGINT,
                average_inventory BIGINT,
                days_sales_outstanding DOUBLE PRECISION,
                days_payables_outstanding DOUBLE PRECISION,
                days_of_inventory_on_hand DOUBLE PRECISION,
                receivables_turnover DOUBLE PRECISION,
                payables_turnover DOUBLE PRECISION,
                inventory_turnover DOUBLE PRECISION,
                roe DOUBLE PRECISION,
                capex_per_share DOUBLE PRECISION,
                PRIMARY KEY (symbol, date)
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

    # Include all FMP tables plus the additional ones
    tables = list(FMP_CSV_TABLES.values()) + [
        'etfs_quotes', 'equity_quotes', 'equity_balance_growth', 'equity_cashflow_growth',
        'equity_financial_growth', 'equity_income_growth'
    ]

    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                logger.info(f"Cleared table: {table}")
            except Exception as e:
                logger.warning(f"Could not clear table {table}: {e}")

        conn.commit()

def load_etf_quotes_directory(conn) -> bool:
    """Load all ETF quote files from etfs_quotes directory."""
    logger.info("Loading ETF quotes from directory...")

    etf_quotes_dir = os.path.join(get_fmp_csv_directory(), 'etfs_quotes')

    if not os.path.exists(etf_quotes_dir):
        logger.warning("ETF quotes directory not found")
        return True  # Not an error if directory doesn't exist

    success_count = 0
    total_files = 0

    for filename in os.listdir(etf_quotes_dir):
        if filename.endswith('.csv'):
            total_files += 1
            file_path = os.path.join(etf_quotes_dir, filename)

            logger.info(f"Loading ETF quotes from {filename}...")

            try:
                with conn.cursor() as cur:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Skip header line
                        next(f)

                        # Use COPY FROM for fast loading
                        cur.copy_from(
                            f,
                            'etfs_quotes',
                            sep=',',
                            null='',
                            columns=None
                        )

                success_count += 1
                logger.info(f"Successfully loaded {filename}")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

    if total_files > 0:
        logger.info(f"ETF quotes loading: {success_count}/{total_files} files loaded")

    return success_count == total_files

def load_equity_quotes_directory(conn) -> bool:
    """Load all equity quote files from equity_quotes directory."""
    logger.info("Loading equity quotes from directory...")

    equity_quotes_dir = os.path.join(get_fmp_csv_directory(), 'equity_quotes')

    if not os.path.exists(equity_quotes_dir):
        logger.warning("Equity quotes directory not found")
        return True  # Not an error if directory doesn't exist

    success_count = 0
    total_files = 0

    for filename in os.listdir(equity_quotes_dir):
        if filename.endswith('.csv'):
            total_files += 1
            file_path = os.path.join(equity_quotes_dir, filename)

            logger.info(f"Loading equity quotes from {filename}...")

            try:
                with conn.cursor() as cur:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # Skip header line
                        next(f)

                        # Use COPY FROM for fast loading
                        cur.copy_from(
                            f,
                            'equity_quotes',
                            sep=',',
                            null='',
                            columns=None
                        )

                success_count += 1
                logger.info(f"Successfully loaded {filename}")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

    if total_files > 0:
        logger.info(f"Equity quotes loading: {success_count}/{total_files} files loaded")

    return success_count == total_files

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

        # Load ETF quotes directory
        etf_quotes_success = load_etf_quotes_directory(conn)

        # Load equity quotes directory
        equity_quotes_success = load_equity_quotes_directory(conn)

        conn.close()

        total_expected = len(FMP_CSV_TABLES)
        logger.info(f"FMP CSV loading completed: {success_count}/{total_expected} files loaded successfully")

        if etf_quotes_success:
            logger.info("ETF quotes loading completed successfully")

        if equity_quotes_success:
            logger.info("Equity quotes loading completed successfully")

        return success_count == total_expected and etf_quotes_success and equity_quotes_success

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
            # Check main FMP tables
            for table_name in FMP_CSV_TABLES.values():
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cur.fetchone()[0]
                    status[table_name] = count
                except Exception as e:
                    status[table_name] = f"Error: {e}"

            # Check directory-based tables
            for table_name in ['etfs_quotes', 'equity_quotes']:
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