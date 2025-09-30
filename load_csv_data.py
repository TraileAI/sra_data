"""
Unified CSV loader for both FMP and fundata.
Handles complete initial seeding from CSV files stored in the repository.
"""
import os
import psycopg2
import psycopg2.extras
import logging
import sys
import time
from typing import Dict, List, Tuple

# Add FMP module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FMP'))

# Import configuration management
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
logger.info("Loading database configuration for CSV loader...")
DB_CONFIG = config.db_config
DB_HOST = DB_CONFIG['host']
DB_PORT = DB_CONFIG['port']
DB_NAME = DB_CONFIG['database']
DB_USER = DB_CONFIG['user']
DB_PASSWORD = DB_CONFIG['password']

logger.info(f"Database config loaded: {DB_HOST}:{DB_PORT}/{DB_NAME} as {DB_USER}")

def download_required_csv_files() -> bool:
    """Download required CSV files from B2 if they don't exist locally."""
    try:
        from download_csv_files import download_all_csv_files
        logger.info("Checking for required CSV files and downloading if needed...")
        return download_all_csv_files()
    except Exception as e:
        logger.warning(f"Could not download CSV files: {e}")
        logger.warning("Continuing with existing files...")
        return False

def download_selective_csv_files(under_seeded_tables: List[str]) -> bool:
    """Download only CSV files needed for specific under-seeded tables."""
    try:
        from download_csv_files import download_csv_files_for_tables
        logger.info(f"Downloading CSV files for {len(under_seeded_tables)} under-seeded tables...")
        return download_csv_files_for_tables(under_seeded_tables)
    except Exception as e:
        logger.warning(f"Could not download selective CSV files: {e}")
        logger.warning("Continuing with existing files...")
        return False

def load_fmp_csvs(selective_tables: List[str] = None) -> bool:
    """Load FMP CSV files to PostgreSQL.

    Args:
        selective_tables: If provided, only process these specific tables instead of all FMP tables
    """
    if selective_tables:
        logger.info(f"=== Starting Selective FMP CSV Loading for {len(selective_tables)} tables ===")
        logger.info(f"Selective tables: {selective_tables}")
    else:
        logger.info("=== Starting FMP CSV Loading ===")

    try:
        # Check if FMP tables are already adequately seeded
        counts = get_all_table_counts()

        # Use selective_tables if provided, otherwise use all FMP tables
        if selective_tables:
            fmp_tables = selective_tables
        else:
            fmp_tables = ['equity_profile', 'equity_income', 'equity_balance', 'equity_cashflow',
                          'equity_earnings', 'equity_peers', 'equity_financial_ratio',
                          'equity_key_metrics', 'equity_financial_scores', 'etfs_profile',
                          'etfs_peers', 'etfs_data', 'equity_quotes', 'etfs_quotes']

        # Expected minimums for FMP tables
        fmp_minimums = {
            'equity_profile': 4000,
            'equity_income': 270000,
            'equity_balance': 260000,
            'equity_cashflow': 260000,
            'equity_earnings': 250000,
            'equity_peers': 25000,
            'equity_financial_ratio': 275000,
            'equity_key_metrics': 275000,
            'equity_financial_scores': 4000,
            'etfs_profile': 1500,
            'etfs_peers': 5000,
            'etfs_data': 700000,
            'equity_quotes': 10000000,
            'etfs_quotes': 500000,
        }

        fmp_adequately_seeded = True
        critical_failures = []
        for table in fmp_tables:
            current_count = counts.get(table, 0)
            expected_min = fmp_minimums.get(table, 0)
            if isinstance(current_count, str) or current_count < expected_min:
                fmp_adequately_seeded = False
                critical_failures.append(f"{table}: {current_count} < {expected_min}")

        if fmp_adequately_seeded:
            logger.info("FMP tables are already adequately seeded - skipping FMP CSV loading")
            return True
        else:
            logger.warning(f"FMP tables need reseeding due to failures: {critical_failures}")
            logger.info("Proceeding with FMP CSV loading...")

        from load_from_csv import load_all_fmp_csvs, get_loading_status

        success = load_all_fmp_csvs(selective_tables=fmp_tables if selective_tables else None)

        if success:
            logger.info("FMP CSV loading completed successfully")
            status = get_loading_status()
            for table, count in status.items():
                logger.info(f"  {table}: {count} rows")
        else:
            logger.error("FMP CSV loading failed - check logs for critical table failures")

        return success

    except Exception as e:
        logger.error(f"Error loading FMP CSVs: {e}")
        return False

def load_fundata_csvs(drop_all_tables: bool = True) -> bool:
    """Load fundata CSV files to PostgreSQL."""
    logger.info("=== Starting Fundata CSV Loading ===")

    # Check if fundata tables already have adequate data
    try:
        counts = get_all_table_counts()
        # Check ALL fundata tables, not just the original 3
        fundata_minimums = {
            'fund_general': 1000,           # Expect at least 1000 funds
            'fund_daily_nav': 10000,        # Expect at least 10K NAV records
            'fund_quotes': 50000,           # Expect at least 50K quote records
            'benchmark_general': 100,
            'instrument_identifier': 5000,
            'fund_performance_summary': 10000,
            'fund_allocation': 10000,
            'fund_expenses': 10000,
            'fund_yearly_performance': 10000,
            # New tables that need to be loaded
            'benchmark_yearly_performance': 1000,
            'fund_advanced_performance': 5000,
            'fund_assets': 10000,
            'fund_associate_benchmark': 5000,
            'fund_distribution': 10000,
            'fund_equity_style': 5000,
            'fund_fixed_income_style': 5000,
            'fund_loads': 5000,
            'fund_other_fees': 5000,
            'fund_risk_yearly_performance': 5000,
            'fund_top_holdings': 20000,
            'fund_trailer_schedule': 5000,
            'fund_yearly_performance_ranking': 10000,
        }

        fundata_adequately_seeded = True
        under_seeded = []
        for table, min_expected in fundata_minimums.items():
            current_count = counts.get(table, 0)
            if isinstance(current_count, str) or current_count < min_expected:
                fundata_adequately_seeded = False
                under_seeded.append(f"{table}: {current_count} < {min_expected}")

        if not fundata_adequately_seeded:
            logger.info(f"Fundata tables need seeding: {under_seeded[:3]}...")
        else:
            logger.info("All fundata tables are adequately seeded - skipping fundata CSV loading")
            return True
    except Exception as e:
        logger.warning(f"Could not check fundata seeding status: {e} - proceeding with loading")

    # Fundata CSV mappings - ALL 21 seed files
    fundata_tables = {
        # Currently mapped (8 files)
        'FundGeneralSeed.csv': 'fund_general',
        'BenchmarkGeneralSeed.csv': 'benchmark_general',
        'FundDailyNAVPSSeed.csv': 'fund_daily_nav',
        'InstrumentIdentifierSeed.csv': 'instrument_identifier',
        'FundPerformanceSummarySeed.csv': 'fund_performance_summary',
        'FundAllocationSeed.csv': 'fund_allocation',
        'FundExpensesSeed.csv': 'fund_expenses',
        'FundYearlyPerformanceSeed.csv': 'fund_yearly_performance',

        # Previously unmapped (13 files) - now adding
        'BenchmarkYearlyPerformanceSeed.csv': 'benchmark_yearly_performance',
        'FundAdvancedPerformanceSeed.csv': 'fund_advanced_performance',
        'FundAssetsSeed.csv': 'fund_assets',
        'FundAssociateBenchmarkSeed.csv': 'fund_associate_benchmark',
        'FundDistributionSeed.csv': 'fund_distribution',
        'FundEquityStyleSeed.csv': 'fund_equity_style',
        'FundFixedIncomeStyleSeed.csv': 'fund_fixed_income_style',
        'FundLoadsSeed.csv': 'fund_loads',
        'FundOtherFeeSeed.csv': 'fund_other_fees',
        'FundRiskYearlyPerformanceSeed.csv': 'fund_risk_yearly_performance',
        'FundTopHoldingSeed.csv': 'fund_top_holdings',
        'FundTrailerScheduleSeed.csv': 'fund_trailer_schedule',
        'FundYearlyPerformanceRankingByClassSeed.csv': 'fund_yearly_performance_ranking',
    }

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Drop existing tables only if requested (for full reload)
        if drop_all_tables:
            logger.info("Dropping existing fundata tables and creating proper schemas...")
            with conn.cursor() as cur:
                for table_name in fundata_tables.values():
                    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                cur.execute("DROP TABLE IF EXISTS fund_quotes CASCADE")
            conn.commit()

            # Create proper table schemas
            if not create_fundata_table_schemas(conn):
                return False
        else:
            logger.info("Selective loading - keeping existing tables, only loading files that exist...")
            # Ensure tables exist for files we're about to load
            if not create_fundata_table_schemas(conn):
                return False

        success_count = 0

        # Load data files
        fundata_data_dir = os.path.join('fundata', 'data')
        for csv_file, table_name in fundata_tables.items():
            csv_path = os.path.join(fundata_data_dir, csv_file)

            if os.path.exists(csv_path):
                if load_fundata_csv_to_table(conn, csv_path, table_name):
                    success_count += 1
            else:
                logger.warning(f"Fundata CSV not found: {csv_path}")

        # Load quotes files
        fundata_quotes_dir = os.path.join('fundata', 'quotes')
        quotes_files = [
            'FundDailyNAVPSSeed.csv',
            'Pricing2015to2025/Pricing2015to2017.csv',
            'Pricing2015to2025/Pricing2018to2019.csv',
            'Pricing2015to2025/Pricing2020to2021.csv',
            'Pricing2015to2025/Pricing2022to2023.csv',
            'Pricing2015to2025/Pricing2024to2025.csv'
        ]

        for quotes_file in quotes_files:
            csv_path = os.path.join(fundata_quotes_dir, quotes_file)
            if os.path.exists(csv_path):
                # Use generic quotes table name
                table_name = 'fund_quotes' if 'Pricing' in quotes_file else 'fund_daily_nav'
                if load_fundata_csv_to_table(conn, csv_path, table_name):
                    success_count += 1

        conn.close()

        total_files = len(fundata_tables) + len(quotes_files)
        logger.info(f"Fundata CSV loading completed: {success_count}/{total_files} files loaded")
        return success_count > 0

    except Exception as e:
        logger.error(f"Error loading fundata CSVs: {e}")
        return False

def create_fundata_table_schemas(conn) -> bool:
    """Create proper structured tables for fundata matching actual CSV structure."""
    try:
        with conn.cursor() as cur:
            # Fund General table - matches FundGeneralSeed.csv exactly
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_general (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    language VARCHAR(10),
                    legal_name TEXT,
                    family_name TEXT,
                    series_name TEXT,
                    company TEXT,
                    instrument_type VARCHAR(50),
                    inception_date VARCHAR(20),  -- Some dates might be malformed
                    objective TEXT,
                    strategy TEXT,
                    cifsc_type TEXT,
                    properties TEXT,
                    min_investment VARCHAR(50),  -- Handle empty numeric values
                    sub_investment VARCHAR(50),  -- Handle empty numeric values
                    distribution_frequency TEXT,
                    management_fee VARCHAR(20),  -- Handle empty numeric values
                    legal_status TEXT,
                    management_type TEXT,
                    share_class TEXT,
                    fixed_distribution VARCHAR(10),
                    high_net_worth VARCHAR(10),
                    currency VARCHAR(10),
                    document_type TEXT,
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Benchmark General table - matches BenchmarkGeneralSeed.csv exactly
            cur.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_general (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    language VARCHAR(10),
                    legal_name TEXT,
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Daily NAV table - matches FundDailyNAVPSSeed.csv structure
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_daily_nav (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    current_yield VARCHAR(20),
                    current_yield_percent_change VARCHAR(20),
                    date VARCHAR(20),
                    navps VARCHAR(20),
                    navps_penny_change VARCHAR(20),
                    navps_percent_change VARCHAR(20),
                    previous_date VARCHAR(20),
                    previous_navps VARCHAR(20),
                    split VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Instrument Identifier table - keep existing structure as it worked
            cur.execute("""
                CREATE TABLE IF NOT EXISTS instrument_identifier (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    identifier_type VARCHAR(50),
                    identifier_value VARCHAR(100),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Performance Summary table - match actual CSV structure exactly
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_performance_summary (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    month_end_date VARCHAR(20),
                    performance_start_date VARCHAR(20),
                    one_month_return VARCHAR(20),
                    three_month_return VARCHAR(20),
                    six_month_return VARCHAR(20),
                    ytd_return VARCHAR(20),
                    inception_return VARCHAR(20),
                    inception_return_date VARCHAR(20),
                    ytd_total_distribution VARCHAR(20),
                    fund_grade VARCHAR(10),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Allocation table - flexible to handle extra columns
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_allocation (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    language VARCHAR(10),
                    allocation_date VARCHAR(20),
                    allocation_type TEXT,
                    allocation_name TEXT,
                    allocation_value VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Expenses table - keep existing as it worked
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_expenses (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    expense_date VARCHAR(20),
                    expense_type VARCHAR(50),
                    expense_value VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Yearly Performance table - flexible to handle extra columns
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_yearly_performance (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    performance_date VARCHAR(20),
                    performance_year VARCHAR(10),
                    annual_return VARCHAR(20),
                    benchmark_return VARCHAR(20),
                    reference_date VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # ===== NEW TABLES FOR PREVIOUSLY UNMAPPED FUNDATA FILES =====

            # Benchmark Yearly Performance table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_yearly_performance (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    month_end_date VARCHAR(20),
                    year VARCHAR(10),
                    compound_return VARCHAR(20),
                    calendar_simple_return VARCHAR(20),
                    calendar_return_year VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Advanced Performance table - FIXED with all 32 columns
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_advanced_performance (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    monthend_date VARCHAR(20),
                    inception_return_date VARCHAR(20),
                    worst_monthly_return VARCHAR(20),
                    uninterrupted_drawdown VARCHAR(20),
                    max_percent_change VARCHAR(20),
                    max_start_date VARCHAR(20),
                    max_end_date VARCHAR(20),
                    max_down_length VARCHAR(20),
                    max_recovery_time VARCHAR(20),
                    one_year_max_drawdown VARCHAR(20),
                    three_year_max_drawdown VARCHAR(20),
                    five_year_max_drawdown VARCHAR(20),
                    ten_year_max_drawdown VARCHAR(20),
                    one_year_category_batting VARCHAR(20),
                    three_year_category_batting VARCHAR(20),
                    five_year_category_batting VARCHAR(20),
                    ten_year_category_batting VARCHAR(20),
                    one_year_absolute_batting VARCHAR(20),
                    three_year_absolute_batting VARCHAR(20),
                    five_year_absolute_batting VARCHAR(20),
                    ten_year_absolute_batting VARCHAR(20),
                    one_year_up_capture VARCHAR(20),
                    three_year_up_capture VARCHAR(20),
                    five_year_up_capture VARCHAR(20),
                    ten_year_up_capture VARCHAR(20),
                    one_year_down_capture VARCHAR(20),
                    three_year_down_capture VARCHAR(20),
                    five_year_down_capture VARCHAR(20),
                    ten_year_down_capture VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Assets table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_assets (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    date VARCHAR(20),
                    fund_assets_value VARCHAR(30),
                    series_assets_value VARCHAR(30),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Associate Benchmark table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_associate_benchmark (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    language VARCHAR(10),
                    benchmark_type VARCHAR(50),
                    benchmark_instrument_key VARCHAR(50),  -- Can be text like "234408"
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Distribution table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_distribution (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    date VARCHAR(20),
                    total_distribution VARCHAR(20),
                    dividend_income VARCHAR(20),
                    foreign_dividend_income VARCHAR(20),
                    capital_gains VARCHAR(20),
                    interest_income VARCHAR(20),
                    return_of_principle VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Equity Style table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_equity_style (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    month_end_date VARCHAR(20),
                    capitalization VARCHAR(20),
                    style VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Fixed Income Style table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_fixed_income_style (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    month_end_date VARCHAR(20),
                    duration_class VARCHAR(20),
                    credit_rating_class VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Loads table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_loads (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    language VARCHAR(10),
                    fundserv_code VARCHAR(20),
                    sales_status VARCHAR(20),
                    load_type VARCHAR(20),
                    max_percent VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Other Fees table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_other_fees (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    language VARCHAR(10),
                    fee_type VARCHAR(100),
                    value VARCHAR(20),
                    days VARCHAR(20),
                    paid_by VARCHAR(50),
                    fee_note TEXT,
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Risk Yearly Performance table - FIXED with all 19 columns
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_risk_yearly_performance (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    month_end_date VARCHAR(20),
                    year VARCHAR(10),
                    annualized_std VARCHAR(20),
                    annualized_alpha VARCHAR(20),
                    beta VARCHAR(20),
                    r_squared VARCHAR(20),
                    annualized_sharpe VARCHAR(20),
                    annualized_treynor VARCHAR(20),
                    annualized_sortino VARCHAR(20),
                    annualized_information_ratio VARCHAR(20),
                    annualized_jensen_alpha VARCHAR(20),
                    annualized_tracking_error VARCHAR(20),
                    tax_efficiency VARCHAR(20),
                    calmar VARCHAR(20),
                    downside_deviation VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Top Holdings table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_top_holdings (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    language VARCHAR(10),
                    date VARCHAR(20),
                    security TEXT,
                    market_percent VARCHAR(20),
                    holding_number VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Trailer Schedule table - FIXED to match actual CSV
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_trailer_schedule (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    fundserv_code VARCHAR(20),
                    start_month VARCHAR(20),
                    end_month VARCHAR(20),
                    description TEXT,
                    fee VARCHAR(20),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Yearly Performance Ranking by Class table - FIXED with all 12 columns
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_yearly_performance_ranking (
                    record_id INTEGER,
                    instrument_key INTEGER,
                    month_end_date VARCHAR(20),
                    year VARCHAR(10),
                    compound_return_quartile VARCHAR(10),
                    compound_return_fund_count VARCHAR(10),
                    compound_return_ranking VARCHAR(10),
                    calendar_return_quartile VARCHAR(10),
                    calendar_return_fund_count VARCHAR(10),
                    calendar_return_ranking VARCHAR(10),
                    record_state VARCHAR(20),
                    change_date VARCHAR(20),
                    PRIMARY KEY (record_id)
                )
            """)

            # Fund Quotes table - matches Pricing CSV structure exactly
            cur.execute("""
                CREATE TABLE IF NOT EXISTS fund_quotes (
                    instrument_key INTEGER,
                    date VARCHAR(20),
                    navps VARCHAR(20),
                    daily_total_distribution VARCHAR(20),
                    daily_dividend_income_distribution VARCHAR(20),
                    daily_foreign_dividend_income_distribution VARCHAR(20),
                    daily_capital_gains_distribution VARCHAR(20),
                    daily_interest_income_distribution VARCHAR(20),
                    daily_return_of_principle_distribution VARCHAR(20),
                    distribution_pay_date VARCHAR(20),
                    split_factor VARCHAR(20),
                    navps_percent_change VARCHAR(20),
                    penny_change_day VARCHAR(20),
                    current_yield VARCHAR(20),
                    current_yield_percent_change VARCHAR(20),
                    PRIMARY KEY (instrument_key, date)
                )
            """)

            conn.commit()
            logger.info("Fundata table schemas created successfully")
            return True

    except Exception as e:
        logger.error(f"Error creating fundata table schemas: {e}")
        conn.rollback()
        return False

def load_fundata_csv_to_table(conn, csv_path: str, table_name: str) -> bool:
    """Load a single fundata CSV file to PostgreSQL table using proper schema with CSV cleaning."""
    logger.info(f"Loading {os.path.basename(csv_path)} to {table_name}...")

    try:
        # Import clean_csv_line from FMP module
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), 'FMP'))
        from load_from_csv import clean_csv_line

        # Create a temporary cleaned file
        temp_csv_path = csv_path + '.cleaned'

        with open(csv_path, 'r', encoding='utf-8') as infile:
            with open(temp_csv_path, 'w', encoding='utf-8') as outfile:
                header = infile.readline()
                outfile.write(header)

                # Count expected columns from header
                expected_columns = len(header.strip().split(','))

                for line_num, line in enumerate(infile, start=2):
                    cleaned_line = clean_csv_line(line, expected_columns)
                    outfile.write(cleaned_line + '\n')

        with conn.cursor() as cur:
            # Use PostgreSQL COPY for efficient loading
            copy_sql = f"""
                COPY {table_name} FROM STDIN
                WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
            """

            with open(temp_csv_path, 'r', encoding='utf-8') as f:
                cur.copy_expert(copy_sql, f)

            conn.commit()

            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
            logger.info(f"Successfully loaded {row_count} total rows in {table_name}")

        # Clean up temporary file
        os.remove(temp_csv_path)
        return True

    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        conn.rollback()
        # Clean up temporary file on error
        temp_csv_path = csv_path + '.cleaned'
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        return False

def initial_csv_seeding() -> bool:
    """Perform complete initial seeding from CSV files."""
    logger.info("=== Starting Complete CSV Seeding Process ===")

    start_time = time.time()

    # Step 0: Get specific tables that need seeding
    under_seeded_tables = get_under_seeded_tables()
    if not under_seeded_tables:
        logger.info("Database is already adequately seeded - skipping download and seeding")
        return True

    # Step 1: Only download CSV files for tables that actually need seeding
    logger.info(f"Database needs seeding for {len(under_seeded_tables)} tables - downloading selective CSV files...")
    download_success = download_selective_csv_files(under_seeded_tables)
    if not download_success:
        logger.warning("Selective CSV download had issues, but continuing with existing files...")

    # Step 2: Only run FMP loading if download was successful or files exist
    fmp_success = True  # Default to success for cases where no FMP tables need loading

    # Check if any under-seeded tables require FMP loading
    fmp_tables_needed = [table for table in under_seeded_tables
                        if table.startswith(('equity_', 'etfs_'))]

    if fmp_tables_needed:
        if download_success:
            logger.info(f"Running FMP loading for {len(fmp_tables_needed)} tables: {fmp_tables_needed}")
            fmp_success = load_fmp_csvs(selective_tables=fmp_tables_needed)
        else:
            logger.warning(f"Skipping FMP loading - downloads failed for needed tables: {fmp_tables_needed}")
            fmp_success = False

    # Step 3: Only run fundata loading if needed
    fundata_success = True  # Default to success for cases where no fundata tables need loading

    # Check if any under-seeded tables require fundata loading
    fundata_tables_needed = [table for table in under_seeded_tables
                           if table.startswith(('fund_', 'benchmark_', 'instrument_'))]

    if fundata_tables_needed:
        logger.info(f"Running fundata loading for {len(fundata_tables_needed)} tables: {fundata_tables_needed}")
        # Don't drop all tables when doing selective loading
        fundata_success = load_fundata_csvs(drop_all_tables=False)
    else:
        logger.info("No fundata tables need reseeding - skipping fundata loading")

    elapsed_time = time.time() - start_time

    if fmp_success and fundata_success:
        logger.info(f"=== Complete CSV Seeding Successful in {elapsed_time:.1f} seconds ===")
        return True
    elif fmp_success:
        logger.info(f"=== FMP CSV Seeding Successful in {elapsed_time:.1f} seconds ===")
        logger.warning("Fundata loading failed, but FMP data loaded successfully")
        return True  # Consider FMP-only success as acceptable for basic functionality
    else:
        logger.error(f"=== CSV Seeding FAILED - FMP: {fmp_success}, Fundata: {fundata_success} ===")
        logger.error("Critical tables remain empty - system is not functional")
        return False

def get_all_table_counts() -> Dict[str, int]:
    """Get row counts for all loaded tables."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        counts = {}
        with conn.cursor() as cur:
            # Get all table names
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """)

            tables = [row[0] for row in cur.fetchall()]

            for table in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    counts[table] = count
                except Exception as e:
                    counts[table] = f"Error: {e}"

        conn.close()
        return counts

    except Exception as e:
        logger.error(f"Error getting table counts: {e}")
        return {}

def check_tables_need_seeding() -> bool:
    """Check if any core tables are under-seeded and need reseeding."""
    logger.info("Checking if database needs seeding...")

    # Expected minimum row counts for fully seeded tables (based on our test results)
    EXPECTED_MINIMUMS = {
        # Core FMP tables
        'equity_profile': 4000,        # Had 4133
        'equity_income': 270000,       # Had 277290
        'equity_balance': 260000,      # Had 263727
        'equity_cashflow': 260000,     # Had 263587
        'equity_earnings': 250000,     # New earnings data similar to other financial tables
        'equity_peers': 25000,         # Had 26210
        'equity_financial_ratio': 275000,  # Had 279125
        'equity_key_metrics': 275000,  # Had 279125
        'equity_financial_scores': 4000,    # One record per symbol, similar to equity_profile
        'etfs_profile': 1500,          # Had 1548
        'etfs_peers': 5000,            # Had 5934
        'etfs_data': 700000,           # Had 726188

        # Growth tables (critical for financial analysis)
        'equity_balance_growth': 200000,    # Growth data for balance sheet metrics
        'equity_cashflow_growth': 200000,   # Growth data for cashflow metrics
        'equity_financial_growth': 200000,  # General financial growth metrics
        'equity_income_growth': 200000,     # Growth data for income statement metrics

        # Quote tables (critical ones that were failing)
        'equity_quotes': 10000000,     # Had 14553046 - at least 10M
        'etfs_quotes': 500000,         # Had 623904 - at least 500K

        # Existing fundata tables
        'fund_general': 2000,          # Had 44626 - at least 2K funds
        'benchmark_general': 100,      # Benchmark data
        'fund_daily_nav': 100000,      # Had 6629248 - at least 100K NAV records
        'instrument_identifier': 5000, # Instrument identifiers
        'fund_performance_summary': 10000, # Performance summaries
        'fund_allocation': 10000,      # Asset allocation data
        'fund_expenses': 10000,        # Expense data
        'fund_yearly_performance': 10000, # Yearly performance data
        'fund_quotes': 100000,         # Quote data from pricing files

        # New fundata tables (will be populated once deployed) - lowered expectations
        'benchmark_yearly_performance': 100,     # Reduced expectation
        'fund_advanced_performance': 100,        # Reduced expectation
        'fund_assets': 100,                     # Reduced expectation
        'fund_associate_benchmark': 100,         # Reduced expectation
        'fund_distribution': 100,                # Reduced expectation
        'fund_equity_style': 100,                # Reduced expectation
        'fund_fixed_income_style': 100,          # Reduced expectation
        'fund_loads': 100,                       # Reduced expectation
        'fund_other_fees': 100,                  # Reduced expectation
        'fund_risk_yearly_performance': 100,     # Reduced expectation
        'fund_top_holdings': 100,                # Reduced expectation
        'fund_trailer_schedule': 100,            # Reduced expectation
        'fund_yearly_performance_ranking': 100,  # Reduced expectation
    }

    try:
        counts = get_all_table_counts()

        under_seeded_tables = []

        for table, min_expected in EXPECTED_MINIMUMS.items():
            current_count = counts.get(table, 0)

            # Handle error cases
            if isinstance(current_count, str):
                logger.warning(f"Error getting count for {table}: {current_count}")
                under_seeded_tables.append(table)
                continue

            if current_count < min_expected:
                logger.warning(f"Table {table} is under-seeded: {current_count} < {min_expected}")
                under_seeded_tables.append(table)
            else:
                logger.info(f"Table {table} is adequately seeded: {current_count} rows")

        if under_seeded_tables:
            logger.warning(f"Found {len(under_seeded_tables)} under-seeded tables: {under_seeded_tables}")
            return True
        else:
            logger.info("All core tables are adequately seeded")
            return False

    except Exception as e:
        logger.error(f"Error checking table seeding status: {e}")
        # If we can't check, assume we need seeding for safety
        return True

def get_under_seeded_tables() -> List[str]:
    """Get list of tables that are under-seeded and need reseeding."""
    logger.info("Getting list of under-seeded tables...")

    # Expected minimum row counts for fully seeded tables (based on our test results)
    EXPECTED_MINIMUMS = {
        # Core FMP tables
        'equity_profile': 4000,        # Had 4133
        'equity_income': 270000,       # Had 277290
        'equity_balance': 260000,      # Had 263727
        'equity_cashflow': 260000,     # Had 263587
        'equity_earnings': 250000,     # New earnings data similar to other financial tables
        'equity_peers': 25000,         # Had 26210
        'equity_financial_ratio': 275000,  # Had 279125
        'equity_key_metrics': 275000,  # Had 279125
        'equity_financial_scores': 4000,    # One record per symbol, similar to equity_profile
        'etfs_profile': 1500,          # Had 1548
        'etfs_peers': 5000,            # Had 5934
        'etfs_data': 700000,           # Had 726188

        # Growth tables (critical for financial analysis)
        'equity_balance_growth': 200000,    # Growth data for balance sheet metrics
        'equity_cashflow_growth': 200000,   # Growth data for cashflow metrics
        'equity_financial_growth': 200000,  # General financial growth metrics
        'equity_income_growth': 200000,     # Growth data for income statement metrics

        # Quote tables (critical ones that were failing)
        'equity_quotes': 10000000,     # Had 14553046 - at least 10M
        'etfs_quotes': 500000,         # Had 623904 - at least 500K

        # Existing fundata tables
        'fund_general': 2000,          # Had 44626 - at least 2K funds
        'benchmark_general': 100,      # Benchmark data
        'fund_daily_nav': 100000,      # Had 6629248 - at least 100K NAV records
        'instrument_identifier': 5000, # Instrument identifiers
        'fund_performance_summary': 10000, # Performance summaries
        'fund_allocation': 10000,      # Asset allocation data
        'fund_expenses': 10000,        # Expense data
        'fund_yearly_performance': 10000, # Yearly performance data
        'fund_quotes': 100000,         # Quote data from pricing files

        # New fundata tables (will be populated once deployed) - lowered expectations
        'benchmark_yearly_performance': 100,     # Reduced expectation
        'fund_advanced_performance': 100,        # Reduced expectation
        'fund_assets': 100,                     # Reduced expectation
        'fund_associate_benchmark': 100,         # Reduced expectation
        'fund_distribution': 100,                # Reduced expectation
        'fund_equity_style': 100,                # Reduced expectation
        'fund_fixed_income_style': 100,          # Reduced expectation
        'fund_loads': 100,                       # Reduced expectation
        'fund_other_fees': 100,                  # Reduced expectation
        'fund_risk_yearly_performance': 100,     # Reduced expectation
        'fund_top_holdings': 100,                # Reduced expectation
        'fund_trailer_schedule': 100,            # Reduced expectation
        'fund_yearly_performance_ranking': 100,  # Reduced expectation
    }

    try:
        counts = get_all_table_counts()
        under_seeded_tables = []

        for table, min_expected in EXPECTED_MINIMUMS.items():
            current_count = counts.get(table, 0)

            # Handle error cases
            if isinstance(current_count, str):
                logger.warning(f"Error getting count for {table}: {current_count}")
                under_seeded_tables.append(table)
                continue

            if current_count < min_expected:
                logger.warning(f"Table {table} is under-seeded: {current_count} < {min_expected}")
                under_seeded_tables.append(table)
            else:
                logger.info(f"Table {table} is adequately seeded: {current_count} rows")

        return under_seeded_tables

    except Exception as e:
        logger.error(f"Error getting under-seeded tables: {e}")
        # Return all tables as under-seeded on error to be safe
        return list(EXPECTED_MINIMUMS.keys())

def auto_seed_if_needed():
    """Check if seeding is needed and run it automatically if so."""
    logger.info("=== Auto-Seeding Check ===")

    needs_seeding = check_tables_need_seeding()

    if needs_seeding:
        logger.info("🚀 Tables are under-seeded. Starting automatic seeding...")
        success = initial_csv_seeding()
        return success
    else:
        logger.info("✅ Database is already adequately seeded. Skipping seeding process.")
        return True

if __name__ == "__main__":
    import time
    import sys

    # Check if we should force seeding
    force_seed = "--force" in sys.argv

    if force_seed:
        logger.info("🔧 Force seeding requested via --force flag")
        success = initial_csv_seeding()
    else:
        # Smart auto-seeding based on table counts
        success = auto_seed_if_needed()

    if success:
        print("\n=== Database Loading Summary ===")
        counts = get_all_table_counts()

        # Separate FMP and fundata tables
        fmp_tables = [t for t in counts.keys() if t.startswith(('equity_', 'etfs_'))]
        fundata_tables = [t for t in counts.keys() if t.startswith(('fund_', 'benchmark_', 'instrument_'))]
        other_tables = [t for t in counts.keys() if t not in fmp_tables and t not in fundata_tables]

        if fmp_tables:
            print("\nFMP Tables:")
            for table in sorted(fmp_tables):
                print(f"  {table}: {counts[table]} rows")

        if fundata_tables:
            print("\nFundata Tables:")
            for table in sorted(fundata_tables):
                print(f"  {table}: {counts[table]} rows")

        if other_tables:
            print("\nOther Tables:")
            for table in sorted(other_tables):
                print(f"  {table}: {counts[table]} rows")

        print("\n✅ Database is ready!")
    else:
        print("❌ Database seeding failed - check logs for details")
        exit(1)