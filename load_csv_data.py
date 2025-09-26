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
from dotenv import load_dotenv

# Add FMP module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FMP'))

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def load_fmp_csvs() -> bool:
    """Load FMP CSV files to PostgreSQL."""
    logger.info("=== Starting FMP CSV Loading ===")

    try:
        from load_from_csv import load_all_fmp_csvs, get_loading_status

        success = load_all_fmp_csvs()

        if success:
            logger.info("FMP CSV loading completed successfully")
            status = get_loading_status()
            for table, count in status.items():
                logger.info(f"  {table}: {count} rows")
        else:
            logger.error("FMP CSV loading failed")

        return success

    except Exception as e:
        logger.error(f"Error loading FMP CSVs: {e}")
        return False

def load_fundata_csvs() -> bool:
    """Load fundata CSV files to PostgreSQL."""
    logger.info("=== Starting Fundata CSV Loading ===")

    # Fundata CSV mappings (add more as needed)
    fundata_tables = {
        'FundGeneralSeed.csv': 'fund_general',
        'BenchmarkGeneralSeed.csv': 'benchmark_general',
        'FundDailyNAVPSSeed.csv': 'fund_daily_nav',
        'InstrumentIdentifierSeed.csv': 'instrument_identifier',
        'FundPerformanceSummarySeed.csv': 'fund_performance_summary',
        'FundAllocationSeed.csv': 'fund_allocation',
        'FundExpensesSeed.csv': 'fund_expenses',
        'FundYearlyPerformanceSeed.csv': 'fund_yearly_performance',
    }

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        success_count = 0

        # Load data files
        fundata_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fundata', 'data')
        for csv_file, table_name in fundata_tables.items():
            csv_path = os.path.join(fundata_data_dir, csv_file)

            if os.path.exists(csv_path):
                if load_fundata_csv_to_table(conn, csv_path, table_name):
                    success_count += 1
            else:
                logger.warning(f"Fundata CSV not found: {csv_path}")

        # Load quotes files
        fundata_quotes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fundata', 'quotes')
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

def load_fundata_csv_to_table(conn, csv_path: str, table_name: str) -> bool:
    """Load a single fundata CSV file to PostgreSQL table."""
    logger.info(f"Loading {os.path.basename(csv_path)} to {table_name}...")

    try:
        with conn.cursor() as cur:
            # Create table dynamically based on CSV headers (simplified approach)
            # In production, you'd want proper table schemas
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Create table if it doesn't exist (basic version)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        data JSONB
                    )
                """)

                # Reset file pointer and skip header for actual data loading
                f.seek(0)
                header = next(f).strip().split(',')

                # For now, load as JSONB for flexibility
                # You can create proper schemas later
                row_count = 0
                for line in f:
                    values = line.strip().split(',')
                    if len(values) == len(header):
                        data = dict(zip(header, values))
                        cur.execute(
                            f"INSERT INTO {table_name} (data) VALUES (%s)",
                            (psycopg2.extras.Json(data),)
                        )
                        row_count += 1

            conn.commit()
            logger.info(f"Successfully loaded {row_count} rows into {table_name}")

        return True

    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        conn.rollback()
        return False

def initial_csv_seeding() -> bool:
    """Perform complete initial seeding from CSV files."""
    logger.info("=== Starting Complete CSV Seeding Process ===")

    start_time = time.time()

    # Step 1: Load FMP data
    fmp_success = load_fmp_csvs()

    # Step 2: Load fundata
    fundata_success = load_fundata_csvs()

    elapsed_time = time.time() - start_time

    if fmp_success and fundata_success:
        logger.info(f"=== Complete CSV Seeding Successful in {elapsed_time:.1f} seconds ===")
        return True
    else:
        logger.error(f"=== CSV Seeding Failed - FMP: {fmp_success}, Fundata: {fundata_success} ===")
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

if __name__ == "__main__":
    import time

    success = initial_csv_seeding()

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

        print("\n✅ CSV seeding completed successfully!")
    else:
        print("❌ CSV seeding failed - check logs for details")
        exit(1)