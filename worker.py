import schedule
import time
import subprocess
import os
import sys
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD')
}

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def is_db_seeded():
    """Check if database has been seeded by verifying key tables have data."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM equity_profile")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return count > 0
    except Exception:
        return False

def run_script(script_path):
    """Run a Python script using subprocess."""
    try:
        result = subprocess.run(['python', script_path], capture_output=True, text=True, cwd=os.path.dirname(script_path))
        if result.returncode == 0:
            print(f"Successfully ran {script_path}")
        else:
            print(f"Error running {script_path}: {result.stderr}")
    except Exception as e:
        print(f"Failed to run {script_path}: {e}")

def initial_seeding():
    """Run initial database seeding using CSV loading (fast, no API calls)."""
    if is_db_seeded():
        print("Database already seeded. Skipping initial seeding.")
        return

    print("Running initial database seeding from CSV files...")

    # Check if CSV files are available (Git LFS working)
    csv_files_available = check_csv_files_available()

    if not csv_files_available:
        print("⚠️ CSV files not available (likely Git LFS bandwidth limit exceeded)")
        print("🔄 Falling back to API-based seeding...")
        initial_seeding_fallback()
        return

    try:
        # Import the CSV loader
        from load_csv_data import initial_csv_seeding

        # Load all CSV data (FMP + fundata)
        success = initial_csv_seeding()

        if success:
            print("✅ Initial CSV seeding completed successfully!")

            # Run scoring models after data is loaded
            print("Running scoring models...")
            scoring_scripts = [
                'scoring_models/equity/1.equity_batch_scoring.py',
                'scoring_models/equity/ETFS_history_to_db.py'
            ]
            for script in scoring_scripts:
                run_script(script)

            print("✅ Initial seeding process completed!")
        else:
            print("❌ CSV seeding failed - falling back to API-based seeding")
            initial_seeding_fallback()

    except Exception as e:
        print(f"❌ Error during initial seeding: {e}")

        # Fallback to old method if CSV loading fails
        print("Falling back to API-based seeding...")
        initial_seeding_fallback()

def check_csv_files_available():
    """Check if key CSV files are available (not just LFS pointers)."""
    import os

    key_files = [
        'fmp_data/equity_profile.csv',
        'fmp_data/equity_income.csv'
    ]

    for file_path in key_files:
        if not os.path.exists(file_path):
            return False

        # Check if file is just an LFS pointer (small size)
        file_size = os.path.getsize(file_path)
        if file_size < 1000:  # LFS pointers are typically < 200 bytes
            return False

    return True

def initial_seeding_fallback():
    """Fallback to original API-based seeding if CSV loading fails."""
    print("Running fallback API-based seeding...")
    scripts = [
        'FMP/market_and_sector_quotes.py',
        'FMP/equity/1.equity_profile.py',
        'FMP/equity/2.income.py',
        'FMP/equity/3.balance.py',
        'FMP/equity/4.cashflow.py',
        'FMP/equity/5.financial_ratio.py',
        'FMP/equity/6.key_metrics.py',
        'FMP/equity/7.financial_scores.py',
        'FMP/equity/8.equity_quotes.py',
        'FMP/equity/9.balance_growth.py',
        'FMP/equity/10.cashflow_growth.py',
        'FMP/equity/11.financial_growth.py',
        'FMP/equity/12.income_growth.py',
        'FMP/equity/13.equity_peers.py',
        'FMP/etfs/1.etfs_profile.py',
        'FMP/etfs/2.etfs_data.py',
        'FMP/etfs/3.etfs_quotes.py',
        'FMP/etfs/4.etfs_peers.py',
        'FMP/treasury/treasury.py',
        'scoring_models/equity/1.equity_batch_scoring.py',
        'scoring_models/equity/ETFS_history_to_db.py'
    ]
    for script in scripts:
        run_script(script)
    print("Fallback seeding completed.")

def daily_quotes():
    """Run daily quote updates sequentially."""
    print("Running daily quotes update...")
    scripts = [
        'FMP/market_and_sector_quotes.py',
        'FMP/equity/8.equity_quotes.py',
        'FMP/etfs/3.etfs_quotes.py'
    ]
    for script in scripts:
        run_script(script)

def weekly_fundamentals():
    """Run weekly fundamentals updates sequentially."""
    print("Running weekly fundamentals update...")
    scripts = [
        'FMP/equity/1.equity_profile.py',
        'FMP/equity/2.income.py',
        'FMP/equity/3.balance.py',
        'FMP/equity/4.cashflow.py',
        'FMP/equity/5.financial_ratio.py',
        'FMP/equity/6.key_metrics.py',
        'FMP/equity/7.financial_scores.py',
        'FMP/equity/9.balance_growth.py',
        'FMP/equity/10.cashflow_growth.py',
        'FMP/equity/11.financial_growth.py',
        'FMP/equity/12.income_growth.py',
        'FMP/equity/13.equity_peers.py',
        'FMP/etfs/1.etfs_profile.py',
        'FMP/etfs/2.etfs_data.py',
        'FMP/etfs/4.etfs_peers.py',
        'FMP/treasury/treasury.py'
    ]
    for script in scripts:
        run_script(script)

def weekly_scoring():
    """Run weekly scoring updates sequentially."""
    print("Running weekly scoring update...")
    scripts = [
        'scoring_models/equity/1.equity_batch_scoring.py',
        'scoring_models/equity/ETFS_history_to_db.py'
    ]
    for script in scripts:
        run_script(script)

def fmp_seeding():
    """Load only FMP data from CSV files."""
    print("Loading FMP data from CSV files...")
    try:
        from load_csv_data import load_fmp_csvs
        success = load_fmp_csvs()
        if success:
            print("✅ FMP CSV loading completed!")
        else:
            print("❌ FMP CSV loading failed")
    except Exception as e:
        print(f"❌ Error loading FMP CSVs: {e}")

def fundata_seeding():
    """Load only fundata from CSV files."""
    print("Loading fundata from CSV files...")
    try:
        from load_csv_data import load_fundata_csvs
        success = load_fundata_csvs()
        if success:
            print("✅ Fundata CSV loading completed!")
        else:
            print("❌ Fundata CSV loading failed")
    except Exception as e:
        print(f"❌ Error loading fundata CSVs: {e}")

def force_fresh_seeding():
    """Clear seeding status to force fresh seeding."""
    print("Clearing seeding status to force fresh seeding...")
    try:
        conn = connect_db()
        cur = conn.cursor()

        # Clear some key tables to reset seeding status
        tables_to_clear = ['equity_profile', 'fund_general']
        for table in tables_to_clear:
            try:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                print(f"Cleared table: {table}")
            except Exception as e:
                print(f"Could not clear {table}: {e}")

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Seeding status cleared - next run will do fresh seeding")
    except Exception as e:
        print(f"❌ Error clearing seeding status: {e}")

if __name__ == "__main__":
    # Run initial seeding if needed
    initial_seeding()

    # Schedule tasks at staggered times to avoid overlaps
    schedule.every().day.at("00:00").do(daily_quotes)  # Daily at midnight
    schedule.every().sunday.at("01:00").do(weekly_fundamentals)  # Sunday 1 AM
    schedule.every().sunday.at("03:00").do(weekly_scoring)  # Sunday 3 AM (after fundamentals)

    print("Background worker started. Scheduled tasks: daily quotes, weekly fundamentals/scoring.")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
