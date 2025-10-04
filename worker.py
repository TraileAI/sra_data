import schedule
import time
import subprocess
import os
import sys
from datetime import datetime
import psycopg2

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our configuration management
from config import config

print("🔧 Loading database configuration...")
DB_CONFIG = config.db_config
print(f"✅ Database configuration loaded successfully")

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def inspect_database():
    """Inspect database state for debugging."""
    try:
        conn = connect_db()
        cur = conn.cursor()

        print("📊 Database inspection:")

        # Check what tables exist
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"   Tables found: {len(tables)}")
        for table in tables[:10]:  # Show first 10 tables
            print(f"     - {table}")
        if len(tables) > 10:
            print(f"     ... and {len(tables) - 10} more")

        # Check specific key tables
        key_tables = ['equity_profile', 'equity_income', 'fund_general']
        for table in key_tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"   {table}: {count} rows")
            except Exception as e:
                print(f"   {table}: not accessible ({e})")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Database inspection failed: {e}")

def is_db_seeded():
    """Check if database has been seeded using comprehensive table analysis."""
    try:
        # Import the proper seeding detection logic from load_csv_data
        from load_csv_data import check_tables_need_seeding

        print("🔍 Checking database seeding status using comprehensive analysis...")

        # check_tables_need_seeding returns True if seeding is needed, False if already seeded
        needs_seeding = check_tables_need_seeding()
        is_seeded = not needs_seeding

        if is_seeded:
            print("✅ Database comprehensively seeded - skipping initial seeding")
        else:
            print("🌱 Database needs seeding based on table analysis - will run initial seeding")

        return is_seeded
    except Exception as e:
        print(f"⚠️ Could not check seeding status: {e}")
        print("🔍 Running database inspection...")
        inspect_database()
        print("🌱 Assuming database needs seeding")
        return False

def run_script(script_path):
    """Run a Python script using subprocess (works locally and on Render.com)."""
    print(f"🚀 Starting script: {script_path}")
    try:
        # Use python3 for Render.com compatibility, run from project root
        result = subprocess.run(['python3', script_path], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print(f"✅ Successfully completed {script_path}")
            if result.stdout:
                # Show last few lines of output for progress visibility
                output_lines = result.stdout.strip().split('\n')
                if len(output_lines) > 3:
                    print(f"   Last output: ...{output_lines[-1]}")
                else:
                    print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ Error running {script_path} (exit code: {result.returncode})")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"   Output: {result.stdout.strip()}")
    except subprocess.TimeoutExpired:
        print(f"⏰ Script {script_path} timed out after 10 minutes")
    except Exception as e:
        print(f"💥 Failed to run {script_path}: {e}")

def initial_seeding():
    """Run initial database seeding using CSV loading (fast, no API calls)."""
    print("🚀 === INITIAL_SEEDING FUNCTION STARTED ===")
    print(f"🔧 Current working directory: {os.getcwd()}")
    print(f"🔧 Python path: {sys.path[:3]}")  # Show first 3 entries

    # Check for force seeding from environment
    force_seeding = os.getenv('FORCE_SEEDING', 'false').lower() == 'true'
    print(f"🔄 Force seeding setting: {force_seeding}")

    seeding_needed = False

    if force_seeding:
        print("🔄 FORCE_SEEDING enabled - will run seeding regardless of current state")
        seeding_needed = True
    elif not is_db_seeded():
        print("Database needs seeding - will run CSV seeding")
        seeding_needed = True
    else:
        print("✅ Database already seeded - skipping CSV loading")

    # Run CSV seeding if needed
    if seeding_needed:
        print("Running initial database seeding from CSV files...")

        try:
            # Import the CSV loader
            from load_csv_data import initial_csv_seeding

            # Load all CSV data (FMP + fundata)
            success = initial_csv_seeding()

            if not success:
                print("❌ CSV seeding failed - falling back to API-based seeding")
                initial_seeding_fallback()
                return  # Fallback handles its own updates

        except Exception as e:
            print(f"❌ Error during initial seeding: {e}")
            # Fallback to old method if CSV loading fails
            print("Falling back to API-based seeding...")
            initial_seeding_fallback()
            return  # Fallback handles its own updates

    # ALWAYS run updates on first worker.py startup (regardless of seeding status)
    # This ensures data is current even if seeding was done previously
    print("")
    print("🔄 === RUNNING INITIAL DATA UPDATES ===")
    print("Updating all data to current date...")

    # Run daily updates to populate calculated tables and get latest prices
    print("")
    print("📊 Running daily quotes update...")
    daily_quotes()

    # Run weekly updates to get all fundamentals
    print("")
    print("📈 Running weekly fundamentals update...")
    weekly_fundamentals()

    # Run scoring models after all data is loaded
    print("")
    print("🏆 Running weekly scoring update...")
    weekly_scoring()

    print("")
    print("✅ Initial startup process completed - all data is current!")


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
        'FMP/equity/15. equity_financial_scores.py',
        'FMP/equity/16. equity_earnings.py',
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

def log_progress(phase, current, total, script_name=None):
    """Log progress for monitoring via ps or logs."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    if script_name:
        print(f"📊 [{timestamp}] PROGRESS: {phase} - {current}/{total} - Running: {script_name}")
    else:
        print(f"📊 [{timestamp}] PROGRESS: {phase} - {current}/{total} completed")

def daily_quotes():
    """Run daily quote updates sequentially."""
    print("🔥 === DAILY QUOTES UPDATE PHASE ===")
    scripts = [
        # Base quote updates - FMP
        'FMP/market_and_sector_quotes.py',
        'FMP/bulk_eod_quotes.py',  # Bulk EOD quotes for equity and ETFs (replaces 8.equity_quotes.py and 3.etfs_quotes.py)
        # Base quote updates - Fundata
        'fundata/update_fund_quotes.py',
        # Market/sector data collection
        'FMP/market_sector_quotes.py',
        # Calculated metrics (run after base data)
        'FMP/index_performance.py',
        'FMP/top_10.py',
        'FMP/market_sector_pages_performance.py',
        'FMP/sector_top_holding.py'
    ]
    
    total_scripts = len(scripts)
    for i, script in enumerate(scripts, 1):
        log_progress("DAILY_QUOTES", i, total_scripts, script)
        run_script(script)
        log_progress("DAILY_QUOTES", i, total_scripts)
    
    print("✅ Daily quotes update completed!")

def weekly_fundamentals():
    """Run weekly fundamentals updates sequentially."""
    print("📈 === WEEKLY FUNDAMENTALS UPDATE PHASE ===")
    scripts = [
        # Core equity data
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
        'FMP/equity/15. equity_financial_scores.py',
        'FMP/equity/16. equity_earnings.py',
        # ETF data
        'FMP/etfs/1.etfs_profile.py',
        'FMP/etfs/2.etfs_data.py',
        'FMP/etfs/4.etfs_peers.py',
        # Treasury data
        'FMP/treasury/treasury.py',
        # Calculated metrics (run after all base data is collected)
        'FMP/equity/18.equity_standard_deviation.py',
        # Advanced scores (run NEXT-TO-LAST, after all base and calculated data)
        'FMP/equity/17.equity_financial_scores.py'
    ]
    
    total_scripts = len(scripts)
    for i, script in enumerate(scripts, 1):
        log_progress("WEEKLY_FUNDAMENTALS", i, total_scripts, script)
        run_script(script)
        log_progress("WEEKLY_FUNDAMENTALS", i, total_scripts)
    
    print("✅ Weekly fundamentals update completed!")

def weekly_scoring():
    """Run weekly scoring updates sequentially."""
    print("🏆 === WEEKLY SCORING UPDATE PHASE ===")
    scripts = [
        'scoring_models/equity/1.equity_batch_scoring.py',
        'scoring_models/equity/ETFS_history_to_db.py',
        'FMP/equity/equity_scoring.py',
        'FMP/equity/equity_scores_average.py'
    ]

    total_scripts = len(scripts)
    for i, script in enumerate(scripts, 1):
        log_progress("WEEKLY_SCORING", i, total_scripts, script)
        run_script(script)
        log_progress("WEEKLY_SCORING", i, total_scripts)

    print("✅ Weekly scoring update completed!")

def monthly_scoring():
    """Run monthly scoring updates on the first day of the month."""
    from datetime import datetime

    # Only run if it's the first day of the month
    if datetime.now().day != 1:
        print(f"⏭️  Skipping monthly scoring - not first day of month (today is day {datetime.now().day})")
        return

    print("📅 === MONTHLY SCORING UPDATE PHASE ===")
    print(f"Running monthly scoring for {datetime.now().strftime('%B %Y')}")

    scripts = [
        'FMP/equity/equity_scoring.py',
        'FMP/equity/equity_scores_average.py'
    ]

    total_scripts = len(scripts)
    for i, script in enumerate(scripts, 1):
        log_progress("MONTHLY_SCORING", i, total_scripts, script)
        run_script(script)
        log_progress("MONTHLY_SCORING", i, total_scripts)

    print("✅ Monthly scoring update completed!")

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
    schedule.every().day.at("03:00").do(monthly_scoring)  # Daily 3 AM, only executes on 1st of month

    print("Background worker started. Scheduled tasks: daily quotes, weekly fundamentals/scoring, monthly equity scoring.")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
