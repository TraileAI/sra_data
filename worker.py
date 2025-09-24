import schedule
import time
import subprocess
import os
from datetime import datetime
import psycopg2
from dotenv import load_dotenv

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
    """Run initial database seeding if not already done."""
    if is_db_seeded():
        print("Database already seeded. Skipping initial seeding.")
        return
    print("Running initial database seeding...")
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
    print("Initial seeding completed.")

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
