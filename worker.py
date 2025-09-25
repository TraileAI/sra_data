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

def ensure_seeding_status_table():
    """Create seeding_status table if it doesn't exist."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS seeding_status (
                id SERIAL PRIMARY KEY,
                data_source VARCHAR(50) UNIQUE NOT NULL,
                is_completed BOOLEAN NOT NULL DEFAULT FALSE,
                completed_at TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            );
        """)

        # Initialize default records if they don't exist
        cur.execute("""
            INSERT INTO seeding_status (data_source, is_completed)
            VALUES ('FMP', FALSE), ('FUNDATA', FALSE)
            ON CONFLICT (data_source) DO NOTHING;
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Seeding status table initialized")
    except Exception as e:
        print(f"⚠️ Failed to create seeding status table: {e}")

def is_fmp_seeded():
    """Check if FMP data has been seeded."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT is_completed FROM seeding_status WHERE data_source = 'FMP'")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result and result[0] if result else False
    except Exception as e:
        print(f"⚠️ Failed to check FMP seeding status: {e}")
        return False

def is_fundata_seeded():
    """Check if FUNDATA has been seeded."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT is_completed FROM seeding_status WHERE data_source = 'FUNDATA'")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result and result[0] if result else False
    except Exception as e:
        print(f"⚠️ Failed to check FUNDATA seeding status: {e}")
        return False

def mark_seeding_completed(data_source, notes=None):
    """Mark a data source as seeded."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            UPDATE seeding_status
            SET is_completed = TRUE,
                completed_at = CURRENT_TIMESTAMP,
                last_updated = CURRENT_TIMESTAMP,
                notes = %s
            WHERE data_source = %s
        """, (notes, data_source))
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Marked {data_source} seeding as completed")
    except Exception as e:
        print(f"❌ Failed to mark {data_source} as completed: {e}")


def run_script(script_path):
    """Run a Python script using subprocess with live output."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting: {script_path}")
    print(f"{'='*60}")

    try:
        # Use Popen for live output streaming
        process = subprocess.Popen(
            ['python', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())

        # Wait for process to complete
        return_code = process.wait()

        if return_code == 0:
            print(f"✅ Successfully completed: {script_path}")
        else:
            print(f"❌ Failed: {script_path} (exit code: {return_code})")

    except Exception as e:
        print(f"💥 Exception running {script_path}: {e}")

    print(f"{'='*60}\n")

def fmp_seeding():
    """Run FMP data seeding if not already done."""
    print("\n🔍 Checking FMP seeding status...")
    if is_fmp_seeded():
        print("✅ FMP data already seeded. Skipping FMP seeding.")
        return

    print("🚀 Running FMP data seeding...")
    fmp_scripts = [
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

    failed_scripts = []
    for script in fmp_scripts:
        print(f"\n📊 FMP Progress: {fmp_scripts.index(script) + 1}/{len(fmp_scripts)} scripts")
        try:
            run_script(script)
        except Exception as e:
            print(f"❌ Failed to run {script}: {e}")
            failed_scripts.append(script)

    if not failed_scripts:
        mark_seeding_completed('FMP', f'Completed {len(fmp_scripts)} scripts successfully')
        print("✅ FMP seeding completed successfully!")
    else:
        print(f"⚠️ FMP seeding completed with {len(failed_scripts)} failures: {failed_scripts}")

def fundata_seeding():
    """Run FUNDATA seeding if not already done."""
    print("\n🔍 Checking FUNDATA seeding status...")
    if is_fundata_seeded():
        print("✅ FUNDATA already seeded. Skipping FUNDATA seeding.")
        return

    print("🚀 Running FUNDATA seeding...")
    # TODO: Add fundata scripts here when available
    # For now, just mark as completed for testing
    print("📝 FUNDATA seeding scripts not yet implemented")
    # mark_seeding_completed('FUNDATA', 'Scripts not yet implemented')

def initial_seeding():
    """Run initial database seeding for both FMP and FUNDATA."""
    print("🎯 Starting comprehensive database seeding...")

    # Ensure seeding status table exists
    ensure_seeding_status_table()

    # Run FMP seeding
    fmp_seeding()

    # Run FUNDATA seeding
    fundata_seeding()

    print("\n🎉 Initial seeding process completed!")

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
