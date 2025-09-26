import schedule
import time
import subprocess
import os
import gc
import psutil
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


def run_module_function(module_path, function_name="main"):
    """Import and run a function from a module directly."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting: {module_path}.{function_name}()")
    print(f"{'='*60}")

    try:
        # Convert file path to module path (FMP/equity/1.equity_profile.py -> FMP.equity.1.equity_profile)
        module_name = module_path.replace('/', '.').replace('.py', '')

        # Import the module
        module = __import__(module_name, fromlist=[function_name])

        # Get the function or run the main block
        if hasattr(module, function_name):
            func = getattr(module, function_name)
            func()
        else:
            # Execute the module's main block by importing it
            exec(open(f"{module_path}").read())

        print(f"✅ Successfully completed: {module_path}")

    except Exception as e:
        print(f"💥 Exception running {module_path}: {e}")
        import traceback
        traceback.print_exc()

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
            run_module_function(script)
            # Memory cleanup and system monitoring
            gc.collect()
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"📊 System Status - Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%")

            # Dynamic cooling based on system load
            if memory_percent > 80 or cpu_percent > 80:
                print("🔥 High system load detected - extended cooling (60 seconds)...")
                time.sleep(60)
            else:
                print("⏳ Standard CPU cooldown (30 seconds)...")
                time.sleep(30)
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
        run_module_function(script)
        # Add delay between quote scripts to prevent CPU spikes
        print("⏳ Brief CPU cooldown (10 seconds)...")
        time.sleep(10)

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
        run_module_function(script)
        # Add delay between fundamentals scripts to prevent CPU overload
        print("⏳ CPU cooldown (20 seconds)...")
        time.sleep(20)

def weekly_scoring():
    """Run weekly scoring updates sequentially."""
    print("Running weekly scoring update...")
    scripts = [
        'scoring_models/equity/1.equity_batch_scoring.py',
        'scoring_models/equity/ETFS_history_to_db.py'
    ]
    for script in scripts:
        run_module_function(script)
        # Add delay between scoring scripts to prevent CPU overload
        print("⏳ CPU cooldown (15 seconds)...")
        time.sleep(15)

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
