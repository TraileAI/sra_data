import schedule
import time
import subprocess
import os
import gc
import psutil
import argparse
import sys
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
    try:
        print(f"🔗 Attempting database connection with config: {DB_CONFIG}")
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Database connection successful")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print(f"   Config used: {DB_CONFIG}")
        raise

def ensure_seeding_status_table():
    """Create seeding_status table with distributed lock support."""
    try:
        conn = connect_db()
        cur = conn.cursor()

        # Create table if it doesn't exist
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

        # Add lock columns if they don't exist (for existing tables)
        try:
            cur.execute("ALTER TABLE seeding_status ADD COLUMN IF NOT EXISTS is_locked BOOLEAN NOT NULL DEFAULT FALSE;")
            cur.execute("ALTER TABLE seeding_status ADD COLUMN IF NOT EXISTS locked_by VARCHAR(100);")
            cur.execute("ALTER TABLE seeding_status ADD COLUMN IF NOT EXISTS locked_at TIMESTAMP;")
        except Exception as alter_error:
            print(f"⚠️ Note: Could not add lock columns (may already exist): {alter_error}")

        # Initialize default records if they don't exist
        cur.execute("""
            INSERT INTO seeding_status (data_source, is_completed)
            VALUES ('FMP', FALSE), ('FUNDATA', FALSE)
            ON CONFLICT (data_source) DO NOTHING;
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Seeding status table initialized with lock support")
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

def acquire_seeding_lock(data_source, instance_id):
    """Acquire distributed lock for seeding a data source."""
    try:
        conn = connect_db()
        cur = conn.cursor()

        # Try to acquire lock atomically
        cur.execute("""
            UPDATE seeding_status
            SET is_locked = TRUE,
                locked_by = %s,
                locked_at = CURRENT_TIMESTAMP,
                last_updated = CURRENT_TIMESTAMP
            WHERE data_source = %s
            AND (is_locked = FALSE OR locked_at < NOW() - INTERVAL '30 minutes')
            AND is_completed = FALSE
        """, (instance_id, data_source))

        acquired = cur.rowcount > 0
        conn.commit()
        cur.close()
        conn.close()

        if acquired:
            print(f"🔒 Acquired seeding lock for {data_source} (instance: {instance_id})")
        else:
            print(f"⏳ Could not acquire lock for {data_source} - another instance may be processing")

        return acquired
    except Exception as e:
        print(f"❌ Failed to acquire lock for {data_source}: {e}")
        return False

def release_seeding_lock(data_source, instance_id):
    """Release distributed lock for seeding."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            UPDATE seeding_status
            SET is_locked = FALSE,
                locked_by = NULL,
                locked_at = NULL,
                last_updated = CURRENT_TIMESTAMP
            WHERE data_source = %s AND locked_by = %s
        """, (data_source, instance_id))
        conn.commit()
        cur.close()
        conn.close()
        print(f"🔓 Released seeding lock for {data_source}")
    except Exception as e:
        print(f"❌ Failed to release lock for {data_source}: {e}")

def mark_seeding_completed(data_source, notes=None):
    """Mark a data source as seeded and release lock."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            UPDATE seeding_status
            SET is_completed = TRUE,
                completed_at = CURRENT_TIMESTAMP,
                last_updated = CURRENT_TIMESTAMP,
                notes = %s,
                is_locked = FALSE,
                locked_by = NULL,
                locked_at = NULL
            WHERE data_source = %s
        """, (notes, data_source))
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Marked {data_source} seeding as completed")
    except Exception as e:
        print(f"❌ Failed to mark {data_source} as completed: {e}")

def reset_seeding_status(data_source, reason="Manual reset"):
    """Reset seeding status to allow re-seeding."""
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            UPDATE seeding_status
            SET is_completed = FALSE,
                completed_at = NULL,
                last_updated = CURRENT_TIMESTAMP,
                notes = %s,
                is_locked = FALSE,
                locked_by = NULL,
                locked_at = NULL
            WHERE data_source = %s
        """, (reason, data_source))
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Reset seeding status for {data_source}: {reason}")
    except Exception as e:
        print(f"❌ Failed to reset seeding status for {data_source}: {e}")

def force_fresh_seeding():
    """Force fresh seeding by resetting all completion statuses."""
    print("🔄 Forcing fresh seeding by resetting completion statuses...")
    reset_seeding_status('FMP', 'Force fresh start due to incomplete seeding')
    reset_seeding_status('FUNDATA', 'Force fresh start due to incomplete seeding')


def run_module_function(module_path, function_name="main"):
    """Import and run a function from a module directly."""
    print(f"\n{'='*60}")
    print(f"🚀 Starting: {module_path}")
    print(f"{'='*60}")

    try:
        # Execute the script directly using exec with proper globals/locals
        import os
        import sys

        # Get absolute path
        abs_path = os.path.abspath(module_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Script not found: {abs_path}")

        # Set up execution environment
        script_dir = os.path.dirname(abs_path)
        script_name = os.path.basename(abs_path)

        # Add script directory to sys.path temporarily
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        try:
            # Execute the script
            with open(abs_path, 'r') as f:
                script_content = f.read()

            # Create execution environment
            exec_globals = {
                '__file__': abs_path,
                '__name__': '__main__',
                '__package__': None
            }

            # Execute the script
            exec(script_content, exec_globals)

        finally:
            # Remove script directory from sys.path
            if script_dir in sys.path:
                sys.path.remove(script_dir)

        print(f"✅ Successfully completed: {module_path}")

    except Exception as e:
        print(f"💥 Exception running {module_path}: {e}")
        import traceback
        traceback.print_exc()

    print(f"{'='*60}\n")

def check_fmp_api_quota():
    """Check FMP API quota before starting seeding."""
    import requests
    try:
        # Test API call with small request
        test_url = f"https://financialmodelingprep.com/api/v3/stock-screener?limit=1&apikey={os.getenv('FMP_API_KEY', 'Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT')}"
        response = requests.get(test_url, timeout=10)

        if response.status_code == 429:
            print("❌ FMP API quota exceeded. Cannot proceed with seeding.")
            return False
        elif response.status_code != 200:
            print(f"⚠️ FMP API returned status {response.status_code}: {response.text}")
            return False

        print("✅ FMP API quota check passed")
        return True
    except Exception as e:
        print(f"⚠️ Failed to check FMP API quota: {e}")
        return False

def fmp_seeding():
    """Run FMP data seeding if not already done."""
    print("\n🔍 Checking FMP seeding status...", flush=True)
    try:
        fmp_seeded = is_fmp_seeded()
        print(f"   - FMP seeding status: {fmp_seeded}", flush=True)
        if fmp_seeded:
            print("✅ FMP data already seeded. Skipping FMP seeding.", flush=True)
            return

        # Generate unique instance ID
        import socket
        import uuid
        instance_id = f"{socket.gethostname()}-{str(uuid.uuid4())[:8]}"
        print(f"   - Generated instance ID: {instance_id}", flush=True)

        # Try to acquire distributed lock
        print("🔒 Attempting to acquire seeding lock...", flush=True)
        if not acquire_seeding_lock('FMP', instance_id):
            print("⏳ Another instance is already processing FMP seeding. Skipping.", flush=True)
            return

        # Check API quota before proceeding
        print("📡 Checking FMP API quota...", flush=True)
        if not check_fmp_api_quota():
            print("❌ API quota check failed. Releasing lock and exiting.", flush=True)
            release_seeding_lock('FMP', instance_id)
            return

        print(f"🚀 Running FMP data seeding (instance: {instance_id})...", flush=True)
    except Exception as e:
        print(f"💥 Error in FMP seeding initialization: {e}")
        import traceback
        traceback.print_exc()
        return
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
    try:
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
                if memory_percent > 85 or cpu_percent > 85:
                    print("🔥 High system load detected - brief cooling (10 seconds)...")
                    time.sleep(10)
            except Exception as e:
                print(f"❌ Failed to run {script}: {e}")
                failed_scripts.append(script)

        if not failed_scripts:
            mark_seeding_completed('FMP', f'Completed {len(fmp_scripts)} scripts successfully')
            print("✅ FMP seeding completed successfully!")
        else:
            print(f"⚠️ FMP seeding completed with {len(failed_scripts)} failures: {failed_scripts}")
            # Don't mark as completed if there were failures, just release lock
            release_seeding_lock('FMP', instance_id)

    except Exception as e:
        print(f"💥 Critical error during FMP seeding: {e}")
        release_seeding_lock('FMP', instance_id)

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
    print("🎯 Starting comprehensive database seeding...", flush=True)

    # Debug environment variables
    print("🔍 Environment variable debug:", flush=True)
    print(f"   - DB_HOST: {os.getenv('DB_HOST', 'NOT SET')}", flush=True)
    print(f"   - DB_PORT: {os.getenv('DB_PORT', 'NOT SET')}", flush=True)
    print(f"   - DB_NAME: {os.getenv('DB_NAME', 'NOT SET')}", flush=True)
    print(f"   - DB_USER: {os.getenv('DB_USER', 'NOT SET')}", flush=True)
    print(f"   - DB_PASSWORD: {'SET' if os.getenv('DB_PASSWORD') else 'NOT SET'}", flush=True)
    print(f"   - FMP_API_KEY: {'SET' if os.getenv('FMP_API_KEY') else 'NOT SET'}", flush=True)

    try:
        # Check for force fresh seeding environment variable
        if os.getenv('FORCE_FRESH_SEEDING', '').lower() == 'true':
            print("🔄 FORCE_FRESH_SEEDING enabled - forcing fresh start...", flush=True)
            force_fresh_seeding()

        # Ensure seeding status table exists
        print("📋 Ensuring seeding status table exists...", flush=True)
        ensure_seeding_status_table()

        # Check current seeding status
        print("🔍 Checking current seeding status...", flush=True)
        fmp_status = is_fmp_seeded()
        fundata_status = is_fundata_seeded()
        print(f"   - FMP already seeded: {fmp_status}", flush=True)
        print(f"   - FUNDATA already seeded: {fundata_status}", flush=True)

        # If both show as completed but user says seeding is not complete, reset statuses
        if fmp_status and fundata_status:
            print("⚠️ Seeding shows complete but may be stale. Forcing fresh start...", flush=True)
            force_fresh_seeding()
        elif fmp_status:
            print("⚠️ FMP shows complete but may be incomplete. Resetting FMP status...", flush=True)
            reset_seeding_status('FMP', 'Incomplete seeding detected - forcing restart')

        # Run FMP seeding
        print("🚀 Starting FMP seeding process...", flush=True)
        fmp_seeding()

        # Run FUNDATA seeding
        print("🚀 Starting FUNDATA seeding process...", flush=True)
        fundata_seeding()

        print("\n🎉 Initial seeding process completed!", flush=True)
    except Exception as e:
        print(f"💥 Critical error during initial seeding: {e}")
        import traceback
        traceback.print_exc()

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
        # Brief pause between scripts
        time.sleep(2)

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
        # Brief pause between scripts
        time.sleep(2)

def weekly_scoring():
    """Run weekly scoring updates sequentially."""
    print("Running weekly scoring update...")
    scripts = [
        'scoring_models/equity/1.equity_batch_scoring.py',
        'scoring_models/equity/ETFS_history_to_db.py'
    ]
    for script in scripts:
        run_module_function(script)
        # Brief pause between scripts
        time.sleep(2)

def main():
    """Main function for running initial seeding and starting scheduler."""
    print("🚀 SRA Data Worker - Starting main process...")

    # Run initial seeding if needed
    initial_seeding()

    # Schedule tasks at staggered times to avoid overlaps
    schedule.every().day.at("00:00").do(daily_quotes)  # Daily at midnight
    schedule.every().sunday.at("01:00").do(weekly_fundamentals)  # Sunday 1 AM
    schedule.every().sunday.at("03:00").do(weekly_scoring)  # Sunday 3 AM (after fundamentals)

    print("📅 Background worker started. Scheduled tasks: daily quotes, weekly fundamentals/scoring.")
    print("💡 Manual seeding commands:")
    print("   python worker.py                    # Full seeding + scheduler (default)")
    print("   python worker.py seeding            # Full seeding only")
    print("   python worker.py fmp                # FMP seeding only")
    print("   python worker.py fundata            # FUNDATA seeding only")
    print("   python worker.py reset              # Reset seeding status")
    print("   python worker.py seeding --force    # Reset status + full seeding")
    print("   python -m worker fmp                # Module-style execution")

    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def cli():
    """Command line interface for manual seeding operations."""
    parser = argparse.ArgumentParser(description='SRA Data Processing Worker')
    parser.add_argument('command', nargs='?', choices=['seeding', 'fmp', 'fundata', 'reset', 'scheduler'],
                       default='scheduler', help='Command to execute')
    parser.add_argument('--force', action='store_true', help='Force reset status before seeding')

    args = parser.parse_args()

    print(f"🚀 SRA Data Worker - Command: {args.command}")

    if args.force or args.command == 'reset':
        print("🔄 Forcing fresh seeding status...")
        force_fresh_seeding()

    if args.command == 'seeding':
        print("📊 Running full seeding process...")
        initial_seeding()
    elif args.command == 'fmp':
        print("📈 Running FMP seeding only...")
        fmp_seeding()
    elif args.command == 'fundata':
        print("📋 Running FUNDATA seeding only...")
        fundata_seeding()
    elif args.command == 'reset':
        print("✅ Seeding status reset completed")
    elif args.command == 'scheduler':
        print("📅 Starting full process with scheduler...")
        main()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli()
    else:
        main()
