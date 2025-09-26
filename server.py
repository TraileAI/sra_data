"""
Production server configuration for SRA Data Processing API with autoscaling support.

Combines FastAPI web service with background scheduling for optimal resource usage.
Worker runs in-process as imported module.
"""

import os
import sys
import logging
import threading
import uvicorn
import argparse

# Add current directory to Python path to ensure worker module can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from packages.sra_data.api.skeleton import app
import worker
import schedule
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def background_scheduler(run_initial_seeding=True, force_reset=False):
    """Run background scheduler in separate thread with direct worker module imports."""
    logger.info("Starting background scheduler with in-process worker...")

    if run_initial_seeding:
        if force_reset:
            logger.info("Force reset requested - clearing seeding status...")
            try:
                worker.force_fresh_seeding()
            except Exception as e:
                logger.error(f"Error during force reset: {e}")

        # Run initial seeding on startup (directly from worker module)
        logger.info("Running initial seeding on startup...")
        try:
            worker.initial_seeding()
            logger.info("Initial seeding completed - starting scheduled tasks...")
        except Exception as e:
            logger.error(f"Error during initial seeding: {e}")

    # Schedule recurring tasks using worker module functions
    schedule.every().day.at("08:00").do(worker.daily_quotes)
    schedule.every().sunday.at("08:00").do(worker.weekly_fundamentals)
    schedule.every().sunday.at("10:00").do(worker.weekly_scoring)

    logger.info("Background scheduler initialized with daily/weekly tasks")

    # Main scheduler loop
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
        except Exception as e:
            logger.error(f"Error in background scheduler: {e}")
            time.sleep(60)  # Continue despite errors

def get_server_config():
    """Get server configuration from environment variables."""
    return {
        "host": os.getenv("HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "10000")),
        "workers": int(os.getenv("WORKERS", "1")),
        "log_level": os.getenv("LOG_LEVEL", "info"),
        "access_log": os.getenv("ACCESS_LOG", "true").lower() == "true",
        "timeout_keep_alive": int(os.getenv("TIMEOUT_KEEP_ALIVE", "30"))
    }

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SRA Data Processing Server with Worker')
    parser.add_argument('command', nargs='?', default='server',
                       choices=['server', 'seeding', 'fmp', 'fundata', 'reset'],
                       help='Command to run (default: server)')
    parser.add_argument('--force', action='store_true',
                       help='Force fresh seeding (resets status)')
    parser.add_argument('--no-scheduler', action='store_true',
                       help='Run without background scheduler')
    return parser.parse_args()

def run_worker_only(command, force=False):
    """Run worker commands without starting the server."""
    logger.info(f"Running worker command: {command}")

    try:
        if command == 'reset':
            worker.force_fresh_seeding()
            logger.info("Reset completed")
        elif command == 'seeding':
            if force:
                worker.force_fresh_seeding()
            worker.initial_seeding()
            logger.info("Seeding completed")
        elif command == 'fmp':
            if force:
                worker.force_fresh_seeding()
            worker.fmp_seeding()
            logger.info("FMP seeding completed")
        elif command == 'fundata':
            if force:
                worker.force_fresh_seeding()
            worker.fundata_seeding()
            logger.info("FUNDATA seeding completed")
    except Exception as e:
        logger.error(f"Error running {command}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_arguments()

    # If not running as server, execute worker command and exit
    if args.command != 'server':
        run_worker_only(args.command, args.force)
        sys.exit(0)

    # Server mode with optional worker
    config = get_server_config()
    logger.info(f"Starting SRA Data Processing API server with in-process worker: {config}")

    if not args.no_scheduler:
        # Start background scheduler in separate thread (daemon=True ensures it stops with main process)
        logger.info("Starting background worker thread...")
        scheduler_thread = threading.Thread(
            target=background_scheduler,
            args=(True, args.force),  # run_initial_seeding=True, force_reset=args.force
            daemon=True,
            name="WorkerScheduler"
        )
        scheduler_thread.start()
        logger.info("Background worker thread started successfully")
    else:
        logger.info("Background scheduler disabled")

    # Start FastAPI server (main thread - this keeps the process alive)
    logger.info("Starting FastAPI server for autoscaling triggers...")
    uvicorn.run(
        "packages.sra_data.api.skeleton:app",
        host=config["host"],
        port=config["port"],
        log_level=config["log_level"],
        access_log=config["access_log"],
        timeout_keep_alive=config["timeout_keep_alive"]
    )