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

def background_scheduler():
    """Run background scheduler in separate thread with direct worker module imports."""
    logger.info("Starting background scheduler with in-process worker...")

    # Run initial seeding on startup (directly from worker module)
    logger.info("Running initial seeding on startup...")
    worker.initial_seeding()
    logger.info("Initial seeding completed - starting scheduled tasks...")

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

if __name__ == "__main__":
    config = get_server_config()
    logger.info(f"Starting SRA Data Processing API server with in-process worker: {config}")

    # Start background scheduler in separate thread (daemon=True ensures it stops with main process)
    logger.info("Starting background worker thread...")
    scheduler_thread = threading.Thread(target=background_scheduler, daemon=True, name="WorkerScheduler")
    scheduler_thread.start()
    logger.info("Background worker thread started successfully")

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