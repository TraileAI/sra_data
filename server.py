"""
Production server configuration for SRA Data Processing API.

This script provides the production server entry point for deployment on Render.com.
"""

import os
import logging
import uvicorn
from packages.sra_data.api.skeleton import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

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
    logger.info(f"Starting SRA Data Processing API server with config: {config}")

    uvicorn.run(
        "packages.sra_data.api.skeleton:app",
        host=config["host"],
        port=config["port"],
        log_level=config["log_level"],
        access_log=config["access_log"],
        timeout_keep_alive=config["timeout_keep_alive"]
        # Note: workers > 1 not supported with app instance
    )