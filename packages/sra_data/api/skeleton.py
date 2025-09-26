"""
Minimal FastAPI skeleton for deployment stability.

This module provides a minimal FastAPI application with essential health check
endpoints to prevent Render.com service suspension. It is NOT a full client API,
but rather a deployment stability layer.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# Application metadata
APP_NAME = "SRA Data Processing Service"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Minimal FastAPI skeleton for deployment stability"

# Global startup time for uptime calculation
APP_START_TIME = time.time()

def create_fastapi_app() -> FastAPI:
    """
    Create and configure the minimal FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=APP_NAME,
        version=APP_VERSION,
        description=APP_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Add CORS middleware for deployment compatibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Include routes
    _add_routes(app)

    return app

def _add_routes(app: FastAPI) -> None:
    """Add routes to the FastAPI application."""

    @app.get("/")
    async def root() -> Dict[str, Any]:
        """
        Root endpoint with basic service information.

        Returns:
            Basic service metadata and available endpoints
        """
        return {
            "service": APP_NAME,
            "version": APP_VERSION,
            "description": "Data processing service for FMP API and fundata CSV ingestion",
            "status": "running",
            "deployment": "render.com",
            "endpoints": [
                "/",
                "/health",
                "/status",
                "/seeding-status"
            ],
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc"
            }
        }

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """
        Simple health check endpoint.

        Returns:
            Basic health status for load balancer/monitoring
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": APP_NAME,
            "version": APP_VERSION
        }

    @app.get("/status")
    async def detailed_status() -> Dict[str, Any]:
        """
        Detailed service status endpoint.

        Returns:
            Comprehensive status including system metrics and service health
        """
        current_time = time.time()
        uptime_seconds = current_time - APP_START_TIME

        # Basic system status
        status_data = {
            "service": APP_NAME,
            "version": APP_VERSION,
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(uptime_seconds),
            "uptime_human": _format_uptime(uptime_seconds)
        }

        # Try to get database status (graceful fallback if not available)
        try:
            database_status = await _get_database_status()
            status_data["database"] = database_status
        except Exception as e:
            logger.warning(f"Unable to get database status: {e}")
            status_data["database"] = {
                "status": "unknown",
                "error": "Database connectivity check unavailable"
            }

        # Try to get data services status
        try:
            services_status = await _get_data_services_status()
            status_data["data_services"] = services_status
        except Exception as e:
            logger.warning(f"Unable to get data services status: {e}")
            status_data["data_services"] = {
                "status": "unknown",
                "error": "Data services status check unavailable"
            }

        # Determine overall status
        overall_status = _determine_overall_status(status_data)
        status_data["status"] = overall_status

        return status_data

    @app.get("/seeding-status")
    async def seeding_status() -> Dict[str, Any]:
        """Get current seeding status and lock information."""
        try:
            # Import worker module
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            import psycopg2

            # Get database connection
            DB_CONFIG = {
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            }

            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            # Get seeding status
            cur.execute("""
                SELECT data_source, is_completed, completed_at, last_updated, notes,
                       is_locked, locked_by, locked_at
                FROM seeding_status
                ORDER BY data_source
            """)

            results = cur.fetchall()
            cur.close()
            conn.close()

            status_info = {}
            for row in results:
                data_source, is_completed, completed_at, last_updated, notes, is_locked, locked_by, locked_at = row
                status_info[data_source] = {
                    "completed": is_completed,
                    "completed_at": completed_at.isoformat() if completed_at else None,
                    "last_updated": last_updated.isoformat() if last_updated else None,
                    "notes": notes,
                    "locked": is_locked or False,
                    "locked_by": locked_by,
                    "locked_at": locked_at.isoformat() if locked_at else None
                }

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seeding_status": status_info,
                "environment": {
                    "force_fresh_seeding": os.getenv('FORCE_FRESH_SEEDING', 'false')
                }
            }

        except Exception as e:
            return {
                "error": f"Failed to get seeding status: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

async def _get_database_status() -> Dict[str, Any]:
    """
    Get database connection status.

    Returns:
        Database status information
    """
    try:
        # Import here to avoid dependency issues in minimal deployment
        from packages.sra_data.repositories.database_infrastructure import create_database_manager

        db_manager = create_database_manager()
        is_connected = await db_manager.check_connection()

        if is_connected:
            connection_info = await db_manager.get_connection_info()
            return {
                "status": "connected",
                "connected": True,
                "pool_info": connection_info
            }
        else:
            return {
                "status": "disconnected",
                "connected": False,
                "error": "Database connection check failed"
            }

    except ImportError:
        return {
            "status": "unavailable",
            "connected": False,
            "error": "Database infrastructure not available"
        }
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "error": str(e)
        }

async def _get_data_services_status() -> Dict[str, Any]:
    """
    Get data processing services status.

    Returns:
        Data services status information
    """
    services_status = {}

    try:
        # FMP Integration Service Status
        try:
            from packages.sra_data.services.data_processing import create_data_processing_service

            fmp_service = create_data_processing_service()
            services_status["fmp_integration"] = {
                "status": "available",
                "last_check": datetime.now(timezone.utc).isoformat(),
                "service_type": "equity_data_processing"
            }
        except ImportError:
            services_status["fmp_integration"] = {
                "status": "unavailable",
                "error": "FMP service not available"
            }

        # Fundata Processing Service Status
        try:
            from packages.sra_data.services.csv_processing import create_csv_processing_service

            fundata_service = create_csv_processing_service()
            services_status["fundata_processing"] = {
                "status": "available",
                "last_check": datetime.now(timezone.utc).isoformat(),
                "service_type": "csv_data_processing"
            }
        except ImportError:
            services_status["fundata_processing"] = {
                "status": "unavailable",
                "error": "Fundata service not available"
            }

    except Exception as e:
        logger.warning(f"Error checking data services: {e}")
        services_status["error"] = str(e)

    return services_status

def _determine_overall_status(status_data: Dict[str, Any]) -> str:
    """
    Determine overall service status based on components.

    Args:
        status_data: Status data dictionary

    Returns:
        Overall status: "healthy", "degraded", or "unhealthy"
    """
    # Check database status
    db_status = status_data.get("database", {}).get("status", "unknown")

    # Check data services
    services = status_data.get("data_services", {})
    available_services = sum(
        1 for service_data in services.values()
        if isinstance(service_data, dict) and service_data.get("status") == "available"
    )

    # Determine status
    if db_status == "connected" and available_services >= 1:
        return "healthy"
    elif available_services >= 1:
        return "degraded"  # Some services available
    else:
        return "unhealthy"

def _format_uptime(uptime_seconds: float) -> str:
    """
    Format uptime in human-readable format.

    Args:
        uptime_seconds: Uptime in seconds

    Returns:
        Formatted uptime string
    """
    uptime = int(uptime_seconds)

    days = uptime // 86400
    hours = (uptime % 86400) // 3600
    minutes = (uptime % 3600) // 60
    seconds = uptime % 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

# Create the application instance
app = create_fastapi_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "packages.sra_data.api.skeleton:app",
        host="0.0.0.0",
        port=10000,  # Render.com default port
        reload=False,
        log_level="info"
    )