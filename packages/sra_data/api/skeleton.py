"""
FastAPI skeleton for SRA Data Processing service.
Provides health checks and basic endpoints for Render autoscaling.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import psycopg2
from datetime import datetime

app = FastAPI(
    title="SRA Data Processing API",
    description="Financial data processing service with CSV-based loading",
    version="1.0.0"
)

def get_db_connection():
    """Get database connection for health checks."""
    try:
        return psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT', '5432'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
    except Exception:
        return None

@app.get("/")
async def root():
    """Root endpoint for basic health check."""
    return {"message": "SRA Data Processing API", "status": "running", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health")
async def health_check():
    """Health check endpoint for Render monitoring."""

    # Check database connectivity
    db_status = "disconnected"
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.close()
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    status = {
        "status": "healthy" if db_status == "connected" else "unhealthy",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat(),
        "service": "sra-data-processing"
    }

    status_code = 200 if db_status == "connected" else 503
    return JSONResponse(content=status, status_code=status_code)

@app.get("/status")
async def service_status():
    """Service status endpoint with data loading information."""

    # Check data loading status
    data_status = {}
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                # Check key tables for data presence
                tables_to_check = [
                    'equity_profile', 'equity_quotes', 'etfs_profile', 'etfs_quotes'
                ]

                for table in tables_to_check:
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        data_status[table] = count
                    except Exception:
                        data_status[table] = 0

            conn.close()
    except Exception as e:
        data_status = {"error": str(e)}

    return {
        "service": "sra-data-processing",
        "status": "running",
        "data_status": data_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint for monitoring."""
    return {
        "service": "sra-data-processing",
        "uptime": "running",
        "timestamp": datetime.utcnow().isoformat()
    }