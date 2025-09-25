"""Test fixtures for FastAPI skeleton testing."""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def mock_database_manager():
    """Mock database manager for API testing."""
    mock_db = AsyncMock()
    mock_db.check_connection.return_value = True
    mock_db.get_connection_status.return_value = {
        "connected": True,
        "pool_size": 10,
        "active_connections": 3,
        "last_check": datetime.now(timezone.utc)
    }
    return mock_db

@pytest.fixture
def mock_data_processing_service():
    """Mock data processing service for API testing."""
    mock_service = AsyncMock()
    mock_service.get_status.return_value = {
        "service": "data_processing",
        "status": "healthy",
        "last_run": datetime.now(timezone.utc),
        "success_rate": 98.5,
        "processed_symbols": 1250
    }
    return mock_service

@pytest.fixture
def mock_csv_processing_service():
    """Mock CSV processing service for API testing."""
    mock_service = AsyncMock()
    mock_service.get_status.return_value = {
        "service": "csv_processing",
        "status": "healthy",
        "last_run": datetime.now(timezone.utc),
        "processed_files": 12,
        "processed_records": 45678
    }
    return mock_service

@pytest.fixture
def healthy_service_status():
    """Sample healthy service status response."""
    return {
        "service": "SRA Data Processing",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": 86400,
        "database": {
            "connected": True,
            "pool_size": 10,
            "active_connections": 3
        },
        "data_services": {
            "fmp_integration": {
                "status": "healthy",
                "last_run": datetime.now(timezone.utc).isoformat(),
                "success_rate": 98.5
            },
            "fundata_processing": {
                "status": "healthy",
                "last_run": datetime.now(timezone.utc).isoformat(),
                "processed_files": 12
            }
        }
    }

@pytest.fixture
def degraded_service_status():
    """Sample degraded service status response."""
    return {
        "service": "SRA Data Processing",
        "version": "1.0.0",
        "status": "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": 86400,
        "database": {
            "connected": True,
            "pool_size": 10,
            "active_connections": 8
        },
        "data_services": {
            "fmp_integration": {
                "status": "healthy",
                "last_run": datetime.now(timezone.utc).isoformat(),
                "success_rate": 98.5
            },
            "fundata_processing": {
                "status": "error",
                "last_run": datetime.now(timezone.utc).isoformat(),
                "error": "Failed to process CSV file: connection timeout"
            }
        }
    }

@pytest.fixture
def basic_service_info():
    """Basic service information for root endpoint."""
    return {
        "service": "SRA Data Processing",
        "version": "1.0.0",
        "description": "Data processing service for FMP API and fundata CSV ingestion",
        "status": "running",
        "deployment": "render.com",
        "endpoints": [
            "/",
            "/health",
            "/status"
        ]
    }

@pytest.fixture
def api_test_client():
    """Mock API test client configuration."""
    return {
        "base_url": "http://testserver",
        "timeout": 30,
        "headers": {
            "User-Agent": "SRA-Data-Processing-Test/1.0",
            "Accept": "application/json"
        }
    }

@pytest.fixture
def cors_headers():
    """Expected CORS headers for deployment."""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400"
    }

@pytest.fixture
def performance_thresholds():
    """Performance thresholds for API testing."""
    return {
        "health_endpoint_max_response_time": 1.0,  # seconds
        "status_endpoint_max_response_time": 2.0,  # seconds
        "root_endpoint_max_response_time": 0.5,    # seconds
        "max_memory_usage_mb": 512,
        "max_cpu_usage_percent": 80
    }

@pytest.fixture
def mock_system_metrics():
    """Mock system metrics for monitoring."""
    return {
        "memory": {
            "used_mb": 256,
            "available_mb": 1024,
            "usage_percent": 25.0
        },
        "cpu": {
            "usage_percent": 15.5,
            "load_average": [0.5, 0.8, 1.2]
        },
        "disk": {
            "used_gb": 2.5,
            "available_gb": 20.0,
            "usage_percent": 12.5
        },
        "uptime_seconds": 86400
    }

@pytest.fixture
def deployment_config():
    """Deployment configuration for testing."""
    return {
        "host": "0.0.0.0",
        "port": 10000,  # Render.com default port
        "workers": 1,
        "log_level": "info",
        "access_log": True,
        "timeout_keep_alive": 30
    }