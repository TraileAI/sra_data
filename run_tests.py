#!/usr/bin/env python3
"""
Simple test runner for the testing framework without pytest dependency.
"""

import asyncio
import logging
import sys
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_api_health_check():
    """Simple API health check test."""
    try:
        from fastapi.testclient import TestClient
        from packages.sra_data.api.skeleton import app

        client = TestClient(app)

        # Test root endpoint
        logger.info("Testing root endpoint...")
        root_response = client.get("/")
        assert root_response.status_code == 200, f"Root endpoint failed: {root_response.status_code}"

        root_data = root_response.json()
        assert root_data["service"] == "SRA Data Processing Service"
        assert root_data["status"] == "running"

        # Test health endpoint
        logger.info("Testing health endpoint...")
        health_response = client.get("/health")
        assert health_response.status_code == 200, f"Health endpoint failed: {health_response.status_code}"

        health_data = health_response.json()
        assert health_data["status"] == "healthy"

        # Test status endpoint
        logger.info("Testing status endpoint...")
        status_response = client.get("/status")
        assert status_response.status_code == 200, f"Status endpoint failed: {status_response.status_code}"

        status_data = status_response.json()
        assert status_data["status"] in ["healthy", "degraded", "unhealthy"]

        logger.info("API health check tests passed!")
        return True

    except Exception as e:
        logger.error(f"API health check test failed: {e}")
        return False


async def test_database_connection():
    """Test database connection."""
    try:
        from packages.sra_data.repositories.database_infrastructure import create_database_manager

        logger.info("Testing database connection...")
        db_manager = create_database_manager()
        await db_manager.initialize()

        # Test connection
        connected = await db_manager.check_connection()
        assert connected, "Database connection failed"

        # Get connection info
        connection_info = await db_manager.get_connection_info()
        assert isinstance(connection_info, dict), "Connection info not returned"

        await db_manager.close()

        logger.info("Database connection test passed!")
        return True

    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def test_repository_operations():
    """Test basic repository operations."""
    try:
        from packages.sra_data.repositories.database_infrastructure import create_database_manager
        from packages.sra_data.repositories.equity_repository import EquityRepository
        from packages.sra_data.models import EquityProfile

        logger.info("Testing repository operations...")

        # Initialize database and repository
        db_manager = create_database_manager()
        await db_manager.initialize()

        equity_repo = EquityRepository(db_manager.get_session_factory())

        # Test create operation
        test_profile = EquityProfile(
            symbol='TEST001',
            name='Test Company',
            market_cap=1000000,
            sector='Technology',
            industry='Software',
            country='US',
            exchange='NASDAQ',
            price=100.0,
            pe_ratio=20.0,
            beta=1.0
        )

        created = await equity_repo.create(test_profile)
        assert created.id is not None, "Created profile should have an ID"
        assert created.symbol == 'TEST001', "Symbol should match"

        # Test read operation
        retrieved = await equity_repo.get_by_id(created.id)
        assert retrieved is not None, "Should retrieve created profile"
        assert retrieved.symbol == 'TEST001', "Retrieved symbol should match"

        # Test update operation
        retrieved.price = 110.0
        updated = await equity_repo.update(retrieved)
        assert updated.price == 110.0, "Price should be updated"

        # Test delete operation
        await equity_repo.delete(created.id)
        deleted_check = await equity_repo.get_by_id(created.id)
        assert deleted_check is None, "Profile should be deleted"

        await db_manager.close()

        logger.info("Repository operations test passed!")
        return True

    except Exception as e:
        logger.error(f"Repository operations test failed: {e}")
        return False


async def test_integration_flow():
    """Test basic integration flow."""
    try:
        from fastapi.testclient import TestClient
        from packages.sra_data.api.skeleton import app
        from packages.sra_data.repositories.database_infrastructure import create_database_manager

        logger.info("Testing integration flow...")

        # Test API while database is available
        client = TestClient(app)
        db_manager = create_database_manager()
        await db_manager.initialize()

        # API should be responsive with database available
        status_response = client.get("/status")
        assert status_response.status_code == 200

        status_data = status_response.json()

        # Check that database status is reported
        assert "database" in status_data

        await db_manager.close()

        logger.info("Integration flow test passed!")
        return True

    except Exception as e:
        logger.error(f"Integration flow test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    logger.info("=== Starting comprehensive test suite ===")
    start_time = time.time()

    tests = [
        ("API Health Check", test_api_health_check),
        ("Database Connection", test_database_connection),
        ("Repository Operations", test_repository_operations),
        ("Integration Flow", test_integration_flow)
    ]

    passed_tests = 0
    total_tests = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        test_start = time.time()

        try:
            result = await test_func()
            test_end = time.time()
            test_duration = test_end - test_start

            if result:
                logger.info(f"✅ {test_name} PASSED ({test_duration:.2f}s)")
                passed_tests += 1
            else:
                logger.error(f"❌ {test_name} FAILED ({test_duration:.2f}s)")

        except Exception as e:
            test_end = time.time()
            test_duration = test_end - test_start
            logger.error(f"❌ {test_name} ERROR ({test_duration:.2f}s): {e}")

    end_time = time.time()
    total_duration = end_time - start_time

    logger.info(f"\n=== Test Suite Complete ===")
    logger.info(f"Tests passed: {passed_tests}/{total_tests}")
    logger.info(f"Success rate: {(passed_tests/total_tests*100):.1f}%")
    logger.info(f"Total duration: {total_duration:.2f}s")

    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! System is ready.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed.")
        return False


if __name__ == "__main__":
    # Run the test suite
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test suite interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        sys.exit(1)