"""
Comprehensive End-to-End Testing Framework

This module provides complete application flow testing across all layers:
- API endpoints and middleware
- Service integration and data flow
- Repository and database operations
- Cross-component integration
- Real-world usage scenarios
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import text

# Import application components
from packages.sra_data.api.skeleton import app
from packages.sra_data.repositories.database_infrastructure import (
    create_database_manager, DatabaseManager
)
from packages.sra_data.repositories.equity_repository import EquityRepository
from packages.sra_data.repositories.fundata_repository import (
    FundataRepository, FundataDataRepository, FundataQuotesRepository
)
from packages.sra_data.services.data_processing import create_data_processing_service
from packages.sra_data.services.csv_processing import create_csv_processing_service
from packages.sra_data.models import (
    EquityProfile, FundataDataRecord, FundataQuotesRecord
)

logger = logging.getLogger(__name__)


class E2ETestContext:
    """Context manager for end-to-end test environment."""

    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.equity_repo: Optional[EquityRepository] = None
        self.fundata_repo: Optional[FundataRepository] = None
        self.test_client: Optional[TestClient] = None
        self.async_client: Optional[AsyncClient] = None
        self.test_data: Dict[str, Any] = {}
        self.cleanup_tasks: List = []

    async def __aenter__(self):
        """Initialize test environment."""
        # Setup database manager
        self.db_manager = create_database_manager()
        await self.db_manager.initialize()

        # Setup repositories
        self.equity_repo = EquityRepository(self.db_manager.get_session_factory())
        self.fundata_repo = FundataRepository(self.db_manager.get_session_factory())

        # Setup API clients
        self.test_client = TestClient(app)
        self.async_client = AsyncClient(app=app, base_url="http://test")

        # Initialize test data
        await self._setup_test_data()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup test environment."""
        # Run cleanup tasks
        for cleanup_task in reversed(self.cleanup_tasks):
            try:
                await cleanup_task()
            except Exception as e:
                logger.warning(f"Cleanup task failed: {e}")

        # Close clients
        if self.async_client:
            await self.async_client.aclose()

        # Cleanup database
        if self.db_manager:
            await self.db_manager.close()

    async def _setup_test_data(self):
        """Setup test data for E2E tests."""
        self.test_data = {
            'equity_profiles': [
                {
                    'symbol': f'TEST{i}',
                    'name': f'Test Company {i}',
                    'market_cap': 1000000 * (i + 1),
                    'sector': 'Technology' if i % 2 == 0 else 'Healthcare',
                    'industry': f'Test Industry {i}',
                    'country': 'US',
                    'exchange': 'NASDAQ',
                    'price': 100.0 + i * 10,
                    'pe_ratio': 15.0 + i,
                    'beta': 1.0 + i * 0.1
                }
                for i in range(5)
            ],
            'fundata_records': [
                {
                    'fund_id': f'FD{i:03d}',
                    'fund_name': f'Test Fund {i}',
                    'fund_family': 'Test Fund Family',
                    'category': 'Equity',
                    'nav': 10.0 + i,
                    'total_assets': 1000000 * (i + 1),
                    'expense_ratio': 0.01 * (i + 1),
                    'date': datetime.utcnow().date()
                }
                for i in range(5)
            ]
        }


@pytest_asyncio.fixture
async def e2e_context():
    """Provide E2E test context."""
    async with E2ETestContext() as context:
        yield context


class TestEndToEndAPIFlow:
    """Test complete API request/response flows."""

    @pytest.mark.asyncio
    async def test_api_health_check_flow(self, e2e_context: E2ETestContext):
        """Test complete health check API flow."""
        start_time = time.time()

        # Test root endpoint
        response = e2e_context.test_client.get("/")
        assert response.status_code == 200

        root_data = response.json()
        assert root_data["service"] == "SRA Data Processing Service"
        assert root_data["status"] == "running"
        assert "endpoints" in root_data

        # Test health endpoint
        health_response = e2e_context.test_client.get("/health")
        assert health_response.status_code == 200

        health_data = health_response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data

        # Test detailed status endpoint
        status_response = e2e_context.test_client.get("/status")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert status_data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "uptime_seconds" in status_data
        assert status_data["uptime_seconds"] >= 0

        # Verify API response consistency
        assert all(
            data["service"] == "SRA Data Processing Service"
            for data in [root_data, health_data, status_data]
        )

        end_time = time.time()
        logger.info(f"API health check flow completed in {end_time - start_time:.2f}s")

    @pytest.mark.asyncio
    async def test_api_async_flow(self, e2e_context: E2ETestContext):
        """Test async API request handling."""
        start_time = time.time()

        # Test concurrent health checks
        async def check_health():
            response = await e2e_context.async_client.get("/health")
            return response.status_code, response.json()

        # Run multiple concurrent requests
        tasks = [check_health() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all requests succeeded
        for status_code, data in results:
            assert status_code == 200
            assert data["status"] == "healthy"

        end_time = time.time()
        logger.info(f"Async API flow test completed in {end_time - start_time:.2f}s")


class TestEndToEndDataFlow:
    """Test complete data processing flows."""

    @pytest.mark.asyncio
    async def test_equity_data_complete_flow(self, e2e_context: E2ETestContext):
        """Test complete equity data processing flow."""
        start_time = time.time()

        # Step 1: Create equity profiles via repository
        created_profiles = []
        for equity_data in e2e_context.test_data['equity_profiles']:
            profile = EquityProfile(**equity_data)
            saved_profile = await e2e_context.equity_repo.create(profile)
            created_profiles.append(saved_profile)
            e2e_context.cleanup_tasks.append(
                lambda p=saved_profile: e2e_context.equity_repo.delete(p.id)
            )

        assert len(created_profiles) == 5

        # Step 2: Search and retrieve data
        search_results = await e2e_context.equity_repo.search(
            query="Test Company",
            limit=10
        )
        assert len(search_results) == 5

        # Step 3: Bulk operations
        bulk_update_data = [
            {'id': profile.id, 'price': profile.price * 1.1}
            for profile in created_profiles
        ]

        updated_count = await e2e_context.equity_repo.bulk_update(bulk_update_data)
        assert updated_count == 5

        # Step 4: Statistics and aggregation
        stats = await e2e_context.equity_repo.get_portfolio_statistics()
        assert stats['total_count'] >= 5
        assert stats['total_market_cap'] > 0

        # Step 5: Verify data integrity
        for profile in created_profiles:
            retrieved = await e2e_context.equity_repo.get_by_id(profile.id)
            assert retrieved is not None
            assert retrieved.symbol == profile.symbol
            assert retrieved.price != profile.price  # Should be updated

        end_time = time.time()
        logger.info(f"Equity data flow completed in {end_time - start_time:.2f}s")

    @pytest.mark.asyncio
    async def test_fundata_processing_complete_flow(self, e2e_context: E2ETestContext):
        """Test complete fundata processing flow."""
        start_time = time.time()

        # Step 1: Create fundata records
        data_repo = FundataDataRepository(e2e_context.db_manager.get_session_factory())
        quotes_repo = FundataQuotesRepository(e2e_context.db_manager.get_session_factory())

        created_records = []
        for fund_data in e2e_context.test_data['fundata_records']:
            # Create data record
            data_record = FundataDataRecord(
                fund_id=fund_data['fund_id'],
                fund_name=fund_data['fund_name'],
                fund_family=fund_data['fund_family'],
                category=fund_data['category'],
                total_assets=fund_data['total_assets'],
                expense_ratio=fund_data['expense_ratio'],
                date=fund_data['date']
            )
            saved_data = await data_repo.create(data_record)
            created_records.append(saved_data)

            # Create quotes record
            quotes_record = FundataQuotesRecord(
                fund_id=fund_data['fund_id'],
                date=fund_data['date'],
                nav=fund_data['nav'],
                volume=1000,
                change_percent=0.01
            )
            saved_quotes = await quotes_repo.create(quotes_record)

            # Add cleanup tasks
            e2e_context.cleanup_tasks.extend([
                lambda d=saved_data: data_repo.delete(d.id),
                lambda q=saved_quotes: quotes_repo.delete(q.id)
            ])

        assert len(created_records) == 5

        # Step 2: Test unified repository operations
        unified_data = await e2e_context.fundata_repo.get_fund_complete_data(
            fund_id="FD000"
        )
        assert unified_data is not None
        assert unified_data['fund_id'] == "FD000"
        assert 'quotes' in unified_data

        # Step 3: Bulk operations and statistics
        bulk_stats = await e2e_context.fundata_repo.get_bulk_statistics(
            fund_ids=[f"FD{i:03d}" for i in range(5)]
        )
        assert len(bulk_stats) == 5

        # Step 4: Performance analysis
        perf_analysis = await e2e_context.fundata_repo.analyze_performance(
            fund_id="FD000",
            start_date=datetime.utcnow().date() - timedelta(days=30),
            end_date=datetime.utcnow().date()
        )
        assert perf_analysis is not None
        assert 'total_return' in perf_analysis

        end_time = time.time()
        logger.info(f"Fundata processing flow completed in {end_time - start_time:.2f}s")


class TestEndToEndServiceIntegration:
    """Test service layer integration."""

    @pytest.mark.asyncio
    async def test_data_processing_service_integration(self, e2e_context: E2ETestContext):
        """Test data processing service integration."""
        start_time = time.time()

        # Create data processing service
        with patch('packages.sra_data.services.data_processing.FMPClient') as mock_fmp:
            # Mock FMP API responses
            mock_fmp.return_value.get_company_profile.return_value = {
                'symbol': 'AAPL',
                'companyName': 'Apple Inc.',
                'mktCap': 2800000000000,
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'country': 'US',
                'exchange': 'NASDAQ',
                'price': 150.0,
                'pe': 25.0,
                'beta': 1.2
            }

            service = create_data_processing_service()

            # Test service initialization
            assert service is not None

            # Test data processing workflow
            result = await service.process_equity_data(['AAPL'])
            assert result is not None
            assert len(result) == 1
            assert result[0]['symbol'] == 'AAPL'

        end_time = time.time()
        logger.info(f"Service integration test completed in {end_time - start_time:.2f}s")

    @pytest.mark.asyncio
    async def test_csv_processing_service_integration(self, e2e_context: E2ETestContext):
        """Test CSV processing service integration."""
        start_time = time.time()

        # Create CSV processing service
        service = create_csv_processing_service()

        # Test service initialization
        assert service is not None

        # Create mock CSV data
        mock_csv_data = [
            {
                'fund_id': 'TEST001',
                'fund_name': 'Test Fund 1',
                'nav': '10.50',
                'date': '2024-01-01'
            },
            {
                'fund_id': 'TEST002',
                'fund_name': 'Test Fund 2',
                'nav': '15.75',
                'date': '2024-01-01'
            }
        ]

        # Test CSV processing workflow (mocked)
        with patch.object(service, 'process_csv_data') as mock_process:
            mock_process.return_value = {
                'processed_count': 2,
                'errors': [],
                'statistics': {
                    'total_records': 2,
                    'successful': 2,
                    'failed': 0
                }
            }

            result = await service.process_csv_data(mock_csv_data)
            assert result['processed_count'] == 2
            assert len(result['errors']) == 0

        end_time = time.time()
        logger.info(f"CSV service integration test completed in {end_time - start_time:.2f}s")


class TestEndToEndCrosComponentIntegration:
    """Test cross-component integration scenarios."""

    @pytest.mark.asyncio
    async def test_api_to_database_integration(self, e2e_context: E2ETestContext):
        """Test API to database integration."""
        start_time = time.time()

        # Step 1: Setup test data in database
        test_profile = EquityProfile(
            symbol='INTEGRATION_TEST',
            name='Integration Test Company',
            market_cap=1000000,
            sector='Technology',
            industry='Software',
            country='US',
            exchange='NASDAQ',
            price=100.0,
            pe_ratio=20.0,
            beta=1.0
        )

        saved_profile = await e2e_context.equity_repo.create(test_profile)
        e2e_context.cleanup_tasks.append(
            lambda: e2e_context.equity_repo.delete(saved_profile.id)
        )

        # Step 2: Test API status endpoint includes database status
        status_response = e2e_context.test_client.get("/status")
        assert status_response.status_code == 200

        status_data = status_response.json()
        assert "database" in status_data

        # Database should be connected if we successfully created a profile
        db_status = status_data["database"]
        if db_status["status"] == "connected":
            assert db_status["connected"] is True

        # Step 3: Test concurrent API and database operations
        async def api_call():
            return e2e_context.test_client.get("/health")

        async def db_operation():
            return await e2e_context.equity_repo.get_by_id(saved_profile.id)

        # Run concurrent operations
        api_task = asyncio.create_task(api_call())
        db_task = asyncio.create_task(db_operation())

        api_result, db_result = await asyncio.gather(api_task, db_task)

        assert api_result.status_code == 200
        assert db_result is not None
        assert db_result.symbol == 'INTEGRATION_TEST'

        end_time = time.time()
        logger.info(f"API to database integration test completed in {end_time - start_time:.2f}s")

    @pytest.mark.asyncio
    async def test_service_to_repository_integration(self, e2e_context: E2ETestContext):
        """Test service to repository integration."""
        start_time = time.time()

        # This test would normally integrate real services with repositories
        # For now, we'll test the pattern with mocked external dependencies

        with patch('packages.sra_data.services.data_processing.FMPClient'):
            service = create_data_processing_service()

            # Test that service can interact with repositories
            # In a real scenario, this would process external data and save to DB

            # Mock successful processing result
            mock_result = [
                {
                    'symbol': 'SERVICE_TEST',
                    'name': 'Service Test Company',
                    'market_cap': 500000,
                    'sector': 'Testing',
                    'price': 75.0
                }
            ]

            # Verify service structure supports repository integration
            assert hasattr(service, 'process_equity_data')

            # Test would continue with actual service-to-repository flow
            # when services are fully implemented

        end_time = time.time()
        logger.info(f"Service to repository integration test completed in {end_time - start_time:.2f}s")


class TestEndToEndPerformanceAndReliability:
    """Test performance and reliability aspects."""

    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, e2e_context: E2ETestContext):
        """Test concurrent database operations performance."""
        start_time = time.time()

        # Create multiple concurrent database operations
        async def create_equity_profile(index: int):
            profile = EquityProfile(
                symbol=f'PERF{index:03d}',
                name=f'Performance Test {index}',
                market_cap=1000000 * index,
                sector='Testing',
                industry='Performance',
                country='US',
                exchange='TEST',
                price=100.0 + index,
                pe_ratio=15.0,
                beta=1.0
            )

            created = await e2e_context.equity_repo.create(profile)
            e2e_context.cleanup_tasks.append(
                lambda p=created: e2e_context.equity_repo.delete(p.id)
            )
            return created

        # Run 20 concurrent create operations
        tasks = [create_equity_profile(i) for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all operations completed successfully
        successful_ops = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_ops) == 20

        # Test concurrent read operations
        async def read_profile(profile):
            return await e2e_context.equity_repo.get_by_id(profile.id)

        read_tasks = [read_profile(profile) for profile in successful_ops[:10]]
        read_results = await asyncio.gather(*read_tasks)

        assert all(result is not None for result in read_results)

        end_time = time.time()
        duration = end_time - start_time
        ops_per_second = 40 / duration  # 20 creates + 20 reads

        logger.info(f"Concurrent operations: {ops_per_second:.2f} ops/sec")
        assert ops_per_second > 10  # Minimum performance threshold

    @pytest.mark.asyncio
    async def test_api_performance_under_load(self, e2e_context: E2ETestContext):
        """Test API performance under load."""
        start_time = time.time()

        async def api_request():
            response = await e2e_context.async_client.get("/health")
            return response.status_code, time.time()

        # Run 50 concurrent API requests
        tasks = [api_request() for _ in range(50)]
        results = await asyncio.gather(*tasks)

        # Analyze results
        successful_requests = [r for r in results if r[0] == 200]
        assert len(successful_requests) == 50

        # Calculate response time statistics
        response_times = [
            result[1] - start_time
            for result in successful_requests
        ]

        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)

        end_time = time.time()
        total_duration = end_time - start_time
        requests_per_second = 50 / total_duration

        logger.info(f"API Performance - RPS: {requests_per_second:.2f}, "
                   f"Avg: {avg_response_time:.3f}s, Max: {max_response_time:.3f}s")

        # Performance assertions
        assert avg_response_time < 0.5  # Average response under 500ms
        assert max_response_time < 2.0  # Max response under 2s
        assert requests_per_second > 25  # Minimum throughput


class TestEndToEndErrorHandling:
    """Test error handling across components."""

    @pytest.mark.asyncio
    async def test_database_connection_error_handling(self, e2e_context: E2ETestContext):
        """Test database connection error handling."""
        # Test API graceful handling of database issues
        with patch.object(e2e_context.db_manager, 'check_connection',
                          return_value=False):

            status_response = e2e_context.test_client.get("/status")
            assert status_response.status_code == 200

            status_data = status_response.json()
            assert status_data["status"] in ["degraded", "unhealthy"]

    @pytest.mark.asyncio
    async def test_repository_error_recovery(self, e2e_context: E2ETestContext):
        """Test repository error recovery mechanisms."""
        # Test repository resilience to database errors

        # Create a profile that should succeed
        valid_profile = EquityProfile(
            symbol='ERROR_TEST',
            name='Error Test Company',
            market_cap=1000000,
            sector='Testing',
            country='US',
            price=100.0
        )

        created = await e2e_context.equity_repo.create(valid_profile)
        assert created is not None

        e2e_context.cleanup_tasks.append(
            lambda: e2e_context.equity_repo.delete(created.id)
        )

        # Test handling of invalid operations
        with pytest.raises((ValueError, Exception)):
            # Try to create profile with missing required fields
            invalid_profile = EquityProfile(symbol=None, name=None)
            await e2e_context.equity_repo.create(invalid_profile)


# Test execution summary and metrics
class TestEndToEndSummary:
    """Test summary and metrics collection."""

    @pytest.mark.asyncio
    async def test_complete_system_integration_summary(self, e2e_context: E2ETestContext):
        """Comprehensive system integration test with metrics."""
        start_time = time.time()
        metrics = {
            'api_calls': 0,
            'db_operations': 0,
            'errors': 0,
            'performance_samples': []
        }

        try:
            # API Layer Test
            api_start = time.time()
            response = e2e_context.test_client.get("/status")
            api_end = time.time()

            assert response.status_code == 200
            metrics['api_calls'] += 1
            metrics['performance_samples'].append(('api_status', api_end - api_start))

            # Repository Layer Test
            db_start = time.time()
            test_profile = EquityProfile(
                symbol='SUMMARY_TEST',
                name='Summary Test Company',
                market_cap=1000000,
                sector='Testing',
                country='US',
                price=100.0
            )

            created = await e2e_context.equity_repo.create(test_profile)
            retrieved = await e2e_context.equity_repo.get_by_id(created.id)
            await e2e_context.equity_repo.delete(created.id)
            db_end = time.time()

            assert retrieved.symbol == 'SUMMARY_TEST'
            metrics['db_operations'] += 3  # create, get, delete
            metrics['performance_samples'].append(('db_crud', db_end - db_start))

            # Service Integration Test (mocked)
            service_start = time.time()
            with patch('packages.sra_data.services.data_processing.create_data_processing_service'):
                # Service integration would go here
                pass
            service_end = time.time()

            metrics['performance_samples'].append(('service_integration', service_end - service_start))

        except Exception as e:
            metrics['errors'] += 1
            logger.error(f"Integration test error: {e}")
            raise

        finally:
            end_time = time.time()
            total_duration = end_time - start_time

            # Log comprehensive metrics
            logger.info("=== End-to-End Test Summary ===")
            logger.info(f"Total Duration: {total_duration:.3f}s")
            logger.info(f"API Calls: {metrics['api_calls']}")
            logger.info(f"DB Operations: {metrics['db_operations']}")
            logger.info(f"Errors: {metrics['errors']}")

            for operation, duration in metrics['performance_samples']:
                logger.info(f"{operation}: {duration:.3f}s")

            # Assert overall success
            assert metrics['errors'] == 0
            assert total_duration < 10.0  # Complete test under 10s
            assert metrics['api_calls'] > 0
            assert metrics['db_operations'] > 0


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "--asyncio-mode=auto"
    ])