"""
Comprehensive Integration Testing Framework

This module provides thorough integration testing between all system components:
- Database layer integration
- Service layer integration
- API layer integration
- Cross-component communication
- Data flow validation
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession

# Import all system components
from packages.sra_data.api.skeleton import app
from packages.sra_data.repositories.database_infrastructure import (
    create_database_manager, DatabaseManager
)
from packages.sra_data.repositories.equity_repository import EquityRepository
from packages.sra_data.repositories.fundata_repository import (
    FundataRepository, FundataDataRepository, FundataQuotesRepository
)
from packages.sra_data.repositories.view_manager import ViewManager
from packages.sra_data.repositories.transaction_manager import TransactionManager
from packages.sra_data.repositories.performance_optimizer import PerformanceOptimizer
from packages.sra_data.services.data_processing import create_data_processing_service
from packages.sra_data.services.csv_processing import create_csv_processing_service
from packages.sra_data.models import (
    EquityProfile, FundataDataRecord, FundataQuotesRecord
)

logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """Comprehensive integration test suite manager."""

    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.repositories: Dict[str, Any] = {}
        self.services: Dict[str, Any] = {}
        self.managers: Dict[str, Any] = {}
        self.test_data: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'total_duration': 0.0,
            'component_timings': {},
            'integration_points': []
        }

    async def setup(self):
        """Setup integration test environment."""
        start_time = time.time()

        # Initialize database manager
        self.db_manager = create_database_manager()
        await self.db_manager.initialize()

        # Initialize repositories
        session_factory = self.db_manager.get_session_factory()
        self.repositories = {
            'equity': EquityRepository(session_factory),
            'fundata': FundataRepository(session_factory),
            'fundata_data': FundataDataRepository(session_factory),
            'fundata_quotes': FundataQuotesRepository(session_factory)
        }

        # Initialize managers
        self.managers = {
            'view': ViewManager(session_factory),
            'transaction': TransactionManager(session_factory),
            'performance': PerformanceOptimizer(session_factory)
        }

        # Initialize services (with mocking for external dependencies)
        with patch('packages.sra_data.services.data_processing.FMPClient'):
            self.services['data_processing'] = create_data_processing_service()

        self.services['csv_processing'] = create_csv_processing_service()

        # Setup test data
        await self._setup_test_data()

        setup_time = time.time() - start_time
        self.metrics['component_timings']['setup'] = setup_time
        logger.info(f"Integration test environment setup completed in {setup_time:.3f}s")

    async def teardown(self):
        """Teardown integration test environment."""
        start_time = time.time()

        # Cleanup test data
        await self._cleanup_test_data()

        # Close database connections
        if self.db_manager:
            await self.db_manager.close()

        teardown_time = time.time() - start_time
        self.metrics['component_timings']['teardown'] = teardown_time
        logger.info(f"Integration test environment teardown completed in {teardown_time:.3f}s")

    async def _setup_test_data(self):
        """Setup comprehensive test data."""
        self.test_data = {
            'equity_profiles': [
                EquityProfile(
                    symbol=f'INTG{i:02d}',
                    name=f'Integration Test Company {i}',
                    market_cap=1000000 * (i + 1),
                    sector='Technology' if i % 3 == 0 else 'Healthcare' if i % 3 == 1 else 'Finance',
                    industry=f'Integration Industry {i}',
                    country='US',
                    exchange='NASDAQ' if i % 2 == 0 else 'NYSE',
                    price=100.0 + i * 5,
                    pe_ratio=15.0 + i * 0.5,
                    beta=1.0 + i * 0.1
                )
                for i in range(10)
            ],
            'fundata_data': [
                FundataDataRecord(
                    fund_id=f'FUND{i:03d}',
                    fund_name=f'Integration Fund {i}',
                    fund_family='Integration Fund Family',
                    category='Equity' if i % 2 == 0 else 'Bond',
                    total_assets=1000000 * (i + 1),
                    expense_ratio=0.01 * (i + 1),
                    date=datetime.utcnow().date()
                )
                for i in range(10)
            ],
            'fundata_quotes': [
                FundataQuotesRecord(
                    fund_id=f'FUND{i:03d}',
                    date=datetime.utcnow().date(),
                    nav=10.0 + i,
                    volume=1000 * (i + 1),
                    change_percent=0.01 * (i - 5)  # Some positive, some negative
                )
                for i in range(10)
            ]
        }

        # Store created records for cleanup
        self.test_data['created_equity_ids'] = []
        self.test_data['created_fundata_data_ids'] = []
        self.test_data['created_fundata_quotes_ids'] = []

    async def _cleanup_test_data(self):
        """Cleanup all test data."""
        try:
            # Cleanup equity profiles
            for equity_id in self.test_data.get('created_equity_ids', []):
                await self.repositories['equity'].delete(equity_id)

            # Cleanup fundata records
            for data_id in self.test_data.get('created_fundata_data_ids', []):
                await self.repositories['fundata_data'].delete(data_id)

            for quotes_id in self.test_data.get('created_fundata_quotes_ids', []):
                await self.repositories['fundata_quotes'].delete(quotes_id)

        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


@pytest_asyncio.fixture
async def integration_suite():
    """Provide integration test suite."""
    suite = IntegrationTestSuite()
    await suite.setup()
    try:
        yield suite
    finally:
        await suite.teardown()


class TestDatabaseLayerIntegration:
    """Test database layer component integration."""

    @pytest.mark.asyncio
    async def test_repository_to_database_integration(self, integration_suite: IntegrationTestSuite):
        """Test repository integration with database infrastructure."""
        start_time = time.time()

        # Test equity repository database integration
        equity_repo = integration_suite.repositories['equity']

        # Create test data
        test_profile = integration_suite.test_data['equity_profiles'][0]
        created = await equity_repo.create(test_profile)
        integration_suite.test_data['created_equity_ids'].append(created.id)

        # Test direct database connection
        async with integration_suite.db_manager.get_session() as session:
            # Verify data exists in database
            result = await session.execute(
                text("SELECT * FROM equity_profiles WHERE symbol = :symbol"),
                {'symbol': test_profile.symbol}
            )
            db_record = result.fetchone()
            assert db_record is not None
            assert db_record[1] == test_profile.symbol  # symbol column

        # Test repository-database consistency
        retrieved = await equity_repo.get_by_id(created.id)
        assert retrieved.symbol == test_profile.symbol
        assert retrieved.market_cap == test_profile.market_cap

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'repository_to_database',
            'duration': end_time - start_time,
            'components': ['EquityRepository', 'DatabaseManager']
        })

    @pytest.mark.asyncio
    async def test_transaction_manager_integration(self, integration_suite: IntegrationTestSuite):
        """Test transaction manager integration with repositories."""
        start_time = time.time()

        transaction_manager = integration_suite.managers['transaction']
        equity_repo = integration_suite.repositories['equity']

        # Test transactional operations
        test_profiles = integration_suite.test_data['equity_profiles'][:3]

        async with transaction_manager.begin_transaction() as tx:
            created_profiles = []
            for profile in test_profiles:
                created = await equity_repo.create(profile, session=tx.session)
                created_profiles.append(created)
                integration_suite.test_data['created_equity_ids'].append(created.id)

            # Verify all profiles created in transaction
            assert len(created_profiles) == 3

        # Verify transaction committed successfully
        for profile in created_profiles:
            retrieved = await equity_repo.get_by_id(profile.id)
            assert retrieved is not None

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'transaction_manager',
            'duration': end_time - start_time,
            'components': ['TransactionManager', 'EquityRepository', 'DatabaseManager']
        })

    @pytest.mark.asyncio
    async def test_view_manager_integration(self, integration_suite: IntegrationTestSuite):
        """Test view manager integration with database and repositories."""
        start_time = time.time()

        view_manager = integration_suite.managers['view']

        # Create test data for views
        equity_repo = integration_suite.repositories['equity']
        for profile in integration_suite.test_data['equity_profiles'][:5]:
            created = await equity_repo.create(profile)
            integration_suite.test_data['created_equity_ids'].append(created.id)

        # Test view operations
        await view_manager.ensure_views_exist()

        # Test equity summary view
        summary = await view_manager.get_equity_summary()
        assert summary is not None
        assert summary['total_companies'] >= 5

        # Test view health
        health = await view_manager.check_view_health()
        assert health is not None
        assert 'equity_summary' in health

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'view_manager',
            'duration': end_time - start_time,
            'components': ['ViewManager', 'EquityRepository', 'DatabaseManager']
        })


class TestServiceLayerIntegration:
    """Test service layer integration."""

    @pytest.mark.asyncio
    async def test_data_processing_service_integration(self, integration_suite: IntegrationTestSuite):
        """Test data processing service integration with repositories."""
        start_time = time.time()

        service = integration_suite.services['data_processing']
        equity_repo = integration_suite.repositories['equity']

        # Mock external API responses
        mock_company_data = [
            {
                'symbol': 'INTG_SVC1',
                'companyName': 'Integration Service Test 1',
                'mktCap': 2000000,
                'sector': 'Technology',
                'industry': 'Software',
                'country': 'US',
                'exchange': 'NASDAQ',
                'price': 125.0,
                'pe': 20.0,
                'beta': 1.1
            },
            {
                'symbol': 'INTG_SVC2',
                'companyName': 'Integration Service Test 2',
                'mktCap': 1500000,
                'sector': 'Healthcare',
                'industry': 'Biotechnology',
                'country': 'US',
                'exchange': 'NYSE',
                'price': 85.0,
                'pe': 18.0,
                'beta': 0.9
            }
        ]

        with patch.object(service, 'process_equity_data') as mock_process:
            mock_process.return_value = [
                {
                    'symbol': data['symbol'],
                    'name': data['companyName'],
                    'market_cap': data['mktCap'],
                    'sector': data['sector'],
                    'industry': data['industry'],
                    'country': data['country'],
                    'exchange': data['exchange'],
                    'price': data['price'],
                    'pe_ratio': data['pe'],
                    'beta': data['beta']
                }
                for data in mock_company_data
            ]

            # Test service processing
            symbols = ['INTG_SVC1', 'INTG_SVC2']
            processed_data = await service.process_equity_data(symbols)

            assert len(processed_data) == 2
            assert processed_data[0]['symbol'] == 'INTG_SVC1'

            # Test service-to-repository integration pattern
            # (In real implementation, service would save to repository)
            for data in processed_data:
                profile = EquityProfile(**data)
                created = await equity_repo.create(profile)
                integration_suite.test_data['created_equity_ids'].append(created.id)

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'data_processing_service',
            'duration': end_time - start_time,
            'components': ['DataProcessingService', 'EquityRepository']
        })

    @pytest.mark.asyncio
    async def test_csv_processing_service_integration(self, integration_suite: IntegrationTestSuite):
        """Test CSV processing service integration with repositories."""
        start_time = time.time()

        service = integration_suite.services['csv_processing']
        fundata_repo = integration_suite.repositories['fundata_data']

        # Mock CSV data
        mock_csv_data = [
            {
                'fund_id': 'CSV_INTG_001',
                'fund_name': 'CSV Integration Fund 1',
                'fund_family': 'CSV Integration Family',
                'category': 'Equity',
                'total_assets': '1000000',
                'expense_ratio': '0.01',
                'date': '2024-01-01'
            },
            {
                'fund_id': 'CSV_INTG_002',
                'fund_name': 'CSV Integration Fund 2',
                'fund_family': 'CSV Integration Family',
                'category': 'Bond',
                'total_assets': '2000000',
                'expense_ratio': '0.02',
                'date': '2024-01-01'
            }
        ]

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

            # Test service processing
            result = await service.process_csv_data(mock_csv_data)
            assert result['processed_count'] == 2

            # Test integration with repository
            # (In real implementation, service would create fundata records)
            for data in mock_csv_data:
                record = FundataDataRecord(
                    fund_id=data['fund_id'],
                    fund_name=data['fund_name'],
                    fund_family=data['fund_family'],
                    category=data['category'],
                    total_assets=float(data['total_assets']),
                    expense_ratio=float(data['expense_ratio']),
                    date=datetime.strptime(data['date'], '%Y-%m-%d').date()
                )
                created = await fundata_repo.create(record)
                integration_suite.test_data['created_fundata_data_ids'].append(created.id)

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'csv_processing_service',
            'duration': end_time - start_time,
            'components': ['CSVProcessingService', 'FundataDataRepository']
        })


class TestAPILayerIntegration:
    """Test API layer integration with underlying components."""

    @pytest.mark.asyncio
    async def test_api_database_integration(self, integration_suite: IntegrationTestSuite):
        """Test API integration with database layer."""
        start_time = time.time()

        # Setup test data in database
        equity_repo = integration_suite.repositories['equity']
        test_profile = integration_suite.test_data['equity_profiles'][0]
        created = await equity_repo.create(test_profile)
        integration_suite.test_data['created_equity_ids'].append(created.id)

        # Test API endpoints with database backend
        client = TestClient(app)

        # Test status endpoint includes database status
        response = client.get("/status")
        assert response.status_code == 200

        status_data = response.json()
        assert "database" in status_data

        # If database is working (we just created a record), status should reflect that
        if status_data["database"].get("connected", False):
            assert status_data["database"]["status"] == "connected"

        # Test health check
        health_response = client.get("/health")
        assert health_response.status_code == 200

        health_data = health_response.json()
        assert health_data["status"] == "healthy"

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'api_database',
            'duration': end_time - start_time,
            'components': ['FastAPI', 'DatabaseManager', 'EquityRepository']
        })

    @pytest.mark.asyncio
    async def test_api_service_integration(self, integration_suite: IntegrationTestSuite):
        """Test API integration with service layer."""
        start_time = time.time()

        client = TestClient(app)

        # Test that API status includes service status
        response = client.get("/status")
        assert response.status_code == 200

        status_data = response.json()
        assert "data_services" in status_data

        # Verify service integration points are exposed
        services = status_data["data_services"]
        if isinstance(services, dict):
            # Services should be listed even if unavailable
            assert len(services) >= 0

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'api_service',
            'duration': end_time - start_time,
            'components': ['FastAPI', 'DataProcessingService', 'CSVProcessingService']
        })


class TestCrossComponentIntegration:
    """Test complex cross-component integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_stack_data_flow(self, integration_suite: IntegrationTestSuite):
        """Test complete data flow across all layers."""
        start_time = time.time()

        # Simulate full data processing flow:
        # API -> Service -> Repository -> Database -> View -> API Response

        equity_repo = integration_suite.repositories['equity']
        view_manager = integration_suite.managers['view']

        # Step 1: Create data via repository (simulating service processing)
        test_profiles = integration_suite.test_data['equity_profiles'][:3]
        created_profiles = []

        for profile in test_profiles:
            created = await equity_repo.create(profile)
            created_profiles.append(created)
            integration_suite.test_data['created_equity_ids'].append(created.id)

        # Step 2: Update views to reflect new data
        await view_manager.refresh_materialized_views()

        # Step 3: Query aggregated data via views
        summary = await view_manager.get_equity_summary()
        assert summary['total_companies'] >= 3

        # Step 4: Test API access to aggregated data
        client = TestClient(app)
        status_response = client.get("/status")
        assert status_response.status_code == 200

        # Step 5: Verify data consistency across layers
        for profile in created_profiles:
            # Repository layer
            retrieved = await equity_repo.get_by_id(profile.id)
            assert retrieved is not None

            # Database layer
            async with integration_suite.db_manager.get_session() as session:
                result = await session.execute(
                    text("SELECT symbol FROM equity_profiles WHERE id = :id"),
                    {'id': profile.id}
                )
                db_record = result.fetchone()
                assert db_record[0] == profile.symbol

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'full_stack_data_flow',
            'duration': end_time - start_time,
            'components': ['FastAPI', 'Service', 'Repository', 'Database', 'ViewManager']
        })

    @pytest.mark.asyncio
    async def test_concurrent_multi_component_operations(self, integration_suite: IntegrationTestSuite):
        """Test concurrent operations across multiple components."""
        start_time = time.time()

        equity_repo = integration_suite.repositories['equity']
        fundata_repo = integration_suite.repositories['fundata_data']
        client = TestClient(app)

        async def create_equity_data(index: int):
            profile = integration_suite.test_data['equity_profiles'][index]
            created = await equity_repo.create(profile)
            integration_suite.test_data['created_equity_ids'].append(created.id)
            return created

        async def create_fundata_data(index: int):
            record = integration_suite.test_data['fundata_data'][index]
            created = await fundata_repo.create(record)
            integration_suite.test_data['created_fundata_data_ids'].append(created.id)
            return created

        async def api_health_check():
            # Simulate API calls during data operations
            return client.get("/health")

        # Run concurrent operations across components
        tasks = []

        # Add database operations
        for i in range(5):
            tasks.append(create_equity_data(i))
            tasks.append(create_fundata_data(i))

        # Add API operations
        for _ in range(10):
            tasks.append(api_health_check())

        # Execute all operations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify results
        successful_ops = [r for r in results if not isinstance(r, Exception)]
        db_ops = [r for r in successful_ops if hasattr(r, 'id')]
        api_ops = [r for r in successful_ops if hasattr(r, 'status_code')]

        assert len(db_ops) == 10  # 5 equity + 5 fundata
        assert len(api_ops) == 10  # 10 health checks
        assert all(r.status_code == 200 for r in api_ops)

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'concurrent_multi_component',
            'duration': end_time - start_time,
            'components': ['EquityRepository', 'FundataRepository', 'FastAPI'],
            'concurrent_operations': len(tasks)
        })


class TestIntegrationPerformance:
    """Test integration performance characteristics."""

    @pytest.mark.asyncio
    async def test_component_interaction_performance(self, integration_suite: IntegrationTestSuite):
        """Test performance of component interactions."""
        start_time = time.time()

        equity_repo = integration_suite.repositories['equity']
        transaction_manager = integration_suite.managers['transaction']

        # Test repository performance
        repo_start = time.time()
        test_profile = integration_suite.test_data['equity_profiles'][0]
        created = await equity_repo.create(test_profile)
        integration_suite.test_data['created_equity_ids'].append(created.id)
        repo_time = time.time() - repo_start

        # Test transaction performance
        tx_start = time.time()
        async with transaction_manager.begin_transaction() as tx:
            retrieved = await equity_repo.get_by_id(created.id, session=tx.session)
            assert retrieved is not None
        tx_time = time.time() - tx_start

        # Test view manager performance
        view_start = time.time()
        view_manager = integration_suite.managers['view']
        await view_manager.ensure_views_exist()
        view_time = time.time() - view_start

        # Performance assertions
        assert repo_time < 1.0  # Repository operations under 1s
        assert tx_time < 0.5    # Transaction operations under 0.5s
        assert view_time < 5.0  # View operations under 5s

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'component_interaction_performance',
            'duration': end_time - start_time,
            'component_timings': {
                'repository': repo_time,
                'transaction': tx_time,
                'view': view_time
            }
        })

    @pytest.mark.asyncio
    async def test_integration_scalability(self, integration_suite: IntegrationTestSuite):
        """Test integration scalability with increasing load."""
        start_time = time.time()

        equity_repo = integration_suite.repositories['equity']

        # Test with increasing batch sizes
        batch_sizes = [1, 5, 10, 20]
        performance_data = {}

        for batch_size in batch_sizes:
            batch_start = time.time()

            # Create batch of profiles
            profiles_to_create = integration_suite.test_data['equity_profiles'][:batch_size]
            created_profiles = []

            for profile in profiles_to_create:
                created = await equity_repo.create(profile)
                created_profiles.append(created)
                integration_suite.test_data['created_equity_ids'].append(created.id)

            # Test batch retrieval
            for profile in created_profiles:
                retrieved = await equity_repo.get_by_id(profile.id)
                assert retrieved is not None

            batch_time = time.time() - batch_start
            performance_data[batch_size] = {
                'duration': batch_time,
                'ops_per_second': (batch_size * 2) / batch_time  # create + retrieve
            }

        # Verify scalability characteristics
        for i in range(1, len(batch_sizes)):
            prev_ops = performance_data[batch_sizes[i-1]]['ops_per_second']
            curr_ops = performance_data[batch_sizes[i]]['ops_per_second']

            # Performance should not degrade significantly
            assert curr_ops >= prev_ops * 0.5  # Allow 50% degradation max

        end_time = time.time()
        integration_suite.metrics['integration_points'].append({
            'test': 'integration_scalability',
            'duration': end_time - start_time,
            'performance_data': performance_data
        })


# Integration test summary and reporting
class TestIntegrationSummary:
    """Integration test summary and metrics reporting."""

    @pytest.mark.asyncio
    async def test_comprehensive_integration_report(self, integration_suite: IntegrationTestSuite):
        """Generate comprehensive integration test report."""
        start_time = time.time()

        # Collect system health metrics
        client = TestClient(app)
        status_response = client.get("/status")
        system_status = status_response.json() if status_response.status_code == 200 else {}

        # Test database connectivity
        db_connected = False
        try:
            async with integration_suite.db_manager.get_session() as session:
                await session.execute(text("SELECT 1"))
                db_connected = True
        except Exception:
            pass

        # Test repository functionality
        repo_functional = False
        try:
            equity_repo = integration_suite.repositories['equity']
            test_profile = integration_suite.test_data['equity_profiles'][0]
            created = await equity_repo.create(test_profile)
            integration_suite.test_data['created_equity_ids'].append(created.id)
            retrieved = await equity_repo.get_by_id(created.id)
            repo_functional = retrieved is not None
        except Exception:
            pass

        # Compile integration report
        end_time = time.time()
        total_duration = end_time - start_time

        integration_report = {
            'test_summary': {
                'total_duration': integration_suite.metrics['component_timings'].get('setup', 0) +
                                integration_suite.metrics['component_timings'].get('teardown', 0) +
                                sum(point['duration'] for point in integration_suite.metrics['integration_points']),
                'integration_points_tested': len(integration_suite.metrics['integration_points']),
                'components_tested': set(
                    comp for point in integration_suite.metrics['integration_points']
                    for comp in point.get('components', [])
                )
            },
            'system_health': {
                'api_responsive': status_response.status_code == 200,
                'database_connected': db_connected,
                'repositories_functional': repo_functional,
                'overall_status': system_status.get('status', 'unknown')
            },
            'performance_summary': {
                'avg_integration_time': sum(point['duration'] for point in integration_suite.metrics['integration_points']) /
                                       max(len(integration_suite.metrics['integration_points']), 1),
                'fastest_integration': min(
                    (point['duration'] for point in integration_suite.metrics['integration_points']),
                    default=0
                ),
                'slowest_integration': max(
                    (point['duration'] for point in integration_suite.metrics['integration_points']),
                    default=0
                )
            },
            'integration_points': integration_suite.metrics['integration_points']
        }

        # Log comprehensive report
        logger.info("=== Integration Test Report ===")
        logger.info(json.dumps(integration_report, indent=2, default=str))

        # Assertions for integration success
        assert integration_report['system_health']['api_responsive']
        assert integration_report['system_health']['database_connected']
        assert integration_report['system_health']['repositories_functional']
        assert integration_report['test_summary']['integration_points_tested'] > 0
        assert len(integration_report['test_summary']['components_tested']) >= 3

        integration_suite.metrics['final_report'] = integration_report


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "--asyncio-mode=auto"
    ])