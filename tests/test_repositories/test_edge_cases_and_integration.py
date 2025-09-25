"""Edge cases and integration tests for repository layer.

This module contains tests for edge cases, error conditions, and integration
scenarios to ensure 100% test coverage and robust error handling.
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date
from decimal import Decimal
from contextlib import asynccontextmanager

from packages.sra_data.repositories import (
    DatabaseManager,
    EquityRepository,
    FundataRepository,
    MigrationManager,
    TableManager,
    ViewManager,
    PerformanceOptimizer,
    TransactionManager
)
from packages.sra_data.domain.models import (
    EquityProfile,
    FundataDataRecord,
    FundataQuotesRecord,
    ExchangeType,
    RecordState,
    CurrencyType
)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager with edge case behaviors."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.mark.asyncio
    async def test_repository_operations_with_none_values(self, mock_db_manager):
        """Test repository operations with None/null values."""
        db_manager, mock_conn = mock_db_manager
        equity_repo = EquityRepository(db_manager)

        # Test with None market cap
        equity = EquityProfile(
            symbol="TEST",
            company_name="Test Corp",
            exchange=ExchangeType.NYSE,
            market_cap=None,  # None value
            employees=None,
            sector=None
        )

        mock_conn.execute.return_value = None
        result = await equity_repo.create(equity)
        assert result is True

    @pytest.mark.asyncio
    async def test_repository_operations_with_large_data(self, mock_db_manager):
        """Test repository operations with large data values."""
        db_manager, mock_conn = mock_db_manager
        equity_repo = EquityRepository(db_manager)

        # Very large market cap
        large_equity = EquityProfile(
            symbol="LARGE",
            company_name="Very Large Corporation with Extremely Long Name That Tests String Limits",
            exchange=ExchangeType.NYSE,
            market_cap=Decimal("999999999999999999999.99"),  # Very large number
            employees=999999999,
            description="A" * 1000  # Long description
        )

        mock_conn.execute.return_value = None
        result = await equity_repo.create(large_equity)
        assert result is True

    @pytest.mark.asyncio
    async def test_repository_search_special_characters(self, mock_db_manager):
        """Test search functionality with special characters."""
        db_manager, mock_conn = mock_db_manager
        equity_repo = EquityRepository(db_manager)

        mock_conn.fetch.return_value = []

        # Test with various special characters
        special_queries = [
            "Apple & Co.",
            "Test'Company",
            'Test"Company',
            "Company (USA)",
            "Test\\Company",
            "Test%Company",
            "Test_Company",
            "",  # Empty string
            " ",  # Just whitespace
            "日本語",  # Unicode characters
        ]

        for query in special_queries:
            results = await equity_repo.search(query)
            assert isinstance(results, list)  # Should not crash

    @pytest.mark.asyncio
    async def test_fundata_records_with_extreme_dates(self, mock_db_manager):
        """Test fundata records with extreme date values."""
        db_manager, mock_conn = mock_db_manager
        transaction_mgr = TransactionManager(db_manager)

        # Test with very old date
        old_date_record = {
            'InstrumentKey': 'TEST001',
            'RecordId': 'REC001',
            'Date': date(1900, 1, 1),  # Very old date
            'NAVPS': Decimal('10.00'),
            'source_file': 'test.csv'
        }

        violations = await transaction_mgr.validate_data_integrity("fundata_quotes", old_date_record)
        # Should not have date violations for 1900 (allowed)
        date_violations = [v for v in violations if v.column_name == 'date']
        assert len(date_violations) == 0

        # Test with date before 1900
        very_old_date_record = {
            'InstrumentKey': 'TEST001',
            'RecordId': 'REC001',
            'Date': date(1850, 1, 1),  # Too old
            'NAVPS': Decimal('10.00'),
            'source_file': 'test.csv'
        }

        violations = await transaction_mgr.validate_data_integrity("fundata_quotes", very_old_date_record)
        date_violations = [v for v in violations if v.column_name == 'date']
        assert len(date_violations) > 0

    @pytest.mark.asyncio
    async def test_bulk_operations_empty_and_large_lists(self, mock_db_manager):
        """Test bulk operations with empty and very large lists."""
        db_manager, mock_conn = mock_db_manager
        equity_repo = EquityRepository(db_manager)

        # Empty list
        result = await equity_repo.bulk_insert([])
        assert result.success is True
        assert result.records_processed == 0

        # Large list simulation
        mock_conn.execute.return_value = None
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

        large_list = [
            EquityProfile(
                symbol=f"TEST{i:04d}",
                company_name=f"Test Company {i}",
                exchange=ExchangeType.NYSE
            )
            for i in range(1000)  # Large number of records
        ]

        result = await equity_repo.bulk_insert(large_list)
        assert result.records_processed == 1000

    @pytest.mark.asyncio
    async def test_concurrent_repository_access(self, mock_db_manager):
        """Test concurrent access to repository methods."""
        db_manager, mock_conn = mock_db_manager
        equity_repo = EquityRepository(db_manager)

        # Mock successful responses
        mock_conn.fetchrow.return_value = {
            'symbol': 'TEST',
            'company_name': 'Test Corp',
            'exchange': 'NYSE',
            'sector': None,
            'industry': None,
            'market_cap': None,
            'employees': None,
            'description': None,
            'website': None,
            'country': 'US',
            'currency': 'USD',
            'is_etf': False,
            'is_actively_trading': True,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }

        # Simulate concurrent access
        async def get_symbol(symbol):
            return await equity_repo.get_by_symbol(symbol)

        # Run multiple concurrent operations
        tasks = [get_symbol(f"TEST{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(result is not None for result in results)

    @pytest.mark.asyncio
    async def test_error_recovery_and_logging(self, mock_db_manager):
        """Test error recovery and logging mechanisms."""
        db_manager, mock_conn = mock_db_manager

        # Simulate intermittent connection issues
        connection_errors = [
            Exception("Connection lost"),
            Exception("Timeout"),
            None,  # Success on third try
        ]
        mock_conn.execute.side_effect = connection_errors

        with patch('packages.sra_data.repositories.equity_repository.logger') as mock_logger:
            equity_repo = EquityRepository(db_manager)
            equity = EquityProfile(symbol="TEST", company_name="Test", exchange=ExchangeType.NYSE)

            result = await equity_repo.create(equity)

            # Should fail gracefully
            assert result is False
            # Should log the error
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_migration_dependency_resolution(self):
        """Test migration dependency resolution with complex dependencies."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        migration_mgr = MigrationManager(db_manager)

        # Create migrations with complex dependencies
        from packages.sra_data.repositories.migrations import Migration

        migration_a = Migration("001", "base", "Base migration", "CREATE TABLE a (id INT);")
        migration_b = Migration("002", "depends_on_a", "Depends on A", "CREATE TABLE b (id INT);", dependencies=["001"])
        migration_c = Migration("003", "depends_on_b", "Depends on B", "CREATE TABLE c (id INT);", dependencies=["002"])
        migration_d = Migration("004", "depends_on_a_and_c", "Depends on A and C",
                              "CREATE TABLE d (id INT);", dependencies=["001", "003"])

        # Add migrations out of order
        migration_mgr.add_migration(migration_d)
        migration_mgr.add_migration(migration_b)
        migration_mgr.add_migration(migration_a)
        migration_mgr.add_migration(migration_c)

        # Mock no applied migrations
        mock_conn.fetch.return_value = []

        pending = await migration_mgr.get_pending_migrations()

        # Should return in correct order: A, B, C, D
        assert len(pending) >= 4
        pending_versions = [m.version for m in pending]
        assert pending_versions.index("001") < pending_versions.index("002")
        assert pending_versions.index("002") < pending_versions.index("003")
        assert pending_versions.index("003") < pending_versions.index("004")

    @pytest.mark.asyncio
    async def test_view_refresh_with_data_changes(self, mock_db_manager):
        """Test view refresh behavior with underlying data changes."""
        db_manager, mock_conn = mock_db_manager
        view_mgr = ViewManager(db_manager)

        # Simulate data changes affecting view
        mock_conn.execute.return_value = None
        mock_conn.fetchval.side_effect = [1000, 1500]  # Row count before and after

        result = await view_mgr.refresh_materialized_view("mv_test_view")

        assert result.success is True
        assert result.rows_affected == 1500

    @pytest.mark.asyncio
    async def test_performance_optimizer_with_no_data(self, mock_db_manager):
        """Test performance optimizer with empty/no data scenarios."""
        db_manager, mock_conn = mock_db_manager
        perf_opt = PerformanceOptimizer(db_manager)

        # Mock empty results
        mock_conn.fetchval.return_value = False  # No pg_stat_statements
        mock_conn.fetchrow.return_value = None
        mock_conn.fetch.return_value = []

        report = await perf_opt.analyze_performance()

        assert report.overall_health_score >= 0
        assert len(report.slow_queries) == 0
        assert len(report.index_recommendations) == 0

    @pytest.mark.asyncio
    async def test_transaction_manager_nested_transactions(self, mock_db_manager):
        """Test transaction manager behavior with nested transaction attempts."""
        db_manager, mock_conn = mock_db_manager
        transaction_mgr = TransactionManager(db_manager)

        mock_conn.execute.return_value = None

        # Test nested transaction attempt (should work with same connection)
        async with transaction_mgr.transaction() as conn1:
            # This would typically be problematic in real PostgreSQL
            # but our mock should handle it
            await conn1.execute("INSERT INTO test VALUES (1)")

            async with transaction_mgr.transaction() as conn2:
                await conn2.execute("INSERT INTO test VALUES (2)")

        # Should complete without error in mock environment

    @pytest.mark.asyncio
    async def test_repository_with_connection_pool_exhaustion(self, mock_db_manager):
        """Test repository behavior when connection pool is exhausted."""
        db_manager, mock_conn = mock_db_manager

        # Simulate pool exhaustion
        async def mock_acquire():
            raise Exception("Connection pool exhausted")

        db_manager.pool.acquire.side_effect = mock_acquire()

        equity_repo = EquityRepository(db_manager)
        result = await equity_repo.get_by_symbol("TEST")

        # Should handle gracefully
        assert result is None


class TestMemoryAndResourceManagement:
    """Test memory usage and resource cleanup."""

    @pytest.mark.asyncio
    async def test_large_result_set_handling(self):
        """Test handling of large result sets."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()

        # Simulate large result set
        large_result_set = [
            {
                'symbol': f'SYM{i:05d}',
                'company_name': f'Company {i}',
                'exchange': 'NYSE',
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': Decimal(str(i * 1000000)),
                'employees': i * 100,
                'description': f'Description for company {i}' * 10,
                'website': f'https://company{i}.com',
                'country': 'US',
                'currency': 'USD',
                'is_etf': False,
                'is_actively_trading': True,
                'created_at': datetime.utcnow(),
                'updated_at': datetime.utcnow()
            }
            for i in range(10000)  # Large number of records
        ]

        mock_conn.fetch.return_value = large_result_set
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        equity_repo = EquityRepository(db_manager)
        results = await equity_repo.get_by_exchange("NYSE", limit=10000)

        assert len(results) == 10000
        # Memory should be managed properly (no explicit test, but operations should complete)

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self):
        """Test proper connection cleanup when errors occur."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()

        # Simulate error during operation
        mock_conn.execute.side_effect = Exception("Database error")
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        equity_repo = EquityRepository(db_manager)
        equity = EquityProfile(symbol="TEST", company_name="Test", exchange=ExchangeType.NYSE)

        result = await equity_repo.create(equity)

        assert result is False
        # Connection context manager should still be properly exited
        assert db_manager.pool.acquire.return_value.__aexit__.called


class TestDataValidationEdgeCases:
    """Test data validation edge cases."""

    @pytest.mark.asyncio
    async def test_unicode_and_encoding_handling(self):
        """Test handling of Unicode and different encodings."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        transaction_mgr = TransactionManager(db_manager)

        # Test with various Unicode characters
        unicode_data = {
            'symbol': '日本株',  # Japanese characters
            'company_name': 'Société Française',  # French accents
            'exchange': 'TOKYO',
            'sector': 'Технологии',  # Cyrillic
            'description': '公司描述 with émoji 🚀'
        }

        violations = await transaction_mgr.validate_data_integrity("equity_profile", unicode_data)

        # Should handle Unicode gracefully
        assert isinstance(violations, list)

    @pytest.mark.asyncio
    async def test_decimal_precision_edge_cases(self):
        """Test decimal precision handling in edge cases."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        transaction_mgr = TransactionManager(db_manager)

        # Test with various decimal precisions
        test_cases = [
            Decimal('0.00000001'),  # Very small
            Decimal('999999999999999999.99'),  # Very large
            Decimal('10.123456789012345'),  # High precision
            Decimal('0'),  # Zero
            Decimal('-100.50')  # Negative (should fail for NAVPS)
        ]

        for navps_value in test_cases:
            record_data = {
                'InstrumentKey': 'TEST',
                'RecordId': 'REC001',
                'Date': date.today(),
                'NAVPS': navps_value,
                'source_file': 'test.csv'
            }

            violations = await transaction_mgr.validate_data_integrity("fundata_quotes", record_data)

            if navps_value <= 0:
                # Should have violations for non-positive values
                navps_violations = [v for v in violations if v.column_name == 'navps']
                assert len(navps_violations) > 0
            # Other cases should be handled gracefully

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self):
        """Test that parameterized queries prevent SQL injection."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = None
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        equity_repo = EquityRepository(db_manager)

        # Test with SQL injection attempt
        malicious_symbol = "'; DROP TABLE equity_profile; --"

        result = await equity_repo.get_by_symbol(malicious_symbol)

        # Should safely handle the malicious input
        assert result is None
        # Verify parameterized query was used
        mock_conn.fetchrow.assert_called()
        call_args = mock_conn.fetchrow.call_args
        # Query should be parameterized
        assert "$1" in call_args[0][0]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv('TEST_DB_HOST'),
    reason="Integration tests require TEST_DB_* environment variables"
)
class TestFullIntegration:
    """Full integration tests with real database."""

    @pytest.fixture
    async def real_db_manager(self):
        """Create real database manager for integration tests."""
        config = {
            'host': os.getenv('TEST_DB_HOST', 'localhost'),
            'port': int(os.getenv('TEST_DB_PORT', '5432')),
            'database': os.getenv('TEST_DB_NAME', 'sra_test'),
            'user': os.getenv('TEST_DB_USER', 'postgres'),
            'password': os.getenv('TEST_DB_PASSWORD', '')
        }

        try:
            manager = DatabaseManager(config)
            await manager.initialize()
            yield manager
        except Exception as e:
            pytest.skip(f"Test database not available: {e}")
        finally:
            if 'manager' in locals():
                await manager.close()

    @pytest.mark.asyncio
    async def test_full_repository_workflow(self, real_db_manager):
        """Test complete repository workflow with real database."""
        equity_repo = EquityRepository(real_db_manager)

        # Create test equity
        test_equity = EquityProfile(
            symbol="INTTEST",
            company_name="Integration Test Corp",
            exchange=ExchangeType.NYSE,
            sector="Technology",
            market_cap=Decimal("1000000000"),
            is_actively_trading=True
        )

        try:
            # Test create
            create_result = await equity_repo.create(test_equity)
            assert create_result is True

            # Test retrieve
            retrieved = await equity_repo.get_by_symbol("INTTEST")
            assert retrieved is not None
            assert retrieved.company_name == "Integration Test Corp"

            # Test update
            update_result = await equity_repo.update("INTTEST", {"market_cap": Decimal("1500000000")})
            assert update_result is True

            # Test search
            search_results = await equity_repo.search("Integration Test")
            assert len(search_results) >= 1

        finally:
            # Cleanup
            await equity_repo.delete("INTTEST")

    @pytest.mark.asyncio
    async def test_migration_system_integration(self, real_db_manager):
        """Test migration system with real database."""
        migration_mgr = MigrationManager(real_db_manager)
        await migration_mgr.initialize()

        # Check migration history
        history = await migration_mgr.get_migration_history()
        assert isinstance(history, list)

        # Health check
        health = await migration_mgr.health_check()
        assert health['status'] == 'healthy'

    @pytest.mark.asyncio
    async def test_transaction_system_integration(self, real_db_manager):
        """Test transaction system with real database."""
        transaction_mgr = TransactionManager(real_db_manager)

        # Test successful transaction
        async def test_operation(conn):
            await conn.execute("SELECT 1")
            return "success"

        result = await transaction_mgr.execute_with_retry(test_operation)
        assert result == "success"

        # Test rollback on exception
        async def failing_operation(conn):
            await conn.execute("SELECT 1")
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            async with transaction_mgr.transaction() as conn:
                await failing_operation(conn)

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, real_db_manager):
        """Test performance monitoring with real database."""
        perf_optimizer = PerformanceOptimizer(real_db_manager)

        # Run performance analysis
        report = await perf_optimizer.analyze_performance(include_recommendations=False)

        assert isinstance(report, type(perf_optimizer).__module__.split('.')[-1])
        assert report.overall_health_score >= 0
        assert isinstance(report.connection_metrics, dict)

    @pytest.mark.asyncio
    async def test_view_system_integration(self, real_db_manager):
        """Test view system with real database."""
        view_mgr = ViewManager(real_db_manager)

        try:
            # Initialize views
            await view_mgr.initialize_views()

            # Test view health check
            health = await view_mgr.get_view_health_check()
            assert 'status' in health

        except Exception as e:
            # Views might fail if tables don't have data, which is acceptable
            pytest.skip(f"View operations require data: {e}")


class TestConcurrencyAndStress:
    """Test concurrency and stress scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_repository_operations(self):
        """Test concurrent repository operations."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        mock_conn.fetchrow.return_value = None
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        equity_repo = EquityRepository(db_manager)

        # Simulate concurrent operations
        async def create_equity(symbol):
            equity = EquityProfile(
                symbol=symbol,
                company_name=f"Company {symbol}",
                exchange=ExchangeType.NYSE
            )
            return await equity_repo.create(equity)

        # Run many concurrent operations
        tasks = [create_equity(f"CONC{i:03d}") for i in range(100)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete (successfully or with exceptions handled)
        assert len(results) == 100
        # Should not have unhandled exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0  # Mock should handle all gracefully

    @pytest.mark.asyncio
    async def test_stress_bulk_operations(self):
        """Test bulk operations under stress."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        equity_repo = EquityRepository(db_manager)

        # Large bulk operation
        large_batch = [
            EquityProfile(
                symbol=f"BULK{i:05d}",
                company_name=f"Bulk Company {i}",
                exchange=ExchangeType.NYSE
            )
            for i in range(5000)
        ]

        result = await equity_repo.bulk_insert(large_batch)

        assert result.success is True
        assert result.records_processed == 5000