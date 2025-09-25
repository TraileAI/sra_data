"""Tests for advanced repository features.

Test coverage for migrations, table management, views, performance optimization,
and transaction management components.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date
from decimal import Decimal

from packages.sra_data.repositories.migrations import (
    MigrationManager,
    Migration,
    MigrationStatus,
    MigrationResult,
    create_migration_manager
)
from packages.sra_data.repositories.table_manager import (
    TableManager,
    PartitionType,
    TableMaintenanceResult
)
from packages.sra_data.repositories.view_manager import (
    ViewManager,
    ViewType,
    ViewRefreshResult
)
from packages.sra_data.repositories.performance_optimizer import (
    PerformanceOptimizer,
    OptimizationLevel,
    QueryPerformanceMetric,
    IndexRecommendation,
    PerformanceReport
)
from packages.sra_data.repositories.transaction_manager import (
    TransactionManager,
    TransactionIsolationLevel,
    IntegrityViolationType,
    IntegrityViolation,
    TransactionResult
)


class TestMigrationManager:
    """Test migration system functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.fixture
    def migration_manager(self, mock_db_manager):
        """Create migration manager for testing."""
        db_manager, _ = mock_db_manager
        return MigrationManager(db_manager)

    @pytest.fixture
    def sample_migration(self):
        """Create sample migration."""
        return Migration(
            version="001",
            name="test_migration",
            description="Test migration for unit tests",
            up_sql="CREATE TABLE test (id SERIAL PRIMARY KEY);",
            down_sql="DROP TABLE IF EXISTS test;"
        )

    @pytest.mark.asyncio
    async def test_initialize(self, migration_manager, mock_db_manager):
        """Test migration manager initialization."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        await migration_manager.initialize()

        # Should create migrations table and load built-in migrations
        assert mock_conn.execute.call_count >= 1
        assert len(migration_manager._migrations) >= 3  # Built-in migrations

    def test_add_migration_success(self, migration_manager, sample_migration):
        """Test adding a migration successfully."""
        migration_manager.add_migration(sample_migration)

        assert sample_migration.version in migration_manager._migrations
        assert migration_manager._migrations[sample_migration.version] == sample_migration

    def test_add_migration_duplicate_version(self, migration_manager, sample_migration):
        """Test adding migration with duplicate version."""
        migration_manager.add_migration(sample_migration)

        with pytest.raises(ValueError, match="Migration version .* already exists"):
            migration_manager.add_migration(sample_migration)

    @pytest.mark.asyncio
    async def test_get_pending_migrations(self, migration_manager, mock_db_manager):
        """Test getting pending migrations."""
        _, mock_conn = mock_db_manager
        # Mock no applied migrations
        mock_conn.fetch.return_value = []

        # Add test migrations
        migration1 = Migration("001", "first", "First migration", "SELECT 1;")
        migration2 = Migration("002", "second", "Second migration", "SELECT 2;", dependencies=["001"])
        migration_manager.add_migration(migration1)
        migration_manager.add_migration(migration2)

        pending = await migration_manager.get_pending_migrations()

        # Should return migration 1 first (no dependencies)
        assert len(pending) >= 1
        assert pending[0].version == "001"

    @pytest.mark.asyncio
    async def test_apply_migration_success(self, migration_manager, sample_migration, mock_db_manager):
        """Test successful migration application."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        result = await migration_manager.apply_migration(sample_migration)

        assert isinstance(result, MigrationResult)
        assert result.success is True
        assert result.status == MigrationStatus.COMPLETED
        assert result.version == sample_migration.version
        assert mock_conn.execute.call_count >= 2  # Migration SQL + status update

    @pytest.mark.asyncio
    async def test_apply_migration_failure(self, migration_manager, sample_migration, mock_db_manager):
        """Test migration application failure."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.side_effect = [None, Exception("SQL error")]  # Status update succeeds, migration fails

        result = await migration_manager.apply_migration(sample_migration)

        assert result.success is False
        assert result.status == MigrationStatus.FAILED
        assert "SQL error" in result.error_message

    @pytest.mark.asyncio
    async def test_rollback_migration_success(self, migration_manager, sample_migration, mock_db_manager):
        """Test successful migration rollback."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        result = await migration_manager.rollback_migration(sample_migration.version)

        assert result.success is True
        assert result.status == MigrationStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_rollback_migration_no_down_sql(self, migration_manager, mock_db_manager):
        """Test rollback with no down SQL."""
        migration = Migration("test", "test", "Test", "SELECT 1;")  # No down_sql
        migration_manager.add_migration(migration)

        with pytest.raises(ValueError, match="has no rollback SQL"):
            await migration_manager.rollback_migration("test")

    @pytest.mark.asyncio
    async def test_migrate_up(self, migration_manager, mock_db_manager):
        """Test applying all pending migrations."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.fetch.return_value = []  # No applied migrations

        # Add test migration
        migration = Migration("test", "test", "Test", "SELECT 1;")
        migration_manager.add_migration(migration)

        results = await migration_manager.migrate_up()

        assert len(results) >= 1
        assert all(isinstance(result, MigrationResult) for result in results)

    @pytest.mark.asyncio
    async def test_get_migration_history(self, migration_manager, mock_db_manager):
        """Test getting migration history."""
        _, mock_conn = mock_db_manager
        mock_rows = [
            {
                'version': '001',
                'name': 'initial_schema',
                'status': 'completed',
                'applied_at': datetime.utcnow()
            }
        ]
        mock_conn.fetch.return_value = mock_rows

        history = await migration_manager.get_migration_history()

        assert len(history) == 1
        assert history[0]['version'] == '001'

    @pytest.mark.asyncio
    async def test_health_check(self, migration_manager, mock_db_manager):
        """Test migration system health check."""
        _, mock_conn = mock_db_manager
        mock_conn.fetch.return_value = []

        # Add test migration
        migration_manager.add_migration(Migration("test", "test", "Test", "SELECT 1;"))

        health = await migration_manager.health_check()

        assert health['status'] == 'healthy'
        assert 'total_migrations' in health
        assert 'pending_migrations' in health


class TestTableManager:
    """Test table management functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.fixture
    def table_manager(self, mock_db_manager):
        """Create table manager for testing."""
        db_manager, _ = mock_db_manager
        return TableManager(db_manager)

    @pytest.mark.asyncio
    async def test_get_table_info_exists(self, table_manager, mock_db_manager):
        """Test getting table info when table exists."""
        _, mock_conn = mock_db_manager

        # Mock table info queries
        mock_conn.fetchrow.side_effect = [
            {'schemaname': 'public', 'tablename': 'test_table', 'tableowner': 'postgres',
             'hasindexes': True, 'hasrules': False, 'hastriggers': False},  # Basic info
            {'total_size': '1 MB', 'table_size': '800 kB', 'indexes_size': '200 kB'},  # Size info
            1000  # Row count
        ]
        mock_conn.fetchval.return_value = 1000
        mock_conn.fetch.side_effect = [[], []]  # Columns, indexes, constraints

        result = await table_manager.get_table_info("test_table")

        assert result['exists'] is True
        assert result['basic_info']['tablename'] == 'test_table'
        assert result['estimated_rows'] == 1000

    @pytest.mark.asyncio
    async def test_get_table_info_not_exists(self, table_manager, mock_db_manager):
        """Test getting table info when table doesn't exist."""
        _, mock_conn = mock_db_manager
        mock_conn.fetchrow.return_value = None

        result = await table_manager.get_table_info("nonexistent_table")

        assert result['exists'] is False

    @pytest.mark.asyncio
    async def test_analyze_table_success(self, table_manager, mock_db_manager):
        """Test successful table analysis."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.fetchrow.side_effect = [
            {'reltuples': 900, 'relpages': 10, 'last_analyze': None, 'last_autoanalyze': None},  # Pre-stats
            {'reltuples': 1000, 'relpages': 12, 'last_analyze': datetime.utcnow(), 'last_autoanalyze': None}  # Post-stats
        ]

        result = await table_manager.analyze_table("test_table")

        assert isinstance(result, TableMaintenanceResult)
        assert result.success is True
        assert result.operation == "ANALYZE"
        assert 'rows_analyzed' in result.details

    @pytest.mark.asyncio
    async def test_vacuum_table_success(self, table_manager, mock_db_manager):
        """Test successful table vacuum."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.fetchval.side_effect = [1000000, 900000]  # Before and after size

        result = await table_manager.vacuum_table("test_table", full=False)

        assert result.success is True
        assert result.operation == "VACUUM"
        assert result.details['space_reclaimed_bytes'] == 100000

    @pytest.mark.asyncio
    async def test_reindex_table_concurrently(self, table_manager, mock_db_manager):
        """Test concurrent table reindexing."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.fetch.return_value = [
            {'indexname': 'idx_test_1', 'index_size': '100 kB'},
            {'indexname': 'idx_test_2', 'index_size': '200 kB'}
        ]

        result = await table_manager.reindex_table("test_table", concurrently=True)

        assert result.success is True
        assert result.operation == "REINDEX"
        assert result.details['total_indexes'] == 2

    @pytest.mark.asyncio
    async def test_get_table_statistics(self, table_manager, mock_db_manager):
        """Test getting comprehensive table statistics."""
        _, mock_conn = mock_db_manager
        mock_stats = {
            'seq_scan': 100, 'seq_tup_read': 10000, 'idx_scan': 1000, 'idx_tup_fetch': 50000,
            'n_tup_ins': 1000, 'n_live_tup': 900, 'n_dead_tup': 100
        }
        mock_io_stats = {
            'heap_blks_read': 100, 'heap_blks_hit': 9000, 'idx_blks_read': 50, 'idx_blks_hit': 4500
        }
        mock_size_stats = {
            'total_size': '1 MB', 'table_size': '800 kB', 'total_size_bytes': 1048576, 'table_size_bytes': 819200
        }

        mock_conn.fetchrow.side_effect = [mock_stats, mock_io_stats, mock_size_stats]
        mock_conn.fetch.return_value = []

        result = await table_manager.get_table_statistics("test_table")

        assert 'basic_stats' in result
        assert 'cache_hit_ratio_percent' in result
        assert result['cache_hit_ratio_percent'] > 95  # High hit ratio

    @pytest.mark.asyncio
    async def test_create_partition(self, table_manager, mock_db_manager):
        """Test partition creation."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.fetch.return_value = []  # No indexes to inherit

        result = await table_manager.create_partition(
            "parent_table", "partition_table", PartitionType.RANGE,
            "date", "FROM ('2023-01-01') TO ('2023-02-01')"
        )

        assert result.success is True
        assert result.operation == "CREATE_PARTITION"
        assert result.details['parent_table'] == "parent_table"

    @pytest.mark.asyncio
    async def test_optimize_table_for_bulk_operations(self, table_manager, mock_db_manager):
        """Test table optimization for bulk operations."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        result = await table_manager.optimize_table_for_bulk_operations("test_table")

        assert result.success is True
        assert result.operation == "BULK_OPTIMIZE"
        assert 'storage_optimizations' in result.details

    @pytest.mark.asyncio
    async def test_maintenance_health_check(self, table_manager, mock_db_manager):
        """Test maintenance health check."""
        _, mock_conn = mock_db_manager
        mock_conn.fetch.side_effect = [[], [], []]  # No tables needing maintenance, no unused indexes

        health = await table_manager.maintenance_health_check()

        assert health['status'] == 'healthy'
        assert 'recommendations' in health


class TestViewManager:
    """Test view management functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.fixture
    def view_manager(self, mock_db_manager):
        """Create view manager for testing."""
        db_manager, _ = mock_db_manager
        return ViewManager(db_manager)

    @pytest.mark.asyncio
    async def test_initialize_views(self, view_manager, mock_db_manager):
        """Test view initialization."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        await view_manager.initialize_views()

        # Should create multiple views
        assert mock_conn.execute.call_count >= 6  # All predefined views

    @pytest.mark.asyncio
    async def test_refresh_materialized_view_success(self, view_manager, mock_db_manager):
        """Test successful materialized view refresh."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.fetchval.side_effect = [1000, 1100]  # Before and after row counts

        result = await view_manager.refresh_materialized_view("mv_test_view")

        assert isinstance(result, ViewRefreshResult)
        assert result.success is True
        assert result.rows_affected == 1100

    @pytest.mark.asyncio
    async def test_refresh_materialized_view_failure(self, view_manager, mock_db_manager):
        """Test materialized view refresh failure."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.side_effect = Exception("View refresh failed")

        result = await view_manager.refresh_materialized_view("mv_test_view")

        assert result.success is False
        assert "View refresh failed" in result.error_message

    @pytest.mark.asyncio
    async def test_refresh_all_materialized_views(self, view_manager, mock_db_manager):
        """Test refreshing all materialized views."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.fetchval.return_value = 1000

        results = await view_manager.refresh_all_materialized_views()

        assert len(results) >= 3  # At least 3 materialized views
        assert all(isinstance(result, ViewRefreshResult) for result in results)

    @pytest.mark.asyncio
    async def test_get_view_info_exists(self, view_manager, mock_db_manager):
        """Test getting view info when view exists."""
        _, mock_conn = mock_db_manager
        mock_view_info = {
            'schemaname': 'public',
            'viewname': 'test_view',
            'viewowner': 'postgres',
            'definition': 'SELECT * FROM test_table'
        }
        mock_conn.fetchrow.side_effect = [mock_view_info, None]  # View info, no size info (not materialized)
        mock_conn.fetchval.side_effect = [1000, False]  # Row count, not materialized

        result = await view_manager.get_view_info("test_view")

        assert result['exists'] is True
        assert result['is_materialized'] is False
        assert result['row_count'] == 1000

    @pytest.mark.asyncio
    async def test_search_unified_success(self, view_manager, mock_db_manager):
        """Test unified search functionality."""
        _, mock_conn = mock_db_manager
        mock_results = [
            {
                'record_type': 'equity',
                'primary_key': 'AAPL',
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'exchange': 'NASDAQ',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap': Decimal('3000000000000'),
                'is_etf': False,
                'country': 'US',
                'base_rank': 1.0,
                'relevance_score': 4.0,
                'last_updated': datetime.utcnow()
            }
        ]
        mock_conn.fetch.return_value = mock_results

        results = await view_manager.search_unified("Apple", limit=10)

        assert len(results) == 1
        assert results[0]['name'] == 'Apple Inc.'
        assert results[0]['relevance_score'] == 4.0

    @pytest.mark.asyncio
    async def test_search_unified_empty_query(self, view_manager, mock_db_manager):
        """Test unified search with empty query."""
        results = await view_manager.search_unified("", limit=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_get_view_health_check(self, view_manager, mock_db_manager):
        """Test view health check."""
        _, mock_conn = mock_db_manager

        # Mock view info for each view
        mock_view_responses = [
            {'exists': True, 'row_count': 1000, 'is_materialized': False},
            {'exists': True, 'row_count': 500, 'is_materialized': True},
            {'exists': False, 'error': 'View does not exist'}
        ]

        with patch.object(view_manager, 'get_view_info', side_effect=mock_view_responses):
            health = await view_manager.get_view_health_check()

            assert 'status' in health
            assert 'view_details' in health
            assert health['total_views'] >= 6


class TestPerformanceOptimizer:
    """Test performance optimization functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.fixture
    def performance_optimizer(self, mock_db_manager):
        """Create performance optimizer for testing."""
        db_manager, _ = mock_db_manager
        return PerformanceOptimizer(db_manager)

    @pytest.mark.asyncio
    async def test_analyze_performance_complete(self, performance_optimizer, mock_db_manager):
        """Test complete performance analysis."""
        _, mock_conn = mock_db_manager

        # Mock pg_stat_statements availability
        mock_conn.fetchval.side_effect = [True, 1000, 1000, 1000]  # Extension exists, row counts
        mock_conn.fetchrow.side_effect = [
            # Connection stats
            {'numbackends': 5, 'xact_commit': 10000, 'xact_rollback': 100,
             'blks_read': 1000, 'blks_hit': 90000, 'tup_returned': 100000,
             'tup_fetched': 50000, 'tup_inserted': 1000, 'tup_updated': 500, 'tup_deleted': 100},
            # Table stats (multiple calls)
            {'n_tup_ins': 1000, 'n_tup_upd': 500, 'n_tup_del': 100, 'n_live_tup': 10000, 'n_dead_tup': 500,
             'seq_scan': 100, 'seq_tup_read': 10000, 'idx_scan': 1000, 'idx_tup_fetch': 50000,
             'last_vacuum': datetime.utcnow(), 'last_autovacuum': None,
             'last_analyze': datetime.utcnow(), 'last_autoanalyze': None},
            {'heap_blks_read': 100, 'heap_blks_hit': 9000, 'idx_blks_read': 50, 'idx_blks_hit': 4500,
             'toast_blks_read': 0, 'toast_blks_hit': 0, 'tidx_blks_read': 0, 'tidx_blks_hit': 0},
            {'total_size': '1 MB', 'table_size': '800 kB', 'total_size_bytes': 1048576, 'table_size_bytes': 819200}
        ]
        mock_conn.fetch.side_effect = [
            # Slow queries
            [{'query_hash': '123', 'query_text': 'SELECT * FROM test', 'calls': 100, 'total_time': 5000,
              'mean_time': 50, 'min_time': 10, 'max_time': 200, 'stddev_time': 30, 'rows_returned': 1000,
              'cache_hit_ratio': 95.0}],
            # Server config
            [{'name': 'max_connections', 'setting': '100', 'unit': None}],
            # Missing FK indexes, frequent columns, etc.
            [], [], []
        ]

        report = await performance_optimizer.analyze_performance()

        assert isinstance(report, PerformanceReport)
        assert report.overall_health_score > 0
        assert len(report.slow_queries) >= 0
        assert 'cache_hit_ratio' in report.connection_metrics

    @pytest.mark.asyncio
    async def test_analyze_performance_no_pg_stat_statements(self, performance_optimizer, mock_db_manager):
        """Test performance analysis when pg_stat_statements is not available."""
        _, mock_conn = mock_db_manager
        mock_conn.fetchval.return_value = False  # Extension not available

        # Mock other required data
        mock_conn.fetchrow.side_effect = [
            {'numbackends': 5, 'xact_commit': 10000, 'xact_rollback': 100,
             'blks_read': 1000, 'blks_hit': 90000, 'tup_returned': 100000,
             'tup_fetched': 50000, 'tup_inserted': 1000, 'tup_updated': 500, 'tup_deleted': 100},
        ]
        mock_conn.fetch.side_effect = [[], []]

        with patch.object(performance_optimizer, '_analyze_table_performance', return_value={}), \
             patch.object(performance_optimizer, '_generate_index_recommendations', return_value=[]):

            report = await performance_optimizer.analyze_performance()

            assert isinstance(report, PerformanceReport)
            # Should still work without pg_stat_statements

    @pytest.mark.asyncio
    async def test_optimize_automatically_basic(self, performance_optimizer, mock_db_manager):
        """Test basic automatic optimization."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        # Mock performance report
        mock_report = PerformanceReport()
        mock_report.table_metrics = {
            'test_table': {'needs_analyze': True, 'needs_vacuum': False}
        }
        mock_report.index_recommendations = [
            IndexRecommendation(
                table_name="test_table",
                columns=["test_column"],
                estimated_benefit="high",
                create_sql="CREATE INDEX idx_test ON test_table(test_column);"
            )
        ]

        with patch.object(performance_optimizer, 'analyze_performance', return_value=mock_report):
            result = await performance_optimizer.optimize_automatically(OptimizationLevel.BASIC)

            assert 'operations' in result
            assert 'errors' in result
            assert len(result['operations']) >= 1  # Should analyze the table

    @pytest.mark.asyncio
    async def test_optimize_automatically_aggressive(self, performance_optimizer, mock_db_manager):
        """Test aggressive automatic optimization."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        mock_report = PerformanceReport()
        mock_report.table_metrics = {
            'test_table': {'needs_analyze': True, 'needs_vacuum': True}
        }
        mock_report.index_recommendations = [
            IndexRecommendation(
                table_name="test_table",
                columns=["test_column"],
                estimated_benefit="high",
                create_sql="CREATE INDEX idx_test ON test_table(test_column);"
            )
        ]

        with patch.object(performance_optimizer, 'analyze_performance', return_value=mock_report):
            result = await performance_optimizer.optimize_automatically(OptimizationLevel.AGGRESSIVE)

            assert len(result['operations']) >= 2  # Should analyze and vacuum


class TestTransactionManager:
    """Test transaction management functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.fixture
    def transaction_manager(self, mock_db_manager):
        """Create transaction manager for testing."""
        db_manager, _ = mock_db_manager
        return TransactionManager(db_manager)

    @pytest.mark.asyncio
    async def test_transaction_success(self, transaction_manager, mock_db_manager):
        """Test successful transaction execution."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        async with transaction_manager.transaction() as conn:
            await conn.execute("INSERT INTO test VALUES (1)")

        # Should call BEGIN, INSERT, and COMMIT
        calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert any("BEGIN" in call for call in calls)
        assert any("COMMIT" in call for call in calls)

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_exception(self, transaction_manager, mock_db_manager):
        """Test transaction rollback on exception."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        with pytest.raises(ValueError, match="Test error"):
            async with transaction_manager.transaction() as conn:
                await conn.execute("INSERT INTO test VALUES (1)")
                raise ValueError("Test error")

        # Should call BEGIN, INSERT, and ROLLBACK
        calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert any("BEGIN" in call for call in calls)
        assert any("ROLLBACK" in call for call in calls)

    @pytest.mark.asyncio
    async def test_transaction_readonly(self, transaction_manager, mock_db_manager):
        """Test readonly transaction."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        async with transaction_manager.transaction(readonly=True) as conn:
            await conn.fetchval("SELECT COUNT(*) FROM test")

        # Should call BEGIN READ ONLY
        calls = [call[0][0] for call in mock_conn.execute.call_args_list]
        assert any("BEGIN READ ONLY" in call for call in calls)

    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, transaction_manager, mock_db_manager):
        """Test retry mechanism with successful operation."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        async def test_operation(conn):
            await conn.execute("INSERT INTO test VALUES (1)")
            return "success"

        result = await transaction_manager.execute_with_retry(test_operation)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_execute_with_retry_transient_error(self, transaction_manager, mock_db_manager):
        """Test retry mechanism with transient errors."""
        _, mock_conn = mock_db_manager

        # Mock serialization error on first attempt, success on second
        attempt_count = 0

        async def test_operation(conn):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                from packages.sra_data.repositories.transaction_manager import asyncpg
                raise asyncpg.SerializationError("Serialization failure")
            return "success"

        with patch('packages.sra_data.repositories.transaction_manager.asyncpg') as mock_asyncpg:
            mock_asyncpg.SerializationError = Exception
            mock_asyncpg.DeadlockDetectedError = Exception

            # Mock the transaction context manager
            with patch.object(transaction_manager, 'transaction') as mock_transaction:
                mock_transaction.return_value.__aenter__.return_value = mock_conn
                mock_transaction.return_value.__aexit__.return_value = None

                result = await transaction_manager.execute_with_retry(test_operation, max_retries=1)

                assert result == "success"
                assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_validate_data_integrity_equity_profile(self, transaction_manager, mock_db_manager):
        """Test data integrity validation for equity profile."""
        _, mock_conn = mock_db_manager
        mock_conn.fetchval.return_value = None  # No existing symbol

        record_data = {
            'symbol': 'AAPL',
            'company_name': 'Apple Inc.',
            'exchange': 'NASDAQ',
            'market_cap': 3000000000000
        }

        violations = await transaction_manager.validate_data_integrity("equity_profile", record_data)

        # Should pass validation with valid data
        assert len(violations) == 0

    @pytest.mark.asyncio
    async def test_validate_data_integrity_missing_required_fields(self, transaction_manager, mock_db_manager):
        """Test validation with missing required fields."""
        _, mock_conn = mock_db_manager

        record_data = {
            'symbol': '',  # Empty symbol
            'company_name': 'Test Company'
            # Missing exchange
        }

        violations = await transaction_manager.validate_data_integrity("equity_profile", record_data)

        assert len(violations) >= 2  # Missing symbol and exchange
        assert any(v.violation_type == IntegrityViolationType.NOT_NULL for v in violations)

    @pytest.mark.asyncio
    async def test_validate_data_integrity_duplicate_key(self, transaction_manager, mock_db_manager):
        """Test validation with duplicate key."""
        _, mock_conn = mock_db_manager
        mock_conn.fetchval.return_value = "AAPL"  # Symbol already exists

        record_data = {
            'symbol': 'AAPL',
            'company_name': 'Apple Inc.',
            'exchange': 'NASDAQ'
        }

        violations = await transaction_manager.validate_data_integrity("equity_profile", record_data)

        assert len(violations) >= 1
        duplicate_violations = [v for v in violations if v.violation_type == IntegrityViolationType.DUPLICATE_KEY]
        assert len(duplicate_violations) == 1

    @pytest.mark.asyncio
    async def test_batch_operation_with_integrity_success(self, transaction_manager, mock_db_manager):
        """Test successful batch operation with integrity checking."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = "INSERT 0 1"
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

        async def operation1(conn):
            return await conn.execute("INSERT INTO test VALUES (1)")

        async def operation2(conn):
            return await conn.execute("INSERT INTO test VALUES (2)")

        result = await transaction_manager.batch_operation_with_integrity([operation1, operation2])

        assert isinstance(result, TransactionResult)
        assert result.success is True
        assert result.operations_count == 2

    @pytest.mark.asyncio
    async def test_get_constraint_violations_report(self, transaction_manager, mock_db_manager):
        """Test getting constraint violations report."""
        _, mock_conn = mock_db_manager
        mock_constraints = [
            {
                'table_name': 'equity_profile',
                'constraint_name': 'equity_profile_pkey',
                'constraint_type': 'PRIMARY KEY',
                'definition': 'PRIMARY KEY (symbol)'
            }
        ]
        mock_fk_relationships = [
            {
                'source_table': 'child_table',
                'source_column': 'parent_id',
                'target_table': 'parent_table',
                'target_column': 'id',
                'constraint_name': 'fk_child_parent'
            }
        ]
        mock_conn.fetch.side_effect = [mock_constraints, mock_fk_relationships]

        report = await transaction_manager.get_constraint_violations_report()

        assert 'constraints_summary' in report
        assert 'constraints_detail' in report
        assert 'foreign_key_relationships' in report

    @pytest.mark.asyncio
    async def test_health_check(self, transaction_manager, mock_db_manager):
        """Test transaction manager health check."""
        health = await transaction_manager.health_check()

        assert health['status'] == 'healthy'
        assert 'features_available' in health
        assert health['features_available']['transaction_isolation'] is True


@pytest.mark.asyncio
async def test_create_migration_manager_factory(mock_db_manager):
    """Test migration manager factory function."""
    db_manager, mock_conn = mock_db_manager
    mock_conn.execute.return_value = None

    with patch('packages.sra_data.repositories.migrations.MigrationManager') as mock_manager_class:
        mock_manager = AsyncMock()
        mock_manager_class.return_value = mock_manager

        result = await create_migration_manager(db_manager)

        mock_manager_class.assert_called_once_with(db_manager)
        mock_manager.initialize.assert_called_once()
        assert result == mock_manager