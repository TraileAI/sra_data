"""Comprehensive tests for database infrastructure.

Test coverage for DatabaseManager, ConnectionPool, SchemaManager with
both unit tests and integration tests including error conditions.
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from packages.sra_data.repositories.database import (
    DatabaseConfig,
    ConnectionPool,
    SchemaManager,
    DatabaseManager,
    create_database_manager,
    ASYNCPG_AVAILABLE
)


class TestDatabaseConfig:
    """Test database configuration management."""

    def test_init_with_config(self):
        """Test initialization with provided configuration."""
        config_dict = {
            'host': 'test-host',
            'port': 5433,
            'database': 'test-db',
            'user': 'test-user',
            'password': 'test-pass'
        }
        config = DatabaseConfig(config_dict)
        assert config.config['host'] == 'test-host'
        assert config.config['port'] == 5433

    def test_init_from_environment(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'DB_HOST': 'env-host',
            'DB_PORT': '5434',
            'DB_NAME': 'env-db',
            'DB_USER': 'env-user',
            'DB_PASSWORD': 'env-pass'
        }):
            config = DatabaseConfig()
            assert config.config['host'] == 'env-host'
            assert config.config['port'] == 5434
            assert config.config['database'] == 'env-db'

    def test_validate_config_success(self):
        """Test successful configuration validation."""
        valid_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'user': 'test'
        }
        config = DatabaseConfig(valid_config)
        # Should not raise exception
        assert config.config is not None

    def test_validate_config_missing_fields(self):
        """Test validation with missing required fields."""
        invalid_config = {'host': 'localhost'}
        with pytest.raises(ValueError, match="Missing required database config fields"):
            DatabaseConfig(invalid_config)

    def test_validate_config_invalid_port(self):
        """Test validation with invalid port."""
        invalid_config = {
            'host': 'localhost',
            'port': -1,
            'database': 'test',
            'user': 'test'
        }
        with pytest.raises(ValueError, match="Invalid port number"):
            DatabaseConfig(invalid_config)

    def test_validate_config_invalid_pool_size(self):
        """Test validation with invalid pool sizes."""
        invalid_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'user': 'test',
            'min_size': -1
        }
        with pytest.raises(ValueError, match="min_size cannot be negative"):
            DatabaseConfig(invalid_config)

    def test_dsn_property(self):
        """Test DSN string generation."""
        config_dict = {
            'host': 'test-host',
            'port': 5432,
            'database': 'test-db',
            'user': 'test-user',
            'password': 'secret'
        }
        config = DatabaseConfig(config_dict)
        expected_dsn = "postgresql://test-user:secret@test-host:5432/test-db"
        assert config.dsn == expected_dsn


class TestConnectionPool:
    """Test connection pool management."""

    @pytest.fixture
    def config(self):
        """Create test database configuration."""
        return DatabaseConfig({
            'host': 'localhost',
            'port': 5432,
            'database': 'test',
            'user': 'test',
            'password': 'test'
        })

    @pytest.fixture
    def connection_pool(self, config):
        """Create connection pool for testing."""
        return ConnectionPool(config)

    @pytest.mark.asyncio
    async def test_initialize_success(self, connection_pool):
        """Test successful pool initialization."""
        if not ASYNCPG_AVAILABLE:
            # Pool should handle missing asyncpg gracefully
            await connection_pool.initialize()
            assert connection_pool._pool is None
        else:
            with patch('packages.sra_data.repositories.database.asyncpg.create_pool') as mock_create:
                mock_pool = AsyncMock()
                mock_create.return_value = mock_pool

                await connection_pool.initialize()
                assert connection_pool._pool == mock_pool
                mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, connection_pool):
        """Test pool initialization failure handling."""
        if ASYNCPG_AVAILABLE:
            with patch('packages.sra_data.repositories.database.asyncpg.create_pool') as mock_create:
                mock_create.side_effect = Exception("Connection failed")

                with pytest.raises(Exception, match="Connection failed"):
                    await connection_pool.initialize()

    @pytest.mark.asyncio
    async def test_acquire_context_manager(self, connection_pool):
        """Test connection acquisition through context manager."""
        await connection_pool.initialize()

        async with connection_pool.acquire() as conn:
            assert conn is not None
            # Mock connection should be returned

    @pytest.mark.asyncio
    async def test_acquire_with_retry(self, connection_pool):
        """Test connection acquisition with retry logic."""
        if ASYNCPG_AVAILABLE:
            mock_pool = AsyncMock()
            # First call fails, second succeeds
            mock_conn = AsyncMock()
            mock_pool.acquire.side_effect = [Exception("Connection lost"), mock_conn]
            connection_pool._pool = mock_pool

            with patch.object(connection_pool, '_handle_connection_loss') as mock_handle:
                mock_handle.return_value = None  # Simulate successful recovery

                async with connection_pool.acquire() as conn:
                    assert conn == mock_conn
                    mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, connection_pool):
        """Test pool closure."""
        await connection_pool.initialize()

        if ASYNCPG_AVAILABLE:
            mock_pool = AsyncMock()
            connection_pool._pool = mock_pool

            await connection_pool.close()
            mock_pool.close.assert_called_once()
            assert connection_pool._pool is None

    def test_is_connected_property(self, connection_pool):
        """Test connection status property."""
        # Initially not connected
        assert not connection_pool.is_connected

        if ASYNCPG_AVAILABLE:
            # Mock connected state
            mock_pool = MagicMock()
            mock_pool._closed = False
            connection_pool._pool = mock_pool
            assert connection_pool.is_connected


class TestSchemaManager:
    """Test database schema management."""

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = AsyncMock()
        mock_conn = AsyncMock()
        pool.acquire.return_value.__aenter__.return_value = mock_conn
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return pool

    @pytest.fixture
    def schema_manager(self, mock_pool):
        """Create schema manager for testing."""
        connection_pool = MagicMock()
        connection_pool.acquire = mock_pool.acquire
        return SchemaManager(connection_pool)

    @pytest.mark.asyncio
    async def test_initialize_schema(self, schema_manager, mock_pool):
        """Test schema initialization."""
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value

        await schema_manager.initialize_schema()

        # Verify table creation calls
        assert mock_conn.execute.call_count >= 3  # Tables + indexes
        calls = [call.args[0] for call in mock_conn.execute.call_args_list]

        # Check for table creation
        equity_table_created = any("CREATE TABLE IF NOT EXISTS equity_profile" in call for call in calls)
        fundata_data_created = any("CREATE TABLE IF NOT EXISTS fundata_data" in call for call in calls)
        fundata_quotes_created = any("CREATE TABLE IF NOT EXISTS fundata_quotes" in call for call in calls)

        assert equity_table_created
        assert fundata_data_created
        assert fundata_quotes_created

    @pytest.mark.asyncio
    async def test_check_schema_exists_complete(self, schema_manager, mock_pool):
        """Test schema existence check when all tables exist."""
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        # Mock 3 tables existing
        mock_conn.fetch.return_value = [
            {'table_name': 'equity_profile'},
            {'table_name': 'fundata_data'},
            {'table_name': 'fundata_quotes'}
        ]

        result = await schema_manager.check_schema_exists()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_schema_exists_incomplete(self, schema_manager, mock_pool):
        """Test schema existence check when tables are missing."""
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        # Mock only 2 tables existing
        mock_conn.fetch.return_value = [
            {'table_name': 'equity_profile'},
            {'table_name': 'fundata_data'}
        ]

        result = await schema_manager.check_schema_exists()
        assert result is False

    @pytest.mark.asyncio
    async def test_create_equity_tables(self, schema_manager, mock_pool):
        """Test equity table creation."""
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value

        await schema_manager._create_equity_tables(mock_conn)

        # Verify equity_profile table creation
        mock_conn.execute.assert_called()
        call_args = mock_conn.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS equity_profile" in call_args

    @pytest.mark.asyncio
    async def test_create_fundata_tables(self, schema_manager, mock_pool):
        """Test fundata tables creation."""
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value

        await schema_manager._create_fundata_tables(mock_conn)

        # Should create both fundata_data and fundata_quotes
        assert mock_conn.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_create_indexes_with_errors(self, schema_manager, mock_pool):
        """Test index creation with some failures."""
        mock_conn = mock_pool.acquire.return_value.__aenter__.return_value
        # Mock some index creations failing
        mock_conn.execute.side_effect = [None, Exception("Index exists"), None]

        # Should not raise exception, just log warnings
        await schema_manager._create_indexes(mock_conn)

        # Verify multiple index creation attempts
        assert mock_conn.execute.call_count >= 3


class TestDatabaseManager:
    """Test high-level database manager."""

    @pytest.fixture
    def mock_config(self):
        """Create mock database configuration."""
        config = MagicMock()
        config.dsn = "postgresql://test:test@localhost:5432/test"
        return config

    @pytest.fixture
    def database_manager(self, mock_config):
        """Create database manager for testing."""
        with patch('packages.sra_data.repositories.database.DatabaseConfig') as mock_config_class:
            mock_config_class.return_value = mock_config
            return DatabaseManager()

    @pytest.mark.asyncio
    async def test_initialize_new_schema(self, database_manager):
        """Test initialization with new schema creation."""
        with patch.object(database_manager.pool, 'initialize') as mock_pool_init, \
             patch.object(database_manager.schema, 'check_schema_exists', return_value=False) as mock_check, \
             patch.object(database_manager.schema, 'initialize_schema') as mock_schema_init:

            await database_manager.initialize()

            mock_pool_init.assert_called_once()
            mock_check.assert_called_once()
            mock_schema_init.assert_called_once()
            assert database_manager._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_existing_schema(self, database_manager):
        """Test initialization with existing schema."""
        with patch.object(database_manager.pool, 'initialize') as mock_pool_init, \
             patch.object(database_manager.schema, 'check_schema_exists', return_value=True) as mock_check, \
             patch.object(database_manager.schema, 'initialize_schema') as mock_schema_init:

            await database_manager.initialize()

            mock_pool_init.assert_called_once()
            mock_check.assert_called_once()
            mock_schema_init.assert_not_called()  # Schema already exists
            assert database_manager._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_failure(self, database_manager):
        """Test initialization failure handling."""
        with patch.object(database_manager.pool, 'initialize') as mock_pool_init:
            mock_pool_init.side_effect = Exception("Pool init failed")

            with pytest.raises(Exception, match="Pool init failed"):
                await database_manager.initialize()

            assert database_manager._initialized is False

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, database_manager):
        """Test health check when database is healthy."""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_conn.fetch.return_value = [
            {'table_name': 'equity_profile'},
            {'table_name': 'fundata_data'}
        ]

        with patch.object(database_manager.pool, 'acquire') as mock_acquire:
            mock_acquire.return_value.__aenter__.return_value = mock_conn
            mock_acquire.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await database_manager.health_check()

            assert result['status'] == 'healthy'
            assert result['connected'] is True
            assert result['tables_count'] == 2
            assert 'timestamp' in result

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, database_manager):
        """Test health check when database is unhealthy."""
        with patch.object(database_manager.pool, 'acquire') as mock_acquire:
            mock_acquire.side_effect = Exception("Connection failed")

            result = await database_manager.health_check()

            assert result['status'] == 'unhealthy'
            assert 'error' in result
            assert 'timestamp' in result

    @pytest.mark.asyncio
    async def test_close(self, database_manager):
        """Test database manager closure."""
        database_manager._initialized = True

        with patch.object(database_manager.pool, 'close') as mock_close:
            await database_manager.close()

            mock_close.assert_called_once()
            assert database_manager._initialized is False

    def test_destructor_warning(self, database_manager):
        """Test destructor warning for unclosed manager."""
        database_manager._initialized = True
        database_manager.pool.is_connected = True

        with patch('packages.sra_data.repositories.database.logger') as mock_logger:
            database_manager.__del__()
            mock_logger.warning.assert_called_once()


class TestFactoryFunction:
    """Test factory function for database manager creation."""

    @pytest.mark.asyncio
    async def test_create_database_manager_default_config(self):
        """Test creating database manager with default configuration."""
        with patch('packages.sra_data.repositories.database.DatabaseManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            result = await create_database_manager()

            mock_manager_class.assert_called_once_with(None)
            mock_manager.initialize.assert_called_once()
            assert result == mock_manager

    @pytest.mark.asyncio
    async def test_create_database_manager_custom_config(self):
        """Test creating database manager with custom configuration."""
        config = {'host': 'custom-host'}

        with patch('packages.sra_data.repositories.database.DatabaseManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            result = await create_database_manager(config)

            mock_manager_class.assert_called_once_with(config)
            mock_manager.initialize.assert_called_once()
            assert result == mock_manager


class TestAsyncpgAvailability:
    """Test behavior when asyncpg is not available."""

    @pytest.mark.asyncio
    async def test_mock_connection_behavior(self):
        """Test mock connection when asyncpg is unavailable."""
        if not ASYNCPG_AVAILABLE:
            # This tests the mock implementation
            config = DatabaseConfig({
                'host': 'localhost',
                'port': 5432,
                'database': 'test',
                'user': 'test'
            })
            pool = ConnectionPool(config)

            # Should not raise exception
            async with pool.acquire() as conn:
                assert conn is not None
                # Mock methods should be available
                result = await conn.execute("SELECT 1")
                assert result is not None

    def test_mock_types_available(self):
        """Test that mock types are available when asyncpg is not."""
        if not ASYNCPG_AVAILABLE:
            from packages.sra_data.repositories.database import asyncpg

            # Mock classes should be available
            assert hasattr(asyncpg, 'Pool')
            assert hasattr(asyncpg, 'Connection')
            assert hasattr(asyncpg, 'ConnectionDoesNotExistError')
            assert hasattr(asyncpg, 'InterfaceError')


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests requiring actual database connection."""

    @pytest.fixture
    async def db_manager(self):
        """Create database manager for integration tests."""
        # Only run if we have a test database available
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
        except Exception:
            pytest.skip("Test database not available")
        finally:
            if 'manager' in locals():
                await manager.close()

    @pytest.mark.asyncio
    async def test_full_initialization(self, db_manager):
        """Test full database initialization with real connection."""
        health = await db_manager.health_check()
        assert health['status'] == 'healthy'
        assert health['connected'] is True

    @pytest.mark.asyncio
    async def test_schema_creation_and_check(self, db_manager):
        """Test schema creation and verification."""
        # Schema should be created during initialization
        exists = await db_manager.schema.check_schema_exists()
        assert exists is True

    @pytest.mark.asyncio
    async def test_connection_pool_operations(self, db_manager):
        """Test connection pool acquire and release operations."""
        # Test multiple concurrent connections
        async def test_connection():
            async with db_manager.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result

        # Run multiple concurrent operations
        results = await asyncio.gather(*[test_connection() for _ in range(5)])
        assert all(result == 1 for result in results)