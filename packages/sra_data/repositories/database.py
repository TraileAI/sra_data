"""Database infrastructure for SRA Data processing service.

This module provides async database connectivity, connection pooling,
and schema management for data processing operations.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
import json

# Import requirements that should be available
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    # Create mock types for development
    class MockConnection:
        async def execute(self, query): pass
        async def fetchval(self, query): return 1
        async def fetch(self, query): return []
        async def fetchrow(self, query): return None
        async def set_type_codec(self, *args, **kwargs): pass

    class MockPool:
        def __init__(self):
            self._closed = False
        async def acquire(self): return MockConnection()
        async def release(self, conn): pass
        async def close(self): self._closed = True

    # Mock asyncpg module
    class MockAsyncpg:
        Pool = MockPool
        Connection = MockConnection
        ConnectionDoesNotExistError = Exception
        InterfaceError = Exception
        @staticmethod
        async def create_pool(*args, **kwargs):
            return MockPool()

    asyncpg = MockAsyncpg()


logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database configuration.

        Args:
            config: Database configuration dictionary
        """
        self.config = config or self._load_from_environment()
        self._validate_config()

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load database configuration from environment variables."""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'sra_data'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'min_size': int(os.getenv('DB_POOL_MIN_SIZE', '5')),
            'max_size': int(os.getenv('DB_POOL_MAX_SIZE', '20')),
            'command_timeout': int(os.getenv('DB_COMMAND_TIMEOUT', '60')),
            'server_settings': {
                'application_name': 'sra_data_processor',
                'search_path': 'public'
            }
        }

    def _validate_config(self) -> None:
        """Validate database configuration."""
        required_fields = ['host', 'port', 'database', 'user']
        missing_fields = [field for field in required_fields if not self.config.get(field)]

        if missing_fields:
            raise ValueError(f"Missing required database config fields: {missing_fields}")

        if self.config['port'] <= 0 or self.config['port'] > 65535:
            raise ValueError(f"Invalid port number: {self.config['port']}")

        if self.config.get('min_size', 0) < 0:
            raise ValueError("min_size cannot be negative")

        if self.config.get('max_size', 0) <= 0:
            raise ValueError("max_size must be positive")

    @property
    def dsn(self) -> str:
        """Get database connection DSN."""
        return (
            f"postgresql://{self.config['user']}:{self.config['password']}@"
            f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )


class ConnectionPool:
    """Async connection pool manager with resilience features."""

    def __init__(self, config: DatabaseConfig):
        """Initialize connection pool manager.

        Args:
            config: Database configuration
        """
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._initialization_lock = asyncio.Lock()
        self._reconnection_attempts = 0
        self._max_reconnection_attempts = 3

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available - using mock pool for development")
            return

        async with self._initialization_lock:
            if self._pool is not None:
                return

            try:
                logger.info("Initializing database connection pool...")

                # Create connection pool with optimized settings for data processing
                self._pool = await asyncpg.create_pool(
                    dsn=self.config.dsn,
                    min_size=self.config.config['min_size'],
                    max_size=self.config.config['max_size'],
                    command_timeout=self.config.config['command_timeout'],
                    server_settings=self.config.config.get('server_settings', {}),
                    # Optimize for bulk operations
                    init=self._setup_connection
                )

                logger.info(
                    f"Database pool initialized: {self.config.config['min_size']}-"
                    f"{self.config.config['max_size']} connections"
                )
                self._reconnection_attempts = 0

            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise

    async def _setup_connection(self, connection: asyncpg.Connection) -> None:
        """Setup individual connection for optimal data processing.

        Args:
            connection: Database connection to configure
        """
        # Set connection-level optimizations for bulk operations
        await connection.execute("SET synchronous_commit = OFF")  # Faster bulk inserts
        await connection.execute("SET wal_buffers = '16MB'")      # Better WAL performance
        await connection.execute("SET checkpoint_completion_target = 0.9")

        # Set up JSON handling
        await connection.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Acquire a connection from the pool with automatic retry logic.

        Yields:
            Database connection
        """
        if not ASYNCPG_AVAILABLE:
            # Return a mock connection for development
            from unittest.mock import AsyncMock
            mock_conn = AsyncMock()
            yield mock_conn
            return

        if self._pool is None:
            await self.initialize()

        connection = None
        try:
            connection = await self._pool.acquire()
            yield connection

        except (asyncpg.ConnectionDoesNotExistError,
                asyncpg.InterfaceError,
                ConnectionResetError) as e:
            logger.warning(f"Connection issue detected: {e}")

            if self._reconnection_attempts < self._max_reconnection_attempts:
                await self._handle_connection_loss()
                # Retry acquisition
                connection = await self._pool.acquire()
                yield connection
            else:
                raise

        except Exception as e:
            logger.error(f"Unexpected error in connection acquisition: {e}")
            raise

        finally:
            if connection:
                try:
                    await self._pool.release(connection)
                except Exception as e:
                    logger.warning(f"Error releasing connection: {e}")

    async def _handle_connection_loss(self) -> None:
        """Handle connection pool recovery."""
        self._reconnection_attempts += 1
        logger.info(f"Attempting pool recovery (attempt {self._reconnection_attempts})")

        try:
            # Close existing pool
            if self._pool:
                await self._pool.close()
                self._pool = None

            # Wait before reconnection
            await asyncio.sleep(2 ** self._reconnection_attempts)

            # Reinitialize
            await self.initialize()

        except Exception as e:
            logger.error(f"Pool recovery failed: {e}")
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            logger.info("Closing database connection pool...")
            await self._pool.close()
            self._pool = None

    @property
    def is_connected(self) -> bool:
        """Check if pool is connected."""
        return self._pool is not None and not self._pool._closed


class SchemaManager:
    """Database schema management for data processing."""

    def __init__(self, pool: ConnectionPool):
        """Initialize schema manager.

        Args:
            pool: Database connection pool
        """
        self.pool = pool

    async def initialize_schema(self) -> None:
        """Initialize database schema for data processing."""
        logger.info("Initializing database schema...")

        async with self.pool.acquire() as connection:
            # Create tables in order of dependencies
            await self._create_equity_tables(connection)
            await self._create_fundata_tables(connection)
            await self._create_indexes(connection)

        logger.info("Database schema initialization complete")

    async def _create_equity_tables(self, connection: asyncpg.Connection) -> None:
        """Create equity-related tables."""
        equity_profile_sql = """
        CREATE TABLE IF NOT EXISTS equity_profile (
            symbol VARCHAR(10) PRIMARY KEY,
            company_name VARCHAR(255) NOT NULL,
            exchange VARCHAR(10) NOT NULL,
            sector VARCHAR(100),
            industry VARCHAR(100),
            market_cap DECIMAL(20,2),
            employees INTEGER,
            description TEXT,
            website VARCHAR(255),
            country VARCHAR(3) DEFAULT 'US',
            currency VARCHAR(3) DEFAULT 'USD',
            is_etf BOOLEAN DEFAULT FALSE,
            is_actively_trading BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """

        await connection.execute(equity_profile_sql)
        logger.debug("Created equity_profile table")

    async def _create_fundata_tables(self, connection: asyncpg.Connection) -> None:
        """Create fundata-related tables with flat denormalized structure."""

        # Fundata general data table
        fundata_data_sql = """
        CREATE TABLE IF NOT EXISTS fundata_data (
            id SERIAL PRIMARY KEY,
            instrument_key VARCHAR(20) NOT NULL,
            record_id VARCHAR(20) NOT NULL,
            language VARCHAR(5),
            legal_name VARCHAR(500),
            family_name VARCHAR(255),
            series_name VARCHAR(255),
            company VARCHAR(255),
            inception_date DATE,
            change_date DATE,
            currency VARCHAR(3),
            record_state VARCHAR(20) DEFAULT 'Active',
            source_file VARCHAR(255) NOT NULL,
            file_hash VARCHAR(64),
            processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            additional_data JSONB,
            CONSTRAINT unique_fundata_data UNIQUE (instrument_key, record_id)
        );
        """

        # Fundata quotes table
        fundata_quotes_sql = """
        CREATE TABLE IF NOT EXISTS fundata_quotes (
            id SERIAL PRIMARY KEY,
            instrument_key VARCHAR(20) NOT NULL,
            record_id VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            navps DECIMAL(12,2) NOT NULL CHECK (navps > 0),
            navps_penny_change DECIMAL(12,2),
            navps_percent_change DECIMAL(8,6),
            previous_date DATE,
            previous_navps DECIMAL(12,2) CHECK (previous_navps > 0 OR previous_navps IS NULL),
            record_state VARCHAR(20) DEFAULT 'Active',
            change_date DATE,
            source_file VARCHAR(255) NOT NULL,
            file_hash VARCHAR(64),
            processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            additional_data JSONB,
            CONSTRAINT unique_fundata_quotes UNIQUE (instrument_key, record_id, date)
        );
        """

        await connection.execute(fundata_data_sql)
        logger.debug("Created fundata_data table")

        await connection.execute(fundata_quotes_sql)
        logger.debug("Created fundata_quotes table")

    async def _create_indexes(self, connection: asyncpg.Connection) -> None:
        """Create optimized indexes for data processing."""

        # Equity profile indexes
        equity_indexes = [
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_equity_exchange ON equity_profile(exchange)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_equity_sector ON equity_profile(sector)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_equity_updated ON equity_profile(updated_at)"
        ]

        # Fundata data indexes - optimized for identifier lookups
        fundata_data_indexes = [
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_instrument ON fundata_data(instrument_key)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_record ON fundata_data(record_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_processed ON fundata_data(processed_at)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_state ON fundata_data(record_state)"
        ]

        # Fundata quotes indexes - optimized for time-series queries
        fundata_quotes_indexes = [
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_instrument ON fundata_quotes(instrument_key)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_date ON fundata_quotes(date)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_navps ON fundata_quotes(navps)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_processed ON fundata_quotes(processed_at)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_composite ON fundata_quotes(instrument_key, date)"
        ]

        all_indexes = equity_indexes + fundata_data_indexes + fundata_quotes_indexes

        for index_sql in all_indexes:
            try:
                await connection.execute(index_sql)
                logger.debug(f"Created index: {index_sql.split('idx_')[1].split()[0] if 'idx_' in index_sql else 'unknown'}")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")

    async def check_schema_exists(self) -> bool:
        """Check if required schema exists."""
        async with self.pool.acquire() as connection:
            result = await connection.fetch("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('equity_profile', 'fundata_data', 'fundata_quotes')
            """)
            return len(result) == 3


class DatabaseManager:
    """High-level database management for SRA data processing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database manager.

        Args:
            config: Database configuration dictionary
        """
        self._initialized = False
        self.config = DatabaseConfig(config)
        self.pool = ConnectionPool(self.config)
        self.schema = SchemaManager(self.pool)

    async def initialize(self) -> None:
        """Initialize database infrastructure."""
        if self._initialized:
            return

        logger.info("Initializing database infrastructure...")

        try:
            await self.pool.initialize()

            # Check if schema exists, create if needed
            if not await self.schema.check_schema_exists():
                await self.schema.initialize_schema()
            else:
                logger.info("Database schema already exists")

            self._initialized = True
            logger.info("Database infrastructure ready")

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check.

        Returns:
            Health check results
        """
        try:
            async with self.pool.acquire() as connection:
                # Simple query to test connectivity
                result = await connection.fetchval("SELECT 1")

                # Check table existence
                tables = await connection.fetch("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """)

                return {
                    'status': 'healthy',
                    'connected': result == 1,
                    'tables_count': len(tables),
                    'tables': [row['table_name'] for row in tables],
                    'timestamp': datetime.utcnow().isoformat()
                }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def close(self) -> None:
        """Close database connections."""
        await self.pool.close()
        self._initialized = False

    def __del__(self):
        """Cleanup on object destruction."""
        if self._initialized and self.pool.is_connected:
            logger.warning("DatabaseManager was not properly closed")


# Factory function for easy instantiation
async def create_database_manager(config: Optional[Dict[str, Any]] = None) -> DatabaseManager:
    """Create and initialize a database manager.

    Args:
        config: Database configuration

    Returns:
        Initialized database manager
    """
    manager = DatabaseManager(config)
    await manager.initialize()
    return manager