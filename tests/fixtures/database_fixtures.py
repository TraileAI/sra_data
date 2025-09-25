"""Database test fixtures for async infrastructure testing."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
import os
from typing import Dict, Any, Optional


@pytest.fixture
async def mock_db_pool():
    """Mock asyncpg connection pool for testing."""
    try:
        # Try to import asyncpg for proper typing
        import asyncpg
        pool_spec = asyncpg.Pool
        connection_spec = asyncpg.Connection
    except ImportError:
        # Fallback if asyncpg not available
        pool_spec = None
        connection_spec = None

    pool = AsyncMock(spec=pool_spec)
    connection = AsyncMock(spec=connection_spec)

    # Configure mock connection
    connection.execute.return_value = None
    connection.fetchval.return_value = 1
    connection.fetch.return_value = []
    connection.fetchrow.return_value = None

    # Configure pool acquire context manager
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    return pool


@pytest.fixture
def test_db_config() -> Dict[str, Any]:
    """Test database configuration."""
    return {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_sra_data',
        'user': 'test_user',
        'password': 'test_pass',
        'min_size': 1,
        'max_size': 10,
        'command_timeout': 60
    }


@pytest.fixture
def invalid_db_config() -> Dict[str, Any]:
    """Invalid database configuration for negative testing."""
    return {
        'host': '',  # Empty host
        'port': -1,  # Invalid port
        'database': '',  # Empty database
        'user': '',  # Empty user
    }


@pytest.fixture
async def in_memory_db():
    """In-memory SQLite database for testing."""
    try:
        import aiosqlite
        db = await aiosqlite.connect(':memory:')
        yield db
        await db.close()
    except ImportError:
        # If aiosqlite not available, yield None
        yield None


@pytest.fixture
def schema_sql_statements() -> Dict[str, str]:
    """SQL statements for schema creation testing."""
    return {
        'equity_profile': '''
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''',
        'fundata_data': '''
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
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                additional_data JSONB,
                UNIQUE(instrument_key, record_id)
            );
        ''',
        'fundata_quotes': '''
            CREATE TABLE IF NOT EXISTS fundata_quotes (
                id SERIAL PRIMARY KEY,
                instrument_key VARCHAR(20) NOT NULL,
                record_id VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                navps DECIMAL(12,2) NOT NULL,
                navps_penny_change DECIMAL(12,2),
                navps_percent_change DECIMAL(8,6),
                previous_date DATE,
                previous_navps DECIMAL(12,2),
                record_state VARCHAR(20) DEFAULT 'Active',
                change_date DATE,
                source_file VARCHAR(255) NOT NULL,
                file_hash VARCHAR(64),
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                additional_data JSONB,
                UNIQUE(instrument_key, record_id, date)
            );
        '''
    }


@pytest.fixture
def index_sql_statements() -> Dict[str, str]:
    """SQL statements for index creation testing."""
    return {
        'equity_profile_indexes': [
            'CREATE INDEX IF NOT EXISTS idx_equity_exchange ON equity_profile(exchange);',
            'CREATE INDEX IF NOT EXISTS idx_equity_sector ON equity_profile(sector);',
            'CREATE INDEX IF NOT EXISTS idx_equity_updated ON equity_profile(updated_at);'
        ],
        'fundata_data_indexes': [
            'CREATE INDEX IF NOT EXISTS idx_fundata_data_instrument ON fundata_data(instrument_key);',
            'CREATE INDEX IF NOT EXISTS idx_fundata_data_processed ON fundata_data(processed_at);',
            'CREATE INDEX IF NOT EXISTS idx_fundata_data_state ON fundata_data(record_state);'
        ],
        'fundata_quotes_indexes': [
            'CREATE INDEX IF NOT EXISTS idx_fundata_quotes_instrument ON fundata_quotes(instrument_key);',
            'CREATE INDEX IF NOT EXISTS idx_fundata_quotes_date ON fundata_quotes(date);',
            'CREATE INDEX IF NOT EXISTS idx_fundata_quotes_navps ON fundata_quotes(navps);',
            'CREATE INDEX IF NOT EXISTS idx_fundata_quotes_processed ON fundata_quotes(processed_at);'
        ]
    }


@pytest.fixture
def mock_connection_error():
    """Mock connection error for resilience testing."""
    return Exception("Connection lost to database server")


@pytest.fixture
def mock_successful_reconnection():
    """Mock successful reconnection scenario."""
    async def reconnect_mock():
        await asyncio.sleep(0.1)  # Simulate reconnection delay
        return True
    return reconnect_mock