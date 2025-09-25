#!/usr/bin/env python3
"""Simple test script to verify database infrastructure works correctly."""

import sys
import os
import asyncio
from unittest.mock import AsyncMock

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages'))

async def test_database_config():
    """Test database configuration."""
    print("Testing DatabaseConfig...")

    try:
        from sra_data.repositories.database import DatabaseConfig

        # Test valid config
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass',
            'min_size': 2,
            'max_size': 5
        }

        db_config = DatabaseConfig(config)
        print(f"✓ Valid DatabaseConfig created: {db_config.config['host']}:{db_config.config['port']}")

        # Test invalid config
        try:
            invalid_config = {
                'host': '',  # Empty host should fail
                'port': -1,  # Invalid port
                'database': 'test',
                'user': 'test'
            }
            DatabaseConfig(invalid_config)
            print("✗ Invalid config validation failed")
            return False
        except ValueError as e:
            print(f"✓ Invalid config properly rejected: {e}")

        return True

    except Exception as e:
        print(f"✗ DatabaseConfig test failed: {e}")
        return False

async def test_connection_pool():
    """Test connection pool."""
    print("\nTesting ConnectionPool...")

    try:
        from sra_data.repositories.database import ConnectionPool, DatabaseConfig

        config = DatabaseConfig({
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass',
            'min_size': 2,
            'max_size': 5
        })

        pool = ConnectionPool(config)
        print(f"✓ ConnectionPool created with config")

        # Test initialization (will use mock since asyncpg may not be available)
        await pool.initialize()
        print("✓ Pool initialization completed")

        # Test connection acquisition
        async with pool.acquire() as connection:
            print("✓ Connection acquired successfully")
            # Connection should be mock or real depending on environment

        return True

    except Exception as e:
        print(f"✗ ConnectionPool test failed: {e}")
        return False

async def test_database_manager():
    """Test database manager."""
    print("\nTesting DatabaseManager...")

    try:
        from sra_data.repositories.database import DatabaseManager

        # Test with mock config
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass',
            'min_size': 2,
            'max_size': 5
        }

        manager = DatabaseManager(config)
        print("✓ DatabaseManager created")

        # Test initialization
        await manager.initialize()
        print("✓ DatabaseManager initialization completed")

        # Test health check
        health = await manager.health_check()
        print(f"✓ Health check completed: {health.get('status', 'unknown')}")

        # Cleanup
        await manager.close()
        print("✓ DatabaseManager closed properly")

        return True

    except Exception as e:
        print(f"✗ DatabaseManager test failed: {e}")
        return False

async def test_schema_manager():
    """Test schema manager functionality."""
    print("\nTesting SchemaManager...")

    try:
        from sra_data.repositories.database import SchemaManager, ConnectionPool, DatabaseConfig

        config = DatabaseConfig({
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_pass',
            'min_size': 2,
            'max_size': 5
        })

        pool = ConnectionPool(config)
        await pool.initialize()

        schema = SchemaManager(pool)
        print("✓ SchemaManager created")

        # Schema operations will use mock connections
        schema_exists = await schema.check_schema_exists()
        print(f"✓ Schema existence check completed: {schema_exists}")

        return True

    except Exception as e:
        print(f"✗ SchemaManager test failed: {e}")
        return False

async def main():
    """Run all database infrastructure tests."""
    print("Running Database Infrastructure Tests")
    print("="*50)

    success = True
    success &= await test_database_config()
    success &= await test_connection_pool()
    success &= await test_database_manager()
    success &= await test_schema_manager()

    print("\n" + "="*50)
    if success:
        print("✓ All database infrastructure tests passed!")
        return 0
    else:
        print("✗ Some database infrastructure tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)