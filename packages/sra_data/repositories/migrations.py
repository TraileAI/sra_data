"""Database migration system for SRA Data processing service.

This module provides comprehensive database migration functionality with
version tracking, rollback support, and automated schema evolution.
"""

import logging
import os
import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from packages.sra_data.repositories.database import DatabaseManager

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    # Mock for development
    class MockConnection:
        async def execute(self, query, *args): return "MOCK"
        async def fetch(self, query, *args): return []
        async def fetchrow(self, query, *args): return None
        async def fetchval(self, query, *args): return None

    asyncpg = type('MockAsyncpg', (), {
        'Connection': MockConnection,
        'Record': dict
    })()

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    """Migration execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """Single database migration definition."""
    version: str
    name: str
    description: str
    up_sql: str
    down_sql: str = ""
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate migration after initialization."""
        if not self.version:
            raise ValueError("Migration version cannot be empty")
        if not self.name:
            raise ValueError("Migration name cannot be empty")
        if not self.up_sql.strip():
            raise ValueError("Migration up_sql cannot be empty")


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    version: str
    status: MigrationStatus
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    applied_at: Optional[datetime] = None


class MigrationManager:
    """Comprehensive database migration manager."""

    def __init__(self, db_manager: DatabaseManager, migrations_table: str = "schema_migrations"):
        """Initialize migration manager.

        Args:
            db_manager: Database manager instance
            migrations_table: Name of migrations tracking table
        """
        self.db_manager = db_manager
        self.migrations_table = migrations_table
        self._migrations: Dict[str, Migration] = {}
        self._migration_order: List[str] = []

    async def initialize(self) -> None:
        """Initialize migration system and create tracking table."""
        logger.info("Initializing migration system...")

        await self._create_migrations_table()
        await self._load_built_in_migrations()

        logger.info(f"Migration system ready with {len(self._migrations)} available migrations")

    async def _create_migrations_table(self) -> None:
        """Create the migrations tracking table."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migrations_table} (
            version VARCHAR(50) PRIMARY KEY,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            applied_at TIMESTAMP WITH TIME ZONE,
            execution_time_seconds DECIMAL(10,3),
            error_message TEXT,
            rollback_sql TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_{self.migrations_table}_status
        ON {self.migrations_table}(status);

        CREATE INDEX IF NOT EXISTS idx_{self.migrations_table}_applied
        ON {self.migrations_table}(applied_at);
        """

        async with self.db_manager.pool.acquire() as connection:
            await connection.execute(create_table_sql)

    async def _load_built_in_migrations(self) -> None:
        """Load built-in migrations for the SRA data system."""

        # Migration 001: Initial schema
        self.add_migration(Migration(
            version="001",
            name="initial_schema",
            description="Create initial tables for equity profiles, fundata data and quotes",
            up_sql="""
            -- Create equity_profile table
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

            -- Create fundata_data table
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

            -- Create fundata_quotes table
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
            """,
            down_sql="""
            DROP TABLE IF EXISTS fundata_quotes;
            DROP TABLE IF EXISTS fundata_data;
            DROP TABLE IF EXISTS equity_profile;
            """
        ))

        # Migration 002: Add indexes for performance
        self.add_migration(Migration(
            version="002",
            name="performance_indexes",
            description="Add optimized indexes for query performance",
            dependencies=["001"],
            up_sql="""
            -- Equity profile indexes
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_equity_exchange ON equity_profile(exchange);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_equity_sector ON equity_profile(sector);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_equity_updated ON equity_profile(updated_at);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_equity_market_cap ON equity_profile(market_cap DESC NULLS LAST);

            -- Fundata data indexes
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_instrument ON fundata_data(instrument_key);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_record ON fundata_data(record_id);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_processed ON fundata_data(processed_at);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_state ON fundata_data(record_state);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_data_family ON fundata_data(family_name);

            -- Fundata quotes indexes
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_instrument ON fundata_quotes(instrument_key);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_date ON fundata_quotes(date DESC);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_navps ON fundata_quotes(navps);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_processed ON fundata_quotes(processed_at);
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_fundata_quotes_composite ON fundata_quotes(instrument_key, date DESC);
            """,
            down_sql="""
            DROP INDEX IF EXISTS idx_fundata_quotes_composite;
            DROP INDEX IF EXISTS idx_fundata_quotes_processed;
            DROP INDEX IF EXISTS idx_fundata_quotes_navps;
            DROP INDEX IF EXISTS idx_fundata_quotes_date;
            DROP INDEX IF EXISTS idx_fundata_quotes_instrument;
            DROP INDEX IF EXISTS idx_fundata_data_family;
            DROP INDEX IF EXISTS idx_fundata_data_state;
            DROP INDEX IF EXISTS idx_fundata_data_processed;
            DROP INDEX IF EXISTS idx_fundata_data_record;
            DROP INDEX IF EXISTS idx_fundata_data_instrument;
            DROP INDEX IF EXISTS idx_equity_market_cap;
            DROP INDEX IF EXISTS idx_equity_updated;
            DROP INDEX IF EXISTS idx_equity_sector;
            DROP INDEX IF EXISTS idx_equity_exchange;
            """
        ))

        # Migration 003: Add audit and tracking features
        self.add_migration(Migration(
            version="003",
            name="audit_tracking",
            description="Add audit trails and data lineage tracking",
            dependencies=["001"],
            up_sql="""
            -- Add audit columns to existing tables
            ALTER TABLE equity_profile
            ADD COLUMN IF NOT EXISTS last_updated_by VARCHAR(100),
            ADD COLUMN IF NOT EXISTS data_source VARCHAR(50) DEFAULT 'unknown',
            ADD COLUMN IF NOT EXISTS validation_status VARCHAR(20) DEFAULT 'pending';

            -- Create audit log table
            CREATE TABLE IF NOT EXISTS audit_log (
                id SERIAL PRIMARY KEY,
                table_name VARCHAR(100) NOT NULL,
                record_id VARCHAR(100) NOT NULL,
                operation VARCHAR(10) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
                old_values JSONB,
                new_values JSONB,
                changed_by VARCHAR(100),
                changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                change_reason TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_audit_log_table ON audit_log(table_name);
            CREATE INDEX IF NOT EXISTS idx_audit_log_record ON audit_log(record_id);
            CREATE INDEX IF NOT EXISTS idx_audit_log_changed_at ON audit_log(changed_at DESC);

            -- Create data quality metrics table
            CREATE TABLE IF NOT EXISTS data_quality_metrics (
                id SERIAL PRIMARY KEY,
                table_name VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(15,6) NOT NULL,
                measurement_date DATE NOT NULL DEFAULT CURRENT_DATE,
                details JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT unique_daily_metric UNIQUE (table_name, metric_name, measurement_date)
            );
            """,
            down_sql="""
            DROP TABLE IF EXISTS data_quality_metrics;
            DROP TABLE IF EXISTS audit_log;
            ALTER TABLE equity_profile
            DROP COLUMN IF EXISTS validation_status,
            DROP COLUMN IF EXISTS data_source,
            DROP COLUMN IF EXISTS last_updated_by;
            """
        ))

    def add_migration(self, migration: Migration) -> None:
        """Add a migration to the manager.

        Args:
            migration: Migration to add
        """
        if migration.version in self._migrations:
            raise ValueError(f"Migration version {migration.version} already exists")

        self._migrations[migration.version] = migration
        self._migration_order.append(migration.version)
        self._migration_order.sort()  # Keep versions in order

    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations in dependency order.

        Returns:
            List of pending migrations
        """
        async with self.db_manager.pool.acquire() as connection:
            applied_versions = await connection.fetch(
                f"SELECT version FROM {self.migrations_table} WHERE status = 'completed'"
            )
            applied_set = {row['version'] for row in applied_versions}

        pending = []
        for version in self._migration_order:
            if version not in applied_set:
                migration = self._migrations[version]
                # Check dependencies
                if all(dep in applied_set for dep in migration.dependencies):
                    pending.append(migration)

        return pending

    async def apply_migration(self, migration: Migration) -> MigrationResult:
        """Apply a single migration.

        Args:
            migration: Migration to apply

        Returns:
            Migration result
        """
        start_time = datetime.utcnow()
        logger.info(f"Applying migration {migration.version}: {migration.name}")

        try:
            # Record migration as running
            await self._record_migration_status(
                migration, MigrationStatus.RUNNING, start_time
            )

            # Execute migration in transaction
            async with self.db_manager.pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(migration.up_sql)

            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            # Record successful completion
            await self._record_migration_status(
                migration, MigrationStatus.COMPLETED, end_time, execution_time
            )

            logger.info(f"Migration {migration.version} applied successfully in {execution_time:.2f}s")

            return MigrationResult(
                version=migration.version,
                status=MigrationStatus.COMPLETED,
                execution_time_seconds=execution_time,
                applied_at=end_time
            )

        except Exception as e:
            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            error_msg = f"Migration {migration.version} failed: {e}"
            logger.error(error_msg)

            # Record failure
            await self._record_migration_status(
                migration, MigrationStatus.FAILED, end_time, execution_time, error_msg
            )

            return MigrationResult(
                version=migration.version,
                status=MigrationStatus.FAILED,
                execution_time_seconds=execution_time,
                error_message=error_msg,
                applied_at=end_time
            )

    async def rollback_migration(self, version: str) -> MigrationResult:
        """Rollback a specific migration.

        Args:
            version: Version to rollback

        Returns:
            Migration result
        """
        if version not in self._migrations:
            raise ValueError(f"Migration version {version} not found")

        migration = self._migrations[version]
        if not migration.down_sql.strip():
            raise ValueError(f"Migration {version} has no rollback SQL")

        start_time = datetime.utcnow()
        logger.info(f"Rolling back migration {version}: {migration.name}")

        try:
            # Execute rollback in transaction
            async with self.db_manager.pool.acquire() as connection:
                async with connection.transaction():
                    await connection.execute(migration.down_sql)

            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            # Update migration status
            await self._record_migration_status(
                migration, MigrationStatus.ROLLED_BACK, end_time, execution_time
            )

            logger.info(f"Migration {version} rolled back successfully in {execution_time:.2f}s")

            return MigrationResult(
                version=version,
                status=MigrationStatus.ROLLED_BACK,
                execution_time_seconds=execution_time,
                applied_at=end_time
            )

        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()

            error_msg = f"Rollback of migration {version} failed: {e}"
            logger.error(error_msg)

            return MigrationResult(
                version=version,
                status=MigrationStatus.FAILED,
                execution_time_seconds=execution_time,
                error_message=error_msg,
                applied_at=end_time
            )

    async def migrate_up(self) -> List[MigrationResult]:
        """Apply all pending migrations.

        Returns:
            List of migration results
        """
        pending_migrations = await self.get_pending_migrations()

        if not pending_migrations:
            logger.info("No pending migrations to apply")
            return []

        logger.info(f"Applying {len(pending_migrations)} pending migrations")
        results = []

        for migration in pending_migrations:
            result = await self.apply_migration(migration)
            results.append(result)

            # Stop if migration failed
            if result.status == MigrationStatus.FAILED:
                logger.error(f"Migration {migration.version} failed, stopping migration")
                break

        successful_count = sum(1 for r in results if r.status == MigrationStatus.COMPLETED)
        logger.info(f"Migration batch completed: {successful_count}/{len(results)} successful")

        return results

    async def get_migration_history(self) -> List[Dict[str, Any]]:
        """Get complete migration history.

        Returns:
            List of migration records
        """
        async with self.db_manager.pool.acquire() as connection:
            rows = await connection.fetch(
                f"""
                SELECT * FROM {self.migrations_table}
                ORDER BY version, applied_at DESC
                """
            )

            return [dict(row) for row in rows]

    async def _record_migration_status(
        self,
        migration: Migration,
        status: MigrationStatus,
        timestamp: datetime,
        execution_time: float = 0.0,
        error_message: str = None
    ) -> None:
        """Record migration status in tracking table.

        Args:
            migration: Migration instance
            status: Migration status
            timestamp: Status change timestamp
            execution_time: Execution time in seconds
            error_message: Error message if failed
        """
        async with self.db_manager.pool.acquire() as connection:
            await connection.execute(
                f"""
                INSERT INTO {self.migrations_table} (
                    version, name, description, status, applied_at,
                    execution_time_seconds, error_message, rollback_sql
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (version) DO UPDATE SET
                    status = EXCLUDED.status,
                    applied_at = EXCLUDED.applied_at,
                    execution_time_seconds = EXCLUDED.execution_time_seconds,
                    error_message = EXCLUDED.error_message
                """,
                migration.version,
                migration.name,
                migration.description,
                status.value,
                timestamp,
                execution_time,
                error_message,
                migration.down_sql
            )

    async def health_check(self) -> Dict[str, Any]:
        """Perform migration system health check.

        Returns:
            Health check results
        """
        try:
            pending = await self.get_pending_migrations()
            history = await self.get_migration_history()

            completed_migrations = [h for h in history if h['status'] == 'completed']
            failed_migrations = [h for h in history if h['status'] == 'failed']

            return {
                'status': 'healthy',
                'total_migrations': len(self._migrations),
                'pending_migrations': len(pending),
                'completed_migrations': len(completed_migrations),
                'failed_migrations': len(failed_migrations),
                'latest_migration': completed_migrations[-1]['version'] if completed_migrations else None,
                'migrations_table_exists': True,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


# Factory function for easy instantiation
async def create_migration_manager(db_manager: DatabaseManager) -> MigrationManager:
    """Create and initialize a migration manager.

    Args:
        db_manager: Database manager instance

    Returns:
        Initialized migration manager
    """
    manager = MigrationManager(db_manager)
    await manager.initialize()
    return manager