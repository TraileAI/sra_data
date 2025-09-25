"""Table management utilities for SRA Data processing service.

This module provides advanced table management functionality including
partitioning, maintenance, statistics, and performance optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date, timedelta
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


class PartitionType(str, Enum):
    """Supported table partitioning types."""
    RANGE = "RANGE"
    LIST = "LIST"
    HASH = "HASH"


class TableMaintenanceResult:
    """Result of table maintenance operation."""

    def __init__(self, table_name: str, operation: str):
        self.table_name = table_name
        self.operation = operation
        self.success = True
        self.details = {}
        self.duration_seconds = 0.0
        self.error_message = None
        self.timestamp = datetime.utcnow()


class TableManager:
    """Advanced table management with partitioning and optimization."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize table manager.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    async def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table information.

        Args:
            table_name: Name of table to analyze

        Returns:
            Dictionary with table metadata
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                # Basic table information
                table_info = await connection.fetchrow("""
                    SELECT
                        schemaname,
                        tablename,
                        tableowner,
                        hasindexes,
                        hasrules,
                        hastriggers
                    FROM pg_tables
                    WHERE tablename = $1
                """, table_name)

                if not table_info:
                    return {'exists': False}

                # Table size information
                size_info = await connection.fetchrow("""
                    SELECT
                        pg_size_pretty(pg_total_relation_size($1)) as total_size,
                        pg_size_pretty(pg_relation_size($1)) as table_size,
                        pg_size_pretty(pg_total_relation_size($1) - pg_relation_size($1)) as indexes_size
                """, table_name)

                # Row count estimation
                row_count = await connection.fetchval("""
                    SELECT reltuples::bigint
                    FROM pg_class
                    WHERE relname = $1
                """, table_name)

                # Column information
                columns = await connection.fetch("""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default,
                        character_maximum_length
                    FROM information_schema.columns
                    WHERE table_name = $1
                    ORDER BY ordinal_position
                """, table_name)

                # Index information
                indexes = await connection.fetch("""
                    SELECT
                        indexname,
                        indexdef
                    FROM pg_indexes
                    WHERE tablename = $1
                """, table_name)

                # Constraints information
                constraints = await connection.fetch("""
                    SELECT
                        conname,
                        contype,
                        pg_get_constraintdef(c.oid) as definition
                    FROM pg_constraint c
                    JOIN pg_class t ON c.conrelid = t.oid
                    WHERE t.relname = $1
                """, table_name)

                return {
                    'exists': True,
                    'basic_info': dict(table_info),
                    'size_info': dict(size_info),
                    'estimated_rows': row_count,
                    'columns': [dict(col) for col in columns],
                    'indexes': [dict(idx) for idx in indexes],
                    'constraints': [dict(const) for const in constraints],
                    'analyzed_at': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            return {'exists': False, 'error': str(e)}

    async def analyze_table(self, table_name: str) -> TableMaintenanceResult:
        """Update table statistics for query optimization.

        Args:
            table_name: Name of table to analyze

        Returns:
            Maintenance result
        """
        result = TableMaintenanceResult(table_name, "ANALYZE")
        start_time = datetime.utcnow()

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Get pre-analyze stats
                pre_stats = await connection.fetchrow("""
                    SELECT
                        reltuples,
                        relpages,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE relname = $1
                """, table_name)

                # Run ANALYZE
                await connection.execute(f"ANALYZE {table_name}")

                # Get post-analyze stats
                post_stats = await connection.fetchrow("""
                    SELECT
                        reltuples,
                        relpages,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE relname = $1
                """, table_name)

                result.details = {
                    'pre_analyze': dict(pre_stats) if pre_stats else {},
                    'post_analyze': dict(post_stats) if post_stats else {},
                    'rows_analyzed': int(post_stats['reltuples']) if post_stats else 0
                }

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Failed to analyze table {table_name}: {e}")

        finally:
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        return result

    async def vacuum_table(self, table_name: str, full: bool = False) -> TableMaintenanceResult:
        """Vacuum table to reclaim space and update statistics.

        Args:
            table_name: Name of table to vacuum
            full: Whether to perform VACUUM FULL (blocks table access)

        Returns:
            Maintenance result
        """
        result = TableMaintenanceResult(table_name, "VACUUM FULL" if full else "VACUUM")
        start_time = datetime.utcnow()

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Get pre-vacuum size
                pre_size = await connection.fetchval(
                    "SELECT pg_total_relation_size($1)", table_name
                )

                # Run vacuum
                vacuum_cmd = f"VACUUM {'FULL ' if full else ''}ANALYZE {table_name}"
                await connection.execute(vacuum_cmd)

                # Get post-vacuum size
                post_size = await connection.fetchval(
                    "SELECT pg_total_relation_size($1)", table_name
                )

                result.details = {
                    'vacuum_type': 'FULL' if full else 'STANDARD',
                    'size_before_bytes': pre_size,
                    'size_after_bytes': post_size,
                    'space_reclaimed_bytes': pre_size - post_size,
                    'space_reclaimed_mb': round((pre_size - post_size) / (1024 * 1024), 2)
                }

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Failed to vacuum table {table_name}: {e}")

        finally:
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        return result

    async def reindex_table(self, table_name: str, concurrently: bool = True) -> TableMaintenanceResult:
        """Rebuild all indexes on a table.

        Args:
            table_name: Name of table to reindex
            concurrently: Whether to rebuild indexes concurrently (non-blocking)

        Returns:
            Maintenance result
        """
        result = TableMaintenanceResult(table_name, "REINDEX")
        start_time = datetime.utcnow()

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Get index information before reindex
                indexes_info = await connection.fetch("""
                    SELECT
                        indexname,
                        pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size
                    FROM pg_indexes
                    WHERE tablename = $1
                """, table_name)

                if concurrently:
                    # Rebuild each index individually with CONCURRENTLY
                    rebuilt_indexes = []
                    for index_info in indexes_info:
                        index_name = index_info['indexname']
                        try:
                            await connection.execute(f"REINDEX INDEX CONCURRENTLY {index_name}")
                            rebuilt_indexes.append(index_name)
                        except Exception as e:
                            logger.warning(f"Failed to reindex {index_name}: {e}")

                    result.details['rebuilt_indexes'] = rebuilt_indexes
                else:
                    # Use REINDEX TABLE (blocking)
                    await connection.execute(f"REINDEX TABLE {table_name}")
                    result.details['rebuilt_indexes'] = [idx['indexname'] for idx in indexes_info]

                result.details['total_indexes'] = len(indexes_info)
                result.details['indexes_info'] = [dict(idx) for idx in indexes_info]

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Failed to reindex table {table_name}: {e}")

        finally:
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        return result

    async def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive table usage statistics.

        Args:
            table_name: Name of table to analyze

        Returns:
            Dictionary with table statistics
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                # Basic stats
                basic_stats = await connection.fetchrow("""
                    SELECT
                        schemaname,
                        relname,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch,
                        n_tup_ins,
                        n_tup_upd,
                        n_tup_del,
                        n_tup_hot_upd,
                        n_live_tup,
                        n_dead_tup,
                        last_vacuum,
                        last_autovacuum,
                        last_analyze,
                        last_autoanalyze,
                        vacuum_count,
                        autovacuum_count,
                        analyze_count,
                        autoanalyze_count
                    FROM pg_stat_user_tables
                    WHERE relname = $1
                """, table_name)

                # I/O stats
                io_stats = await connection.fetchrow("""
                    SELECT
                        heap_blks_read,
                        heap_blks_hit,
                        idx_blks_read,
                        idx_blks_hit,
                        toast_blks_read,
                        toast_blks_hit,
                        tidx_blks_read,
                        tidx_blks_hit
                    FROM pg_statio_user_tables
                    WHERE relname = $1
                """, table_name)

                # Size information
                size_stats = await connection.fetchrow("""
                    SELECT
                        pg_size_pretty(pg_total_relation_size($1)) as total_size,
                        pg_size_pretty(pg_relation_size($1)) as table_size,
                        pg_size_pretty(pg_total_relation_size($1) - pg_relation_size($1)) as indexes_size,
                        pg_total_relation_size($1) as total_size_bytes,
                        pg_relation_size($1) as table_size_bytes
                """, table_name)

                # Index usage stats
                index_stats = await connection.fetch("""
                    SELECT
                        indexrelname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE relname = $1
                    ORDER BY idx_scan DESC
                """, table_name)

                # Calculate cache hit ratios
                cache_hit_ratio = 0.0
                if io_stats and (io_stats['heap_blks_read'] + io_stats['heap_blks_hit']) > 0:
                    cache_hit_ratio = (
                        io_stats['heap_blks_hit'] /
                        (io_stats['heap_blks_read'] + io_stats['heap_blks_hit']) * 100
                    )

                return {
                    'basic_stats': dict(basic_stats) if basic_stats else {},
                    'io_stats': dict(io_stats) if io_stats else {},
                    'size_stats': dict(size_stats) if size_stats else {},
                    'index_stats': [dict(idx) for idx in index_stats],
                    'cache_hit_ratio_percent': round(cache_hit_ratio, 2),
                    'collected_at': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get table statistics for {table_name}: {e}")
            return {'error': str(e)}

    async def create_partition(
        self,
        parent_table: str,
        partition_name: str,
        partition_type: PartitionType,
        partition_key: str,
        partition_bounds: str
    ) -> TableMaintenanceResult:
        """Create a table partition.

        Args:
            parent_table: Parent table name
            partition_name: Name for the partition
            partition_type: Type of partitioning
            partition_key: Column(s) to partition on
            partition_bounds: Partition bounds (e.g., "FROM ('2023-01-01') TO ('2023-02-01')")

        Returns:
            Maintenance result
        """
        result = TableMaintenanceResult(partition_name, "CREATE_PARTITION")
        start_time = datetime.utcnow()

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Create partition table
                partition_sql = f"""
                CREATE TABLE {partition_name} PARTITION OF {parent_table}
                FOR VALUES {partition_bounds}
                """

                await connection.execute(partition_sql)

                # Create indexes on partition (inherit from parent)
                parent_indexes = await connection.fetch("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = $1
                    AND indexdef NOT LIKE '%UNIQUE%'
                """, parent_table)

                created_indexes = []
                for index_info in parent_indexes:
                    try:
                        # Modify index definition for partition
                        index_def = index_info['indexdef'].replace(
                            f" ON {parent_table} ",
                            f" ON {partition_name} "
                        )
                        # Generate unique index name
                        new_index_name = f"{partition_name}_{index_info['indexname'].split('_', 1)[-1]}"
                        index_def = index_def.replace(
                            f"INDEX {index_info['indexname']}",
                            f"INDEX {new_index_name}"
                        )

                        await connection.execute(index_def)
                        created_indexes.append(new_index_name)
                    except Exception as e:
                        logger.warning(f"Failed to create index on partition: {e}")

                result.details = {
                    'parent_table': parent_table,
                    'partition_type': partition_type.value,
                    'partition_key': partition_key,
                    'partition_bounds': partition_bounds,
                    'indexes_created': created_indexes
                }

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Failed to create partition {partition_name}: {e}")

        finally:
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        return result

    async def optimize_table_for_bulk_operations(self, table_name: str) -> TableMaintenanceResult:
        """Optimize table configuration for bulk data operations.

        Args:
            table_name: Name of table to optimize

        Returns:
            Maintenance result
        """
        result = TableMaintenanceResult(table_name, "BULK_OPTIMIZE")
        start_time = datetime.utcnow()

        try:
            async with self.db_manager.pool.acquire() as connection:
                optimizations = []

                # Set table storage parameters for bulk operations
                storage_params = [
                    "ALTER TABLE {} SET (fillfactor = 85)".format(table_name),
                    "ALTER TABLE {} SET (autovacuum_vacuum_scale_factor = 0.1)".format(table_name),
                    "ALTER TABLE {} SET (autovacuum_analyze_scale_factor = 0.05)".format(table_name)
                ]

                for param_sql in storage_params:
                    try:
                        await connection.execute(param_sql)
                        optimizations.append(param_sql.split("SET")[1].strip())
                    except Exception as e:
                        logger.warning(f"Storage parameter failed: {e}")

                # Create partial indexes for common queries if they don't exist
                partial_indexes = [
                    {
                        'name': f"idx_{table_name}_active_records",
                        'sql': f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{table_name}_active_records ON {table_name}(record_state) WHERE record_state = 'Active'"
                    },
                    {
                        'name': f"idx_{table_name}_recent_processed",
                        'sql': f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{table_name}_recent_processed ON {table_name}(processed_at) WHERE processed_at >= CURRENT_DATE - INTERVAL '30 days'"
                    }
                ]

                created_indexes = []
                for index_info in partial_indexes:
                    try:
                        await connection.execute(index_info['sql'])
                        created_indexes.append(index_info['name'])
                    except Exception as e:
                        logger.warning(f"Partial index creation failed: {e}")

                result.details = {
                    'storage_optimizations': optimizations,
                    'partial_indexes_created': created_indexes,
                    'optimization_type': 'bulk_operations'
                }

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Failed to optimize table {table_name}: {e}")

        finally:
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        return result

    async def get_slow_queries(self, table_name: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slow queries affecting specific table or all tables.

        Args:
            table_name: Optional table name to filter queries
            limit: Maximum number of queries to return

        Returns:
            List of slow query information
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                base_query = """
                SELECT
                    query,
                    calls,
                    total_time,
                    mean_time,
                    min_time,
                    max_time,
                    stddev_time,
                    rows
                FROM pg_stat_statements
                WHERE query IS NOT NULL
                """

                params = []
                if table_name:
                    base_query += " AND query ILIKE $1"
                    params.append(f"%{table_name}%")

                base_query += " ORDER BY mean_time DESC LIMIT ${}".format(len(params) + 1)
                params.append(limit)

                rows = await connection.fetch(base_query, *params)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get slow queries: {e}")
            return []

    async def maintenance_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive maintenance health check.

        Returns:
            Health check results with maintenance recommendations
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                # Get tables needing maintenance
                tables_needing_vacuum = await connection.fetch("""
                    SELECT
                        relname,
                        n_dead_tup,
                        n_live_tup,
                        ROUND(n_dead_tup::numeric / GREATEST(n_live_tup, 1) * 100, 2) as dead_tuple_ratio
                    FROM pg_stat_user_tables
                    WHERE n_dead_tup > 1000
                    AND ROUND(n_dead_tup::numeric / GREATEST(n_live_tup, 1) * 100, 2) > 10
                    ORDER BY dead_tuple_ratio DESC
                """)

                tables_needing_analyze = await connection.fetch("""
                    SELECT
                        relname,
                        last_analyze,
                        last_autoanalyze
                    FROM pg_stat_user_tables
                    WHERE (last_analyze IS NULL OR last_analyze < CURRENT_TIMESTAMP - INTERVAL '7 days')
                    AND (last_autoanalyze IS NULL OR last_autoanalyze < CURRENT_TIMESTAMP - INTERVAL '7 days')
                """)

                # Check for unused indexes
                unused_indexes = await connection.fetch("""
                    SELECT
                        indexrelname,
                        relname,
                        idx_scan,
                        pg_size_pretty(pg_relation_size(indexrelid)) as index_size
                    FROM pg_stat_user_indexes
                    WHERE idx_scan < 10
                    AND pg_relation_size(indexrelid) > 1024 * 1024  -- > 1MB
                    ORDER BY pg_relation_size(indexrelid) DESC
                """)

                return {
                    'status': 'healthy',
                    'tables_needing_vacuum': [dict(row) for row in tables_needing_vacuum],
                    'tables_needing_analyze': [dict(row) for row in tables_needing_analyze],
                    'unused_indexes': [dict(row) for row in unused_indexes],
                    'recommendations': {
                        'vacuum_needed': len(tables_needing_vacuum) > 0,
                        'analyze_needed': len(tables_needing_analyze) > 0,
                        'index_cleanup_needed': len(unused_indexes) > 0
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }