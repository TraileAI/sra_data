"""Database performance optimization and monitoring for SRA Data processing.

This module provides advanced performance optimization including query analysis,
index recommendations, connection tuning, and performance monitoring.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
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


class OptimizationLevel(str, Enum):
    """Levels of database optimization."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


@dataclass
class QueryPerformanceMetric:
    """Performance metrics for a database query."""
    query_hash: str
    query_text: str
    calls: int
    total_time: float
    mean_time: float
    min_time: float
    max_time: float
    stddev_time: float
    rows_returned: int
    cache_hit_ratio: float = 0.0
    recommendation: Optional[str] = None


@dataclass
class IndexRecommendation:
    """Index creation recommendation."""
    table_name: str
    columns: List[str]
    index_type: str = "btree"
    reason: str = ""
    estimated_benefit: str = "medium"
    create_sql: str = ""
    estimated_size_mb: float = 0.0


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    overall_health_score: float = 0.0
    slow_queries: List[QueryPerformanceMetric] = field(default_factory=list)
    index_recommendations: List[IndexRecommendation] = field(default_factory=list)
    connection_metrics: Dict[str, Any] = field(default_factory=dict)
    table_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimization_suggestions: List[str] = field(default_factory=list)


class PerformanceOptimizer:
    """Advanced database performance optimization and monitoring."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize performance optimizer.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self._performance_baseline = {}

    async def analyze_performance(self, include_recommendations: bool = True) -> PerformanceReport:
        """Perform comprehensive performance analysis.

        Args:
            include_recommendations: Whether to include optimization recommendations

        Returns:
            Detailed performance report
        """
        logger.info("Starting comprehensive performance analysis...")
        report = PerformanceReport()

        try:
            # Analyze slow queries
            report.slow_queries = await self._analyze_slow_queries()

            # Get connection and database metrics
            report.connection_metrics = await self._get_connection_metrics()

            # Analyze table performance
            report.table_metrics = await self._analyze_table_performance()

            # Generate index recommendations if requested
            if include_recommendations:
                report.index_recommendations = await self._generate_index_recommendations()

            # Calculate overall health score
            report.overall_health_score = await self._calculate_health_score(report)

            # Generate optimization suggestions
            report.optimization_suggestions = self._generate_optimization_suggestions(report)

            logger.info(f"Performance analysis completed. Health score: {report.overall_health_score:.1f}/100")

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            raise

        return report

    async def _analyze_slow_queries(self, limit: int = 20) -> List[QueryPerformanceMetric]:
        """Analyze slow-running queries using pg_stat_statements."""
        try:
            async with self.db_manager.pool.acquire() as connection:
                # Check if pg_stat_statements is available
                extension_exists = await connection.fetchval("""
                    SELECT EXISTS (
                        SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                    )
                """)

                if not extension_exists:
                    logger.warning("pg_stat_statements extension not available, using alternative analysis")
                    return await self._analyze_slow_queries_alternative()

                # Get slow queries from pg_stat_statements
                rows = await connection.fetch("""
                    SELECT
                        queryid::text as query_hash,
                        LEFT(query, 200) as query_text,
                        calls,
                        total_exec_time as total_time,
                        mean_exec_time as mean_time,
                        min_exec_time as min_time,
                        max_exec_time as max_time,
                        stddev_exec_time as stddev_time,
                        rows as rows_returned,
                        -- Calculate cache hit ratio
                        CASE
                            WHEN (shared_blks_hit + shared_blks_read) > 0
                            THEN shared_blks_hit::numeric / (shared_blks_hit + shared_blks_read) * 100
                            ELSE 0
                        END as cache_hit_ratio
                    FROM pg_stat_statements
                    WHERE query IS NOT NULL
                    AND query NOT LIKE '%pg_stat_statements%'
                    AND calls > 1
                    ORDER BY mean_exec_time DESC
                    LIMIT $1
                """, limit)

                metrics = []
                for row in rows:
                    metric = QueryPerformanceMetric(
                        query_hash=row['query_hash'],
                        query_text=row['query_text'],
                        calls=row['calls'],
                        total_time=row['total_time'],
                        mean_time=row['mean_time'],
                        min_time=row['min_time'],
                        max_time=row['max_time'],
                        stddev_time=row['stddev_time'] or 0.0,
                        rows_returned=row['rows_returned'],
                        cache_hit_ratio=row['cache_hit_ratio'],
                        recommendation=self._get_query_recommendation(row)
                    )
                    metrics.append(metric)

                return metrics

        except Exception as e:
            logger.error(f"Failed to analyze slow queries: {e}")
            return []

    async def _analyze_slow_queries_alternative(self) -> List[QueryPerformanceMetric]:
        """Alternative slow query analysis when pg_stat_statements is not available."""
        # Return empty list for now - could implement log file analysis
        logger.info("Using basic query analysis (pg_stat_statements not available)")
        return []

    def _get_query_recommendation(self, query_row: Dict[str, Any]) -> str:
        """Generate recommendation for a slow query."""
        mean_time = query_row['mean_time']
        cache_hit_ratio = query_row['cache_hit_ratio']
        query_text = query_row['query_text'].lower()

        recommendations = []

        if mean_time > 1000:  # > 1 second
            recommendations.append("Consider query optimization or indexing")

        if cache_hit_ratio < 90:
            recommendations.append("Poor cache hit ratio - consider indexing")

        if 'order by' in query_text and 'limit' not in query_text:
            recommendations.append("Consider adding LIMIT clause to ORDER BY queries")

        if 'ilike' in query_text or 'like' in query_text:
            recommendations.append("Consider full-text search or trigram indexes for LIKE queries")

        if query_text.count('join') > 3:
            recommendations.append("Complex joins detected - verify index coverage")

        return "; ".join(recommendations) if recommendations else "Query performance acceptable"

    async def _get_connection_metrics(self) -> Dict[str, Any]:
        """Get database connection and server metrics."""
        try:
            async with self.db_manager.pool.acquire() as connection:
                # Connection statistics
                conn_stats = await connection.fetchrow("""
                    SELECT
                        numbackends as active_connections,
                        xact_commit as transactions_committed,
                        xact_rollback as transactions_rolled_back,
                        blks_read as blocks_read,
                        blks_hit as blocks_hit,
                        tup_returned as tuples_returned,
                        tup_fetched as tuples_fetched,
                        tup_inserted as tuples_inserted,
                        tup_updated as tuples_updated,
                        tup_deleted as tuples_deleted
                    FROM pg_stat_database
                    WHERE datname = current_database()
                """)

                # Server configuration
                server_config = await connection.fetch("""
                    SELECT name, setting, unit
                    FROM pg_settings
                    WHERE name IN (
                        'max_connections', 'shared_buffers', 'effective_cache_size',
                        'work_mem', 'maintenance_work_mem', 'checkpoint_completion_target',
                        'wal_buffers', 'default_statistics_target'
                    )
                """)

                # Calculate cache hit ratio
                cache_hit_ratio = 0.0
                if conn_stats and (conn_stats['blocks_hit'] + conn_stats['blocks_read']) > 0:
                    cache_hit_ratio = (
                        conn_stats['blocks_hit'] /
                        (conn_stats['blocks_hit'] + conn_stats['blocks_read']) * 100
                    )

                return {
                    'connection_stats': dict(conn_stats) if conn_stats else {},
                    'server_config': {row['name']: {'value': row['setting'], 'unit': row['unit']}
                                    for row in server_config},
                    'cache_hit_ratio': cache_hit_ratio,
                    'commit_ratio': (
                        conn_stats['transactions_committed'] /
                        max(conn_stats['transactions_committed'] + conn_stats['transactions_rolled_back'], 1) * 100
                        if conn_stats else 0
                    )
                }

        except Exception as e:
            logger.error(f"Failed to get connection metrics: {e}")
            return {}

    async def _analyze_table_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance metrics for each table."""
        tables = ['equity_profile', 'fundata_data', 'fundata_quotes']
        table_metrics = {}

        try:
            async with self.db_manager.pool.acquire() as connection:
                for table_name in tables:
                    # Get table statistics
                    stats = await connection.fetchrow("""
                        SELECT
                            n_tup_ins as inserts,
                            n_tup_upd as updates,
                            n_tup_del as deletes,
                            n_live_tup as live_tuples,
                            n_dead_tup as dead_tuples,
                            seq_scan as sequential_scans,
                            seq_tup_read as seq_tuples_read,
                            idx_scan as index_scans,
                            idx_tup_fetch as index_tuples_fetched,
                            last_vacuum,
                            last_autovacuum,
                            last_analyze,
                            last_autoanalyze
                        FROM pg_stat_user_tables
                        WHERE relname = $1
                    """, table_name)

                    # Get I/O statistics
                    io_stats = await connection.fetchrow("""
                        SELECT
                            heap_blks_read,
                            heap_blks_hit,
                            idx_blks_read,
                            idx_blks_hit
                        FROM pg_statio_user_tables
                        WHERE relname = $1
                    """, table_name)

                    # Get table size
                    size_info = await connection.fetchrow("""
                        SELECT
                            pg_total_relation_size($1) as total_size_bytes,
                            pg_relation_size($1) as table_size_bytes
                    """, table_name)

                    # Calculate metrics
                    table_cache_hit_ratio = 0.0
                    if io_stats and (io_stats['heap_blks_hit'] + io_stats['heap_blks_read']) > 0:
                        table_cache_hit_ratio = (
                            io_stats['heap_blks_hit'] /
                            (io_stats['heap_blks_hit'] + io_stats['heap_blks_read']) * 100
                        )

                    index_usage_ratio = 0.0
                    if stats and (stats['sequential_scans'] + stats['index_scans']) > 0:
                        index_usage_ratio = (
                            stats['index_scans'] /
                            (stats['sequential_scans'] + stats['index_scans']) * 100
                        )

                    dead_tuple_ratio = 0.0
                    if stats and stats['live_tuples'] > 0:
                        dead_tuple_ratio = (stats['dead_tuples'] / stats['live_tuples']) * 100

                    table_metrics[table_name] = {
                        'stats': dict(stats) if stats else {},
                        'io_stats': dict(io_stats) if io_stats else {},
                        'size_info': dict(size_info) if size_info else {},
                        'cache_hit_ratio': table_cache_hit_ratio,
                        'index_usage_ratio': index_usage_ratio,
                        'dead_tuple_ratio': dead_tuple_ratio,
                        'needs_vacuum': dead_tuple_ratio > 10,
                        'needs_analyze': self._needs_analyze(stats)
                    }

        except Exception as e:
            logger.error(f"Failed to analyze table performance: {e}")

        return table_metrics

    def _needs_analyze(self, stats: Optional[Dict[str, Any]]) -> bool:
        """Determine if table needs ANALYZE based on statistics."""
        if not stats:
            return True

        last_analyze = stats.get('last_analyze')
        last_autoanalyze = stats.get('last_autoanalyze')

        # Check if analyzed in last 7 days
        recent_threshold = datetime.utcnow() - timedelta(days=7)

        if last_analyze and last_analyze > recent_threshold:
            return False
        if last_autoanalyze and last_autoanalyze > recent_threshold:
            return False

        return True

    async def _generate_index_recommendations(self) -> List[IndexRecommendation]:
        """Generate index recommendations based on query patterns and table analysis."""
        recommendations = []

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Check for missing indexes on foreign key columns
                missing_fk_indexes = await connection.fetch("""
                    SELECT
                        t.table_name,
                        t.column_name,
                        'btree' as index_type,
                        'Missing index on foreign key column' as reason
                    FROM (
                        SELECT
                            kcu.table_name,
                            kcu.column_name
                        FROM information_schema.key_column_usage kcu
                        JOIN information_schema.table_constraints tc ON kcu.constraint_name = tc.constraint_name
                        WHERE tc.constraint_type = 'FOREIGN KEY'
                        AND kcu.table_schema = 'public'
                    ) t
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM pg_indexes i
                        WHERE i.tablename = t.table_name
                        AND i.indexdef LIKE '%' || t.column_name || '%'
                    )
                """)

                for row in missing_fk_indexes:
                    rec = IndexRecommendation(
                        table_name=row['table_name'],
                        columns=[row['column_name']],
                        index_type=row['index_type'],
                        reason=row['reason'],
                        estimated_benefit="high",
                        create_sql=f"CREATE INDEX CONCURRENTLY idx_{row['table_name']}_{row['column_name']} ON {row['table_name']}({row['column_name']});",
                        estimated_size_mb=10.0  # Rough estimate
                    )
                    recommendations.append(rec)

                # Recommend indexes for frequent WHERE clauses
                frequent_where_columns = [
                    ('equity_profile', 'sector', 'Frequently filtered column'),
                    ('equity_profile', 'industry', 'Frequently filtered column'),
                    ('fundata_data', 'family_name', 'Frequently searched column'),
                    ('fundata_quotes', 'date', 'Time-series queries'),
                ]

                for table_name, column_name, reason in frequent_where_columns:
                    # Check if index exists
                    index_exists = await connection.fetchval("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_indexes
                            WHERE tablename = $1
                            AND indexdef LIKE '%' || $2 || '%'
                        )
                    """, table_name, column_name)

                    if not index_exists:
                        rec = IndexRecommendation(
                            table_name=table_name,
                            columns=[column_name],
                            reason=reason,
                            estimated_benefit="medium",
                            create_sql=f"CREATE INDEX CONCURRENTLY idx_{table_name}_{column_name} ON {table_name}({column_name});",
                            estimated_size_mb=5.0
                        )
                        recommendations.append(rec)

                # Recommend partial indexes for active records
                partial_index_candidates = [
                    ('equity_profile', "is_actively_trading = TRUE", 'Active equities filter'),
                    ('fundata_data', "record_state = 'Active'", 'Active funds filter'),
                    ('fundata_quotes', "record_state = 'Active'", 'Active quotes filter'),
                ]

                for table_name, condition, reason in partial_index_candidates:
                    index_name = f"idx_{table_name}_active"
                    index_exists = await connection.fetchval("""
                        SELECT EXISTS (
                            SELECT 1 FROM pg_indexes
                            WHERE tablename = $1 AND indexname = $2
                        )
                    """, table_name, index_name)

                    if not index_exists:
                        rec = IndexRecommendation(
                            table_name=table_name,
                            columns=['record_state' if 'record_state' in condition else 'is_actively_trading'],
                            reason=f"Partial index for {reason}",
                            estimated_benefit="high",
                            create_sql=f"CREATE INDEX CONCURRENTLY {index_name} ON {table_name}(record_state) WHERE {condition};",
                            estimated_size_mb=2.0
                        )
                        recommendations.append(rec)

        except Exception as e:
            logger.error(f"Failed to generate index recommendations: {e}")

        return recommendations

    async def _calculate_health_score(self, report: PerformanceReport) -> float:
        """Calculate overall database health score (0-100)."""
        score = 100.0
        penalties = []

        # Penalize for slow queries
        if report.slow_queries:
            avg_query_time = sum(q.mean_time for q in report.slow_queries) / len(report.slow_queries)
            if avg_query_time > 1000:  # > 1 second
                penalty = min(30, avg_query_time / 100)
                score -= penalty
                penalties.append(f"Slow queries detected (avg: {avg_query_time:.1f}ms)")

        # Penalize for low cache hit ratio
        cache_hit_ratio = report.connection_metrics.get('cache_hit_ratio', 100)
        if cache_hit_ratio < 90:
            penalty = (90 - cache_hit_ratio) * 2
            score -= penalty
            penalties.append(f"Low cache hit ratio ({cache_hit_ratio:.1f}%)")

        # Penalize for tables needing maintenance
        tables_needing_vacuum = sum(
            1 for metrics in report.table_metrics.values()
            if metrics.get('needs_vacuum', False)
        )
        if tables_needing_vacuum > 0:
            penalty = tables_needing_vacuum * 10
            score -= penalty
            penalties.append(f"{tables_needing_vacuum} tables need vacuum")

        # Penalize for missing indexes
        if len(report.index_recommendations) > 5:
            penalty = min(20, len(report.index_recommendations) * 2)
            score -= penalty
            penalties.append(f"{len(report.index_recommendations)} index recommendations")

        logger.info(f"Health score calculation: {score:.1f}/100. Penalties: {penalties}")
        return max(0, score)

    def _generate_optimization_suggestions(self, report: PerformanceReport) -> List[str]:
        """Generate actionable optimization suggestions."""
        suggestions = []

        # Slow query suggestions
        if report.slow_queries:
            slow_count = len([q for q in report.slow_queries if q.mean_time > 1000])
            if slow_count > 0:
                suggestions.append(f"Optimize {slow_count} slow queries (>1s average)")

        # Cache optimization
        cache_ratio = report.connection_metrics.get('cache_hit_ratio', 100)
        if cache_ratio < 95:
            suggestions.append(f"Improve cache hit ratio ({cache_ratio:.1f}% - target >95%)")

        # Table maintenance
        vacuum_needed = sum(
            1 for metrics in report.table_metrics.values()
            if metrics.get('needs_vacuum', False)
        )
        if vacuum_needed > 0:
            suggestions.append(f"Run VACUUM on {vacuum_needed} tables with high dead tuple ratio")

        analyze_needed = sum(
            1 for metrics in report.table_metrics.values()
            if metrics.get('needs_analyze', False)
        )
        if analyze_needed > 0:
            suggestions.append(f"Run ANALYZE on {analyze_needed} tables with outdated statistics")

        # Index recommendations
        high_impact_indexes = [
            idx for idx in report.index_recommendations
            if idx.estimated_benefit == "high"
        ]
        if high_impact_indexes:
            suggestions.append(f"Create {len(high_impact_indexes)} high-impact indexes")

        # Connection tuning
        active_conn = report.connection_metrics.get('connection_stats', {}).get('active_connections', 0)
        if active_conn > 80:  # Assuming max_connections around 100
            suggestions.append("Consider connection pooling - high active connection count")

        return suggestions

    async def optimize_automatically(self, level: OptimizationLevel = OptimizationLevel.STANDARD) -> Dict[str, Any]:
        """Perform automatic optimization based on analysis.

        Args:
            level: Level of optimization to perform

        Returns:
            Results of optimization operations
        """
        logger.info(f"Starting automatic optimization (level: {level.value})...")
        results = {
            'level': level.value,
            'started_at': datetime.utcnow(),
            'operations': [],
            'errors': []
        }

        try:
            # Get performance report
            report = await self.analyze_performance()

            # Perform optimizations based on level
            if level in [OptimizationLevel.BASIC, OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
                # Basic: Update table statistics
                for table_name, metrics in report.table_metrics.items():
                    if metrics.get('needs_analyze', False):
                        try:
                            async with self.db_manager.pool.acquire() as connection:
                                await connection.execute(f"ANALYZE {table_name}")
                            results['operations'].append(f"Analyzed table {table_name}")
                        except Exception as e:
                            results['errors'].append(f"Failed to analyze {table_name}: {e}")

            if level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
                # Standard: Create high-impact indexes
                high_impact_indexes = [
                    idx for idx in report.index_recommendations
                    if idx.estimated_benefit == "high"
                ]

                for idx in high_impact_indexes[:3]:  # Limit to 3 indexes per run
                    try:
                        async with self.db_manager.pool.acquire() as connection:
                            await connection.execute(idx.create_sql)
                        results['operations'].append(f"Created index on {idx.table_name}({', '.join(idx.columns)})")
                    except Exception as e:
                        results['errors'].append(f"Failed to create index on {idx.table_name}: {e}")

            if level == OptimizationLevel.AGGRESSIVE:
                # Aggressive: Vacuum tables with high dead tuple ratio
                for table_name, metrics in report.table_metrics.items():
                    if metrics.get('needs_vacuum', False):
                        try:
                            async with self.db_manager.pool.acquire() as connection:
                                await connection.execute(f"VACUUM ANALYZE {table_name}")
                            results['operations'].append(f"Vacuumed table {table_name}")
                        except Exception as e:
                            results['errors'].append(f"Failed to vacuum {table_name}: {e}")

            results['completed_at'] = datetime.utcnow()
            results['duration_seconds'] = (results['completed_at'] - results['started_at']).total_seconds()

            logger.info(
                f"Automatic optimization completed: {len(results['operations'])} operations, "
                f"{len(results['errors'])} errors in {results['duration_seconds']:.1f}s"
            )

        except Exception as e:
            results['errors'].append(f"Optimization failed: {e}")
            logger.error(f"Automatic optimization failed: {e}")

        return results

    async def monitor_performance_trends(self, hours_back: int = 24) -> Dict[str, Any]:
        """Monitor performance trends over time.

        Args:
            hours_back: Hours of history to analyze

        Returns:
            Performance trend analysis
        """
        # This is a placeholder for trend monitoring
        # In a real implementation, you'd store historical metrics
        logger.info(f"Monitoring performance trends for last {hours_back} hours")

        return {
            'period_hours': hours_back,
            'trend_analysis': 'Placeholder for trend monitoring',
            'recommendations': ['Implement historical metrics storage for trend analysis'],
            'timestamp': datetime.utcnow().isoformat()
        }