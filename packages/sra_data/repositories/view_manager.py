"""Database view manager for modelized client data consumption.

This module provides materialized and regular views for efficient client
data access with pre-aggregated data, optimized queries, and caching.
"""

import logging
from typing import List, Dict, Any, Optional, Union
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


class ViewType(str, Enum):
    """Types of database views."""
    REGULAR = "VIEW"
    MATERIALIZED = "MATERIALIZED VIEW"


class ViewRefreshResult:
    """Result of view refresh operation."""

    def __init__(self, view_name: str):
        self.view_name = view_name
        self.success = True
        self.rows_affected = 0
        self.duration_seconds = 0.0
        self.error_message = None
        self.refreshed_at = datetime.utcnow()


class ViewManager:
    """Database view management for optimized client data access."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize view manager.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    async def initialize_views(self) -> None:
        """Initialize all predefined views for client consumption."""
        logger.info("Initializing client consumption views...")

        views_created = []

        try:
            # Create equity summary view
            await self._create_equity_summary_view()
            views_created.append("equity_summary")

            # Create fund performance view
            await self._create_fund_performance_view()
            views_created.append("fund_performance")

            # Create fund family overview view
            await self._create_fund_family_overview()
            views_created.append("fund_family_overview")

            # Create market statistics view
            await self._create_market_statistics_view()
            views_created.append("market_statistics")

            # Create data freshness view
            await self._create_data_freshness_view()
            views_created.append("data_freshness")

            # Create search optimization view
            await self._create_search_optimization_view()
            views_created.append("search_optimization")

            logger.info(f"Successfully created {len(views_created)} views: {views_created}")

        except Exception as e:
            logger.error(f"Failed to initialize views: {e}")
            raise

    async def _create_equity_summary_view(self) -> None:
        """Create equity profile summary view for client consumption."""
        view_sql = """
        CREATE OR REPLACE VIEW v_equity_summary AS
        SELECT
            ep.symbol,
            ep.company_name,
            ep.exchange,
            ep.sector,
            ep.industry,
            ep.market_cap,
            CASE
                WHEN ep.market_cap >= 200000000000 THEN 'Mega Cap'
                WHEN ep.market_cap >= 10000000000 THEN 'Large Cap'
                WHEN ep.market_cap >= 2000000000 THEN 'Mid Cap'
                WHEN ep.market_cap >= 300000000 THEN 'Small Cap'
                WHEN ep.market_cap >= 50000000 THEN 'Micro Cap'
                ELSE 'Nano Cap'
            END as market_cap_category,
            ep.employees,
            CASE
                WHEN ep.employees >= 100000 THEN 'Very Large'
                WHEN ep.employees >= 10000 THEN 'Large'
                WHEN ep.employees >= 1000 THEN 'Medium'
                WHEN ep.employees >= 100 THEN 'Small'
                ELSE 'Very Small'
            END as company_size,
            ep.country,
            ep.currency,
            ep.is_etf,
            ep.is_actively_trading,
            ep.created_at,
            ep.updated_at,
            -- Data quality indicators
            CASE
                WHEN ep.market_cap IS NOT NULL AND ep.sector IS NOT NULL
                     AND ep.industry IS NOT NULL AND ep.employees IS NOT NULL
                THEN 'Complete'
                WHEN ep.market_cap IS NOT NULL AND ep.sector IS NOT NULL
                THEN 'Good'
                WHEN ep.sector IS NOT NULL
                THEN 'Basic'
                ELSE 'Minimal'
            END as data_completeness,
            -- Search relevance score
            CASE
                WHEN ep.is_actively_trading AND ep.market_cap IS NOT NULL
                THEN 100
                WHEN ep.is_actively_trading
                THEN 75
                WHEN ep.market_cap IS NOT NULL
                THEN 50
                ELSE 25
            END as search_relevance_score
        FROM equity_profile ep
        WHERE ep.is_actively_trading = TRUE
        ORDER BY ep.market_cap DESC NULLS LAST, ep.symbol;

        -- Create index for common queries
        CREATE INDEX IF NOT EXISTS idx_v_equity_summary_exchange
        ON equity_profile(exchange) WHERE is_actively_trading = TRUE;

        CREATE INDEX IF NOT EXISTS idx_v_equity_summary_sector
        ON equity_profile(sector) WHERE is_actively_trading = TRUE;
        """

        async with self.db_manager.pool.acquire() as connection:
            await connection.execute(view_sql)

    async def _create_fund_performance_view(self) -> None:
        """Create materialized view for fund performance metrics."""
        view_sql = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_fund_performance AS
        SELECT
            fd.instrument_key,
            fd.legal_name,
            fd.family_name,
            fd.series_name,
            fd.company,
            fd.currency,
            fd.inception_date,
            -- Latest quote information
            latest.date as latest_quote_date,
            latest.navps as latest_navps,
            latest.navps_penny_change as latest_penny_change,
            latest.navps_percent_change as latest_percent_change,
            -- Performance metrics
            perf.quotes_count,
            perf.first_quote_date,
            perf.last_quote_date,
            perf.days_of_data,
            perf.avg_navps,
            perf.min_navps,
            perf.max_navps,
            perf.navps_volatility,
            -- Period returns (if enough data)
            perf.return_1d,
            perf.return_7d,
            perf.return_30d,
            perf.return_90d,
            perf.return_ytd,
            -- Data quality indicators
            CASE
                WHEN perf.quotes_count >= 252 THEN 'Excellent'  -- 1 year of daily data
                WHEN perf.quotes_count >= 90 THEN 'Good'        -- 3 months
                WHEN perf.quotes_count >= 30 THEN 'Fair'        -- 1 month
                ELSE 'Limited'
            END as data_quality,
            -- Fund classification
            CASE
                WHEN fd.legal_name ILIKE '%bond%' OR fd.legal_name ILIKE '%fixed income%' THEN 'Fixed Income'
                WHEN fd.legal_name ILIKE '%equity%' OR fd.legal_name ILIKE '%stock%' THEN 'Equity'
                WHEN fd.legal_name ILIKE '%balanced%' OR fd.legal_name ILIKE '%allocation%' THEN 'Balanced'
                WHEN fd.legal_name ILIKE '%money market%' THEN 'Money Market'
                WHEN fd.legal_name ILIKE '%alternative%' THEN 'Alternative'
                ELSE 'Other'
            END as fund_category,
            -- Freshness indicators
            CASE
                WHEN latest.date >= CURRENT_DATE - INTERVAL '3 days' THEN 'Fresh'
                WHEN latest.date >= CURRENT_DATE - INTERVAL '7 days' THEN 'Recent'
                WHEN latest.date >= CURRENT_DATE - INTERVAL '30 days' THEN 'Stale'
                ELSE 'Very Stale'
            END as data_freshness,
            fd.processed_at as fund_data_processed_at,
            latest.processed_at as latest_quote_processed_at
        FROM fundata_data fd
        INNER JOIN (
            -- Get latest quote for each fund
            SELECT DISTINCT ON (instrument_key)
                instrument_key,
                date,
                navps,
                navps_penny_change,
                navps_percent_change,
                processed_at,
                ROW_NUMBER() OVER (PARTITION BY instrument_key ORDER BY date DESC, processed_at DESC) as rn
            FROM fundata_quotes
            WHERE record_state = 'Active'
        ) latest ON fd.instrument_key = latest.instrument_key AND latest.rn = 1
        INNER JOIN (
            -- Calculate performance metrics
            SELECT
                instrument_key,
                COUNT(*) as quotes_count,
                MIN(date) as first_quote_date,
                MAX(date) as last_quote_date,
                MAX(date) - MIN(date) as days_of_data,
                AVG(navps) as avg_navps,
                MIN(navps) as min_navps,
                MAX(navps) as max_navps,
                STDDEV(navps) as navps_volatility,
                -- Calculate period returns
                (SELECT (latest_navps.navps - prev_1d.navps) / prev_1d.navps * 100
                 FROM fundata_quotes latest_navps
                 LEFT JOIN fundata_quotes prev_1d ON prev_1d.instrument_key = latest_navps.instrument_key
                     AND prev_1d.date = latest_navps.date - INTERVAL '1 day'
                 WHERE latest_navps.instrument_key = fq.instrument_key
                 ORDER BY latest_navps.date DESC LIMIT 1
                ) as return_1d,
                (SELECT (latest_navps.navps - prev_7d.navps) / prev_7d.navps * 100
                 FROM fundata_quotes latest_navps
                 LEFT JOIN fundata_quotes prev_7d ON prev_7d.instrument_key = latest_navps.instrument_key
                     AND prev_7d.date <= latest_navps.date - INTERVAL '7 days'
                 WHERE latest_navps.instrument_key = fq.instrument_key
                 ORDER BY latest_navps.date DESC, prev_7d.date DESC LIMIT 1
                ) as return_7d,
                (SELECT (latest_navps.navps - prev_30d.navps) / prev_30d.navps * 100
                 FROM fundata_quotes latest_navps
                 LEFT JOIN fundata_quotes prev_30d ON prev_30d.instrument_key = latest_navps.instrument_key
                     AND prev_30d.date <= latest_navps.date - INTERVAL '30 days'
                 WHERE latest_navps.instrument_key = fq.instrument_key
                 ORDER BY latest_navps.date DESC, prev_30d.date DESC LIMIT 1
                ) as return_30d,
                (SELECT (latest_navps.navps - prev_90d.navps) / prev_90d.navps * 100
                 FROM fundata_quotes latest_navps
                 LEFT JOIN fundata_quotes prev_90d ON prev_90d.instrument_key = latest_navps.instrument_key
                     AND prev_90d.date <= latest_navps.date - INTERVAL '90 days'
                 WHERE latest_navps.instrument_key = fq.instrument_key
                 ORDER BY latest_navps.date DESC, prev_90d.date DESC LIMIT 1
                ) as return_90d,
                (SELECT (latest_navps.navps - ytd_start.navps) / ytd_start.navps * 100
                 FROM fundata_quotes latest_navps
                 LEFT JOIN fundata_quotes ytd_start ON ytd_start.instrument_key = latest_navps.instrument_key
                     AND ytd_start.date >= DATE_TRUNC('year', CURRENT_DATE)
                 WHERE latest_navps.instrument_key = fq.instrument_key
                 ORDER BY latest_navps.date DESC, ytd_start.date ASC LIMIT 1
                ) as return_ytd
            FROM fundata_quotes fq
            WHERE record_state = 'Active'
            GROUP BY instrument_key
        ) perf ON fd.instrument_key = perf.instrument_key
        WHERE fd.record_state = 'Active';

        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_mv_fund_performance_family
        ON mv_fund_performance(family_name);

        CREATE INDEX IF NOT EXISTS idx_mv_fund_performance_category
        ON mv_fund_performance(fund_category);

        CREATE INDEX IF NOT EXISTS idx_mv_fund_performance_freshness
        ON mv_fund_performance(data_freshness);

        CREATE INDEX IF NOT EXISTS idx_mv_fund_performance_return_30d
        ON mv_fund_performance(return_30d DESC NULLS LAST);
        """

        async with self.db_manager.pool.acquire() as connection:
            await connection.execute(view_sql)

    async def _create_fund_family_overview(self) -> None:
        """Create fund family aggregation view."""
        view_sql = """
        CREATE OR REPLACE VIEW v_fund_family_overview AS
        SELECT
            fd.family_name,
            fd.company as management_company,
            COUNT(DISTINCT fd.instrument_key) as total_funds,
            COUNT(DISTINCT CASE WHEN latest_quotes.latest_date >= CURRENT_DATE - INTERVAL '7 days' THEN fd.instrument_key END) as active_funds,
            MIN(fd.inception_date) as earliest_inception,
            MAX(fd.inception_date) as latest_inception,
            -- Asset size estimation (sum of latest NAVPS * assumed shares)
            SUM(COALESCE(latest_quotes.latest_navps, 0)) as total_estimated_nav,
            AVG(latest_quotes.latest_navps) as avg_fund_navps,
            -- Currency distribution
            STRING_AGG(DISTINCT fd.currency, ', ') as currencies,
            -- Data quality metrics
            ROUND(
                COUNT(DISTINCT CASE WHEN latest_quotes.latest_date >= CURRENT_DATE - INTERVAL '7 days' THEN fd.instrument_key END)::numeric /
                GREATEST(COUNT(DISTINCT fd.instrument_key), 1) * 100, 2
            ) as data_freshness_percent,
            -- Performance metrics (where available)
            AVG(
                CASE WHEN latest_quotes.navps_percent_change BETWEEN -50 AND 50
                THEN latest_quotes.navps_percent_change
                END
            ) as avg_recent_return,
            COUNT(DISTINCT CASE WHEN latest_quotes.navps_percent_change > 0 THEN fd.instrument_key END) as funds_positive_return,
            -- Update tracking
            MAX(fd.processed_at) as latest_data_update,
            MAX(latest_quotes.processed_at) as latest_quotes_update
        FROM fundata_data fd
        LEFT JOIN (
            SELECT DISTINCT ON (instrument_key)
                instrument_key,
                date as latest_date,
                navps as latest_navps,
                navps_percent_change,
                processed_at,
                ROW_NUMBER() OVER (PARTITION BY instrument_key ORDER BY date DESC, processed_at DESC) as rn
            FROM fundata_quotes
            WHERE record_state = 'Active'
        ) latest_quotes ON fd.instrument_key = latest_quotes.instrument_key AND latest_quotes.rn = 1
        WHERE fd.record_state = 'Active'
          AND fd.family_name IS NOT NULL
          AND LENGTH(TRIM(fd.family_name)) > 0
        GROUP BY fd.family_name, fd.company
        HAVING COUNT(DISTINCT fd.instrument_key) >= 2  -- Only families with 2+ funds
        ORDER BY total_funds DESC, family_name;
        """

        async with self.db_manager.pool.acquire() as connection:
            await connection.execute(view_sql)

    async def _create_market_statistics_view(self) -> None:
        """Create market-wide statistics view."""
        view_sql = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_market_statistics AS
        SELECT
            CURRENT_DATE as statistics_date,
            -- Equity statistics
            (SELECT COUNT(*) FROM equity_profile WHERE is_actively_trading = TRUE) as active_equities,
            (SELECT COUNT(DISTINCT exchange) FROM equity_profile WHERE is_actively_trading = TRUE) as active_exchanges,
            (SELECT COUNT(DISTINCT sector) FROM equity_profile WHERE is_actively_trading = TRUE AND sector IS NOT NULL) as sectors_covered,
            (SELECT SUM(market_cap) FROM equity_profile WHERE is_actively_trading = TRUE AND market_cap IS NOT NULL) as total_market_cap,
            (SELECT AVG(market_cap) FROM equity_profile WHERE is_actively_trading = TRUE AND market_cap IS NOT NULL) as avg_market_cap,
            -- Fund statistics
            (SELECT COUNT(*) FROM fundata_data WHERE record_state = 'Active') as total_funds,
            (SELECT COUNT(DISTINCT family_name) FROM fundata_data WHERE record_state = 'Active' AND family_name IS NOT NULL) as fund_families,
            (SELECT COUNT(DISTINCT company) FROM fundata_data WHERE record_state = 'Active' AND company IS NOT NULL) as management_companies,
            -- Quote statistics
            (SELECT COUNT(*) FROM fundata_quotes WHERE record_state = 'Active') as total_fund_quotes,
            (SELECT COUNT(DISTINCT instrument_key) FROM fundata_quotes WHERE record_state = 'Active' AND date >= CURRENT_DATE - INTERVAL '7 days') as funds_with_recent_quotes,
            (SELECT AVG(navps) FROM fundata_quotes WHERE record_state = 'Active' AND date >= CURRENT_DATE - INTERVAL '30 days') as avg_recent_navps,
            -- Data freshness
            (SELECT MAX(processed_at) FROM equity_profile) as latest_equity_update,
            (SELECT MAX(processed_at) FROM fundata_data) as latest_fund_update,
            (SELECT MAX(processed_at) FROM fundata_quotes) as latest_quote_update,
            -- Market cap distribution
            (SELECT COUNT(*) FROM equity_profile WHERE is_actively_trading = TRUE AND market_cap >= 200000000000) as mega_cap_count,
            (SELECT COUNT(*) FROM equity_profile WHERE is_actively_trading = TRUE AND market_cap >= 10000000000 AND market_cap < 200000000000) as large_cap_count,
            (SELECT COUNT(*) FROM equity_profile WHERE is_actively_trading = TRUE AND market_cap >= 2000000000 AND market_cap < 10000000000) as mid_cap_count,
            (SELECT COUNT(*) FROM equity_profile WHERE is_actively_trading = TRUE AND market_cap < 2000000000) as small_cap_count,
            -- Exchange distribution
            (SELECT jsonb_object_agg(exchange, count) FROM (
                SELECT exchange, COUNT(*) as count
                FROM equity_profile
                WHERE is_actively_trading = TRUE
                GROUP BY exchange
                ORDER BY count DESC
                LIMIT 10
            ) top_exchanges) as top_exchanges_distribution,
            -- Sector distribution
            (SELECT jsonb_object_agg(sector, count) FROM (
                SELECT sector, COUNT(*) as count
                FROM equity_profile
                WHERE is_actively_trading = TRUE AND sector IS NOT NULL
                GROUP BY sector
                ORDER BY count DESC
                LIMIT 15
            ) top_sectors) as top_sectors_distribution,
            -- Currency distribution for funds
            (SELECT jsonb_object_agg(currency, count) FROM (
                SELECT currency, COUNT(*) as count
                FROM fundata_data
                WHERE record_state = 'Active' AND currency IS NOT NULL
                GROUP BY currency
                ORDER BY count DESC
            ) currencies) as fund_currencies_distribution,
            -- Generated timestamp
            CURRENT_TIMESTAMP as generated_at;

        -- No indexes needed for single-row materialized view
        """

        async with self.db_manager.pool.acquire() as connection:
            await connection.execute(view_sql)

    async def _create_data_freshness_view(self) -> None:
        """Create data freshness monitoring view."""
        view_sql = """
        CREATE OR REPLACE VIEW v_data_freshness AS
        SELECT
            'equity_profile' as table_name,
            COUNT(*) as total_records,
            MIN(created_at) as oldest_record,
            MAX(updated_at) as newest_record,
            COUNT(CASE WHEN updated_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 1 END) as updated_last_24h,
            COUNT(CASE WHEN updated_at >= CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END) as updated_last_7d,
            ROUND(
                COUNT(CASE WHEN updated_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 1 END)::numeric /
                GREATEST(COUNT(*), 1) * 100, 2
            ) as freshness_score_24h,
            CURRENT_TIMESTAMP as checked_at
        FROM equity_profile

        UNION ALL

        SELECT
            'fundata_data' as table_name,
            COUNT(*) as total_records,
            MIN(processed_at) as oldest_record,
            MAX(processed_at) as newest_record,
            COUNT(CASE WHEN processed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 1 END) as updated_last_24h,
            COUNT(CASE WHEN processed_at >= CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END) as updated_last_7d,
            ROUND(
                COUNT(CASE WHEN processed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 1 END)::numeric /
                GREATEST(COUNT(*), 1) * 100, 2
            ) as freshness_score_24h,
            CURRENT_TIMESTAMP as checked_at
        FROM fundata_data

        UNION ALL

        SELECT
            'fundata_quotes' as table_name,
            COUNT(*) as total_records,
            MIN(processed_at) as oldest_record,
            MAX(processed_at) as newest_record,
            COUNT(CASE WHEN processed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 1 END) as updated_last_24h,
            COUNT(CASE WHEN processed_at >= CURRENT_TIMESTAMP - INTERVAL '7 days' THEN 1 END) as updated_last_7d,
            ROUND(
                COUNT(CASE WHEN processed_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 1 END)::numeric /
                GREATEST(COUNT(*), 1) * 100, 2
            ) as freshness_score_24h,
            CURRENT_TIMESTAMP as checked_at
        FROM fundata_quotes

        ORDER BY table_name;
        """

        async with self.db_manager.pool.acquire() as connection:
            await connection.execute(view_sql)

    async def _create_search_optimization_view(self) -> None:
        """Create search optimization view with text search vectors."""
        view_sql = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS mv_search_optimization AS
        SELECT
            'equity' as record_type,
            ep.symbol as primary_key,
            ep.symbol,
            ep.company_name as name,
            ep.exchange,
            ep.sector,
            ep.industry,
            ep.market_cap,
            ep.is_etf,
            ep.country,
            -- Create search vector for full-text search
            setweight(to_tsvector('english', COALESCE(ep.symbol, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(ep.company_name, '')), 'B') ||
            setweight(to_tsvector('english', COALESCE(ep.sector, '')), 'C') ||
            setweight(to_tsvector('english', COALESCE(ep.industry, '')), 'D') as search_vector,
            -- Search ranking factors
            CASE
                WHEN ep.is_actively_trading AND ep.market_cap IS NOT NULL THEN 1.0
                WHEN ep.is_actively_trading THEN 0.8
                WHEN ep.market_cap IS NOT NULL THEN 0.6
                ELSE 0.4
            END as base_rank,
            -- Searchable text for ILIKE queries
            LOWER(ep.symbol || ' ' || ep.company_name || ' ' || COALESCE(ep.sector, '') || ' ' || COALESCE(ep.industry, '')) as searchable_text,
            ep.updated_at as last_updated
        FROM equity_profile ep
        WHERE ep.is_actively_trading = TRUE

        UNION ALL

        SELECT
            'fund' as record_type,
            fd.instrument_key as primary_key,
            fd.instrument_key as symbol,
            COALESCE(fd.legal_name, fd.series_name, fd.instrument_key) as name,
            fd.family_name as exchange,  -- Using family_name in exchange field for consistency
            fd.company as sector,        -- Using company in sector field
            'Fund' as industry,
            NULL as market_cap,
            FALSE as is_etf,
            'CA' as country,  -- Assuming Canadian funds
            -- Create search vector
            setweight(to_tsvector('english', COALESCE(fd.instrument_key, '')), 'A') ||
            setweight(to_tsvector('english', COALESCE(fd.legal_name, '')), 'B') ||
            setweight(to_tsvector('english', COALESCE(fd.family_name, '')), 'C') ||
            setweight(to_tsvector('english', COALESCE(fd.series_name, '')), 'C') ||
            setweight(to_tsvector('english', COALESCE(fd.company, '')), 'D') as search_vector,
            -- Search ranking factors
            CASE
                WHEN latest_quotes.latest_date >= CURRENT_DATE - INTERVAL '7 days' THEN 1.0
                WHEN latest_quotes.latest_date >= CURRENT_DATE - INTERVAL '30 days' THEN 0.8
                WHEN latest_quotes.latest_date >= CURRENT_DATE - INTERVAL '90 days' THEN 0.6
                ELSE 0.4
            END as base_rank,
            -- Searchable text
            LOWER(
                fd.instrument_key || ' ' ||
                COALESCE(fd.legal_name, '') || ' ' ||
                COALESCE(fd.family_name, '') || ' ' ||
                COALESCE(fd.series_name, '') || ' ' ||
                COALESCE(fd.company, '')
            ) as searchable_text,
            fd.processed_at as last_updated
        FROM fundata_data fd
        LEFT JOIN (
            SELECT DISTINCT ON (instrument_key)
                instrument_key,
                date as latest_date,
                ROW_NUMBER() OVER (PARTITION BY instrument_key ORDER BY date DESC) as rn
            FROM fundata_quotes
            WHERE record_state = 'Active'
        ) latest_quotes ON fd.instrument_key = latest_quotes.instrument_key AND latest_quotes.rn = 1
        WHERE fd.record_state = 'Active';

        -- Create GIN index for full-text search
        CREATE INDEX IF NOT EXISTS idx_mv_search_optimization_fts
        ON mv_search_optimization USING GIN (search_vector);

        -- Create index for traditional text search
        CREATE INDEX IF NOT EXISTS idx_mv_search_optimization_text
        ON mv_search_optimization USING GIN (searchable_text gin_trgm_ops);

        -- Create index for filtering and ranking
        CREATE INDEX IF NOT EXISTS idx_mv_search_optimization_type_rank
        ON mv_search_optimization(record_type, base_rank DESC);
        """

        async with self.db_manager.pool.acquire() as connection:
            await connection.execute(view_sql)

    async def refresh_materialized_view(self, view_name: str, concurrently: bool = True) -> ViewRefreshResult:
        """Refresh a materialized view.

        Args:
            view_name: Name of materialized view to refresh
            concurrently: Whether to refresh concurrently (non-blocking)

        Returns:
            View refresh result
        """
        result = ViewRefreshResult(view_name)
        start_time = datetime.utcnow()

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Get row count before refresh
                pre_count = await connection.fetchval(f"SELECT COUNT(*) FROM {view_name}")

                # Refresh the view
                refresh_sql = f"REFRESH MATERIALIZED VIEW {'CONCURRENTLY ' if concurrently else ''}{view_name}"
                await connection.execute(refresh_sql)

                # Get row count after refresh
                post_count = await connection.fetchval(f"SELECT COUNT(*) FROM {view_name}")

                result.rows_affected = post_count
                logger.info(f"Refreshed materialized view {view_name}: {post_count} rows")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Failed to refresh materialized view {view_name}: {e}")

        finally:
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        return result

    async def refresh_all_materialized_views(self, concurrently: bool = True) -> List[ViewRefreshResult]:
        """Refresh all materialized views.

        Args:
            concurrently: Whether to refresh concurrently

        Returns:
            List of refresh results
        """
        materialized_views = ['mv_fund_performance', 'mv_market_statistics', 'mv_search_optimization']
        results = []

        for view_name in materialized_views:
            try:
                result = await self.refresh_materialized_view(view_name, concurrently)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to refresh {view_name}: {e}")
                result = ViewRefreshResult(view_name)
                result.success = False
                result.error_message = str(e)
                results.append(result)

        successful_refreshes = sum(1 for r in results if r.success)
        logger.info(f"Materialized view refresh completed: {successful_refreshes}/{len(results)} successful")

        return results

    async def get_view_info(self, view_name: str) -> Dict[str, Any]:
        """Get comprehensive view information.

        Args:
            view_name: Name of view to analyze

        Returns:
            Dictionary with view metadata
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                # Check if view exists and get basic info
                view_info = await connection.fetchrow("""
                    SELECT
                        schemaname,
                        viewname,
                        viewowner,
                        definition
                    FROM pg_views
                    WHERE viewname = $1
                    UNION
                    SELECT
                        schemaname,
                        matviewname as viewname,
                        matviewowner as viewowner,
                        definition
                    FROM pg_matviews
                    WHERE matviewname = $1
                """, view_name)

                if not view_info:
                    return {'exists': False}

                # Get size information (for materialized views)
                size_info = await connection.fetchrow("""
                    SELECT
                        pg_size_pretty(pg_total_relation_size($1)) as total_size,
                        pg_total_relation_size($1) as total_size_bytes
                    WHERE EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = $1)
                """, view_name)

                # Get row count
                try:
                    row_count = await connection.fetchval(f"SELECT COUNT(*) FROM {view_name}")
                except:
                    row_count = None

                # Check if it's materialized
                is_materialized = await connection.fetchval("""
                    SELECT EXISTS (SELECT 1 FROM pg_matviews WHERE matviewname = $1)
                """, view_name)

                # Get last refresh info (for materialized views)
                last_refresh = None
                if is_materialized:
                    try:
                        last_refresh = await connection.fetchval("""
                            SELECT stats_reset FROM pg_stat_user_tables WHERE relname = $1
                        """, view_name)
                    except:
                        pass

                return {
                    'exists': True,
                    'is_materialized': is_materialized,
                    'basic_info': dict(view_info),
                    'size_info': dict(size_info) if size_info else {},
                    'row_count': row_count,
                    'last_refresh': last_refresh,
                    'analyzed_at': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get view info for {view_name}: {e}")
            return {'exists': False, 'error': str(e)}

    async def search_unified(
        self,
        query: str,
        record_types: List[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Perform unified search across equity and fund data.

        Args:
            query: Search query string
            record_types: List of record types to search ('equity', 'fund')
            limit: Maximum records to return

        Returns:
            List of search results with relevance scoring
        """
        if not query.strip():
            return []

        record_types = record_types or ['equity', 'fund']
        search_query = f"%{query.strip().lower()}%"

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Use the search optimization view
                type_filter = "WHERE record_type = ANY($2)" if record_types != ['equity', 'fund'] else ""

                search_sql = f"""
                    SELECT
                        record_type,
                        primary_key,
                        symbol,
                        name,
                        exchange,
                        sector,
                        industry,
                        market_cap,
                        is_etf,
                        country,
                        base_rank,
                        -- Calculate relevance score
                        CASE
                            WHEN symbol ILIKE $1 THEN base_rank * 4.0
                            WHEN name ILIKE $1 THEN base_rank * 3.0
                            WHEN searchable_text LIKE $1 THEN base_rank * 2.0
                            ELSE base_rank * 1.0
                        END as relevance_score,
                        last_updated
                    FROM mv_search_optimization
                    {type_filter}
                    AND searchable_text LIKE $1
                    ORDER BY relevance_score DESC, name
                    LIMIT $3
                """

                params = [search_query]
                if record_types != ['equity', 'fund']:
                    params.append(record_types)
                params.append(limit)

                rows = await connection.fetch(search_sql, *params)
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Unified search failed: {e}")
            return []

    async def get_view_health_check(self) -> Dict[str, Any]:
        """Perform health check on all views.

        Returns:
            Health check results
        """
        try:
            views_to_check = [
                'v_equity_summary',
                'v_fund_family_overview',
                'v_data_freshness',
                'mv_fund_performance',
                'mv_market_statistics',
                'mv_search_optimization'
            ]

            view_statuses = {}
            total_healthy = 0

            for view_name in views_to_check:
                try:
                    info = await self.get_view_info(view_name)
                    if info.get('exists'):
                        view_statuses[view_name] = {
                            'status': 'healthy',
                            'row_count': info.get('row_count'),
                            'is_materialized': info.get('is_materialized'),
                            'last_refresh': info.get('last_refresh')
                        }
                        total_healthy += 1
                    else:
                        view_statuses[view_name] = {
                            'status': 'missing',
                            'error': 'View does not exist'
                        }
                except Exception as e:
                    view_statuses[view_name] = {
                        'status': 'error',
                        'error': str(e)
                    }

            return {
                'status': 'healthy' if total_healthy == len(views_to_check) else 'degraded',
                'total_views': len(views_to_check),
                'healthy_views': total_healthy,
                'view_details': view_statuses,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }