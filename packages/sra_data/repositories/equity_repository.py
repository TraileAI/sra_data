"""Equity profile repository with comprehensive CRUD operations.

This module provides database operations for equity profile management
with transaction support, bulk operations, and query optimization.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from packages.sra_data.domain.models import EquityProfile, ProcessingResult
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


class EquityRepository:
    """Repository for equity profile database operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize equity repository.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    async def create(self, equity: EquityProfile) -> bool:
        """Create a new equity profile record.

        Args:
            equity: Equity profile to create

        Returns:
            True if creation successful, False otherwise
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO equity_profile (
                        symbol, company_name, exchange, sector, industry,
                        market_cap, employees, description, website, country,
                        currency, is_etf, is_actively_trading, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                    equity.symbol,
                    equity.company_name,
                    equity.exchange.value,
                    equity.sector,
                    equity.industry,
                    equity.market_cap,
                    equity.employees,
                    equity.description,
                    equity.website,
                    equity.country,
                    equity.currency.value,
                    equity.is_etf,
                    equity.is_actively_trading,
                    equity.created_at,
                    equity.updated_at
                )
            logger.debug(f"Created equity profile: {equity.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to create equity profile {equity.symbol}: {e}")
            return False

    async def get_by_symbol(self, symbol: str) -> Optional[EquityProfile]:
        """Retrieve equity profile by symbol.

        Args:
            symbol: Stock symbol to lookup

        Returns:
            Equity profile if found, None otherwise
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                row = await connection.fetchrow(
                    """
                    SELECT * FROM equity_profile WHERE symbol = $1
                    """,
                    symbol.upper()
                )

                if row:
                    return EquityProfile(**dict(row))
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve equity profile {symbol}: {e}")
            return None

    async def get_by_exchange(self, exchange: str, limit: int = 100) -> List[EquityProfile]:
        """Retrieve equity profiles by exchange.

        Args:
            exchange: Exchange code
            limit: Maximum records to return

        Returns:
            List of equity profiles
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT * FROM equity_profile
                    WHERE exchange = $1
                    ORDER BY symbol
                    LIMIT $2
                    """,
                    exchange,
                    limit
                )

                return [EquityProfile(**dict(row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to retrieve equity profiles by exchange {exchange}: {e}")
            return []

    async def get_by_sector(self, sector: str, limit: int = 100) -> List[EquityProfile]:
        """Retrieve equity profiles by sector.

        Args:
            sector: Business sector
            limit: Maximum records to return

        Returns:
            List of equity profiles
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT * FROM equity_profile
                    WHERE sector = $1
                    ORDER BY market_cap DESC NULLS LAST, symbol
                    LIMIT $2
                    """,
                    sector,
                    limit
                )

                return [EquityProfile(**dict(row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to retrieve equity profiles by sector {sector}: {e}")
            return []

    async def update(self, symbol: str, updates: Dict[str, Any]) -> bool:
        """Update equity profile fields.

        Args:
            symbol: Stock symbol to update
            updates: Fields to update

        Returns:
            True if update successful, False otherwise
        """
        if not updates:
            return True

        try:
            # Build dynamic update query
            set_clauses = []
            values = []
            param_count = 1

            for field, value in updates.items():
                if field not in ['symbol', 'created_at']:  # Immutable fields
                    set_clauses.append(f"{field} = ${param_count}")
                    values.append(value)
                    param_count += 1

            if not set_clauses:
                return True

            # Always update timestamp
            set_clauses.append(f"updated_at = ${param_count}")
            values.append(datetime.utcnow())
            param_count += 1

            # Add WHERE clause
            values.append(symbol.upper())

            query = f"""
                UPDATE equity_profile
                SET {', '.join(set_clauses)}
                WHERE symbol = ${param_count}
            """

            async with self.db_manager.pool.acquire() as connection:
                result = await connection.execute(query, *values)

                # Check if any rows were affected
                rows_affected = int(result.split()[-1]) if result else 0
                if rows_affected > 0:
                    logger.debug(f"Updated equity profile: {symbol}")
                    return True
                else:
                    logger.warning(f"No equity profile found to update: {symbol}")
                    return False

        except Exception as e:
            logger.error(f"Failed to update equity profile {symbol}: {e}")
            return False

    async def delete(self, symbol: str) -> bool:
        """Delete equity profile by symbol.

        Args:
            symbol: Stock symbol to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                result = await connection.execute(
                    "DELETE FROM equity_profile WHERE symbol = $1",
                    symbol.upper()
                )

                rows_affected = int(result.split()[-1]) if result else 0
                if rows_affected > 0:
                    logger.debug(f"Deleted equity profile: {symbol}")
                    return True
                else:
                    logger.warning(f"No equity profile found to delete: {symbol}")
                    return False

        except Exception as e:
            logger.error(f"Failed to delete equity profile {symbol}: {e}")
            return False

    async def bulk_insert(self, equities: List[EquityProfile]) -> ProcessingResult:
        """Bulk insert equity profiles with transaction support.

        Args:
            equities: List of equity profiles to insert

        Returns:
            Processing result with success metrics
        """
        if not equities:
            return ProcessingResult(success=True, records_processed=0)

        start_time = datetime.utcnow()
        records_processed = 0
        records_failed = 0
        errors = []

        try:
            async with self.db_manager.pool.acquire() as connection:
                async with connection.transaction():
                    for equity in equities:
                        try:
                            await connection.execute(
                                """
                                INSERT INTO equity_profile (
                                    symbol, company_name, exchange, sector, industry,
                                    market_cap, employees, description, website, country,
                                    currency, is_etf, is_actively_trading, created_at, updated_at
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                                ON CONFLICT (symbol) DO UPDATE SET
                                    company_name = EXCLUDED.company_name,
                                    exchange = EXCLUDED.exchange,
                                    sector = EXCLUDED.sector,
                                    industry = EXCLUDED.industry,
                                    market_cap = EXCLUDED.market_cap,
                                    employees = EXCLUDED.employees,
                                    description = EXCLUDED.description,
                                    website = EXCLUDED.website,
                                    country = EXCLUDED.country,
                                    currency = EXCLUDED.currency,
                                    is_etf = EXCLUDED.is_etf,
                                    is_actively_trading = EXCLUDED.is_actively_trading,
                                    updated_at = EXCLUDED.updated_at
                                """,
                                equity.symbol,
                                equity.company_name,
                                equity.exchange.value,
                                equity.sector,
                                equity.industry,
                                equity.market_cap,
                                equity.employees,
                                equity.description,
                                equity.website,
                                equity.country,
                                equity.currency.value,
                                equity.is_etf,
                                equity.is_actively_trading,
                                equity.created_at,
                                equity.updated_at
                            )
                            records_processed += 1

                        except Exception as e:
                            records_failed += 1
                            error_msg = f"Failed to insert equity {equity.symbol}: {e}"
                            errors.append(error_msg)
                            logger.warning(error_msg)

            processing_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(f"Bulk insert completed: {records_processed} processed, {records_failed} failed")

            return ProcessingResult(
                success=records_failed == 0,
                records_processed=records_processed,
                records_failed=records_failed,
                errors=errors,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Bulk insert transaction failed: {e}")

            return ProcessingResult(
                success=False,
                records_processed=records_processed,
                records_failed=len(equities) - records_processed,
                errors=errors + [f"Transaction failed: {e}"],
                processing_time_seconds=processing_time
            )

    async def search(
        self,
        query: str,
        fields: List[str] = None,
        limit: int = 50
    ) -> List[EquityProfile]:
        """Search equity profiles across multiple fields.

        Args:
            query: Search query string
            fields: Fields to search in (default: symbol, company_name)
            limit: Maximum records to return

        Returns:
            List of matching equity profiles
        """
        if not query.strip():
            return []

        search_fields = fields or ['symbol', 'company_name']
        search_query = f"%{query.strip()}%"

        try:
            # Build dynamic WHERE clause
            where_conditions = []
            for field in search_fields:
                if field in ['symbol', 'company_name', 'sector', 'industry']:
                    where_conditions.append(f"{field} ILIKE $1")

            if not where_conditions:
                return []

            where_clause = " OR ".join(where_conditions)

            async with self.db_manager.pool.acquire() as connection:
                rows = await connection.fetch(
                    f"""
                    SELECT * FROM equity_profile
                    WHERE {where_clause}
                    ORDER BY
                        CASE WHEN symbol ILIKE $1 THEN 1
                             WHEN company_name ILIKE $1 THEN 2
                             ELSE 3 END,
                        market_cap DESC NULLS LAST,
                        symbol
                    LIMIT $2
                    """,
                    search_query,
                    limit
                )

                return [EquityProfile(**dict(row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to search equity profiles: {e}")
            return []

    async def get_statistics(self) -> Dict[str, Any]:
        """Get equity profile statistics.

        Returns:
            Dictionary with various statistics
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                stats = await connection.fetchrow("""
                    SELECT
                        COUNT(*) as total_count,
                        COUNT(DISTINCT exchange) as exchange_count,
                        COUNT(DISTINCT sector) as sector_count,
                        COUNT(CASE WHEN is_etf THEN 1 END) as etf_count,
                        COUNT(CASE WHEN is_actively_trading THEN 1 END) as active_count,
                        AVG(market_cap) as avg_market_cap,
                        MAX(market_cap) as max_market_cap,
                        MIN(created_at) as oldest_record,
                        MAX(updated_at) as newest_update
                    FROM equity_profile
                """)

                return dict(stats) if stats else {}

        except Exception as e:
            logger.error(f"Failed to get equity statistics: {e}")
            return {}