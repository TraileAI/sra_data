"""Fundata repository with comprehensive CRUD operations for fund data.

This module provides database operations for fundata processing with
separate handling for general data and quotes, including bulk operations.
"""

import logging
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, date

from packages.sra_data.domain.models import (
    FundataDataRecord,
    FundataQuotesRecord,
    ProcessingResult,
    RecordState
)
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


class FundataDataRepository:
    """Repository for fundata general data operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize fundata data repository.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    async def create(self, fund_data: FundataDataRecord) -> bool:
        """Create a new fundata data record.

        Args:
            fund_data: Fundata record to create

        Returns:
            True if creation successful, False otherwise
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO fundata_data (
                        instrument_key, record_id, language, legal_name, family_name,
                        series_name, company, inception_date, change_date, currency,
                        record_state, source_file, file_hash, processed_at, additional_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                    fund_data.InstrumentKey,
                    fund_data.RecordId,
                    fund_data.Language,
                    fund_data.LegalName,
                    fund_data.FamilyName,
                    fund_data.SeriesName,
                    fund_data.Company,
                    fund_data.InceptionDate,
                    fund_data.ChangeDate,
                    fund_data.Currency.value if fund_data.Currency else None,
                    fund_data.RecordState.value,
                    fund_data.source_file,
                    fund_data.file_hash,
                    fund_data.processed_at,
                    fund_data.additional_data
                )
            logger.debug(f"Created fundata record: {fund_data.InstrumentKey}")
            return True

        except Exception as e:
            logger.error(f"Failed to create fundata record {fund_data.InstrumentKey}: {e}")
            return False

    async def get_by_instrument_key(self, instrument_key: str) -> Optional[FundataDataRecord]:
        """Retrieve fundata record by instrument key.

        Args:
            instrument_key: Instrument key to lookup

        Returns:
            Fundata record if found, None otherwise
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                row = await connection.fetchrow(
                    """
                    SELECT * FROM fundata_data WHERE instrument_key = $1
                    ORDER BY processed_at DESC LIMIT 1
                    """,
                    instrument_key
                )

                if row:
                    # Convert database row to model
                    data = dict(row)
                    # Map database column names to model field names
                    model_data = {
                        'InstrumentKey': data['instrument_key'],
                        'RecordId': data['record_id'],
                        'Language': data['language'],
                        'LegalName': data['legal_name'],
                        'FamilyName': data['family_name'],
                        'SeriesName': data['series_name'],
                        'Company': data['company'],
                        'InceptionDate': data['inception_date'],
                        'ChangeDate': data['change_date'],
                        'Currency': data['currency'],
                        'RecordState': data['record_state'],
                        'source_file': data['source_file'],
                        'file_hash': data['file_hash'],
                        'processed_at': data['processed_at'],
                        'additional_data': data['additional_data']
                    }
                    return FundataDataRecord(**model_data)
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve fundata record {instrument_key}: {e}")
            return None

    async def get_by_family_name(self, family_name: str, limit: int = 100) -> List[FundataDataRecord]:
        """Retrieve fundata records by family name.

        Args:
            family_name: Fund family name
            limit: Maximum records to return

        Returns:
            List of fundata records
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT * FROM fundata_data
                    WHERE family_name ILIKE $1
                    ORDER BY legal_name, instrument_key
                    LIMIT $2
                    """,
                    f"%{family_name}%",
                    limit
                )

                return [self._row_to_model(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to retrieve fundata records by family {family_name}: {e}")
            return []

    async def bulk_insert(self, fund_data_list: List[FundataDataRecord]) -> ProcessingResult:
        """Bulk insert fundata records with transaction support.

        Args:
            fund_data_list: List of fundata records to insert

        Returns:
            Processing result with success metrics
        """
        if not fund_data_list:
            return ProcessingResult(success=True, records_processed=0)

        start_time = datetime.utcnow()
        records_processed = 0
        records_failed = 0
        errors = []

        try:
            async with self.db_manager.pool.acquire() as connection:
                async with connection.transaction():
                    for fund_data in fund_data_list:
                        try:
                            await connection.execute(
                                """
                                INSERT INTO fundata_data (
                                    instrument_key, record_id, language, legal_name, family_name,
                                    series_name, company, inception_date, change_date, currency,
                                    record_state, source_file, file_hash, processed_at, additional_data
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                                ON CONFLICT (instrument_key, record_id) DO UPDATE SET
                                    language = EXCLUDED.language,
                                    legal_name = EXCLUDED.legal_name,
                                    family_name = EXCLUDED.family_name,
                                    series_name = EXCLUDED.series_name,
                                    company = EXCLUDED.company,
                                    inception_date = EXCLUDED.inception_date,
                                    change_date = EXCLUDED.change_date,
                                    currency = EXCLUDED.currency,
                                    record_state = EXCLUDED.record_state,
                                    source_file = EXCLUDED.source_file,
                                    file_hash = EXCLUDED.file_hash,
                                    processed_at = EXCLUDED.processed_at,
                                    additional_data = EXCLUDED.additional_data
                                """,
                                fund_data.InstrumentKey,
                                fund_data.RecordId,
                                fund_data.Language,
                                fund_data.LegalName,
                                fund_data.FamilyName,
                                fund_data.SeriesName,
                                fund_data.Company,
                                fund_data.InceptionDate,
                                fund_data.ChangeDate,
                                fund_data.Currency.value if fund_data.Currency else None,
                                fund_data.RecordState.value,
                                fund_data.source_file,
                                fund_data.file_hash,
                                fund_data.processed_at,
                                fund_data.additional_data
                            )
                            records_processed += 1

                        except Exception as e:
                            records_failed += 1
                            error_msg = f"Failed to insert fundata record {fund_data.InstrumentKey}: {e}"
                            errors.append(error_msg)
                            logger.warning(error_msg)

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Fundata bulk insert completed: {records_processed} processed, {records_failed} failed")

            return ProcessingResult(
                success=records_failed == 0,
                records_processed=records_processed,
                records_failed=records_failed,
                errors=errors,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Fundata bulk insert failed: {e}")

            return ProcessingResult(
                success=False,
                records_processed=records_processed,
                records_failed=len(fund_data_list) - records_processed,
                errors=errors + [f"Transaction failed: {e}"],
                processing_time_seconds=processing_time
            )

    def _row_to_model(self, row: asyncpg.Record) -> FundataDataRecord:
        """Convert database row to FundataDataRecord model.

        Args:
            row: Database row

        Returns:
            FundataDataRecord instance
        """
        data = dict(row)
        model_data = {
            'InstrumentKey': data['instrument_key'],
            'RecordId': data['record_id'],
            'Language': data['language'],
            'LegalName': data['legal_name'],
            'FamilyName': data['family_name'],
            'SeriesName': data['series_name'],
            'Company': data['company'],
            'InceptionDate': data['inception_date'],
            'ChangeDate': data['change_date'],
            'Currency': data['currency'],
            'RecordState': data['record_state'],
            'source_file': data['source_file'],
            'file_hash': data['file_hash'],
            'processed_at': data['processed_at'],
            'additional_data': data['additional_data']
        }
        return FundataDataRecord(**model_data)


class FundataQuotesRepository:
    """Repository for fundata quotes operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize fundata quotes repository.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager

    async def create(self, quote: FundataQuotesRecord) -> bool:
        """Create a new fundata quotes record.

        Args:
            quote: Fundata quotes record to create

        Returns:
            True if creation successful, False otherwise
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                await connection.execute(
                    """
                    INSERT INTO fundata_quotes (
                        instrument_key, record_id, date, navps, navps_penny_change,
                        navps_percent_change, previous_date, previous_navps, record_state,
                        change_date, source_file, file_hash, processed_at, additional_data
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    """,
                    quote.InstrumentKey,
                    quote.RecordId,
                    quote.Date,
                    quote.NAVPS,
                    quote.NAVPSPennyChange,
                    quote.NAVPSPercentChange,
                    quote.PreviousDate,
                    quote.PreviousNAVPS,
                    quote.RecordState.value,
                    quote.ChangeDate,
                    quote.source_file,
                    quote.file_hash,
                    quote.processed_at,
                    quote.additional_data
                )
            logger.debug(f"Created fundata quote: {quote.InstrumentKey} {quote.Date}")
            return True

        except Exception as e:
            logger.error(f"Failed to create fundata quote {quote.InstrumentKey} {quote.Date}: {e}")
            return False

    async def get_latest_quote(self, instrument_key: str) -> Optional[FundataQuotesRecord]:
        """Get the latest quote for an instrument.

        Args:
            instrument_key: Instrument key to lookup

        Returns:
            Latest fundata quotes record if found, None otherwise
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                row = await connection.fetchrow(
                    """
                    SELECT * FROM fundata_quotes
                    WHERE instrument_key = $1
                    ORDER BY date DESC, processed_at DESC
                    LIMIT 1
                    """,
                    instrument_key
                )

                if row:
                    return self._row_to_model(row)
                return None

        except Exception as e:
            logger.error(f"Failed to retrieve latest quote for {instrument_key}: {e}")
            return None

    async def get_quotes_date_range(
        self,
        instrument_key: str,
        start_date: date,
        end_date: date
    ) -> List[FundataQuotesRecord]:
        """Get quotes for instrument within date range.

        Args:
            instrument_key: Instrument key
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of quotes in date range
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                rows = await connection.fetch(
                    """
                    SELECT * FROM fundata_quotes
                    WHERE instrument_key = $1
                    AND date BETWEEN $2 AND $3
                    ORDER BY date ASC
                    """,
                    instrument_key,
                    start_date,
                    end_date
                )

                return [self._row_to_model(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to retrieve quotes for {instrument_key} {start_date}-{end_date}: {e}")
            return []

    async def bulk_insert(self, quotes_list: List[FundataQuotesRecord]) -> ProcessingResult:
        """Bulk insert fundata quotes with transaction support.

        Args:
            quotes_list: List of fundata quotes to insert

        Returns:
            Processing result with success metrics
        """
        if not quotes_list:
            return ProcessingResult(success=True, records_processed=0)

        start_time = datetime.utcnow()
        records_processed = 0
        records_failed = 0
        errors = []

        try:
            async with self.db_manager.pool.acquire() as connection:
                async with connection.transaction():
                    for quote in quotes_list:
                        try:
                            await connection.execute(
                                """
                                INSERT INTO fundata_quotes (
                                    instrument_key, record_id, date, navps, navps_penny_change,
                                    navps_percent_change, previous_date, previous_navps, record_state,
                                    change_date, source_file, file_hash, processed_at, additional_data
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                                ON CONFLICT (instrument_key, record_id, date) DO UPDATE SET
                                    navps = EXCLUDED.navps,
                                    navps_penny_change = EXCLUDED.navps_penny_change,
                                    navps_percent_change = EXCLUDED.navps_percent_change,
                                    previous_date = EXCLUDED.previous_date,
                                    previous_navps = EXCLUDED.previous_navps,
                                    record_state = EXCLUDED.record_state,
                                    change_date = EXCLUDED.change_date,
                                    source_file = EXCLUDED.source_file,
                                    file_hash = EXCLUDED.file_hash,
                                    processed_at = EXCLUDED.processed_at,
                                    additional_data = EXCLUDED.additional_data
                                """,
                                quote.InstrumentKey,
                                quote.RecordId,
                                quote.Date,
                                quote.NAVPS,
                                quote.NAVPSPennyChange,
                                quote.NAVPSPercentChange,
                                quote.PreviousDate,
                                quote.PreviousNAVPS,
                                quote.RecordState.value,
                                quote.ChangeDate,
                                quote.source_file,
                                quote.file_hash,
                                quote.processed_at,
                                quote.additional_data
                            )
                            records_processed += 1

                        except Exception as e:
                            records_failed += 1
                            error_msg = f"Failed to insert quote {quote.InstrumentKey} {quote.Date}: {e}"
                            errors.append(error_msg)
                            logger.warning(error_msg)

            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Quotes bulk insert completed: {records_processed} processed, {records_failed} failed")

            return ProcessingResult(
                success=records_failed == 0,
                records_processed=records_processed,
                records_failed=records_failed,
                errors=errors,
                processing_time_seconds=processing_time
            )

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Quotes bulk insert failed: {e}")

            return ProcessingResult(
                success=False,
                records_processed=records_processed,
                records_failed=len(quotes_list) - records_processed,
                errors=errors + [f"Transaction failed: {e}"],
                processing_time_seconds=processing_time
            )

    async def get_statistics(self) -> Dict[str, Any]:
        """Get fundata quotes statistics.

        Returns:
            Dictionary with various statistics
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                stats = await connection.fetchrow("""
                    SELECT
                        COUNT(*) as total_quotes,
                        COUNT(DISTINCT instrument_key) as unique_instruments,
                        MIN(date) as earliest_date,
                        MAX(date) as latest_date,
                        AVG(navps) as avg_navps,
                        MAX(navps) as max_navps,
                        MIN(navps) as min_navps
                    FROM fundata_quotes
                """)

                return dict(stats) if stats else {}

        except Exception as e:
            logger.error(f"Failed to get quotes statistics: {e}")
            return {}

    def _row_to_model(self, row: asyncpg.Record) -> FundataQuotesRecord:
        """Convert database row to FundataQuotesRecord model.

        Args:
            row: Database row

        Returns:
            FundataQuotesRecord instance
        """
        data = dict(row)
        model_data = {
            'InstrumentKey': data['instrument_key'],
            'RecordId': data['record_id'],
            'Date': data['date'],
            'NAVPS': data['navps'],
            'NAVPSPennyChange': data['navps_penny_change'],
            'NAVPSPercentChange': data['navps_percent_change'],
            'PreviousDate': data['previous_date'],
            'PreviousNAVPS': data['previous_navps'],
            'RecordState': data['record_state'],
            'ChangeDate': data['change_date'],
            'source_file': data['source_file'],
            'file_hash': data['file_hash'],
            'processed_at': data['processed_at'],
            'additional_data': data['additional_data']
        }
        return FundataQuotesRecord(**model_data)


class FundataRepository:
    """Unified repository for fundata operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize unified fundata repository.

        Args:
            db_manager: Database manager instance
        """
        self.data_repo = FundataDataRepository(db_manager)
        self.quotes_repo = FundataQuotesRepository(db_manager)
        self.db_manager = db_manager

    async def get_fund_with_latest_quote(self, instrument_key: str) -> Optional[Tuple[FundataDataRecord, FundataQuotesRecord]]:
        """Get fund data with its latest quote.

        Args:
            instrument_key: Instrument key to lookup

        Returns:
            Tuple of (fund_data, latest_quote) if found, None otherwise
        """
        fund_data = await self.data_repo.get_by_instrument_key(instrument_key)
        if fund_data:
            latest_quote = await self.quotes_repo.get_latest_quote(instrument_key)
            if latest_quote:
                return (fund_data, latest_quote)
        return None

    async def get_combined_statistics(self) -> Dict[str, Any]:
        """Get combined statistics from both data and quotes tables.

        Returns:
            Dictionary with combined statistics
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                # Combined stats query
                stats = await connection.fetchrow("""
                    SELECT
                        (SELECT COUNT(*) FROM fundata_data) as total_funds,
                        (SELECT COUNT(*) FROM fundata_quotes) as total_quotes,
                        (SELECT COUNT(DISTINCT instrument_key) FROM fundata_quotes) as funds_with_quotes,
                        (SELECT MIN(date) FROM fundata_quotes) as earliest_quote,
                        (SELECT MAX(date) FROM fundata_quotes) as latest_quote
                """)

                return dict(stats) if stats else {}

        except Exception as e:
            logger.error(f"Failed to get combined statistics: {e}")
            return {}