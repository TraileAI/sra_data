"""Comprehensive tests for repository classes.

Test coverage for EquityRepository and FundataRepository classes with
CRUD operations, error handling, and bulk operations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, date
from decimal import Decimal

from packages.sra_data.repositories.equity_repository import EquityRepository
from packages.sra_data.repositories.fundata_repository import (
    FundataRepository,
    FundataDataRepository,
    FundataQuotesRepository
)
from packages.sra_data.domain.models import (
    EquityProfile,
    FundataDataRecord,
    FundataQuotesRecord,
    ExchangeType,
    RecordState,
    CurrencyType,
    ProcessingResult
)


class TestEquityRepository:
    """Test equity profile repository operations."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.fixture
    def equity_repository(self, mock_db_manager):
        """Create equity repository for testing."""
        db_manager, _ = mock_db_manager
        return EquityRepository(db_manager)

    @pytest.fixture
    def sample_equity(self):
        """Create sample equity profile."""
        return EquityProfile(
            symbol="AAPL",
            company_name="Apple Inc.",
            exchange=ExchangeType.NASDAQ,
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=Decimal("3000000000000"),
            employees=150000,
            description="Apple Inc. designs and manufactures consumer electronics.",
            website="https://www.apple.com",
            country="US",
            currency=CurrencyType.USD,
            is_etf=False,
            is_actively_trading=True
        )

    @pytest.mark.asyncio
    async def test_create_success(self, equity_repository, sample_equity, mock_db_manager):
        """Test successful equity profile creation."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        result = await equity_repository.create(sample_equity)

        assert result is True
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO equity_profile" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_failure(self, equity_repository, sample_equity, mock_db_manager):
        """Test equity profile creation failure."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.side_effect = Exception("Database error")

        result = await equity_repository.create(sample_equity)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_by_symbol_found(self, equity_repository, mock_db_manager):
        """Test retrieving equity profile by symbol when found."""
        _, mock_conn = mock_db_manager
        mock_row = {
            'symbol': 'AAPL',
            'company_name': 'Apple Inc.',
            'exchange': 'NASDAQ',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': Decimal('3000000000000'),
            'employees': 150000,
            'description': 'Apple Inc.',
            'website': 'https://www.apple.com',
            'country': 'US',
            'currency': 'USD',
            'is_etf': False,
            'is_actively_trading': True,
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        mock_conn.fetchrow.return_value = mock_row

        result = await equity_repository.get_by_symbol("AAPL")

        assert result is not None
        assert result.symbol == "AAPL"
        assert result.company_name == "Apple Inc."
        mock_conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_symbol_not_found(self, equity_repository, mock_db_manager):
        """Test retrieving equity profile by symbol when not found."""
        _, mock_conn = mock_db_manager
        mock_conn.fetchrow.return_value = None

        result = await equity_repository.get_by_symbol("NOTFOUND")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_symbol_error(self, equity_repository, mock_db_manager):
        """Test retrieving equity profile with database error."""
        _, mock_conn = mock_db_manager
        mock_conn.fetchrow.side_effect = Exception("Database error")

        result = await equity_repository.get_by_symbol("AAPL")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_exchange(self, equity_repository, mock_db_manager):
        """Test retrieving equity profiles by exchange."""
        _, mock_conn = mock_db_manager
        mock_rows = [
            {'symbol': 'AAPL', 'company_name': 'Apple Inc.', 'exchange': 'NASDAQ',
             'sector': 'Technology', 'industry': 'Consumer Electronics',
             'market_cap': Decimal('3000000000000'), 'employees': 150000,
             'description': 'Apple Inc.', 'website': 'https://www.apple.com',
             'country': 'US', 'currency': 'USD', 'is_etf': False,
             'is_actively_trading': True, 'created_at': datetime.utcnow(),
             'updated_at': datetime.utcnow()},
            {'symbol': 'GOOGL', 'company_name': 'Alphabet Inc.', 'exchange': 'NASDAQ',
             'sector': 'Technology', 'industry': 'Internet Content',
             'market_cap': Decimal('2000000000000'), 'employees': 140000,
             'description': 'Alphabet Inc.', 'website': 'https://www.google.com',
             'country': 'US', 'currency': 'USD', 'is_etf': False,
             'is_actively_trading': True, 'created_at': datetime.utcnow(),
             'updated_at': datetime.utcnow()}
        ]
        mock_conn.fetch.return_value = mock_rows

        results = await equity_repository.get_by_exchange("NASDAQ", limit=10)

        assert len(results) == 2
        assert all(isinstance(equity, EquityProfile) for equity in results)
        assert all(equity.exchange == ExchangeType.NASDAQ for equity in results)

    @pytest.mark.asyncio
    async def test_update_success(self, equity_repository, mock_db_manager):
        """Test successful equity profile update."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = "UPDATE 1"

        updates = {"market_cap": Decimal("3500000000000"), "employees": 160000}
        result = await equity_repository.update("AAPL", updates)

        assert result is True
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0][0]
        assert "UPDATE equity_profile" in call_args
        assert "market_cap = $1" in call_args
        assert "employees = $2" in call_args

    @pytest.mark.asyncio
    async def test_update_no_rows_affected(self, equity_repository, mock_db_manager):
        """Test update when no rows are affected."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = "UPDATE 0"

        result = await equity_repository.update("NOTFOUND", {"market_cap": 1000})

        assert result is False

    @pytest.mark.asyncio
    async def test_update_empty_updates(self, equity_repository, mock_db_manager):
        """Test update with empty updates dictionary."""
        _, mock_conn = mock_db_manager

        result = await equity_repository.update("AAPL", {})

        assert result is True
        mock_conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_success(self, equity_repository, mock_db_manager):
        """Test successful equity profile deletion."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = "DELETE 1"

        result = await equity_repository.delete("AAPL")

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_not_found(self, equity_repository, mock_db_manager):
        """Test deletion when record not found."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = "DELETE 0"

        result = await equity_repository.delete("NOTFOUND")

        assert result is False

    @pytest.mark.asyncio
    async def test_bulk_insert_success(self, equity_repository, mock_db_manager):
        """Test successful bulk insert operation."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

        equities = [
            EquityProfile(symbol="AAPL", company_name="Apple Inc.", exchange=ExchangeType.NASDAQ),
            EquityProfile(symbol="GOOGL", company_name="Alphabet Inc.", exchange=ExchangeType.NASDAQ)
        ]

        result = await equity_repository.bulk_insert(equities)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.records_processed == 2
        assert result.records_failed == 0

    @pytest.mark.asyncio
    async def test_bulk_insert_partial_failure(self, equity_repository, mock_db_manager):
        """Test bulk insert with partial failures."""
        _, mock_conn = mock_db_manager
        # First insert succeeds, second fails
        mock_conn.execute.side_effect = [None, Exception("Duplicate key")]
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

        equities = [
            EquityProfile(symbol="AAPL", company_name="Apple Inc.", exchange=ExchangeType.NASDAQ),
            EquityProfile(symbol="GOOGL", company_name="Alphabet Inc.", exchange=ExchangeType.NASDAQ)
        ]

        result = await equity_repository.bulk_insert(equities)

        assert result.records_processed == 1
        assert result.records_failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_bulk_insert_empty_list(self, equity_repository, mock_db_manager):
        """Test bulk insert with empty list."""
        result = await equity_repository.bulk_insert([])

        assert result.success is True
        assert result.records_processed == 0

    @pytest.mark.asyncio
    async def test_search_success(self, equity_repository, mock_db_manager):
        """Test successful equity search."""
        _, mock_conn = mock_db_manager
        mock_rows = [
            {'symbol': 'AAPL', 'company_name': 'Apple Inc.', 'exchange': 'NASDAQ',
             'sector': 'Technology', 'industry': 'Consumer Electronics',
             'market_cap': Decimal('3000000000000'), 'employees': 150000,
             'description': 'Apple Inc.', 'website': 'https://www.apple.com',
             'country': 'US', 'currency': 'USD', 'is_etf': False,
             'is_actively_trading': True, 'created_at': datetime.utcnow(),
             'updated_at': datetime.utcnow()}
        ]
        mock_conn.fetch.return_value = mock_rows

        results = await equity_repository.search("Apple", limit=10)

        assert len(results) == 1
        assert results[0].company_name == "Apple Inc."

    @pytest.mark.asyncio
    async def test_search_empty_query(self, equity_repository, mock_db_manager):
        """Test search with empty query."""
        results = await equity_repository.search("", limit=10)

        assert results == []

    @pytest.mark.asyncio
    async def test_get_statistics(self, equity_repository, mock_db_manager):
        """Test getting equity profile statistics."""
        _, mock_conn = mock_db_manager
        mock_stats = {
            'total_count': 1000,
            'exchange_count': 5,
            'sector_count': 15,
            'etf_count': 100,
            'active_count': 950,
            'avg_market_cap': Decimal('50000000000'),
            'max_market_cap': Decimal('3000000000000'),
            'oldest_record': datetime(2020, 1, 1),
            'newest_update': datetime.utcnow()
        }
        mock_conn.fetchrow.return_value = mock_stats

        result = await equity_repository.get_statistics()

        assert result['total_count'] == 1000
        assert result['exchange_count'] == 5
        assert result['etf_count'] == 100


class TestFundataDataRepository:
    """Test fundata data repository operations."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.fixture
    def fundata_data_repository(self, mock_db_manager):
        """Create fundata data repository for testing."""
        db_manager, _ = mock_db_manager
        return FundataDataRepository(db_manager)

    @pytest.fixture
    def sample_fundata_data(self):
        """Create sample fundata data record."""
        return FundataDataRecord(
            InstrumentKey="TSX001",
            RecordId="REC001",
            Language="EN",
            LegalName="Sample Fund LP",
            FamilyName="Sample Fund Family",
            SeriesName="Series A",
            Company="Sample Management Inc.",
            InceptionDate=date(2020, 1, 1),
            Currency=CurrencyType.CAD,
            RecordState=RecordState.ACTIVE,
            source_file="fundata_sample.csv",
            file_hash="abc123"
        )

    @pytest.mark.asyncio
    async def test_create_success(self, fundata_data_repository, sample_fundata_data, mock_db_manager):
        """Test successful fundata data creation."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        result = await fundata_data_repository.create(sample_fundata_data)

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_instrument_key_found(self, fundata_data_repository, mock_db_manager):
        """Test retrieving fundata record by instrument key."""
        _, mock_conn = mock_db_manager
        mock_row = {
            'instrument_key': 'TSX001',
            'record_id': 'REC001',
            'language': 'EN',
            'legal_name': 'Sample Fund LP',
            'family_name': 'Sample Fund Family',
            'series_name': 'Series A',
            'company': 'Sample Management Inc.',
            'inception_date': date(2020, 1, 1),
            'change_date': None,
            'currency': 'CAD',
            'record_state': 'Active',
            'source_file': 'fundata_sample.csv',
            'file_hash': 'abc123',
            'processed_at': datetime.utcnow(),
            'additional_data': None
        }
        mock_conn.fetchrow.return_value = mock_row

        result = await fundata_data_repository.get_by_instrument_key("TSX001")

        assert result is not None
        assert result.InstrumentKey == "TSX001"
        assert result.LegalName == "Sample Fund LP"

    @pytest.mark.asyncio
    async def test_get_by_family_name(self, fundata_data_repository, mock_db_manager):
        """Test retrieving fundata records by family name."""
        _, mock_conn = mock_db_manager
        mock_rows = [
            {
                'instrument_key': 'TSX001',
                'record_id': 'REC001',
                'language': 'EN',
                'legal_name': 'Fund A',
                'family_name': 'Sample Family',
                'series_name': 'Series A',
                'company': 'Sample Management',
                'inception_date': date(2020, 1, 1),
                'change_date': None,
                'currency': 'CAD',
                'record_state': 'Active',
                'source_file': 'test.csv',
                'file_hash': 'hash1',
                'processed_at': datetime.utcnow(),
                'additional_data': None
            }
        ]
        mock_conn.fetch.return_value = mock_rows

        results = await fundata_data_repository.get_by_family_name("Sample Family")

        assert len(results) == 1
        assert results[0].FamilyName == "Sample Family"

    @pytest.mark.asyncio
    async def test_bulk_insert_success(self, fundata_data_repository, mock_db_manager):
        """Test successful bulk insert of fundata records."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

        records = [
            FundataDataRecord(InstrumentKey="TSX001", RecordId="REC001", source_file="test1.csv"),
            FundataDataRecord(InstrumentKey="TSX002", RecordId="REC002", source_file="test2.csv")
        ]

        result = await fundata_data_repository.bulk_insert(records)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.records_processed == 2


class TestFundataQuotesRepository:
    """Test fundata quotes repository operations."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager, mock_conn

    @pytest.fixture
    def fundata_quotes_repository(self, mock_db_manager):
        """Create fundata quotes repository for testing."""
        db_manager, _ = mock_db_manager
        return FundataQuotesRepository(db_manager)

    @pytest.fixture
    def sample_quote(self):
        """Create sample fundata quotes record."""
        return FundataQuotesRecord(
            InstrumentKey="TSX001",
            RecordId="REC001",
            Date=date(2023, 1, 15),
            NAVPS=Decimal("25.50"),
            NAVPSPennyChange=Decimal("0.25"),
            NAVPSPercentChange=Decimal("0.99"),
            PreviousDate=date(2023, 1, 14),
            PreviousNAVPS=Decimal("25.25"),
            RecordState=RecordState.ACTIVE,
            source_file="quotes_sample.csv"
        )

    @pytest.mark.asyncio
    async def test_create_success(self, fundata_quotes_repository, sample_quote, mock_db_manager):
        """Test successful quote creation."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None

        result = await fundata_quotes_repository.create(sample_quote)

        assert result is True
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_latest_quote(self, fundata_quotes_repository, mock_db_manager):
        """Test retrieving latest quote for instrument."""
        _, mock_conn = mock_db_manager
        mock_row = {
            'instrument_key': 'TSX001',
            'record_id': 'REC001',
            'date': date(2023, 1, 15),
            'navps': Decimal('25.50'),
            'navps_penny_change': Decimal('0.25'),
            'navps_percent_change': Decimal('0.99'),
            'previous_date': date(2023, 1, 14),
            'previous_navps': Decimal('25.25'),
            'record_state': 'Active',
            'change_date': None,
            'source_file': 'test.csv',
            'file_hash': 'hash1',
            'processed_at': datetime.utcnow(),
            'additional_data': None
        }
        mock_conn.fetchrow.return_value = mock_row

        result = await fundata_quotes_repository.get_latest_quote("TSX001")

        assert result is not None
        assert result.NAVPS == Decimal('25.50')
        assert result.Date == date(2023, 1, 15)

    @pytest.mark.asyncio
    async def test_get_quotes_date_range(self, fundata_quotes_repository, mock_db_manager):
        """Test retrieving quotes within date range."""
        _, mock_conn = mock_db_manager
        mock_rows = [
            {
                'instrument_key': 'TSX001',
                'record_id': 'REC001',
                'date': date(2023, 1, 15),
                'navps': Decimal('25.50'),
                'navps_penny_change': Decimal('0.25'),
                'navps_percent_change': Decimal('0.99'),
                'previous_date': date(2023, 1, 14),
                'previous_navps': Decimal('25.25'),
                'record_state': 'Active',
                'change_date': None,
                'source_file': 'test.csv',
                'file_hash': 'hash1',
                'processed_at': datetime.utcnow(),
                'additional_data': None
            }
        ]
        mock_conn.fetch.return_value = mock_rows

        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        results = await fundata_quotes_repository.get_quotes_date_range("TSX001", start_date, end_date)

        assert len(results) == 1
        assert results[0].Date == date(2023, 1, 15)

    @pytest.mark.asyncio
    async def test_bulk_insert_quotes(self, fundata_quotes_repository, mock_db_manager):
        """Test bulk insert of quotes."""
        _, mock_conn = mock_db_manager
        mock_conn.execute.return_value = None
        mock_conn.transaction.return_value.__aenter__ = AsyncMock()
        mock_conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)

        quotes = [
            FundataQuotesRecord(
                InstrumentKey="TSX001", RecordId="REC001", Date=date(2023, 1, 15),
                NAVPS=Decimal("25.50"), source_file="test1.csv"
            ),
            FundataQuotesRecord(
                InstrumentKey="TSX001", RecordId="REC001", Date=date(2023, 1, 16),
                NAVPS=Decimal("25.75"), source_file="test2.csv"
            )
        ]

        result = await fundata_quotes_repository.bulk_insert(quotes)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.records_processed == 2

    @pytest.mark.asyncio
    async def test_get_statistics(self, fundata_quotes_repository, mock_db_manager):
        """Test getting quotes statistics."""
        _, mock_conn = mock_db_manager
        mock_stats = {
            'total_quotes': 10000,
            'unique_instruments': 500,
            'earliest_date': date(2020, 1, 1),
            'latest_date': date(2023, 12, 31),
            'avg_navps': Decimal('25.50'),
            'max_navps': Decimal('100.00'),
            'min_navps': Decimal('10.00')
        }
        mock_conn.fetchrow.return_value = mock_stats

        result = await fundata_quotes_repository.get_statistics()

        assert result['total_quotes'] == 10000
        assert result['unique_instruments'] == 500


class TestFundataRepository:
    """Test unified fundata repository."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = MagicMock()
        return db_manager

    @pytest.fixture
    def fundata_repository(self, mock_db_manager):
        """Create unified fundata repository for testing."""
        return FundataRepository(mock_db_manager)

    @pytest.mark.asyncio
    async def test_get_fund_with_latest_quote_found(self, fundata_repository):
        """Test getting fund with latest quote when both exist."""
        # Mock successful data retrieval
        mock_fund_data = FundataDataRecord(
            InstrumentKey="TSX001", RecordId="REC001", source_file="test.csv"
        )
        mock_quote = FundataQuotesRecord(
            InstrumentKey="TSX001", RecordId="REC001", Date=date(2023, 1, 15),
            NAVPS=Decimal("25.50"), source_file="test.csv"
        )

        with patch.object(fundata_repository.data_repo, 'get_by_instrument_key', return_value=mock_fund_data), \
             patch.object(fundata_repository.quotes_repo, 'get_latest_quote', return_value=mock_quote):

            result = await fundata_repository.get_fund_with_latest_quote("TSX001")

            assert result is not None
            fund_data, latest_quote = result
            assert fund_data.InstrumentKey == "TSX001"
            assert latest_quote.NAVPS == Decimal("25.50")

    @pytest.mark.asyncio
    async def test_get_fund_with_latest_quote_not_found(self, fundata_repository):
        """Test getting fund with latest quote when fund doesn't exist."""
        with patch.object(fundata_repository.data_repo, 'get_by_instrument_key', return_value=None):
            result = await fundata_repository.get_fund_with_latest_quote("NOTFOUND")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_combined_statistics(self, fundata_repository, mock_db_manager):
        """Test getting combined statistics."""
        mock_conn = AsyncMock()
        mock_db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        mock_stats = {
            'total_funds': 1000,
            'total_quotes': 50000,
            'funds_with_quotes': 950,
            'earliest_quote': date(2020, 1, 1),
            'latest_quote': date(2023, 12, 31)
        }
        mock_conn.fetchrow.return_value = mock_stats

        result = await fundata_repository.get_combined_statistics()

        assert result['total_funds'] == 1000
        assert result['total_quotes'] == 50000


class TestRepositoryErrorHandling:
    """Test error handling across repository classes."""

    @pytest.fixture
    def mock_db_manager_with_errors(self):
        """Create mock database manager that generates errors."""
        db_manager = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("Database connection lost")
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        db_manager.pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        return db_manager

    @pytest.mark.asyncio
    async def test_equity_repository_connection_error(self, mock_db_manager_with_errors):
        """Test equity repository handling connection errors."""
        equity_repo = EquityRepository(mock_db_manager_with_errors)
        sample_equity = EquityProfile(
            symbol="TEST", company_name="Test Corp", exchange=ExchangeType.NYSE
        )

        # All operations should handle errors gracefully
        assert await equity_repo.create(sample_equity) is False
        assert await equity_repo.get_by_symbol("TEST") is None
        assert await equity_repo.update("TEST", {"market_cap": 1000}) is False
        assert await equity_repo.delete("TEST") is False

    @pytest.mark.asyncio
    async def test_fundata_repository_connection_error(self, mock_db_manager_with_errors):
        """Test fundata repository handling connection errors."""
        fundata_repo = FundataDataRepository(mock_db_manager_with_errors)
        sample_data = FundataDataRecord(
            InstrumentKey="TEST", RecordId="REC001", source_file="test.csv"
        )

        # All operations should handle errors gracefully
        assert await fundata_repo.create(sample_data) is False
        assert await fundata_repo.get_by_instrument_key("TEST") is None