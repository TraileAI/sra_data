"""Test fixtures for data processing services."""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime, date
from decimal import Decimal
import asyncio
from typing import List, Dict, Any, Optional


@pytest.fixture
def sample_equity_data() -> List[Dict[str, Any]]:
    """Sample equity data from external API."""
    return [
        {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3000000000000
        },
        {
            "symbol": "MSFT",
            "company_name": "Microsoft Corporation",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 2800000000000
        },
        {
            "symbol": "GOOGL",
            "company_name": "Alphabet Inc.",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Internet Services",
            "market_cap": 1800000000000
        }
    ]


@pytest.fixture
def sample_fundata_data_csv() -> str:
    """Sample fundata data CSV content."""
    return """InstrumentKey,RecordId,Language,LegalName,FamilyName,SeriesName,Company,InceptionDate,Currency,RecordState,ChangeDate
412682,4,EN,MD Dividend Income Index,MD Funds,Dividend Income Index,MD Financial Management Inc.,2010-05-01,CAD,Active,2024-01-15
412683,5,EN,TD Balanced Growth Fund,TD Asset Management,Balanced Growth Fund,TD Asset Management Inc.,2008-03-15,CAD,Active,2024-01-15
412684,6,FR,Fonds équilibré croissance,TD Gestion d'actifs,Fonds équilibré croissance,TD Asset Management Inc.,2008-03-15,CAD,Active,2024-01-15"""


@pytest.fixture
def sample_fundata_quotes_csv() -> str:
    """Sample fundata quotes CSV content."""
    return """InstrumentKey,RecordId,Date,NAVPS,NAVPSPennyChange,NAVPSPercentChange,PreviousDate,PreviousNAVPS,RecordState,ChangeDate
4095,26177,2024-01-15,11.58290000,0.00020000,0.00172700,2024-01-14,11.58270000,Active,2024-01-15
4096,26178,2024-01-15,25.42150000,-0.00050000,-0.00196400,2024-01-14,25.42200000,Active,2024-01-15
4097,26179,2024-01-15,18.95670000,0.00120000,0.00633200,2024-01-14,18.95550000,Active,2024-01-15"""


@pytest.fixture
def invalid_fundata_csv() -> str:
    """Invalid CSV content for error testing."""
    return """InstrumentKey,RecordId,Date,NAVPS,NAVPSPennyChange,NAVPSPercentChange,PreviousDate,PreviousNAVPS,RecordState,ChangeDate
,26177,2024-01-15,11.58290000,0.00020000,0.00172700,2024-01-14,11.58270000,Active,2024-01-15
4096,26178,2024-01-15,-5.42150000,-0.00050000,-0.00196400,2024-01-14,25.42200000,Active,2024-01-15
4097,,2024-01-15,18.95670000,0.00120000,0.00633200,2024-01-14,18.95550000,Active,2024-01-15"""


@pytest.fixture
def mock_data_fetcher():
    """Mock external data fetcher."""
    fetcher = AsyncMock()
    fetcher.fetch_equity_data.return_value = [
        {"symbol": "AAPL", "company_name": "Apple Inc.", "exchange": "NASDAQ"},
        {"symbol": "MSFT", "company_name": "Microsoft Corporation", "exchange": "NASDAQ"}
    ]
    fetcher.fetch_market_data.return_value = {
        "AAPL": {"price": 180.50, "volume": 50000000, "market_cap": 3000000000000},
        "MSFT": {"price": 404.75, "volume": 25000000, "market_cap": 2800000000000}
    }
    return fetcher


@pytest.fixture
def mock_csv_processor():
    """Mock CSV processing service."""
    processor = AsyncMock()
    processor.process_file.return_value = {
        "records_processed": 100,
        "records_failed": 2,
        "processing_time": 1.5,
        "errors": ["Invalid NAVPS value", "Missing InstrumentKey"]
    }
    return processor


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiting service."""
    limiter = AsyncMock()
    limiter.acquire.return_value = True
    limiter.wait_time.return_value = 0.2  # 200ms wait
    return limiter


@pytest.fixture
def mock_retry_service():
    """Mock retry service for failure handling."""
    retry = AsyncMock()
    retry.execute_with_retry.return_value = {
        "success": True,
        "attempts": 2,
        "total_time": 3.2,
        "final_error": None
    }
    return retry


@pytest.fixture
def processing_config() -> Dict[str, Any]:
    """Configuration for data processing services."""
    return {
        "batch_size": 10,
        "max_concurrent": 5,
        "timeout_seconds": 30,
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "rate_limit_per_second": 5,
        "api_endpoints": {
            "fmp_base": "https://financialmodelingprep.com/api/v3/",
            "fundata_base": "https://api.fundata.com/"
        }
    }


@pytest.fixture
def large_symbol_list() -> List[str]:
    """Large list of symbols for performance testing."""
    # Generate 100 realistic stock symbols
    base_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ORCL", "CRM"]
    symbols = []
    for i in range(10):
        for base in base_symbols:
            symbols.append(f"{base}{i}" if i > 0 else base)
    return symbols


@pytest.fixture
def mock_database_service():
    """Mock database service for testing."""
    db_service = AsyncMock()
    db_service.store_equity_profiles.return_value = {"stored": 95, "failed": 5}
    db_service.store_fundata_records.return_value = {"stored": 1000, "failed": 12}
    db_service.get_last_update_time.return_value = datetime(2024, 1, 14, 18, 0, 0)
    return db_service


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collection service."""
    metrics = Mock()
    metrics.increment_counter = Mock()
    metrics.record_timing = Mock()
    metrics.record_gauge = Mock()
    metrics.get_summary.return_value = {
        "total_processed": 1000,
        "total_failed": 25,
        "avg_processing_time": 1.2,
        "success_rate": 97.5
    }
    return metrics


@pytest.fixture
async def processing_context():
    """Full processing context for integration testing."""
    context = {
        "start_time": datetime.utcnow(),
        "symbols_processed": 0,
        "files_processed": 0,
        "errors": [],
        "metrics": {},
        "rate_limit_hits": 0,
        "retry_attempts": 0
    }
    return context


@pytest.fixture
def sample_api_response():
    """Sample API response for external service testing."""
    return {
        "status_code": 200,
        "headers": {"content-type": "application/json"},
        "data": [
            {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "exchange": "NASDAQ",
                "marketCap": 3000000000000,
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "price": 180.50,
                "priceChange": 2.30,
                "priceChangePercent": 1.29
            }
        ]
    }


@pytest.fixture
def rate_limit_error_response():
    """Rate limit error response for testing."""
    return {
        "status_code": 429,
        "headers": {"retry-after": "60"},
        "error": "Rate limit exceeded"
    }


@pytest.fixture
def server_error_response():
    """Server error response for retry testing."""
    return {
        "status_code": 500,
        "error": "Internal server error"
    }