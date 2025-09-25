import pytest
from decimal import Decimal
from datetime import datetime, date
from typing import Dict, Any

@pytest.fixture
def valid_equity_profile_data() -> Dict[str, Any]:
    """Valid equity profile data for testing."""
    return {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "exchange": "NASDAQ",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": Decimal("3000000000000")
    }

@pytest.fixture
def invalid_equity_data() -> Dict[str, Any]:
    """Invalid equity data for negative testing."""
    return {
        "symbol": "",  # Invalid empty symbol
        "company_name": "Test",
        "exchange": "INVALID",  # Invalid exchange
        "market_cap": -1000  # Invalid negative market cap
    }

@pytest.fixture
def valid_fundata_data_record() -> Dict[str, Any]:
    """Valid fundata data CSV record."""
    return {
        "InstrumentKey": "412682",
        "RecordId": "4",
        "Language": "EN",
        "LegalName": "MD Dividend Income Index",
        "FamilyName": "MD Funds",
        "SeriesName": "Dividend Income Index",
        "Company": "MD Financial Management Inc.",
        "InceptionDate": date(2010, 5, 1),
        "Currency": "CAD",
        "RecordState": "Active",
        "ChangeDate": date(2024, 1, 15),
        "source_file": "FundGeneralSeed.csv"
    }

@pytest.fixture
def valid_fundata_quotes_record() -> Dict[str, Any]:
    """Valid fundata quotes CSV record."""
    return {
        "InstrumentKey": "4095",
        "RecordId": "26177",
        "Date": date(2024, 1, 15),
        "NAVPS": Decimal("11.58290000"),
        "NAVPSPennyChange": Decimal("0.00020000"),
        "NAVPSPercentChange": Decimal("0.00172700"),
        "PreviousDate": date(2024, 1, 14),
        "PreviousNAVPS": Decimal("11.58270000"),
        "RecordState": "Active",
        "ChangeDate": date(2024, 1, 15),
        "source_file": "FundDailyNAVPSSeed.csv"
    }