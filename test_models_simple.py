#!/usr/bin/env python3
"""Simple test script to verify domain models work correctly."""

import sys
import os
from decimal import Decimal
from datetime import date

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages'))

def test_equity_profile():
    """Test EquityProfile model."""
    print("Testing EquityProfile model...")

    try:
        from sra_data.domain.models import EquityProfile, ExchangeType

        # Test valid data
        valid_data = {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "exchange": ExchangeType.NASDAQ,
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": Decimal("3000000000000")
        }

        equity = EquityProfile(**valid_data)
        print(f"✓ Valid EquityProfile created: {equity.symbol} - {equity.company_name}")

        # Test invalid data
        try:
            invalid_data = {
                "symbol": "",  # Empty symbol should fail
                "company_name": "Test",
                "exchange": "NYSE",
                "market_cap": -1000  # Negative market cap should fail
            }
            EquityProfile(**invalid_data)
            print("✗ Invalid data validation failed - should have raised error")
            return False
        except Exception as e:
            print(f"✓ Invalid data properly rejected: {e}")

        return True
    except Exception as e:
        print(f"✗ EquityProfile test failed: {e}")
        return False

def test_fundata_models():
    """Test Fundata models."""
    print("\nTesting Fundata models...")

    try:
        from sra_data.domain.models import FundataDataRecord, FundataQuotesRecord

        # Test FundataDataRecord
        data_record = FundataDataRecord(
            InstrumentKey="412682",
            RecordId="4",
            Language="EN",
            LegalName="MD Dividend Income Index",
            source_file="FundGeneralSeed.csv"
        )
        print(f"✓ FundataDataRecord created: {data_record.InstrumentKey}")

        # Test FundataQuotesRecord
        quotes_record = FundataQuotesRecord(
            InstrumentKey="4095",
            RecordId="26177",
            Date=date(2024, 1, 15),
            NAVPS=Decimal("11.58290000"),
            source_file="FundDailyNAVPSSeed.csv"
        )
        print(f"✓ FundataQuotesRecord created: {quotes_record.InstrumentKey} - NAVPS: {quotes_record.NAVPS}")

        return True
    except Exception as e:
        print(f"✗ Fundata models test failed: {e}")
        return False

def test_validation_errors():
    """Test validation error handling."""
    print("\nTesting validation errors...")

    try:
        from sra_data.domain.models import FundataQuotesRecord

        # Test negative NAVPS (should fail)
        try:
            FundataQuotesRecord(
                InstrumentKey="TEST",
                RecordId="1",
                Date=date(2024, 1, 15),
                NAVPS=Decimal("-1.00"),  # Negative NAVPS should fail
                source_file="test.csv"
            )
            print("✗ Negative NAVPS validation failed")
            return False
        except Exception as e:
            print(f"✓ Negative NAVPS properly rejected: {e}")

        return True
    except Exception as e:
        print(f"✗ Validation error test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running Domain Models Tests")
    print("="*50)

    success = True
    success &= test_equity_profile()
    success &= test_fundata_models()
    success &= test_validation_errors()

    print("\n" + "="*50)
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)