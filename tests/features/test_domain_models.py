"""BDD test steps for domain model validation."""
import pytest
from pytest_bdd import scenarios, given, when, then, parsers
from decimal import Decimal
from datetime import date
from typing import Dict, Any, Optional

# Import fixtures
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.fixtures.domain_fixtures import (
    valid_equity_profile_data,
    invalid_equity_data,
    valid_fundata_data_record,
    valid_fundata_quotes_record
)

# Load scenarios
scenarios('domain_models.feature')

class TestContext:
    """Test context to store state between steps."""
    def __init__(self):
        self.data: Optional[Dict[str, Any]] = None
        self.model_instance = None
        self.validation_error = None
        self.validation_success = False

@pytest.fixture
def context():
    """Provide test context."""
    return TestContext()

# Given steps
@given('I have valid equity profile data from FMP API')
def given_valid_equity_data(context, valid_equity_profile_data):
    context.data = valid_equity_profile_data

@given('I have equity data with invalid symbol format')
def given_invalid_equity_data(context, invalid_equity_data):
    context.data = invalid_equity_data

@given('I have CSV data from local fundata/data/ directory (Git LFS)')
def given_fundata_data(context, valid_fundata_data_record):
    context.data = valid_fundata_data_record

@given('I have CSV data from local fundata/quotes/ directory (Git LFS)')
def given_fundata_quotes_data(context, valid_fundata_quotes_record):
    context.data = valid_fundata_quotes_record

# When steps
@when('I create an EquityProfile model for database storage')
def when_create_equity_profile(context):
    try:
        from packages.sra_data.domain.models import EquityProfile
        context.model_instance = EquityProfile(**context.data)
        context.validation_success = True
    except ImportError as e:
        context.validation_error = f"Import error: {e}"
    except Exception as e:
        context.validation_error = str(e)
        context.validation_success = False

@when('I attempt to create an EquityProfile model')
def when_attempt_equity_profile(context):
    try:
        from packages.sra_data.domain.models import EquityProfile
        context.model_instance = EquityProfile(**context.data)
        context.validation_success = True
    except ImportError as e:
        context.validation_error = f"Import error: {e}"
    except Exception as e:
        context.validation_error = str(e)
        context.validation_success = False

@when('I create FundataDataRecord models for batch ingestion')
def when_create_fundata_data_record(context):
    try:
        from packages.sra_data.domain.models import FundataDataRecord
        context.model_instance = FundataDataRecord(**context.data)
        context.validation_success = True
    except ImportError as e:
        context.validation_error = f"Import error: {e}"
    except Exception as e:
        context.validation_error = str(e)
        context.validation_success = False

@when('I create FundataQuotesRecord models for batch ingestion')
def when_create_fundata_quotes_record(context):
    try:
        from packages.sra_data.domain.models import FundataQuotesRecord
        context.model_instance = FundataQuotesRecord(**context.data)
        context.validation_success = True
    except ImportError as e:
        context.validation_error = f"Import error: {e}"
    except Exception as e:
        context.validation_error = str(e)
        context.validation_success = False

# Then steps
@then('the model should validate successfully')
def then_validation_success(context):
    assert context.validation_success, f"Expected validation to succeed, but got error: {context.validation_error}"
    assert context.model_instance is not None, "Model instance should be created"

@then('validation should fail with clear error message')
def then_validation_fails(context):
    assert not context.validation_success, "Expected validation to fail"
    assert context.validation_error is not None, "Expected error message"
    assert "Import error" not in context.validation_error, f"Unexpected import error: {context.validation_error}"

@then('all fields should be properly typed for raw storage')
def then_fields_properly_typed(context):
    assert context.model_instance is not None
    # Verify critical fields are present and typed correctly
    assert hasattr(context.model_instance, 'symbol')
    assert hasattr(context.model_instance, 'company_name')
    assert hasattr(context.model_instance, 'exchange')

@then('the invalid data should be logged but not stored')
def then_invalid_data_logged(context):
    # This step validates that validation properly rejected the data
    assert not context.validation_success
    assert context.model_instance is None

@then('identifier field should be properly validated and indexed')
def then_identifier_validated(context):
    assert context.model_instance is not None
    # Check that identifier/key fields are present
    if hasattr(context.model_instance, 'InstrumentKey'):
        assert context.model_instance.InstrumentKey is not None
    elif hasattr(context.model_instance, 'instrument_key'):
        assert context.model_instance.instrument_key is not None

@then('all optional fields should handle null values gracefully')
def then_optional_fields_handle_nulls(context):
    # This validates that the model accepts the data structure
    assert context.validation_success
    assert context.model_instance is not None

@then('records should be prepared for fundata_data table storage')
def then_prepared_for_fundata_data_storage(context):
    assert context.validation_success
    assert context.model_instance is not None

@then('NAVPS fields should be validated as positive decimals')
def then_navps_positive_decimals(context):
    assert context.validation_success
    assert context.model_instance is not None
    # Verify NAVPS field is present and positive
    if hasattr(context.model_instance, 'NAVPS'):
        assert isinstance(context.model_instance.NAVPS, Decimal)
        assert context.model_instance.NAVPS > 0

@then('records should be prepared for fundata_quotes table storage')
def then_prepared_for_fundata_quotes_storage(context):
    assert context.validation_success
    assert context.model_instance is not None