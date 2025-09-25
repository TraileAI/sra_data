"""Domain models for SRA Data processing service.

This module provides Pydantic models for data validation and processing
for equity profiles, fundata records, and other data types.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum


class ExchangeType(str, Enum):
    """Supported exchange types for equity trading."""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    TSX = "TSX"
    TSXV = "TSXV"
    CSE = "CSE"
    OTC = "OTC"


class RecordState(str, Enum):
    """Record state for fundata processing."""
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    SUSPENDED = "Suspended"
    DELISTED = "Delisted"


class CurrencyType(str, Enum):
    """Supported currency types."""
    USD = "USD"
    CAD = "CAD"
    EUR = "EUR"
    GBP = "GBP"


class EquityProfile(BaseModel):
    """Equity profile domain model with comprehensive validation.

    This model validates equity data from FMP API and other sources
    before database storage.
    """
    symbol: str = Field(..., min_length=1, max_length=10, description="Equity symbol")
    company_name: str = Field(..., min_length=1, max_length=255, description="Company name")
    exchange: ExchangeType = Field(..., description="Stock exchange")
    sector: Optional[str] = Field(None, max_length=100, description="Business sector")
    industry: Optional[str] = Field(None, max_length=100, description="Industry classification")
    market_cap: Optional[Decimal] = Field(None, ge=0, description="Market capitalization")
    employees: Optional[int] = Field(None, ge=0, description="Number of employees")
    description: Optional[str] = Field(None, description="Company description")
    website: Optional[str] = Field(None, max_length=255, description="Company website")
    country: str = Field(default="US", max_length=3, description="Country code")
    currency: CurrencyType = Field(default=CurrencyType.USD, description="Trading currency")
    is_etf: bool = Field(default=False, description="Is exchange-traded fund")
    is_actively_trading: bool = Field(default=True, description="Is actively trading")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('symbol')
    def validate_symbol(cls, v):
        """Normalize symbol to uppercase and validate format."""
        if not v or not v.strip():
            raise ValueError('Symbol cannot be empty')
        normalized = v.upper().strip()
        if not normalized.replace('.', '').replace('-', '').isalnum():
            raise ValueError('Symbol contains invalid characters')
        return normalized

    @validator('market_cap')
    def validate_market_cap(cls, v):
        """Ensure market cap is reasonable."""
        if v is not None:
            if v < 0:
                raise ValueError('Market cap cannot be negative')
            if v > Decimal('100000000000000'):  # $100T limit
                raise ValueError('Market cap exceeds reasonable limit')
        return v

    @validator('company_name')
    def validate_company_name(cls, v):
        """Validate company name format."""
        if not v or not v.strip():
            raise ValueError('Company name cannot be empty')
        return v.strip()


class FundataDataRecord(BaseModel):
    """Fundata general data record model for CSV processing.

    This model handles the fundata_data flat table structure for
    denormalized fund information storage.
    """
    # Core identifiers
    InstrumentKey: str = Field(..., min_length=1, max_length=20, description="Unique instrument identifier")
    RecordId: str = Field(..., min_length=1, max_length=20, description="Record identifier")

    # Descriptive fields
    Language: Optional[str] = Field(None, max_length=5, description="Language code")
    LegalName: Optional[str] = Field(None, max_length=500, description="Legal fund name")
    FamilyName: Optional[str] = Field(None, max_length=255, description="Fund family name")
    SeriesName: Optional[str] = Field(None, max_length=255, description="Series name")
    Company: Optional[str] = Field(None, max_length=255, description="Managing company")

    # Date fields
    InceptionDate: Optional[date] = Field(None, description="Fund inception date")
    ChangeDate: Optional[date] = Field(None, description="Last change date")

    # Financial details
    Currency: Optional[CurrencyType] = Field(None, description="Fund currency")

    # Status tracking
    RecordState: RecordState = Field(default=RecordState.ACTIVE, description="Record status")

    # Processing metadata
    source_file: str = Field(..., min_length=1, max_length=255, description="Source CSV file")
    file_hash: Optional[str] = Field(None, max_length=64, description="Source file hash")
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    # Additional flexible data as JSON
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional CSV columns as JSON")

    @validator('InstrumentKey')
    def validate_instrument_key(cls, v):
        """Validate instrument key format."""
        if not v or not v.strip():
            raise ValueError('InstrumentKey cannot be empty')
        return v.strip()

    @validator('RecordId')
    def validate_record_id(cls, v):
        """Validate record ID format."""
        if not v or not v.strip():
            raise ValueError('RecordId cannot be empty')
        return v.strip()

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


class FundataQuotesRecord(BaseModel):
    """Fundata quotes record model for CSV processing.

    This model handles the fundata_quotes flat table structure for
    denormalized quote and pricing information storage.
    """
    # Core identifiers
    InstrumentKey: str = Field(..., min_length=1, max_length=20, description="Unique instrument identifier")
    RecordId: str = Field(..., min_length=1, max_length=20, description="Record identifier")

    # Date and pricing data
    Date: date = Field(..., description="Quote date")
    NAVPS: Decimal = Field(..., ge=0, description="Net Asset Value Per Share")
    NAVPSPennyChange: Optional[Decimal] = Field(None, description="Penny change in NAVPS")
    NAVPSPercentChange: Optional[Decimal] = Field(None, description="Percent change in NAVPS")

    # Previous period data
    PreviousDate: Optional[date] = Field(None, description="Previous quote date")
    PreviousNAVPS: Optional[Decimal] = Field(None, ge=0, description="Previous NAVPS value")

    # Status tracking
    RecordState: RecordState = Field(default=RecordState.ACTIVE, description="Record status")
    ChangeDate: Optional[date] = Field(None, description="Last change date")

    # Processing metadata
    source_file: str = Field(..., min_length=1, max_length=255, description="Source CSV file")
    file_hash: Optional[str] = Field(None, max_length=64, description="Source file hash")
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    # Additional flexible data as JSON
    additional_data: Optional[Dict[str, Any]] = Field(None, description="Additional CSV columns as JSON")

    @validator('InstrumentKey')
    def validate_instrument_key(cls, v):
        """Validate instrument key format."""
        if not v or not v.strip():
            raise ValueError('InstrumentKey cannot be empty')
        return v.strip()

    @validator('RecordId')
    def validate_record_id(cls, v):
        """Validate record ID format."""
        if not v or not v.strip():
            raise ValueError('RecordId cannot be empty')
        return v.strip()

    @validator('NAVPS')
    def validate_navps(cls, v):
        """Validate NAVPS is positive and reasonable."""
        if v <= 0:
            raise ValueError('NAVPS must be positive')
        if v > Decimal('10000'):  # Reasonable upper limit
            raise ValueError('NAVPS value exceeds reasonable limit')
        return v

    @validator('PreviousNAVPS')
    def validate_previous_navps(cls, v):
        """Validate previous NAVPS is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError('PreviousNAVPS must be positive if provided')
        return v

    @validator('Date', 'PreviousDate')
    def validate_dates(cls, v):
        """Validate date is reasonable."""
        if v is not None:
            # Don't allow dates too far in the future or past
            today = date.today()
            if v > today:
                raise ValueError('Date cannot be in the future')
            if v.year < 1900:
                raise ValueError('Date cannot be before 1900')
        return v

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


class ProcessingResult(BaseModel):
    """Result model for data processing operations."""
    success: bool = Field(..., description="Processing success status")
    records_processed: int = Field(default=0, ge=0, description="Number of records processed")
    records_failed: int = Field(default=0, ge=0, description="Number of records failed")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")
    processing_time_seconds: Optional[float] = Field(None, ge=0, description="Processing time")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional processing metadata")

    @property
    def total_records(self) -> int:
        """Total records attempted."""
        return self.records_processed + self.records_failed

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_records == 0:
            return 100.0
        return (self.records_processed / self.total_records) * 100.0


class DataSourceConfig(BaseModel):
    """Configuration model for data sources."""
    source_type: str = Field(..., description="Data source type (fmp, fundata, etc)")
    connection_params: Dict[str, Any] = Field(..., description="Connection parameters")
    rate_limit_per_second: Optional[int] = Field(None, ge=1, description="Rate limit per second")
    timeout_seconds: int = Field(default=30, ge=1, description="Request timeout")
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    enabled: bool = Field(default=True, description="Is source enabled")

    class Config:
        """Pydantic configuration."""
        validate_assignment = True


# Type aliases for common use cases
EquityData = Union[EquityProfile, Dict[str, Any]]
FundataData = Union[FundataDataRecord, FundataQuotesRecord, Dict[str, Any]]
ProcessingData = Union[EquityData, FundataData]