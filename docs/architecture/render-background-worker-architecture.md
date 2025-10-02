# SRA Data Processing Service - Render.com Deployment Architecture

**Standards Compliance**: This document follows architecture-documentation-standard.md and implements function-based design patterns with minimal FastAPI skeleton for deployment stability.

## Executive Summary

### System Purpose and Business Value
The SRA Data Processing Service is a backend data processing system designed for automated data ingestion, transformation, and preparation of financial market data. The system serves as the foundational data layer that feeds external client applications, focusing on data pipeline operations rather than client-facing APIs.

**Business Value**:
- Automated financial data pipeline leveraging existing FMP collection modules and fundata API sources
- Fundata denormalization: JSON preservation + flattened columns with DECIMAL(12,2) precision
- Array handling: Denormalized arrays create multiple rows (one row per array element)
- Historical data tracking: All API updates preserved with versioning
- Identifier sharing: Shared identifier space across data and quotes tables
- CSV seeding: Historical data from GitHub LFS for initial database seeding only
- API-driven updates: All current data sourced from Fundata API calls
- Modelized Pydantic views creation for client-ready data formats
- Background data processing with 99.9% uptime target
- Cost-effective deployment on Render.com with minimal resource usage

### Technology Stack Overview
- **Runtime**: Python 3.13+ with asyncio support
- **Framework**: FastAPI for API endpoints and background task coordination
- **Database**: PostgreSQL 15+ with DECIMAL(12,2) standardization and JSONB support
- **Queue System**: Redis for task queuing and caching
- **Deployment**: Render.com Background Worker service
- **Process Management**: Schedule library for cron-like operations
- **Data Processing**: Pandas, NumPy for denormalization and array explosion
- **API Integration**: Fundata API calls with retry mechanisms and historical versioning
- **CSV Processing**: GitHub LFS integration for historical seeding data only

### Core Architectural Principles
1. **Data Processing Focus**: Primary function is data ingestion and transformation, not client API services
2. **Function-Based Design**: Pure functions for all business logic, minimal classes
3. **Minimal FastAPI**: Skeleton FastAPI app only for deployment stability (health checks, basic endpoints)
4. **Database-Centric**: Focus on building modelized views and data transformations within the database
5. **External Client Ready**: Prepare data in client-ready formats for consumption by separate SRA FastAPI application
6. **Background Processing**: Optimized for batch operations and scheduled data refresh cycles

### Success Metrics and Performance Targets
- **Deployment Stability**: Render.com service remains active and healthy
- **Data Freshness**: Daily quotes updated within 1 hour of market close
- **Processing Speed**: 1,000 securities processed per minute for batch operations
- **Health Check Response**: <100ms for skeleton FastAPI health endpoints
- **Memory Usage**: <512MB baseline, <2GB during peak processing
- **Error Rate**: <0.1% for data processing operations
- **Database Views**: Modelized Pydantic views updated and accessible

## System Overview

### Visual System Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                    ┌─────────────────┐     │
│  │ Financial       │                    │ Git LFS         │     │
│  │ Modeling Prep   │                    │ Repository      │     │
│  │ API             │                    │ fundata/ CSVs   │     │
│  └─────────────────┘                    └─────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        RENDER.COM SERVICES                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ SRA_DATA        │              │ PostgreSQL      │           │
│  │ Processing      │◄────────────►│ Database        │           │
│  │ Service         │              │ (Raw + Views)   │           │
│  │ (Worker +       │              │                 │           │
│  │  Skeleton API)  │              │                 │           │
│  │                 │              │                 │           │
│  │ ┌─────────────┐ │              │ ┌─────────────┐ │           │
│  │ │ Health      │ │              │ │ Modelized   │ │           │
│  │ │ Endpoints   │ │              │ │ Pydantic    │ │           │
│  │ │ (Minimal)   │ │              │ │ Views       │ │           │
│  │ └─────────────┘ │              │ └─────────────┘ │           │
│  │                 │              │                 │           │
│  │ ┌─────────────┐ │              │                 │           │
│  │ │ Git LFS     │ │              │                 │           │
│  │ │ fundata/    │ │              │                 │           │
│  │ │ CSV Files   │ │              │                 │           │
│  │ └─────────────┘ │              │                 │           │
│  └─────────────────┘              └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘

                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EXTERNAL CLIENT                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Separate SRA FastAPI Application                        │    │
│  │ (Client-facing API consuming modelized views)           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow Patterns
```
 Local Sources → Data Ingestion → Raw Storage → Modelized Views → External Access
      │                │                │              │                │
      ▼                ▼                ▼              ▼                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ FMP API     │  │ Python      │  │ PostgreSQL  │ │ Pydantic    │ │ Separate    │
│ Git LFS     │  │ Processing  │  │ Raw Tables: │ │ View Models │ │ SRA Client  │
│ fundata/    │  │ - FMP API   │  │ - FMP data  │ │ (Database)  │ │ FastAPI App │
│ data/ CSVs  │  │ - Local CSV │  │ - fundata_  │ │             │ │             │
│ fundata/    │  │   processing│  │   data      │ │             │ │             │
│ quotes/ CSVs│  │ - Unified   │  │ - fundata_  │ │             │ │             │
│ (Local LFS) │  │   refresh   │  │   quotes    │ │             │ │             │
└─────────────┘  └─────────────┘  └─────────────┘ └─────────────┘ └─────────────┘
                                                         ▲
                                                         │
                                   ┌─────────────┐       │
                                   │ Skeleton    │───────┘
                                   │ FastAPI     │
                                   │ (Health)    │
                                   └─────────────┘
```

### Integration Points
1. **Existing FMP Code Integration**: The system leverages existing, battle-tested FMP data collection code located in `/FMP/` directory:
   - **Preserved Functionality**: All existing FMP modules remain unchanged to maintain stability
   - **Proven Patterns**: Existing threading (ThreadPoolExecutor), error handling, and API patterns
   - **Database Integration**: Direct PostgreSQL connections with established table schemas
   - **Rate Limiting**: Built-in threading limits (max_workers=10-20) with request chunking
   - **Worker Integration**: Current `worker.py` already references FMP scripts for scheduling
2. **Fundata Data Sources**:
   - **Historical Seeding**: CSV files from GitHub LFS for initial database population only
   - **Current Data**: Fundata API calls for all ongoing data updates and maintenance
   - **Data Strategy**: CSVs provide historical seed data, API provides current/live data
   - **Version Control**: All API updates tracked with historical preservation
3. **Git LFS Integration**: Large CSV files stored in Git LFS, accessed locally during deployment
   - **.gitattributes**: Configuration for CSV file LFS tracking
   - **Render deployment**: Automatic Git LFS file pulling during build
4. **Database**: Managed PostgreSQL with raw data tables and modelized views
   - **fundata_data** table: Denormalized flat table indexed by "Identifier" field
   - **fundata_quotes** table: Denormalized flat table indexed by "Identifier" field
5. **Unified Refresh**: Daily refresh schedule for both FMP and fundata at same time
6. **Client Access**: Database views accessible by separate SRA FastAPI application
7. **Monitoring**: Minimal health checks to prevent Render.com service suspension
8. **Logging**: Background processing logs for debugging and monitoring

### Security and Authentication Patterns
- **API Keys**: Environment variable storage with rotation capability
- **Database**: SSL-enforced connections with credential management
- **File Access**: Local file system access to Git LFS pulled CSV files
- **Git LFS**: Secure repository access during Render deployment
- **Network**: VPC-style isolation within Render.com infrastructure
- **Secrets**: Render.com secret management for sensitive configuration

## Technical Architecture Details

### Domain Layer

#### Pydantic Models and Validators
```python
# packages/sra_data/domain/models.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal

class EquityProfile(BaseModel):
    """Equity profile domain model with comprehensive validation."""
    symbol: str = Field(..., min_length=1, max_length=10)
    company_name: str = Field(..., min_length=1, max_length=255)
    exchange: str = Field(..., pattern="^(NYSE|NASDAQ|AMEX|TSX|TSXV)$")
    sector: Optional[str] = Field(None, max_length=100)
    industry: Optional[str] = Field(None, max_length=100)
    market_cap: Optional[Decimal] = Field(None, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().strip()

class FundataDataRecord(BaseModel):
    """
    Fundata data record model with denormalization strategy implementation.

    DENORMALIZATION REQUIREMENTS:
    - Column Strategy: Preserve JSON + flatten with DECIMAL(12,2) standardization
    - Array Handling: Multiple rows for array elements (one row per element)
    - Null Strategy: Keep nulls as NULL (no conversion or omission)
    - Identifier Scope: Shared across data and quotes tables
    - Data Sources: CSV for historical seeding, API for current updates
    """
    identifier: str = Field(..., min_length=1)  # Shared identifier space
    record_id: Optional[str] = Field(None)
    language: Optional[str] = Field(None, max_length=2)
    legal_name: Optional[str] = Field(None, max_length=255)
    family_name: Optional[str] = Field(None, max_length=150)
    series_name: Optional[str] = Field(None, max_length=150)
    company: Optional[str] = Field(None, max_length=100)
    inception_date: Optional[date] = Field(None)
    currency: Optional[str] = Field(None, max_length=3)
    record_state: Optional[str] = Field(None, max_length=20)
    change_date: Optional[date] = Field(None)

    # Denormalized numeric fields - ALL STANDARDIZED TO DECIMAL(12,2)
    management_fee: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    expense_ratio: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    minimum_investment: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    nav_value: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    yield_rate: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)

    # Array handling fields (for denormalized arrays)
    array_source_field: Optional[str] = Field(None, max_length=100)
    array_element_index: Optional[int] = Field(None, ge=0)
    array_element_value: Optional[str] = Field(None)

    # Data source and versioning
    data_source: str = Field(default="API", regex="^(CSV|API)$")
    source_file: Optional[str] = Field(None)
    api_version: Optional[str] = Field(None, max_length=20)
    data_version: int = Field(default=1, ge=1)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # JSON preservation (CRITICAL: preserve original structure)
    raw_json: Dict[str, Any] = Field(default_factory=dict)

    @validator('identifier')
    def validate_identifier(cls, v):
        return str(v).strip()

    @validator('raw_json')
    def validate_raw_json_required(cls, v):
        if not v:
            raise ValueError("raw_json is required to preserve original data structure")
        return v

class FundataQuotesRecord(BaseModel):
    """
    Fundata quotes record model with denormalization strategy implementation.

    DENORMALIZATION REQUIREMENTS:
    - Column Strategy: Preserve JSON + flatten with DECIMAL(12,2) standardization
    - Array Handling: Multiple rows for array elements (one row per element)
    - Null Strategy: Keep nulls as NULL (no conversion or omission)
    - Identifier Scope: Shared across data and quotes tables
    - Data Sources: CSV for historical seeding, API for current updates
    """
    identifier: str = Field(..., min_length=1)  # Shared identifier space with fundata_data
    record_id: Optional[str] = Field(None)
    date: Optional[date] = Field(None)

    # ALL NUMERIC FIELDS STANDARDIZED TO DECIMAL(12,2) - NO EXCEPTIONS
    navps: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    navps_penny_change: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    navps_percent_change: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    previous_navps: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    current_yield: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    current_yield_percent_change: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)

    # Additional standardized numeric fields
    price: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    volume_weighted_price: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    bid_price: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)
    ask_price: Optional[Decimal] = Field(None, decimal_places=2, max_digits=14)

    # Non-numeric fields
    previous_date: Optional[date] = Field(None)
    split: Optional[str] = Field(None, max_length=20)
    record_state: Optional[str] = Field(None, max_length=20)
    change_date: Optional[date] = Field(None)

    # Array handling fields (for denormalized arrays)
    array_source_field: Optional[str] = Field(None, max_length=100)
    array_element_index: Optional[int] = Field(None, ge=0)
    array_element_value: Optional[str] = Field(None)

    # Data source and versioning
    data_source: str = Field(default="API", regex="^(CSV|API)$")
    source_file: Optional[str] = Field(None)
    api_version: Optional[str] = Field(None, max_length=20)
    data_version: int = Field(default=1, ge=1)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # JSON preservation (CRITICAL: preserve original structure)
    raw_json: Dict[str, Any] = Field(default_factory=dict)

    @validator('identifier')
    def validate_identifier(cls, v):
        return str(v).strip()

    @validator('raw_json')
    def validate_raw_json_required(cls, v):
        if not v:
            raise ValueError("raw_json is required to preserve original data structure")
        return v

class ProcessingTask(BaseModel):
    """Background task definition model."""
    task_id: str
    task_type: str = Field(..., pattern="^(daily_quotes|weekly_fundamentals|csv_import|scoring)$")
    priority: int = Field(default=1, ge=1, le=5)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    scheduled_at: datetime
    max_retries: int = Field(default=3, ge=0, le=10)
    timeout_seconds: int = Field(default=3600, ge=60, le=14400)
```

#### Protocols and Interfaces
```python
# packages/sra_data/domain/protocols.py
from typing import Protocol, List, Optional, Dict, Any
from datetime import datetime

class DataFetcher(Protocol):
    """Protocol for external data fetching services."""
    async def fetch_equity_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Fetch equity data from external source."""
        ...

    async def validate_connection(self) -> bool:
        """Validate connection to external service."""
        ...

class DataRepository(Protocol):
    """Protocol for data storage operations."""
    async def upsert_equity_profiles(self, profiles: List[EquityProfile]) -> int:
        """Insert or update equity profiles."""
        ...

    async def get_symbols_by_exchange(self, exchange: str) -> List[str]:
        """Get all symbols for specific exchange."""
        ...

class CSVProcessor(Protocol):
    """Protocol for CSV file processing."""
    async def process_csv_file(self, file_path: str) -> List[FundataRecord]:
        """Process CSV file and return validated records."""
        ...

    async def list_local_csv_files(self, directory: str) -> List[str]:
        """List available local CSV files from Git LFS."""
        ...
```

### Service Layer

#### Business Logic Functions
```python
# packages/sra_data/services/data_processing.py
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import asyncio
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

async def process_daily_market_data(
    data_fetcher: DataFetcher,
    repository: DataRepository,
    symbols: List[str],
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Process daily market data for specified symbols.

    Args:
        data_fetcher: External data source interface
        repository: Database repository interface
        symbols: List of symbols to process
        batch_size: Number of symbols per batch

    Returns:
        Processing results with success/error counts
    """
    results = {
        'processed': 0,
        'errors': 0,
        'start_time': datetime.utcnow(),
        'batches': []
    }

    # Process symbols in batches to avoid API limits
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batch_start = datetime.utcnow()

        try:
            # Fetch data from external API
            raw_data = await data_fetcher.fetch_equity_data(batch)

            # Validate and transform data
            validated_records = [
                EquityProfile(**record) for record in raw_data
                if _validate_record(record)
            ]

            # Store in database
            stored_count = await repository.upsert_equity_profiles(validated_records)

            batch_result = {
                'batch_id': i // batch_size + 1,
                'symbols': len(batch),
                'processed': stored_count,
                'duration_seconds': (datetime.utcnow() - batch_start).total_seconds()
            }
            results['batches'].append(batch_result)
            results['processed'] += stored_count

        except Exception as e:
            logger.error(f"Batch processing error: {e}", extra={'batch': batch})
            results['errors'] += len(batch)

    results['end_time'] = datetime.utcnow()
    results['total_duration'] = (results['end_time'] - results['start_time']).total_seconds()

    return results

async def process_fundata_csv_files(
    csv_processor: CSVProcessor,
    repository: DataRepository,
    fundata_directory: str = "fundata",
    file_pattern: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process fundata CSV files from local Git LFS storage.

    Args:
        csv_processor: CSV processing interface
        repository: Database repository interface
        fundata_directory: Local fundata directory path
        file_pattern: Optional pattern to filter files

    Returns:
        Processing results with file counts and errors
    """
    results = {
        'files_processed': 0,
        'records_processed': 0,
        'errors': [],
        'start_time': datetime.utcnow()
    }

    try:
        # Get list of local CSV files from fundata/data and fundata/quotes
        data_files = await csv_processor.list_local_csv_files(f"{fundata_directory}/data")
        quotes_files = await csv_processor.list_local_csv_files(f"{fundata_directory}/quotes")
        available_files = data_files + quotes_files

        # Filter files if pattern provided
        if file_pattern:
            available_files = [
                f for f in available_files
                if file_pattern in f
            ]

        # Process each file
        for file_path in available_files:
            try:
                # Parse CSV file from local filesystem
                records = await csv_processor.process_csv_file(file_path)

                # Store records in database
                stored_count = await repository.upsert_fundata_records(records)

                results['files_processed'] += 1
                results['records_processed'] += stored_count

                logger.info(f"Processed local file {file_path}: {stored_count} records")

            except Exception as e:
                error_info = {
                    'file': file_path,
                    'error': str(e),
                    'timestamp': datetime.utcnow()
                }
                results['errors'].append(error_info)
                logger.error(f"Error processing {file_path}: {e}")

    except Exception as e:
        results['errors'].append({
            'error': f"Failed to list local CSV files: {str(e)}",
            'timestamp': datetime.utcnow()
        })

    results['end_time'] = datetime.utcnow()
    results['total_duration'] = (results['end_time'] - results['start_time']).total_seconds()

    return results

def _validate_record(record: Dict[str, Any]) -> bool:
    """Validate individual record before processing."""
    required_fields = ['symbol', 'company_name', 'exchange']
    return all(field in record and record[field] for field in required_fields)
```

### FMP Integration Layer

The system integrates with existing FMP data collection modules through wrapper functions that provide modern interface patterns while preserving the battle-tested collection code.

#### Existing FMP Code Architecture

The current FMP implementation consists of proven data collection modules organized by asset type:

**Equity Modules** (`/FMP/equity/`):
- `1.equity_profile.py` - Company profiles, ticker screening, and basic data
- `2.income.py` - Income statement data with quarterly/annual collection
- `3.balance.py` - Balance sheet data collection
- `4.cashflow.py` - Cash flow statement data
- `5.financial_ratio.py` - Financial ratios and metrics
- `6.key_metrics.py` - Key performance metrics
- `7.financial_scores.py` - Financial scoring data
- `8.equity_quotes.py` - Historical price data with date-range chunking
- `9-12.*.py` - Growth metrics across financial statements
- `13-14.equity_peers*.py` - Peer comparison data

**ETF Modules** (`/FMP/etfs/`):
- `1.etfs_profile.py` - ETF profiles with combined API calls
- `2.etfs_data.py` - ETF-specific data collection
- `3.etfs_quotes.py` - ETF price history
- `4-5.etfs_peers*.py` - ETF peer analysis

**Market Data** (`/FMP/`):
- `market_and_sector_quotes.py` - Market indices and sector ETF data

#### FMP Integration Wrapper Functions

Wrapper functions provide modern integration without modifying existing FMP code:

```python
# packages/sra_data/services/fmp_integration.py
import subprocess
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

async def execute_fmp_equity_profile_collection(
    force_refresh: bool = False,
    target_exchanges: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Execute existing FMP equity profile collection with wrapper integration.

    This function calls the existing 1.equity_profile.py script without modification,
    providing modern async interface and enhanced monitoring.
    """
    result = {
        'status': 'started',
        'start_time': datetime.utcnow(),
        'script_path': 'FMP/equity/1.equity_profile.py',
        'process_info': {}
    }

    try:
        # Execute existing FMP script as subprocess
        script_path = Path(__file__).parent.parent.parent / 'FMP' / 'equity' / '1.equity_profile.py'

        process = await asyncio.create_subprocess_exec(
            'python', str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=script_path.parent.parent.parent
        )

        stdout, stderr = await process.communicate()

        result.update({
            'status': 'completed' if process.returncode == 0 else 'error',
            'return_code': process.returncode,
            'stdout': stdout.decode() if stdout else None,
            'stderr': stderr.decode() if stderr else None,
            'end_time': datetime.utcnow(),
            'duration_seconds': (datetime.utcnow() - result['start_time']).total_seconds()
        })

        if process.returncode == 0:
            logger.info(f"FMP equity profile collection completed successfully")
        else:
            logger.error(f"FMP equity profile collection failed: {stderr.decode()}")

    except Exception as e:
        result.update({
            'status': 'exception',
            'error': str(e),
            'end_time': datetime.utcnow()
        })
        logger.error(f"Exception executing FMP equity profile collection: {e}")

    return result

async def execute_fmp_quotes_collection(
    asset_type: str = "equity",  # "equity" or "etf"
    symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Execute existing FMP quotes collection scripts.

    Calls either equity or ETF quotes script without modification.
    """
    script_map = {
        'equity': 'FMP/equity/8.equity_quotes.py',
        'etf': 'FMP/etfs/3.etfs_quotes.py'
    }

    if asset_type not in script_map:
        raise ValueError(f"Invalid asset_type: {asset_type}. Must be 'equity' or 'etf'")

    script_relative_path = script_map[asset_type]
    script_path = Path(__file__).parent.parent.parent / script_relative_path

    result = {
        'status': 'started',
        'start_time': datetime.utcnow(),
        'script_path': script_relative_path,
        'asset_type': asset_type
    }

    try:
        process = await asyncio.create_subprocess_exec(
            'python', str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=script_path.parent.parent.parent
        )

        stdout, stderr = await process.communicate()

        result.update({
            'status': 'completed' if process.returncode == 0 else 'error',
            'return_code': process.returncode,
            'stdout': stdout.decode() if stdout else None,
            'stderr': stderr.decode() if stderr else None,
            'end_time': datetime.utcnow(),
            'duration_seconds': (datetime.utcnow() - result['start_time']).total_seconds()
        })

    except Exception as e:
        result.update({
            'status': 'exception',
            'error': str(e),
            'end_time': datetime.utcnow()
        })
        logger.error(f"Exception executing FMP {asset_type} quotes collection: {e}")

    return result

async def execute_fmp_fundamentals_collection(
    data_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Execute existing FMP fundamental data collection scripts.

    Runs income, balance, cashflow, ratios, and metrics collection
    using existing proven scripts without modification.
    """
    if data_types is None:
        data_types = ['income', 'balance', 'cashflow', 'ratios', 'metrics', 'scores']

    script_map = {
        'income': 'FMP/equity/2.income.py',
        'balance': 'FMP/equity/3.balance.py',
        'cashflow': 'FMP/equity/4.cashflow.py',
        'ratios': 'FMP/equity/5.financial_ratio.py',
        'metrics': 'FMP/equity/6.key_metrics.py',
        'scores': 'FMP/equity/7.financial_scores.py'
    }

    results = {
        'status': 'started',
        'start_time': datetime.utcnow(),
        'collections': [],
        'total_processed': 0,
        'total_errors': 0
    }

    for data_type in data_types:
        if data_type not in script_map:
            logger.warning(f"Unknown data type: {data_type}, skipping")
            continue

        script_path = Path(__file__).parent.parent.parent / script_map[data_type]
        collection_result = {
            'data_type': data_type,
            'script_path': script_map[data_type],
            'start_time': datetime.utcnow()
        }

        try:
            process = await asyncio.create_subprocess_exec(
                'python', str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=script_path.parent.parent.parent
            )

            stdout, stderr = await process.communicate()

            collection_result.update({
                'status': 'completed' if process.returncode == 0 else 'error',
                'return_code': process.returncode,
                'end_time': datetime.utcnow(),
                'duration_seconds': (datetime.utcnow() - collection_result['start_time']).total_seconds()
            })

            if process.returncode == 0:
                results['total_processed'] += 1
            else:
                results['total_errors'] += 1
                collection_result['error'] = stderr.decode() if stderr else "Unknown error"

        except Exception as e:
            collection_result.update({
                'status': 'exception',
                'error': str(e),
                'end_time': datetime.utcnow()
            })
            results['total_errors'] += 1

        results['collections'].append(collection_result)

    results.update({
        'status': 'completed',
        'end_time': datetime.utcnow(),
        'total_duration': (datetime.utcnow() - results['start_time']).total_seconds()
    })

    return results

def get_fmp_collection_status() -> Dict[str, Any]:
    """
    Get status of existing FMP data collection without running scripts.

    Checks database for latest collection timestamps and data availability.
    """
    # This function queries existing FMP tables to determine collection status
    # without modifying or executing any FMP collection code
    pass

class FMPCollectionScheduler:
    """
    Scheduler integration for existing FMP scripts using current worker.py patterns.

    This class provides structured scheduling interface while maintaining
    the exact execution patterns from existing worker.py.
    """

    @staticmethod
    def get_daily_scripts() -> List[str]:
        """Return list of daily FMP scripts from existing worker.py."""
        return [
            'FMP/market_and_sector_quotes.py',
            'FMP/equity/8.equity_quotes.py',
            'FMP/etfs/3.etfs_quotes.py'
        ]

    @staticmethod
    def get_weekly_scripts() -> List[str]:
        """Return list of weekly FMP scripts from existing worker.py."""
        return [
            'FMP/equity/1.equity_profile.py',
            'FMP/equity/2.income.py',
            'FMP/equity/3.balance.py',
            'FMP/equity/4.cashflow.py',
            'FMP/equity/5.financial_ratio.py',
            'FMP/equity/6.key_metrics.py',
            'FMP/equity/7.financial_scores.py',
            'FMP/equity/9.balance_growth.py',
            'FMP/equity/10.cashflow_growth.py',
            'FMP/equity/11.financial_growth.py',
            'FMP/equity/12.income_growth.py',
            'FMP/equity/13.equity_peers.py',
            'FMP/etfs/1.etfs_profile.py',
            'FMP/etfs/2.etfs_data.py',
            'FMP/etfs/4.etfs_peers.py'
        ]
```

#### Integration Strategy

1. **Zero Changes to Existing FMP Code**: All FMP modules in `/FMP/` remain unchanged
2. **Wrapper Functions**: Modern async interfaces call existing scripts via subprocess
3. **Enhanced Monitoring**: Wrapper functions provide detailed execution tracking
4. **Error Handling**: Enhanced error capture and logging without modifying FMP code
5. **Scheduling Integration**: Existing `worker.py` scheduling preserved and enhanced
6. **Database Compatibility**: Existing FMP database schemas maintained exactly

#### Existing FMP Database Schema Integration

The existing FMP code already creates and populates comprehensive database schemas that perfectly align with the new architecture requirements:

**Core Tables Created by Existing FMP Code**:
- `equity_profile` - Company profiles and metadata (existing schema matches architecture)
- `equity_quotes` - Historical price data (existing schema matches architecture)
- `equity_income` - Income statement data with CREATE TABLE definition
- `equity_balance` - Balance sheet financial data
- `equity_cash_flow` - Cash flow statement data
- `equity_ratios` - Financial ratios and metrics
- `equity_key_metrics` - Key performance metrics
- `equity_financial_scores` - Financial scoring data
- `equity_peers` - Peer comparison data with CREATE TABLE definition
- `etfs_profile` - ETF profiles and metadata
- `etfs_quotes` - ETF historical price data
- `etfs_holdings` - ETF holdings composition data
- `etfs_country_weightings` - ETF country allocation data
- `etfs_peers` - ETF peer comparison data
- `market_and_sector_quotes` - Market indices and sector data
- `treasury` - Treasury bond and rate data

**Schema Compatibility Verification**:
1. **equity_profile**: Existing FMP schema matches architecture requirements exactly
2. **equity_quotes**: Existing FMP code uses pandas `to_sql()` to populate table compatible with TimescaleDB hypertable requirements
3. **All Growth Tables**: `equity_income_growth`, `equity_cashflow_growth`, `equity_financial_growth` tables already populated
4. **ETF Integration**: Complete ETF data pipeline with profiles, quotes, holdings, and peer data

**Integration Benefits**:
- **Zero Schema Migration**: No database changes needed
- **Proven Data Quality**: Existing FMP schemas contain production-tested data validation
- **Complete Data Pipeline**: All required tables already defined and populated
- **Performance Optimized**: Existing schemas include appropriate indexing patterns
- **TimescaleDB Ready**: Quote tables compatible with hypertable conversion

### Repository Layer

#### Data Access Functions
```python
# packages/sra_data/repositories/database.py
import asyncpg
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

async def create_connection_pool() -> asyncpg.Pool:
    """Create optimized database connection pool."""
    return await asyncpg.create_pool(
        host=os.getenv('DB_HOST'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        ssl='require',
        min_size=5,
        max_size=20,
        max_queries=50000,
        max_inactive_connection_lifetime=300,
        command_timeout=30
    )

async def initialize_database_schema(pool: asyncpg.Pool) -> bool:
    """
    Initialize database schema if tables don't exist.

    Args:
        pool: Database connection pool

    Returns:
        True if initialization successful
    """
    schema_sql = """
    -- Create equity_profile table
    CREATE TABLE IF NOT EXISTS equity_profile (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10) NOT NULL,
        company_name VARCHAR(255) NOT NULL,
        exchange VARCHAR(20) NOT NULL,
        sector VARCHAR(100),
        industry VARCHAR(100),
        market_cap DECIMAL(20,2),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(symbol, exchange)
    );

    -- Create indexes for performance
    CREATE INDEX IF NOT EXISTS idx_equity_profile_symbol ON equity_profile(symbol);
    CREATE INDEX IF NOT EXISTS idx_equity_profile_exchange ON equity_profile(exchange);
    CREATE INDEX IF NOT EXISTS idx_equity_profile_sector ON equity_profile(sector);

    -- Create fundata_records table
    CREATE TABLE IF NOT EXISTS fundata_records (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(20) NOT NULL,
        date DATE NOT NULL,
        open_price DECIMAL(15,4),
        high_price DECIMAL(15,4),
        low_price DECIMAL(15,4),
        close_price DECIMAL(15,4),
        volume BIGINT,
        source_file VARCHAR(255) NOT NULL,
        processed_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(symbol, date, source_file)
    );

    -- Create indexes for fundata_records
    CREATE INDEX IF NOT EXISTS idx_fundata_symbol_date ON fundata_records(symbol, date);
    CREATE INDEX IF NOT EXISTS idx_fundata_date ON fundata_records(date);
    CREATE INDEX IF NOT EXISTS idx_fundata_source ON fundata_records(source_file);

    -- Create task_execution_log table
    CREATE TABLE IF NOT EXISTS task_execution_log (
        id SERIAL PRIMARY KEY,
        task_type VARCHAR(50) NOT NULL,
        started_at TIMESTAMP NOT NULL,
        completed_at TIMESTAMP,
        status VARCHAR(20) NOT NULL DEFAULT 'running',
        records_processed INTEGER DEFAULT 0,
        errors_count INTEGER DEFAULT 0,
        details JSONB,
        duration_seconds INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_task_log_type_date ON task_execution_log(task_type, started_at);
    """

    try:
        async with pool.acquire() as conn:
            await conn.execute(schema_sql)
        logger.info("Database schema initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database schema: {e}")
        return False

async def upsert_equity_profiles(
    pool: asyncpg.Pool,
    profiles: List[Dict[str, Any]]
) -> int:
    """
    Insert or update equity profiles using efficient upsert.

    Args:
        pool: Database connection pool
        profiles: List of equity profile dictionaries

    Returns:
        Number of records processed
    """
    if not profiles:
        return 0

    upsert_sql = """
    INSERT INTO equity_profile (
        symbol, company_name, exchange, sector, industry, market_cap, updated_at
    ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
    ON CONFLICT (symbol, exchange)
    DO UPDATE SET
        company_name = EXCLUDED.company_name,
        sector = EXCLUDED.sector,
        industry = EXCLUDED.industry,
        market_cap = EXCLUDED.market_cap,
        updated_at = NOW();
    """

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                for profile in profiles:
                    await conn.execute(
                        upsert_sql,
                        profile['symbol'],
                        profile['company_name'],
                        profile['exchange'],
                        profile.get('sector'),
                        profile.get('industry'),
                        profile.get('market_cap')
                    )
        return len(profiles)
    except Exception as e:
        logger.error(f"Error upserting equity profiles: {e}")
        raise

async def check_database_seeded(pool: asyncpg.Pool) -> bool:
    """
    Check if database has been seeded with initial data.

    Args:
        pool: Database connection pool

    Returns:
        True if database contains data
    """
    try:
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM equity_profile")
            return result > 0
    except Exception as e:
        logger.error(f"Error checking database seed status: {e}")
        return False
```

### API Layer

#### FastAPI Route Handlers
```python
# packages/sra_data/api/routes.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncpg
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SRA Data Background Worker API",
    description="Financial data processing and monitoring API",
    version="1.0.0"
)

async def get_db_pool() -> asyncpg.Pool:
    """FastAPI dependency for database connection pool."""
    # This will be injected by the application startup
    return app.state.db_pool

@app.get("/health")
async def health_check(pool: asyncpg.Pool = Depends(get_db_pool)) -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    Performance target: <100ms response time
    """
    start_time = datetime.utcnow()
    health_status = {
        "status": "healthy",
        "timestamp": start_time,
        "checks": {}
    }

    try:
        # Database connectivity check
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Response time check
    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    health_status["response_time_ms"] = response_time
    health_status["checks"]["response_time"] = "healthy" if response_time < 100 else "slow"

    return health_status

@app.post("/tasks/trigger/{task_type}")
async def trigger_task(
    task_type: str,
    background_tasks: BackgroundTasks,
    parameters: Optional[Dict[str, Any]] = None,
    pool: asyncpg.Pool = Depends(get_db_pool)
) -> Dict[str, Any]:
    """
    Trigger background task execution.

    Args:
        task_type: Type of task (daily_quotes, weekly_fundamentals, csv_import)
        parameters: Optional task parameters

    Returns:
        Task execution details
    """
    valid_tasks = ['daily_quotes', 'weekly_fundamentals', 'csv_import', 'scoring']
    if task_type not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"Invalid task type. Must be one of: {valid_tasks}")

    task_id = f"{task_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    # Log task initiation
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO task_execution_log (task_type, started_at, status, details)
            VALUES ($1, $2, 'initiated', $3)
            """,
            task_type,
            datetime.utcnow(),
            parameters or {}
        )

    # Add to background tasks
    background_tasks.add_task(_execute_background_task, task_type, task_id, parameters or {})

    return {
        "task_id": task_id,
        "task_type": task_type,
        "status": "initiated",
        "parameters": parameters
    }

async def _execute_background_task(task_type: str, task_id: str, parameters: Dict[str, Any]):
    """Execute background task with comprehensive logging."""
    # Implementation will call appropriate service layer functions
    # This is where the 8-step process integration occurs
    pass
```

## Implementation Specifications

### File Structure
```
/Users/adam/dev/buckler/sra_data/
├── packages/
│   └── sra_data/
│       ├── domain/              # Pydantic models, protocols
│       │   ├── __init__.py
│       │   ├── models.py        # EquityProfile, FundataRecord, ProcessingTask
│       │   └── protocols.py     # DataFetcher, DataRepository, CSVProcessor
│       ├── services/            # Business logic functions
│       │   ├── __init__.py
│       │   ├── data_processing.py  # Core processing functions
│       │   ├── csv_processing.py   # Local CSV file handling
│       │   └── scheduling.py       # Task scheduling logic
│       ├── repositories/        # Data access functions
│       │   ├── __init__.py
│       │   ├── database.py      # PostgreSQL operations
│       │   └── local_files.py   # Local CSV file access
│       ├── api/                 # FastAPI routes
│       │   ├── __init__.py
│       │   ├── routes.py        # Health checks, task triggers
│       │   └── dependencies.py  # DI container setup
│       └── infrastructure/      # Deployment configurations
│           ├── __init__.py
│           ├── render_config.py # Render.com specific setup
│           └── database_init.py # Schema initialization
├── tests/                      # pytest-BDD test suite
│   ├── features/              # Gherkin feature files
│   ├── fixtures/              # Test fixtures and mocks
│   └── step_definitions/      # BDD step implementations
├── docs/
│   ├── architecture/          # This document
│   └── implementation/        # Implementation plans
├── fundata/                   # Git LFS stored CSV files
│   ├── data/                 # fundata_data CSV files (Git LFS)
│   ├── quotes/               # fundata_quotes CSV files (Git LFS)
│   └── documentation/        # API definitions and procedures
├── render/                    # Render.com deployment files
│   ├── render.yaml           # Service configuration
│   └── start.sh              # Startup script
├── requirements.txt          # Python dependencies
├── worker.py                 # Main application entry point
├── .gitattributes           # Git LFS configuration for CSV files
└── .env.example             # Environment configuration template
```

### Database Schema

#### Complete SQL Schema with Relationships
```sql
-- Equity profile table with comprehensive indexing
CREATE TABLE equity_profile (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    exchange VARCHAR(20) NOT NULL CHECK (exchange IN ('NYSE', 'NASDAQ', 'AMEX', 'TSX', 'TSXV')),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap DECIMAL(20,2) CHECK (market_cap >= 0),
    employees INTEGER CHECK (employees >= 0),
    description TEXT,
    website VARCHAR(255),
    ceo VARCHAR(100),
    country VARCHAR(3) DEFAULT 'US',
    currency VARCHAR(3) DEFAULT 'USD',
    is_etf BOOLEAN DEFAULT FALSE,
    is_actively_trading BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_symbol_exchange UNIQUE(symbol, exchange)
);

-- Performance indexes for equity_profile with clustering optimization
CREATE INDEX idx_equity_profile_symbol ON equity_profile(symbol);
CREATE INDEX idx_equity_profile_exchange ON equity_profile(exchange);
CREATE INDEX idx_equity_profile_sector ON equity_profile(sector);
CREATE INDEX idx_equity_profile_sector_symbol ON equity_profile(sector, symbol); -- Clustering index
CREATE INDEX idx_equity_profile_market_cap ON equity_profile(market_cap DESC);
CREATE INDEX idx_equity_profile_updated ON equity_profile(updated_at DESC);

-- Set default clustering for sector-based analytical queries
ALTER TABLE equity_profile CLUSTER ON idx_equity_profile_sector_symbol;

-- Fundata CSV records table
CREATE TABLE fundata_records (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(15,4) CHECK (open_price >= 0),
    high_price DECIMAL(15,4) CHECK (high_price >= 0),
    low_price DECIMAL(15,4) CHECK (low_price >= 0),
    close_price DECIMAL(15,4) CHECK (close_price >= 0),
    volume BIGINT CHECK (volume >= 0),
    adjusted_close DECIMAL(15,4),
    dividend_amount DECIMAL(10,4) DEFAULT 0,
    split_factor DECIMAL(10,6) DEFAULT 1,
    source_file VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64), -- For duplicate detection
    processed_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT unique_symbol_date_file UNIQUE(symbol, date, source_file)
);

-- Performance indexes for fundata_records
CREATE INDEX idx_fundata_symbol_date ON fundata_records(symbol, date DESC);
CREATE INDEX idx_fundata_date ON fundata_records(date DESC);
CREATE INDEX idx_fundata_source ON fundata_records(source_file);
CREATE INDEX idx_fundata_volume ON fundata_records(volume DESC);

-- Fundata data table (denormalized from fundata API with historical CSV seeding)
-- DENORMALIZATION STRATEGY: Preserve JSON + flatten columns with DECIMAL(12,2) standardization
-- ARRAY HANDLING: Create multiple rows for array elements (one row per element)
-- IDENTIFIER SCOPE: Shared identifier space across data and quotes tables
CREATE TABLE fundata_data (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(50) NOT NULL,  -- Shared identifier space with fundata_quotes
    record_id VARCHAR(50),
    language VARCHAR(2) DEFAULT 'EN',
    legal_name VARCHAR(255),
    family_name VARCHAR(150),
    series_name VARCHAR(150),
    company VARCHAR(100),
    inception_date DATE,
    currency VARCHAR(3) DEFAULT 'CAD',
    record_state VARCHAR(20) DEFAULT 'Active',
    change_date DATE,

    -- Denormalized numeric fields - STANDARDIZED TO DECIMAL(12,2)
    management_fee DECIMAL(12,2),        -- All numeric fields use DECIMAL(12,2)
    expense_ratio DECIMAL(12,2),         -- Consistent precision across all numerics
    minimum_investment DECIMAL(12,2),    -- No exceptions to DECIMAL(12,2) rule
    nav_value DECIMAL(12,2),            -- Absolute standardization
    yield_rate DECIMAL(12,2),           -- DECIMAL(12,2) for all numeric data

    -- Array handling fields (for arrays that create multiple rows)
    array_source_field VARCHAR(100),    -- Which array field this row represents
    array_element_index INTEGER,        -- Position in original array (0-based)
    array_element_value TEXT,           -- The specific array element value

    -- Data source and versioning
    data_source VARCHAR(20) DEFAULT 'API',  -- 'CSV' for historical seed, 'API' for current
    source_file VARCHAR(255),           -- File name for CSV seeds, API endpoint for API data
    api_version VARCHAR(20),            -- API version for tracking changes
    data_version INTEGER DEFAULT 1,     -- Version number for historical tracking
    processed_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(), -- Track when record was last modified

    -- JSON preservation (CRITICAL: preserve original structure)
    raw_json JSONB NOT NULL DEFAULT '{}',  -- Store complete original API response/CSV row

    CONSTRAINT unique_fundata_data_identifier_version UNIQUE(identifier, data_version, array_element_index)
);

-- Performance indexes for fundata_data with clustering optimization
CREATE INDEX idx_fundata_data_identifier ON fundata_data(identifier); -- Primary clustering index
CREATE INDEX idx_fundata_data_legal_name ON fundata_data(legal_name);
CREATE INDEX idx_fundata_data_company ON fundata_data(company);
CREATE INDEX idx_fundata_data_company_identifier ON fundata_data(company, identifier); -- Secondary clustering option
CREATE INDEX idx_fundata_data_currency ON fundata_data(currency);
CREATE INDEX idx_fundata_data_source ON fundata_data(source_file);
CREATE INDEX idx_fundata_data_processed ON fundata_data(processed_at DESC);

-- Set default clustering on identifier for data processing efficiency
ALTER TABLE fundata_data CLUSTER ON idx_fundata_data_identifier;

-- Fundata quotes table (denormalized from fundata API with historical CSV seeding)
-- DENORMALIZATION STRATEGY: Preserve JSON + flatten columns with DECIMAL(12,2) standardization
-- ARRAY HANDLING: Create multiple rows for array elements (one row per element)
-- IDENTIFIER SCOPE: Shared identifier space across data and quotes tables
-- NULL STRATEGY: Keep nulls as NULL (no conversion or omission)
CREATE TABLE fundata_quotes (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(50) NOT NULL,  -- Shared identifier space with fundata_data
    record_id VARCHAR(50),
    date DATE,

    -- ALL NUMERIC FIELDS STANDARDIZED TO DECIMAL(12,2) - NO EXCEPTIONS
    navps DECIMAL(12,2),                    -- Was DECIMAL(15,8), now DECIMAL(12,2)
    navps_penny_change DECIMAL(12,2),       -- Was DECIMAL(15,8), now DECIMAL(12,2)
    navps_percent_change DECIMAL(12,2),     -- Was DECIMAL(10,6), now DECIMAL(12,2)
    previous_navps DECIMAL(12,2),           -- Was DECIMAL(15,8), now DECIMAL(12,2)
    current_yield DECIMAL(12,2),            -- Was DECIMAL(10,6), now DECIMAL(12,2)
    current_yield_percent_change DECIMAL(12,2), -- Was DECIMAL(10,6), now DECIMAL(12,2)

    -- Additional standardized numeric fields
    price DECIMAL(12,2),                    -- Any price fields use DECIMAL(12,2)
    volume_weighted_price DECIMAL(12,2),    -- All numeric standardized
    bid_price DECIMAL(12,2),                -- Consistent DECIMAL(12,2) usage
    ask_price DECIMAL(12,2),                -- Absolute standardization

    -- Non-numeric fields (unchanged)
    previous_date DATE,
    split VARCHAR(20),
    record_state VARCHAR(20) DEFAULT 'Active',
    change_date DATE,

    -- Array handling fields (for arrays that create multiple rows)
    array_source_field VARCHAR(100),        -- Which array field this row represents
    array_element_index INTEGER,            -- Position in original array (0-based)
    array_element_value TEXT,               -- The specific array element value

    -- Data source and versioning
    data_source VARCHAR(20) DEFAULT 'API',  -- 'CSV' for historical seed, 'API' for current
    source_file VARCHAR(255),               -- File name for CSV seeds, API endpoint for API data
    api_version VARCHAR(20),                -- API version for tracking changes
    data_version INTEGER DEFAULT 1,         -- Version number for historical tracking
    processed_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),     -- Track when record was last modified

    -- JSON preservation (CRITICAL: preserve original structure)
    raw_json JSONB NOT NULL DEFAULT '{}',   -- Store complete original API response/CSV row

    CONSTRAINT unique_fundata_quotes_identifier_date_version UNIQUE(identifier, date, data_version, array_element_index)
);

-- Performance indexes for fundata_quotes (will become TimescaleDB hypertable)
CREATE INDEX idx_fundata_quotes_identifier ON fundata_quotes(identifier);
CREATE INDEX idx_fundata_quotes_date ON fundata_quotes(date DESC);
CREATE INDEX idx_fundata_quotes_navps ON fundata_quotes(navps DESC);
CREATE INDEX idx_fundata_quotes_source ON fundata_quotes(source_file);
CREATE INDEX idx_fundata_quotes_processed ON fundata_quotes(processed_at DESC);
CREATE INDEX idx_fundata_quotes_identifier_date ON fundata_quotes(identifier, date DESC);

-- Note: After TimescaleDB conversion, this table will be automatically chunked and clustered
-- TimescaleDB conversion command (run after table creation and data load):
-- SELECT create_hypertable('fundata_quotes', 'date', chunk_time_interval => INTERVAL '30 days');
-- SELECT add_dimension('fundata_quotes', 'identifier', number_partitions => 8);

-- Task execution logging
CREATE TABLE task_execution_log (
    id SERIAL PRIMARY KEY,
    task_type VARCHAR(50) NOT NULL,
    task_id VARCHAR(100) UNIQUE,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    records_processed INTEGER DEFAULT 0,
    errors_count INTEGER DEFAULT 0,
    error_details TEXT,
    performance_metrics JSONB, -- For storing timing and throughput data
    details JSONB,
    duration_seconds INTEGER,
    memory_usage_mb INTEGER,
    peak_memory_mb INTEGER
);

-- Indexes for task monitoring
CREATE INDEX idx_task_log_type_date ON task_execution_log(task_type, started_at DESC);
CREATE INDEX idx_task_log_status ON task_execution_log(status, started_at DESC);
CREATE INDEX idx_task_log_duration ON task_execution_log(duration_seconds DESC);

-- File processing tracker
CREATE TABLE file_processing_tracker (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(500) NOT NULL UNIQUE,
    file_hash VARCHAR(64) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    records_count INTEGER,
    first_processed_at TIMESTAMP DEFAULT NOW(),
    last_processed_at TIMESTAMP DEFAULT NOW(),
    processing_count INTEGER DEFAULT 1,
    status VARCHAR(20) DEFAULT 'completed'
        CHECK (status IN ('processing', 'completed', 'failed', 'skipped'))
);

CREATE INDEX idx_file_tracker_hash ON file_processing_tracker(file_hash);
CREATE INDEX idx_file_tracker_processed ON file_processing_tracker(last_processed_at DESC);
```

### Skeleton API Endpoints (Minimal)

#### Deployment Stability Endpoint Specifications
```python
# Skeleton endpoints for Render.com deployment stability
GET /
# Root endpoint for service identification
# Response time: <50ms

GET /health
# Basic health check to prevent Render.com suspension
# Response time: <100ms, Returns basic system health

GET /status
# Service status with data processing summary
# Response time: <200ms, Returns operational status and data counts
```

**Note**: This service provides minimal API endpoints for deployment stability only. The main client-facing API functionality will be implemented in a separate "sra" FastAPI application that consumes the modelized database views created by this service.

## Performance & Quality Specifications

### Core Performance Requirements

#### Existing FMP Rate Limiting and Control
- **Built-in Threading Limits**: Existing FMP code uses ThreadPoolExecutor with max_workers=10-20
- **Request Chunking**: Proven chunking patterns (20 symbols per request) for API efficiency
- **Natural Rate Limiting**: Threading limits provide effective rate control without additional mechanisms
- **Async Semaphore Implementation**: Token bucket pattern with semaphore controls
- **Rate Monitoring**: Real-time call counting with sliding window measurement
- **Overflow Protection**: Circuit breaker pattern when approaching limits
- **Implementation Strategy**:
  ```python
  # Rate limiting semaphore with 50 calls per second (3000/60)
  fmp_rate_limiter = asyncio.Semaphore(50)
  call_timestamps = collections.deque(maxlen=3000)

  async def rate_limited_fmp_call():
      async with fmp_rate_limiter:
          # Check sliding window for minute-based limits
          await enforce_sliding_window_limit()
          return await make_fmp_api_call()
  ```

#### Database Performance Optimization

##### Traditional PostgreSQL Clustering Strategy
- **Primary Indices**: All lookup columns (symbol, date, identifier) have B-tree indices
- **Composite Indices**: Multi-column indices for common query patterns
- **Physical Clustering**: Table-level clustering on frequently accessed columns
- **Maintenance Scheduling**: Periodic CLUSTER operations for optimal performance

##### TimescaleDB Time-Series Clustering Strategy
- **Hypertable Structure**: Automatic time-based partitioning for time-series tables
- **Chunk-Level Clustering**: Optimized chunk organization by time and space dimensions
- **Continuous Aggregates**: Real-time aggregations for analytical queries
- **Retention Policies**: Automated data lifecycle management

##### Table-Specific Clustering Implementation:

**1. equity_profile Table (Traditional Clustering)**
```sql
-- Create composite index for sector-symbol queries (most common lookup pattern)
CREATE INDEX idx_equity_profile_sector_symbol ON equity_profile(sector, symbol);

-- Cluster table on sector for analytical queries
ALTER TABLE equity_profile CLUSTER ON idx_equity_profile_sector_symbol;

-- Performance benefit: Groups similar companies together physically
-- Use case: Sector analysis, peer comparison queries
-- Maintenance: Weekly CLUSTER operation during off-peak hours
```

**2. equity_quotes Table (TimescaleDB Hypertable)**
```sql
-- Convert to TimescaleDB hypertable with 7-day chunks
SELECT create_hypertable('equity_quotes', 'date', chunk_time_interval => INTERVAL '7 days');

-- Add space dimension for symbol-based partitioning
SELECT add_dimension('equity_quotes', 'symbol', number_partitions => 16);

-- Create continuous aggregate for daily OHLCV summaries
CREATE MATERIALIZED VIEW equity_quotes_daily
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 day', date) AS day,
       symbol,
       first(open, date) AS open,
       max(high) AS high,
       min(low) AS low,
       last(close, date) AS close,
       sum(volume) AS volume
FROM equity_quotes
GROUP BY day, symbol;

-- Retention policy: Keep raw data for 2 years, aggregates for 10 years
SELECT add_retention_policy('equity_quotes', INTERVAL '2 years');
```

**3. fundata_data Table (Traditional Clustering)**
```sql
-- Create composite index for company-based queries
CREATE INDEX idx_fundata_data_company_identifier ON fundata_data(company, identifier);

-- Cluster on identifier for data processing efficiency
ALTER TABLE fundata_data CLUSTER ON idx_fundata_data_identifier;

-- Performance benefit: Groups related fund data together
-- Use case: Fund family analysis, identifier-based lookups
-- Maintenance: Weekly CLUSTER operation after fundata refresh
```

**4. fundata_quotes Table (TimescaleDB Hypertable)**
```sql
-- Convert to TimescaleDB hypertable with 30-day chunks for fund data
SELECT create_hypertable('fundata_quotes', 'date', chunk_time_interval => INTERVAL '30 days');

-- Add space dimension for identifier-based partitioning
SELECT add_dimension('fundata_quotes', 'identifier', number_partitions => 8);

-- Create continuous aggregate for monthly NAV performance
CREATE MATERIALIZED VIEW fundata_monthly_performance
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 month', date) AS month,
       identifier,
       first(navps, date) AS opening_nav,
       last(navps, date) AS closing_nav,
       max(navps) AS max_nav,
       min(navps) AS min_nav,
       avg(current_yield) AS avg_yield
FROM fundata_quotes
GROUP BY month, identifier;

-- Retention policy: Keep raw data for 5 years
SELECT add_retention_policy('fundata_quotes', INTERVAL '5 years');
```

**5. fmp_api_calls Table (TimescaleDB Hypertable)**
```sql
-- Create dedicated API call tracking table with TimescaleDB
CREATE TABLE fmp_api_calls (
    id SERIAL,
    endpoint VARCHAR(255) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    response_time_ms INTEGER,
    status_code INTEGER,
    symbols_requested INTEGER DEFAULT 1,
    rate_limit_remaining INTEGER,
    error_message TEXT
);

-- Convert to hypertable with 1-day chunks for API monitoring
SELECT create_hypertable('fmp_api_calls', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Continuous aggregate for rate limiting monitoring
CREATE MATERIALIZED VIEW api_usage_by_minute
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 minute', timestamp) AS minute,
       endpoint,
       count(*) AS call_count,
       avg(response_time_ms) AS avg_response_time,
       min(rate_limit_remaining) AS min_rate_limit
FROM fmp_api_calls
GROUP BY minute, endpoint;

-- Retention policy: Keep API logs for 90 days
SELECT add_retention_policy('fmp_api_calls', INTERVAL '90 days');
```

**6. csv_processing_records Table (Traditional Clustering)**
```sql
-- Create table for CSV processing tracking
CREATE TABLE csv_processing_records (
    id SERIAL PRIMARY KEY,
    file_path VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    processing_start TIMESTAMPTZ NOT NULL,
    processing_end TIMESTAMPTZ,
    records_processed INTEGER,
    status VARCHAR(20) DEFAULT 'processing',
    error_details JSONB
);

-- Create index for file-based lookups
CREATE INDEX idx_csv_processing_file_hash ON csv_processing_records(file_hash);

-- Cluster on processing_start for chronological access
CREATE INDEX idx_csv_processing_start ON csv_processing_records(processing_start);
ALTER TABLE csv_processing_records CLUSTER ON idx_csv_processing_start;

-- Performance benefit: Groups processing records by time for batch analysis
```

##### Clustering Maintenance Strategy:

**Weekly Maintenance Schedule (Sunday 2:00 AM)**
```sql
-- Traditional table clustering maintenance
CLUSTER equity_profile;
CLUSTER fundata_data;
CLUSTER csv_processing_records;

-- Update table statistics after clustering
ANALYZE equity_profile;
ANALYZE fundata_data;
ANALYZE csv_processing_records;

-- Refresh continuous aggregates
CALL refresh_continuous_aggregate('equity_quotes_daily', NULL, NULL);
CALL refresh_continuous_aggregate('fundata_monthly_performance', NULL, NULL);
CALL refresh_continuous_aggregate('api_usage_by_minute', NULL, NULL);
```

##### Performance Monitoring and Optimization:

**Query Performance Metrics**
```sql
-- Monitor clustering effectiveness
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del,
       last_vacuum, last_autovacuum, last_analyze, last_autoanalyze
FROM pg_stat_user_tables
WHERE tablename IN ('equity_profile', 'fundata_data', 'csv_processing_records');

-- Monitor TimescaleDB chunk health
SELECT hypertable_name, chunk_name, range_start, range_end,
       is_compressed, chunk_size
FROM timescaledb_information.chunks
WHERE hypertable_name IN ('equity_quotes', 'fundata_quotes', 'fmp_api_calls');

-- Monitor continuous aggregate refresh lag
SELECT view_name, materialization_hypertable,
       completed_threshold, invalidation_threshold
FROM timescaledb_information.continuous_aggregates;
```

#### Fundata Resilience and Backoff Strategy
- **Progressive Backoff**: Exponential backoff with jitter for all external calls
- **Timeout Handling**: Treat timeouts as rate limits with adaptive delays
- **Retry Logic**: Maximum 5 retries with increasing delays (1s, 2s, 4s, 8s, 16s)
- **Circuit Breaker**: Automatic fallback after consecutive failures
- **Implementation Pattern**:
  ```python
  async def resilient_fundata_fetch(url: str, max_retries: int = 5) -> Dict:
      for attempt in range(max_retries + 1):
          try:
              async with httpx.AsyncClient(timeout=30) as client:
                  response = await client.get(url)
                  return response.json()
          except (httpx.TimeoutException, httpx.RequestError) as e:
              if attempt == max_retries:
                  raise
              # Progressive backoff with jitter
              wait_time = (2 ** attempt) + random.uniform(0, 1)
              await asyncio.sleep(wait_time)
  ```

#### Database Load Management and Adaptive Throttling
- **Response Time Monitoring**: Track database query execution times
- **Adaptive Batch Sizing**: Reduce batch sizes when DB response times increase
- **Connection Pool Health**: Monitor pool utilization and connection timeouts
- **Dynamic Throttling**: Slow down inserts when database becomes overwhelmed
- **Implementation Framework**:
  ```python
  class AdaptiveThrottler:
      def __init__(self):
          self.current_batch_size = 1000
          self.response_times = collections.deque(maxlen=100)
          self.target_response_time = 0.5  # 500ms target

      async def adjust_batch_size(self, last_response_time: float):
          self.response_times.append(last_response_time)
          avg_response = sum(self.response_times) / len(self.response_times)

          if avg_response > self.target_response_time * 1.5:
              # Database is struggling - reduce batch size
              self.current_batch_size = max(100, self.current_batch_size // 2)
          elif avg_response < self.target_response_time * 0.7:
              # Database is handling well - increase batch size
              self.current_batch_size = min(2000, self.current_batch_size * 1.2)
  ```

### Performance Targets
- **Skeleton API Response Times**:
  - Health checks: <100ms (99th percentile)
  - Status endpoints: <200ms (95th percentile)
  - Service identification: <50ms (99th percentile)

- **Background Processing**:
  - Daily quotes: 1,000 symbols/minute processing rate
  - CSV import: 10,000 records/minute processing rate
  - fundata processing: Process 20+ CSV files within 30 minutes
  - fundata_data ingestion: 5,000 records/minute processing rate
  - fundata_quotes ingestion: 10,000 records/minute processing rate
  - Unified refresh: Complete FMP and fundata refresh within 2-hour window
  - Weekly fundamentals: Complete processing within 2-hour window
  - Memory usage: <512MB baseline, <2GB during peak processing
  - Database seeding: Complete initial seed within 30 minutes

- **Database Performance**:
  - Connection pool: 5-20 connections with <30s command timeout
  - Raw data inserts: >1000 records/second
  - View creation: Complete within 5 minutes after data update
  - Index usage: >95% for all production queries

- **Clustering Performance Targets**:
  - Traditional clustering operations: Complete within 15 minutes per table
  - TimescaleDB chunk creation: <100ms per chunk
  - Continuous aggregate refresh: <2 minutes for daily aggregates
  - Sector-based equity queries: <50ms response time (clustered benefit)
  - Time-range quote queries: <100ms for 1-year ranges (hypertable benefit)
  - Fund identifier lookups: <25ms (clustered fundata_data benefit)

- **Service Reliability**:
  - Render.com deployment: Remains active without suspension
  - Error rate: <0.1% for all data processing operations
  - Data consistency: 100% ACID compliance for raw data storage
  - Recovery time: <5 minutes for service restart
  - View availability: 99.9% uptime for external SRA client access

### Testing Requirements

#### pytest-BDD with Gherkin Scenarios
```gherkin
# tests/features/data_processing.feature
Feature: Daily Market Data Processing
  As a financial data system
  I want to process daily market data reliably
  So that users have access to current information

  Scenario: Successful daily data processing
    Given the system is healthy and connected to external APIs
    When I trigger daily quotes processing for 100 symbols
    Then the task should complete within 2 minutes
    And all 100 symbols should have updated quotes
    And the processing should be logged with metrics

  Scenario: Handling API rate limits
    Given the external API has rate limits
    When I process 1000 symbols with rate limiting
    Then the system should respect rate limits
    And all symbols should eventually be processed
    And no API errors should occur due to rate limiting

  Scenario: CSV file processing with validation
    Given there are 5 CSV files in the private service
    When I trigger CSV processing
    Then all files should be processed successfully
    And invalid records should be logged but not stored
    And duplicate records should be handled correctly
```

#### Test Coverage Targets
- **Overall Coverage**: >90% for all production code
- **Function Coverage**: 100% for all service layer functions
- **Branch Coverage**: >85% for all conditional logic
- **Integration Coverage**: 100% for all external API interactions
- **Performance Tests**: All endpoints must meet response time targets

## Implementation Guidance

**Implementation Focus**: This service should prioritize:
1. **Data Ingestion Functions**: Robust data fetching from external sources
2. **Database Seeding**: Efficient bulk data insertion and updates
3. **fundata Processing**: Denormalized flat tables for fundata_data and fundata_quotes
4. **Unified Refresh**: Daily refresh coordination for FMP and fundata sources
5. **View Creation**: Modelized Pydantic views for client consumption
6. **Background Processing**: Scheduled data refresh and processing
7. **Minimal API**: Just enough FastAPI to prevent deployment suspension

### Background Processing and Data Pipeline

#### Unified Refresh Schedule
The system implements a unified daily refresh schedule for both FMP API data and fundata CSV processing:

1. **Daily Refresh Timing**: Both FMP and fundata data refresh occurs at the same time daily
2. **Processing Order**:
   - FMP API data ingestion (equity profiles, quotes)
   - fundata CSV processing (fundata/data/ → fundata_data table)
   - fundata quotes processing (fundata/quotes/ → fundata_quotes table)
   - Modelized view recreation with updated data
3. **Refresh Coordination**: Single scheduler coordinates all data source refreshes
4. **Error Handling**: Independent error handling per data source without blocking others
5. **Performance Target**: Complete refresh cycle within 2-hour window

#### Fundata Processing Pipeline with Denormalization Strategy
```
CSV Seeding (Historical):     API Updates (Current):
fundata/data/ CSVs          → Fundata API calls → fundata_data table
fundata/quotes/ CSVs        → Fundata API calls → fundata_quotes table

DENORMALIZATION FEATURES:
- Column Strategy: JSON preservation + flattened columns (DECIMAL(12,2))
- Array Handling: Multiple rows per array element (1 row per element)
- Null Strategy: Preserve nulls as NULL (no conversion)
- Identifier Scope: Shared space across data/quotes tables
- Historical Tracking: All API updates versioned and preserved
- Schema Stability: No new CSVs, no schema changes
```

#### Array Denormalization Examples

**Example 1: Single Record with Array Field**
```json
// Original API Response
{
  "identifier": "FUND123",
  "holdings": ["AAPL", "GOOGL", "MSFT"],
  "name": "Tech Fund",
  "nav": 15.75
}

// Denormalized to Multiple Rows
Row 1: identifier=FUND123, array_source_field="holdings", array_element_index=0, array_element_value="AAPL", nav=15.75, raw_json={...}
Row 2: identifier=FUND123, array_source_field="holdings", array_element_index=1, array_element_value="GOOGL", nav=15.75, raw_json={...}
Row 3: identifier=FUND123, array_source_field="holdings", array_element_index=2, array_element_value="MSFT", nav=15.75, raw_json={...}
```

**Example 2: Multiple Arrays in Same Record**
```json
// Original API Response
{
  "identifier": "BOND456",
  "sectors": ["Technology", "Healthcare"],
  "ratings": ["AAA", "AA+"],
  "yield": 3.25
}

// Denormalized to Multiple Rows (6 rows total)
Row 1: identifier=BOND456, array_source_field="sectors", array_element_index=0, array_element_value="Technology", yield=3.25, raw_json={...}
Row 2: identifier=BOND456, array_source_field="sectors", array_element_index=1, array_element_value="Healthcare", yield=3.25, raw_json={...}
Row 3: identifier=BOND456, array_source_field="ratings", array_element_index=0, array_element_value="AAA", yield=3.25, raw_json={...}
Row 4: identifier=BOND456, array_source_field="ratings", array_element_index=1, array_element_value="AA+", yield=3.25, raw_json={...}
```

**Example 3: DECIMAL(12,2) Standardization**
```json
// Original API Response (various numeric precisions)
{
  "identifier": "EQUITY789",
  "price": 123.456789,          // High precision price
  "yield": 0.045,               // Low precision percentage
  "nav": "15.7",                // String numeric
  "volume": null                // Null value preserved
}

// Denormalized with DECIMAL(12,2) Standardization
Row 1: identifier=EQUITY789, price=123.46, yield=0.05, nav=15.70, volume=NULL, raw_json={...}
```

### Database Clustering Implementation Functions

#### Clustering Maintenance Functions
```python
# packages/sra_data/services/database_clustering.py
import asyncpg
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

async def perform_weekly_clustering(pool: asyncpg.Pool) -> Dict[str, Any]:
    """
    Execute weekly clustering maintenance for traditional PostgreSQL tables.

    Performance target: Complete within 45 minutes total.
    Schedule: Sunday 2:00 AM during low usage period.
    """
    results = {
        'start_time': datetime.utcnow(),
        'tables_processed': [],
        'errors': [],
        'total_duration_seconds': 0
    }

    # Tables requiring traditional clustering
    cluster_tables = [
        'equity_profile',
        'fundata_data',
        'csv_processing_records'
    ]

    try:
        async with pool.acquire() as conn:
            for table in cluster_tables:
                table_start = datetime.utcnow()

                try:
                    # Perform clustering operation
                    await conn.execute(f"CLUSTER {table}")

                    # Update statistics after clustering
                    await conn.execute(f"ANALYZE {table}")

                    # Log clustering metrics
                    table_duration = (datetime.utcnow() - table_start).total_seconds()

                    results['tables_processed'].append({
                        'table': table,
                        'duration_seconds': table_duration,
                        'status': 'completed'
                    })

                    logger.info(f"Clustered {table} in {table_duration:.2f} seconds")

                except Exception as e:
                    error_info = {
                        'table': table,
                        'error': str(e),
                        'timestamp': datetime.utcnow()
                    }
                    results['errors'].append(error_info)
                    logger.error(f"Failed to cluster {table}: {e}")

    except Exception as e:
        results['errors'].append({
            'operation': 'clustering_maintenance',
            'error': str(e),
            'timestamp': datetime.utcnow()
        })

    results['end_time'] = datetime.utcnow()
    results['total_duration_seconds'] = (
        results['end_time'] - results['start_time']
    ).total_seconds()

    return results

async def refresh_timescaledb_aggregates(pool: asyncpg.Pool) -> Dict[str, Any]:
    """
    Refresh TimescaleDB continuous aggregates.

    Performance target: Complete within 5 minutes.
    """
    results = {
        'start_time': datetime.utcnow(),
        'aggregates_refreshed': [],
        'errors': []
    }

    # Continuous aggregates to refresh
    aggregates = [
        'equity_quotes_daily',
        'fundata_monthly_performance',
        'api_usage_by_minute'
    ]

    try:
        async with pool.acquire() as conn:
            for aggregate in aggregates:
                try:
                    # Refresh continuous aggregate
                    await conn.execute(
                        f"CALL refresh_continuous_aggregate('{aggregate}', NULL, NULL)"
                    )

                    results['aggregates_refreshed'].append({
                        'aggregate': aggregate,
                        'status': 'refreshed',
                        'timestamp': datetime.utcnow()
                    })

                    logger.info(f"Refreshed continuous aggregate: {aggregate}")

                except Exception as e:
                    results['errors'].append({
                        'aggregate': aggregate,
                        'error': str(e),
                        'timestamp': datetime.utcnow()
                    })
                    logger.error(f"Failed to refresh {aggregate}: {e}")

    except Exception as e:
        results['errors'].append({
            'operation': 'aggregate_refresh',
            'error': str(e),
            'timestamp': datetime.utcnow()
        })

    results['end_time'] = datetime.utcnow()
    results['total_duration'] = (
        results['end_time'] - results['start_time']
    ).total_seconds()

    return results

async def monitor_clustering_performance(pool: asyncpg.Pool) -> Dict[str, Any]:
    """
    Monitor clustering effectiveness and performance metrics.

    Returns clustering health report for optimization decisions.
    """
    metrics = {
        'timestamp': datetime.utcnow(),
        'traditional_tables': [],
        'timescaledb_tables': [],
        'recommendations': []
    }

    try:
        async with pool.acquire() as conn:
            # Monitor traditional table clustering effectiveness
            traditional_stats = await conn.fetch("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del,
                       last_vacuum, last_autovacuum, last_analyze, last_autoanalyze,
                       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_stat_user_tables
                WHERE tablename IN ('equity_profile', 'fundata_data', 'csv_processing_records')
            """)

            for row in traditional_stats:
                table_metrics = dict(row)

                # Calculate fragmentation indicator
                fragmentation_score = (
                    (table_metrics['n_tup_upd'] + table_metrics['n_tup_del']) /
                    max(table_metrics['n_tup_ins'], 1)
                )

                table_metrics['fragmentation_score'] = fragmentation_score

                # Recommend reclustering if fragmentation is high
                if fragmentation_score > 0.3:
                    metrics['recommendations'].append({
                        'table': table_metrics['tablename'],
                        'action': 'recluster',
                        'reason': f'High fragmentation score: {fragmentation_score:.2f}'
                    })

                metrics['traditional_tables'].append(table_metrics)

            # Monitor TimescaleDB chunk health (if extension is available)
            try:
                chunk_stats = await conn.fetch("""
                    SELECT hypertable_name, chunk_name,
                           range_start, range_end, is_compressed,
                           pg_size_pretty(chunk_size) as chunk_size
                    FROM timescaledb_information.chunks
                    WHERE hypertable_name IN ('equity_quotes', 'fundata_quotes', 'fmp_api_calls')
                    ORDER BY range_start DESC
                    LIMIT 20
                """)

                for row in chunk_stats:
                    metrics['timescaledb_tables'].append(dict(row))

            except Exception as e:
                # TimescaleDB extension may not be installed yet
                metrics['timescaledb_tables'] = [{'note': 'TimescaleDB not yet configured'}]

    except Exception as e:
        metrics['error'] = str(e)
        logger.error(f"Failed to collect clustering metrics: {e}")

    return metrics
```

#### TimescaleDB Migration Functions
```python
# packages/sra_data/services/timescaledb_migration.py
import asyncpg
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

async def migrate_to_timescaledb(
    pool: asyncpg.Pool,
    table_name: str,
    time_column: str,
    chunk_interval: str,
    space_dimension: str = None,
    space_partitions: int = None
) -> Dict[str, Any]:
    """
    Migrate a regular PostgreSQL table to TimescaleDB hypertable.

    Args:
        pool: Database connection pool
        table_name: Name of table to migrate
        time_column: Time-based partitioning column
        chunk_interval: Chunk time interval (e.g., '7 days', '1 month')
        space_dimension: Optional space dimension column
        space_partitions: Number of space partitions

    Returns:
        Migration results with success/error status
    """
    results = {
        'table': table_name,
        'start_time': datetime.utcnow(),
        'status': 'started',
        'steps_completed': [],
        'error': None
    }

    try:
        async with pool.acquire() as conn:
            # Step 1: Check if TimescaleDB extension is available
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            results['steps_completed'].append('timescaledb_extension_enabled')

            # Step 2: Convert table to hypertable
            convert_sql = f"""
                SELECT create_hypertable('{table_name}', '{time_column}',
                                       chunk_time_interval => INTERVAL '{chunk_interval}')
            """
            await conn.execute(convert_sql)
            results['steps_completed'].append('hypertable_created')

            # Step 3: Add space dimension if specified
            if space_dimension and space_partitions:
                space_sql = f"""
                    SELECT add_dimension('{table_name}', '{space_dimension}',
                                       number_partitions => {space_partitions})
                """
                await conn.execute(space_sql)
                results['steps_completed'].append('space_dimension_added')

            # Step 4: Create optimized indexes for hypertable
            if table_name == 'equity_quotes':
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_equity_quotes_time_symbol
                    ON equity_quotes (date DESC, symbol)
                """)
                results['steps_completed'].append('hypertable_indexes_created')

            elif table_name == 'fundata_quotes':
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_fundata_quotes_time_identifier
                    ON fundata_quotes (date DESC, identifier)
                """)
                results['steps_completed'].append('hypertable_indexes_created')

            results['status'] = 'completed'
            logger.info(f"Successfully migrated {table_name} to TimescaleDB hypertable")

    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        logger.error(f"Failed to migrate {table_name} to TimescaleDB: {e}")

    results['end_time'] = datetime.utcnow()
    results['duration_seconds'] = (
        results['end_time'] - results['start_time']
    ).total_seconds()

    return results

async def setup_continuous_aggregates(pool: asyncpg.Pool) -> Dict[str, Any]:
    """
    Setup continuous aggregates for TimescaleDB hypertables.

    Performance benefit: Pre-computed aggregations for fast analytical queries.
    """
    results = {
        'start_time': datetime.utcnow(),
        'aggregates_created': [],
        'errors': []
    }

    aggregate_definitions = [
        {
            'name': 'equity_quotes_daily',
            'sql': """
                CREATE MATERIALIZED VIEW equity_quotes_daily
                WITH (timescaledb.continuous) AS
                SELECT time_bucket('1 day', date) AS day,
                       symbol,
                       first(open, date) AS open,
                       max(high) AS high,
                       min(low) AS low,
                       last(close, date) AS close,
                       sum(volume) AS volume,
                       count(*) AS data_points
                FROM equity_quotes
                GROUP BY day, symbol;
            """
        },
        {
            'name': 'fundata_monthly_performance',
            'sql': """
                CREATE MATERIALIZED VIEW fundata_monthly_performance
                WITH (timescaledb.continuous) AS
                SELECT time_bucket('1 month', date) AS month,
                       identifier,
                       first(navps, date) AS opening_nav,
                       last(navps, date) AS closing_nav,
                       max(navps) AS max_nav,
                       min(navps) AS min_nav,
                       avg(current_yield) AS avg_yield,
                       count(*) AS trading_days
                FROM fundata_quotes
                WHERE navps IS NOT NULL
                GROUP BY month, identifier;
            """
        }
    ]

    try:
        async with pool.acquire() as conn:
            for aggregate_def in aggregate_definitions:
                try:
                    await conn.execute(aggregate_def['sql'])

                    results['aggregates_created'].append({
                        'name': aggregate_def['name'],
                        'status': 'created',
                        'timestamp': datetime.utcnow()
                    })

                    logger.info(f"Created continuous aggregate: {aggregate_def['name']}")

                except Exception as e:
                    results['errors'].append({
                        'aggregate': aggregate_def['name'],
                        'error': str(e),
                        'timestamp': datetime.utcnow()
                    })
                    logger.error(f"Failed to create {aggregate_def['name']}: {e}")

    except Exception as e:
        results['errors'].append({
            'operation': 'setup_continuous_aggregates',
            'error': str(e),
            'timestamp': datetime.utcnow()
        })

    results['end_time'] = datetime.utcnow()

    return results
```

### Function-Based Code Examples

#### Data Processing Service Pattern
```python
# Example: Function-based service with FastAPI DI
from typing import List, Dict, Any
from fastapi import Depends
import asyncpg

async def process_equity_batch(
    symbols: List[str],
    db_pool: asyncpg.Pool = Depends(get_db_pool),
    batch_size: int = 100,
    timeout_seconds: int = 300
) -> Dict[str, Any]:
    """
    Process equity data batch with comprehensive error handling.

    Function-based design: Pure function with injected dependencies.
    Performance target: 1000 symbols per minute.
    """
    start_time = datetime.utcnow()
    results = {
        'processed': 0,
        'errors': 0,
        'duration_seconds': 0,
        'batches': []
    }

    try:
        # Process in optimized batches
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]

            # Fetch external data with retry logic
            external_data = await _fetch_with_retry(batch_symbols, max_retries=3)

            # Validate using Pydantic models
            validated_profiles = [
                EquityProfile(**record)
                for record in external_data
                if _is_valid_record(record)
            ]

            # Bulk database operation
            stored_count = await _bulk_upsert_profiles(db_pool, validated_profiles)

            results['processed'] += stored_count
            results['batches'].append({
                'batch_id': len(results['batches']) + 1,
                'symbols_count': len(batch_symbols),
                'stored_count': stored_count
            })

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        results['errors'] += 1
        raise

    finally:
        results['duration_seconds'] = (datetime.utcnow() - start_time).total_seconds()

    return results
```

### Configuration

#### Environment Variables
```bash
# Database configuration
DB_HOST=your-render-postgres-host
DB_PORT=5432
DB_NAME=sra_data_prod
DB_USER=your-db-user
DB_PASSWORD=your-secure-password
DATABASE_URL=postgresql://user:pass@host:port/db  # Render format

# Redis configuration
REDIS_URL=redis://your-render-redis:6379/0

# External API configuration
FMP_API_KEY=your-fmp-api-key
FMP_BASE_URL=https://financialmodelingprep.com/api/v3
FMP_RATE_LIMIT_PER_MINUTE=300

# Local file configuration
FUNDATA_BASE_PATH=fundata
FUNDATA_DATA_PATH=fundata/data
FUNDATA_QUOTES_PATH=fundata/quotes

# Application configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_TASKS=5
TASK_TIMEOUT_SECONDS=3600
HEALTH_CHECK_INTERVAL=30

# Performance tuning
DB_POOL_MIN_SIZE=5
DB_POOL_MAX_SIZE=20
BATCH_SIZE_EQUITY=100
BATCH_SIZE_CSV=1000

# Monitoring
ENABLE_METRICS=true
METRICS_EXPORT_INTERVAL=60
```

#### Render.com Configuration
```yaml
# render.yaml
services:
  - type: worker
    name: sra-data-worker
    env: python
    buildCommand: "git lfs pull && pip install -r requirements.txt"
    startCommand: "python worker.py"
    plan: standard  # Upgrade to standard for production
    envVars:
      - key: PYTHON_VERSION
        value: 3.13
      - key: LOG_LEVEL
        value: INFO
      - key: FUNDATA_BASE_PATH
        value: fundata
    autoDeploy: true

databases:
  - name: sra-data-db
    databaseName: sra_data_prod
    user: sra_user
    plan: standard  # Production database

  - name: sra-data-redis
    plan: starter
    type: redis
```

## Deployment

### Production Considerations

#### Render.com Service Configuration
1. **Background Worker Service**:
   - Plan: Standard (1GB RAM, 0.5 CPU)
   - Auto-scaling: Disabled (single worker instance)
   - Health checks: Custom endpoint `/health`
   - Environment: Production with SSL certificates
   - Git LFS: Automatic pulling during build process

2. **Database**:
   - Plan: Standard PostgreSQL (shared CPU, 1GB RAM)
   - Backup: Automatic daily backups with 7-day retention
   - SSL: Required for all connections
   - Connection pooling: pgBouncer enabled

3. **Redis Cache**:
   - Plan: Starter (25MB memory)
   - Persistence: RDB snapshots enabled
   - Key expiration: Configured for optimal memory usage

4. **Git LFS Integration**:
   - Build command includes `git lfs pull` for CSV files
   - fundata/ directory populated with LFS files during deployment
   - No additional storage service required

#### Monitoring and Logging Setup
```python
# packages/sra_data/infrastructure/monitoring.py
import logging
import structlog
from datetime import datetime
import os

def configure_logging():
    """Configure structured logging for production."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set log level from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(message)s'
    )

async def log_task_metrics(task_type: str, metrics: Dict[str, Any]):
    """Log task performance metrics for monitoring."""
    logger = structlog.get_logger()
    await logger.ainfo(
        "task_completed",
        task_type=task_type,
        duration_seconds=metrics.get('duration_seconds'),
        records_processed=metrics.get('processed'),
        error_count=metrics.get('errors'),
        memory_usage_mb=metrics.get('memory_usage'),
        timestamp=datetime.utcnow().isoformat()
    )
```

#### Git LFS Configuration and Management
```python
# packages/sra_data/infrastructure/git_lfs.py
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def verify_git_lfs_files(fundata_path: str = "fundata") -> Dict[str, Any]:
    """
    Verify that Git LFS files are properly pulled and available.

    Args:
        fundata_path: Path to fundata directory

    Returns:
        Verification results with file counts and status
    """
    results = {
        'status': 'success',
        'data_files': 0,
        'quotes_files': 0,
        'total_size_mb': 0,
        'missing_files': []
    }

    try:
        # Check data directory
        data_path = Path(fundata_path) / "data"
        if data_path.exists():
            data_files = list(data_path.glob("*.csv"))
            results['data_files'] = len(data_files)

            # Calculate total size
            for file_path in data_files:
                if file_path.is_file():
                    results['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                else:
                    results['missing_files'].append(str(file_path))

        # Check quotes directory
        quotes_path = Path(fundata_path) / "quotes"
        if quotes_path.exists():
            quotes_files = list(quotes_path.glob("*.csv"))
            results['quotes_files'] = len(quotes_files)

            # Calculate total size
            for file_path in quotes_files:
                if file_path.is_file():
                    results['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                else:
                    results['missing_files'].append(str(file_path))

        # Check if any files are missing
        if results['missing_files']:
            results['status'] = 'partial'
            logger.warning(f"Missing LFS files: {results['missing_files']}")

        logger.info(
            f"Git LFS verification: {results['data_files']} data files, "
            f"{results['quotes_files']} quotes files, "
            f"{results['total_size_mb']:.2f}MB total"
        )

    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
        logger.error(f"Git LFS verification failed: {e}")

    return results

def list_local_csv_files(directory_path: str) -> List[str]:
    """
    List all CSV files in a local directory.

    Args:
        directory_path: Path to directory to scan

    Returns:
        List of CSV file paths
    """
    csv_files = []
    try:
        path = Path(directory_path)
        if path.exists() and path.is_dir():
            csv_files = [str(p) for p in path.glob("*.csv")]
            logger.info(f"Found {len(csv_files)} CSV files in {directory_path}")
        else:
            logger.warning(f"Directory not found or not accessible: {directory_path}")
    except Exception as e:
        logger.error(f"Error listing CSV files in {directory_path}: {e}")

    return csv_files
```

#### Error Handling and Recovery
```python
# packages/sra_data/infrastructure/resilience.py
import asyncio
from typing import Callable, Any, Optional
import logging

logger = logging.getLogger(__name__)

async def with_retry(
    func: Callable,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Execute function with exponential backoff retry logic.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to retry on

    Returns:
        Function result or raises final exception
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()

        except exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break

            wait_time = backoff_base ** attempt
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}"
            )
            await asyncio.sleep(wait_time)

    logger.error(f"All {max_retries + 1} attempts failed")
    raise last_exception

async def graceful_shutdown(cleanup_tasks: list):
    """Handle graceful shutdown with cleanup."""
    logger.info("Initiating graceful shutdown")

    for task in cleanup_tasks:
        try:
            if asyncio.iscoroutinefunction(task):
                await task()
            else:
                task()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    logger.info("Graceful shutdown completed")
```

This architecture document provides comprehensive specifications for deploying the SRA Data Processing Service on Render.com with Git LFS integration for fundata storage. The service focuses on data ingestion, transformation, and database management with minimal client-facing functionality. Key architectural decisions include:

1. **Primary Purpose**: Data processing service, not client-facing API
2. **Skeleton FastAPI**: Minimal endpoints to prevent Render.com service suspension
3. **Git LFS Integration**: Local fundata CSV file storage within the project repository
4. **Simplified Deployment**: Single service architecture without separate file hosting
5. **Database Focus**: Raw data storage and modelized view creation
6. **External Client Ready**: Prepared for consumption by separate SRA FastAPI application
7. **Background Processing**: Automated data refresh and transformation cycles
8. **Cost Optimization**: Reduced from 3 services to 2 services by eliminating private file service

**Architecture Benefits**:
- **Simplified Deployment**: One fewer service to manage and monitor
- **Reduced Costs**: No separate file hosting service subscription required
- **Improved Reliability**: No network dependencies for CSV file access
- **Version Control**: Fundata files tracked with code changes
- **Faster Processing**: Local file system access instead of remote API calls

The implementation plan will detail the specific 8-step process for each component development phase, with emphasis on data pipeline functionality over API endpoints and Git LFS configuration.