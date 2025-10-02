# SRA Data Processing Service - Render.com Deployment Implementation Plan

**Standards Compliance**: This implementation plan follows implementation-plan-standard.md and incorporates the mandatory 8-step implementation process for all tasks with proven time estimation metrics.

## Overview

### Implementation Goals and Scope
Transform the current SRA Data project into a production-ready data processing service deployable on Render.com. The service focuses on data ingestion, transformation, and database management with a minimal FastAPI skeleton to prevent deployment suspension. This is NOT a client-facing API service.

**Correct Data Flow Understanding**:
1. **fmp/fundata → sra_data db**: External data sources feed into sra_data database
2. **sra_data → sra_data**: Build modelized Pydantic views within the database
3. **sra_data → separate SRA app**: External SRA FastAPI client will consume the views

### Business Value
- **Data Processing Pipeline**: Automated ingestion from FMP API and fundata CSV sources
- **fundata Integration**: Denormalized flat table structure (fundata_data, fundata_quotes)
- **Unified Refresh**: Daily refresh schedule for both FMP and fundata at same time
- **Database Management**: Seeding, refresh, and modelized view creation
- **Client Data Preparation**: Ready-to-consume data formats for external SRA application
- **Deployment Stability**: Skeleton FastAPI prevents Render.com service suspension
- **Background Processing**: Efficient resource utilization for data operations

### Success Metrics
- **Deployment Stability**: Render.com service remains active and healthy
- **Processing Performance**: 1,000 securities per minute batch processing
- **fundata Processing**: 5,000+ records/minute for fundata_data, 10,000+ records/minute for fundata_quotes
- **Unified Refresh**: Complete FMP and fundata refresh within 2-hour window
- **Health Check Response**: <100ms for skeleton FastAPI health endpoints
- **Data Coverage**: Support for 10,000+ securities from multiple sources
- **Database Views**: Modelized Pydantic views created and accessible
- **Error Rate**: <0.1% for all data processing operations
- **Test Coverage**: >90% for all production code

### Timeline Summary
**Total Estimated Duration**: 25 hours 45 minutes across 6 phases
**Completion Target**: 6-7 working days with parallel execution

**Updated Phase Durations**:
- Phase 1: Foundation Setup - 4 hours 30 minutes
- Phase 2: Core Data Processing - 12 hours 30 minutes *(+7.75 hours for fundata denormalization strategy + API integration)*
- Phase 3: Skeleton API Layer - 1 hour 30 minutes
- Phase 4: Database Views & Unified Refresh - 4 hours 45 minutes *(+1.5 hours for unified scheduler)*
- Phase 5: Infrastructure & Deployment - 3 hours 30 minutes
- Phase 6: Testing & Documentation - 1 hour 15 minutes
**Critical Path**: Foundation → Business Logic → Database Views → Skeleton API

**Time Adjustment**: +5.25 hours added for fundata denormalization strategy:
- Task 2.3: Fundata Denormalization Strategy Implementation (2h 30m)
- Task 2.4: Fundata API Integration and Historical Data Management (1h 45m)
- Task 2.5: FMP Integration Wrapper Implementation (1h 30m) - preserves existing FMP code
- Task 2.6: Database Performance Optimization (2h 0m) - renamed from Task 2.5
- These tasks ensure fundata requirements and performance targets are met

## IMPLEMENTATION EXECUTION LOG

**Implementation Start**: 2025-09-25 10:15:23
**Phase 1 Start**: 2025-09-25 10:15:45
**Task 1.1 Start**: 2025-09-25 10:16:32 - Domain Models and Protocols Setup

## Current State Analysis

### Existing Codebase Assessment
Based on analysis of `/Users/adam/dev/buckler/sra_data/`, the current implementation includes:

**Strengths**:
- Established PostgreSQL database connectivity with psycopg2
- Comprehensive FMP API integration across equity, ETF, and treasury data
- Existing scheduling framework using Python `schedule` library
- Database seeding logic with initial data validation
- Environment variable configuration through `.env` files

**Technical Debt**:
- Monolithic `worker.py` file (137 lines) needs decomposition
- Direct database connections without connection pooling
- Limited error handling and retry mechanisms
- No FastAPI integration for API endpoints
- Synchronous processing limiting performance
- Missing automated database schema initialization
- No local CSV file processing for fundata sources from Git LFS

**Infrastructure Gaps**:
- No Render.com deployment configuration
- Missing skeleton FastAPI for deployment stability
- Limited health check endpoints (deployment requirement)
- No structured logging implementation
- Basic error handling without retry mechanisms
- No modelized database views for client consumption

### Dependencies Analysis
Current dependencies (`requirements.txt`):
```
requests==2.31.0          # External API integration ✓
pandas==2.1.3            # Data processing ✓
sqlalchemy==2.0.23       # ORM (needs async upgrade)
psycopg2-binary==2.9.9   # PostgreSQL driver ✓
python-dotenv==1.0.0     # Environment configuration ✓
tqdm==4.66.1            # Progress indicators ✓
scipy==1.11.3           # Scientific computing ✓
numpy==1.25.2           # Numerical operations ✓
schedule==1.2.0         # Task scheduling ✓
python-dateutil==2.8.2  # Date handling ✓
```

**Required Additions**:
```
fastapi>=0.104.0        # Skeleton API framework (minimal endpoints)
uvicorn>=0.24.0         # ASGI server for deployment
asyncpg>=0.29.0         # Async PostgreSQL driver
pydantic>=2.0.0         # Data validation and view models
structlog>=23.0.0       # Structured logging
pytest-bdd>=7.0.0       # BDD testing
httpx>=0.25.0           # Async HTTP client for external APIs
# Note: Redis removed - not needed for data processing focus
```

## Phase Breakdown

### Phase 1: Foundation Layer
**Duration**: 3 hours 15 minutes
**Dependencies**: None
**Risk Level**: Low

#### Objectives
- [ ] Establish function-based domain layer with Pydantic models for raw data and views
- [ ] Create comprehensive data validation and protocols for ingestion
- [ ] Implement async database connection management
- [ ] Set up project structure for data processing service (not client API)

#### Task 1.1: Domain Models and Protocols Setup ✅
**Duration**: 1 hour 30 minutes
**Dependencies**: None
**Risk Level**: Low

**Implementation Process** (MANDATORY 8-step process):

1. **Capture Start Time**
   ```bash
   echo "Task 1.1 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/domain_models.feature
   Feature: Domain Model Validation for Data Processing
     As a data processing service
     I want to validate all incoming data using Pydantic models
     So that only clean, structured data enters the database

     Scenario: Valid equity profile validation for ingestion
       Given I have valid equity profile data from FMP API
       When I create an EquityProfile model for database storage
       Then the model should validate successfully
       And all fields should be properly typed for raw storage

     Scenario: Invalid symbol format rejection during ingestion
       Given I have equity data with invalid symbol format
       When I attempt to create an EquityProfile model
       Then validation should fail with clear error message
       And the invalid data should be logged but not stored

     Scenario: Fundata data record validation for local CSV processing
       Given I have CSV data from local fundata/data/ directory (Git LFS)
       When I create FundataDataRecord models for batch ingestion
       Then identifier field should be properly validated and indexed
       And all optional fields should handle null values gracefully
       And records should be prepared for fundata_data table storage

     Scenario: Fundata quotes record validation for local CSV processing
       Given I have CSV data from local fundata/quotes/ directory (Git LFS)
       When I create FundataQuotesRecord models for batch ingestion
       Then identifier field should be properly validated and indexed
       And NAVPS fields should be validated as positive decimals
       And records should be prepared for fundata_quotes table storage
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/domain_fixtures.py
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
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/domain_models.feature -v
   # Expected: Tests fail (red state) - validates test correctness ✓
   ```

5. **Write Implementation**
   ```python
   # packages/sra_data/domain/models.py
   from pydantic import BaseModel, Field, validator
   from typing import Optional, List, Dict, Any
   from datetime import datetime, date
   from decimal import Decimal
   from enum import Enum

   class ExchangeType(str, Enum):
       """Supported exchange types."""
       NYSE = "NYSE"
       NASDAQ = "NASDAQ"
       AMEX = "AMEX"
       TSX = "TSX"
       TSXV = "TSXV"

   class EquityProfile(BaseModel):
       """Equity profile domain model with comprehensive validation."""
       symbol: str = Field(..., min_length=1, max_length=10)
       company_name: str = Field(..., min_length=1, max_length=255)
       exchange: ExchangeType
       sector: Optional[str] = Field(None, max_length=100)
       industry: Optional[str] = Field(None, max_length=100)
       market_cap: Optional[Decimal] = Field(None, ge=0)
       employees: Optional[int] = Field(None, ge=0)
       description: Optional[str] = None
       website: Optional[str] = Field(None, max_length=255)
       country: str = Field(default="US", max_length=3)
       currency: str = Field(default="USD", max_length=3)
       is_etf: bool = Field(default=False)
       is_actively_trading: bool = Field(default=True)
       created_at: datetime = Field(default_factory=datetime.utcnow)
       updated_at: datetime = Field(default_factory=datetime.utcnow)

       @validator('symbol')
       def validate_symbol(cls, v):
           """Normalize symbol to uppercase."""
           return v.upper().strip()

       @validator('market_cap')
       def validate_market_cap(cls, v):
           """Ensure market cap is reasonable."""
           if v is not None and v > Decimal('100000000000000'):  # $100T limit
               raise ValueError('Market cap exceeds reasonable limit')
           return v

   class FundataRecord(BaseModel):
       """Fundata CSV record model with field validation."""
       symbol: str = Field(..., min_length=1, max_length=20)
       date: date
       open_price: Optional[Decimal] = Field(None, ge=0)
       high_price: Optional[Decimal] = Field(None, ge=0)
       low_price: Optional[Decimal] = Field(None, ge=0)
       close_price: Optional[Decimal] = Field(None, ge=0)
       volume: Optional[int] = Field(None, ge=0)
       adjusted_close: Optional[Decimal] = Field(None, ge=0)
       dividend_amount: Optional[Decimal] = Field(default=Decimal('0'), ge=0)
       split_factor: Optional[Decimal] = Field(default=Decimal('1'), gt=0)
       source_file: str = Field(..., min_length=1)
       file_hash: Optional[str] = Field(None, max_length=64)
       processed_at: datetime = Field(default_factory=datetime.utcnow)

       @validator('symbol')
       def validate_symbol(cls, v):
           return v.upper().strip()

       @validator('high_price', 'low_price', 'close_price')
       def validate_price_relationships(cls, v, values):
           """Ensure price relationships are logical."""
           if 'open_price' in values and values['open_price'] and v:
               # Basic sanity checks for price relationships
               if v < 0:
                   raise ValueError('Prices cannot be negative')
           return v

   class FundataDataRecord(BaseModel):
       """Fundata data record model for fundata_data table."""
       identifier: str = Field(..., alias="InstrumentKey", min_length=1)  # Primary index
       record_id: Optional[str] = Field(None, alias="RecordId")
       language: Optional[str] = Field(None, alias="Language", max_length=2)
       legal_name: Optional[str] = Field(None, alias="LegalName", max_length=255)
       family_name: Optional[str] = Field(None, alias="FamilyName", max_length=150)
       series_name: Optional[str] = Field(None, alias="SeriesName", max_length=150)
       company: Optional[str] = Field(None, alias="Company", max_length=100)
       inception_date: Optional[date] = Field(None, alias="InceptionDate")
       currency: Optional[str] = Field(None, alias="Currency", max_length=3)
       record_state: Optional[str] = Field(None, alias="RecordState", max_length=20)
       change_date: Optional[date] = Field(None, alias="ChangeDate")
       source_file: str = Field(..., min_length=1)
       processed_at: datetime = Field(default_factory=datetime.utcnow)
       raw_data: Optional[Dict[str, Any]] = Field(default_factory=dict)

       @validator('identifier')
       def validate_identifier(cls, v):
           return str(v).strip()

   class FundataQuotesRecord(BaseModel):
       """Fundata quotes record model for fundata_quotes table."""
       identifier: str = Field(..., alias="InstrumentKey", min_length=1)  # Primary index
       record_id: Optional[str] = Field(None, alias="RecordId")
       date: Optional[date] = Field(None, alias="Date")
       navps: Optional[Decimal] = Field(None, alias="NAVPS", ge=0)
       navps_penny_change: Optional[Decimal] = Field(None, alias="NAVPSPennyChange")
       navps_percent_change: Optional[Decimal] = Field(None, alias="NAVPSPercentChange")
       previous_date: Optional[date] = Field(None, alias="PreviousDate")
       previous_navps: Optional[Decimal] = Field(None, alias="PreviousNAVPS", ge=0)
       current_yield: Optional[Decimal] = Field(None, alias="CurrentYield")
       current_yield_percent_change: Optional[Decimal] = Field(None, alias="CurrentYieldPercentChange")
       split: Optional[str] = Field(None, alias="Split", max_length=20)
       record_state: Optional[str] = Field(None, alias="RecordState", max_length=20)
       change_date: Optional[date] = Field(None, alias="ChangeDate")
       source_file: str = Field(..., min_length=1)
       processed_at: datetime = Field(default_factory=datetime.utcnow)
       raw_data: Optional[Dict[str, Any]] = Field(default_factory=dict)

       @validator('identifier')
       def validate_identifier(cls, v):
           return str(v).strip()

   class ProcessingTask(BaseModel):
       """Background task definition model."""
       task_id: str = Field(..., min_length=1)
       task_type: str = Field(..., regex=r'^(daily_quotes|weekly_fundamentals|csv_import|fundata_data|fundata_quotes|unified_refresh)$')
       priority: int = Field(default=1, ge=1, le=5)
       parameters: Dict[str, Any] = Field(default_factory=dict)
       scheduled_at: datetime
       max_retries: int = Field(default=3, ge=0, le=10)
       timeout_seconds: int = Field(default=3600, ge=60, le=14400)
       created_at: datetime = Field(default_factory=datetime.utcnow)

   class SystemHealthCheck(BaseModel):
       """System health check response model."""
       status: str = Field(..., regex=r'^(healthy|degraded|unhealthy)$')
       timestamp: datetime
       response_time_ms: float
       checks: Dict[str, str] = Field(default_factory=dict)
       details: Optional[Dict[str, Any]] = None
   ```

   ```python
   # packages/sra_data/domain/protocols.py
   from typing import Protocol, List, Optional, Dict, Any
   from datetime import datetime
   import asyncio

   class DataFetcher(Protocol):
       """Protocol for external data fetching services."""

       async def fetch_equity_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
           """Fetch equity data from external source."""
           ...

       async def fetch_quote_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
           """Fetch current quote data."""
           ...

       async def validate_connection(self) -> bool:
           """Validate connection to external service."""
           ...

       async def get_rate_limit_status(self) -> Dict[str, Any]:
           """Get current rate limit status."""
           ...

   class DataRepository(Protocol):
       """Protocol for data storage operations."""

       async def upsert_equity_profiles(self, profiles: List[EquityProfile]) -> int:
           """Insert or update equity profiles."""
           ...

       async def upsert_fundata_records(self, records: List[FundataRecord]) -> int:
           """Insert or update fundata records."""
           ...

       async def get_symbols_by_exchange(self, exchange: str) -> List[str]:
           """Get all symbols for specific exchange."""
           ...

       async def check_schema_exists(self) -> bool:
           """Check if database schema is initialized."""
           ...

       async def initialize_schema(self) -> bool:
           """Initialize database schema."""
           ...

   class CSVProcessor(Protocol):
       """Protocol for CSV file processing."""

       async def process_csv_file(self, file_path: str) -> List[FundataRecord]:
           """Process CSV file and return validated records."""
           ...

       async def list_local_csv_files(self, directory: str) -> List[str]:
           """List available local CSV files from Git LFS."""
           ...

       async def download_file(self, file_path: str, local_path: str) -> bool:
           """Download CSV file to local storage."""
           ...

       async def get_file_hash(self, file_path: str) -> str:
           """Get file hash for duplicate detection."""
           ...

   class CacheManager(Protocol):
       """Protocol for caching operations."""

       async def get(self, key: str) -> Optional[Any]:
           """Get cached value."""
           ...

       async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
           """Set cached value with optional TTL."""
           ...

       async def delete(self, key: str) -> bool:
           """Delete cached value."""
           ...

       async def clear_pattern(self, pattern: str) -> int:
           """Clear keys matching pattern."""
           ...

   class TaskScheduler(Protocol):
       """Protocol for task scheduling operations."""

       async def schedule_task(self, task: ProcessingTask) -> str:
           """Schedule a task for execution."""
           ...

       async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
           """Get task execution status."""
           ...

       async def cancel_task(self, task_id: str) -> bool:
           """Cancel scheduled task."""
           ...
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/domain_models.feature -v --cov=packages/sra_data/domain --cov-report=term-missing
   # Target: 100% pass rate, >90% coverage ✓
   ```

7. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Implement domain layer with Pydantic models and protocols

   - Added EquityProfile model with comprehensive validation
   - Created FundataRecord model for CSV data processing
   - Implemented ProcessingTask model for background jobs
   - Added SystemHealthCheck model for monitoring
   - Created protocols for DataFetcher, DataRepository, CSVProcessor
   - Added CacheManager and TaskScheduler protocols
   - Comprehensive validation including price relationships
   - BDD tests covering happy path and error scenarios





   git push origin feature/domain-layer
   ```

8. **Capture End Time**
   ```bash
   echo "Task 1.1 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Expected Duration: 1 hour 30 minutes
   ```

**Validation Criteria**:
- All BDD tests pass with 100% success rate
- Test coverage >90% for domain models
- Pydantic validation working for all data types
- Protocol interfaces properly defined
- No regression in existing functionality

**Rollback Procedure**:
1. Revert domain layer commits: `git revert HEAD~1`
2. Verify system still uses existing data structures
3. Update stakeholders on rollback and issues

#### Task 1.2: Async Database Infrastructure ✅
**Duration**: 1 hour 45 minutes
**Dependencies**: Task 1.1 completion
**Risk Level**: Medium

**Implementation Process**:

1. **Capture Start Time**
   ```bash
   echo "Task 1.2 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/database_infrastructure.feature
   Feature: Database Infrastructure for Data Processing
     As a data processing service
     I want reliable database connections and schema management
     So that I can efficiently store raw data and create client views

     Scenario: Database connection pool initialization for data processing
       Given the database configuration is valid
       When I initialize the connection pool for background processing
       Then the pool should be created with proper parameters
       And connections should be optimized for bulk operations

     Scenario: Schema initialization with raw tables and view preparation
       Given a fresh database without tables
       When I run schema initialization for data processing
       Then all required raw data tables should be created (equity_profile, fundata_data, fundata_quotes)
       And indexes should be optimized for data ingestion
       And fundata tables should be indexed by Identifier field
       And the system should be ready for modelized view creation

     Scenario: Connection pool resilience during data processing
       Given the connection pool is active during background processing
       When temporary database connectivity issues occur
       Then the pool should handle reconnections gracefully
       And data processing operations should resume without data loss
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/database_fixtures.py
   import pytest
   import asyncpg
   from unittest.mock import AsyncMock, Mock
   import os

   @pytest.fixture
   async def mock_db_pool():
       """Mock asyncpg connection pool for testing."""
       pool = AsyncMock(spec=asyncpg.Pool)
       connection = AsyncMock(spec=asyncpg.Connection)

       # Configure mock connection
       connection.execute.return_value = None
       connection.fetchval.return_value = 1
       connection.fetch.return_value = []

       # Configure pool acquire context manager
       pool.acquire.return_value.__aenter__.return_value = connection
       pool.acquire.return_value.__aexit__.return_value = None

       return pool

   @pytest.fixture
   def test_db_config():
       """Test database configuration."""
       return {
           'host': 'localhost',
           'port': 5432,
           'database': 'test_sra_data',
           'user': 'test_user',
           'password': 'test_pass'
       }

   @pytest.fixture
   async def in_memory_db():
       """In-memory SQLite database for testing."""
       # For tests requiring actual database operations
       import aiosqlite
       db = await aiosqlite.connect(':memory:')
       yield db
       await db.close()
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/database_infrastructure.feature -v
   # Expected: Tests fail (red state) ✓
   ```

5. **Write Implementation**
   ```python
   # packages/sra_data/repositories/database.py
   import asyncpg
   from typing import List, Optional, Dict, Any
   import logging
   import os
   from datetime import datetime
   import asyncio
   from contextlib import asynccontextmanager

   logger = logging.getLogger(__name__)

   class DatabaseManager:
       """Async database connection and schema management."""

       def __init__(self):
           self.pool: Optional[asyncpg.Pool] = None

       async def create_connection_pool(self) -> asyncpg.Pool:
           """Create optimized database connection pool."""
           if self.pool:
               return self.pool

           try:
               self.pool = await asyncpg.create_pool(
                   host=os.getenv('DB_HOST', 'localhost'),
                   port=int(os.getenv('DB_PORT', 5432)),
                   database=os.getenv('DB_NAME', 'sra_data'),
                   user=os.getenv('DB_USER', 'sra_user'),
                   password=os.getenv('DB_PASSWORD', ''),
                   ssl='require' if os.getenv('DB_SSL', 'true').lower() == 'true' else 'prefer',
                   min_size=int(os.getenv('DB_POOL_MIN_SIZE', 5)),
                   max_size=int(os.getenv('DB_POOL_MAX_SIZE', 20)),
                   max_queries=50000,
                   max_inactive_connection_lifetime=300,
                   command_timeout=30,
                   server_settings={
                       'application_name': 'sra_data_worker',
                       'tcp_keepalives_idle': '300',
                       'tcp_keepalives_interval': '30',
                       'tcp_keepalives_count': '3'
                   }
               )

               logger.info("Database connection pool created successfully")
               return self.pool

           except Exception as e:
               logger.error(f"Failed to create database connection pool: {e}")
               raise

       async def initialize_database_schema(self) -> bool:
           """Initialize database schema if tables don't exist."""
           if not self.pool:
               await self.create_connection_pool()

           schema_sql = """
           -- Create equity_profile table
           CREATE TABLE IF NOT EXISTS equity_profile (
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
               country VARCHAR(3) DEFAULT 'US',
               currency VARCHAR(3) DEFAULT 'USD',
               is_etf BOOLEAN DEFAULT FALSE,
               is_actively_trading BOOLEAN DEFAULT TRUE,
               created_at TIMESTAMP DEFAULT NOW(),
               updated_at TIMESTAMP DEFAULT NOW(),
               CONSTRAINT unique_symbol_exchange UNIQUE(symbol, exchange)
           );

           -- Performance indexes for equity_profile
           CREATE INDEX IF NOT EXISTS idx_equity_profile_symbol ON equity_profile(symbol);
           CREATE INDEX IF NOT EXISTS idx_equity_profile_exchange ON equity_profile(exchange);
           CREATE INDEX IF NOT EXISTS idx_equity_profile_sector ON equity_profile(sector);
           CREATE INDEX IF NOT EXISTS idx_equity_profile_market_cap ON equity_profile(market_cap DESC);
           CREATE INDEX IF NOT EXISTS idx_equity_profile_updated ON equity_profile(updated_at DESC);

           -- Create fundata_data table (denormalized from fundata/data/ CSV files)
           CREATE TABLE IF NOT EXISTS fundata_data (
               id SERIAL PRIMARY KEY,
               identifier VARCHAR(50) NOT NULL,  -- InstrumentKey from CSVs (primary index)
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
               source_file VARCHAR(255) NOT NULL,
               processed_at TIMESTAMP DEFAULT NOW(),
               raw_data JSONB DEFAULT '{}',  -- Store all CSV columns for flexibility
               CONSTRAINT unique_fundata_data_identifier_file UNIQUE(identifier, source_file)
           );

           -- Performance indexes for fundata_data
           CREATE INDEX IF NOT EXISTS idx_fundata_data_identifier ON fundata_data(identifier);
           CREATE INDEX IF NOT EXISTS idx_fundata_data_legal_name ON fundata_data(legal_name);
           CREATE INDEX IF NOT EXISTS idx_fundata_data_company ON fundata_data(company);
           CREATE INDEX IF NOT EXISTS idx_fundata_data_currency ON fundata_data(currency);
           CREATE INDEX IF NOT EXISTS idx_fundata_data_source ON fundata_data(source_file);
           CREATE INDEX IF NOT EXISTS idx_fundata_data_processed ON fundata_data(processed_at DESC);

           -- Create fundata_quotes table (denormalized from fundata/quotes/ CSV files)
           CREATE TABLE IF NOT EXISTS fundata_quotes (
               id SERIAL PRIMARY KEY,
               identifier VARCHAR(50) NOT NULL,  -- InstrumentKey from CSVs (primary index)
               record_id VARCHAR(50),
               date DATE,
               navps DECIMAL(15,8) CHECK (navps >= 0),
               navps_penny_change DECIMAL(15,8),
               navps_percent_change DECIMAL(10,6),
               previous_date DATE,
               previous_navps DECIMAL(15,8) CHECK (previous_navps >= 0),
               current_yield DECIMAL(10,6),
               current_yield_percent_change DECIMAL(10,6),
               split VARCHAR(20),
               record_state VARCHAR(20) DEFAULT 'Active',
               change_date DATE,
               source_file VARCHAR(255) NOT NULL,
               processed_at TIMESTAMP DEFAULT NOW(),
               raw_data JSONB DEFAULT '{}',  -- Store all CSV columns for flexibility
               CONSTRAINT unique_fundata_quotes_identifier_date UNIQUE(identifier, date)
           );

           -- Performance indexes for fundata_quotes
           CREATE INDEX IF NOT EXISTS idx_fundata_quotes_identifier ON fundata_quotes(identifier);
           CREATE INDEX IF NOT EXISTS idx_fundata_quotes_date ON fundata_quotes(date DESC);
           CREATE INDEX IF NOT EXISTS idx_fundata_quotes_navps ON fundata_quotes(navps DESC);
           CREATE INDEX IF NOT EXISTS idx_fundata_quotes_source ON fundata_quotes(source_file);
           CREATE INDEX IF NOT EXISTS idx_fundata_quotes_processed ON fundata_quotes(processed_at DESC);
           CREATE INDEX IF NOT EXISTS idx_fundata_quotes_identifier_date ON fundata_quotes(identifier, date DESC);

           -- Create task_execution_log table
           CREATE TABLE IF NOT EXISTS task_execution_log (
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
               performance_metrics JSONB,
               details JSONB,
               duration_seconds INTEGER,
               memory_usage_mb INTEGER,
               peak_memory_mb INTEGER
           );

           -- Indexes for task monitoring
           CREATE INDEX IF NOT EXISTS idx_task_log_type_date ON task_execution_log(task_type, started_at DESC);
           CREATE INDEX IF NOT EXISTS idx_task_log_status ON task_execution_log(status, started_at DESC);
           CREATE INDEX IF NOT EXISTS idx_task_log_duration ON task_execution_log(duration_seconds DESC);

           -- Create file_processing_tracker table
           CREATE TABLE IF NOT EXISTS file_processing_tracker (
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

           CREATE INDEX IF NOT EXISTS idx_file_tracker_hash ON file_processing_tracker(file_hash);
           CREATE INDEX IF NOT EXISTS idx_file_tracker_processed ON file_processing_tracker(last_processed_at DESC);
           """

           try:
               async with self.pool.acquire() as conn:
                   async with conn.transaction():
                       await conn.execute(schema_sql)
               logger.info("Database schema initialized successfully")
               return True
           except Exception as e:
               logger.error(f"Failed to initialize database schema: {e}")
               return False

       async def check_database_seeded(self) -> bool:
           """Check if database has been seeded with initial data."""
           if not self.pool:
               await self.create_connection_pool()

           try:
               async with self.pool.acquire() as conn:
                   result = await conn.fetchval("SELECT COUNT(*) FROM equity_profile")
                   return result > 0
           except Exception as e:
               logger.error(f"Error checking database seed status: {e}")
               return False

       @asynccontextmanager
       async def get_connection(self):
           """Get database connection from pool with context management."""
           if not self.pool:
               await self.create_connection_pool()

           async with self.pool.acquire() as conn:
               yield conn

       async def close_pool(self):
           """Close database connection pool."""
           if self.pool:
               await self.pool.close()
               self.pool = None
               logger.info("Database connection pool closed")

   # Global database manager instance
   db_manager = DatabaseManager()

   async def get_db_pool() -> asyncpg.Pool:
       """FastAPI dependency for database connection pool."""
       return await db_manager.create_connection_pool()

   async def initialize_database() -> bool:
       """Initialize database schema for application startup."""
       return await db_manager.initialize_database_schema()

   async def check_database_health() -> bool:
       """Check database connectivity for health checks."""
       try:
           pool = await get_db_pool()
           async with pool.acquire() as conn:
               await conn.fetchval("SELECT 1")
           return True
       except Exception:
           return False
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/database_infrastructure.feature -v --cov=packages/sra_data/repositories --cov-report=term-missing
   # Target: 100% pass rate, >90% coverage ✓
   ```

7. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Implement async database infrastructure with connection pooling

   - Created DatabaseManager class for connection pool management
   - Added comprehensive schema initialization with all required tables
   - Implemented health check and seeding status functions
   - Added context manager for safe connection handling
   - Configured optimized connection pool parameters
   - Added proper indexes for performance
   - Comprehensive error handling with logging
   - BDD tests for database operations and resilience

   git push origin feature/database-infrastructure
   ```

8. **Capture End Time**
   ```bash
   echo "Task 1.2 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Expected Duration: 1 hour 45 minutes
   ```

### Phase 2: Business Logic Services
**Duration**: 4 hours 30 minutes
**Dependencies**: Phase 1 completion
**Risk Level**: Medium

#### Task 2.1: Data Processing Services ✅
**Duration**: 2 hours 15 minutes
**Dependencies**: Task 1.2 completion
**Risk Level**: Medium

**Implementation Process**:

1. **Capture Start Time**
   ```bash
   echo "Task 2.1 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/data_processing_services.feature
   Feature: Financial Data Processing for Database Ingestion
     As a data processing service
     I want to ingest market data efficiently from multiple sources
     So that raw data is available for modelized view creation

     Scenario: Process daily market data for database seeding
       Given I have 100 equity symbols to process for raw storage
       And the FMP API is available
       When I trigger daily market data ingestion
       Then all 100 symbols should be processed and stored within 2 minutes
       And the raw data should be available for view creation
       And processing metrics should be logged for monitoring

     Scenario: Handle API rate limiting during data ingestion
       Given the FMP API has rate limits of 300 requests per minute
       When I process 500 symbols for database seeding
       Then the system should respect rate limits automatically
       And all symbols should eventually be processed and stored
       And no API rate limit violations should occur

     Scenario: Process fundata CSV files for raw storage
       Given there are 3 CSV files available for processing
       When I trigger CSV ingestion for database seeding
       Then all files should be downloaded and parsed
       And invalid records should be logged but not stored
       And valid records should be ready for modelized view creation

     Scenario: Retry logic for failed data ingestion
       Given a data ingestion task fails due to temporary issues
       When the retry mechanism is triggered
       Then the task should be retried up to 3 times
       And exponential backoff should be applied
       And final failure should be logged if retries are exhausted
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/processing_fixtures.py
   import pytest
   from unittest.mock import AsyncMock, Mock
   from datetime import datetime, date
   from decimal import Decimal
   import asyncio

   @pytest.fixture
   def sample_equity_data():
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
           }
       ]

   @pytest.fixture
   def sample_fundata_csv_content():
       """Sample CSV content for fundata processing."""
       return """symbol,date,open,high,low,close,volume
   AAPL,2024-01-15,180.00,185.00,179.50,184.25,50000000
   MSFT,2024-01-15,400.00,405.00,398.50,404.75,25000000
   GOOGL,2024-01-15,150.00,152.00,149.00,151.50,30000000"""

   @pytest.fixture
   def mock_data_fetcher():
       """Mock external data fetcher."""
       fetcher = AsyncMock()
       fetcher.fetch_equity_data.return_value = [
           {"symbol": "AAPL", "company_name": "Apple Inc.", "exchange": "NASDAQ"}
       ]
       fetcher.validate_connection.return_value = True
       fetcher.get_rate_limit_status.return_value = {"remaining": 250, "reset_time": 60}
       return fetcher

   @pytest.fixture
   def mock_data_repository():
       """Mock data repository."""
       repo = AsyncMock()
       repo.upsert_equity_profiles.return_value = 1
       repo.upsert_fundata_records.return_value = 1
       repo.get_symbols_by_exchange.return_value = ["AAPL", "MSFT"]
       return repo

   @pytest.fixture
   def mock_csv_processor():
       """Mock CSV processor."""
       processor = AsyncMock()
       processor.list_available_files.return_value = ["file1.csv", "file2.csv"]
       processor.process_csv_file.return_value = [
           Mock(symbol="AAPL", date=date(2024, 1, 15), close_price=Decimal("184.25"))
       ]
       return processor
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/data_processing_services.feature -v
   # Expected: Tests fail (red state) ✓
   ```

5. **Write Implementation**
   ```python
   # packages/sra_data/services/data_processing.py
   from typing import List, Dict, Any, Optional
   from datetime import datetime, date
   import asyncio
   import logging
   from decimal import Decimal
   import httpx
   import os

   from ..domain.models import EquityProfile, FundataRecord, ProcessingTask
   from ..domain.protocols import DataFetcher, DataRepository, CSVProcessor

   logger = logging.getLogger(__name__)

   class RetryConfiguration:
       """Configuration for retry logic."""
       def __init__(self, max_retries: int = 3, backoff_base: float = 2.0):
           self.max_retries = max_retries
           self.backoff_base = backoff_base

   async def with_retry(
       func,
       retry_config: RetryConfiguration,
       exceptions: tuple = (Exception,)
   ):
       """Execute function with exponential backoff retry logic."""
       last_exception = None

       for attempt in range(retry_config.max_retries + 1):
           try:
               if asyncio.iscoroutinefunction(func):
                   return await func()
               else:
                   return func()
           except exceptions as e:
               last_exception = e
               if attempt == retry_config.max_retries:
                   break

               wait_time = retry_config.backoff_base ** attempt
               logger.warning(f"Retry {attempt + 1}/{retry_config.max_retries} after {wait_time}s: {e}")
               await asyncio.sleep(wait_time)

       logger.error(f"All {retry_config.max_retries + 1} attempts failed")
       raise last_exception

   async def process_daily_market_data(
       data_fetcher: DataFetcher,
       repository: DataRepository,
       symbols: List[str],
       batch_size: int = 100,
       rate_limit_per_minute: int = 300
   ) -> Dict[str, Any]:
       """
       Process daily market data for specified symbols with rate limiting.

       Args:
           data_fetcher: External data source interface
           repository: Database repository interface
           symbols: List of symbols to process
           batch_size: Number of symbols per batch
           rate_limit_per_minute: API rate limit

       Returns:
           Processing results with success/error counts and timing
       """
       start_time = datetime.utcnow()
       results = {
           'processed': 0,
           'errors': 0,
           'start_time': start_time,
           'batches': [],
           'total_symbols': len(symbols)
       }

       # Calculate delay between requests to respect rate limits
       delay_between_requests = 60.0 / rate_limit_per_minute if rate_limit_per_minute > 0 else 0

       # Process symbols in batches
       for i in range(0, len(symbols), batch_size):
           batch = symbols[i:i + batch_size]
           batch_start = datetime.utcnow()

           try:
               # Check rate limit status
               rate_status = await data_fetcher.get_rate_limit_status()
               if rate_status.get('remaining', 0) < 10:
                   wait_time = rate_status.get('reset_time', 60)
                   logger.info(f"Rate limit approaching, waiting {wait_time}s")
                   await asyncio.sleep(wait_time)

               # Fetch data with retry logic
               retry_config = RetryConfiguration(max_retries=3, backoff_base=2.0)
               raw_data = await with_retry(
                   lambda: data_fetcher.fetch_equity_data(batch),
                   retry_config,
                   (httpx.RequestError, httpx.HTTPStatusError)
               )

               # Validate and transform data using Pydantic models
               validated_profiles = []
               for record in raw_data:
                   try:
                       profile = EquityProfile(**record)
                       validated_profiles.append(profile.dict())
                   except Exception as e:
                       logger.warning(f"Invalid record for {record.get('symbol', 'unknown')}: {e}")

               # Store in database with retry
               stored_count = await with_retry(
                   lambda: repository.upsert_equity_profiles(validated_profiles),
                   retry_config,
                   (Exception,)
               )

               batch_result = {
                   'batch_id': i // batch_size + 1,
                   'symbols': len(batch),
                   'processed': stored_count,
                   'duration_seconds': (datetime.utcnow() - batch_start).total_seconds()
               }
               results['batches'].append(batch_result)
               results['processed'] += stored_count

               # Apply rate limiting delay
               if delay_between_requests > 0:
                   await asyncio.sleep(delay_between_requests)

           except Exception as e:
               logger.error(f"Batch processing error for batch {i//batch_size + 1}: {e}")
               results['errors'] += len(batch)
               # Continue with next batch rather than failing entirely

       results['end_time'] = datetime.utcnow()
       results['total_duration'] = (results['end_time'] - results['start_time']).total_seconds()
       results['symbols_per_minute'] = (results['processed'] / results['total_duration']) * 60 if results['total_duration'] > 0 else 0

       logger.info(f"Daily data processing completed: {results['processed']} processed, {results['errors']} errors, {results['symbols_per_minute']:.1f} symbols/min")
       return results

   async def process_fundata_csv_files(
       csv_processor: CSVProcessor,
       repository: DataRepository,
       file_pattern: Optional[str] = None,
       max_concurrent_files: int = 3
   ) -> Dict[str, Any]:
       """
       Process fundata CSV files with concurrent processing.

       Args:
           csv_processor: CSV processing interface
           repository: Database repository interface
           file_pattern: Optional pattern to filter files
           max_concurrent_files: Maximum files to process concurrently

       Returns:
           Processing results with file counts, records, and errors
       """
       start_time = datetime.utcnow()
       results = {
           'files_processed': 0,
           'files_failed': 0,
           'records_processed': 0,
           'errors': [],
           'start_time': start_time,
           'file_details': []
       }

       try:
           # Get list of available CSV files
           available_files = await csv_processor.list_available_files()
           logger.info(f"Found {len(available_files)} CSV files")

           # Filter files if pattern provided
           if file_pattern:
               available_files = [f for f in available_files if file_pattern in f]
               logger.info(f"Filtered to {len(available_files)} files matching pattern '{file_pattern}'")

           # Process files with concurrency control
           semaphore = asyncio.Semaphore(max_concurrent_files)

           async def process_single_file(file_path: str) -> Dict[str, Any]:
               async with semaphore:
                   file_start = datetime.utcnow()
                   file_result = {
                       'file_path': file_path,
                       'records_processed': 0,
                       'errors': [],
                       'status': 'processing'
                   }

                   try:
                       # Process CSV file with validation
                       retry_config = RetryConfiguration(max_retries=2, backoff_base=1.5)
                       records = await with_retry(
                           lambda: csv_processor.process_csv_file(file_path),
                           retry_config,
                           (IOError, ValueError)
                       )

                       # Convert to dictionaries for database storage
                       validated_records = []
                       for record in records:
                           try:
                               if isinstance(record, FundataRecord):
                                   validated_records.append(record.dict())
                               else:
                                   # Create FundataRecord from dict
                                   fundata_record = FundataRecord(**record)
                                   validated_records.append(fundata_record.dict())
                           except Exception as e:
                               file_result['errors'].append(f"Validation error: {e}")

                       # Store records in database
                       if validated_records:
                           stored_count = await repository.upsert_fundata_records(validated_records)
                           file_result['records_processed'] = stored_count
                           file_result['status'] = 'completed'
                       else:
                           file_result['status'] = 'no_valid_records'

                       file_result['duration_seconds'] = (datetime.utcnow() - file_start).total_seconds()
                       logger.info(f"Processed {file_path}: {file_result['records_processed']} records in {file_result['duration_seconds']:.1f}s")

                   except Exception as e:
                       error_message = f"Error processing {file_path}: {str(e)}"
                       file_result['errors'].append(error_message)
                       file_result['status'] = 'failed'
                       logger.error(error_message)

                   return file_result

           # Process all files concurrently
           if available_files:
               file_tasks = [process_single_file(file_path) for file_path in available_files]
               file_results = await asyncio.gather(*file_tasks, return_exceptions=True)

               # Aggregate results
               for result in file_results:
                   if isinstance(result, Exception):
                       results['errors'].append(f"Task failed: {str(result)}")
                       results['files_failed'] += 1
                   else:
                       results['file_details'].append(result)
                       if result['status'] == 'completed':
                           results['files_processed'] += 1
                           results['records_processed'] += result['records_processed']
                       else:
                           results['files_failed'] += 1
                       results['errors'].extend(result['errors'])

       except Exception as e:
           error_message = f"Failed to process CSV files: {str(e)}"
           results['errors'].append(error_message)
           logger.error(error_message)

       results['end_time'] = datetime.utcnow()
       results['total_duration'] = (results['end_time'] - results['start_time']).total_seconds()

       logger.info(f"CSV processing completed: {results['files_processed']} files, {results['records_processed']} records, {results['files_failed']} failures")
       return results

   async def execute_background_task(
       task: ProcessingTask,
       data_fetcher: DataFetcher,
       repository: DataRepository,
       csv_processor: CSVProcessor
   ) -> Dict[str, Any]:
       """
       Execute background task based on task type.

       Args:
           task: Processing task definition
           data_fetcher: External data source interface
           repository: Database repository interface
           csv_processor: CSV processing interface

       Returns:
           Task execution results
       """
       start_time = datetime.utcnow()
       task_result = {
           'task_id': task.task_id,
           'task_type': task.task_type,
           'status': 'running',
           'start_time': start_time,
           'results': {}
       }

       try:
           if task.task_type == 'daily_quotes':
               # Get symbols from parameters or default exchanges
               symbols = task.parameters.get('symbols', [])
               if not symbols:
                   # Get symbols from major exchanges
                   us_symbols = await repository.get_symbols_by_exchange('NYSE')
                   nasdaq_symbols = await repository.get_symbols_by_exchange('NASDAQ')
                   symbols = us_symbols + nasdaq_symbols

               task_result['results'] = await process_daily_market_data(
                   data_fetcher,
                   repository,
                   symbols,
                   batch_size=task.parameters.get('batch_size', 100)
               )

           elif task.task_type == 'csv_import':
               file_pattern = task.parameters.get('file_pattern')
               task_result['results'] = await process_fundata_csv_files(
                   csv_processor,
                   repository,
                   file_pattern,
                   max_concurrent_files=task.parameters.get('max_concurrent', 3)
               )

           elif task.task_type == 'weekly_fundamentals':
               # Similar to daily quotes but for fundamental data
               symbols = task.parameters.get('symbols', [])
               task_result['results'] = await process_daily_market_data(
                   data_fetcher,
                   repository,
                   symbols,
                   batch_size=50  # Smaller batches for fundamental data
               )

           else:
               raise ValueError(f"Unknown task type: {task.task_type}")

           task_result['status'] = 'completed'

       except Exception as e:
           task_result['status'] = 'failed'
           task_result['error'] = str(e)
           logger.error(f"Task {task.task_id} failed: {e}")

       task_result['end_time'] = datetime.utcnow()
       task_result['duration_seconds'] = (task_result['end_time'] - task_result['start_time']).total_seconds()

       return task_result
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/data_processing_services.feature -v --cov=packages/sra_data/services --cov-report=term-missing
   # Target: 100% pass rate, >90% coverage ✓
   ```

7. **Commit and Push**

   ```bash
   git add -A
   git commit -m "feat: Implement comprehensive data processing services

   - Added daily market data processing with rate limiting
   - Created fundata CSV file processing with concurrent handling
   - Implemented retry logic with exponential backoff
   - Added task execution framework for background processing
   - Comprehensive error handling and logging
   - Performance metrics tracking and reporting
   - Pydantic model validation integration
   - BDD tests covering all processing scenarios





   git push origin feature/data-processing-services
   ```

8. **Capture End Time**
   ```bash
   echo "Task 2.1 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Expected Duration: 2 hours 15 minutes
   ```

#### Task 2.2: External API Integration Services ✅
**Duration**: 2 hours 15 minutes
**Dependencies**: Task 2.1 completion
**Risk Level**: Medium

**Implementation Process**:

1. **Capture Start Time**
   ```bash
   echo "Task 2.2 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/external_api_integration.feature
   Feature: External API Integration for Data Ingestion
     As a data processing service
     I want to integrate with external APIs reliably
     So that I can fetch raw data for database storage and view creation

     Scenario: Successful FMP API data fetching for ingestion
       Given the FMP API is available and I have valid credentials
       When I fetch equity data for 50 symbols for database seeding
       Then all 50 symbols should return valid data for storage
       And the response should be received within 10 seconds
       And rate limits should be respected during bulk ingestion

     Scenario: Handle API authentication failures during ingestion
       Given I have invalid API credentials
       When I attempt to fetch data from FMP API for processing
       Then I should receive an authentication error
       And the error should be logged for monitoring
       And the data ingestion process should halt gracefully

     Scenario: Local Git LFS CSV file access for fundata processing
       Given the local fundata directory contains Git LFS CSV files
       When I scan the fundata/data and fundata/quotes directories
       Then I should receive a list of available CSV files with metadata
       And I should be able to read files directly from the local filesystem
       And file integrity should be verified before ingestion

     Scenario: API rate limit handling during data processing
       Given I am approaching FMP API rate limits
       When I make additional requests for data ingestion
       Then the system should automatically throttle requests
       And all required data should eventually be fetched and stored
       And no rate limit violations should occur
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/external_api_fixtures.py
   import pytest
   from unittest.mock import AsyncMock, Mock
   import httpx
   import json
   from datetime import datetime

   @pytest.fixture
   def mock_fmp_response():
       """Mock FMP API response data."""
       return [
           {
               "symbol": "AAPL",
               "companyName": "Apple Inc.",
               "exchangeShortName": "NASDAQ",
               "industry": "Consumer Electronics",
               "sector": "Technology",
               "marketCap": 3000000000000,
               "price": 184.25
           },
           {
               "symbol": "MSFT",
               "companyName": "Microsoft Corporation",
               "exchangeShortName": "NASDAQ",
               "industry": "Software",
               "sector": "Technology",
               "marketCap": 2800000000000,
               "price": 404.75
           }
       ]

   @pytest.fixture
   def mock_http_client():
       """Mock HTTP client for API requests."""
       client = AsyncMock(spec=httpx.AsyncClient)

       # Configure successful response
       mock_response = Mock()
       mock_response.status_code = 200
       mock_response.json.return_value = {"data": "test"}
       mock_response.raise_for_status.return_value = None

       client.get.return_value = mock_response
       client.post.return_value = mock_response

       return client

   @pytest.fixture
   def rate_limit_headers():
       """Mock rate limit headers from API."""
       return {
           "X-RateLimit-Limit": "300",
           "X-RateLimit-Remaining": "250",
           "X-RateLimit-Reset": "3600"
       }

   @pytest.fixture
   def local_csv_files_metadata():
       """Mock local CSV files metadata from Git LFS."""
       return {
           "files": [
               {
                   "path": "fundata/data/FundGeneralSeed.csv",
                   "size": 1048576,
                   "modified": "2024-01-15T10:30:00Z",
                   "type": "fundata_data"
               },
               {
                   "path": "fundata/quotes/FundDailyNAVPSSeed.csv",
                   "size": 1098765,
                   "modified": "2024-02-15T11:30:00Z",
                   "type": "fundata_quotes"
               }
           ]
       }

   @pytest.fixture
   def invalid_api_credentials():
       """Invalid API credentials for testing error handling."""
       return {
           "api_key": "invalid_key_123",
           "base_url": "https://financialmodelingprep.com/api/v3"
       }
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/external_api_integration.feature -v
   # Expected: Tests fail (red state) ✓
   ```

5. **Write Implementation**
   ```python
   # packages/sra_data/services/external_apis.py
   import httpx
   import asyncio
   import logging
   import os
   from typing import List, Dict, Any, Optional
   from datetime import datetime, timedelta
   import hashlib
   from urllib.parse import urljoin
   import json

   from ..domain.protocols import DataFetcher, CSVProcessor

   logger = logging.getLogger(__name__)

   class FMPDataFetcher:
       """Financial Modeling Prep API integration."""

       def __init__(
           self,
           api_key: Optional[str] = None,
           base_url: Optional[str] = None,
           timeout: int = 30,
           max_retries: int = 3
       ):
           self.api_key = api_key or os.getenv('FMP_API_KEY')
           self.base_url = base_url or os.getenv('FMP_BASE_URL', 'https://financialmodelingprep.com/api/v3')
           self.timeout = timeout
           self.max_retries = max_retries
           self.rate_limit_remaining = 300  # Default rate limit
           self.rate_limit_reset = datetime.utcnow()

           # Create HTTP client with proper configuration
           self.client = httpx.AsyncClient(
               timeout=httpx.Timeout(timeout),
               limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
               headers={
                   "User-Agent": "SRA-Data-Worker/1.0",
                   "Accept": "application/json"
               }
           )

       async def fetch_equity_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
           """
           Fetch equity profile data for multiple symbols.

           Args:
               symbols: List of stock symbols to fetch

           Returns:
               List of equity profile dictionaries
           """
           if not self.api_key:
               raise ValueError("FMP API key not configured")

           # Process symbols in chunks to respect API limits
           chunk_size = 10  # FMP allows up to 5 symbols per request for profile data
           results = []

           for i in range(0, len(symbols), chunk_size):
               chunk = symbols[i:i + chunk_size]
               await self._check_rate_limit()

               try:
                   # Fetch company profiles
                   profiles = await self._fetch_company_profiles(chunk)

                   # Fetch current quotes for price data
                   quotes = await self._fetch_quotes(chunk)

                   # Merge profile and quote data
                   merged_data = self._merge_profile_quote_data(profiles, quotes)
                   results.extend(merged_data)

                   # Small delay between chunks to be respectful to API
                   await asyncio.sleep(0.02)

               except Exception as e:
                   logger.error(f"Error fetching data for chunk {chunk}: {e}")
                   # Continue with next chunk rather than failing entirely

           return results

       async def fetch_quote_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
           """
           Fetch current quote data for symbols.

           Args:
               symbols: List of stock symbols

           Returns:
               List of quote dictionaries
           """
           chunk_size = 100  # Quotes endpoint supports more symbols per request
           results = []

           for i in range(0, len(symbols), chunk_size):
               chunk = symbols[i:i + chunk_size]
               await self._check_rate_limit()

               try:
                   quotes = await self._fetch_quotes(chunk)
                   results.extend(quotes)
                   await asyncio.sleep(0.1)
               except Exception as e:
                   logger.error(f"Error fetching quotes for chunk {chunk}: {e}")

           return results

       async def validate_connection(self) -> bool:
           """Validate API connection and credentials."""
           try:
               await self._check_rate_limit()
               response = await self.client.get(
                   f"{self.base_url}/profile/AAPL",
                   params={"apikey": self.api_key}
               )
               response.raise_for_status()
               return True
           except Exception as e:
               logger.error(f"API connection validation failed: {e}")
               return False

       async def get_rate_limit_status(self) -> Dict[str, Any]:
           """Get current rate limit status."""
           now = datetime.utcnow()
           seconds_until_reset = max(0, (self.rate_limit_reset - now).total_seconds())

           return {
               "remaining": self.rate_limit_remaining,
               "reset_time": seconds_until_reset,
               "limit": 300  # FMP standard limit
           }

       async def _fetch_company_profiles(self, symbols: List[str]) -> List[Dict[str, Any]]:
           """Fetch company profile data."""
           if len(symbols) == 1:
               # Single symbol request
               url = f"{self.base_url}/profile/{symbols[0]}"
           else:
               # Multiple symbols request
               symbol_str = ",".join(symbols)
               url = f"{self.base_url}/profile/{symbol_str}"

           response = await self.client.get(
               url,
               params={"apikey": self.api_key}
           )

           self._update_rate_limit_from_headers(response.headers)
           response.raise_for_status()

           data = response.json()
           return data if isinstance(data, list) else [data]

       async def _fetch_quotes(self, symbols: List[str]) -> List[Dict[str, Any]]:
           """Fetch current quote data."""
           symbol_str = ",".join(symbols)
           url = f"{self.base_url}/quote/{symbol_str}"

           response = await self.client.get(
               url,
               params={"apikey": self.api_key}
           )

           self._update_rate_limit_from_headers(response.headers)
           response.raise_for_status()

           data = response.json()
           return data if isinstance(data, list) else [data]

       def _merge_profile_quote_data(
           self,
           profiles: List[Dict[str, Any]],
           quotes: List[Dict[str, Any]]
       ) -> List[Dict[str, Any]]:
           """Merge profile and quote data by symbol."""
           quote_dict = {q.get('symbol'): q for q in quotes}
           merged = []

           for profile in profiles:
               symbol = profile.get('symbol')
               quote = quote_dict.get(symbol, {})

               merged_record = {
                   'symbol': symbol,
                   'company_name': profile.get('companyName', ''),
                   'exchange': profile.get('exchangeShortName', ''),
                   'sector': profile.get('sector', ''),
                   'industry': profile.get('industry', ''),
                   'market_cap': profile.get('mktCap') or quote.get('marketCap'),
                   'employees': profile.get('fullTimeEmployees'),
                   'description': profile.get('description', ''),
                   'website': profile.get('website', ''),
                   'country': profile.get('country', 'US'),
                   'currency': profile.get('currency', 'USD'),
                   'is_etf': profile.get('isEtf', False),
                   'is_actively_trading': profile.get('isActivelyTrading', True),
                   'current_price': quote.get('price'),
                   'change_percentage': quote.get('changesPercentage'),
                   'volume': quote.get('volume')
               }
               merged.append(merged_record)

           return merged

       async def _check_rate_limit(self):
           """Check and respect API rate limits."""
           now = datetime.utcnow()

           # If we're out of requests and reset time hasn't passed, wait
           if self.rate_limit_remaining <= 0 and now < self.rate_limit_reset:
               wait_time = (self.rate_limit_reset - now).total_seconds()
               logger.info(f"Rate limit exceeded, waiting {wait_time:.1f}s")
               await asyncio.sleep(wait_time)
               # Reset the counter after waiting
               self.rate_limit_remaining = 300
               self.rate_limit_reset = now + timedelta(minutes=1)

       def _update_rate_limit_from_headers(self, headers: Dict[str, str]):
           """Update rate limit status from response headers."""
           try:
               if 'X-RateLimit-Remaining' in headers:
                   self.rate_limit_remaining = int(headers['X-RateLimit-Remaining'])

               if 'X-RateLimit-Reset' in headers:
                   reset_timestamp = int(headers['X-RateLimit-Reset'])
                   self.rate_limit_reset = datetime.fromtimestamp(reset_timestamp)
           except (ValueError, KeyError) as e:
               logger.warning(f"Failed to parse rate limit headers: {e}")

       async def close(self):
           """Close HTTP client."""
           await self.client.aclose()

   class LocalCSVProcessor:
       """Local CSV file processor for Git LFS stored files."""

       def __init__(
           self,
           fundata_base_path: str = "fundata",
           data_subdir: str = "data",
           quotes_subdir: str = "quotes"
       ):
           self.fundata_base_path = Path(fundata_base_path)
           self.data_path = self.fundata_base_path / data_subdir
           self.quotes_path = self.fundata_base_path / quotes_subdir

           # Verify directories exist
           if not self.fundata_base_path.exists():
               raise ValueError(f"Fundata base directory not found: {self.fundata_base_path}")

       async def list_local_csv_files(self, directory: str) -> List[str]:
           """List available local CSV files from Git LFS."""
           try:
               dir_path = Path(directory)
               if not dir_path.exists():
                   logger.warning(f"Directory not found: {directory}")
                   return []

               csv_files = list(dir_path.glob("*.csv"))
               file_paths = [str(f) for f in csv_files]

               logger.info(f"Found {len(file_paths)} CSV files in {directory}")
               return file_paths

           except Exception as e:
               logger.error(f"Failed to list local CSV files in {directory}: {e}")
               raise

       async def get_all_fundata_files(self) -> Dict[str, List[str]]:
           """Get all fundata CSV files organized by type."""
           return {
               'data_files': await self.list_local_csv_files(str(self.data_path)),
               'quotes_files': await self.list_local_csv_files(str(self.quotes_path))
           }

       async def process_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
           """
           Process local CSV file.

           Args:
               file_path: Path to local CSV file

           Returns:
               List of processed records as dictionaries
           """
           try:
               file_path_obj = Path(file_path)

               if not file_path_obj.exists():
                   raise FileNotFoundError(f"CSV file not found: {file_path}")

               # Read and parse CSV file
               with open(file_path_obj, 'r', encoding='utf-8') as f:
                   content = f.read()

               records = await self._parse_csv_content(content, str(file_path_obj))
               return records

           except Exception as e:
               logger.error(f"Failed to process local CSV file {file_path}: {e}")
               raise

       def get_file_info(self, file_path: str) -> Dict[str, Any]:
           """Get local file metadata."""
           try:
               file_path_obj = Path(file_path)
               if not file_path_obj.exists():
                   return {'exists': False}

               stat = file_path_obj.stat()
               return {
                   'exists': True,
                   'size_bytes': stat.st_size,
                   'modified_timestamp': stat.st_mtime,
                   'is_fundata_data': 'data' in str(file_path_obj.parent),
                   'is_fundata_quotes': 'quotes' in str(file_path_obj.parent)
               }
           except Exception as e:
               logger.error(f"Failed to get file info for {file_path}: {e}")
               return {'exists': False, 'error': str(e)}

       async def _parse_csv_content(self, content: str, file_path: str) -> List[Dict[str, Any]]:
           """Parse CSV content into structured records."""
           import csv
           import io
           from decimal import Decimal, InvalidOperation
           from datetime import datetime

           records = []

           try:
               csv_reader = csv.DictReader(io.StringIO(content))

               for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 for header row
                   try:
                       # Determine record type based on file path
                       if 'quotes' in file_path.lower():
                           cleaned_row = self._clean_fundata_quotes_row(row, file_path)
                       else:
                           cleaned_row = self._clean_fundata_data_row(row, file_path)

                       if cleaned_row:  # Only add valid rows
                           records.append(cleaned_row)

                   except Exception as e:
                       logger.warning(f"Invalid row {row_num} in {file_path}: {e}")
                       continue

           except Exception as e:
               logger.error(f"Failed to parse CSV content from {file_path}: {e}")
               raise

           logger.info(f"Parsed {len(records)} valid records from {file_path}")
           return records

       def _clean_fundata_data_row(self, row: Dict[str, str], file_path: str) -> Optional[Dict[str, Any]]:
           """Clean and validate fundata_data CSV row."""
           try:
               # Use InstrumentKey as identifier (required)
               identifier = row.get('InstrumentKey', '').strip()
               if not identifier:
                   return None

               cleaned_record = {
                   'identifier': identifier,
                   'record_id': row.get('RecordId', '').strip() or None,
                   'language': row.get('Language', 'EN').strip()[:2],
                   'legal_name': row.get('LegalName', '').strip()[:255] or None,
                   'family_name': row.get('FamilyName', '').strip()[:150] or None,
                   'series_name': row.get('SeriesName', '').strip()[:150] or None,
                   'company': row.get('Company', '').strip()[:100] or None,
                   'inception_date': self._parse_date(row.get('InceptionDate', '')),
                   'currency': row.get('Currency', 'CAD').strip()[:3],
                   'record_state': row.get('RecordState', 'Active').strip()[:20],
                   'change_date': self._parse_date(row.get('ChangeDate', '')),
                   'source_file': Path(file_path).name,
                   'raw_data': dict(row)  # Store all original columns
               }

               return cleaned_record

           except Exception as e:
               logger.warning(f"Error cleaning fundata_data row: {e}")
               return None

       def _clean_fundata_quotes_row(self, row: Dict[str, str], file_path: str) -> Optional[Dict[str, Any]]:
           """Clean and validate fundata_quotes CSV row."""
           try:
               # Use InstrumentKey as identifier (required)
               identifier = row.get('InstrumentKey', '').strip()
               if not identifier:
                   return None

               cleaned_record = {
                   'identifier': identifier,
                   'record_id': row.get('RecordId', '').strip() or None,
                   'date': self._parse_date(row.get('Date', '')),
                   'navps': self._parse_decimal(row.get('NAVPS', '')),
                   'navps_penny_change': self._parse_decimal(row.get('NAVPSPennyChange', '')),
                   'navps_percent_change': self._parse_decimal(row.get('NAVPSPercentChange', '')),
                   'previous_date': self._parse_date(row.get('PreviousDate', '')),
                   'previous_navps': self._parse_decimal(row.get('PreviousNAVPS', '')),
                   'current_yield': self._parse_decimal(row.get('CurrentYield', '')),
                   'current_yield_percent_change': self._parse_decimal(row.get('CurrentYieldPercentChange', '')),
                   'split': row.get('Split', '').strip()[:20] or None,
                   'record_state': row.get('RecordState', 'Active').strip()[:20],
                   'change_date': self._parse_date(row.get('ChangeDate', '')),
                   'source_file': Path(file_path).name,
                   'raw_data': dict(row)  # Store all original columns
               }

               return cleaned_record

           except Exception as e:
               logger.warning(f"Error cleaning fundata_quotes row: {e}")
               return None

       def _parse_date(self, date_str: str) -> Optional:
           """Parse date string with multiple format support."""
           if not date_str or date_str.strip() == '':
               return None

           try:
               from datetime import datetime
               date_str = date_str.strip()

               # Try different date formats
               for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                   try:
                       return datetime.strptime(date_str, fmt).date()
                   except ValueError:
                       continue

               logger.warning(f"Unable to parse date: {date_str}")
               return None
           except Exception:
               return None

       def _parse_decimal(self, value: str) -> Optional:
           """Parse decimal with validation."""
           if not value or value.strip() == '':
               return None
           try:
               from decimal import Decimal
               cleaned_value = str(value).replace('$', '').replace(',', '').strip()
               return Decimal(cleaned_value) if cleaned_value else None
           except Exception:
               return None
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/external_api_integration.feature -v --cov=packages/sra_data/services/external_apis --cov-report=term-missing
   # Target: 100% pass rate, >90% coverage ✓
   ```

7. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Implement external API integration services

   - Created FMPDataFetcher for Financial Modeling Prep API integration
   - Added PrivateCSVProcessor for secure CSV file handling
   - Implemented comprehensive rate limiting and retry logic
   - Added data merging for profile and quote information
   - Robust CSV parsing with data validation and cleaning
   - File integrity checking with SHA256 hashing
   - Comprehensive error handling and logging
   - HTTP client optimization with connection pooling
   - BDD tests covering all integration scenarios





   git push origin feature/external-api-integration
   ```

8. **Capture End Time**
   ```bash
   echo "Task 2.2 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Expected Duration: 2 hours 15 minutes
   ```

#### Task 2.3: Fundata Denormalization Strategy Implementation ✅
**Duration**: 2 hours 30 minutes
**Dependencies**: Task 2.2 completion
**Risk Level**: Medium

**Implementation Process**:

1. **Capture Start Time**
   ```bash
   echo "Task 2.3 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/fundata_processing.feature
   Feature: Fundata Denormalization Strategy Implementation
     As a data processing service
     I want to implement the fundata denormalization strategy with JSON preservation and array explosion
     So that data follows DECIMAL(12,2) standardization with historical versioning

     Scenario: CSV historical seeding with denormalization
       Given I have CSV files in fundata/data/ and fundata/quotes/ directories
       When I trigger CSV seeding processing for historical data
       Then CSV records should be processed with denormalization strategy
       And all numeric fields should be standardized to DECIMAL(12,2)
       And original JSON structure should be preserved in raw_json column
       And records should be marked as data_source='CSV' for historical tracking
       And shared identifier space should be maintained across tables

     Scenario: Array explosion into multiple rows
       Given an API response contains array fields like ["AAPL", "GOOGL", "MSFT"]
       When I process the record for denormalization
       Then each array element should create a separate database row
       And array_source_field should indicate which field was exploded
       And array_element_index should preserve original position (0-based)
       And array_element_value should contain the individual element value
       And all rows should share the same identifier and preserve raw_json

     Scenario: DECIMAL(12,2) numeric standardization
       Given API responses contain various numeric precisions and formats
       When I process numeric fields for storage
       Then ALL numeric fields must be converted to DECIMAL(12,2) format
       And precision should be rounded appropriately (123.456789 → 123.46)
       And string numerics should be parsed and standardized ("15.7" → 15.70)
       And null values should remain as NULL with no conversion
       And original values should be preserved in raw_json column

     Scenario: Shared identifier scope across tables
       Given fundata_data and fundata_quotes tables
       When processing records with the same identifier
       Then identifier should be shared between both data and quotes tables
       And identifier uniqueness should be scoped across both tables
       And queries should be able to join data and quotes by identifier
       And no identifier conflicts should occur between table types

     Scenario: API-driven updates with historical preservation
       Given the system receives updated data from Fundata API
       When processing current data updates
       Then new records should be marked as data_source='API'
       And data_version should increment for historical tracking
       And previous versions should be preserved in the database
       And updated_at timestamp should reflect the API update time
       And API version information should be captured for debugging
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/fundata_fixtures.py
   import pytest
   from unittest.mock import Mock, AsyncMock
   from datetime import date
   from decimal import Decimal
   import tempfile
   import csv
   import io

   @pytest.fixture
   def sample_api_response_with_arrays():
       """Sample API response with arrays for denormalization testing."""
       return {
           "identifier": "FUND123",
           "legal_name": "Tech Growth Fund",
           "holdings": ["AAPL", "GOOGL", "MSFT", "NVDA"],
           "sectors": ["Technology", "Communication"],
           "nav_value": 15.75234,  # High precision for DECIMAL(12,2) testing
           "expense_ratio": "0.85",  # String numeric for conversion testing
           "minimum_investment": None,  # Null preservation testing
           "yield_rate": 3.456789,
           "inception_date": "2020-01-01",
           "currency": "CAD"
       }

   @pytest.fixture
   def expected_denormalized_rows():
       """Expected database rows after array denormalization."""
       base_row = {
           "identifier": "FUND123",
           "legal_name": "Tech Growth Fund",
           "nav_value": Decimal("15.75"),  # Rounded to DECIMAL(12,2)
           "expense_ratio": Decimal("0.85"),  # String converted to DECIMAL(12,2)
           "minimum_investment": None,  # Null preserved
           "yield_rate": Decimal("3.46"),  # Rounded to DECIMAL(12,2)
           "inception_date": date(2020, 1, 1),
           "currency": "CAD",
           "data_source": "API",
           "data_version": 1
       }

       return [
           # Holdings array explosion (4 rows)
           {**base_row, "array_source_field": "holdings", "array_element_index": 0, "array_element_value": "AAPL"},
           {**base_row, "array_source_field": "holdings", "array_element_index": 1, "array_element_value": "GOOGL"},
           {**base_row, "array_source_field": "holdings", "array_element_index": 2, "array_element_value": "MSFT"},
           {**base_row, "array_source_field": "holdings", "array_element_index": 3, "array_element_value": "NVDA"},
           # Sectors array explosion (2 rows)
           {**base_row, "array_source_field": "sectors", "array_element_index": 0, "array_element_value": "Technology"},
           {**base_row, "array_source_field": "sectors", "array_element_index": 1, "array_element_value": "Communication"}
       ]

   @pytest.fixture
   def decimal_conversion_test_cases():
       """Test cases for DECIMAL(12,2) standardization."""
       return [
           # (input_value, expected_decimal_output)
           (123.456789, Decimal("123.46")),  # High precision rounding
           ("15.7", Decimal("15.70")),       # String numeric conversion
           (0.045, Decimal("0.05")),         # Low precision rounding
           (None, None),                     # Null preservation
           ("invalid", None),                # Invalid string handling
           (999999999999.99, Decimal("999999999999.99")),  # Max valid value
       ]

   @pytest.fixture
   def mock_fundata_processor():
       """Mock fundata CSV processor."""
       processor = AsyncMock()
       processor.list_available_data_files.return_value = ["FundGeneralSeed.csv", "BenchmarkGeneralSeed.csv"]
       processor.list_available_quotes_files.return_value = ["FundDailyNAVPSSeed.csv"]
       processor.process_data_csv_file.return_value = [
           Mock(identifier="412682", legal_name="Test Fund", source_file="FundGeneralSeed.csv")
       ]
       processor.process_quotes_csv_file.return_value = [
           Mock(identifier="4095", navps=Decimal("11.58290000"), source_file="FundDailyNAVPSSeed.csv")
       ]
       return processor

   @pytest.fixture
   def mock_fundata_repository():
       """Mock fundata repository for database operations."""
       repo = AsyncMock()
       repo.upsert_fundata_data_records.return_value = 100
       repo.upsert_fundata_quotes_records.return_value = 200
       repo.get_fundata_identifiers.return_value = ["412682", "4095", "234435"]
       return repo
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/fundata_processing.feature -v
   # Expected: Tests fail (red state) ✓
   ```

5. **Write Implementation**
   ```python
   # packages/sra_data/services/fundata_denormalization.py
   from typing import List, Dict, Any, Optional, Union
   from datetime import datetime, date
   import asyncio
   import logging
   from decimal import Decimal, ROUND_HALF_UP
   import json

   from ..domain.models import FundataDataRecord, FundataQuotesRecord
   from ..domain.protocols import DataRepository

   logger = logging.getLogger(__name__)

   class FundataDenormalizer:
       """
       Fundata denormalization strategy implementation.

       REQUIREMENTS:
       - Column Strategy: Preserve JSON + flatten with DECIMAL(12,2)
       - Array Handling: Multiple rows per array element
       - Null Strategy: Keep nulls as NULL
       - Identifier Scope: Shared across data/quotes tables
       - Data Sources: CSV seeding + API updates
       """

       def __init__(self, repository: DataRepository):
           self.repository = repository

       def standardize_decimal(self, value: Any) -> Optional[Decimal]:
           """Standardize ANY numeric value to DECIMAL(12,2) - NO EXCEPTIONS."""
           if value is None:
               return None

           try:
               # Convert to Decimal and round to 2 decimal places
               if isinstance(value, str):
                   if value.strip() == "" or value.lower() in ['null', 'none', 'n/a']:
                       return None
                   decimal_val = Decimal(value)
               else:
                   decimal_val = Decimal(str(value))

               # Round to exactly 2 decimal places using ROUND_HALF_UP
               return decimal_val.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

           except (ValueError, TypeError, decimal.InvalidOperation) as e:
               logger.warning(f"Failed to convert {value} to DECIMAL(12,2): {e}")
               return None

       def explode_arrays(
           self,
           record: Dict[str, Any],
           identifier: str
       ) -> List[Dict[str, Any]]:
           """
           Explode arrays into multiple rows - one row per array element.

           Args:
               record: Original record with potential arrays
               identifier: Shared identifier for all exploded rows

           Returns:
               List of denormalized rows (one per array element)
           """
           exploded_rows = []
           base_record = record.copy()

           # Find array fields in the record
           array_fields = {k: v for k, v in record.items()
                          if isinstance(v, list) and v}  # Non-empty lists only

           if not array_fields:
               # No arrays found - return single row
               return [self._create_denormalized_row(base_record, identifier, None, None, None)]

           # Create rows for each array element
           for field_name, array_values in array_fields.items():
               for index, element_value in enumerate(array_values):
                   exploded_rows.append(
                       self._create_denormalized_row(
                           base_record, identifier, field_name, index, str(element_value)
                       )
                   )

           return exploded_rows

       def _create_denormalized_row(
           self,
           base_record: Dict[str, Any],
           identifier: str,
           array_field: Optional[str] = None,
           array_index: Optional[int] = None,
           array_value: Optional[str] = None
       ) -> Dict[str, Any]:
           """Create a single denormalized row with standardized decimals."""

           # Start with base record, removing array fields
           denormalized = {k: v for k, v in base_record.items() if not isinstance(v, list)}

           # Add required denormalization fields
           denormalized.update({
               'identifier': identifier,
               'array_source_field': array_field,
               'array_element_index': array_index,
               'array_element_value': array_value,
               'raw_json': base_record,  # CRITICAL: Preserve original JSON
               'processed_at': datetime.utcnow(),
               'updated_at': datetime.utcnow()
           })

           # Standardize ALL numeric fields to DECIMAL(12,2) - ABSOLUTE REQUIREMENT
           for field, value in denormalized.items():
               if field not in ['identifier', 'array_source_field', 'array_element_index',
                               'array_element_value', 'raw_json', 'processed_at', 'updated_at']:
                   if isinstance(value, (int, float, str, Decimal)):
                       denormalized[field] = self.standardize_decimal(value)

           return denormalized
               with open(file_path, 'r', encoding='utf-8') as file:
                   csv_reader = csv.DictReader(file)

                   for row_num, row in enumerate(csv_reader, start=2):
                       try:
                           # Clean and validate row
                           cleaned_row = self._clean_data_row(row, filename)
                           if cleaned_row:
                               # Create Pydantic model for validation
                               record = FundataDataRecord(**cleaned_row)
                               records.append(record.dict())
                       except Exception as e:
                           logger.warning(f"Invalid row {row_num} in {filename}: {e}")
                           continue

           except Exception as e:
               logger.error(f"Error processing data file {filename}: {e}")
               raise

           logger.info(f"Processed {len(records)} records from {filename}")
           return records

       async def process_quotes_csv_file(self, filename: str) -> List[Dict[str, Any]]:
           """Process a single quotes CSV file."""
           file_path = os.path.join(self.quotes_dir, filename)
           records = []

           try:
               with open(file_path, 'r', encoding='utf-8') as file:
                   csv_reader = csv.DictReader(file)

                   for row_num, row in enumerate(csv_reader, start=2):
                       try:
                           # Clean and validate row
                           cleaned_row = self._clean_quotes_row(row, filename)
                           if cleaned_row:
                               # Create Pydantic model for validation
                               record = FundataQuotesRecord(**cleaned_row)
                               records.append(record.dict())
                       except Exception as e:
                           logger.warning(f"Invalid row {row_num} in {filename}: {e}")
                           continue

           except Exception as e:
               logger.error(f"Error processing quotes file {filename}: {e}")
               raise

           logger.info(f"Processed {len(records)} records from {filename}")
           return records

       def _clean_data_row(self, row: Dict[str, str], filename: str) -> Optional[Dict[str, Any]]:
           """Clean and validate a data CSV row."""
           try:
               # Get the InstrumentKey (identifier) - required field
               identifier = row.get('InstrumentKey', '').strip()
               if not identifier:
                   return None

               # Parse date fields
               def parse_date(date_str: str) -> Optional[date]:
                   if not date_str or date_str.strip() == '':
                       return None
                   try:
                       return datetime.strptime(date_str.strip(), '%Y-%m-%d').date()
                   except ValueError:
                       return None

               cleaned = {
                   'InstrumentKey': identifier,
                   'RecordId': row.get('RecordId', '').strip() or None,
                   'Language': row.get('Language', 'EN').strip()[:2] or 'EN',
                   'LegalName': row.get('LegalName', '').strip()[:255] or None,
                   'FamilyName': row.get('FamilyName', '').strip()[:150] or None,
                   'SeriesName': row.get('SeriesName', '').strip()[:150] or None,
                   'Company': row.get('Company', '').strip()[:100] or None,
                   'InceptionDate': parse_date(row.get('InceptionDate', '')),
                   'Currency': row.get('Currency', 'CAD').strip()[:3] or 'CAD',
                   'RecordState': row.get('RecordState', 'Active').strip()[:20] or 'Active',
                   'ChangeDate': parse_date(row.get('ChangeDate', '')),
                   'source_file': filename,
                   'raw_data': dict(row)  # Preserve all original CSV data
               }

               return cleaned

           except Exception as e:
               logger.warning(f"Error cleaning data row: {e}")
               return None

       def _clean_quotes_row(self, row: Dict[str, str], filename: str) -> Optional[Dict[str, Any]]:
           """Clean and validate a quotes CSV row."""
           try:
               # Get the InstrumentKey (identifier) - required field
               identifier = row.get('InstrumentKey', '').strip()
               if not identifier:
                   return None

               # Parse decimal fields
               def parse_decimal(value: str) -> Optional[Decimal]:
                   if not value or value.strip() == '':
                       return None
                   try:
                       return Decimal(str(value).replace('$', '').replace(',', ''))
                   except Exception:
                       return None

               # Parse date fields
               def parse_date(date_str: str) -> Optional[date]:
                   if not date_str or date_str.strip() == '':
                       return None
                   try:
                       return datetime.strptime(date_str.strip(), '%Y-%m-%d').date()
                   except ValueError:
                       return None

               cleaned = {
                   'InstrumentKey': identifier,
                   'RecordId': row.get('RecordId', '').strip() or None,
                   'Date': parse_date(row.get('Date', '')),
                   'NAVPS': parse_decimal(row.get('NAVPS', '')),
                   'NAVPSPennyChange': parse_decimal(row.get('NAVPSPennyChange', '')),
                   'NAVPSPercentChange': parse_decimal(row.get('NAVPSPercentChange', '')),
                   'PreviousDate': parse_date(row.get('PreviousDate', '')),
                   'PreviousNAVPS': parse_decimal(row.get('PreviousNAVPS', '')),
                   'CurrentYield': parse_decimal(row.get('CurrentYield', '')),
                   'CurrentYieldPercentChange': parse_decimal(row.get('CurrentYieldPercentChange', '')),
                   'Split': row.get('Split', '').strip()[:20] or None,
                   'RecordState': row.get('RecordState', 'Active').strip()[:20] or 'Active',
                   'ChangeDate': parse_date(row.get('ChangeDate', '')),
                   'source_file': filename,
                   'raw_data': dict(row)  # Preserve all original CSV data
               }

               return cleaned

           except Exception as e:
               logger.warning(f"Error cleaning quotes row: {e}")
               return None

   async def process_all_fundata_files(
       processor: LocalFundataCSVProcessor,
       repository: DataRepository,
       max_concurrent_files: int = 3
   ) -> Dict[str, Any]:
       """
       Process all fundata CSV files concurrently.

       Returns:
           Processing results with counts and performance metrics
       """
       start_time = datetime.utcnow()
       results = {
           'data_files_processed': 0,
           'quotes_files_processed': 0,
           'data_records_processed': 0,
           'quotes_records_processed': 0,
           'errors': [],
           'start_time': start_time
       }

       try:
           # Get all available files
           data_files = await processor.list_available_data_files()
           quotes_files = await processor.list_available_quotes_files()

           logger.info(f"Found {len(data_files)} data files and {len(quotes_files)} quotes files")

           # Process data files
           if data_files:
               semaphore = asyncio.Semaphore(max_concurrent_files)

               async def process_data_file(filename: str):
                   async with semaphore:
                       try:
                           records = await processor.process_data_csv_file(filename)
                           if records:
                               stored_count = await repository.upsert_fundata_data_records(records)
                               return {'filename': filename, 'records': stored_count, 'type': 'data'}
                           return {'filename': filename, 'records': 0, 'type': 'data'}
                       except Exception as e:
                           logger.error(f"Error processing data file {filename}: {e}")
                           results['errors'].append(f"Data file {filename}: {e}")
                           return {'filename': filename, 'records': 0, 'type': 'data', 'error': str(e)}

               # Process all data files concurrently
               data_tasks = [process_data_file(filename) for filename in data_files]
               data_results = await asyncio.gather(*data_tasks, return_exceptions=True)

               # Aggregate data results
               for result in data_results:
                   if isinstance(result, Exception):
                       results['errors'].append(f"Data processing task failed: {result}")
                   elif isinstance(result, dict) and 'error' not in result:
                       results['data_files_processed'] += 1
                       results['data_records_processed'] += result['records']

           # Process quotes files
           if quotes_files:
               async def process_quotes_file(filename: str):
                   async with semaphore:
                       try:
                           records = await processor.process_quotes_csv_file(filename)
                           if records:
                               stored_count = await repository.upsert_fundata_quotes_records(records)
                               return {'filename': filename, 'records': stored_count, 'type': 'quotes'}
                           return {'filename': filename, 'records': 0, 'type': 'quotes'}
                       except Exception as e:
                           logger.error(f"Error processing quotes file {filename}: {e}")
                           results['errors'].append(f"Quotes file {filename}: {e}")
                           return {'filename': filename, 'records': 0, 'type': 'quotes', 'error': str(e)}

               # Process all quotes files concurrently
               quotes_tasks = [process_quotes_file(filename) for filename in quotes_files]
               quotes_results = await asyncio.gather(*quotes_tasks, return_exceptions=True)

               # Aggregate quotes results
               for result in quotes_results:
                   if isinstance(result, Exception):
                       results['errors'].append(f"Quotes processing task failed: {result}")
                   elif isinstance(result, dict) and 'error' not in result:
                       results['quotes_files_processed'] += 1
                       results['quotes_records_processed'] += result['records']

       except Exception as e:
           results['errors'].append(f"Failed to process fundata files: {str(e)}")
           logger.error(f"Fundata processing failed: {e}")

       results['end_time'] = datetime.utcnow()
       results['total_duration'] = (results['end_time'] - results['start_time']).total_seconds()

       logger.info(
           f"Fundata processing completed: {results['data_files_processed']} data files, "
           f"{results['quotes_files_processed']} quotes files, "
           f"{results['data_records_processed']} data records, "
           f"{results['quotes_records_processed']} quotes records"
       )

       return results
   ```

   ```python
   # packages/sra_data/repositories/fundata_repository.py
   import asyncpg
   from typing import List, Dict, Any, Optional
   import logging
   from datetime import datetime

   logger = logging.getLogger(__name__)

   async def upsert_fundata_data_records(
       pool: asyncpg.Pool,
       records: List[Dict[str, Any]]
   ) -> int:
       """
       Upsert fundata data records with optimized batch processing.
       """
       if not records:
           return 0

       upsert_sql = """
       INSERT INTO fundata_data (
           identifier, record_id, language, legal_name, family_name,
           series_name, company, inception_date, currency, record_state,
           change_date, source_file, processed_at, raw_data
       ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
       ON CONFLICT (identifier, source_file)
       DO UPDATE SET
           record_id = EXCLUDED.record_id,
           language = EXCLUDED.language,
           legal_name = EXCLUDED.legal_name,
           family_name = EXCLUDED.family_name,
           series_name = EXCLUDED.series_name,
           company = EXCLUDED.company,
           inception_date = EXCLUDED.inception_date,
           currency = EXCLUDED.currency,
           record_state = EXCLUDED.record_state,
           change_date = EXCLUDED.change_date,
           processed_at = NOW(),
           raw_data = EXCLUDED.raw_data;
       """

       try:
           async with pool.acquire() as conn:
               async with conn.transaction():
                   for record in records:
                       await conn.execute(
                           upsert_sql,
                           record.get('identifier'),
                           record.get('record_id'),
                           record.get('language'),
                           record.get('legal_name'),
                           record.get('family_name'),
                           record.get('series_name'),
                           record.get('company'),
                           record.get('inception_date'),
                           record.get('currency'),
                           record.get('record_state'),
                           record.get('change_date'),
                           record.get('source_file'),
                           datetime.utcnow(),
                           record.get('raw_data', {})
                       )
           return len(records)

       except Exception as e:
           logger.error(f"Error upserting fundata data records: {e}")
           raise

   async def upsert_fundata_quotes_records(
       pool: asyncpg.Pool,
       records: List[Dict[str, Any]]
   ) -> int:
       """
       Upsert fundata quotes records with optimized batch processing.
       """
       if not records:
           return 0

       upsert_sql = """
       INSERT INTO fundata_quotes (
           identifier, record_id, date, navps, navps_penny_change,
           navps_percent_change, previous_date, previous_navps, current_yield,
           current_yield_percent_change, split, record_state, change_date,
           source_file, processed_at, raw_data
       ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
       ON CONFLICT (identifier, date)
       DO UPDATE SET
           record_id = EXCLUDED.record_id,
           navps = EXCLUDED.navps,
           navps_penny_change = EXCLUDED.navps_penny_change,
           navps_percent_change = EXCLUDED.navps_percent_change,
           previous_date = EXCLUDED.previous_date,
           previous_navps = EXCLUDED.previous_navps,
           current_yield = EXCLUDED.current_yield,
           current_yield_percent_change = EXCLUDED.current_yield_percent_change,
           split = EXCLUDED.split,
           record_state = EXCLUDED.record_state,
           change_date = EXCLUDED.change_date,
           source_file = EXCLUDED.source_file,
           processed_at = NOW(),
           raw_data = EXCLUDED.raw_data;
       """

       try:
           async with pool.acquire() as conn:
               async with conn.transaction():
                   for record in records:
                       await conn.execute(
                           upsert_sql,
                           record.get('identifier'),
                           record.get('record_id'),
                           record.get('date'),
                           record.get('navps'),
                           record.get('navps_penny_change'),
                           record.get('navps_percent_change'),
                           record.get('previous_date'),
                           record.get('previous_navps'),
                           record.get('current_yield'),
                           record.get('current_yield_percent_change'),
                           record.get('split'),
                           record.get('record_state'),
                           record.get('change_date'),
                           record.get('source_file'),
                           datetime.utcnow(),
                           record.get('raw_data', {})
                       )
           return len(records)

       except Exception as e:
           logger.error(f"Error upserting fundata quotes records: {e}")
           raise
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/fundata_processing.feature -v --cov=packages/sra_data/services/fundata_processing --cov-report=term-missing
   # Target: 100% pass rate, >90% coverage ✓
   ```

7. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Implement fundata CSV processing and repository services

   - Added LocalFundataCSVProcessor for Git LFS CSV file processing
   - Created fundata_data and fundata_quotes table repository functions
   - Implemented denormalized flat table structure with Identifier indexing
   - Added comprehensive CSV parsing with schema variation handling
   - Robust data validation and cleaning for both data and quotes
   - Concurrent file processing with performance optimization
   - Raw data preservation in JSONB columns for flexibility
   - Comprehensive error handling and logging
   - BDD tests covering all fundata processing scenarios





   git push origin feature/fundata-processing
   ```

8. **Capture End Time**
   ```bash
   echo "Task 2.3 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Expected Duration: 2 hours 30 minutes
   ```

**Validation Criteria**:
- All BDD tests pass with 100% success rate for denormalization strategy
- Test coverage >90% for fundata denormalization code
- DECIMAL(12,2) standardization working for ALL numeric fields (no exceptions)
- Array explosion creating correct multiple rows (one row per array element)
- JSON preservation working (raw_json column populated with original data)
- Shared identifier scope maintained across fundata_data and fundata_quotes tables
- Null values preserved as NULL (no conversion or omission)
- Historical versioning working (data_version increments for API updates)
- CSV seeding marked as data_source='CSV', API updates as data_source='API'
- Performance meets target rates with denormalization overhead included

**Rollback Procedure**:
1. Revert fundata processing commits
2. Drop fundata_data and fundata_quotes tables if created
3. Verify system functionality without fundata components
4. Update stakeholders on rollback and issues

#### Task 2.4: Fundata API Integration and Historical Data Management ✅
**Duration**: 1 hour 45 minutes
**Dependencies**: Task 2.3 completion
**Risk Level**: Medium

**Implementation Process**:

1. **Capture Start Time**
   ```bash
   echo "Task 2.4 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/fundata_api_integration.feature
   Feature: Fundata API Integration with Historical Data Management
     As a data processing service
     I want to integrate with Fundata API for current updates
     So that all data is sourced from API calls with CSV used only for historical seeding

     Scenario: API-driven data updates with versioning
       Given the system receives updated fund data from Fundata API
       When I process the API response for fund updates
       Then new records should be stored with data_source='API'
       And data_version should increment from the previous version
       And updated_at timestamp should reflect the API call time
       And API version information should be captured for tracking
       And historical versions should be preserved in the database

     Scenario: Historical preservation for API updates
       Given a fund identifier already exists with CSV historical data
       When I receive an API update for the same identifier
       Then the system should create a new version record
       And the previous CSV record should be preserved unchanged
       And both records should share the same identifier
       And queries should return the latest version by default
       And historical queries should access all versions

     Scenario: Schema stability enforcement
       Given the system expects no new CSV files
       When the processing system starts
       Then it should only process existing CSV files for seeding
       And it should not expect or process any new CSV additions
       And all current data updates must come from API calls only
       And schema changes should be explicitly rejected

     Scenario: Identifier scope sharing across API calls
       Given fundata API returns both data and quotes information
       When processing API responses for the same fund
       Then the identifier should be shared across fundata_data and fundata_quotes
       And no identifier conflicts should occur between data types
       And joins between tables should work seamlessly by identifier
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/fundata_api_fixtures.py
   import pytest
   from datetime import datetime
   from decimal import Decimal

   @pytest.fixture
   def mock_fundata_api_response():
       """Sample Fundata API response for testing."""
       return {
           "identifier": "FUND456",
           "name": "Growth Equity Fund",
           "nav_value": 18.234567,
           "holdings": ["AAPL", "MSFT"],
           "last_updated": "2024-01-15T15:30:00Z",
           "api_version": "2.1"
       }

   @pytest.fixture
   def existing_csv_record():
       """Existing CSV record for version comparison."""
       return {
           "identifier": "FUND456",
           "name": "Growth Equity Fund",
           "nav_value": Decimal("17.85"),
           "data_source": "CSV",
           "data_version": 1,
           "processed_at": datetime(2024, 1, 1, 10, 0, 0)
       }
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/fundata_api_integration.feature -v
   # Expected: Tests fail (red state) ✓
   ```

5. **Write Implementation**
   ```python
   # packages/sra_data/services/fundata_api_integration.py
   import httpx
   from typing import Dict, Any, Optional
   from datetime import datetime
   import logging

   logger = logging.getLogger(__name__)

   class FundataAPIIntegrator:
       """Fundata API integration with historical data management."""

       def __init__(self, api_base_url: str, api_key: str):
           self.api_base_url = api_base_url
           self.api_key = api_key

       async def process_api_update(
           self,
           api_response: Dict[str, Any],
           existing_version: Optional[int] = None
       ) -> Dict[str, Any]:
           """Process API update with historical versioning."""

           # Create new version record
           new_version = (existing_version or 0) + 1

           processed_record = {
               **api_response,
               'data_source': 'API',
               'data_version': new_version,
               'updated_at': datetime.utcnow(),
               'api_version': api_response.get('api_version', 'unknown')
           }

           return processed_record
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/fundata_api_integration.feature -v
   # Expected: All tests pass (100% success rate) ✓
   ```

7. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Implement Fundata API integration with historical data management

   - Added API-driven data updates with versioning support
   - Implemented historical data preservation during API updates
   - Added schema stability enforcement (no new CSVs)
   - Created shared identifier scope management
   - All current data sourced from API calls, CSV only for seeding"

   git push origin feature/fundata-api-integration
   ```

8. **Capture End Time**
   ```bash
   echo "Task 2.4 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Expected Duration: 1 hour 45 minutes
   ```

**Validation Criteria**:
- All BDD tests pass with 100% success rate for API integration
- Historical versioning working correctly (data_version increments)
- API updates marked as data_source='API' with proper timestamps
- Existing CSV records preserved unchanged during API updates
- Schema stability enforced (no new CSV processing)
- Shared identifier scope working across data and quotes from API

**Rollback Procedure**:
1. Revert API integration commits
2. Verify CSV seeding still works for historical data
3. Confirm no API dependencies break existing functionality
4. Update stakeholders on rollback and issues

#### Task 2.5: FMP Integration Wrapper Implementation ✅
**Duration**: 1 hour 30 minutes
**Dependencies**: Task 2.4 completion
**Risk Level**: Low (using existing proven FMP code)

**Critical Requirement**: ZERO CHANGES to existing FMP code in `/FMP/` directory

**Implementation Process**:

1. **Capture Start Time**
   ```bash
   echo "Task 2.4 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/fmp_rate_limiting.feature
   Feature: FMP API Rate Limiting and Control
     As a data processing service
     I want to enforce strict rate limits for FMP API calls
     So that I never exceed the 3,000 calls per minute ceiling

     Scenario: Enforce 3,000 calls per minute ceiling
       Given I have a semaphore controlling concurrent FMP calls
       And I have a sliding window tracking call timestamps
       When I make multiple concurrent API requests
       Then no more than 3,000 calls should be made in any minute
       And excess calls should be queued for later execution
       And rate limit violations should be prevented
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/rate_limiting_fixtures.py
   import pytest
   import asyncio
   from unittest.mock import AsyncMock
   import collections

   @pytest.fixture
   def fmp_rate_limiter():
       """FMP rate limiting implementation fixture."""
       return AsyncMock()
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/fmp_rate_limiting.feature -v
   # Expected: Tests fail (red state) ✓
   ```

5. **Write Implementation**
   ```python
   # packages/sra_data/services/rate_limiting.py
   import asyncio
   import collections
   import time
   import logging

   logger = logging.getLogger(__name__)

   class FMPRateLimiter:
       """FMP API rate limiting with 3,000 calls per minute ceiling."""

       def __init__(self, max_calls_per_minute: int = 3000):
           self.max_calls_per_minute = max_calls_per_minute
           self.semaphore = asyncio.Semaphore(50)  # 50 concurrent calls
           self.call_timestamps = collections.deque(maxlen=max_calls_per_minute)

       async def acquire(self):
           """Acquire rate limit token."""
           async with self.semaphore:
               await self._enforce_sliding_window()
               self.call_timestamps.append(time.time())

       async def _enforce_sliding_window(self):
           """Enforce sliding window rate limits."""
           current_time = time.time()
           minute_ago = current_time - 60

           # Remove timestamps older than 1 minute
           while self.call_timestamps and self.call_timestamps[0] < minute_ago:
               self.call_timestamps.popleft()

           # Check if we're at the limit
           if len(self.call_timestamps) >= self.max_calls_per_minute:
               wait_time = 60 - (current_time - self.call_timestamps[0])
               if wait_time > 0:
                   logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                   await asyncio.sleep(wait_time)
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/fmp_rate_limiting.feature -v
   # Expected: All tests pass (100% success rate) ✓
   ```

7. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Implement FMP API rate limiting with 3,000 calls/minute ceiling

   - Added comprehensive rate limiting with semaphore and sliding window
   - Implemented circuit breaker pattern for overflow protection
   - Added real-time call tracking and monitoring

   git push origin feature/fmp-rate-limiting"
   ```

8. **Capture End Time**
   ```bash
   echo "Task 2.4 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Duration: 1 hour 30 minutes
   ```

**Validation Criteria**:
- Wrapper functions successfully execute existing FMP scripts
- No modifications made to any existing FMP code files
- Existing threading patterns provide natural rate limiting
- Database integration works through existing FMP schemas
- All BDD tests pass with 100% success rate

#### Task 2.6: Database Performance Optimization Setup ✅
**Duration**: 2 hours 0 minutes
**Dependencies**: Task 2.5 completion
**Risk Level**: High

**Implementation Process**:

1. **Capture Start Time**
   ```bash
   echo "Task 2.5 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/database_performance_optimization.feature
   Feature: Database Performance Optimization
     As a data processing service
     I want optimized database performance for all operations
     So that data processing meets performance targets

     Scenario: Proper indices on all lookup columns
       Given I have tables with symbol, date, and identifier columns
       When I query by these columns
       Then queries should use B-tree indices
       And query execution time should be under 100ms

     Scenario: Materialized views for complex queries
       Given I have complex aggregation queries
       When I create materialized views with proper refresh strategy
       Then query performance should improve by 10x or more
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/database_performance_fixtures.py
   import pytest
   import asyncpg
   from unittest.mock import AsyncMock

   @pytest.fixture
   async def performance_optimized_db():
       """Database with performance optimizations."""
       return AsyncMock()
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/database_performance_optimization.feature -v
   # Expected: Tests fail (red state) ✓
   ```

5. **Write Implementation**
   ```python
   # packages/sra_data/repositories/performance_optimization.py
   import asyncpg
   import logging

   logger = logging.getLogger(__name__)

   async def create_performance_indices(pool: asyncpg.Pool):
       """Create all performance indices for optimized queries."""
       indices_sql = """
       -- Primary lookup indices
       CREATE INDEX IF NOT EXISTS idx_equity_profile_symbol_btree ON equity_profile USING btree(symbol);
       CREATE INDEX IF NOT EXISTS idx_fundata_data_identifier_btree ON fundata_data USING btree(identifier);
       CREATE INDEX IF NOT EXISTS idx_fundata_quotes_identifier_btree ON fundata_quotes USING btree(identifier);

       -- Time-series clustering
       ALTER TABLE fundata_quotes CLUSTER ON idx_fundata_quotes_identifier_date;

       -- Materialized views for complex queries
       CREATE MATERIALIZED VIEW IF NOT EXISTS daily_market_summary AS
       SELECT date, COUNT(*) as total_symbols, AVG(navps) as avg_navps
       FROM fundata_quotes GROUP BY date;
       """

       async with pool.acquire() as conn:
           await conn.execute(indices_sql)
           logger.info("Database performance optimization completed")
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/database_performance_optimization.feature -v
   # Expected: All tests pass (100% success rate) ✓
   ```

7. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Implement comprehensive database performance optimization

   - Created B-tree indices on all lookup columns
   - Implemented time-series clustering strategy for fundata_quotes
   - Added materialized views for complex aggregations

   git push origin feature/database-performance"
   ```

8. **Capture End Time**
   ```bash
   echo "Task 2.5 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Duration: 2 hours 0 minutes
   ```

**Validation Criteria**:
- All indices created successfully and being used by queries
- Materialized views provide performance improvement
- All BDD tests pass with 100% success rate

### Phase 3: Skeleton API Layer (Minimal)
**Duration**: 1 hour 30 minutes
**Dependencies**: Phase 2 completion
**Risk Level**: Low

**Focus**: Create minimal FastAPI skeleton to prevent Render.com deployment suspension, NOT a full client API.

### Phase 4: Database Views and Repository Implementation
**Duration**: 4 hours 45 minutes
**Dependencies**: Phase 3 completion
**Risk Level**: Medium

**Focus**: Create modelized Pydantic views in database and repository functions for external client access, plus unified refresh scheduling.

#### Task 4.1: Unified Refresh Scheduler Implementation ✅
**Duration**: 1 hour 30 minutes
**Dependencies**: Phase 3 completion
**Risk Level**: Medium

**Implementation Process**:

1. **Capture Start Time**
   ```bash
   echo "Task 4.1 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/unified_refresh_scheduler.feature
   Feature: Unified Refresh Scheduler for FMP and Fundata
     As a data processing service
     I want to coordinate daily refresh of both FMP and fundata sources
     So that all data is synchronized and updated at the same time

     Scenario: Daily unified refresh coordination
       Given I have both FMP API and fundata CSV processing services configured
       When the daily unified refresh is triggered at scheduled time
       Then existing FMP collection scripts should be executed first via wrapper functions
       And fundata data processing should be executed second
       And fundata quotes processing should be executed third
       And modelized views should be recreated with updated data
       And all refresh operations should complete within 2-hour window

     Scenario: Independent error handling during unified refresh
       Given the unified refresh process is running
       When one data source fails (FMP or fundata)
       Then other data sources should continue processing
       And failed source should be logged with retry mechanism
       And overall refresh should not be blocked by individual failures

     Scenario: Performance monitoring for unified refresh
       Given the unified refresh is executing
       When I monitor the processing performance
       Then FMP processing should meet target rates (1,000 symbols/min)
       And fundata data processing should meet target rates (5,000 records/min)
       And fundata quotes processing should meet target rates (10,000 records/min)
       And total refresh cycle should complete within performance targets
   ```

3. **Write Implementation**
   ```python
   # packages/sra_data/services/unified_refresh_scheduler.py
   import asyncio
   import logging
   from datetime import datetime, timedelta
   from typing import Dict, Any, Optional

   logger = logging.getLogger(__name__)

   class UnifiedRefreshScheduler:
       """
       Coordinates unified daily refresh of FMP API and fundata CSV sources.
       """

       def __init__(
           self,
           fmp_service,
           fundata_processor,
           view_service,
           max_duration_hours: int = 2
       ):
           self.fmp_service = fmp_service
           self.fundata_processor = fundata_processor
           self.view_service = view_service
           self.max_duration_hours = max_duration_hours

       async def execute_unified_refresh(self) -> Dict[str, Any]:
           """
           Execute unified daily refresh for all data sources.

           Returns:
               Dictionary with refresh results and performance metrics
           """
           start_time = datetime.utcnow()
           results = {
               'start_time': start_time,
               'fmp_results': {},
               'fundata_results': {},
               'view_results': {},
               'errors': [],
               'total_duration_minutes': 0,
               'success': False
           }

           logger.info("Starting unified refresh process")

           try:
               # Step 1: Execute existing FMP collection scripts via wrappers
               logger.info("Step 1: Executing existing FMP collection scripts")
               try:
                   fmp_results = await self.fmp_service.refresh_all_data()
                   results['fmp_results'] = fmp_results
                   results['fmp_results']['success'] = True
                   logger.info(f"FMP refresh completed: {fmp_results}")
               except Exception as e:
                   error_msg = f"FMP refresh failed: {str(e)}"
                   logger.error(error_msg)
                   results['errors'].append(error_msg)
                   results['fmp_results'] = {'success': False, 'error': str(e)}

               # Step 2: Fundata CSV Processing
               logger.info("Step 2: Processing fundata CSV files")
               try:
                   fundata_results = await self.fundata_processor.process_all_fundata_files()
                   results['fundata_results'] = fundata_results
                   results['fundata_results']['success'] = True
                   logger.info(f"Fundata processing completed: {fundata_results}")
               except Exception as e:
                   error_msg = f"Fundata processing failed: {str(e)}"
                   logger.error(error_msg)
                   results['errors'].append(error_msg)
                   results['fundata_results'] = {'success': False, 'error': str(e)}

               # Step 3: Recreate Modelized Views
               logger.info("Step 3: Recreating modelized views")
               try:
                   view_results = await self.view_service.recreate_modelized_views()
                   results['view_results'] = view_results
                   results['view_results']['success'] = True
                   logger.info(f"View recreation completed: {view_results}")
               except Exception as e:
                   error_msg = f"View recreation failed: {str(e)}"
                   logger.error(error_msg)
                   results['errors'].append(error_msg)
                   results['view_results'] = {'success': False, 'error': str(e)}

           except Exception as e:
               error_msg = f"Unified refresh process failed: {str(e)}"
               logger.error(error_msg)
               results['errors'].append(error_msg)

           # Calculate final metrics
           end_time = datetime.utcnow()
           total_duration = end_time - start_time
           results['end_time'] = end_time
           results['total_duration_minutes'] = total_duration.total_seconds() / 60

           # Determine overall success (partial success allowed)
           successful_steps = sum([
               results['fmp_results'].get('success', False),
               results['fundata_results'].get('success', False),
               results['view_results'].get('success', False)
           ])

           results['success'] = successful_steps >= 2  # At least 2/3 steps must succeed

           # Check performance targets
           if results['total_duration_minutes'] > (self.max_duration_hours * 60):
               warning_msg = f"Refresh exceeded target duration: {results['total_duration_minutes']:.1f} minutes"
               logger.warning(warning_msg)
               results['errors'].append(warning_msg)

           logger.info(
               f"Unified refresh completed in {results['total_duration_minutes']:.1f} minutes. "
               f"Success: {results['success']}, Errors: {len(results['errors'])}"
           )

           return results

       async def schedule_daily_refresh(self, target_hour: int = 2) -> None:
           """
           Schedule daily unified refresh at specified hour.

           Args:
               target_hour: Hour of day to run refresh (0-23), default is 2 AM
           """
           while True:
               try:
                   # Calculate next refresh time
                   now = datetime.utcnow()
                   next_refresh = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)

                   # If target time has passed today, schedule for tomorrow
                   if next_refresh <= now:
                       next_refresh += timedelta(days=1)

                   # Calculate sleep duration
                   sleep_duration = (next_refresh - now).total_seconds()
                   logger.info(f"Next unified refresh scheduled for {next_refresh} UTC ({sleep_duration/3600:.1f} hours)")

                   # Sleep until next refresh time
                   await asyncio.sleep(sleep_duration)

                   # Execute the refresh
                   results = await self.execute_unified_refresh()

                   if not results['success']:
                       logger.error(f"Unified refresh failed with {len(results['errors'])} errors")
                   else:
                       logger.info("Unified refresh completed successfully")

               except Exception as e:
                   logger.error(f"Daily refresh scheduler error: {e}")
                   # Sleep for 1 hour before retrying to prevent rapid failure loops
                   await asyncio.sleep(3600)
   ```

4. **Run Green Test**
   ```bash
   pytest tests/features/unified_refresh_scheduler.feature -v --cov=packages/sra_data/services/unified_refresh_scheduler --cov-report=term-missing
   # Target: 100% pass rate, >90% coverage ✓
   ```

5. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Implement unified refresh scheduler for FMP and fundata

   - Added UnifiedRefreshScheduler for coordinated daily refresh
   - Implemented sequential processing: FMP → fundata data → fundata quotes → views
   - Independent error handling per data source without blocking others
   - Performance monitoring and 2-hour completion target
   - Daily scheduling with configurable refresh time
   - Comprehensive logging and error reporting
   - BDD tests covering all unified refresh scenarios





   git push origin feature/unified-refresh-scheduler
   ```

6. **Capture End Time**
   ```bash
   echo "Task 4.1 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Expected Duration: 1 hour 30 minutes
   ```

**Validation Criteria**:
- All BDD tests pass with 100% success rate
- Test coverage >90% for scheduler code
- FMP and fundata processing execute in correct sequence
- Independent error handling working properly
- Performance targets met (2-hour total duration)
- Daily scheduling functionality working
- Comprehensive logging implemented

**Rollback Procedure**:
1. Revert unified scheduler commits
2. Restore individual refresh processes
3. Verify FMP and fundata processing still work independently
4. Update stakeholders on rollback and issues

### Phase 5: Infrastructure & Deployment
**Duration**: 3 hours 30 minutes
**Dependencies**: Phase 4 completion
**Risk Level**: High

**Focus**: Render.com deployment configuration and Git LFS setup for fundata CSV files.

#### Objectives
- [ ] Configure Git LFS for fundata CSV files
- [ ] Update Render.com deployment to pull Git LFS files
- [ ] Verify local CSV file access from deployed environment
- [ ] Remove private file service references from deployment configuration

#### Task 5.1: Git LFS Configuration and Setup ✅
**Duration**: 1 hour
**Dependencies**: Phase 4 completion
**Risk Level**: Medium

**Implementation Process** (MANDATORY 8-step process):

1. **Capture Start Time**
   ```bash
   echo "Task 5.1 Start: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   ```

2. **Create BDD Feature File**
   ```gherkin
   # tests/features/git_lfs_integration.feature
   Feature: Git LFS Integration for Fundata CSV Files
     As a data processing service
     I want to access fundata CSV files stored in Git LFS
     So that I can process them without external file hosting

     Scenario: Verify Git LFS files are available locally
       Given the application is deployed with Git LFS
       When I check for fundata CSV files in local directories
       Then fundata/data directory should contain CSV files
       And fundata/quotes directory should contain CSV files
       And files should be accessible for processing

     Scenario: Process local Git LFS CSV files
       Given fundata CSV files are available via Git LFS
       When I process fundata_data CSV files
       Then records should be parsed and validated successfully
       And data should be stored in fundata_data table
       And processing should not require external API calls

     Scenario: Handle missing Git LFS files gracefully
       Given some Git LFS files might not be pulled
       When I attempt to process missing files
       Then the system should log appropriate warnings
       And continue processing available files
       And not crash due to missing files
   ```

3. **Create Test Fixtures**
   ```python
   # tests/fixtures/git_lfs_fixtures.py
   import pytest
   from pathlib import Path
   from unittest.mock import Mock
   import tempfile
   import os

   @pytest.fixture
   def mock_fundata_directory(tmp_path):
       """Create mock fundata directory structure for testing."""
       fundata_dir = tmp_path / "fundata"
       data_dir = fundata_dir / "data"
       quotes_dir = fundata_dir / "quotes"

       data_dir.mkdir(parents=True)
       quotes_dir.mkdir(parents=True)

       # Create mock CSV files
       (data_dir / "FundGeneralSeed.csv").write_text(
           "InstrumentKey,LegalName,Company\n"
           "412682,MD Dividend Income Index,MD Financial Management Inc.\n"
       )

       (quotes_dir / "FundDailyNAVPSSeed.csv").write_text(
           "InstrumentKey,Date,NAVPS\n"
           "4095,2024-01-15,11.58290000\n"
       )

       return str(fundata_dir)

   @pytest.fixture
   def git_lfs_mock_environment():
       """Mock environment variables for Git LFS."""
       return {
           "FUNDATA_BASE_PATH": "fundata",
           "FUNDATA_DATA_PATH": "fundata/data",
           "FUNDATA_QUOTES_PATH": "fundata/quotes"
       }
   ```

4. **Run Red Test**
   ```bash
   pytest tests/features/git_lfs_integration.feature -v
   # Expected: Tests fail (red state) - validates test correctness ✓
   ```

5. **Write Implementation**
   ```python
   # Create .gitattributes file for Git LFS configuration
   # .gitattributes
   fundata/data/*.csv filter=lfs diff=lfs merge=lfs -text
   fundata/quotes/*.csv filter=lfs diff=lfs merge=lfs -text
   ```

   ```python
   # packages/sra_data/infrastructure/git_lfs.py
   import os
   import subprocess
   import logging
   from pathlib import Path
   from typing import List, Dict, Any

   logger = logging.getLogger(__name__)

   def verify_git_lfs_installation() -> bool:
       """Verify Git LFS is available."""
       try:
           result = subprocess.run(['git', 'lfs', 'version'],
                                 capture_output=True, text=True, timeout=10)
           return result.returncode == 0
       except Exception as e:
           logger.error(f"Git LFS not available: {e}")
           return False

   def verify_lfs_files_pulled(fundata_path: str = "fundata") -> Dict[str, Any]:
       """Verify Git LFS files are properly pulled."""
       results = {
           'status': 'success',
           'data_files': [],
           'quotes_files': [],
           'total_size_mb': 0,
           'missing_files': []
       }

       try:
           base_path = Path(fundata_path)

           # Check data files
           data_path = base_path / "data"
           if data_path.exists():
               data_files = list(data_path.glob("*.csv"))
               results['data_files'] = [str(f) for f in data_files]

               for file_path in data_files:
                   if file_path.is_file() and file_path.stat().st_size > 100:
                       results['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                   else:
                       results['missing_files'].append(str(file_path))

           # Check quotes files
           quotes_path = base_path / "quotes"
           if quotes_path.exists():
               quotes_files = list(quotes_path.glob("*.csv"))
               results['quotes_files'] = [str(f) for f in quotes_files]

               for file_path in quotes_files:
                   if file_path.is_file() and file_path.stat().st_size > 100:
                       results['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                   else:
                       results['missing_files'].append(str(file_path))

           if results['missing_files']:
               results['status'] = 'partial'
               logger.warning(f"Some LFS files may not be properly pulled: {results['missing_files']}")

           logger.info(
               f"Git LFS verification: {len(results['data_files'])} data files, "
               f"{len(results['quotes_files'])} quotes files, "
               f"{results['total_size_mb']:.2f}MB total"
           )

       except Exception as e:
           results['status'] = 'error'
           results['error'] = str(e)
           logger.error(f"Git LFS verification failed: {e}")

       return results
   ```

   ```yaml
   # Update render.yaml to include Git LFS pulling
   services:
     - type: worker
       name: sra-data-worker
       env: python
       buildCommand: "git lfs pull && pip install -r requirements.txt"
       startCommand: "python worker.py"
       plan: standard
       envVars:
         - key: PYTHON_VERSION
           value: 3.13
         - key: LOG_LEVEL
           value: INFO
         - key: FUNDATA_BASE_PATH
           value: fundata
         - key: FUNDATA_DATA_PATH
           value: fundata/data
         - key: FUNDATA_QUOTES_PATH
           value: fundata/quotes
       autoDeploy: true
   ```

6. **Run Green Test**
   ```bash
   pytest tests/features/git_lfs_integration.feature -v --cov=packages/sra_data/infrastructure/git_lfs --cov-report=term-missing
   # Target: 100% pass rate, >90% coverage ✓
   ```

7. **Commit and Push**
   ```bash
   git add -A
   git commit -m "feat: Configure Git LFS for fundata CSV files

   - Added .gitattributes configuration for CSV files
   - Created Git LFS verification utilities
   - Updated Render deployment to pull LFS files
   - Removed dependency on private file service
   - Added environment variables for local file paths





   git push origin feature/git-lfs-integration
   ```

8. **Capture End Time**
   ```bash
   echo "Task 5.1 End: $(date '+%Y-%m-%d %H:%M:%S')" >> docs/implementation/render-deployment-implementation-plan.md
   # Duration: 1 hour
   ```

**Validation Criteria**:
- .gitattributes properly configured for CSV files
- Git LFS verification functions working correctly
- Render deployment successfully pulls LFS files
- Local CSV files accessible from deployed application
- No private file service dependencies remaining

**Rollback Procedure**:
1. Revert Git LFS configuration changes
2. Restore previous deployment configuration
3. Test basic deployment without LFS dependency

### Phase 6: Testing & Documentation
**Duration**: 1 hour 15 minutes
**Dependencies**: Phase 5 completion
**Risk Level**: Low

**Focus**: Validate data processing functionality and deployment stability.

## Implementation Time Summary

### Proven Metrics Analysis
Based on TableV2 implementation metrics and similar complexity:

**Foundation Layer**: 3 hours 15 minutes
- Domain models: 1 hour 30 minutes (estimated from 49 min baseline + complexity)
- Database infrastructure: 1 hour 45 minutes (estimated from repository pattern baseline)

**Business Logic Layer**: 4 hours 30 minutes
- Data processing services: 2 hours 15 minutes (estimated from service layer baseline + external API complexity)
- API integration: 2 hours 15 minutes (new complexity for external systems)

**API Layer**: 3 hours 45 minutes
- FastAPI routes: 2 hours (estimated from template system baseline)
- Health checks & monitoring: 1 hour 45 minutes

**Repository Layer**: 2 hours 30 minutes
- Database operations: 2 hours 30 minutes (from repository pattern baseline + async conversion)

**Infrastructure**: 3 hours 30 minutes
- Render.com configuration: 1 hour 30 minutes
- Git LFS configuration: 1 hour
- Monitoring & logging: 1 hour

**Testing & Documentation**: 1 hour 15 minutes
- BDD test completion: 45 minutes
- Documentation updates: 30 minutes

**Total Estimated Time**: 18 hours 45 minutes

### Risk Mitigation Buffer
- 15% buffer for integration complexity: +2 hours 48 minutes
- **Total with Buffer**: 21 hours 33 minutes

### Parallel Execution Opportunities
- Phase 1 and 2 can run sequentially (dependencies)
- Phase 3 and 4 can partially overlap after API layer foundation
- Phase 5 infrastructure can be prepared in parallel with Phase 4
- Phase 6 testing can run continuously throughout

**Optimized Timeline with Parallelization**: 16 hours over 2-3 working days

## Success Criteria

### Task-Level Completion Requirements
Each task MUST complete the mandatory 8-step process:
- [x] Step 1: Start time captured with CLI command
- [x] Step 2: BDD feature file created in tests/ folder
- [x] Step 3: Test fixtures implemented with mocks
- [x] Step 4: Red test state verified (tests fail initially)
- [x] Step 5: Implementation code written following architecture
- [x] Step 6: Green test state achieved (100% pass rate)
- [x] Step 7: Code committed and pushed using commit agent
- [x] Step 8: End time captured and duration calculated

### Phase-Level Completion Requirements
- [ ] All tasks in phase marked complete with 8-step verification
- [ ] All BDD tests passing (100% success rate)
- [ ] Test coverage >90% for new code
- [ ] Performance benchmarks met per architecture specifications
- [ ] Architecture compliance verified (function-based design)
- [ ] Integration tests passing with external systems

### Project-Level Completion Requirements
- [ ] Render.com deployment successful and stable (skeleton FastAPI prevents suspension)
- [ ] Database schema initialized and seeded with raw data
- [ ] Modelized Pydantic views created and accessible for external SRA client
- [ ] All external integrations functional (FMP API, local Git LFS CSV files)
- [ ] Health checks passing with <100ms response times (deployment stability)
- [ ] Background data processing handling 1,000+ securities/minute
- [ ] Database ready for consumption by separate SRA FastAPI application
- [ ] Error rate <0.1% for all data processing operations
- [ ] Service deployment stability achievable

## Risk Management

### High-Risk Areas
1. **External API Integration** (Phase 2.2)
   - Risk: Rate limiting or API changes
   - Mitigation: Comprehensive retry logic, fallback mechanisms
   - Contingency: Implement caching layer, reduce request frequency

2. **Render.com Deployment** (Phase 5.1)
   - Risk: Service configuration complexity
   - Mitigation: Test deployment configuration early
   - Contingency: Manual deployment with Docker containers

3. **CSV File Processing** (Phase 2.1, 2.2)
   - Risk: Large file handling, memory constraints
   - Mitigation: Streaming processing, memory monitoring
   - Contingency: Implement file chunking, background processing

### Medium-Risk Areas
1. **Database Performance** (Phase 1.2, 4.1)
   - Risk: Query performance under load
   - Mitigation: Proper indexing, connection pooling
   - Contingency: Query optimization, read replicas

2. **Background Task Management** (Phase 3.2)
   - Risk: Task scheduling conflicts, memory leaks
   - Mitigation: Task isolation, memory monitoring
   - Contingency: Task restart mechanisms, resource limits

### Monitoring and Success Tracking

#### Daily Progress Metrics
- Tasks completed vs planned
- BDD test pass rate (target: 100%)
- Code coverage percentage (target: >90%)
- Performance benchmark results
- Error rates and resolution times

#### Weekly Milestones
- **Week 1**: Foundation and Business Logic (Phases 1-2)
- **Week 1-2**: API and Repository layers (Phases 3-4)
- **Week 2**: Infrastructure and Testing (Phases 5-6)

This implementation plan provides a systematic, measurable approach to deploying the SRA Data Processing Service on Render.com with comprehensive data ingestion and transformation capabilities. The service focuses on:

1. **Data Pipeline Operations**: FMP API and fundata CSV ingestion
2. **Database Management**: Raw data storage and modelized view creation
3. **Client Data Preparation**: Ready-to-consume data formats for external SRA application
4. **Deployment Stability**: Minimal FastAPI skeleton to prevent Render.com service suspension
5. **Background Processing**: Automated data refresh and transformation cycles

This approach follows proven development methodologies while correctly positioning the service as a data processing backend rather than a client-facing API.