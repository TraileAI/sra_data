"""Transaction management and data integrity for SRA Data processing.

This module provides comprehensive transaction management, data validation,
constraint enforcement, and integrity checking for database operations.
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union, Callable, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from packages.sra_data.repositories.database import DatabaseManager
from packages.sra_data.domain.models import ProcessingResult

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


class TransactionIsolationLevel(str, Enum):
    """Database transaction isolation levels."""
    READ_UNCOMMITTED = "READ UNCOMMITTED"
    READ_COMMITTED = "READ COMMITTED"
    REPEATABLE_READ = "REPEATABLE READ"
    SERIALIZABLE = "SERIALIZABLE"


class IntegrityViolationType(str, Enum):
    """Types of data integrity violations."""
    DUPLICATE_KEY = "duplicate_key"
    FOREIGN_KEY = "foreign_key"
    CHECK_CONSTRAINT = "check_constraint"
    NOT_NULL = "not_null"
    DATA_TYPE = "data_type"
    BUSINESS_RULE = "business_rule"


@dataclass
class IntegrityViolation:
    """Data integrity violation record."""
    violation_type: IntegrityViolationType
    table_name: str
    column_name: Optional[str]
    constraint_name: Optional[str]
    error_message: str
    data_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TransactionResult:
    """Result of a transaction operation."""
    success: bool
    transaction_id: Optional[str] = None
    operations_count: int = 0
    affected_rows: int = 0
    duration_seconds: float = 0.0
    integrity_violations: List[IntegrityViolation] = field(default_factory=list)
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None


class TransactionManager:
    """Advanced transaction management with data integrity enforcement."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize transaction manager.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self._active_transactions = {}
        self._transaction_counter = 0

    @asynccontextmanager
    async def transaction(
        self,
        isolation_level: TransactionIsolationLevel = TransactionIsolationLevel.READ_COMMITTED,
        readonly: bool = False,
        validate_constraints: bool = True
    ) -> AsyncGenerator[asyncpg.Connection, None]:
        """Create a managed database transaction with integrity checking.

        Args:
            isolation_level: Transaction isolation level
            readonly: Whether transaction is read-only
            validate_constraints: Whether to validate business constraints

        Yields:
            Database connection within transaction context
        """
        self._transaction_counter += 1
        transaction_id = f"txn_{self._transaction_counter}_{datetime.utcnow().strftime('%H%M%S')}"
        start_time = datetime.utcnow()

        logger.debug(f"Starting transaction {transaction_id} (isolation: {isolation_level.value})")

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Begin transaction with specified isolation level
                if readonly:
                    await connection.execute(f"BEGIN READ ONLY ISOLATION LEVEL {isolation_level.value}")
                else:
                    await connection.execute(f"BEGIN ISOLATION LEVEL {isolation_level.value}")

                self._active_transactions[transaction_id] = {
                    'start_time': start_time,
                    'isolation_level': isolation_level,
                    'readonly': readonly,
                    'validate_constraints': validate_constraints
                }

                try:
                    yield connection
                    # Transaction completed successfully
                    await connection.execute("COMMIT")
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    logger.debug(f"Transaction {transaction_id} committed successfully in {duration:.3f}s")

                except Exception as e:
                    # Transaction failed, rollback
                    await connection.execute("ROLLBACK")
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    logger.warning(f"Transaction {transaction_id} rolled back after {duration:.3f}s: {e}")
                    raise

                finally:
                    # Clean up transaction tracking
                    self._active_transactions.pop(transaction_id, None)

        except Exception as e:
            logger.error(f"Transaction {transaction_id} failed: {e}")
            raise

    async def execute_with_retry(
        self,
        operation: Callable[[asyncpg.Connection], Any],
        max_retries: int = 3,
        retry_delay: float = 0.1,
        isolation_level: TransactionIsolationLevel = TransactionIsolationLevel.READ_COMMITTED
    ) -> Any:
        """Execute operation with automatic retry on transient failures.

        Args:
            operation: Async callable that takes a connection and performs database operations
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            isolation_level: Transaction isolation level

        Returns:
            Result of the operation

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                async with self.transaction(isolation_level=isolation_level) as conn:
                    result = await operation(conn)
                    logger.debug(f"Operation succeeded on attempt {attempt + 1}")
                    return result

            except (asyncpg.SerializationError, asyncpg.DeadlockDetectedError) as e:
                # Transient errors that can be retried
                last_exception = e
                if attempt < max_retries:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Transient error on attempt {attempt + 1}, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                    break

            except Exception as e:
                # Non-transient errors should not be retried
                logger.error(f"Non-retryable error on attempt {attempt + 1}: {e}")
                raise

        # If we get here, all retry attempts failed
        raise last_exception or Exception("Operation failed after all retry attempts")

    async def validate_data_integrity(
        self,
        table_name: str,
        record_data: Dict[str, Any],
        operation_type: str = "INSERT"
    ) -> List[IntegrityViolation]:
        """Validate data integrity before database operations.

        Args:
            table_name: Target table name
            record_data: Data to validate
            operation_type: Type of operation (INSERT, UPDATE, DELETE)

        Returns:
            List of integrity violations found
        """
        violations = []

        try:
            async with self.db_manager.pool.acquire() as connection:
                # Validate based on table type
                if table_name == "equity_profile":
                    violations.extend(await self._validate_equity_profile(connection, record_data))
                elif table_name == "fundata_data":
                    violations.extend(await self._validate_fundata_data(connection, record_data))
                elif table_name == "fundata_quotes":
                    violations.extend(await self._validate_fundata_quotes(connection, record_data))

                # Common validations for all tables
                violations.extend(await self._validate_common_constraints(connection, table_name, record_data))

        except Exception as e:
            logger.error(f"Data integrity validation failed for {table_name}: {e}")
            violations.append(IntegrityViolation(
                violation_type=IntegrityViolationType.BUSINESS_RULE,
                table_name=table_name,
                column_name=None,
                constraint_name=None,
                error_message=f"Validation process failed: {e}",
                data_context=record_data
            ))

        return violations

    async def _validate_equity_profile(
        self,
        connection: asyncpg.Connection,
        record_data: Dict[str, Any]
    ) -> List[IntegrityViolation]:
        """Validate equity profile specific business rules."""
        violations = []

        # Check for required fields
        required_fields = ['symbol', 'company_name', 'exchange']
        for field in required_fields:
            if not record_data.get(field):
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.NOT_NULL,
                    table_name="equity_profile",
                    column_name=field,
                    constraint_name=f"equity_profile_{field}_not_null",
                    error_message=f"Required field {field} is missing or empty",
                    data_context=record_data
                ))

        # Validate symbol format
        symbol = record_data.get('symbol', '').strip().upper()
        if symbol and not symbol.replace('.', '').replace('-', '').isalnum():
            violations.append(IntegrityViolation(
                violation_type=IntegrityViolationType.CHECK_CONSTRAINT,
                table_name="equity_profile",
                column_name="symbol",
                constraint_name="equity_profile_symbol_format",
                error_message=f"Invalid symbol format: {symbol}",
                data_context=record_data
            ))

        # Validate market cap
        market_cap = record_data.get('market_cap')
        if market_cap is not None and market_cap < 0:
            violations.append(IntegrityViolation(
                violation_type=IntegrityViolationType.CHECK_CONSTRAINT,
                table_name="equity_profile",
                column_name="market_cap",
                constraint_name="equity_profile_market_cap_positive",
                error_message=f"Market cap cannot be negative: {market_cap}",
                data_context=record_data
            ))

        # Check for duplicate symbol
        if symbol:
            existing = await connection.fetchval(
                "SELECT symbol FROM equity_profile WHERE symbol = $1",
                symbol
            )
            if existing:
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.DUPLICATE_KEY,
                    table_name="equity_profile",
                    column_name="symbol",
                    constraint_name="equity_profile_pkey",
                    error_message=f"Duplicate symbol: {symbol}",
                    data_context=record_data
                ))

        return violations

    async def _validate_fundata_data(
        self,
        connection: asyncpg.Connection,
        record_data: Dict[str, Any]
    ) -> List[IntegrityViolation]:
        """Validate fundata data specific business rules."""
        violations = []

        # Check required fields
        required_fields = ['InstrumentKey', 'RecordId', 'source_file']
        for field in required_fields:
            if not record_data.get(field):
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.NOT_NULL,
                    table_name="fundata_data",
                    column_name=field.lower(),
                    constraint_name=f"fundata_data_{field.lower()}_not_null",
                    error_message=f"Required field {field} is missing or empty",
                    data_context=record_data
                ))

        # Validate InstrumentKey format
        instrument_key = record_data.get('InstrumentKey', '').strip()
        if instrument_key and len(instrument_key) > 20:
            violations.append(IntegrityViolation(
                violation_type=IntegrityViolationType.CHECK_CONSTRAINT,
                table_name="fundata_data",
                column_name="instrument_key",
                constraint_name="fundata_data_instrument_key_length",
                error_message=f"InstrumentKey too long: {len(instrument_key)} chars (max 20)",
                data_context=record_data
            ))

        # Check for duplicate combination
        if instrument_key and record_data.get('RecordId'):
            existing = await connection.fetchval(
                "SELECT id FROM fundata_data WHERE instrument_key = $1 AND record_id = $2",
                instrument_key,
                record_data['RecordId']
            )
            if existing:
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.DUPLICATE_KEY,
                    table_name="fundata_data",
                    column_name="instrument_key,record_id",
                    constraint_name="unique_fundata_data",
                    error_message=f"Duplicate instrument_key/record_id: {instrument_key}/{record_data['RecordId']}",
                    data_context=record_data
                ))

        return violations

    async def _validate_fundata_quotes(
        self,
        connection: asyncpg.Connection,
        record_data: Dict[str, Any]
    ) -> List[IntegrityViolation]:
        """Validate fundata quotes specific business rules."""
        violations = []

        # Check required fields
        required_fields = ['InstrumentKey', 'RecordId', 'Date', 'NAVPS', 'source_file']
        for field in required_fields:
            if field not in record_data or record_data[field] is None:
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.NOT_NULL,
                    table_name="fundata_quotes",
                    column_name=field.lower(),
                    constraint_name=f"fundata_quotes_{field.lower()}_not_null",
                    error_message=f"Required field {field} is missing or null",
                    data_context=record_data
                ))

        # Validate NAVPS
        navps = record_data.get('NAVPS')
        if navps is not None:
            try:
                navps_value = float(navps)
                if navps_value <= 0:
                    violations.append(IntegrityViolation(
                        violation_type=IntegrityViolationType.CHECK_CONSTRAINT,
                        table_name="fundata_quotes",
                        column_name="navps",
                        constraint_name="fundata_quotes_navps_positive",
                        error_message=f"NAVPS must be positive: {navps}",
                        data_context=record_data
                    ))
                elif navps_value > 10000:
                    violations.append(IntegrityViolation(
                        violation_type=IntegrityViolationType.CHECK_CONSTRAINT,
                        table_name="fundata_quotes",
                        column_name="navps",
                        constraint_name="fundata_quotes_navps_reasonable",
                        error_message=f"NAVPS value seems unreasonable: {navps}",
                        data_context=record_data
                    ))
            except (ValueError, TypeError):
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.DATA_TYPE,
                    table_name="fundata_quotes",
                    column_name="navps",
                    constraint_name="fundata_quotes_navps_numeric",
                    error_message=f"Invalid NAVPS value (not numeric): {navps}",
                    data_context=record_data
                ))

        # Validate date
        quote_date = record_data.get('Date')
        if quote_date:
            try:
                if hasattr(quote_date, 'date'):  # datetime object
                    quote_date = quote_date.date()

                # Don't allow future dates
                from datetime import date
                if quote_date > date.today():
                    violations.append(IntegrityViolation(
                        violation_type=IntegrityViolationType.CHECK_CONSTRAINT,
                        table_name="fundata_quotes",
                        column_name="date",
                        constraint_name="fundata_quotes_date_not_future",
                        error_message=f"Quote date cannot be in the future: {quote_date}",
                        data_context=record_data
                    ))

                # Don't allow very old dates
                if quote_date.year < 1900:
                    violations.append(IntegrityViolation(
                        violation_type=IntegrityViolationType.CHECK_CONSTRAINT,
                        table_name="fundata_quotes",
                        column_name="date",
                        constraint_name="fundata_quotes_date_reasonable",
                        error_message=f"Quote date too old: {quote_date}",
                        data_context=record_data
                    ))
            except Exception:
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.DATA_TYPE,
                    table_name="fundata_quotes",
                    column_name="date",
                    constraint_name="fundata_quotes_date_valid",
                    error_message=f"Invalid date format: {quote_date}",
                    data_context=record_data
                ))

        # Check for duplicate quote
        if all(record_data.get(field) for field in ['InstrumentKey', 'RecordId', 'Date']):
            existing = await connection.fetchval(
                "SELECT id FROM fundata_quotes WHERE instrument_key = $1 AND record_id = $2 AND date = $3",
                record_data['InstrumentKey'],
                record_data['RecordId'],
                record_data['Date']
            )
            if existing:
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.DUPLICATE_KEY,
                    table_name="fundata_quotes",
                    column_name="instrument_key,record_id,date",
                    constraint_name="unique_fundata_quotes",
                    error_message=f"Duplicate quote: {record_data['InstrumentKey']}/{record_data['RecordId']}/{record_data['Date']}",
                    data_context=record_data
                ))

        return violations

    async def _validate_common_constraints(
        self,
        connection: asyncpg.Connection,
        table_name: str,
        record_data: Dict[str, Any]
    ) -> List[IntegrityViolation]:
        """Validate common constraints across all tables."""
        violations = []

        # Check string field lengths
        string_fields = {
            'equity_profile': {
                'symbol': 10,
                'company_name': 255,
                'exchange': 10,
                'sector': 100,
                'industry': 100,
                'website': 255,
                'country': 3,
                'currency': 3
            },
            'fundata_data': {
                'instrument_key': 20,
                'record_id': 20,
                'language': 5,
                'legal_name': 500,
                'family_name': 255,
                'series_name': 255,
                'company': 255,
                'currency': 3,
                'record_state': 20,
                'source_file': 255,
                'file_hash': 64
            },
            'fundata_quotes': {
                'instrument_key': 20,
                'record_id': 20,
                'record_state': 20,
                'source_file': 255,
                'file_hash': 64
            }
        }

        table_fields = string_fields.get(table_name, {})
        for field, max_length in table_fields.items():
            # Convert field name for checking (handle case differences)
            check_field = field
            for key in record_data.keys():
                if key.lower() == field.lower():
                    check_field = key
                    break

            value = record_data.get(check_field)
            if value and len(str(value)) > max_length:
                violations.append(IntegrityViolation(
                    violation_type=IntegrityViolationType.CHECK_CONSTRAINT,
                    table_name=table_name,
                    column_name=field,
                    constraint_name=f"{table_name}_{field}_length",
                    error_message=f"Field {field} exceeds maximum length {max_length}: {len(str(value))} chars",
                    data_context=record_data
                ))

        return violations

    async def batch_operation_with_integrity(
        self,
        operations: List[Callable[[asyncpg.Connection], Any]],
        stop_on_first_violation: bool = False,
        validate_each_operation: bool = True
    ) -> TransactionResult:
        """Execute batch operations with comprehensive integrity checking.

        Args:
            operations: List of database operations to execute
            stop_on_first_violation: Whether to stop on first integrity violation
            validate_each_operation: Whether to validate each operation individually

        Returns:
            Transaction result with integrity violation details
        """
        start_time = datetime.utcnow()
        result = TransactionResult(success=False, operations_count=len(operations))

        try:
            async with self.transaction() as conn:
                affected_rows = 0
                violations_found = []

                for i, operation in enumerate(operations):
                    try:
                        # Execute operation
                        op_result = await operation(conn)

                        # Count affected rows if result indicates it
                        if hasattr(op_result, 'split') and op_result.split():
                            try:
                                rows = int(op_result.split()[-1])
                                affected_rows += rows
                            except (ValueError, IndexError):
                                pass

                    except Exception as e:
                        error_msg = str(e)

                        # Categorize the error
                        violation_type = self._categorize_database_error(error_msg)

                        violation = IntegrityViolation(
                            violation_type=violation_type,
                            table_name="unknown",
                            column_name=None,
                            constraint_name=None,
                            error_message=f"Operation {i+1}: {error_msg}",
                            data_context={"operation_index": i}
                        )
                        violations_found.append(violation)

                        if stop_on_first_violation:
                            logger.warning(f"Stopping batch operation on violation: {error_msg}")
                            break

                result.success = len(violations_found) == 0
                result.affected_rows = affected_rows
                result.integrity_violations = violations_found

                if not result.success:
                    result.rollback_reason = f"Integrity violations found: {len(violations_found)}"

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Batch operation failed: {e}")

        finally:
            result.duration_seconds = (datetime.utcnow() - start_time).total_seconds()

        return result

    def _categorize_database_error(self, error_message: str) -> IntegrityViolationType:
        """Categorize database error into integrity violation type."""
        error_lower = error_message.lower()

        if "duplicate key" in error_lower or "unique constraint" in error_lower:
            return IntegrityViolationType.DUPLICATE_KEY
        elif "foreign key" in error_lower:
            return IntegrityViolationType.FOREIGN_KEY
        elif "check constraint" in error_lower:
            return IntegrityViolationType.CHECK_CONSTRAINT
        elif "not null" in error_lower or "null value" in error_lower:
            return IntegrityViolationType.NOT_NULL
        elif "invalid input" in error_lower or "type" in error_lower:
            return IntegrityViolationType.DATA_TYPE
        else:
            return IntegrityViolationType.BUSINESS_RULE

    async def get_constraint_violations_report(self) -> Dict[str, Any]:
        """Generate a report of constraint violations across all tables.

        Returns:
            Comprehensive constraint violations report
        """
        try:
            async with self.db_manager.pool.acquire() as connection:
                # Get constraint information
                constraints = await connection.fetch("""
                    SELECT
                        tc.table_name,
                        tc.constraint_name,
                        tc.constraint_type,
                        pg_get_constraintdef(c.oid) as definition
                    FROM information_schema.table_constraints tc
                    JOIN pg_constraint c ON c.conname = tc.constraint_name
                    WHERE tc.table_schema = 'public'
                    AND tc.table_name IN ('equity_profile', 'fundata_data', 'fundata_quotes')
                    ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name
                """)

                # Get foreign key relationships
                fk_relationships = await connection.fetch("""
                    SELECT
                        kcu.table_name as source_table,
                        kcu.column_name as source_column,
                        ccu.table_name as target_table,
                        ccu.column_name as target_column,
                        tc.constraint_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_schema = 'public'
                """)

                return {
                    'constraints_summary': {
                        'total_constraints': len(constraints),
                        'by_type': {},
                        'by_table': {}
                    },
                    'constraints_detail': [dict(row) for row in constraints],
                    'foreign_key_relationships': [dict(row) for row in fk_relationships],
                    'generated_at': datetime.utcnow().isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to generate constraints report: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.utcnow().isoformat()
            }

    async def health_check(self) -> Dict[str, Any]:
        """Perform transaction manager health check.

        Returns:
            Health check results
        """
        try:
            return {
                'status': 'healthy',
                'active_transactions': len(self._active_transactions),
                'transaction_counter': self._transaction_counter,
                'features_available': {
                    'transaction_isolation': True,
                    'retry_mechanism': True,
                    'integrity_validation': True,
                    'batch_operations': True,
                    'constraint_reporting': True
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }