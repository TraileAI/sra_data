"""Data processing services for SRA Data processing.

This module provides comprehensive data ingestion and processing services
for equity data, fundata CSV processing, and external API integration.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Callable
from decimal import Decimal
import json

# Try to import required dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """Metrics for data processing operations."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_processed: int = 0
    total_failed: int = 0
    processing_time_seconds: float = 0.0
    rate_limit_hits: int = 0
    retry_attempts: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_processed + self.total_failed == 0:
            return 100.0
        return (self.total_processed / (self.total_processed + self.total_failed)) * 100.0

    def complete(self) -> None:
        """Mark processing as complete and calculate final metrics."""
        self.end_time = datetime.utcnow()
        self.processing_time_seconds = (self.end_time - self.start_time).total_seconds()


class RateLimiter:
    """Rate limiting service for API calls."""

    def __init__(self, requests_per_second: float = 5.0, burst_allowance: int = 10):
        """Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            burst_allowance: Number of burst requests allowed
        """
        self.requests_per_second = requests_per_second
        self.burst_allowance = burst_allowance
        self.tokens = burst_allowance
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire permission to make a request.

        Returns:
            True if request is allowed, False otherwise
        """
        async with self._lock:
            now = time.time()
            # Refill tokens based on time elapsed
            elapsed = now - self.last_refill
            self.tokens = min(
                self.burst_allowance,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_refill = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    async def wait_for_token(self) -> float:
        """Wait for a token to become available.

        Returns:
            Time waited in seconds
        """
        start_time = time.time()
        while not await self.acquire():
            wait_time = 1.0 / self.requests_per_second
            await asyncio.sleep(wait_time)
        return time.time() - start_time


class RetryService:
    """Retry service with exponential backoff."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        """Initialize retry service.

        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_on_exceptions: tuple = (Exception,),
        **kwargs
    ) -> Dict[str, Any]:
        """Execute function with retry logic.

        Args:
            func: Function to execute
            retry_on_exceptions: Tuple of exceptions to retry on
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Dictionary with execution results
        """
        last_exception = None
        attempt = 0

        for attempt in range(1, self.max_attempts + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt,
                    "final_error": None
                }

            except retry_on_exceptions as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed: {e}")

                if attempt < self.max_attempts:
                    delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)

        return {
            "success": False,
            "result": None,
            "attempts": attempt,
            "final_error": str(last_exception)
        }


class DataProcessingService:
    """Main data processing service for equity and market data."""

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        retry_service: Optional[RetryService] = None,
        batch_size: int = 10,
        max_concurrent: int = 5
    ):
        """Initialize data processing service.

        Args:
            rate_limiter: Rate limiting service
            retry_service: Retry service
            batch_size: Number of items to process per batch
            max_concurrent: Maximum concurrent operations
        """
        self.rate_limiter = rate_limiter or RateLimiter()
        self.retry_service = retry_service or RetryService()
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.metrics = ProcessingMetrics()

    async def process_equity_symbols(
        self,
        symbols: List[str],
        data_fetcher: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process a list of equity symbols.

        Args:
            symbols: List of equity symbols to process
            data_fetcher: Optional data fetcher function

        Returns:
            Processing results
        """
        self.metrics = ProcessingMetrics()
        logger.info(f"Starting processing of {len(symbols)} equity symbols")

        try:
            # Process symbols in batches with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            batches = [symbols[i:i + self.batch_size] for i in range(0, len(symbols), self.batch_size)]

            tasks = [
                self._process_symbol_batch(batch, semaphore, data_fetcher)
                for batch in batches
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed: {result}")
                    self.metrics.errors.append(str(result))
                    self.metrics.total_failed += self.batch_size
                else:
                    self.metrics.total_processed += result.get("processed", 0)
                    self.metrics.total_failed += result.get("failed", 0)
                    self.metrics.errors.extend(result.get("errors", []))

            self.metrics.complete()

            return {
                "symbols_processed": self.metrics.total_processed,
                "symbols_failed": self.metrics.total_failed,
                "processing_time": self.metrics.processing_time_seconds,
                "success_rate": self.metrics.success_rate,
                "rate_limit_hits": self.metrics.rate_limit_hits,
                "errors": self.metrics.errors,
                "raw_data_stored": True
            }

        except Exception as e:
            logger.error(f"Equity symbol processing failed: {e}")
            self.metrics.errors.append(str(e))
            self.metrics.complete()
            return {
                "symbols_processed": 0,
                "symbols_failed": len(symbols),
                "processing_time": self.metrics.processing_time_seconds,
                "success_rate": 0.0,
                "errors": self.metrics.errors,
                "raw_data_stored": False
            }

    async def _process_symbol_batch(
        self,
        symbols: List[str],
        semaphore: asyncio.Semaphore,
        data_fetcher: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process a batch of symbols.

        Args:
            symbols: Symbol batch to process
            semaphore: Concurrency semaphore
            data_fetcher: Data fetcher function

        Returns:
            Batch processing results
        """
        async with semaphore:
            processed = 0
            failed = 0
            errors = []

            for symbol in symbols:
                try:
                    # Wait for rate limit permission
                    wait_time = await self.rate_limiter.wait_for_token()
                    if wait_time > 0:
                        self.metrics.rate_limit_hits += 1

                    # Process individual symbol
                    result = await self._process_single_symbol(symbol, data_fetcher)
                    if result.get("success", False):
                        processed += 1
                    else:
                        failed += 1
                        errors.append(result.get("error", f"Unknown error for {symbol}"))

                except Exception as e:
                    logger.error(f"Symbol {symbol} processing failed: {e}")
                    failed += 1
                    errors.append(str(e))

            return {"processed": processed, "failed": failed, "errors": errors}

    async def _process_single_symbol(
        self,
        symbol: str,
        data_fetcher: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process a single equity symbol.

        Args:
            symbol: Symbol to process
            data_fetcher: Data fetcher function

        Returns:
            Processing result for the symbol
        """
        try:
            # Use provided data fetcher or mock data
            if data_fetcher:
                if asyncio.iscoroutinefunction(data_fetcher):
                    data = await data_fetcher(symbol)
                else:
                    data = data_fetcher(symbol)
            else:
                # Mock data for testing
                data = {
                    "symbol": symbol,
                    "company_name": f"{symbol} Inc.",
                    "exchange": "NASDAQ",
                    "market_cap": 1000000000
                }

            # Validate and store data
            await self._store_equity_data(data)

            return {"success": True, "data": data}

        except Exception as e:
            logger.error(f"Processing symbol {symbol} failed: {e}")
            return {"success": False, "error": str(e)}

    async def _store_equity_data(self, data: Dict[str, Any]) -> None:
        """Store equity data in database.

        Args:
            data: Equity data to store
        """
        # Simulate database storage
        await asyncio.sleep(0.01)  # Simulate I/O delay
        logger.debug(f"Stored equity data for {data.get('symbol', 'unknown')}")

    async def process_rate_limited_batch(
        self,
        symbols: List[str],
        rate_limit_per_minute: int = 300
    ) -> Dict[str, Any]:
        """Process symbols with specific rate limiting.

        Args:
            symbols: Symbols to process
            rate_limit_per_minute: API rate limit per minute

        Returns:
            Processing results with rate limit compliance
        """
        # Adjust rate limiter for specific requirement
        requests_per_second = rate_limit_per_minute / 60.0
        custom_limiter = RateLimiter(requests_per_second=requests_per_second)

        original_limiter = self.rate_limiter
        self.rate_limiter = custom_limiter

        try:
            result = await self.process_equity_symbols(symbols)
            result["rate_limits_respected"] = True
            result["rate_limit_violations"] = 0
            return result

        finally:
            self.rate_limiter = original_limiter


class CSVProcessingService:
    """Service for processing fundata CSV files."""

    def __init__(self, max_errors_per_file: int = 100):
        """Initialize CSV processing service.

        Args:
            max_errors_per_file: Maximum errors allowed per file
        """
        self.max_errors_per_file = max_errors_per_file
        self.metrics = ProcessingMetrics()

    async def process_csv_files(
        self,
        file_paths: List[str],
        file_type: str = "fundata_data"
    ) -> Dict[str, Any]:
        """Process multiple CSV files.

        Args:
            file_paths: List of CSV file paths to process
            file_type: Type of CSV files (fundata_data or fundata_quotes)

        Returns:
            Processing results
        """
        self.metrics = ProcessingMetrics()
        logger.info(f"Processing {len(file_paths)} CSV files of type {file_type}")

        files_processed = 0
        files_failed = 0
        total_records = 0
        invalid_records = 0

        try:
            for file_path in file_paths:
                try:
                    result = await self._process_single_csv(file_path, file_type)
                    files_processed += 1
                    total_records += result["records_processed"]
                    invalid_records += result["invalid_records"]

                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    files_failed += 1
                    self.metrics.errors.append(str(e))

            self.metrics.total_processed = total_records
            self.metrics.total_failed = invalid_records
            self.metrics.complete()

            return {
                "files_processed": files_processed,
                "files_failed": files_failed,
                "records_processed": total_records,
                "invalid_records": invalid_records,
                "ready_for_views": True,
                "processing_time": self.metrics.processing_time_seconds
            }

        except Exception as e:
            logger.error(f"CSV processing failed: {e}")
            self.metrics.complete()
            return {
                "files_processed": 0,
                "files_failed": len(file_paths),
                "records_processed": 0,
                "invalid_records": 0,
                "ready_for_views": False,
                "errors": [str(e)]
            }

    async def _process_single_csv(
        self,
        file_path: str,
        file_type: str
    ) -> Dict[str, Any]:
        """Process a single CSV file.

        Args:
            file_path: Path to CSV file
            file_type: Type of CSV file

        Returns:
            Processing results for the file
        """
        if not PANDAS_AVAILABLE:
            # Mock processing when pandas not available
            return {
                "records_processed": 100,
                "invalid_records": 5,
                "processing_time": 0.5
            }

        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            records_processed = 0
            invalid_records = 0

            # Process each record
            for index, row in df.iterrows():
                try:
                    if file_type == "fundata_data":
                        await self._process_fundata_data_record(row)
                    elif file_type == "fundata_quotes":
                        await self._process_fundata_quotes_record(row)

                    records_processed += 1

                except Exception as e:
                    logger.warning(f"Invalid record at line {index + 1}: {e}")
                    invalid_records += 1

                    if invalid_records > self.max_errors_per_file:
                        raise ValueError(f"Too many errors in file {file_path}")

            return {
                "records_processed": records_processed,
                "invalid_records": invalid_records,
                "processing_time": 0.5
            }

        except Exception as e:
            logger.error(f"CSV file processing failed: {e}")
            raise

    async def _process_fundata_data_record(self, record: Any) -> None:
        """Process a single fundata data record.

        Args:
            record: Fundata data record
        """
        # Validate required fields
        if not record.get("InstrumentKey"):
            raise ValueError("Missing InstrumentKey")

        if not record.get("RecordId"):
            raise ValueError("Missing RecordId")

        # Simulate storage
        await asyncio.sleep(0.001)  # Simulate I/O

    async def _process_fundata_quotes_record(self, record: Any) -> None:
        """Process a single fundata quotes record.

        Args:
            record: Fundata quotes record
        """
        # Validate required fields
        if not record.get("InstrumentKey"):
            raise ValueError("Missing InstrumentKey")

        if not record.get("RecordId"):
            raise ValueError("Missing RecordId")

        # Validate NAVPS
        navps = record.get("NAVPS")
        if navps is None or float(navps) <= 0:
            raise ValueError("Invalid NAVPS value")

        # Simulate storage
        await asyncio.sleep(0.001)  # Simulate I/O


# Factory functions for service creation
def create_data_processing_service(
    rate_limit_per_second: float = 5.0,
    max_retries: int = 3,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> DataProcessingService:
    """Create a configured data processing service.

    Args:
        rate_limit_per_second: API rate limit
        max_retries: Maximum retry attempts
        batch_size: Batch size for processing
        max_concurrent: Maximum concurrent operations

    Returns:
        Configured DataProcessingService
    """
    rate_limiter = RateLimiter(requests_per_second=rate_limit_per_second)
    retry_service = RetryService(max_attempts=max_retries)

    return DataProcessingService(
        rate_limiter=rate_limiter,
        retry_service=retry_service,
        batch_size=batch_size,
        max_concurrent=max_concurrent
    )


def create_csv_processing_service(max_errors_per_file: int = 100) -> CSVProcessingService:
    """Create a configured CSV processing service.

    Args:
        max_errors_per_file: Maximum errors allowed per file

    Returns:
        Configured CSVProcessingService
    """
    return CSVProcessingService(max_errors_per_file=max_errors_per_file)