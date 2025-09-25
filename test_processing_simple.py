#!/usr/bin/env python3
"""Simple test script to verify data processing services work correctly."""

import sys
import os
import asyncio

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'packages'))

async def test_data_processing_service():
    """Test data processing service."""
    print("Testing DataProcessingService...")

    try:
        from sra_data.services.data_processing import DataProcessingService

        # Create service
        service = DataProcessingService(batch_size=5, max_concurrent=2)
        print("✓ DataProcessingService created")

        # Test processing small batch of symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        result = await service.process_equity_symbols(symbols)

        print(f"✓ Processed {result['symbols_processed']} symbols")
        print(f"✓ Success rate: {result['success_rate']:.1f}%")
        print(f"✓ Processing time: {result['processing_time']:.2f}s")

        return result['symbols_processed'] == len(symbols)

    except Exception as e:
        print(f"✗ DataProcessingService test failed: {e}")
        return False

async def test_rate_limiter():
    """Test rate limiting functionality."""
    print("\nTesting RateLimiter...")

    try:
        from sra_data.services.data_processing import RateLimiter

        # Create rate limiter (allow 2 requests per second)
        limiter = RateLimiter(requests_per_second=2.0, burst_allowance=3)
        print("✓ RateLimiter created")

        # Test burst allowance
        burst_count = 0
        for _ in range(3):
            if await limiter.acquire():
                burst_count += 1

        print(f"✓ Burst requests allowed: {burst_count}/3")

        # Test rate limiting
        start_time = asyncio.get_event_loop().time()
        wait_time = await limiter.wait_for_token()
        end_time = asyncio.get_event_loop().time()

        print(f"✓ Rate limiting working, wait time: {wait_time:.2f}s")

        return burst_count == 3 and wait_time > 0

    except Exception as e:
        print(f"✗ RateLimiter test failed: {e}")
        return False

async def test_retry_service():
    """Test retry service functionality."""
    print("\nTesting RetryService...")

    try:
        from sra_data.services.data_processing import RetryService

        # Create retry service
        retry_service = RetryService(max_attempts=3, base_delay=0.1)
        print("✓ RetryService created")

        # Test successful operation
        async def successful_operation():
            return {"result": "success"}

        result = await retry_service.execute_with_retry(successful_operation)
        print(f"✓ Successful operation: attempts={result['attempts']}")

        # Test failing operation (mock)
        attempt_count = 0
        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary failure")
            return {"result": "success after retries"}

        result = await retry_service.execute_with_retry(failing_operation)
        print(f"✓ Retry logic working: attempts={result['attempts']}, success={result['success']}")

        return result['success'] and result['attempts'] == 3

    except Exception as e:
        print(f"✗ RetryService test failed: {e}")
        return False

async def test_csv_processing_service():
    """Test CSV processing service."""
    print("\nTesting CSVProcessingService...")

    try:
        from sra_data.services.data_processing import CSVProcessingService

        # Create CSV service
        service = CSVProcessingService(max_errors_per_file=10)
        print("✓ CSVProcessingService created")

        # Test processing mock files (will use mock processing since files don't exist)
        file_paths = ["fundata_data.csv", "fundata_quotes.csv"]
        result = await service.process_csv_files(file_paths, file_type="fundata_data")

        print(f"✓ Processed {result['files_processed']} files")
        print(f"✓ Records processed: {result['records_processed']}")
        print(f"✓ Ready for views: {result['ready_for_views']}")

        # Files don't exist, so service should handle gracefully
        return result['files_processed'] == 0 and 'ready_for_views' in result

    except Exception as e:
        print(f"✗ CSVProcessingService test failed: {e}")
        return False

async def test_rate_limited_processing():
    """Test rate-limited processing."""
    print("\nTesting rate-limited processing...")

    try:
        from sra_data.services.data_processing import DataProcessingService

        service = DataProcessingService()
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        # Test with rate limiting
        result = await service.process_rate_limited_batch(symbols, rate_limit_per_minute=300)

        print(f"✓ Rate limits respected: {result.get('rate_limits_respected', False)}")
        print(f"✓ Rate limit violations: {result.get('rate_limit_violations', 0)}")
        print(f"✓ Symbols processed: {result['symbols_processed']}")

        return result.get('rate_limits_respected', False) and result['symbols_processed'] > 0

    except Exception as e:
        print(f"✗ Rate-limited processing test failed: {e}")
        return False

async def main():
    """Run all data processing service tests."""
    print("Running Data Processing Services Tests")
    print("="*50)

    success = True
    success &= await test_data_processing_service()
    success &= await test_rate_limiter()
    success &= await test_retry_service()
    success &= await test_csv_processing_service()
    success &= await test_rate_limited_processing()

    print("\n" + "="*50)
    if success:
        print("✓ All data processing service tests passed!")
        return 0
    else:
        print("✗ Some data processing service tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)