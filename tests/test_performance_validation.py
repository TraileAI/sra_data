"""
Performance Validation and Load Testing Framework

This module provides comprehensive performance testing and validation:
- Load testing and stress testing
- Performance benchmarking
- Resource utilization monitoring
- Scalability validation
- Performance regression detection
"""

import asyncio
import gc
import logging
import memory_profiler
import psutil
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import text

# Import system components
from packages.sra_data.api.skeleton import app
from packages.sra_data.repositories.database_infrastructure import (
    create_database_manager, DatabaseManager
)
from packages.sra_data.repositories.equity_repository import EquityRepository
from packages.sra_data.repositories.fundata_repository import FundataRepository
from packages.sra_data.repositories.performance_optimizer import PerformanceOptimizer
from packages.sra_data.models import EquityProfile, FundataDataRecord

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    operation: str
    duration: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    error_rate: float
    p50_latency: float
    p95_latency: float
    p99_latency: float


@dataclass
class LoadTestResult:
    """Load test result container."""
    test_name: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration: float
    requests_per_second: float
    average_response_time: float
    error_rate: float
    resource_usage: Dict[str, float]
    performance_metrics: PerformanceMetrics


class PerformanceMonitor:
    """Performance monitoring utilities."""

    @staticmethod
    def get_system_metrics() -> Dict[str, float]:
        """Get current system resource metrics."""
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'open_files': len(process.open_files()),
            'threads': process.num_threads(),
            'system_cpu': psutil.cpu_percent(),
            'system_memory': psutil.virtual_memory().percent
        }

    @staticmethod
    @asynccontextmanager
    async def measure_performance(operation: str):
        """Context manager for measuring operation performance."""
        start_metrics = PerformanceMonitor.get_system_metrics()
        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            end_metrics = PerformanceMonitor.get_system_metrics()

            duration = end_time - start_time
            cpu_usage = max(end_metrics['cpu_percent'] - start_metrics['cpu_percent'], 0)
            memory_usage = end_metrics['memory_mb'] - start_metrics['memory_mb']

            logger.info(f"Performance - {operation}: {duration:.3f}s, "
                       f"CPU: +{cpu_usage:.1f}%, Memory: +{memory_usage:.1f}MB")


class PerformanceTestSuite:
    """Comprehensive performance test suite."""

    def __init__(self):
        self.db_manager: Optional[DatabaseManager] = None
        self.repositories: Dict[str, Any] = {}
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.test_data: Dict[str, Any] = {}
        self.results: List[LoadTestResult] = []
        self.baseline_metrics: Dict[str, float] = {}

    async def setup(self):
        """Setup performance test environment."""
        async with PerformanceMonitor.measure_performance("test_setup"):
            # Initialize database manager
            self.db_manager = create_database_manager()
            await self.db_manager.initialize()

            # Initialize repositories
            session_factory = self.db_manager.get_session_factory()
            self.repositories = {
                'equity': EquityRepository(session_factory),
                'fundata': FundataRepository(session_factory)
            }

            # Initialize performance optimizer
            self.performance_optimizer = PerformanceOptimizer(session_factory)

            # Setup test data
            await self._setup_performance_test_data()

            # Establish baseline metrics
            await self._establish_baseline_metrics()

    async def teardown(self):
        """Teardown performance test environment."""
        async with PerformanceMonitor.measure_performance("test_teardown"):
            # Cleanup test data
            await self._cleanup_test_data()

            # Close database connections
            if self.db_manager:
                await self.db_manager.close()

    async def _setup_performance_test_data(self):
        """Setup comprehensive test data for performance testing."""
        self.test_data = {
            'equity_profiles': [
                EquityProfile(
                    symbol=f'PERF{i:04d}',
                    name=f'Performance Test Company {i}',
                    market_cap=1000000 * (i + 1),
                    sector=f'Sector{i % 10}',
                    industry=f'Industry{i % 20}',
                    country='US',
                    exchange='TEST',
                    price=100.0 + (i % 100),
                    pe_ratio=15.0 + (i % 30),
                    beta=0.5 + (i % 20) * 0.1
                )
                for i in range(1000)  # Large dataset for performance testing
            ],
            'created_ids': []
        }

    async def _establish_baseline_metrics(self):
        """Establish baseline performance metrics."""
        equity_repo = self.repositories['equity']

        # Test single operation baseline
        start_time = time.time()
        start_metrics = PerformanceMonitor.get_system_metrics()

        test_profile = self.test_data['equity_profiles'][0]
        created = await equity_repo.create(test_profile)
        retrieved = await equity_repo.get_by_id(created.id)
        await equity_repo.delete(created.id)

        end_time = time.time()
        end_metrics = PerformanceMonitor.get_system_metrics()

        self.baseline_metrics = {
            'single_crud_time': end_time - start_time,
            'baseline_cpu': start_metrics['cpu_percent'],
            'baseline_memory': start_metrics['memory_mb']
        }

    async def _cleanup_test_data(self):
        """Cleanup performance test data."""
        if self.test_data.get('created_ids'):
            equity_repo = self.repositories['equity']
            for record_id in self.test_data['created_ids']:
                try:
                    await equity_repo.delete(record_id)
                except Exception as e:
                    logger.warning(f"Cleanup error for ID {record_id}: {e}")


@pytest_asyncio.fixture
async def performance_suite():
    """Provide performance test suite."""
    suite = PerformanceTestSuite()
    await suite.setup()
    try:
        yield suite
    finally:
        await suite.teardown()


class TestDatabasePerformance:
    """Test database layer performance."""

    @pytest.mark.asyncio
    async def test_repository_crud_performance(self, performance_suite: PerformanceTestSuite):
        """Test repository CRUD operation performance."""
        equity_repo = performance_suite.repositories['equity']
        test_profiles = performance_suite.test_data['equity_profiles'][:100]

        # Measure create performance
        create_times = []
        async with PerformanceMonitor.measure_performance("bulk_create"):
            for profile in test_profiles:
                start_time = time.time()
                created = await equity_repo.create(profile)
                end_time = time.time()

                create_times.append(end_time - start_time)
                performance_suite.test_data['created_ids'].append(created.id)

        # Measure read performance
        read_times = []
        async with PerformanceMonitor.measure_performance("bulk_read"):
            for record_id in performance_suite.test_data['created_ids']:
                start_time = time.time()
                retrieved = await equity_repo.get_by_id(record_id)
                end_time = time.time()

                read_times.append(end_time - start_time)
                assert retrieved is not None

        # Measure search performance
        search_times = []
        search_queries = ['Performance Test', 'Company', 'Sector1', 'Industry5']

        async with PerformanceMonitor.measure_performance("search_operations"):
            for query in search_queries:
                start_time = time.time()
                results = await equity_repo.search(query, limit=50)
                end_time = time.time()

                search_times.append(end_time - start_time)
                assert len(results) >= 0

        # Performance assertions
        avg_create_time = statistics.mean(create_times)
        avg_read_time = statistics.mean(read_times)
        avg_search_time = statistics.mean(search_times)

        assert avg_create_time < 0.1   # Average create under 100ms
        assert avg_read_time < 0.05    # Average read under 50ms
        assert avg_search_time < 0.5   # Average search under 500ms

        # Throughput calculations
        create_throughput = 100 / sum(create_times)
        read_throughput = 100 / sum(read_times)

        logger.info(f"CRUD Performance - Create: {create_throughput:.1f} ops/s, "
                   f"Read: {read_throughput:.1f} ops/s")

        assert create_throughput > 50  # Minimum 50 creates/second
        assert read_throughput > 100   # Minimum 100 reads/second

    @pytest.mark.asyncio
    async def test_concurrent_database_performance(self, performance_suite: PerformanceTestSuite):
        """Test concurrent database operation performance."""
        equity_repo = performance_suite.repositories['equity']

        async def concurrent_create_operation(profile: EquityProfile) -> Tuple[float, bool]:
            start_time = time.time()
            try:
                created = await equity_repo.create(profile)
                performance_suite.test_data['created_ids'].append(created.id)
                end_time = time.time()
                return end_time - start_time, True
            except Exception as e:
                end_time = time.time()
                logger.warning(f"Concurrent create failed: {e}")
                return end_time - start_time, False

        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 50]
        performance_results = {}

        for concurrency in concurrency_levels:
            test_profiles = performance_suite.test_data['equity_profiles'][
                len(performance_suite.test_data['created_ids']):
                len(performance_suite.test_data['created_ids']) + concurrency
            ]

            start_time = time.time()
            start_metrics = PerformanceMonitor.get_system_metrics()

            # Execute concurrent operations
            tasks = [concurrent_create_operation(profile) for profile in test_profiles]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()
            end_metrics = PerformanceMonitor.get_system_metrics()

            # Analyze results
            successful_ops = [r for r in results if isinstance(r, tuple) and r[1]]
            failed_ops = [r for r in results if not (isinstance(r, tuple) and r[1])]

            total_duration = end_time - start_time
            throughput = len(successful_ops) / total_duration
            error_rate = len(failed_ops) / len(results)

            performance_results[concurrency] = {
                'throughput': throughput,
                'total_duration': total_duration,
                'error_rate': error_rate,
                'cpu_usage': end_metrics['cpu_percent'] - start_metrics['cpu_percent'],
                'memory_usage': end_metrics['memory_mb'] - start_metrics['memory_mb']
            }

        # Performance validation
        for concurrency, results in performance_results.items():
            logger.info(f"Concurrency {concurrency}: {results['throughput']:.1f} ops/s, "
                       f"Error rate: {results['error_rate']:.2%}")

            # Validate performance doesn't degrade excessively
            assert results['error_rate'] < 0.05  # Less than 5% errors
            assert results['throughput'] > 10    # Minimum throughput

        # Validate scalability
        single_thread_throughput = performance_results[1]['throughput']
        max_throughput = max(r['throughput'] for r in performance_results.values())

        # Should achieve some level of concurrent improvement
        assert max_throughput > single_thread_throughput * 1.2

    @pytest.mark.asyncio
    async def test_database_performance_optimization(self, performance_suite: PerformanceTestSuite):
        """Test database performance optimization features."""
        optimizer = performance_suite.performance_optimizer
        equity_repo = performance_suite.repositories['equity']

        # Create test data for optimization analysis
        test_profiles = performance_suite.test_data['equity_profiles'][:50]
        for profile in test_profiles:
            created = await equity_repo.create(profile)
            performance_suite.test_data['created_ids'].append(created.id)

        # Test query analysis
        async with PerformanceMonitor.measure_performance("query_analysis"):
            slow_queries = await optimizer.analyze_slow_queries(limit=10)
            assert isinstance(slow_queries, list)

        # Test index recommendations
        async with PerformanceMonitor.measure_performance("index_recommendations"):
            recommendations = await optimizer.get_index_recommendations()
            assert isinstance(recommendations, dict)

        # Test performance health check
        async with PerformanceMonitor.measure_performance("health_check"):
            health_score = await optimizer.get_performance_health_score()
            assert isinstance(health_score, (int, float))
            assert 0 <= health_score <= 100

        # Test connection metrics
        async with PerformanceMonitor.measure_performance("connection_metrics"):
            connection_metrics = await optimizer.get_connection_metrics()
            assert isinstance(connection_metrics, dict)
            assert 'active_connections' in connection_metrics


class TestAPIPerformance:
    """Test API layer performance."""

    @pytest.mark.asyncio
    async def test_api_response_performance(self):
        """Test API endpoint response performance."""
        client = TestClient(app)

        # Test individual endpoint performance
        endpoints = ["/", "/health", "/status"]
        response_times = {endpoint: [] for endpoint in endpoints}

        # Measure baseline performance
        for endpoint in endpoints:
            for _ in range(10):
                start_time = time.time()
                response = client.get(endpoint)
                end_time = time.time()

                response_times[endpoint].append(end_time - start_time)
                assert response.status_code == 200

        # Performance assertions
        for endpoint, times in response_times.items():
            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile

            logger.info(f"API {endpoint} - Avg: {avg_time:.3f}s, P95: {p95_time:.3f}s")

            assert avg_time < 0.1   # Average under 100ms
            assert p95_time < 0.2   # 95th percentile under 200ms

    @pytest.mark.asyncio
    async def test_api_concurrent_load(self):
        """Test API performance under concurrent load."""
        async def make_api_request(endpoint: str) -> Tuple[int, float]:
            async with AsyncClient(app=app, base_url="http://test") as client:
                start_time = time.time()
                response = await client.get(endpoint)
                end_time = time.time()
                return response.status_code, end_time - start_time

        # Test concurrent requests to different endpoints
        test_scenarios = [
            {"endpoint": "/health", "concurrent_requests": 50},
            {"endpoint": "/status", "concurrent_requests": 25},
            {"endpoint": "/", "concurrent_requests": 25}
        ]

        for scenario in test_scenarios:
            endpoint = scenario["endpoint"]
            concurrent_requests = scenario["concurrent_requests"]

            start_time = time.time()
            start_metrics = PerformanceMonitor.get_system_metrics()

            # Execute concurrent requests
            tasks = [make_api_request(endpoint) for _ in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            end_metrics = PerformanceMonitor.get_system_metrics()

            # Analyze results
            successful_requests = [r for r in results if r[0] == 200]
            response_times = [r[1] for r in successful_requests]

            total_duration = end_time - start_time
            requests_per_second = len(successful_requests) / total_duration
            avg_response_time = statistics.mean(response_times)
            error_rate = (len(results) - len(successful_requests)) / len(results)

            logger.info(f"API Load Test {endpoint}: {requests_per_second:.1f} RPS, "
                       f"Avg response: {avg_response_time:.3f}s, Error rate: {error_rate:.2%}")

            # Performance assertions
            assert requests_per_second > 100    # Minimum 100 RPS
            assert avg_response_time < 0.1      # Average response under 100ms
            assert error_rate < 0.01           # Less than 1% error rate

    @pytest.mark.asyncio
    async def test_api_stress_testing(self):
        """Test API under stress conditions."""
        async def stress_test_request() -> Dict[str, Any]:
            async with AsyncClient(app=app, base_url="http://test") as client:
                start_time = time.time()
                try:
                    response = await client.get("/status")
                    end_time = time.time()
                    return {
                        'success': True,
                        'status_code': response.status_code,
                        'response_time': end_time - start_time
                    }
                except Exception as e:
                    end_time = time.time()
                    return {
                        'success': False,
                        'error': str(e),
                        'response_time': end_time - start_time
                    }

        # Stress test parameters
        stress_levels = [100, 200, 500]  # Concurrent requests
        stress_results = {}

        for stress_level in stress_levels:
            logger.info(f"Starting stress test with {stress_level} concurrent requests")

            start_time = time.time()
            start_metrics = PerformanceMonitor.get_system_metrics()

            # Execute stress test
            tasks = [stress_test_request() for _ in range(stress_level)]
            results = await asyncio.gather(*tasks)

            end_time = time.time()
            end_metrics = PerformanceMonitor.get_system_metrics()

            # Analyze stress test results
            successful_requests = [r for r in results if r['success']]
            failed_requests = [r for r in results if not r['success']]

            total_duration = end_time - start_time
            requests_per_second = len(successful_requests) / total_duration
            error_rate = len(failed_requests) / len(results)

            response_times = [r['response_time'] for r in successful_requests]
            avg_response_time = statistics.mean(response_times) if response_times else 0
            p95_response_time = (statistics.quantiles(response_times, n=20)[18]
                               if len(response_times) >= 20 else avg_response_time)

            stress_results[stress_level] = {
                'requests_per_second': requests_per_second,
                'error_rate': error_rate,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'cpu_usage': end_metrics['cpu_percent'] - start_metrics['cpu_percent'],
                'memory_usage': end_metrics['memory_mb'] - start_metrics['memory_mb']
            }

            logger.info(f"Stress level {stress_level} results: "
                       f"RPS: {requests_per_second:.1f}, "
                       f"Error rate: {error_rate:.2%}, "
                       f"Avg response: {avg_response_time:.3f}s")

        # Stress test validation
        for stress_level, results in stress_results.items():
            # Even under stress, should maintain reasonable performance
            if stress_level <= 200:
                assert results['error_rate'] < 0.05    # Less than 5% errors
                assert results['avg_response_time'] < 1.0  # Under 1s average
            else:
                # High stress - allow higher error rates but should not crash
                assert results['error_rate'] < 0.2     # Less than 20% errors


class TestResourceUtilizationPerformance:
    """Test resource utilization and efficiency."""

    @pytest.mark.asyncio
    async def test_memory_usage_performance(self, performance_suite: PerformanceTestSuite):
        """Test memory usage patterns and efficiency."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        equity_repo = performance_suite.repositories['equity']

        # Test memory usage during bulk operations
        memory_snapshots = []

        # Create data and monitor memory
        for i in range(0, 100, 10):
            profiles = performance_suite.test_data['equity_profiles'][i:i+10]

            for profile in profiles:
                created = await equity_repo.create(profile)
                performance_suite.test_data['created_ids'].append(created.id)

            # Force garbage collection and measure memory
            gc.collect()
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_snapshots.append(current_memory - initial_memory)

        # Test memory cleanup after operations
        pre_cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Cleanup some data
        for i in range(0, min(50, len(performance_suite.test_data['created_ids']))):
            record_id = performance_suite.test_data['created_ids'][i]
            await equity_repo.delete(record_id)

        gc.collect()
        post_cleanup_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Memory usage assertions
        max_memory_growth = max(memory_snapshots)
        memory_freed = pre_cleanup_memory - post_cleanup_memory

        logger.info(f"Memory usage - Max growth: {max_memory_growth:.1f}MB, "
                   f"Freed after cleanup: {memory_freed:.1f}MB")

        # Memory should not grow excessively
        assert max_memory_growth < 100  # Less than 100MB growth for 100 records

        # Memory should be freed after cleanup (allowing some overhead)
        assert memory_freed > -10  # Should not increase significantly

    @pytest.mark.asyncio
    async def test_cpu_usage_efficiency(self, performance_suite: PerformanceTestSuite):
        """Test CPU usage efficiency."""
        process = psutil.Process()

        # Monitor CPU usage during operations
        initial_cpu_times = process.cpu_times()

        equity_repo = performance_suite.repositories['equity']

        # Perform CPU-intensive operations
        test_profiles = performance_suite.test_data['equity_profiles'][:50]

        start_time = time.time()
        for profile in test_profiles:
            created = await equity_repo.create(profile)
            retrieved = await equity_repo.get_by_id(created.id)
            performance_suite.test_data['created_ids'].append(created.id)
            assert retrieved is not None
        end_time = time.time()

        final_cpu_times = process.cpu_times()

        # Calculate CPU efficiency
        total_duration = end_time - start_time
        cpu_time_used = (final_cpu_times.user + final_cpu_times.system) - \
                       (initial_cpu_times.user + initial_cpu_times.system)

        cpu_efficiency = (cpu_time_used / total_duration) * 100  # CPU utilization %
        operations_per_cpu_second = (len(test_profiles) * 2) / cpu_time_used  # create + read

        logger.info(f"CPU Efficiency - Utilization: {cpu_efficiency:.1f}%, "
                   f"Ops per CPU second: {operations_per_cpu_second:.1f}")

        # CPU efficiency assertions
        assert cpu_efficiency < 50        # Should not use more than 50% CPU
        assert operations_per_cpu_second > 100  # At least 100 ops per CPU second


class TestPerformanceRegression:
    """Test for performance regressions."""

    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, performance_suite: PerformanceTestSuite):
        """Test for performance regressions against baseline."""
        equity_repo = performance_suite.repositories['equity']

        # Baseline single operation performance
        baseline_time = performance_suite.baseline_metrics['single_crud_time']

        # Current performance test
        test_profile = performance_suite.test_data['equity_profiles'][0]

        start_time = time.time()
        created = await equity_repo.create(test_profile)
        retrieved = await equity_repo.get_by_id(created.id)
        await equity_repo.delete(created.id)
        end_time = time.time()

        current_time = end_time - start_time

        # Performance regression analysis
        performance_ratio = current_time / baseline_time
        regression_threshold = 1.5  # Allow 50% performance degradation

        logger.info(f"Performance Regression Test - "
                   f"Baseline: {baseline_time:.3f}s, "
                   f"Current: {current_time:.3f}s, "
                   f"Ratio: {performance_ratio:.2f}x")

        # Regression assertion
        assert performance_ratio < regression_threshold, \
            f"Performance regression detected: {performance_ratio:.2f}x slower than baseline"

        # Additional regression tests for batch operations
        batch_profiles = performance_suite.test_data['equity_profiles'][1:11]

        batch_start = time.time()
        created_ids = []
        for profile in batch_profiles:
            created = await equity_repo.create(profile)
            created_ids.append(created.id)

        for record_id in created_ids:
            retrieved = await equity_repo.get_by_id(record_id)
            assert retrieved is not None

        for record_id in created_ids:
            await equity_repo.delete(record_id)

        batch_end = time.time()

        batch_time = batch_end - batch_start
        expected_batch_time = baseline_time * len(batch_profiles) * 2  # Allow linear scaling

        batch_efficiency = expected_batch_time / batch_time

        logger.info(f"Batch Performance - Time: {batch_time:.3f}s, "
                   f"Efficiency: {batch_efficiency:.2f}x")

        # Batch should be at least as efficient as individual operations
        assert batch_efficiency >= 0.8  # Allow 20% overhead for batch operations


# Performance test summary and reporting
class TestPerformanceSummary:
    """Performance test summary and benchmarking."""

    @pytest.mark.asyncio
    async def test_comprehensive_performance_benchmark(self, performance_suite: PerformanceTestSuite):
        """Comprehensive performance benchmark and reporting."""
        benchmark_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,
                'python_version': psutil.__version__
            },
            'baseline_metrics': performance_suite.baseline_metrics,
            'performance_tests': {}
        }

        # Repository performance benchmark
        equity_repo = performance_suite.repositories['equity']
        repo_start_time = time.time()

        # Create benchmark data
        benchmark_profiles = performance_suite.test_data['equity_profiles'][:100]
        created_ids = []

        for profile in benchmark_profiles:
            created = await equity_repo.create(profile)
            created_ids.append(created.id)

        repo_create_time = time.time() - repo_start_time

        # Read benchmark
        read_start_time = time.time()
        for record_id in created_ids:
            retrieved = await equity_repo.get_by_id(record_id)
            assert retrieved is not None
        read_time = time.time() - read_start_time

        # Search benchmark
        search_start_time = time.time()
        search_results = await equity_repo.search("Performance", limit=50)
        search_time = time.time() - search_start_time

        # Cleanup benchmark data
        cleanup_start_time = time.time()
        for record_id in created_ids:
            await equity_repo.delete(record_id)
        cleanup_time = time.time() - cleanup_start_time

        # API benchmark
        client = TestClient(app)
        api_start_time = time.time()

        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200

        api_time = time.time() - api_start_time

        # Compile benchmark results
        benchmark_results['performance_tests'] = {
            'repository_create': {
                'operations': 100,
                'total_time': repo_create_time,
                'ops_per_second': 100 / repo_create_time,
                'avg_time_per_op': repo_create_time / 100
            },
            'repository_read': {
                'operations': 100,
                'total_time': read_time,
                'ops_per_second': 100 / read_time,
                'avg_time_per_op': read_time / 100
            },
            'repository_search': {
                'operations': 1,
                'total_time': search_time,
                'results_found': len(search_results),
                'avg_time_per_op': search_time
            },
            'repository_delete': {
                'operations': 100,
                'total_time': cleanup_time,
                'ops_per_second': 100 / cleanup_time,
                'avg_time_per_op': cleanup_time / 100
            },
            'api_health_check': {
                'operations': 100,
                'total_time': api_time,
                'ops_per_second': 100 / api_time,
                'avg_time_per_op': api_time / 100
            }
        }

        # Log comprehensive benchmark report
        logger.info("=== Performance Benchmark Report ===")
        for test_name, results in benchmark_results['performance_tests'].items():
            logger.info(f"{test_name}: {results['ops_per_second']:.1f} ops/s "
                       f"({results['avg_time_per_op']*1000:.1f}ms avg)")

        # Performance assertions
        assert benchmark_results['performance_tests']['repository_create']['ops_per_second'] > 50
        assert benchmark_results['performance_tests']['repository_read']['ops_per_second'] > 100
        assert benchmark_results['performance_tests']['api_health_check']['ops_per_second'] > 200
        assert benchmark_results['performance_tests']['repository_search']['avg_time_per_op'] < 0.5

        # Store benchmark for future comparison
        performance_suite.benchmark_results = benchmark_results


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "--asyncio-mode=auto"
    ])