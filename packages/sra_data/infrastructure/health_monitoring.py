"""
Health Checks and Service Reliability System for SRA Data Processing.

This module provides comprehensive health monitoring and reliability including:
- Service health checks and endpoint monitoring
- Database connectivity and performance monitoring
- External service dependency checks
- Circuit breaker patterns for resilience
- Service recovery and auto-healing
- Reliability metrics and reporting
"""

import os
import json
import logging
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import socket
import aiohttp
from datetime import datetime, timedelta
from collections import deque, defaultdict
import psutil
import asyncpg


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ServiceType(Enum):
    """Service types for monitoring."""
    DATABASE = "database"
    API = "api"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    NETWORK = "network"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    service_type: ServiceType
    check_function: Callable
    interval: int = 30  # seconds
    timeout: int = 5    # seconds
    enabled: bool = True
    critical: bool = False  # If critical, affects overall health


@dataclass
class HealthResult:
    """Health check result."""
    check_name: str
    status: HealthStatus
    timestamp: str
    response_time: float
    message: str
    details: Dict[str, Any] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: int = 60      # Seconds before trying half-open
    success_threshold: int = 3      # Successes to close from half-open
    timeout: int = 30               # Operation timeout


class CircuitBreaker:
    """
    Circuit breaker implementation for service resilience.
    """

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker name
            config: Configuration parameters
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_request_time = None
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function through circuit breaker.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        with self._lock:
            current_time = time.time()

            # Check if we should transition from OPEN to HALF_OPEN
            if (self.state == CircuitState.OPEN and
                self.last_failure_time and
                current_time - self.last_failure_time >= self.config.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0

            # Reject calls if circuit is OPEN
            if self.state == CircuitState.OPEN:
                raise Exception(f"Circuit breaker {self.name} is OPEN")

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    def _on_failure(self):
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state information."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time,
                "config": asdict(self.config)
            }


class HealthMonitor:
    """
    Comprehensive health monitoring and service reliability system.

    Provides health checks, circuit breakers, and service monitoring
    for all components of the SRA Data Processing System.
    """

    def __init__(self,
                 check_interval: int = 30,
                 max_history: int = 1000,
                 enable_circuit_breakers: bool = True):
        """
        Initialize health monitor.

        Args:
            check_interval: Default interval between health checks (seconds)
            max_history: Maximum number of health results to keep
            enable_circuit_breakers: Whether to enable circuit breakers
        """
        self.check_interval = check_interval
        self.max_history = max_history
        self.enable_circuit_breakers = enable_circuit_breakers

        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}

        # Health results history
        self.health_history: deque = deque(maxlen=max_history)

        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None

        # Current health status
        self.current_health: Dict[str, HealthResult] = {}
        self._lock = threading.Lock()

        # Register default health checks
        self._register_default_health_checks()

        logger.info("Health monitoring system initialized")

    def _register_default_health_checks(self):
        """Register default health checks."""

        # Database health check
        self.register_health_check(HealthCheck(
            name="database_connectivity",
            service_type=ServiceType.DATABASE,
            check_function=self._check_database_health,
            interval=30,
            timeout=10,
            critical=True
        ))

        # System health check
        self.register_health_check(HealthCheck(
            name="system_resources",
            service_type=ServiceType.FILE_SYSTEM,
            check_function=self._check_system_health,
            interval=60,
            timeout=5,
            critical=False
        ))

        # API health check (self-check)
        self.register_health_check(HealthCheck(
            name="api_endpoints",
            service_type=ServiceType.API,
            check_function=self._check_api_health,
            interval=60,
            timeout=10,
            critical=True
        ))

        # File system health check
        self.register_health_check(HealthCheck(
            name="file_system",
            service_type=ServiceType.FILE_SYSTEM,
            check_function=self._check_file_system_health,
            interval=120,
            timeout=5,
            critical=False
        ))

    def register_health_check(self, health_check: HealthCheck):
        """
        Register a health check.

        Args:
            health_check: Health check to register
        """
        with self._lock:
            self.health_checks[health_check.name] = health_check

        logger.info(f"Registered health check: {health_check.name}")

        # Create circuit breaker if enabled
        if self.enable_circuit_breakers:
            cb_config = CircuitBreakerConfig()
            self.circuit_breakers[health_check.name] = CircuitBreaker(
                health_check.name, cb_config
            )

    def remove_health_check(self, check_name: str):
        """
        Remove a health check.

        Args:
            check_name: Name of health check to remove
        """
        with self._lock:
            if check_name in self.health_checks:
                del self.health_checks[check_name]
                if check_name in self.circuit_breakers:
                    del self.circuit_breakers[check_name]
                if check_name in self.current_health:
                    del self.current_health[check_name]

        logger.info(f"Removed health check: {check_name}")

    async def run_health_check(self, check_name: str) -> HealthResult:
        """
        Run a specific health check.

        Args:
            check_name: Name of health check to run

        Returns:
            Health check result
        """
        if check_name not in self.health_checks:
            return HealthResult(
                check_name=check_name,
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.utcnow().isoformat() + "Z",
                response_time=0.0,
                message="Health check not found",
                error="Check not registered"
            )

        health_check = self.health_checks[check_name]

        if not health_check.enabled:
            return HealthResult(
                check_name=check_name,
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.utcnow().isoformat() + "Z",
                response_time=0.0,
                message="Health check disabled"
            )

        start_time = time.time()
        timestamp = datetime.utcnow().isoformat() + "Z"

        try:
            # Run through circuit breaker if enabled
            if self.enable_circuit_breakers and check_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[check_name]

                try:
                    result = circuit_breaker.call(health_check.check_function)
                except Exception as e:
                    if "Circuit breaker" in str(e):
                        return HealthResult(
                            check_name=check_name,
                            status=HealthStatus.CRITICAL,
                            timestamp=timestamp,
                            response_time=time.time() - start_time,
                            message="Service unavailable - circuit breaker open",
                            error=str(e)
                        )
                    raise
            else:
                result = await health_check.check_function()

            response_time = time.time() - start_time

            # Process result
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", "healthy"))
                message = result.get("message", "Health check passed")
                details = result.get("details", {})
                error = result.get("error")
            else:
                status = HealthStatus.HEALTHY
                message = "Health check passed"
                details = {}
                error = None

            return HealthResult(
                check_name=check_name,
                status=status,
                timestamp=timestamp,
                response_time=response_time,
                message=message,
                details=details,
                error=error
            )

        except asyncio.TimeoutError:
            return HealthResult(
                check_name=check_name,
                status=HealthStatus.CRITICAL,
                timestamp=timestamp,
                response_time=time.time() - start_time,
                message="Health check timed out",
                error="Timeout exceeded"
            )

        except Exception as e:
            return HealthResult(
                check_name=check_name,
                status=HealthStatus.CRITICAL,
                timestamp=timestamp,
                response_time=time.time() - start_time,
                message=f"Health check failed: {str(e)}",
                error=str(e)
            )

    async def run_all_health_checks(self) -> Dict[str, HealthResult]:
        """
        Run all registered health checks.

        Returns:
            Dictionary of health check results
        """
        results = {}
        tasks = []

        for check_name in self.health_checks.keys():
            tasks.append(self.run_health_check(check_name))

        # Run checks concurrently
        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(check_results):
            check_name = list(self.health_checks.keys())[i]

            if isinstance(result, Exception):
                results[check_name] = HealthResult(
                    check_name=check_name,
                    status=HealthStatus.CRITICAL,
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    response_time=0.0,
                    message=f"Health check execution failed: {str(result)}",
                    error=str(result)
                )
            else:
                results[check_name] = result

        # Update current health status
        with self._lock:
            self.current_health.update(results)

        # Add to history
        for result in results.values():
            self.health_history.append(result)

        return results

    def get_overall_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.

        Returns:
            Overall health status summary
        """
        with self._lock:
            if not self.current_health:
                return {
                    "status": HealthStatus.UNKNOWN.value,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "message": "No health checks have been performed",
                    "checks": {}
                }

            # Determine overall status
            critical_failures = []
            warnings = []
            healthy_checks = []

            for check_name, result in self.current_health.items():
                health_check = self.health_checks.get(check_name)

                if result.status == HealthStatus.CRITICAL:
                    critical_failures.append(check_name)
                elif result.status == HealthStatus.WARNING:
                    warnings.append(check_name)
                elif result.status == HealthStatus.HEALTHY:
                    healthy_checks.append(check_name)

            # Determine overall status
            if critical_failures:
                # Check if any critical failures are from critical checks
                critical_check_failures = [
                    check for check in critical_failures
                    if self.health_checks.get(check, HealthCheck("", ServiceType.API, lambda: None)).critical
                ]

                if critical_check_failures:
                    overall_status = HealthStatus.CRITICAL
                    message = f"Critical health check failures: {', '.join(critical_check_failures)}"
                else:
                    overall_status = HealthStatus.WARNING
                    message = f"Health check failures: {', '.join(critical_failures)}"
            elif warnings:
                overall_status = HealthStatus.WARNING
                message = f"Health check warnings: {', '.join(warnings)}"
            else:
                overall_status = HealthStatus.HEALTHY
                message = f"All {len(healthy_checks)} health checks passing"

            return {
                "status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "message": message,
                "summary": {
                    "total_checks": len(self.current_health),
                    "healthy": len(healthy_checks),
                    "warnings": len(warnings),
                    "critical": len(critical_failures)
                },
                "checks": {name: {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time": result.response_time,
                    "last_check": result.timestamp
                } for name, result in self.current_health.items()},
                "circuit_breakers": {
                    name: cb.get_state()
                    for name, cb in self.circuit_breakers.items()
                } if self.enable_circuit_breakers else {}
            }

    def start_monitoring(self):
        """Start background health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return

        self.monitoring_active = True

        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Run health checks
                    asyncio.run(self.run_all_health_checks())

                    # Wait for next check cycle
                    time.sleep(self.check_interval)

                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    time.sleep(self.check_interval)

        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop background health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    # Default health check implementations
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                return {
                    "status": "warning",
                    "message": "DATABASE_URL not configured",
                    "details": {"configuration": "missing"}
                }

            # Test connection
            start_time = time.time()
            conn = await asyncpg.connect(database_url)

            # Test query
            result = await conn.fetchval("SELECT 1")
            connection_time = time.time() - start_time

            await conn.close()

            if connection_time > 2.0:  # Slow connection
                return {
                    "status": "warning",
                    "message": f"Slow database connection: {connection_time:.2f}s",
                    "details": {
                        "connection_time": connection_time,
                        "query_result": result
                    }
                }

            return {
                "status": "healthy",
                "message": "Database connection successful",
                "details": {
                    "connection_time": connection_time,
                    "query_result": result
                }
            }

        except Exception as e:
            return {
                "status": "critical",
                "message": f"Database connection failed: {str(e)}",
                "error": str(e)
            }

    async def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage('/')

            details = {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available,
                "disk_usage": disk.percent,
                "disk_free": disk.free
            }

            # Determine status
            if cpu_percent > 90 or memory.percent > 95 or disk.percent > 95:
                return {
                    "status": "critical",
                    "message": "Critical resource usage detected",
                    "details": details
                }
            elif cpu_percent > 80 or memory.percent > 85 or disk.percent > 85:
                return {
                    "status": "warning",
                    "message": "High resource usage detected",
                    "details": details
                }
            else:
                return {
                    "status": "healthy",
                    "message": "System resources normal",
                    "details": details
                }

        except Exception as e:
            return {
                "status": "critical",
                "message": f"System health check failed: {str(e)}",
                "error": str(e)
            }

    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API endpoint health."""
        try:
            port = os.getenv('PORT', '10000')
            host = os.getenv('HOST', 'localhost')

            url = f"http://{host}:{port}/health"

            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(url, timeout=5) as response:
                    response_time = time.time() - start_time

                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "message": "API endpoint responding",
                            "details": {
                                "response_time": response_time,
                                "status_code": response.status
                            }
                        }
                    else:
                        return {
                            "status": "warning",
                            "message": f"API returned status {response.status}",
                            "details": {
                                "response_time": response_time,
                                "status_code": response.status
                            }
                        }

        except Exception as e:
            return {
                "status": "critical",
                "message": f"API health check failed: {str(e)}",
                "error": str(e)
            }

    async def _check_file_system_health(self) -> Dict[str, Any]:
        """Check file system health."""
        try:
            # Test write access
            test_file = Path("/tmp/health_check_test")

            start_time = time.time()

            # Write test
            with open(test_file, 'w') as f:
                f.write("health check test")

            # Read test
            with open(test_file, 'r') as f:
                content = f.read()

            # Cleanup
            test_file.unlink()

            operation_time = time.time() - start_time

            if operation_time > 1.0:  # Slow file system
                return {
                    "status": "warning",
                    "message": f"Slow file system operations: {operation_time:.2f}s",
                    "details": {"operation_time": operation_time}
                }

            return {
                "status": "healthy",
                "message": "File system operations normal",
                "details": {"operation_time": operation_time}
            }

        except Exception as e:
            return {
                "status": "critical",
                "message": f"File system check failed: {str(e)}",
                "error": str(e)
            }


# Example usage and testing functions
def main():
    """Example usage of HealthMonitor."""
    # Initialize health monitor
    health_monitor = HealthMonitor(
        check_interval=30,
        enable_circuit_breakers=True
    )

    print("Health monitoring system started")

    # Add custom health check
    def custom_check():
        return {
            "status": "healthy",
            "message": "Custom check passed",
            "details": {"timestamp": datetime.utcnow().isoformat()}
        }

    health_monitor.register_health_check(HealthCheck(
        name="custom_service",
        service_type=ServiceType.EXTERNAL_API,
        check_function=custom_check,
        interval=60
    ))

    # Run health checks
    async def run_checks():
        print("\\nRunning health checks...")
        results = await health_monitor.run_all_health_checks()

        for check_name, result in results.items():
            print(f"  {check_name}: {result.status.value} ({result.response_time:.3f}s) - {result.message}")

        # Get overall status
        print("\\nOverall Health Status:")
        overall_status = health_monitor.get_overall_health_status()
        print(json.dumps(overall_status, indent=2))

    # Run the checks
    asyncio.run(run_checks())

    print("\\nHealth monitoring test completed")


if __name__ == "__main__":
    main()