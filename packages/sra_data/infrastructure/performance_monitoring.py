"""
Performance Monitoring and Alerting System for SRA Data Processing.

This module provides comprehensive performance monitoring including:
- Real-time performance metrics collection
- Application performance monitoring (APM)
- Database query performance tracking
- API response time monitoring
- Resource utilization tracking
- Performance alerting and notifications
- Performance trend analysis and reporting
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
import statistics
from datetime import datetime, timedelta
from collections import deque, defaultdict
import psutil
import asyncpg
import functools


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Performance metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    PERCENTAGE = "percentage"


class AlertThreshold(Enum):
    """Alert threshold types."""
    ABOVE = "above"
    BELOW = "below"
    BETWEEN = "between"
    OUTSIDE = "outside"


class PerformanceCategory(Enum):
    """Performance monitoring categories."""
    API = "api"
    DATABASE = "database"
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    CACHE = "cache"


@dataclass
class PerformanceMetric:
    """Performance metric definition."""
    name: str
    value: float
    metric_type: MetricType
    category: PerformanceCategory
    timestamp: str
    tags: Dict[str, str] = None
    unit: str = ""
    description: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class PerformanceThreshold:
    """Performance threshold for alerting."""
    metric_name: str
    threshold_type: AlertThreshold
    warning_value: float
    critical_value: float
    duration: int = 60  # seconds
    enabled: bool = True


@dataclass
class PerformanceAlert:
    """Performance alert."""
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # warning, critical
    timestamp: str
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class PerformanceCollector:
    """
    Collects and aggregates performance metrics.
    """

    def __init__(self, max_metrics: int = 50000):
        """
        Initialize performance collector.

        Args:
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.metric_aggregations: Dict[str, Any] = defaultdict(dict)
        self._lock = threading.Lock()

    def record_metric(self, name: str, value: float, metric_type: MetricType,
                     category: PerformanceCategory, tags: Optional[Dict[str, str]] = None,
                     unit: str = "", description: str = ""):
        """
        Record a performance metric.

        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            category: Performance category
            tags: Optional tags
            unit: Unit of measurement
            description: Metric description
        """
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                metric_type=metric_type,
                category=category,
                timestamp=datetime.utcnow().isoformat() + "Z",
                tags=tags or {},
                unit=unit,
                description=description
            )

            self.metrics.append(metric)
            self._update_aggregations(metric)

    def _update_aggregations(self, metric: PerformanceMetric):
        """Update metric aggregations."""
        key = f"{metric.name}_{metric.category.value}"

        if key not in self.metric_aggregations:
            self.metric_aggregations[key] = {
                "count": 0,
                "sum": 0,
                "min": float('inf'),
                "max": float('-inf'),
                "mean": 0,
                "last_value": 0,
                "last_timestamp": None,
                "values": deque(maxlen=1000)  # Keep last 1000 values for calculations
            }

        agg = self.metric_aggregations[key]
        agg["count"] += 1
        agg["sum"] += metric.value
        agg["min"] = min(agg["min"], metric.value)
        agg["max"] = max(agg["max"], metric.value)
        agg["mean"] = agg["sum"] / agg["count"]
        agg["last_value"] = metric.value
        agg["last_timestamp"] = metric.timestamp
        agg["values"].append(metric.value)

        # Calculate additional statistics if we have enough values
        if len(agg["values"]) >= 2:
            values = list(agg["values"])
            agg["median"] = statistics.median(values)
            if len(values) >= 10:
                agg["p95"] = statistics.quantiles(values, n=20)[18]  # 95th percentile
                agg["p99"] = statistics.quantiles(values, n=100)[98]  # 99th percentile
                agg["std_dev"] = statistics.stdev(values)

    def get_metrics(self, name_filter: Optional[str] = None,
                   category_filter: Optional[PerformanceCategory] = None,
                   since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics with optional filtering."""
        with self._lock:
            filtered = []

            for metric in self.metrics:
                # Name filter
                if name_filter and name_filter not in metric.name:
                    continue

                # Category filter
                if category_filter and metric.category != category_filter:
                    continue

                # Time filter
                if since:
                    metric_time = datetime.fromisoformat(metric.timestamp.replace('Z', '+00:00'))
                    if metric_time < since:
                        continue

                filtered.append(metric)

            return filtered

    def get_aggregations(self) -> Dict[str, Any]:
        """Get metric aggregations."""
        with self._lock:
            return dict(self.metric_aggregations)


def performance_timer(category: PerformanceCategory = PerformanceCategory.APPLICATION,
                     tags: Optional[Dict[str, str]] = None):
    """
    Decorator for timing function performance.

    Args:
        category: Performance category
        tags: Optional tags for the metric
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Get collector from global context (you'd need to set this)
                collector = getattr(performance_timer, 'collector', None)
                if collector:
                    collector.record_metric(
                        name=f"{func.__module__}.{func.__name__}.duration",
                        value=execution_time,
                        metric_type=MetricType.TIMER,
                        category=category,
                        tags=tags or {},
                        unit="seconds",
                        description=f"Execution time for {func.__name__}"
                    )

                return result
            except Exception as e:
                execution_time = time.time() - start_time

                # Record error metric
                collector = getattr(performance_timer, 'collector', None)
                if collector:
                    collector.record_metric(
                        name=f"{func.__module__}.{func.__name__}.errors",
                        value=1,
                        metric_type=MetricType.COUNTER,
                        category=category,
                        tags={**(tags or {}), "error_type": type(e).__name__},
                        description=f"Error count for {func.__name__}"
                    )

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Record metric
                collector = getattr(performance_timer, 'collector', None)
                if collector:
                    collector.record_metric(
                        name=f"{func.__module__}.{func.__name__}.duration",
                        value=execution_time,
                        metric_type=MetricType.TIMER,
                        category=category,
                        tags=tags or {},
                        unit="seconds",
                        description=f"Execution time for {func.__name__}"
                    )

                return result
            except Exception as e:
                execution_time = time.time() - start_time

                # Record error metric
                collector = getattr(performance_timer, 'collector', None)
                if collector:
                    collector.record_metric(
                        name=f"{func.__module__}.{func.__name__}.errors",
                        value=1,
                        metric_type=MetricType.COUNTER,
                        category=category,
                        tags={**(tags or {}), "error_type": type(e).__name__},
                        description=f"Error count for {func.__name__}"
                    )

                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class PerformanceMonitor:
    """
    Comprehensive performance monitoring and alerting system.

    Provides real-time performance monitoring, alerting, and analysis
    for all components of the SRA Data Processing System.
    """

    def __init__(self,
                 collection_interval: int = 15,
                 alert_check_interval: int = 30,
                 enable_system_monitoring: bool = True):
        """
        Initialize performance monitor.

        Args:
            collection_interval: Interval for automatic metric collection (seconds)
            alert_check_interval: Interval for checking alert thresholds (seconds)
            enable_system_monitoring: Whether to collect system metrics automatically
        """
        self.collection_interval = collection_interval
        self.alert_check_interval = alert_check_interval
        self.enable_system_monitoring = enable_system_monitoring

        # Performance collector
        self.collector = PerformanceCollector()

        # Set global collector for decorator
        performance_timer.collector = self.collector

        # Performance thresholds
        self.thresholds: Dict[str, PerformanceThreshold] = {}

        # Active alerts
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []

        # Monitoring control
        self.monitoring_active = False
        self.monitoring_threads = []

        # Lock for thread safety
        self._lock = threading.Lock()

        # Setup default thresholds
        self._setup_default_thresholds()

        logger.info("Performance monitoring system initialized")

    def _setup_default_thresholds(self):
        """Setup default performance thresholds."""
        default_thresholds = [
            # API performance thresholds
            PerformanceThreshold(
                "api.response_time", AlertThreshold.ABOVE, 1.0, 3.0
            ),
            PerformanceThreshold(
                "api.error_rate", AlertThreshold.ABOVE, 0.05, 0.1  # 5% warning, 10% critical
            ),

            # Database performance thresholds
            PerformanceThreshold(
                "database.query_duration", AlertThreshold.ABOVE, 0.5, 2.0
            ),
            PerformanceThreshold(
                "database.connection_time", AlertThreshold.ABOVE, 0.1, 0.5
            ),

            # System resource thresholds
            PerformanceThreshold(
                "system.cpu_usage", AlertThreshold.ABOVE, 80.0, 95.0
            ),
            PerformanceThreshold(
                "system.memory_usage", AlertThreshold.ABOVE, 85.0, 95.0
            ),
            PerformanceThreshold(
                "system.disk_usage", AlertThreshold.ABOVE, 85.0, 95.0
            ),

            # Application thresholds
            PerformanceThreshold(
                "application.processing_time", AlertThreshold.ABOVE, 5.0, 15.0
            )
        ]

        for threshold in default_thresholds:
            self.thresholds[threshold.metric_name] = threshold

    def add_threshold(self, threshold: PerformanceThreshold):
        """
        Add a performance threshold.

        Args:
            threshold: Performance threshold to add
        """
        with self._lock:
            self.thresholds[threshold.metric_name] = threshold

        logger.info(f"Added performance threshold: {threshold.metric_name}")

    def remove_threshold(self, metric_name: str):
        """
        Remove a performance threshold.

        Args:
            metric_name: Name of metric threshold to remove
        """
        with self._lock:
            if metric_name in self.thresholds:
                del self.thresholds[metric_name]

        logger.info(f"Removed performance threshold: {metric_name}")

    def record_api_request(self, method: str, endpoint: str, status_code: int,
                          response_time: float, user_id: Optional[str] = None):
        """
        Record API request performance.

        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: Response status code
            response_time: Response time in seconds
            user_id: Optional user ID
        """
        tags = {
            "method": method,
            "endpoint": endpoint.split('?')[0],  # Remove query params
            "status_code": str(status_code)
        }

        if user_id:
            tags["user_id"] = user_id

        # Record response time
        self.collector.record_metric(
            name="api.response_time",
            value=response_time,
            metric_type=MetricType.TIMER,
            category=PerformanceCategory.API,
            tags=tags,
            unit="seconds",
            description="API response time"
        )

        # Record request count
        self.collector.record_metric(
            name="api.requests",
            value=1,
            metric_type=MetricType.COUNTER,
            category=PerformanceCategory.API,
            tags=tags,
            description="API request count"
        )

        # Record error if applicable
        if status_code >= 400:
            self.collector.record_metric(
                name="api.errors",
                value=1,
                metric_type=MetricType.COUNTER,
                category=PerformanceCategory.API,
                tags=tags,
                description="API error count"
            )

    def record_database_operation(self, operation: str, table: str,
                                duration: float, rows_affected: int = 0,
                                connection_time: Optional[float] = None):
        """
        Record database operation performance.

        Args:
            operation: Database operation (SELECT, INSERT, UPDATE, DELETE)
            table: Table name
            duration: Operation duration in seconds
            rows_affected: Number of rows affected
            connection_time: Optional connection time
        """
        tags = {
            "operation": operation,
            "table": table
        }

        # Record query duration
        self.collector.record_metric(
            name="database.query_duration",
            value=duration,
            metric_type=MetricType.TIMER,
            category=PerformanceCategory.DATABASE,
            tags=tags,
            unit="seconds",
            description="Database query duration"
        )

        # Record query count
        self.collector.record_metric(
            name="database.queries",
            value=1,
            metric_type=MetricType.COUNTER,
            category=PerformanceCategory.DATABASE,
            tags=tags,
            description="Database query count"
        )

        # Record rows affected
        if rows_affected > 0:
            self.collector.record_metric(
                name="database.rows_affected",
                value=rows_affected,
                metric_type=MetricType.GAUGE,
                category=PerformanceCategory.DATABASE,
                tags=tags,
                description="Database rows affected"
            )

        # Record connection time if provided
        if connection_time is not None:
            self.collector.record_metric(
                name="database.connection_time",
                value=connection_time,
                metric_type=MetricType.TIMER,
                category=PerformanceCategory.DATABASE,
                tags=tags,
                unit="seconds",
                description="Database connection time"
            )

    def collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.collector.record_metric(
                "system.cpu_usage", cpu_percent, MetricType.PERCENTAGE,
                PerformanceCategory.SYSTEM, unit="%", description="CPU usage percentage"
            )

            # Memory metrics
            memory = psutil.virtual_memory()
            self.collector.record_metric(
                "system.memory_usage", memory.percent, MetricType.PERCENTAGE,
                PerformanceCategory.SYSTEM, unit="%", description="Memory usage percentage"
            )
            self.collector.record_metric(
                "system.memory_available", memory.available, MetricType.GAUGE,
                PerformanceCategory.SYSTEM, unit="bytes", description="Available memory"
            )

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.collector.record_metric(
                "system.disk_usage", disk.percent, MetricType.PERCENTAGE,
                PerformanceCategory.SYSTEM, unit="%", description="Disk usage percentage"
            )

            # Network I/O (if available)
            try:
                net_io = psutil.net_io_counters()
                self.collector.record_metric(
                    "system.network_bytes_sent", net_io.bytes_sent, MetricType.COUNTER,
                    PerformanceCategory.NETWORK, unit="bytes", description="Network bytes sent"
                )
                self.collector.record_metric(
                    "system.network_bytes_recv", net_io.bytes_recv, MetricType.COUNTER,
                    PerformanceCategory.NETWORK, unit="bytes", description="Network bytes received"
                )
            except:
                pass  # Network metrics not available

            # Process metrics
            process = psutil.Process()
            self.collector.record_metric(
                "process.cpu_percent", process.cpu_percent(), MetricType.PERCENTAGE,
                PerformanceCategory.APPLICATION, unit="%", description="Process CPU usage"
            )
            self.collector.record_metric(
                "process.memory_percent", process.memory_percent(), MetricType.PERCENTAGE,
                PerformanceCategory.APPLICATION, unit="%", description="Process memory usage"
            )

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def check_thresholds(self):
        """Check performance thresholds and trigger alerts."""
        aggregations = self.collector.get_aggregations()

        with self._lock:
            for metric_name, threshold in self.thresholds.items():
                if not threshold.enabled:
                    continue

                # Find matching aggregations
                matching_aggs = [
                    (key, agg) for key, agg in aggregations.items()
                    if metric_name in key
                ]

                for agg_key, agg in matching_aggs:
                    current_value = agg["last_value"]

                    # Check warning threshold
                    warning_triggered = self._check_threshold_condition(
                        current_value, threshold.warning_value, threshold.threshold_type
                    )

                    # Check critical threshold
                    critical_triggered = self._check_threshold_condition(
                        current_value, threshold.critical_value, threshold.threshold_type
                    )

                    # Create alert if threshold exceeded
                    if critical_triggered:
                        self._create_alert(
                            metric_name, current_value, threshold.critical_value,
                            "critical", agg_key, threshold
                        )
                    elif warning_triggered:
                        self._create_alert(
                            metric_name, current_value, threshold.warning_value,
                            "warning", agg_key, threshold
                        )

    def _check_threshold_condition(self, value: float, threshold: float,
                                 threshold_type: AlertThreshold) -> bool:
        """Check if a threshold condition is met."""
        if threshold_type == AlertThreshold.ABOVE:
            return value > threshold
        elif threshold_type == AlertThreshold.BELOW:
            return value < threshold
        # Add other threshold types as needed
        return False

    def _create_alert(self, metric_name: str, current_value: float,
                     threshold_value: float, severity: str, agg_key: str,
                     threshold: PerformanceThreshold):
        """Create a performance alert."""
        alert_key = f"{metric_name}_{severity}"

        # Check if alert already exists and is recent
        if alert_key in self.active_alerts:
            last_alert = self.active_alerts[alert_key]
            last_alert_time = datetime.fromisoformat(last_alert.timestamp.replace('Z', '+00:00'))
            if (datetime.utcnow() - last_alert_time).seconds < threshold.duration:
                return  # Don't spam alerts

        alert = PerformanceAlert(
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            severity=severity,
            timestamp=datetime.utcnow().isoformat() + "Z",
            message=f"Performance {severity}: {metric_name} is {current_value:.2f} (threshold: {threshold_value:.2f})",
            details={
                "aggregation_key": agg_key,
                "threshold_type": threshold.threshold_type.value,
                "duration": threshold.duration
            }
        )

        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Log the alert
        logger.warning(f"PERFORMANCE ALERT: {alert.message}", extra={"alert": asdict(alert)})

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        aggregations = self.collector.get_aggregations()

        # Categorize metrics
        by_category = defaultdict(dict)
        for agg_key, agg in aggregations.items():
            # Extract category from key (format: metric_name_category)
            parts = agg_key.split('_')
            if len(parts) >= 2:
                category = parts[-1]
                metric_name = '_'.join(parts[:-1])
                by_category[category][metric_name] = {
                    "current": agg["last_value"],
                    "mean": agg["mean"],
                    "min": agg["min"],
                    "max": agg["max"],
                    "count": agg["count"],
                    "last_updated": agg["last_timestamp"]
                }

        active_alerts = list(self.active_alerts.values())

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "monitoring_active": self.monitoring_active,
            "total_metrics": len(aggregations),
            "metrics_by_category": dict(by_category),
            "active_alerts": len(active_alerts),
            "alert_summary": {
                "critical": len([a for a in active_alerts if a.severity == "critical"]),
                "warning": len([a for a in active_alerts if a.severity == "warning"])
            },
            "recent_alerts": active_alerts[-5:],  # Last 5 alerts
            "thresholds_configured": len(self.thresholds)
        }

    def start_monitoring(self):
        """Start background performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self.monitoring_active = True

        # System metrics collection thread
        if self.enable_system_monitoring:
            def system_metrics_loop():
                while self.monitoring_active:
                    try:
                        self.collect_system_metrics()
                        time.sleep(self.collection_interval)
                    except Exception as e:
                        logger.error(f"System metrics collection error: {e}")
                        time.sleep(self.collection_interval)

            system_thread = threading.Thread(target=system_metrics_loop, daemon=True)
            system_thread.start()
            self.monitoring_threads.append(system_thread)

        # Alert checking thread
        def alert_check_loop():
            while self.monitoring_active:
                try:
                    self.check_thresholds()
                    time.sleep(self.alert_check_interval)
                except Exception as e:
                    logger.error(f"Alert checking error: {e}")
                    time.sleep(self.alert_check_interval)

        alert_thread = threading.Thread(target=alert_check_loop, daemon=True)
        alert_thread.start()
        self.monitoring_threads.append(alert_thread)

        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.monitoring_active = False

        for thread in self.monitoring_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        self.monitoring_threads.clear()
        logger.info("Performance monitoring stopped")

    def export_metrics(self, output_path: str,
                      since: Optional[datetime] = None) -> bool:
        """
        Export performance metrics to file.

        Args:
            output_path: Output file path
            since: Export metrics since this timestamp

        Returns:
            True if exported successfully
        """
        try:
            metrics = self.collector.get_metrics(since=since)
            aggregations = self.collector.get_aggregations()

            export_data = {
                "export_timestamp": datetime.utcnow().isoformat() + "Z",
                "metrics_count": len(metrics),
                "aggregations": aggregations,
                "raw_metrics": [asdict(metric) for metric in metrics],
                "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
                "thresholds": {
                    name: asdict(threshold) for name, threshold in self.thresholds.items()
                }
            }

            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported {len(metrics)} metrics to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False


# Example usage and testing functions
def main():
    """Example usage of PerformanceMonitor."""
    # Initialize performance monitor
    perf_monitor = PerformanceMonitor(
        collection_interval=10,
        alert_check_interval=20,
        enable_system_monitoring=True
    )

    print("Performance monitoring system started")

    # Test API request recording
    perf_monitor.record_api_request("GET", "/api/equity/AAPL", 200, 0.15, "user123")
    perf_monitor.record_api_request("POST", "/api/equity/search", 200, 0.85)

    # Test database operation recording
    perf_monitor.record_database_operation("SELECT", "equity_profiles", 0.05, 1)

    # Start monitoring
    perf_monitor.start_monitoring()

    # Wait for some metrics collection
    time.sleep(3)

    # Test performance timer decorator
    @performance_timer(PerformanceCategory.APPLICATION, {"test": "example"})
    def test_function():
        time.sleep(0.1)  # Simulate work
        return "result"

    result = test_function()
    print(f"Function result: {result}")

    # Get performance summary
    print("\\nPerformance Summary:")
    summary = perf_monitor.get_performance_summary()
    print(json.dumps(summary, indent=2))

    # Export metrics
    if perf_monitor.export_metrics("performance_metrics.json"):
        print("\\nMetrics exported to performance_metrics.json")

    # Stop monitoring
    perf_monitor.stop_monitoring()

    print("\\nPerformance monitoring test completed")


if __name__ == "__main__":
    main()