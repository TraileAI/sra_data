"""
Comprehensive tests for Performance Monitoring infrastructure component.

Tests cover:
- Performance metrics collection and aggregation
- Performance thresholds and alerting
- API request monitoring
- Database operation monitoring
- System metrics collection
- Performance timer decorator
"""

import pytest
import time
import tempfile
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from packages.sra_data.infrastructure.performance_monitoring import (
    PerformanceMonitor,
    PerformanceCollector,
    MetricType,
    AlertThreshold,
    PerformanceCategory,
    PerformanceMetric,
    PerformanceThreshold,
    PerformanceAlert,
    performance_timer
)


class TestPerformanceCollector:
    """Test performance collector functionality."""

    @pytest.fixture
    def collector(self):
        """Create performance collector for testing."""
        return PerformanceCollector(max_metrics=1000)

    def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert collector.max_metrics == 1000
        assert len(collector.metrics) == 0
        assert len(collector.metric_aggregations) == 0

    def test_record_metric(self, collector):
        """Test recording a performance metric."""
        collector.record_metric(
            name="test.metric",
            value=123.45,
            metric_type=MetricType.GAUGE,
            category=PerformanceCategory.API,
            tags={"endpoint": "/test"},
            unit="ms",
            description="Test metric"
        )

        assert len(collector.metrics) == 1
        metric = collector.metrics[0]

        assert metric.name == "test.metric"
        assert metric.value == 123.45
        assert metric.metric_type == MetricType.GAUGE
        assert metric.category == PerformanceCategory.API
        assert metric.tags["endpoint"] == "/test"
        assert metric.unit == "ms"
        assert metric.description == "Test metric"

    def test_metric_aggregations(self, collector):
        """Test metric aggregation calculations."""
        # Record multiple values for the same metric
        values = [10.0, 20.0, 30.0, 15.0, 25.0]
        for value in values:
            collector.record_metric(
                name="test.aggregation",
                value=value,
                metric_type=MetricType.TIMER,
                category=PerformanceCategory.API
            )

        aggregations = collector.get_aggregations()
        key = "test.aggregation_api"

        assert key in aggregations
        agg = aggregations[key]

        assert agg["count"] == 5
        assert agg["sum"] == sum(values)
        assert agg["min"] == min(values)
        assert agg["max"] == max(values)
        assert agg["mean"] == sum(values) / len(values)
        assert agg["last_value"] == values[-1]

    def test_metric_filtering(self, collector):
        """Test metric filtering by name and category."""
        # Record various metrics
        collector.record_metric("api.requests", 1, MetricType.COUNTER, PerformanceCategory.API)
        collector.record_metric("api.response_time", 0.5, MetricType.TIMER, PerformanceCategory.API)
        collector.record_metric("db.queries", 1, MetricType.COUNTER, PerformanceCategory.DATABASE)

        # Filter by name
        api_metrics = collector.get_metrics(name_filter="api")
        assert len(api_metrics) == 2

        # Filter by category
        db_metrics = collector.get_metrics(category_filter=PerformanceCategory.DATABASE)
        assert len(db_metrics) == 1
        assert db_metrics[0].name == "db.queries"

    def test_metric_time_filtering(self, collector):
        """Test metric filtering by time."""
        # Record a metric
        collector.record_metric("test.metric", 1, MetricType.COUNTER, PerformanceCategory.API)

        # Get metrics since future time (should be empty)
        future_time = datetime.utcnow() + timedelta(minutes=1)
        future_metrics = collector.get_metrics(since=future_time)
        assert len(future_metrics) == 0

        # Get metrics since past time (should include our metric)
        past_time = datetime.utcnow() - timedelta(minutes=1)
        past_metrics = collector.get_metrics(since=past_time)
        assert len(past_metrics) == 1

    def test_max_metrics_limit(self):
        """Test that collector respects max metrics limit."""
        collector = PerformanceCollector(max_metrics=5)

        # Record more metrics than the limit
        for i in range(10):
            collector.record_metric(
                f"metric.{i}", i, MetricType.GAUGE, PerformanceCategory.APPLICATION
            )

        # Should only keep the last 5 metrics
        assert len(collector.metrics) == 5
        assert collector.metrics[-1].name == "metric.9"
        assert collector.metrics[0].name == "metric.5"


class TestPerformanceMonitor:
    """Test performance monitor functionality."""

    @pytest.fixture
    def monitor(self):
        """Create performance monitor for testing."""
        return PerformanceMonitor(
            collection_interval=1,
            alert_check_interval=1,
            enable_system_monitoring=False  # Disable for testing
        )

    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.collection_interval == 1
        assert monitor.alert_check_interval == 1
        assert monitor.enable_system_monitoring is False
        assert monitor.collector is not None
        assert len(monitor.thresholds) > 0  # Default thresholds

    def test_add_remove_threshold(self, monitor):
        """Test adding and removing performance thresholds."""
        threshold = PerformanceThreshold(
            "custom.metric", AlertThreshold.ABOVE, 50.0, 100.0
        )

        # Add threshold
        monitor.add_threshold(threshold)
        assert "custom.metric" in monitor.thresholds
        assert monitor.thresholds["custom.metric"] == threshold

        # Remove threshold
        monitor.remove_threshold("custom.metric")
        assert "custom.metric" not in monitor.thresholds

    def test_record_api_request(self, monitor):
        """Test API request recording."""
        monitor.record_api_request(
            method="GET",
            endpoint="/api/test",
            status_code=200,
            response_time=0.15,
            user_id="user123"
        )

        metrics = monitor.collector.get_metrics()
        assert len(metrics) >= 2  # At least response_time and requests

        # Check response time metric
        response_time_metrics = [m for m in metrics if m.name == "api.response_time"]
        assert len(response_time_metrics) == 1
        assert response_time_metrics[0].value == 0.15
        assert response_time_metrics[0].tags["method"] == "GET"
        assert response_time_metrics[0].tags["endpoint"] == "/api/test"
        assert response_time_metrics[0].tags["status_code"] == "200"
        assert response_time_metrics[0].tags["user_id"] == "user123"

    def test_record_api_error(self, monitor):
        """Test API error recording."""
        monitor.record_api_request(
            method="POST",
            endpoint="/api/test",
            status_code=500,
            response_time=1.0
        )

        metrics = monitor.collector.get_metrics()
        error_metrics = [m for m in metrics if m.name == "api.errors"]
        assert len(error_metrics) == 1
        assert error_metrics[0].value == 1
        assert error_metrics[0].tags["status_code"] == "500"

    def test_record_database_operation(self, monitor):
        """Test database operation recording."""
        monitor.record_database_operation(
            operation="SELECT",
            table="users",
            duration=0.05,
            rows_affected=10,
            connection_time=0.01
        )

        metrics = monitor.collector.get_metrics()

        # Check query duration metric
        duration_metrics = [m for m in metrics if m.name == "database.query_duration"]
        assert len(duration_metrics) == 1
        assert duration_metrics[0].value == 0.05
        assert duration_metrics[0].tags["operation"] == "SELECT"
        assert duration_metrics[0].tags["table"] == "users"

        # Check connection time metric
        conn_metrics = [m for m in metrics if m.name == "database.connection_time"]
        assert len(conn_metrics) == 1
        assert conn_metrics[0].value == 0.01

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.Process')
    def test_collect_system_metrics(self, mock_process, mock_disk, mock_memory, mock_cpu, monitor):
        """Test system metrics collection."""
        # Mock system information
        mock_cpu.return_value = 75.5

        mock_memory_info = MagicMock()
        mock_memory_info.percent = 60.0
        mock_memory_info.available = 1024 * 1024 * 1024
        mock_memory.return_value = mock_memory_info

        mock_disk_info = MagicMock()
        mock_disk_info.percent = 45.0
        mock_disk.return_value = mock_disk_info

        mock_process_info = MagicMock()
        mock_process_info.cpu_percent.return_value = 25.0
        mock_process_info.memory_percent.return_value = 15.0
        mock_process.return_value = mock_process_info

        # Collect system metrics
        monitor.collect_system_metrics()

        metrics = monitor.collector.get_metrics()
        metric_names = [m.name for m in metrics]

        assert "system.cpu_usage" in metric_names
        assert "system.memory_usage" in metric_names
        assert "system.disk_usage" in metric_names
        assert "process.cpu_percent" in metric_names
        assert "process.memory_percent" in metric_names

        # Check specific values
        cpu_metric = next(m for m in metrics if m.name == "system.cpu_usage")
        assert cpu_metric.value == 75.5

    def test_threshold_checking_above(self, monitor):
        """Test threshold checking for ABOVE threshold type."""
        # Add a threshold
        threshold = PerformanceThreshold(
            "test.metric", AlertThreshold.ABOVE, 50.0, 100.0, duration=1
        )
        monitor.add_threshold(threshold)

        # Record metric below threshold
        monitor.collector.record_metric(
            "test.metric", 25.0, MetricType.GAUGE, PerformanceCategory.APPLICATION
        )

        # Check thresholds (should not trigger)
        monitor.check_thresholds()
        assert len(monitor.active_alerts) == 0

        # Record metric above warning threshold
        monitor.collector.record_metric(
            "test.metric", 75.0, MetricType.GAUGE, PerformanceCategory.APPLICATION
        )

        # Check thresholds (should trigger warning)
        monitor.check_thresholds()
        assert len(monitor.active_alerts) > 0

        # Check alert details
        alert_keys = list(monitor.active_alerts.keys())
        warning_alert = monitor.active_alerts[alert_keys[0]]
        assert warning_alert.severity == "warning"
        assert warning_alert.current_value == 75.0

    def test_threshold_checking_critical(self, monitor):
        """Test critical threshold triggering."""
        threshold = PerformanceThreshold(
            "critical.metric", AlertThreshold.ABOVE, 50.0, 100.0
        )
        monitor.add_threshold(threshold)

        # Record metric above critical threshold
        monitor.collector.record_metric(
            "critical.metric", 150.0, MetricType.GAUGE, PerformanceCategory.APPLICATION
        )

        monitor.check_thresholds()

        # Should have critical alert
        critical_alerts = [
            alert for alert in monitor.active_alerts.values()
            if alert.severity == "critical"
        ]
        assert len(critical_alerts) > 0
        assert critical_alerts[0].current_value == 150.0

    def test_performance_summary(self, monitor):
        """Test performance summary generation."""
        # Record some metrics
        monitor.record_api_request("GET", "/api/test", 200, 0.1)
        monitor.record_database_operation("SELECT", "users", 0.05)

        summary = monitor.get_performance_summary()

        assert "timestamp" in summary
        assert "monitoring_active" in summary
        assert "total_metrics" in summary
        assert "metrics_by_category" in summary
        assert "active_alerts" in summary
        assert "alert_summary" in summary
        assert "thresholds_configured" in summary

        # Check that we have metrics by category
        metrics_by_category = summary["metrics_by_category"]
        assert len(metrics_by_category) > 0

    def test_export_metrics(self, monitor):
        """Test metrics export functionality."""
        # Record some metrics
        monitor.record_api_request("POST", "/api/data", 201, 0.25)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            result = monitor.export_metrics(tmp_file.name)
            assert result is True

            # Check exported content
            with open(tmp_file.name, 'r') as f:
                exported = json.load(f)

            assert "export_timestamp" in exported
            assert "metrics_count" in exported
            assert "aggregations" in exported
            assert "raw_metrics" in exported
            assert "active_alerts" in exported
            assert "thresholds" in exported

            # Check that we have metrics
            assert exported["metrics_count"] > 0
            assert len(exported["raw_metrics"]) > 0

    def test_monitoring_start_stop(self, monitor):
        """Test starting and stopping background monitoring."""
        assert monitor.monitoring_active is False

        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring_active is True
        assert len(monitor.monitoring_threads) > 0

        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor.monitoring_active is False
        assert len(monitor.monitoring_threads) == 0

    def test_alert_cooldown(self, monitor):
        """Test alert cooldown functionality."""
        threshold = PerformanceThreshold(
            "cooldown.metric", AlertThreshold.ABOVE, 50.0, 100.0, duration=60
        )
        monitor.add_threshold(threshold)

        # Trigger alert
        monitor.collector.record_metric(
            "cooldown.metric", 75.0, MetricType.GAUGE, PerformanceCategory.APPLICATION
        )
        monitor.check_thresholds()

        initial_alert_count = len(monitor.active_alerts)
        assert initial_alert_count > 0

        # Trigger again immediately (should not create new alert due to cooldown)
        monitor.collector.record_metric(
            "cooldown.metric", 80.0, MetricType.GAUGE, PerformanceCategory.APPLICATION
        )
        monitor.check_thresholds()

        # Alert count should not increase due to cooldown
        assert len(monitor.active_alerts) == initial_alert_count


class TestPerformanceTimer:
    """Test performance timer decorator."""

    @pytest.fixture
    def monitor(self):
        """Create monitor and set up decorator."""
        monitor = PerformanceMonitor(enable_system_monitoring=False)
        performance_timer.collector = monitor.collector
        return monitor

    def test_sync_function_timing(self, monitor):
        """Test timing of synchronous functions."""
        @performance_timer(PerformanceCategory.APPLICATION, {"test": "sync"})
        def test_function():
            time.sleep(0.1)
            return "result"

        result = test_function()
        assert result == "result"

        # Check that timing metric was recorded
        metrics = monitor.collector.get_metrics()
        duration_metrics = [m for m in metrics if "duration" in m.name]
        assert len(duration_metrics) == 1

        duration_metric = duration_metrics[0]
        assert duration_metric.value >= 0.1  # Should be at least 0.1 seconds
        assert duration_metric.metric_type == MetricType.TIMER
        assert duration_metric.category == PerformanceCategory.APPLICATION
        assert duration_metric.tags["test"] == "sync"

    @pytest.mark.asyncio
    async def test_async_function_timing(self, monitor):
        """Test timing of asynchronous functions."""
        @performance_timer(PerformanceCategory.API, {"test": "async"})
        async def async_test_function():
            await asyncio.sleep(0.1)
            return "async_result"

        import asyncio
        result = await async_test_function()
        assert result == "async_result"

        # Check timing metric
        metrics = monitor.collector.get_metrics()
        duration_metrics = [m for m in metrics if "duration" in m.name]
        assert len(duration_metrics) == 1

        duration_metric = duration_metrics[0]
        assert duration_metric.value >= 0.1
        assert duration_metric.tags["test"] == "async"

    def test_function_error_tracking(self, monitor):
        """Test error tracking in timed functions."""
        @performance_timer(PerformanceCategory.DATABASE)
        def error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            error_function()

        # Check that error metric was recorded
        metrics = monitor.collector.get_metrics()
        error_metrics = [m for m in metrics if "errors" in m.name]
        assert len(error_metrics) == 1

        error_metric = error_metrics[0]
        assert error_metric.value == 1
        assert error_metric.metric_type == MetricType.COUNTER
        assert error_metric.tags["error_type"] == "ValueError"

    def test_decorator_without_collector(self):
        """Test decorator behavior when collector is not set."""
        # Temporarily remove collector
        original_collector = getattr(performance_timer, 'collector', None)
        performance_timer.collector = None

        @performance_timer(PerformanceCategory.APPLICATION)
        def test_function():
            return "result"

        # Should not crash even without collector
        result = test_function()
        assert result == "result"

        # Restore collector
        performance_timer.collector = original_collector


# Integration tests
class TestPerformanceMonitoringIntegration:
    """Integration tests for performance monitoring."""

    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        monitor = PerformanceMonitor(
            collection_interval=1,
            enable_system_monitoring=True
        )

        try:
            # Start monitoring
            monitor.start_monitoring()

            # Record various performance metrics
            monitor.record_api_request("GET", "/api/users", 200, 0.15)
            monitor.record_api_request("POST", "/api/users", 201, 0.25)
            monitor.record_api_request("GET", "/api/users/1", 404, 0.05)

            monitor.record_database_operation("SELECT", "users", 0.03, 5)
            monitor.record_database_operation("INSERT", "users", 0.08, 1)

            # Wait for system metrics collection
            time.sleep(2)

            # Get performance summary
            summary = monitor.get_performance_summary()
            assert summary["total_metrics"] > 0
            assert len(summary["metrics_by_category"]) > 0

            # Check for API metrics
            if "api" in summary["metrics_by_category"]:
                api_metrics = summary["metrics_by_category"]["api"]
                assert len(api_metrics) > 0

            # Export metrics
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                result = monitor.export_metrics(tmp_file.name)
                assert result is True

                # Verify export content
                with open(tmp_file.name) as f:
                    exported = json.load(f)
                assert exported["metrics_count"] > 0

        finally:
            # Clean up
            monitor.stop_monitoring()

    def test_alert_integration(self):
        """Test alert integration with real metrics."""
        monitor = PerformanceMonitor(enable_system_monitoring=False)

        # Add sensitive thresholds
        monitor.add_threshold(PerformanceThreshold(
            "api.response_time", AlertThreshold.ABOVE, 0.1, 0.2
        ))

        # Record slow API requests
        monitor.record_api_request("GET", "/api/slow", 200, 0.15)  # Warning
        monitor.record_api_request("GET", "/api/very-slow", 200, 0.25)  # Critical

        # Check thresholds
        monitor.check_thresholds()

        # Should have both warning and critical alerts
        alerts = list(monitor.active_alerts.values())
        assert len(alerts) >= 1

        # Check alert history
        assert len(monitor.alert_history) >= 1

    @patch('psutil.cpu_percent', return_value=95.0)
    @patch('psutil.virtual_memory')
    def test_system_resource_alerting(self, mock_memory, mock_cpu):
        """Test system resource alerting."""
        # Mock high memory usage
        mock_memory_info = MagicMock()
        mock_memory_info.percent = 98.0
        mock_memory_info.available = 1024 * 1024  # Low available memory
        mock_memory.return_value = mock_memory_info

        monitor = PerformanceMonitor(
            collection_interval=1,
            enable_system_monitoring=True
        )

        # Collect system metrics
        monitor.collect_system_metrics()

        # Check thresholds
        monitor.check_thresholds()

        # Should have alerts for high CPU and memory
        alerts = list(monitor.active_alerts.values())
        alert_metrics = [alert.metric_name for alert in alerts]

        # Should have system resource alerts
        system_alerts = [metric for metric in alert_metrics if "system." in metric]
        assert len(system_alerts) > 0


if __name__ == "__main__":
    pytest.main([__file__])