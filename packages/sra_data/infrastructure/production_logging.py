"""
Production Logging and Monitoring System for SRA Data Processing.

This module provides comprehensive production-ready logging and monitoring including:
- Structured logging with multiple handlers
- Performance monitoring and metrics collection
- Error tracking and alerting
- Log aggregation and analysis
- Health checks and service monitoring
- Production-grade log management
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
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil


# Configure logging
logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    process_id: int
    thread_id: int
    extra_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_data is None:
            self.extra_data = {}


@dataclass
class MetricEntry:
    """Performance metric entry."""
    name: str
    value: float
    timestamp: str
    tags: Dict[str, str] = None
    metric_type: str = "gauge"  # gauge, counter, histogram, timer

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: str
    severity: AlertSeverity
    threshold: float
    duration: int  # seconds
    enabled: bool = True
    cooldown: int = 300  # seconds between alerts


class StructuredFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def __init__(self, include_extra: bool = True):
        """
        Initialize formatter.

        Args:
            include_extra: Whether to include extra fields in log output
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""

        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }

        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {
                k: v for k, v in record.__dict__.items()
                if k not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'stack_info', 'exc_info',
                    'exc_text', 'message'
                }
            }
            if extra_fields:
                log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class MetricsCollector:
    """
    Collects and manages performance metrics.
    """

    def __init__(self, max_entries: int = 10000):
        """
        Initialize metrics collector.

        Args:
            max_entries: Maximum number of metric entries to keep in memory
        """
        self.max_entries = max_entries
        self.metrics: deque = deque(maxlen=max_entries)
        self.aggregated_metrics: Dict[str, Any] = defaultdict(dict)
        self._lock = threading.Lock()

    def record_metric(self, name: str, value: float,
                     tags: Optional[Dict[str, str]] = None,
                     metric_type: str = "gauge"):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            metric_type: Type of metric (gauge, counter, histogram, timer)
        """
        with self._lock:
            metric = MetricEntry(
                name=name,
                value=value,
                timestamp=datetime.utcnow().isoformat() + "Z",
                tags=tags or {},
                metric_type=metric_type
            )
            self.metrics.append(metric)

            # Update aggregated metrics
            self._update_aggregated_metrics(metric)

    def _update_aggregated_metrics(self, metric: MetricEntry):
        """Update aggregated metrics with new entry."""
        metric_key = f"{metric.name}_{metric.metric_type}"

        if metric_key not in self.aggregated_metrics:
            self.aggregated_metrics[metric_key] = {
                "count": 0,
                "sum": 0,
                "min": float('inf'),
                "max": float('-inf'),
                "avg": 0,
                "last_value": 0,
                "last_updated": None
            }

        stats = self.aggregated_metrics[metric_key]
        stats["count"] += 1
        stats["sum"] += metric.value
        stats["min"] = min(stats["min"], metric.value)
        stats["max"] = max(stats["max"], metric.value)
        stats["avg"] = stats["sum"] / stats["count"]
        stats["last_value"] = metric.value
        stats["last_updated"] = metric.timestamp

    def get_metrics(self, name_filter: Optional[str] = None,
                   since: Optional[datetime] = None) -> List[MetricEntry]:
        """
        Get metrics with optional filtering.

        Args:
            name_filter: Filter by metric name (substring match)
            since: Only return metrics after this timestamp

        Returns:
            List of matching metrics
        """
        with self._lock:
            filtered_metrics = []

            for metric in self.metrics:
                # Apply name filter
                if name_filter and name_filter not in metric.name:
                    continue

                # Apply time filter
                if since:
                    metric_time = datetime.fromisoformat(metric.timestamp.replace('Z', '+00:00'))
                    if metric_time < since:
                        continue

                filtered_metrics.append(metric)

            return filtered_metrics

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        with self._lock:
            return dict(self.aggregated_metrics)

    def record_system_metrics(self):
        """Record system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system.cpu.usage", cpu_percent, metric_type="gauge")

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system.memory.usage", memory.percent, metric_type="gauge")
            self.record_metric("system.memory.available", memory.available, metric_type="gauge")

            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system.disk.usage", disk.percent, metric_type="gauge")
            self.record_metric("system.disk.free", disk.free, metric_type="gauge")

            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self.record_metric("system.network.bytes_sent", net_io.bytes_sent, metric_type="counter")
                self.record_metric("system.network.bytes_recv", net_io.bytes_recv, metric_type="counter")
            except:
                pass  # Network metrics not available

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")


class AlertManager:
    """
    Manages alerts and notifications based on metrics and logs.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, datetime] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_alert_rule(self, rule: AlertRule):
        """
        Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        with self._lock:
            self.alert_rules.append(rule)

    def check_alerts(self, metrics: List[MetricEntry]):
        """
        Check metrics against alert rules.

        Args:
            metrics: List of metrics to check
        """
        with self._lock:
            current_time = datetime.utcnow()

            for rule in self.alert_rules:
                if not rule.enabled:
                    continue

                # Check if rule is in cooldown
                last_alert = self.active_alerts.get(rule.name)
                if last_alert and (current_time - last_alert).seconds < rule.cooldown:
                    continue

                # Evaluate rule condition
                if self._evaluate_rule(rule, metrics, current_time):
                    self._trigger_alert(rule, current_time)

    def _evaluate_rule(self, rule: AlertRule, metrics: List[MetricEntry],
                      current_time: datetime) -> bool:
        """
        Evaluate a rule against metrics.

        Args:
            rule: Alert rule to evaluate
            metrics: Metrics to check
            current_time: Current timestamp

        Returns:
            True if rule condition is met
        """
        try:
            # Simple condition evaluation (can be extended)
            if ">" in rule.condition:
                metric_name, threshold_str = rule.condition.split(">")
                metric_name = metric_name.strip()
                threshold = float(threshold_str.strip())

                # Find relevant metrics
                relevant_metrics = [
                    m for m in metrics
                    if m.name == metric_name and
                    (current_time - datetime.fromisoformat(m.timestamp.replace('Z', '+00:00'))).seconds <= rule.duration
                ]

                if relevant_metrics:
                    latest_value = relevant_metrics[-1].value
                    return latest_value > threshold

        except Exception as e:
            logger.warning(f"Failed to evaluate alert rule {rule.name}: {e}")

        return False

    def _trigger_alert(self, rule: AlertRule, current_time: datetime):
        """
        Trigger an alert.

        Args:
            rule: Alert rule that was triggered
            current_time: Current timestamp
        """
        self.active_alerts[rule.name] = current_time

        alert_data = {
            "rule_name": rule.name,
            "severity": rule.severity.value,
            "condition": rule.condition,
            "threshold": rule.threshold,
            "timestamp": current_time.isoformat() + "Z",
            "message": f"Alert triggered: {rule.name} - {rule.condition}"
        }

        self.alert_history.append(alert_data)

        # Log the alert
        logger.warning(f"ALERT TRIGGERED: {alert_data['message']}",
                      extra={"alert": alert_data})

        # TODO: Send notification (email, Slack, etc.)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        with self._lock:
            current_time = datetime.utcnow()
            active = []

            for rule_name, alert_time in self.active_alerts.items():
                # Check if alert is still in cooldown
                if (current_time - alert_time).seconds < 3600:  # 1 hour max
                    active.append({
                        "rule_name": rule_name,
                        "alert_time": alert_time.isoformat() + "Z",
                        "duration": (current_time - alert_time).seconds
                    })

            return active


class ProductionLogger:
    """
    Comprehensive production logging and monitoring system.

    Provides structured logging, metrics collection, alerting, and monitoring
    for production deployment of SRA Data Processing System.
    """

    def __init__(self,
                 log_dir: str = "logs",
                 metrics_enabled: bool = True,
                 alerts_enabled: bool = True,
                 structured_logging: bool = True):
        """
        Initialize production logger.

        Args:
            log_dir: Directory for log files
            metrics_enabled: Whether to enable metrics collection
            alerts_enabled: Whether to enable alerting
            structured_logging: Whether to use structured JSON logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.metrics_enabled = metrics_enabled
        self.alerts_enabled = alerts_enabled
        self.structured_logging = structured_logging

        # Initialize components
        self.metrics_collector = MetricsCollector() if metrics_enabled else None
        self.alert_manager = AlertManager() if alerts_enabled else None

        # Setup logging
        self._setup_logging()

        # Setup default alert rules
        if self.alert_manager:
            self._setup_default_alerts()

        # Start background tasks
        self._start_background_tasks()

        logger.info("Production logging system initialized")

    def _setup_logging(self):
        """Setup logging configuration."""
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Console handler
        console_handler = logging.StreamHandler()
        if self.structured_logging:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        root_logger.addHandler(console_handler)

        # File handlers
        # Main application log
        app_handler = logging.FileHandler(self.log_dir / "application.log")
        if self.structured_logging:
            app_handler.setFormatter(StructuredFormatter())
        else:
            app_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        root_logger.addHandler(app_handler)

        # Error log
        error_handler = logging.FileHandler(self.log_dir / "error.log")
        error_handler.setLevel(logging.ERROR)
        if self.structured_logging:
            error_handler.setFormatter(StructuredFormatter())
        else:
            error_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        root_logger.addHandler(error_handler)

        # Access log (for API requests)
        access_logger = logging.getLogger("access")
        access_handler = logging.FileHandler(self.log_dir / "access.log")
        if self.structured_logging:
            access_handler.setFormatter(StructuredFormatter())
        else:
            access_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(message)s')
            )
        access_logger.addHandler(access_handler)
        access_logger.setLevel(logging.INFO)

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        if not self.alert_manager:
            return

        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                condition="system.cpu.usage > 85",
                severity=AlertSeverity.HIGH,
                threshold=85.0,
                duration=300  # 5 minutes
            ),
            AlertRule(
                name="high_memory_usage",
                condition="system.memory.usage > 90",
                severity=AlertSeverity.CRITICAL,
                threshold=90.0,
                duration=300  # 5 minutes
            ),
            AlertRule(
                name="low_disk_space",
                condition="system.disk.usage > 90",
                severity=AlertSeverity.CRITICAL,
                threshold=90.0,
                duration=60   # 1 minute
            ),
            AlertRule(
                name="high_error_rate",
                condition="application.errors > 10",
                severity=AlertSeverity.HIGH,
                threshold=10.0,
                duration=600  # 10 minutes
            )
        ]

        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)

    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        if self.metrics_enabled:
            # Start system metrics collection
            def collect_system_metrics():
                while True:
                    try:
                        if self.metrics_collector:
                            self.metrics_collector.record_system_metrics()
                    except Exception as e:
                        logger.warning(f"System metrics collection failed: {e}")
                    time.sleep(60)  # Collect every minute

            metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
            metrics_thread.start()

        if self.alerts_enabled and self.alert_manager:
            # Start alert checking
            def check_alerts():
                while True:
                    try:
                        if self.metrics_collector:
                            recent_metrics = self.metrics_collector.get_metrics(
                                since=datetime.utcnow() - timedelta(minutes=10)
                            )
                            self.alert_manager.check_alerts(recent_metrics)
                    except Exception as e:
                        logger.warning(f"Alert checking failed: {e}")
                    time.sleep(30)  # Check every 30 seconds

            alerts_thread = threading.Thread(target=check_alerts, daemon=True)
            alerts_thread.start()

    def log_request(self, method: str, path: str, status_code: int,
                   response_time: float, user_id: Optional[str] = None):
        """
        Log API request.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            response_time: Response time in seconds
            user_id: Optional user ID
        """
        access_logger = logging.getLogger("access")

        request_data = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "response_time": response_time,
            "user_id": user_id
        }

        access_logger.info("API Request", extra={"request": request_data})

        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "api.requests", 1,
                tags={"method": method, "status": str(status_code)},
                metric_type="counter"
            )
            self.metrics_collector.record_metric(
                "api.response_time", response_time,
                tags={"method": method, "path": path},
                metric_type="timer"
            )

    def log_database_operation(self, operation: str, table: str,
                             duration: float, rows_affected: int = 0):
        """
        Log database operation.

        Args:
            operation: Database operation (SELECT, INSERT, UPDATE, DELETE)
            table: Table name
            duration: Operation duration in seconds
            rows_affected: Number of rows affected
        """
        logger.info("Database Operation", extra={
            "operation": operation,
            "table": table,
            "duration": duration,
            "rows_affected": rows_affected
        })

        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "database.operations", 1,
                tags={"operation": operation, "table": table},
                metric_type="counter"
            )
            self.metrics_collector.record_metric(
                "database.duration", duration,
                tags={"operation": operation, "table": table},
                metric_type="timer"
            )

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Log application error with context.

        Args:
            error: Exception that occurred
            context: Additional context information
        """
        logger.error(f"Application Error: {str(error)}",
                    extra={"error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "context": context or {}
                    }}, exc_info=error)

        # Record error metric
        if self.metrics_collector:
            self.metrics_collector.record_metric(
                "application.errors", 1,
                tags={"error_type": type(error).__name__},
                metric_type="counter"
            )

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.

        Returns:
            Health status information
        """
        health_status = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "healthy",
            "components": {},
            "metrics": {},
            "alerts": {}
        }

        # System metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            health_status["components"]["system"] = {
                "status": "healthy",
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent
            }

            # Determine overall health
            if cpu_percent > 90 or memory.percent > 95 or disk.percent > 95:
                health_status["status"] = "critical"
            elif cpu_percent > 80 or memory.percent > 85 or disk.percent > 85:
                health_status["status"] = "warning"

        except Exception as e:
            health_status["components"]["system"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "error"

        # Metrics status
        if self.metrics_collector:
            aggregated_metrics = self.metrics_collector.get_aggregated_metrics()
            health_status["metrics"] = {
                "total_metrics": len(aggregated_metrics),
                "last_collection": datetime.utcnow().isoformat() + "Z"
            }

        # Alert status
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            health_status["alerts"] = {
                "active_count": len(active_alerts),
                "active_alerts": active_alerts[:5]  # Show first 5
            }

            if active_alerts:
                critical_alerts = [a for a in active_alerts if "critical" in str(a)]
                if critical_alerts:
                    health_status["status"] = "critical"
                elif health_status["status"] == "healthy":
                    health_status["status"] = "warning"

        return health_status

    def export_logs(self, output_path: str, since: Optional[datetime] = None,
                   level: Optional[LogLevel] = None) -> bool:
        """
        Export logs to file.

        Args:
            output_path: Output file path
            since: Export logs since this timestamp
            level: Minimum log level to export

        Returns:
            True if exported successfully
        """
        try:
            exported_logs = []

            # Read log files
            log_files = [
                self.log_dir / "application.log",
                self.log_dir / "error.log"
            ]

            for log_file in log_files:
                if not log_file.exists():
                    continue

                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            if self.structured_logging:
                                log_entry = json.loads(line.strip())

                                # Apply time filter
                                if since:
                                    log_time = datetime.fromisoformat(
                                        log_entry["timestamp"].replace('Z', '+00:00')
                                    )
                                    if log_time < since:
                                        continue

                                # Apply level filter
                                if level and log_entry["level"] != level.value:
                                    continue

                                exported_logs.append(log_entry)
                            else:
                                # For non-structured logs, just include the line
                                exported_logs.append({"raw_log": line.strip()})

                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue

            # Write exported logs
            with open(output_path, 'w') as f:
                json.dump({
                    "export_timestamp": datetime.utcnow().isoformat() + "Z",
                    "total_logs": len(exported_logs),
                    "logs": exported_logs
                }, f, indent=2)

            logger.info(f"Exported {len(exported_logs)} log entries to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            return False


# Example usage and testing functions
def main():
    """Example usage of ProductionLogger."""
    # Initialize production logger
    prod_logger = ProductionLogger(
        log_dir="logs",
        metrics_enabled=True,
        alerts_enabled=True,
        structured_logging=True
    )

    print("Production logging system started")

    # Test logging
    logger.info("Test info message")
    logger.warning("Test warning message", extra={"test_data": "example"})

    # Test request logging
    prod_logger.log_request("GET", "/api/equity/AAPL", 200, 0.15, "user123")

    # Test database logging
    prod_logger.log_database_operation("SELECT", "equity_profiles", 0.05, 1)

    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        prod_logger.log_error(e, {"context": "testing"})

    # Get health status
    print("\\nHealth Status:")
    health = prod_logger.get_health_status()
    print(json.dumps(health, indent=2))

    # Wait a bit for metrics collection
    time.sleep(2)

    # Check metrics
    if prod_logger.metrics_collector:
        metrics = prod_logger.metrics_collector.get_aggregated_metrics()
        print(f"\\nCollected {len(metrics)} metric types")
        for metric_name, stats in list(metrics.items())[:5]:
            print(f"  {metric_name}: {stats}")

    print("\\nProduction logging test completed")


if __name__ == "__main__":
    main()