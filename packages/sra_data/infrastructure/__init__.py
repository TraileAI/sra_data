"""
Infrastructure package for SRA Data Processing System.

This package contains comprehensive infrastructure components including:
- Git LFS management for large CSV files
- Deployment configuration and optimization for Render.com
- Environment variable management and secrets handling
- Production logging and monitoring systems
- Health checks and service reliability monitoring
- Performance monitoring and alerting
"""

__version__ = "1.0.0"
__author__ = "SRA Data Processing Team"

# Infrastructure component imports
from .git_lfs_manager import GitLFSManager
from .deployment_config import (
    DeploymentConfig,
    ServiceType,
    PlanType,
    RuntimeType,
    EnvironmentVariable as DeploymentEnvironmentVariable,
    ServiceConfig,
    DatabaseConfig,
    DeploymentEnvironment
)
from .environment_manager import (
    EnvironmentManager,
    EnvironmentType,
    VariableType,
    EnvironmentVariable,
    EnvironmentSchema,
    ValidationResult,
    EncryptionManager
)
from .production_logging import (
    ProductionLogger,
    LogLevel,
    AlertSeverity,
    LogEntry,
    MetricEntry,
    AlertRule,
    StructuredFormatter,
    MetricsCollector,
    AlertManager
)
from .health_monitoring import (
    HealthMonitor,
    HealthStatus,
    ServiceType as HealthServiceType,
    CircuitState,
    HealthCheck,
    HealthResult,
    CircuitBreakerConfig,
    CircuitBreaker
)
from .performance_monitoring import (
    PerformanceMonitor,
    MetricType,
    AlertThreshold,
    PerformanceCategory,
    PerformanceMetric,
    PerformanceThreshold,
    PerformanceAlert,
    PerformanceCollector,
    performance_timer
)

__all__ = [
    # Git LFS Management
    "GitLFSManager",

    # Deployment Configuration
    "DeploymentConfig",
    "ServiceType",
    "PlanType",
    "RuntimeType",
    "DeploymentEnvironmentVariable",
    "ServiceConfig",
    "DatabaseConfig",
    "DeploymentEnvironment",

    # Environment Management
    "EnvironmentManager",
    "EnvironmentType",
    "VariableType",
    "EnvironmentVariable",
    "EnvironmentSchema",
    "ValidationResult",
    "EncryptionManager",

    # Production Logging
    "ProductionLogger",
    "LogLevel",
    "AlertSeverity",
    "LogEntry",
    "MetricEntry",
    "AlertRule",
    "StructuredFormatter",
    "MetricsCollector",
    "AlertManager",

    # Health Monitoring
    "HealthMonitor",
    "HealthStatus",
    "HealthServiceType",
    "CircuitState",
    "HealthCheck",
    "HealthResult",
    "CircuitBreakerConfig",
    "CircuitBreaker",

    # Performance Monitoring
    "PerformanceMonitor",
    "MetricType",
    "AlertThreshold",
    "PerformanceCategory",
    "PerformanceMetric",
    "PerformanceThreshold",
    "PerformanceAlert",
    "PerformanceCollector",
    "performance_timer"
]