"""Repositories package for SRA Data processing service.

This package provides database repositories with CRUD operations for
equity profiles, fundata records, and other domain models.
"""

from .database import DatabaseManager, DatabaseConfig, ConnectionPool, SchemaManager
from .equity_repository import EquityRepository
from .fundata_repository import FundataRepository, FundataDataRepository, FundataQuotesRepository
from .migrations import MigrationManager, Migration, MigrationStatus, create_migration_manager
from .table_manager import TableManager, PartitionType, TableMaintenanceResult
from .view_manager import ViewManager, ViewType, ViewRefreshResult
from .performance_optimizer import (
    PerformanceOptimizer,
    OptimizationLevel,
    QueryPerformanceMetric,
    IndexRecommendation,
    PerformanceReport
)
from .transaction_manager import (
    TransactionManager,
    TransactionIsolationLevel,
    IntegrityViolationType,
    IntegrityViolation,
    TransactionResult
)

__all__ = [
    # Database infrastructure
    'DatabaseManager',
    'DatabaseConfig',
    'ConnectionPool',
    'SchemaManager',
    # Repositories
    'EquityRepository',
    'FundataRepository',
    'FundataDataRepository',
    'FundataQuotesRepository',
    # Migration system
    'MigrationManager',
    'Migration',
    'MigrationStatus',
    'create_migration_manager',
    # Table management
    'TableManager',
    'PartitionType',
    'TableMaintenanceResult',
    # View management
    'ViewManager',
    'ViewType',
    'ViewRefreshResult',
    # Performance optimization
    'PerformanceOptimizer',
    'OptimizationLevel',
    'QueryPerformanceMetric',
    'IndexRecommendation',
    'PerformanceReport',
    # Transaction management
    'TransactionManager',
    'TransactionIsolationLevel',
    'IntegrityViolationType',
    'IntegrityViolation',
    'TransactionResult'
]