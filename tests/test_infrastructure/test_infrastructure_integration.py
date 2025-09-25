"""
Integration tests for Infrastructure components.

Tests the integration and interaction between:
- Environment Management
- Deployment Configuration
- Production Logging
- Health Monitoring
- Performance Monitoring
- Git LFS Management
"""

import pytest
import tempfile
import os
import time
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from packages.sra_data.infrastructure import (
    # Environment Management
    EnvironmentManager,
    EnvironmentType,
    VariableType,

    # Deployment Configuration
    DeploymentConfig,
    ServiceType,
    PlanType,
    RuntimeType,

    # Production Logging
    ProductionLogger,
    LogLevel,

    # Health Monitoring
    HealthMonitor,
    HealthStatus,

    # Performance Monitoring
    PerformanceMonitor,
    PerformanceCategory,
    MetricType,

    # Git LFS Management
    GitLFSManager
)


class TestInfrastructureIntegration:
    """Integration tests for infrastructure components."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)

            # Create basic project structure
            (project_path / "config").mkdir()
            (project_path / "logs").mkdir()
            (project_path / "data").mkdir()

            # Create dummy files for LFS testing
            (project_path / "data" / "large_file.csv").write_text("dummy,csv,data\n1,2,3")

            yield project_path

    def test_environment_and_deployment_integration(self, temp_project_dir):
        """Test integration between environment management and deployment configuration."""
        # Initialize environment manager
        env_manager = EnvironmentManager(
            config_dir=str(temp_project_dir / "config"),
            environment=EnvironmentType.PRODUCTION,
            encryption_enabled=True
        )

        # Create environment files
        for env_type in EnvironmentType:
            result = env_manager.create_environment_file(env_type, template=True)
            assert result is True

        # Initialize deployment config
        deploy_config = DeploymentConfig(
            config_path=str(temp_project_dir / "render.yaml"),
            project_path=str(temp_project_dir)
        )

        # Create production deployment configuration
        prod_env = deploy_config.create_production_config()
        assert prod_env.name == "production"
        assert len(prod_env.services) > 0

        # Validate deployment configuration
        validation = deploy_config.validate_config(prod_env)

        # Should have some warnings but no critical errors
        assert len(validation["errors"]) == 0

        # Save deployment configuration
        result = deploy_config.save_config(prod_env)
        assert result is True

        # Check that render.yaml was created
        render_file = temp_project_dir / "render.yaml"
        assert render_file.exists()

    def test_logging_and_performance_integration(self, temp_project_dir):
        """Test integration between production logging and performance monitoring."""
        # Initialize production logger
        prod_logger = ProductionLogger(
            log_dir=str(temp_project_dir / "logs"),
            metrics_enabled=True,
            alerts_enabled=True,
            structured_logging=True
        )

        # Initialize performance monitor
        perf_monitor = PerformanceMonitor(
            collection_interval=1,
            enable_system_monitoring=False  # Disable for testing
        )

        # Record performance metrics and logs
        perf_monitor.record_api_request("GET", "/api/test", 200, 0.15, "user123")
        prod_logger.log_request("GET", "/api/test", 200, 0.15, "user123")

        perf_monitor.record_database_operation("SELECT", "users", 0.05, 10, 0.01)
        prod_logger.log_database_operation("SELECT", "users", 0.05, 10)

        # Simulate an error
        try:
            raise ValueError("Test integration error")
        except Exception as e:
            prod_logger.log_error(e, {"component": "integration_test"})

        # Check that both systems recorded metrics
        perf_metrics = perf_monitor.collector.get_metrics()
        assert len(perf_metrics) > 0

        # Check log files were created
        log_files = list((temp_project_dir / "logs").glob("*.log"))
        assert len(log_files) > 0

        # Get health status from logger
        health_status = prod_logger.get_health_status()
        assert "status" in health_status
        assert "components" in health_status

        # Get performance summary
        perf_summary = perf_monitor.get_performance_summary()
        assert perf_summary["total_metrics"] > 0

    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, temp_project_dir):
        """Test integration with health monitoring system."""
        # Set up environment variables for health checks
        os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
        os.environ["PORT"] = "8000"
        os.environ["HOST"] = "localhost"

        # Initialize health monitor
        health_monitor = HealthMonitor(
            check_interval=1,
            enable_circuit_breakers=True
        )

        # Initialize performance monitor for health metrics
        perf_monitor = PerformanceMonitor(enable_system_monitoring=False)

        try:
            # Run health checks
            results = await health_monitor.run_all_health_checks()
            assert len(results) > 0

            # Check overall health status
            overall_status = health_monitor.get_overall_health_status()
            assert "status" in overall_status
            assert "checks" in overall_status

            # Record health metrics in performance monitor
            for check_name, result in results.items():
                perf_monitor.collector.record_metric(
                    f"health.{check_name}.response_time",
                    result.response_time,
                    MetricType.TIMER,
                    PerformanceCategory.SYSTEM,
                    tags={"status": result.status.value}
                )

            # Check that health metrics were recorded
            health_metrics = perf_monitor.collector.get_metrics(name_filter="health")
            assert len(health_metrics) > 0

            # Test circuit breaker integration
            circuit_breakers = health_monitor.circuit_breakers
            assert len(circuit_breakers) > 0

        finally:
            # Clean up environment variables
            for key in ["DATABASE_URL", "PORT", "HOST"]:
                if key in os.environ:
                    del os.environ[key]

    def test_git_lfs_and_deployment_integration(self, temp_project_dir):
        """Test integration between Git LFS management and deployment."""
        # Initialize Git LFS manager
        lfs_manager = GitLFSManager(str(temp_project_dir))

        # Initialize deployment config
        deploy_config = DeploymentConfig(project_path=str(temp_project_dir))

        # Create LFS configuration
        result = lfs_manager.initialize_lfs()
        assert result is True

        # Configure CSV tracking
        result = lfs_manager.configure_csv_tracking()
        assert result is True

        # Get LFS health status
        lfs_health = lfs_manager.health_check()
        assert "overall_health" in lfs_health

        # Create deployment scripts that include LFS checks
        result = deploy_config.create_deployment_scripts()
        assert result is True

        # Check that deployment scripts directory exists
        scripts_dir = temp_project_dir / "scripts"
        assert scripts_dir.exists()

        # Check pre-deploy script includes LFS checks
        pre_deploy_script = scripts_dir / "pre_deploy_checks.py"
        assert pre_deploy_script.exists()

        script_content = pre_deploy_script.read_text()
        assert "GitLFSManager" in script_content

    def test_full_infrastructure_stack(self, temp_project_dir):
        """Test the complete infrastructure stack working together."""
        # 1. Environment Management
        env_manager = EnvironmentManager(
            config_dir=str(temp_project_dir / "config"),
            environment=EnvironmentType.PRODUCTION
        )

        # Create environment configurations
        for env_type in EnvironmentType:
            env_manager.create_environment_file(env_type, template=True)

        # 2. Production Logging
        prod_logger = ProductionLogger(
            log_dir=str(temp_project_dir / "logs"),
            metrics_enabled=True,
            alerts_enabled=True
        )

        # 3. Performance Monitoring
        perf_monitor = PerformanceMonitor(
            collection_interval=1,
            enable_system_monitoring=False
        )

        # 4. Health Monitoring
        health_monitor = HealthMonitor(check_interval=2)

        # 5. Deployment Configuration
        deploy_config = DeploymentConfig(project_path=str(temp_project_dir))

        # 6. Git LFS Management
        lfs_manager = GitLFSManager(str(temp_project_dir))

        try:
            # Initialize Git LFS
            lfs_manager.initialize_lfs()
            lfs_manager.configure_csv_tracking()

            # Start monitoring systems
            perf_monitor.start_monitoring()

            # Simulate application activity
            # API requests
            for i in range(5):
                perf_monitor.record_api_request(
                    "GET", f"/api/test/{i}", 200, 0.1 + i * 0.05, f"user{i}"
                )
                prod_logger.log_request(
                    "GET", f"/api/test/{i}", 200, 0.1 + i * 0.05, f"user{i}"
                )

            # Database operations
            for operation in ["SELECT", "INSERT", "UPDATE"]:
                perf_monitor.record_database_operation(
                    operation, "test_table", 0.02 + hash(operation) % 10 / 100
                )
                prod_logger.log_database_operation(
                    operation, "test_table", 0.02, 1
                )

            # Wait for metrics collection
            time.sleep(2)

            # Create deployment configuration
            prod_env = deploy_config.create_production_config()
            prod_env = deploy_config.optimize_for_production(prod_env)

            # Validate all components
            # 1. Environment validation
            env_validation = env_manager.validate_environment()

            # 2. Deployment validation
            deploy_validation = deploy_config.validate_config(prod_env)
            assert len(deploy_validation["errors"]) == 0

            # 3. Performance summary
            perf_summary = perf_monitor.get_performance_summary()
            assert perf_summary["total_metrics"] > 0

            # 4. Health status
            prod_health = prod_logger.get_health_status()
            assert "status" in prod_health

            # 5. LFS health
            lfs_health = lfs_manager.health_check()
            assert "overall_health" in lfs_health

            # Export all configurations and reports
            # Export environment config
            env_export_path = temp_project_dir / "environment_config.json"
            env_manager.export_configuration(str(env_export_path))
            assert env_export_path.exists()

            # Export performance metrics
            perf_export_path = temp_project_dir / "performance_metrics.json"
            perf_monitor.export_metrics(str(perf_export_path))
            assert perf_export_path.exists()

            # Export logs
            log_export_path = temp_project_dir / "exported_logs.json"
            prod_logger.export_logs(str(log_export_path))
            assert log_export_path.exists()

            # Save deployment configuration
            deploy_config.save_config(prod_env)
            render_file = temp_project_dir / "render.yaml"
            assert render_file.exists()

            # Create comprehensive status report
            status_report = {
                "timestamp": time.time(),
                "environment": {
                    "current": env_manager.current_environment.value,
                    "validation": {
                        "valid": env_validation.is_valid,
                        "missing_vars": len(env_validation.missing_variables),
                        "warnings": len(env_validation.warnings)
                    }
                },
                "deployment": {
                    "environment": prod_env.name,
                    "services": len(prod_env.services),
                    "databases": len(prod_env.databases),
                    "validation_errors": len(deploy_validation["errors"]),
                    "validation_warnings": len(deploy_validation["warnings"])
                },
                "performance": {
                    "total_metrics": perf_summary["total_metrics"],
                    "active_alerts": perf_summary["active_alerts"],
                    "monitoring_active": perf_summary["monitoring_active"]
                },
                "logging": {
                    "health_status": prod_health["status"],
                    "components": list(prod_health["components"].keys())
                },
                "git_lfs": {
                    "health": lfs_health["overall_health"],
                    "tracked_files": len(lfs_health.get("tracked_files", []))
                }
            }

            # Save status report
            status_report_path = temp_project_dir / "infrastructure_status.json"
            with open(status_report_path, 'w') as f:
                json.dump(status_report, f, indent=2, default=str)

            # Verify comprehensive setup
            assert status_report["deployment"]["services"] > 0
            assert status_report["performance"]["total_metrics"] > 0
            assert status_report["git_lfs"]["health"] in ["excellent", "good", "fair", "poor"]

        finally:
            # Cleanup
            perf_monitor.stop_monitoring()
            health_monitor.stop_monitoring()

    def test_error_handling_integration(self, temp_project_dir):
        """Test error handling across infrastructure components."""
        # Initialize components with potentially failing configurations
        prod_logger = ProductionLogger(
            log_dir=str(temp_project_dir / "logs"),
            structured_logging=True
        )

        perf_monitor = PerformanceMonitor()

        # Test error logging and performance tracking
        errors_to_test = [
            ValueError("Test validation error"),
            ConnectionError("Test connection error"),
            TimeoutError("Test timeout error"),
            RuntimeError("Test runtime error")
        ]

        for error in errors_to_test:
            try:
                raise error
            except Exception as e:
                # Log error in production logger
                prod_logger.log_error(e, {
                    "test_context": "integration_test",
                    "error_type": type(e).__name__
                })

                # Track error in performance monitor
                perf_monitor.collector.record_metric(
                    "integration.errors",
                    1,
                    MetricType.COUNTER,
                    PerformanceCategory.APPLICATION,
                    tags={"error_type": type(e).__name__}
                )

        # Check that all errors were properly logged and tracked
        error_metrics = perf_monitor.collector.get_metrics(name_filter="errors")
        assert len(error_metrics) >= len(errors_to_test)

        # Check log files contain error information
        log_files = list((temp_project_dir / "logs").glob("*.log"))
        assert len(log_files) > 0

        # Check error log specifically
        error_log = temp_project_dir / "logs" / "error.log"
        if error_log.exists():
            error_content = error_log.read_text()
            assert len(error_content) > 0

    def test_configuration_consistency(self, temp_project_dir):
        """Test configuration consistency across infrastructure components."""
        # Initialize all components with consistent configuration
        base_config = {
            "environment": EnvironmentType.PRODUCTION,
            "log_level": "INFO",
            "monitoring_enabled": True,
            "encryption_enabled": True
        }

        # Environment Manager
        env_manager = EnvironmentManager(
            config_dir=str(temp_project_dir / "config"),
            environment=base_config["environment"],
            encryption_enabled=base_config["encryption_enabled"]
        )

        # Production Logger
        prod_logger = ProductionLogger(
            log_dir=str(temp_project_dir / "logs"),
            metrics_enabled=base_config["monitoring_enabled"],
            structured_logging=True
        )

        # Performance Monitor
        perf_monitor = PerformanceMonitor(
            enable_system_monitoring=base_config["monitoring_enabled"]
        )

        # Health Monitor
        health_monitor = HealthMonitor()

        # Verify consistent configuration
        assert env_manager.current_environment == base_config["environment"]
        assert env_manager.encryption_enabled == base_config["encryption_enabled"]
        assert prod_logger.metrics_enabled == base_config["monitoring_enabled"]
        assert perf_monitor.enable_system_monitoring == base_config["monitoring_enabled"]

        # Test configuration export and validation
        config_summary = env_manager.get_configuration_summary()
        assert config_summary["environment"] == base_config["environment"].value
        assert config_summary["encryption_enabled"] == base_config["encryption_enabled"]

        perf_summary = perf_monitor.get_performance_summary()
        assert "monitoring_active" in perf_summary

        health_status = health_monitor.get_overall_health_status()
        assert "status" in health_status


@pytest.mark.asyncio
class TestAsyncInfrastructureIntegration:
    """Async integration tests for infrastructure components."""

    async def test_async_health_and_performance_integration(self):
        """Test async integration between health monitoring and performance tracking."""
        health_monitor = HealthMonitor()
        perf_monitor = PerformanceMonitor(enable_system_monitoring=False)

        # Run health checks
        results = await health_monitor.run_all_health_checks()

        # Record health check performance
        for check_name, result in results.items():
            perf_monitor.collector.record_metric(
                f"health_check.{check_name}.duration",
                result.response_time,
                MetricType.TIMER,
                PerformanceCategory.SYSTEM,
                tags={
                    "status": result.status.value,
                    "check_type": "health"
                }
            )

        # Verify metrics were recorded
        health_metrics = perf_monitor.collector.get_metrics(name_filter="health_check")
        assert len(health_metrics) > 0

        # Check performance summary
        summary = perf_monitor.get_performance_summary()
        assert summary["total_metrics"] > 0


if __name__ == "__main__":
    pytest.main([__file__])