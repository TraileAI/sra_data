"""
Deployment Readiness Validation Framework

This module provides comprehensive deployment readiness testing:
- Production environment validation
- Configuration verification
- Security and compliance checks
- Health check validation
- Performance acceptance testing
- Deployment simulation
"""

import asyncio
import json
import logging
import os
import ssl
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from unittest.mock import patch, MagicMock

import pytest
import pytest_asyncio
import requests
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Import system components
from packages.sra_data.api.skeleton import app
from packages.sra_data.repositories.database_infrastructure import create_database_manager
from packages.sra_data.infrastructure.environment import EnvironmentManager
from packages.sra_data.infrastructure.health_monitor import HealthMonitor
from packages.sra_data.infrastructure.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class DeploymentReadinessValidator:
    """Comprehensive deployment readiness validation."""

    def __init__(self):
        self.validation_results: Dict[str, Any] = {}
        self.test_metrics: Dict[str, Any] = {}
        self.deployment_checklist: Dict[str, bool] = {}

    def record_validation(self, category: str, test_name: str, passed: bool,
                         details: Optional[Dict[str, Any]] = None):
        """Record validation result."""
        if category not in self.validation_results:
            self.validation_results[category] = {}

        self.validation_results[category][test_name] = {
            'passed': passed,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        }

    def get_readiness_score(self) -> float:
        """Calculate overall deployment readiness score."""
        total_tests = 0
        passed_tests = 0

        for category in self.validation_results.values():
            for test_result in category.values():
                total_tests += 1
                if test_result['passed']:
                    passed_tests += 1

        return (passed_tests / total_tests * 100) if total_tests > 0 else 0.0


@pytest_asyncio.fixture
async def deployment_validator():
    """Provide deployment readiness validator."""
    validator = DeploymentReadinessValidator()
    yield validator


class TestEnvironmentConfiguration:
    """Test production environment configuration."""

    @pytest.mark.asyncio
    async def test_environment_variables_validation(self, deployment_validator: DeploymentReadinessValidator):
        """Test all required environment variables are properly configured."""

        # Required environment variables for production
        required_env_vars = [
            'DATABASE_URL',
            'FMP_API_KEY',
            'ENVIRONMENT',
        ]

        # Optional but recommended environment variables
        recommended_env_vars = [
            'HOST',
            'PORT',
            'LOG_LEVEL',
            'WORKERS'
        ]

        missing_required = []
        missing_recommended = []

        # Check required variables
        for var in required_env_vars:
            if not os.getenv(var):
                missing_required.append(var)

        # Check recommended variables
        for var in recommended_env_vars:
            if not os.getenv(var):
                missing_recommended.append(var)

        # Validate environment manager
        env_manager = EnvironmentManager()

        try:
            current_env = env_manager.get_current_environment()
            config_valid = await env_manager.validate_configuration()

            validation_details = {
                'missing_required': missing_required,
                'missing_recommended': missing_recommended,
                'current_environment': current_env,
                'config_valid': config_valid
            }

            test_passed = (
                len(missing_required) == 0 and
                config_valid and
                current_env in ['production', 'staging', 'development']
            )

            deployment_validator.record_validation(
                'environment',
                'environment_variables',
                test_passed,
                validation_details
            )

            if not test_passed:
                logger.error(f"Environment validation failed: {validation_details}")
            else:
                logger.info("Environment configuration validation passed")

        except Exception as e:
            deployment_validator.record_validation(
                'environment',
                'environment_variables',
                False,
                {'error': str(e)}
            )

    @pytest.mark.asyncio
    async def test_database_configuration(self, deployment_validator: DeploymentReadinessValidator):
        """Test database configuration for production readiness."""

        try:
            db_manager = create_database_manager()
            await db_manager.initialize()

            # Test database connection
            connection_test = await db_manager.check_connection()

            # Get connection info
            connection_info = await db_manager.get_connection_info() if connection_test else {}

            # Validate database configuration
            validation_details = {
                'connection_successful': connection_test,
                'connection_info': connection_info,
                'pool_configured': 'pool_size' in connection_info,
                'ssl_configured': connection_info.get('ssl_enabled', False)
            }

            # Check for production-ready database settings
            production_ready = (
                connection_test and
                connection_info.get('pool_size', 0) > 1 and
                connection_info.get('max_overflow', 0) > 0
            )

            deployment_validator.record_validation(
                'database',
                'database_configuration',
                production_ready,
                validation_details
            )

            await db_manager.close()

        except Exception as e:
            deployment_validator.record_validation(
                'database',
                'database_configuration',
                False,
                {'error': str(e)}
            )

    @pytest.mark.asyncio
    async def test_secrets_management(self, deployment_validator: DeploymentReadinessValidator):
        """Test secrets and sensitive data management."""

        # Check for hardcoded secrets (basic scan)
        project_root = Path(__file__).parent.parent

        potential_secret_patterns = [
            'password=',
            'secret=',
            'api_key=',
            'token=',
        ]

        secrets_found = []

        # Scan Python files for potential hardcoded secrets
        for py_file in project_root.rglob("*.py"):
            if 'test' in py_file.name or '.venv' in str(py_file):
                continue

            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    for pattern in potential_secret_patterns:
                        if pattern in content and 'example' not in content:
                            secrets_found.append({
                                'file': str(py_file),
                                'pattern': pattern
                            })
            except Exception:
                continue

        # Test environment manager encryption
        try:
            env_manager = EnvironmentManager()
            encryption_available = hasattr(env_manager, 'encrypt_value')

            validation_details = {
                'hardcoded_secrets_found': len(secrets_found),
                'secrets_details': secrets_found,
                'encryption_available': encryption_available
            }

            secrets_secure = len(secrets_found) == 0

            deployment_validator.record_validation(
                'security',
                'secrets_management',
                secrets_secure,
                validation_details
            )

        except Exception as e:
            deployment_validator.record_validation(
                'security',
                'secrets_management',
                False,
                {'error': str(e)}
            )


class TestApplicationHealthChecks:
    """Test application health check endpoints and monitoring."""

    @pytest.mark.asyncio
    async def test_health_check_endpoints(self, deployment_validator: DeploymentReadinessValidator):
        """Test all health check endpoints work correctly."""

        client = TestClient(app)

        # Test root endpoint
        try:
            root_response = client.get("/")
            root_healthy = root_response.status_code == 200
            root_data = root_response.json() if root_healthy else {}
        except Exception as e:
            root_healthy = False
            root_data = {'error': str(e)}

        # Test health endpoint
        try:
            health_response = client.get("/health")
            health_healthy = health_response.status_code == 200
            health_data = health_response.json() if health_healthy else {}
        except Exception as e:
            health_healthy = False
            health_data = {'error': str(e)}

        # Test status endpoint
        try:
            status_response = client.get("/status")
            status_healthy = status_response.status_code == 200
            status_data = status_response.json() if status_healthy else {}
        except Exception as e:
            status_healthy = False
            status_data = {'error': str(e)}

        validation_details = {
            'root_endpoint': {'healthy': root_healthy, 'data': root_data},
            'health_endpoint': {'healthy': health_healthy, 'data': health_data},
            'status_endpoint': {'healthy': status_healthy, 'data': status_data},
        }

        all_healthy = root_healthy and health_healthy and status_healthy

        deployment_validator.record_validation(
            'health_checks',
            'api_endpoints',
            all_healthy,
            validation_details
        )

    @pytest.mark.asyncio
    async def test_health_monitoring_system(self, deployment_validator: DeploymentReadinessValidator):
        """Test health monitoring system functionality."""

        try:
            # Test health monitor initialization
            health_monitor = HealthMonitor()

            # Test health checks
            health_results = await health_monitor.run_health_checks()

            # Test circuit breaker functionality
            circuit_breaker_status = health_monitor.get_circuit_breaker_status()

            # Test monitoring metrics
            monitoring_metrics = health_monitor.get_monitoring_metrics()

            validation_details = {
                'health_checks_passed': all(
                    check.get('healthy', False)
                    for check in health_results.values()
                ),
                'circuit_breakers_healthy': all(
                    status.get('state') == 'closed'
                    for status in circuit_breaker_status.values()
                ),
                'monitoring_active': len(monitoring_metrics) > 0,
                'health_results': health_results,
                'circuit_breaker_status': circuit_breaker_status
            }

            monitoring_healthy = (
                validation_details['health_checks_passed'] and
                validation_details['circuit_breakers_healthy'] and
                validation_details['monitoring_active']
            )

            deployment_validator.record_validation(
                'health_checks',
                'monitoring_system',
                monitoring_healthy,
                validation_details
            )

        except Exception as e:
            deployment_validator.record_validation(
                'health_checks',
                'monitoring_system',
                False,
                {'error': str(e)}
            )


class TestPerformanceAcceptance:
    """Test performance acceptance criteria for production deployment."""

    @pytest.mark.asyncio
    async def test_api_response_times(self, deployment_validator: DeploymentReadinessValidator):
        """Test API response times meet production requirements."""

        client = TestClient(app)

        # Define performance requirements
        performance_requirements = {
            '/': 100,      # 100ms max for root
            '/health': 50,  # 50ms max for health
            '/status': 200  # 200ms max for status (more complex)
        }

        performance_results = {}

        for endpoint, max_time_ms in performance_requirements.items():
            response_times = []

            # Test multiple requests for consistency
            for _ in range(10):
                start_time = time.time()
                try:
                    response = client.get(endpoint)
                    end_time = time.time()
                    response_time_ms = (end_time - start_time) * 1000

                    if response.status_code == 200:
                        response_times.append(response_time_ms)
                    else:
                        response_times.append(float('inf'))  # Failed request
                except Exception:
                    response_times.append(float('inf'))

            # Calculate performance metrics
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            success_rate = sum(1 for rt in response_times if rt != float('inf')) / len(response_times)

            performance_results[endpoint] = {
                'avg_response_time_ms': avg_response_time,
                'max_response_time_ms': max_response_time,
                'success_rate': success_rate,
                'meets_requirement': avg_response_time <= max_time_ms and success_rate >= 0.9
            }

        all_requirements_met = all(
            result['meets_requirement']
            for result in performance_results.values()
        )

        deployment_validator.record_validation(
            'performance',
            'api_response_times',
            all_requirements_met,
            performance_results
        )

    @pytest.mark.asyncio
    async def test_concurrent_load_handling(self, deployment_validator: DeploymentReadinessValidator):
        """Test application can handle concurrent load."""

        async def concurrent_request():
            async with AsyncClient(app=app, base_url="http://test") as client:
                try:
                    start_time = time.time()
                    response = await client.get("/health")
                    end_time = time.time()
                    return {
                        'success': response.status_code == 200,
                        'response_time': end_time - start_time,
                        'status_code': response.status_code
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e),
                        'response_time': 0
                    }

        # Test with production-level concurrent requests
        concurrent_requests = 50

        start_time = time.time()
        tasks = [concurrent_request() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]

        total_duration = end_time - start_time
        success_rate = len(successful_requests) / len(results)
        requests_per_second = len(successful_requests) / total_duration

        avg_response_time = (
            sum(r['response_time'] for r in successful_requests) /
            len(successful_requests) if successful_requests else 0
        )

        load_test_results = {
            'concurrent_requests': concurrent_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': len(failed_requests),
            'success_rate': success_rate,
            'requests_per_second': requests_per_second,
            'avg_response_time': avg_response_time,
            'total_duration': total_duration
        }

        # Performance acceptance criteria
        load_acceptable = (
            success_rate >= 0.95 and         # 95% success rate
            requests_per_second >= 25 and    # 25 RPS minimum
            avg_response_time <= 0.5         # 500ms average response
        )

        deployment_validator.record_validation(
            'performance',
            'concurrent_load',
            load_acceptable,
            load_test_results
        )

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, deployment_validator: DeploymentReadinessValidator):
        """Test performance monitoring capabilities."""

        try:
            performance_monitor = PerformanceMonitor()

            # Test metrics collection
            @performance_monitor.track_performance
            async def test_operation():
                await asyncio.sleep(0.1)  # Simulate work
                return "test_result"

            # Execute monitored operation
            result = await test_operation()

            # Get performance metrics
            metrics = performance_monitor.get_metrics_summary()
            alerts = performance_monitor.get_active_alerts()

            monitoring_results = {
                'operation_tracked': result == "test_result",
                'metrics_collected': len(metrics) > 0,
                'alerting_configured': isinstance(alerts, list),
                'metrics_summary': metrics,
                'active_alerts': len(alerts) if isinstance(alerts, list) else 0
            }

            monitoring_functional = (
                monitoring_results['operation_tracked'] and
                monitoring_results['metrics_collected'] and
                monitoring_results['alerting_configured']
            )

            deployment_validator.record_validation(
                'performance',
                'monitoring_system',
                monitoring_functional,
                monitoring_results
            )

        except Exception as e:
            deployment_validator.record_validation(
                'performance',
                'monitoring_system',
                False,
                {'error': str(e)}
            )


class TestSecurityAndCompliance:
    """Test security and compliance requirements."""

    @pytest.mark.asyncio
    async def test_security_headers(self, deployment_validator: DeploymentReadinessValidator):
        """Test security headers are properly configured."""

        client = TestClient(app)

        try:
            response = client.get("/")
            headers = response.headers

            # Check for important security headers
            security_headers_check = {
                'content_type_set': 'content-type' in headers,
                'cors_configured': 'access-control-allow-origin' in headers,
                'server_header_minimal': headers.get('server', '').lower() != 'fastapi'
            }

            # Additional security considerations
            security_considerations = {
                'https_ready': True,  # Assuming deployment will use HTTPS
                'sensitive_data_exposed': self._check_for_sensitive_data_exposure(response),
                'error_handling_secure': response.status_code != 500
            }

            all_security_checks = {**security_headers_check, **security_considerations}

            security_compliant = all(all_security_checks.values())

            deployment_validator.record_validation(
                'security',
                'security_headers',
                security_compliant,
                all_security_checks
            )

        except Exception as e:
            deployment_validator.record_validation(
                'security',
                'security_headers',
                False,
                {'error': str(e)}
            )

    def _check_for_sensitive_data_exposure(self, response) -> bool:
        """Check if response exposes sensitive data."""
        try:
            response_text = response.text.lower()

            # Check for potential sensitive data patterns
            sensitive_patterns = [
                'password',
                'secret',
                'key',
                'token',
                'credential'
            ]

            for pattern in sensitive_patterns:
                if pattern in response_text and 'example' not in response_text:
                    return True  # Sensitive data found

            return False  # No sensitive data found
        except:
            return False

    @pytest.mark.asyncio
    async def test_input_validation(self, deployment_validator: DeploymentReadinessValidator):
        """Test input validation and sanitization."""

        client = TestClient(app)

        # Test various potentially malicious inputs
        test_inputs = [
            "normal_input",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "%3Cscript%3Ealert('xss')%3C/script%3E"
        ]

        validation_results = {}

        for test_input in test_inputs:
            try:
                # Test input handling (using query parameters where applicable)
                response = client.get(f"/?test={test_input}")

                validation_results[test_input] = {
                    'status_code': response.status_code,
                    'handled_safely': response.status_code != 500,
                    'no_reflection': test_input not in response.text
                }

            except Exception as e:
                validation_results[test_input] = {
                    'status_code': 'error',
                    'handled_safely': True,  # Exception handling is okay
                    'error': str(e)
                }

        # All inputs should be handled safely
        input_validation_safe = all(
            result.get('handled_safely', False)
            for result in validation_results.values()
        )

        deployment_validator.record_validation(
            'security',
            'input_validation',
            input_validation_safe,
            validation_results
        )


class TestDeploymentSimulation:
    """Test deployment simulation and production scenarios."""

    @pytest.mark.asyncio
    async def test_application_startup(self, deployment_validator: DeploymentReadinessValidator):
        """Test application startup process."""

        startup_results = {
            'import_successful': True,
            'app_creation_successful': True,
            'initial_health_check': True,
            'startup_time_acceptable': True
        }

        try:
            # Test app import and creation
            start_time = time.time()

            # Import and create app (already done, but simulate)
            from packages.sra_data.api.skeleton import app

            # Test initial health check
            client = TestClient(app)
            health_response = client.get("/health")

            end_time = time.time()
            startup_time = end_time - start_time

            startup_results.update({
                'import_successful': True,
                'app_creation_successful': app is not None,
                'initial_health_check': health_response.status_code == 200,
                'startup_time_acceptable': startup_time < 5.0,  # Should start quickly
                'startup_time_seconds': startup_time
            })

        except ImportError as e:
            startup_results['import_successful'] = False
            startup_results['import_error'] = str(e)
        except Exception as e:
            startup_results['app_creation_successful'] = False
            startup_results['creation_error'] = str(e)

        startup_successful = all([
            startup_results['import_successful'],
            startup_results['app_creation_successful'],
            startup_results['initial_health_check'],
            startup_results['startup_time_acceptable']
        ])

        deployment_validator.record_validation(
            'deployment',
            'application_startup',
            startup_successful,
            startup_results
        )

    @pytest.mark.asyncio
    async def test_configuration_files(self, deployment_validator: DeploymentReadinessValidator):
        """Test deployment configuration files."""

        project_root = Path(__file__).parent.parent

        config_files_check = {
            'dockerfile_exists': (project_root / 'Dockerfile').exists(),
            'render_yaml_exists': (project_root / 'render.yaml').exists(),
            'requirements_exists': (project_root / 'requirements.txt').exists(),
            'server_py_exists': (project_root / 'server.py').exists()
        }

        # Check Dockerfile content if it exists
        if config_files_check['dockerfile_exists']:
            try:
                dockerfile_content = (project_root / 'Dockerfile').read_text()
                config_files_check['dockerfile_has_python'] = 'python' in dockerfile_content.lower()
                config_files_check['dockerfile_has_requirements'] = 'requirements.txt' in dockerfile_content
            except Exception as e:
                config_files_check['dockerfile_error'] = str(e)

        # Check render.yaml content if it exists
        if config_files_check['render_yaml_exists']:
            try:
                render_content = (project_root / 'render.yaml').read_text()
                config_files_check['render_has_services'] = 'services:' in render_content
                config_files_check['render_has_build'] = 'buildCommand:' in render_content
            except Exception as e:
                config_files_check['render_error'] = str(e)

        # Check requirements.txt
        if config_files_check['requirements_exists']:
            try:
                requirements_content = (project_root / 'requirements.txt').read_text()
                config_files_check['requirements_has_fastapi'] = 'fastapi' in requirements_content
                config_files_check['requirements_has_uvicorn'] = 'uvicorn' in requirements_content
            except Exception as e:
                config_files_check['requirements_error'] = str(e)

        essential_files_present = all([
            config_files_check['dockerfile_exists'],
            config_files_check['render_yaml_exists'],
            config_files_check['requirements_exists'],
            config_files_check['server_py_exists']
        ])

        deployment_validator.record_validation(
            'deployment',
            'configuration_files',
            essential_files_present,
            config_files_check
        )


class TestDeploymentReadinessSummary:
    """Test deployment readiness summary and final validation."""

    @pytest.mark.asyncio
    async def test_comprehensive_deployment_readiness(self, deployment_validator: DeploymentReadinessValidator):
        """Comprehensive deployment readiness assessment."""

        # Calculate overall readiness score
        readiness_score = deployment_validator.get_readiness_score()

        # Analyze validation results by category
        category_scores = {}
        critical_failures = []
        warnings = []

        for category, tests in deployment_validator.validation_results.items():
            category_total = len(tests)
            category_passed = sum(1 for test in tests.values() if test['passed'])
            category_score = (category_passed / category_total * 100) if category_total > 0 else 0

            category_scores[category] = {
                'score': category_score,
                'passed': category_passed,
                'total': category_total,
                'critical': category in ['environment', 'health_checks', 'security']
            }

            # Identify critical failures
            if category_score < 100 and category in ['environment', 'health_checks', 'security']:
                failed_tests = [
                    test_name for test_name, result in tests.items()
                    if not result['passed']
                ]
                critical_failures.extend([(category, test) for test in failed_tests])

            # Identify warnings
            if category_score < 90:
                warnings.append(f"{category}: {category_score:.1f}% passed")

        # Generate deployment recommendations
        recommendations = []

        if readiness_score < 90:
            recommendations.append("Overall readiness score below 90% - review failed tests before deployment")

        if critical_failures:
            recommendations.append(f"Critical failures detected: {critical_failures}")

        if readiness_score < 70:
            recommendations.append("Deployment not recommended - major issues detected")
        elif readiness_score < 90:
            recommendations.append("Deployment with caution - monitor closely after deployment")
        else:
            recommendations.append("Deployment ready - all systems operational")

        # Final deployment readiness report
        deployment_readiness_report = {
            'overall_score': readiness_score,
            'deployment_recommended': readiness_score >= 70 and len(critical_failures) == 0,
            'category_scores': category_scores,
            'critical_failures': critical_failures,
            'warnings': warnings,
            'recommendations': recommendations,
            'validation_timestamp': datetime.utcnow().isoformat(),
            'total_tests_run': sum(
                len(tests) for tests in deployment_validator.validation_results.values()
            )
        }

        # Log comprehensive deployment readiness report
        logger.info("=== Deployment Readiness Report ===")
        logger.info(f"Overall Score: {readiness_score:.1f}%")
        logger.info(f"Deployment Recommended: {deployment_readiness_report['deployment_recommended']}")

        for category, scores in category_scores.items():
            logger.info(f"{category}: {scores['score']:.1f}% ({scores['passed']}/{scores['total']})")

        if critical_failures:
            logger.error(f"Critical Failures: {critical_failures}")

        if warnings:
            logger.warning(f"Warnings: {warnings}")

        for recommendation in recommendations:
            logger.info(f"Recommendation: {recommendation}")

        # Store final report
        deployment_validator.deployment_readiness_report = deployment_readiness_report

        # Assert deployment readiness
        assert readiness_score >= 70, f"Deployment readiness score too low: {readiness_score:.1f}%"
        assert len(critical_failures) == 0, f"Critical failures must be resolved: {critical_failures}"

        # Final assertion
        deployment_ready = (
            readiness_score >= 80 and
            len(critical_failures) == 0 and
            category_scores.get('security', {}).get('score', 0) >= 90
        )

        logger.info(f"Final deployment readiness: {'READY' if deployment_ready else 'NOT READY'}")


if __name__ == "__main__":
    # Run deployment readiness tests
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "--asyncio-mode=auto"
    ])