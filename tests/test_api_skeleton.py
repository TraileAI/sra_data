"""
Tests for FastAPI skeleton application.

These tests ensure the minimal API endpoints work correctly for deployment.
"""

import pytest
import json
import time
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from packages.sra_data.api.skeleton import create_fastapi_app

# Create test client
app = create_fastapi_app()
client = TestClient(app)

class TestBasicEndpoints:
    """Test basic API endpoints."""

    def test_root_endpoint_returns_service_info(self):
        """Test root endpoint returns basic service information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert data["service"] == "SRA Data Processing Service"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert data["deployment"] == "render.com"
        assert "/" in data["endpoints"]
        assert "/health" in data["endpoints"]
        assert "/status" in data["endpoints"]

    def test_health_endpoint_returns_healthy_status(self):
        """Test health endpoint returns healthy status."""
        start_time = time.time()
        response = client.get("/health")
        response_time = time.time() - start_time

        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "SRA Data Processing Service"
        assert data["version"] == "1.0.0"

    def test_status_endpoint_returns_detailed_info(self):
        """Test status endpoint returns detailed service information."""
        start_time = time.time()
        response = client.get("/status")
        response_time = time.time() - start_time

        assert response.status_code == 200
        assert response_time < 2.0  # Should respond within 2 seconds

        data = response.json()
        assert data["service"] == "SRA Data Processing Service"
        assert data["version"] == "1.0.0"
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "uptime_human" in data
        assert isinstance(data["uptime_seconds"], int)
        assert data["uptime_seconds"] >= 0

    def test_nonexistent_endpoint_returns_404(self):
        """Test that nonexistent endpoints return 404."""
        response = client.get("/nonexistent-endpoint")

        assert response.status_code == 404

    def test_cors_headers_present(self):
        """Test that CORS headers are present for cross-origin requests."""
        response = client.options("/health")

        assert response.status_code == 200
        headers = response.headers

        # Check for CORS headers (FastAPI CORS middleware should add these)
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers

class TestStatusEndpointDetails:
    """Test detailed status endpoint functionality."""

    @patch('packages.sra_data.api.skeleton._get_database_status')
    async def test_status_with_healthy_database(self, mock_db_status):
        """Test status endpoint with healthy database."""
        mock_db_status.return_value = {
            "status": "connected",
            "connected": True,
            "pool_info": {"active_connections": 3, "pool_size": 10}
        }

        response = client.get("/status")
        data = response.json()

        assert response.status_code == 200
        assert "database" in data

    @patch('packages.sra_data.api.skeleton._get_database_status')
    async def test_status_with_database_error(self, mock_db_status):
        """Test status endpoint when database is unavailable."""
        mock_db_status.side_effect = Exception("Database connection failed")

        response = client.get("/status")
        data = response.json()

        assert response.status_code == 200
        assert "database" in data
        # Should still return 200 but with degraded status

    @patch('packages.sra_data.api.skeleton._get_data_services_status')
    async def test_status_with_data_services(self, mock_services_status):
        """Test status endpoint with data services information."""
        mock_services_status.return_value = {
            "fmp_integration": {
                "status": "available",
                "service_type": "equity_data_processing"
            },
            "fundata_processing": {
                "status": "available",
                "service_type": "csv_data_processing"
            }
        }

        response = client.get("/status")
        data = response.json()

        assert response.status_code == 200
        assert "data_services" in data

class TestPerformanceAndReliability:
    """Test performance and reliability requirements."""

    def test_health_endpoint_performance(self):
        """Test health endpoint meets performance requirements."""
        response_times = []

        for _ in range(5):
            start_time = time.time()
            response = client.get("/health")
            response_time = time.time() - start_time

            assert response.status_code == 200
            response_times.append(response_time)

        # Average response time should be under 0.5 seconds
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 0.5

    def test_concurrent_health_checks(self):
        """Test health endpoint handles concurrent requests."""
        import concurrent.futures

        def make_request():
            return client.get("/health")

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200

    def test_api_stability_over_time(self):
        """Test API stability over multiple requests."""
        for i in range(20):
            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"

class TestDeploymentReadiness:
    """Test deployment-specific requirements."""

    def test_all_required_endpoints_available(self):
        """Test that all required endpoints are available."""
        required_endpoints = ["/", "/health", "/status"]

        for endpoint in required_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200

    def test_json_response_format(self):
        """Test that all endpoints return valid JSON."""
        endpoints = ["/", "/health", "/status"]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"

            # Verify valid JSON
            try:
                json.loads(response.content)
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON response from {endpoint}")

    def test_deployment_port_compatibility(self):
        """Test that the server configuration is compatible with Render.com."""
        from packages.sra_data.api.skeleton import app

        # Verify app is configured correctly
        assert app.title == "SRA Data Processing Service"
        assert app.version == "1.0.0"

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_request_handling(self):
        """Test handling of malformed requests."""
        # Test with invalid HTTP method on health endpoint
        response = client.patch("/health")
        assert response.status_code == 405  # Method not allowed

    def test_large_concurrent_load(self):
        """Test handling of large number of concurrent requests."""
        import concurrent.futures

        def make_health_request():
            return client.get("/health")

        # Make 50 concurrent requests to test stability
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_health_request) for _ in range(50)]
            responses = [future.result() for future in futures]

        # Verify all requests succeeded
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count >= 45  # Allow for some potential timeouts

    def test_uptime_calculation_accuracy(self):
        """Test that uptime calculation is accurate."""
        # Get initial status
        response1 = client.get("/status")
        data1 = response1.json()
        uptime1 = data1["uptime_seconds"]

        # Wait briefly
        time.sleep(1)

        # Get status again
        response2 = client.get("/status")
        data2 = response2.json()
        uptime2 = data2["uptime_seconds"]

        # Uptime should have increased
        assert uptime2 > uptime1
        assert (uptime2 - uptime1) >= 1