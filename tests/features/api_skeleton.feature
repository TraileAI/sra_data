Feature: Minimal FastAPI Skeleton for Deployment Stability
  As a deployment service on Render.com
  I want minimal API endpoints that respond with health status
  So that the service remains active and doesn't get suspended

  Scenario: Health check endpoint returns service status
    Given the FastAPI skeleton application is running
    When I send a GET request to "/health"
    Then I should receive a 200 OK response
    And the response should contain service status information
    And the response time should be under 1 second
    And the response should include "status": "healthy"

  Scenario: Root endpoint provides basic service information
    Given the FastAPI skeleton application is running
    When I send a GET request to "/"
    Then I should receive a 200 OK response
    And the response should contain basic service information
    And the response should include service name and version
    And the response should include "service": "SRA Data Processing"

  Scenario: Service status endpoint provides detailed monitoring
    Given the FastAPI skeleton application is running
    When I send a GET request to "/status"
    Then I should receive a 200 OK response
    And the response should contain database connection status
    And the response should contain data processing service status
    And the response should include last successful data refresh timestamp
    And the response should include system uptime information

  Scenario: API handles invalid endpoints gracefully
    Given the FastAPI skeleton application is running
    When I send a GET request to "/nonexistent-endpoint"
    Then I should receive a 404 Not Found response
    And the response should contain proper error information
    And the response should maintain service stability

  Scenario: FastAPI skeleton supports CORS for deployment
    Given the FastAPI skeleton application is running
    When I send an OPTIONS request to "/health"
    Then I should receive appropriate CORS headers
    And the service should handle cross-origin requests properly
    And deployment infrastructure should work correctly