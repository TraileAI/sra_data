Feature: Database Infrastructure for Data Processing
  As a data processing service
  I want reliable database connections and schema management
  So that I can efficiently store raw data and create client views

  Scenario: Database connection pool initialization for data processing
    Given the database configuration is valid
    When I initialize the connection pool for background processing
    Then the pool should be created with proper parameters
    And connections should be optimized for bulk operations

  Scenario: Schema initialization with raw tables and view preparation
    Given a fresh database without tables
    When I run schema initialization for data processing
    Then all required raw data tables should be created (equity_profile, fundata_data, fundata_quotes)
    And indexes should be optimized for data ingestion
    And fundata tables should be indexed by Identifier field
    And the system should be ready for modelized view creation

  Scenario: Connection pool resilience during data processing
    Given the connection pool is active during background processing
    When temporary database connectivity issues occur
    Then the pool should handle reconnections gracefully
    And data processing operations should resume without data loss