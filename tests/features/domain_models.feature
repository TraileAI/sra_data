Feature: Domain Model Validation for Data Processing
  As a data processing service
  I want to validate all incoming data using Pydantic models
  So that only clean, structured data enters the database

  Scenario: Valid equity profile validation for ingestion
    Given I have valid equity profile data from FMP API
    When I create an EquityProfile model for database storage
    Then the model should validate successfully
    And all fields should be properly typed for raw storage

  Scenario: Invalid symbol format rejection during ingestion
    Given I have equity data with invalid symbol format
    When I attempt to create an EquityProfile model
    Then validation should fail with clear error message
    And the invalid data should be logged but not stored

  Scenario: Fundata data record validation for local CSV processing
    Given I have CSV data from local fundata/data/ directory (Git LFS)
    When I create FundataDataRecord models for batch ingestion
    Then identifier field should be properly validated and indexed
    And all optional fields should handle null values gracefully
    And records should be prepared for fundata_data table storage

  Scenario: Fundata quotes record validation for local CSV processing
    Given I have CSV data from local fundata/quotes/ directory (Git LFS)
    When I create FundataQuotesRecord models for batch ingestion
    Then identifier field should be properly validated and indexed
    And NAVPS fields should be validated as positive decimals
    And records should be prepared for fundata_quotes table storage