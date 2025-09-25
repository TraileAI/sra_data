Feature: Financial Data Processing for Database Ingestion
  As a data processing service
  I want to ingest market data efficiently from multiple sources
  So that raw data is available for modelized view creation

  Scenario: Process daily market data for database seeding
    Given I have 100 equity symbols to process for raw storage
    And the FMP API is available
    When I trigger daily market data ingestion
    Then all 100 symbols should be processed and stored within 2 minutes
    And the raw data should be available for view creation
    And processing metrics should be logged for monitoring

  Scenario: Handle API rate limiting during data ingestion
    Given the FMP API has rate limits of 300 requests per minute
    When I process 500 symbols for database seeding
    Then the system should respect rate limits automatically
    And all symbols should eventually be processed and stored
    And no API rate limit violations should occur

  Scenario: Process fundata CSV files for raw storage
    Given there are 3 CSV files available for processing
    When I trigger CSV ingestion for database seeding
    Then all files should be downloaded and parsed
    And invalid records should be logged but not stored
    And valid records should be ready for modelized view creation

  Scenario: Retry logic for failed data ingestion
    Given a data ingestion task fails due to temporary issues
    When the retry mechanism is triggered
    Then the task should be retried up to 3 times
    And exponential backoff should be applied
    And final failure should be logged if retries are exhausted