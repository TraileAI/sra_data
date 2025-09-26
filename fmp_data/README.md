# FMP CSV Data Directory

This directory contains the FMP data in CSV format for fast deployment loading.

## Expected Files:

### Equity Data:
- `equity_profile.csv` - Company profiles and basic information
- `equity_income.csv` - Income statement data (quarterly)
- `equity_balance.csv` - Balance sheet data (quarterly)
- `equity_cashflow.csv` - Cash flow statement data (quarterly)
- `equity_quotes.csv` - Historical stock price data
- `equity_peers.csv` - Peer company relationships
- `equity_financial_ratio.csv` - Financial ratios and metrics

### ETF Data:
- `etfs_profile.csv` - ETF profiles and basic information
- `etfs_peers.csv` - ETF peer relationships

## Loading Process:

The CSV files are loaded during initial seeding using PostgreSQL COPY commands for maximum performance.

1. **Fast Loading**: Uses PostgreSQL COPY FROM for bulk loading
2. **No API Calls**: Eliminates rate limiting during deployment
3. **Predictable Performance**: CSV loading time is consistent
4. **Git LFS**: Large CSV files are managed with Git LFS

## Usage:

```bash
# Load all CSV data (FMP + fundata)
python server.py seeding

# Load only FMP data
python server.py fmp

# Load only fundata
python server.py fundata

# Force fresh loading (clears existing data)
python server.py seeding --force
```

## File Requirements:

- CSV files must have headers matching the database table columns
- Files should be UTF-8 encoded
- Use comma separators
- NULL values should be empty strings

The CSV loader will automatically create tables if they don't exist and handle data type conversions.