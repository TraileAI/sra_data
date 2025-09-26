# FMP CSV Data Directory

This directory contains the FMP data in CSV format for fast deployment loading.

## Current Files:

### Equity Data:
- `equity_profile.csv` - Company profiles and basic information (5.7MB)
- `equity_income.csv` - Income statement data (quarterly) (132MB)
- `equity_balance.csv` - Balance sheet data (quarterly) (176MB)
- `equity_cash_flow.csv` - Cash flow statement data (quarterly) (139MB)
- `equity_peers.csv` - Peer company relationships (252KB)
- `equity_ratios.csv` - Financial ratios and metrics (254MB)
- `equity_key_metrics.csv` - Key financial metrics (252MB)

### Growth Data:
- `equity_balance_growth.csv` - Balance sheet growth metrics (154MB)
- `equity_cashflow_growth.csv` - Cash flow growth metrics (117MB)
- `equity_financial_growth.csv` - Financial growth metrics (159MB)
- `equity_income_growth.csv` - Income growth metrics (125MB)

### ETF Data:
- `etfs_profile.csv` - ETF profiles and basic information (2MB)
- `etfs_peers.csv` - ETF peer relationships (59KB)
- `etfs_data.csv` - ETF detailed data (79MB)
- `etfs_quotes/` - Directory containing ETF quotes by year
  - `etfs_quote_1993.csv` through `etfs_quote_2023.csv`

### Equity Quotes Data:
- `equity_quotes/` - Directory containing equity quotes by year (~1.7GB total)
  - `equity_quote_1962.csv` through `equity_quote_2025.csv`
  - Historical equity price data spanning 63+ years

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