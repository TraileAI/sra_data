# SRA Data - CSV Data Pipeline

This project automatically downloads and loads financial data from B2 cloud storage into PostgreSQL databases. It supports both FMP (Financial Modeling Prep) and Fundata sources.

## Features

- 🚀 **Auto-seeding**: Automatically detects when database needs seeding based on row counts
- 🔄 **Smart downloading**: Downloads CSV files from B2 only when needed
- 🛠️ **CSV error recovery**: Handles malformed CSV files with robust parsing
- 📊 **Structured data**: Creates proper PostgreSQL schemas (no JSONB)
- 🐳 **Docker ready**: Works locally with Docker and deploys to Render.com
- ⚡ **High performance**: Uses PostgreSQL COPY for fast data loading

## Quick Start

### Local Development with Docker

1. **Copy environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your B2 credentials
   ```

2. **Run with Docker:**
   ```bash
   docker-compose up --build
   ```

This will:
- Start PostgreSQL database
- Download all CSV files from B2
- Load FMP and Fundata into structured tables
- Auto-seed only if tables are under-populated

### Local Development without Docker

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up PostgreSQL database:**
   ```bash
   createdb sra_data
   ```

3. **Configure B2 CLI:**
   ```bash
   b2 account authorize
   ```

4. **Run seeding:**
   ```bash
   python load_csv_data.py
   ```

## Deployment to Render.com

### Automatic Deployment

1. **Connect to GitHub**: Link your Render account to this repository

2. **Set environment variables** in Render dashboard:
   - `B2_APPLICATION_KEY_ID`
   - `B2_APPLICATION_KEY`

3. **Deploy**: Render will automatically:
   - Create PostgreSQL database service
   - Create data seeding worker service
   - Download and load all data on first run

### Manual Deployment

Alternatively, use the Render CLI:

```bash
render services create --file render.yaml
```

## Data Sources

### FMP Data (Financial Modeling Prep)
- **Equity profiles**: 4,000+ companies
- **Financial statements**: Income, balance sheet, cash flow
- **Key metrics**: Financial ratios, growth metrics
- **Stock quotes**: 14M+ historical quotes (1962-2025)
- **ETF data**: Profiles, quotes, peer analysis

### Fundata
- **Fund profiles**: 2,700+ Canadian mutual funds
- **Performance data**: Historical returns and metrics
- **Holdings and allocations**: Asset allocation data
- **Pricing data**: Daily NAV and quotes (2015-2025)

## Database Schema

All data is loaded into structured PostgreSQL tables:

**FMP Tables:**
- `equity_profile`, `equity_income`, `equity_balance`, `equity_cashflow`
- `equity_quotes`, `etfs_quotes` (time series data)
- `equity_financial_ratio`, `equity_key_metrics`

**Fundata Tables:**
- `fund_general`, `benchmark_general`
- `fund_daily_nav`, `fund_quotes` (time series data)
- `fund_performance_summary`, `fund_allocation`

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `B2_APPLICATION_KEY_ID` | B2 application key ID | `005aef5952738310000000001` |
| `B2_APPLICATION_KEY` | B2 application key | `K005IalJAukjMj0anZhA2VunBD0Lqac` |
| `B2_BUCKET_URL` | B2 bucket base URL | `https://f005.backblazeb2.com/file/sra-data-csv/` |
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `sra_data` |
| `DB_USER` | Database user | `nauraai` |
| `DB_PASSWORD` | Database password | `password` |

### Smart Auto-Seeding

The system only seeds when tables are under-populated:

```python
EXPECTED_MINIMUMS = {
    'equity_profile': 4000,
    'equity_quotes': 10000000,  # 10M+ quotes
    'fund_general': 2000,
    # ... more thresholds
}
```

## Commands

```bash
# Force seeding (ignore table counts)
python load_csv_data.py --force

# Smart seeding (default - only seed if needed)
python load_csv_data.py

# Check current status
python -c "from load_csv_data import get_all_table_counts; print(get_all_table_counts())"
```

## Performance

- **Download speed**: 112 files (~5GB) in ~6 minutes
- **Loading speed**: 14M+ quotes loaded in ~10 minutes
- **Total seeding time**: ~15-20 minutes for complete database
- **Memory usage**: <2GB RAM during operation

## Troubleshooting

### Common Issues

1. **B2 authentication failed**
   ```bash
   b2 account authorize
   # Enter your key ID and application key
   ```

2. **Database connection failed**
   - Check `DB_*` environment variables
   - Ensure PostgreSQL is running
   - Verify database exists: `createdb sra_data`

3. **CSV parsing errors**
   - The system automatically handles malformed CSV files
   - Check logs for warnings about fixed lines

4. **Slow performance**
   - Ensure adequate disk space (>10GB free)
   - Check network connection for B2 downloads
   - Monitor PostgreSQL performance

### Logs

All operations are logged with timestamps:

```
2025-09-28 12:03:08 - INFO - === Starting Complete CSV Seeding Process ===
2025-09-28 12:03:08 - INFO - ✅ CSV download completed successfully
2025-09-28 12:03:08 - INFO - Successfully loaded 4133 rows into equity_profile
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   B2 Storage    │───▶│  sra_data App   │───▶│  PostgreSQL DB  │
│                 │    │                 │    │                 │
│ • FMP CSVs      │    │ • Auto-download │    │ • Structured    │
│ • Fundata CSVs  │    │ • CSV cleaning  │    │   tables        │
│ • 112 files     │    │ • Smart seeding │    │ • 25+ tables    │
│ • ~5GB data     │    │ • Error recovery│    │ • 50M+ rows     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature-name`
3. **Make changes and test**: `docker-compose up --build`
4. **Submit pull request**

## License

MIT License - see LICENSE file for details.