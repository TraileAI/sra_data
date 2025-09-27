# Deployment Instructions

## Render.com Deployment with Large CSV Files

This project uses large CSV files (7.3GB) that are managed through external storage to avoid Git LFS authentication issues during Render deployment.

### Setup Steps

1. **Upload CSV files to cloud storage** (S3, Google Cloud Storage, etc.)
   - Upload all CSV files from the `FMP/` directory
   - Ensure files are publicly accessible or set up proper authentication

2. **Set environment variable in Render**
   ```
   CSV_BASE_URL=https://your-storage-bucket.amazonaws.com/csv/
   ```

3. **Deploy to Render**
   - The build process will skip CSV download during build
   - CSV files will be downloaded automatically when the application runs
   - First run may take longer as files are downloaded

### CSV Files Location

The following files need to be uploaded to your cloud storage:

**Main FMP Data:**
- equity_income.csv
- equity_balance_sheet.csv
- equity_cash_flow.csv
- equity_ratios.csv
- equity_financial_growth.csv
- equity_peers.csv
- market_and_sector_quotes.csv
- treasury.csv
- crypto_quotes.csv
- etfs_profile.csv
- etfs_peers.csv
- etfs_quotes.csv
- etfs_holding.csv
- forex_quotes.csv

**Stock Quote Files (71 files):**
- equity_quotes_01.csv through equity_quotes_71.csv

### Local Development

For local development, ensure CSV files are present in the `FMP/` directory or the application will attempt to download them from the configured `CSV_BASE_URL`.

### Manual CSV Download

You can manually trigger CSV download:

```bash
python FMP/download_csv_data.py --check          # Check missing files
python FMP/download_csv_data.py                  # Download missing files
python FMP/download_csv_data.py --force          # Force download all files
```

### Environment Variables

Required for Render deployment:
- `DATABASE_URL` - PostgreSQL connection string (automatically provided by Render)
- `CSV_BASE_URL` - Base URL for CSV file downloads

Optional:
- Individual database connection variables (if not using DATABASE_URL)