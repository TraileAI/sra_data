# Deployment Instructions

## Render.com Deployment with Large CSV Files

This project uses large CSV files (7.3GB) stored in Backblaze B2 to avoid Git LFS authentication issues during Render deployment.

### Setup Steps

#### 1. Set up Backblaze B2 credentials

```bash
export B2_APPLICATION_KEY_ID="005aef5952738310000000001"
export B2_APPLICATION_KEY="your-application-key"
```

#### 2. Install B2 SDK (if uploading from local machine)

```bash
pip install b2sdk
```

#### 3. Create B2 bucket

```bash
python scripts/setup_b2_bucket.py --bucket-name sra-data-csv
```

This will create a public bucket and show you the bucket URL.

#### 4. Upload CSV files to B2

```bash
python scripts/upload_to_b2.py --bucket-name sra-data-csv
```

#### 5. Set environment variable in Render

In your Render dashboard, add:
```
B2_BUCKET_URL=https://f{bucket-id}.backblazeb2.com/file/sra-data-csv/
```

(The exact URL will be shown after creating the bucket)

#### 6. Deploy to Render

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
- `B2_BUCKET_URL` - Backblaze B2 bucket URL for CSV downloads

Optional:
- `CSV_BASE_URL` - Alternative to B2_BUCKET_URL if using different storage
- Individual database connection variables (if not using DATABASE_URL)

### Cost Estimation

Backblaze B2 pricing for 7.3GB storage:
- Storage: ~$0.37/month (7.3GB × $0.005/GB)
- Download: ~$0.07 per deployment (7.3GB × $0.01/GB)
- Very cost-effective for this use case!