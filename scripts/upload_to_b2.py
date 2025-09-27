#!/usr/bin/env python3
"""
Upload CSV files to Backblaze B2 bucket.

This script uploads all CSV files from the FMP directory to a Backblaze B2 bucket
for use in Render deployments.

Requirements:
    pip install b2sdk

Usage:
    python scripts/upload_to_b2.py --bucket-name your-bucket-name

Environment Variables:
    B2_APPLICATION_KEY_ID - Your B2 application key ID
    B2_APPLICATION_KEY - Your B2 application key
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

try:
    from b2sdk.v2 import InMemoryAccountInfo, B2Api
except ImportError:
    print("Error: b2sdk not installed. Run: pip install b2sdk")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CSV files to upload (matching download_csv_data.py)
CSV_FILES = {
    # FMP data files
    'equity_income.csv': 'FMP/equity_income.csv',
    'equity_balance_sheet.csv': 'FMP/equity_balance_sheet.csv',
    'equity_cash_flow.csv': 'FMP/equity_cash_flow.csv',
    'equity_ratios.csv': 'FMP/equity_ratios.csv',
    'equity_financial_growth.csv': 'FMP/equity_financial_growth.csv',
    'equity_peers.csv': 'FMP/equity_peers.csv',
    'market_and_sector_quotes.csv': 'FMP/market_and_sector_quotes.csv',
    'treasury.csv': 'FMP/treasury.csv',
    'crypto_quotes.csv': 'FMP/crypto_quotes.csv',
    'etfs_profile.csv': 'FMP/etfs_profile.csv',
    'etfs_peers.csv': 'FMP/etfs_peers.csv',
    'etfs_quotes.csv': 'FMP/etfs_quotes.csv',
    'etfs_holding.csv': 'FMP/etfs_holding.csv',
    'forex_quotes.csv': 'FMP/forex_quotes.csv',
    # Individual stock quote files (71 files)
    **{f'equity_quotes_{i:02d}.csv': f'FMP/equity_quotes_{i:02d}.csv' for i in range(1, 72)}
}

def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

def check_files_exist():
    """Check which CSV files exist locally."""
    project_root = get_project_root()
    existing_files = {}
    missing_files = []

    for filename, local_path in CSV_FILES.items():
        full_path = project_root / local_path
        if full_path.exists():
            existing_files[filename] = full_path
        else:
            missing_files.append(filename)

    return existing_files, missing_files

def setup_b2_api():
    """Set up B2 API client."""
    key_id = os.getenv('B2_APPLICATION_KEY_ID')
    key = os.getenv('B2_APPLICATION_KEY')

    if not key_id or not key:
        logger.error("B2 credentials not found in environment variables")
        logger.error("Please set B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY")
        return None

    info = InMemoryAccountInfo()
    api = B2Api(info)

    try:
        api.authorize_account("production", key_id, key)
        logger.info("Successfully authenticated with Backblaze B2")
        return api
    except Exception as e:
        logger.error(f"Failed to authenticate with B2: {e}")
        return None

def upload_file_to_b2(api, bucket, local_path, remote_filename):
    """Upload a single file to B2."""
    try:
        file_size = local_path.stat().st_size
        logger.info(f"Uploading {remote_filename} ({file_size:,} bytes)")

        with open(local_path, 'rb') as file_data:
            # Upload with progress
            bucket.upload(
                file_data,
                remote_filename,
                content_type='text/csv'
            )

        logger.info(f"✅ Uploaded: {remote_filename}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to upload {remote_filename}: {e}")
        return False

def upload_csv_files(bucket_name, force=False):
    """Upload all CSV files to B2 bucket."""
    # Check local files
    existing_files, missing_files = check_files_exist()

    if missing_files:
        logger.warning(f"Missing {len(missing_files)} local CSV files:")
        for filename in missing_files[:5]:  # Show first 5
            logger.warning(f"  - {filename}")
        if len(missing_files) > 5:
            logger.warning(f"  ... and {len(missing_files) - 5} more")

    if not existing_files:
        logger.error("No CSV files found locally")
        return False

    logger.info(f"Found {len(existing_files)} CSV files to upload")

    # Set up B2 API
    api = setup_b2_api()
    if not api:
        return False

    # Get bucket
    try:
        bucket = api.get_bucket_by_name(bucket_name)
        logger.info(f"Using bucket: {bucket_name}")
    except Exception as e:
        logger.error(f"Failed to access bucket '{bucket_name}': {e}")
        logger.error("Make sure the bucket exists and you have access")
        return False

    # Check existing files in bucket (unless force)
    files_to_upload = existing_files.copy()
    if not force:
        try:
            logger.info("Checking existing files in bucket...")
            existing_b2_files = set()
            for file_version, folder_to_list in bucket.ls(recursive=True):
                existing_b2_files.add(file_version.file_name)

            # Remove files that already exist
            skip_count = 0
            for filename in list(files_to_upload.keys()):
                if filename in existing_b2_files:
                    del files_to_upload[filename]
                    skip_count += 1

            if skip_count > 0:
                logger.info(f"Skipping {skip_count} files that already exist (use --force to overwrite)")

        except Exception as e:
            logger.warning(f"Could not check existing files: {e}")
            logger.info("Proceeding with upload...")

    if not files_to_upload:
        logger.info("All files already exist in bucket")
        return True

    # Upload files
    logger.info(f"Uploading {len(files_to_upload)} files...")
    success_count = 0

    with tqdm(total=len(files_to_upload), desc="Uploading") as pbar:
        for filename, local_path in files_to_upload.items():
            if upload_file_to_b2(api, bucket, local_path, filename):
                success_count += 1
            pbar.update(1)

    logger.info(f"Upload complete: {success_count}/{len(files_to_upload)} files successful")

    if success_count == len(files_to_upload):
        # Show the bucket URL format
        bucket_id = bucket.id_
        bucket_url = f"https://f{bucket_id[:-3]}.backblazeb2.com/file/{bucket_name}/"
        logger.info(f"\n🎉 Upload successful!")
        logger.info(f"Set this environment variable in Render:")
        logger.info(f"B2_BUCKET_URL={bucket_url}")
        return True

    return False

def main():
    parser = argparse.ArgumentParser(description='Upload CSV files to Backblaze B2')
    parser.add_argument('--bucket-name', required=True, help='B2 bucket name')
    parser.add_argument('--force', action='store_true', help='Force upload even if files exist')
    parser.add_argument('--check', action='store_true', help='Only check which files exist locally')

    args = parser.parse_args()

    if args.check:
        existing, missing = check_files_exist()
        print(f"Local CSV files: {len(existing)} found, {len(missing)} missing")
        if missing:
            print("Missing files:")
            for f in missing:
                print(f"  - {f}")
        return

    success = upload_csv_files(args.bucket_name, force=args.force)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()