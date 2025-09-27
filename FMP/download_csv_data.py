#!/usr/bin/env python3
"""
Download CSV data from external storage for Render deployment.

This script downloads CSV files from a cloud storage service when they're needed,
avoiding Git LFS authentication issues during Render builds.
"""

import os
import requests
from pathlib import Path
import logging
from typing import List, Dict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL for CSV files (to be updated with actual cloud storage URL)
CSV_BASE_URL = os.getenv('CSV_BASE_URL', 'https://your-storage-bucket.example.com/csv/')

# CSV files that need to be downloaded
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

def download_file(url: str, local_path: str, chunk_size: int = 8192) -> bool:
    """Download a single file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        with open(local_path, 'wb') as f, tqdm(
            desc=os.path.basename(local_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

        logger.info(f"Downloaded: {local_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def check_files_exist() -> List[str]:
    """Check which CSV files are missing."""
    missing_files = []
    for filename, local_path in CSV_FILES.items():
        if not os.path.exists(local_path):
            missing_files.append(filename)
    return missing_files

def download_csv_files(force: bool = False) -> bool:
    """Download all required CSV files."""
    if not force:
        missing_files = check_files_exist()
        if not missing_files:
            logger.info("All CSV files are already present")
            return True
        logger.info(f"Missing {len(missing_files)} CSV files")
    else:
        missing_files = list(CSV_FILES.keys())
        logger.info("Force download requested - downloading all files")

    if not CSV_BASE_URL or CSV_BASE_URL == 'https://your-storage-bucket.example.com/csv/':
        logger.error("CSV_BASE_URL environment variable not configured")
        logger.error("Please set CSV_BASE_URL to your cloud storage URL")
        return False

    success_count = 0
    for filename in missing_files:
        local_path = CSV_FILES[filename]
        url = f"{CSV_BASE_URL.rstrip('/')}/{filename}"

        if download_file(url, local_path):
            success_count += 1

    logger.info(f"Downloaded {success_count}/{len(missing_files)} files successfully")
    return success_count == len(missing_files)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download CSV data files')
    parser.add_argument('--force', action='store_true',
                       help='Force download all files even if they exist')
    parser.add_argument('--check', action='store_true',
                       help='Only check which files are missing')

    args = parser.parse_args()

    if args.check:
        missing = check_files_exist()
        if missing:
            print(f"Missing {len(missing)} files:")
            for f in missing:
                print(f"  - {f}")
        else:
            print("All CSV files are present")
    else:
        success = download_csv_files(force=args.force)
        if not success:
            exit(1)