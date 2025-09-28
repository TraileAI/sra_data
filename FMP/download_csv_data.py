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

# Try to import B2 SDK for private bucket access
try:
    from b2sdk.v2 import InMemoryAccountInfo, B2Api
    B2_SDK_AVAILABLE = True
except ImportError:
    B2_SDK_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backblaze B2 configuration
# Format: https://f{bucket-id}.backblazeb2.com/file/{bucket-name}/
B2_BUCKET_URL = os.getenv('B2_BUCKET_URL', '')
CSV_BASE_URL = os.getenv('CSV_BASE_URL', B2_BUCKET_URL)

# CSV files that need to be downloaded from B2 (flat structure) to local directories
CSV_FILES = {
    # FMP data files (stored flat in B2, downloaded to fmp_data/)
    'equity_income.csv': 'fmp_data/equity_income.csv',
    'equity_balance.csv': 'fmp_data/equity_balance.csv',
    'equity_balance_growth.csv': 'fmp_data/equity_balance_growth.csv',
    'equity_cash_flow.csv': 'fmp_data/equity_cash_flow.csv',
    'equity_cashflow_growth.csv': 'fmp_data/equity_cashflow_growth.csv',
    'equity_financial_growth.csv': 'fmp_data/equity_financial_growth.csv',
    'equity_income_growth.csv': 'fmp_data/equity_income_growth.csv',
    'equity_key_metrics.csv': 'fmp_data/equity_key_metrics.csv',
    'equity_peers.csv': 'fmp_data/equity_peers.csv',
    'equity_profile.csv': 'fmp_data/equity_profile.csv',
    'equity_ratios.csv': 'fmp_data/equity_ratios.csv',
    'etfs_data.csv': 'fmp_data/etfs_data.csv',
    'etfs_peers.csv': 'fmp_data/etfs_peers.csv',
    'etfs_profile.csv': 'fmp_data/etfs_profile.csv',

    # Individual equity quote files (64 files from 1962-2025)
    **{f'equity_quote_{year}.csv': f'fmp_data/equity_quotes/equity_quote_{year}.csv'
       for year in range(1962, 2026)},

    # ETF quote files (7 files)
    'etfs_quote_1993.csv': 'fmp_data/etfs_quotes/etfs_quote_1993.csv',
    'etfs_quote_1994.csv': 'fmp_data/etfs_quotes/etfs_quote_1994.csv',
    'etfs_quote_1995.csv': 'fmp_data/etfs_quotes/etfs_quote_1995.csv',
    'etfs_quote_1996.csv': 'fmp_data/etfs_quotes/etfs_quote_1996.csv',
    'etfs_quote_1997.csv': 'fmp_data/etfs_quotes/etfs_quote_1997.csv',
    'etfs_quote_2022.csv': 'fmp_data/etfs_quotes/etfs_quote_2022.csv',
    'etfs_quote_2023.csv': 'fmp_data/etfs_quotes/etfs_quote_2023.csv',

    # Fundata files (18 files in fundata/data/)
    'BenchmarkGeneralSeed.csv': 'fundata/data/BenchmarkGeneralSeed.csv',
    'BenchmarkYearlyPerformanceSeed.csv': 'fundata/data/BenchmarkYearlyPerformanceSeed.csv',
    'FundAdvancedPerformanceSeed.csv': 'fundata/data/FundAdvancedPerformanceSeed.csv',
    'FundAllocationSeed.csv': 'fundata/data/FundAllocationSeed.csv',
    'FundAssetsSeed.csv': 'fundata/data/FundAssetsSeed.csv',
    'FundAssociateBenchmarkSeed.csv': 'fundata/data/FundAssociateBenchmarkSeed.csv',
    'FundDistributionSeed.csv': 'fundata/data/FundDistributionSeed.csv',
    'FundEquityStyleSeed.csv': 'fundata/data/FundEquityStyleSeed.csv',
    'FundExpensesSeed.csv': 'fundata/data/FundExpensesSeed.csv',
    'FundFixedIncomeStyleSeed.csv': 'fundata/data/FundFixedIncomeStyleSeed.csv',
    'FundGeneralSeed.csv': 'fundata/data/FundGeneralSeed.csv',
    'FundLoadsSeed.csv': 'fundata/data/FundLoadsSeed.csv',
    'FundOtherFeeSeed.csv': 'fundata/data/FundOtherFeeSeed.csv',
    'FundPerformanceSummarySeed.csv': 'fundata/data/FundPerformanceSummarySeed.csv',
    'FundRiskYearlyPerformanceSeed.csv': 'fundata/data/FundRiskYearlyPerformanceSeed.csv',
    'FundTopHoldingSeed.csv': 'fundata/data/FundTopHoldingSeed.csv',
    'FundTrailerScheduleSeed.csv': 'fundata/data/FundTrailerScheduleSeed.csv',
    'FundYearlyPerformanceRankingByClassSeed.csv': 'fundata/data/FundYearlyPerformanceRankingByClassSeed.csv',
    'FundYearlyPerformanceSeed.csv': 'fundata/data/FundYearlyPerformanceSeed.csv',
    'InstrumentIdentifierSeed.csv': 'fundata/data/InstrumentIdentifierSeed.csv',

    # Fundata quotes (6 files in fundata/quotes/)
    'FundDailyNAVPSSeed.csv': 'fundata/quotes/FundDailyNAVPSSeed.csv',
    'Pricing2015to2017.csv': 'fundata/quotes/Pricing2015to2025/Pricing2015to2017.csv',
    'Pricing2018to2019.csv': 'fundata/quotes/Pricing2015to2025/Pricing2018to2019.csv',
    'Pricing2020to2021.csv': 'fundata/quotes/Pricing2015to2025/Pricing2020to2021.csv',
    'Pricing2022to2023.csv': 'fundata/quotes/Pricing2015to2025/Pricing2022to2023.csv',
    'Pricing2024to2025.csv': 'fundata/quotes/Pricing2015to2025/Pricing2024to2025.csv',
}

def setup_b2_api():
    """Set up B2 API client for private bucket access."""
    if not B2_SDK_AVAILABLE:
        return None

    key_id = os.getenv('B2_APPLICATION_KEY_ID')
    key = os.getenv('B2_APPLICATION_KEY')

    if not key_id or not key:
        logger.warning("B2 credentials not found - falling back to direct URL access")
        return None

    try:
        info = InMemoryAccountInfo()
        api = B2Api(info)
        api.authorize_account("production", key_id, key)
        return api
    except Exception as e:
        logger.warning(f"Failed to authenticate with B2: {e}")
        return None

def download_file_b2(api, bucket_name: str, filename: str, local_path: str) -> bool:
    """Download a file using B2 SDK."""
    try:
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        bucket = api.get_bucket_by_name(bucket_name)
        file_info = bucket.get_file_info_by_name(filename)

        logger.info(f"Downloading {filename} ({file_info.size} bytes) via B2 API...")

        # Download file
        download_dest = str(local_path)
        bucket.download_file_by_name(filename, download_dest)

        logger.info(f"Downloaded: {local_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to download {filename} via B2 API: {e}")
        return False

def download_file(url: str, local_path: str, chunk_size: int = 8192) -> bool:
    """Download a single file with progress bar (direct HTTP)."""
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

    # Try to set up B2 API first for private bucket access
    b2_api = setup_b2_api()
    bucket_name = "sra-data-csv"

    if b2_api:
        logger.info("Using B2 SDK for private bucket access")
        success_count = 0
        for filename in missing_files:
            local_path = CSV_FILES[filename]
            if download_file_b2(b2_api, bucket_name, filename, local_path):
                success_count += 1
    else:
        # Fall back to direct URL access for public buckets
        if not CSV_BASE_URL:
            logger.error("B2_BUCKET_URL or CSV_BASE_URL environment variable not configured")
            logger.error("Please set B2_BUCKET_URL to your Backblaze B2 bucket URL")
            logger.error("Format: https://f{bucket-id}.backblazeb2.com/file/{bucket-name}/")
            return False

        logger.info("Using direct HTTP access (requires public bucket)")
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