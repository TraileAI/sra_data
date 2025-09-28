#!/usr/bin/env python3
"""
Download CSV files from B2 for seeding process.
This ensures all required CSV files are available for loading.
"""
import os
import subprocess
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core FMP files that must be downloaded
CORE_FMP_FILES = [
    'equity_profile.csv',
    'equity_income.csv',
    'equity_balance.csv',
    'equity_cash_flow.csv',
    'equity_earnings.csv',
    'equity_peers.csv',
    'equity_ratios.csv',
    'equity_key_metrics.csv',
    'equity_balance_growth.csv',
    'equity_cashflow_growth.csv',
    'equity_financial_growth.csv',
    'equity_financial_scores.csv',
    'equity_income_growth.csv',
    'etfs_profile.csv',
    'etfs_peers.csv',
    'etfs_data.csv'
]

# Get all quote files dynamically from B2 (will be populated at runtime)
ALL_QUOTE_FILES = []

def ensure_directories():
    """Create necessary directories."""
    os.makedirs('fmp_data', exist_ok=True)
    os.makedirs('fmp_data/equity_quotes', exist_ok=True)
    os.makedirs('fmp_data/etfs_quotes', exist_ok=True)
    os.makedirs('fundata', exist_ok=True)
    os.makedirs('fundata/data', exist_ok=True)
    os.makedirs('fundata/quotes', exist_ok=True)
    os.makedirs('fundata/quotes/Pricing2015to2025', exist_ok=True)
    logger.info("Created required directories")

def check_b2_auth():
    """Check if B2 is properly authenticated."""
    try:
        result = subprocess.run(['b2', 'account', 'get'],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info("B2 authentication verified")
            return True
        else:
            logger.error(f"B2 authentication failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error checking B2 auth: {e}")
        return False

def get_all_quote_files_from_b2() -> Tuple[List[str], List[str]]:
    """Get list of all quote files in B2."""
    try:
        # List all files in B2
        result = subprocess.run(['b2', 'ls', 'b2://sra-data-csv'],
                              capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"Failed to list B2 files: {result.stderr}")
            return [], []

        # Parse the output
        equity_files = []
        etf_files = []

        for line in result.stdout.splitlines():
            if line:
                # B2 ls format includes size, date, and filename
                parts = line.strip().split()
                if parts:
                    filename = parts[-1]  # Last part is the filename
                    if filename.startswith('equity_quote_'):
                        equity_files.append(filename)
                    elif filename.startswith('etfs_quote_'):
                        etf_files.append(filename)

        logger.info(f"Found {len(equity_files)} equity quote files and {len(etf_files)} ETF quote files in B2")
        return equity_files, etf_files

    except Exception as e:
        logger.error(f"Error listing B2 files: {e}")
        return [], []

def download_file_from_b2(filename: str, local_path: str) -> bool:
    """Download a single file from B2."""
    try:
        # Check if file already exists locally
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            if file_size > 0:
                logger.info(f"File {filename} already exists locally ({file_size} bytes), skipping")
                return True

        logger.info(f"Downloading {filename} from B2...")
        cmd = ['b2', 'file', 'download', f'b2://sra-data-csv/{filename}', local_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"Successfully downloaded {filename}")
            return True
        else:
            logger.error(f"Failed to download {filename}: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading {filename}")
        return False
    except Exception as e:
        logger.error(f"Error downloading {filename}: {e}")
        return False

def download_core_fmp_files() -> Tuple[int, int]:
    """Download core FMP CSV files."""
    logger.info("Downloading core FMP files...")
    success_count = 0
    total_count = len(CORE_FMP_FILES)

    for filename in CORE_FMP_FILES:
        local_path = f"fmp_data/{filename}"
        if download_file_from_b2(filename, local_path):
            success_count += 1

    logger.info(f"Core FMP files: {success_count}/{total_count} downloaded successfully")
    return success_count, total_count

def download_all_quotes() -> Tuple[int, int]:
    """Download all quote files from B2."""
    logger.info("Downloading all quote files...")

    # Get list of quote files from B2
    equity_files, etf_files = get_all_quote_files_from_b2()

    if not equity_files and not etf_files:
        logger.warning("No quote files found in B2")
        return 0, 0

    success_count = 0
    total_count = len(equity_files) + len(etf_files)

    # Download equity quote files
    for filename in equity_files:
        local_path = f"fmp_data/equity_quotes/{filename}"
        if download_file_from_b2(filename, local_path):
            success_count += 1

    # Download ETF quote files
    for filename in etf_files:
        local_path = f"fmp_data/etfs_quotes/{filename}"
        if download_file_from_b2(filename, local_path):
            success_count += 1

    logger.info(f"Quote files: {success_count}/{total_count} downloaded successfully")
    return success_count, total_count

def download_fundata_files() -> Tuple[int, int]:
    """Download fundata files from B2."""
    logger.info("Downloading fundata files...")

    # Get list of all fundata files from B2
    try:
        result = subprocess.run(['b2', 'ls', 'b2://sra-data-csv'],
                              capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"Failed to list B2 files: {result.stderr}")
            return 0, 0

        # Find fundata files
        fundata_files = []
        pricing_files = []

        for line in result.stdout.splitlines():
            if line:
                parts = line.strip().split()
                if parts:
                    filename = parts[-1]
                    if (filename.startswith(('Fund', 'Benchmark', 'Instrument')) and
                        filename.endswith('Seed.csv')):
                        fundata_files.append(filename)
                    elif filename.startswith('Pricing') and filename.endswith('.csv'):
                        pricing_files.append(filename)

        logger.info(f"Found {len(fundata_files)} fundata files and {len(pricing_files)} pricing files in B2")

        success_count = 0
        total_count = len(fundata_files) + len(pricing_files)

        # Download main fundata files
        for filename in fundata_files:
            local_path = f"fundata/data/{filename}"
            if download_file_from_b2(filename, local_path):
                success_count += 1

        # Download pricing files
        for filename in pricing_files:
            local_path = f"fundata/quotes/Pricing2015to2025/{filename}"
            if download_file_from_b2(filename, local_path):
                success_count += 1

        # Also download FundDailyNAVPSSeed.csv to quotes directory
        if download_file_from_b2('FundDailyNAVPSSeed.csv', 'fundata/quotes/FundDailyNAVPSSeed.csv'):
            success_count += 1
            total_count += 1

        logger.info(f"Fundata files: {success_count}/{total_count} downloaded successfully")
        return success_count, total_count

    except Exception as e:
        logger.error(f"Error downloading fundata files: {e}")
        return 0, 0

def download_all_csv_files() -> bool:
    """Download all required CSV files for seeding."""
    logger.info("=== Starting CSV File Download Process ===")

    # Ensure directories exist
    ensure_directories()

    # Check B2 authentication
    if not check_b2_auth():
        logger.error("B2 authentication required. Please run: b2 account authorize")
        return False

    # Download core FMP files
    fmp_success, fmp_total = download_core_fmp_files()

    # Download all quote files
    quote_success, quote_total = download_all_quotes()

    # Download fundata files
    fundata_success, fundata_total = download_fundata_files()

    # Calculate overall success
    total_success = fmp_success + quote_success + fundata_success
    total_files = fmp_total + quote_total + fundata_total

    logger.info(f"=== Download Summary: {total_success}/{total_files} files downloaded ===")

    # Consider it successful if we have the core files and reasonable coverage
    core_success = fmp_success >= (fmp_total * 0.8)  # 80% of core files needed
    quote_success_rate = quote_success >= (quote_total * 0.8) if quote_total > 0 else True  # 80% of quote files
    fundata_success_rate = fundata_success >= (fundata_total * 0.8) if fundata_total > 0 else True  # 80% of fundata files

    if core_success and quote_success_rate and fundata_success_rate:
        logger.info("✅ CSV download completed successfully")
        return True
    else:
        logger.error(f"❌ CSV download failed - Core: {fmp_success}/{fmp_total}, Quotes: {quote_success}/{quote_total}, Fundata: {fundata_success}/{fundata_total}")
        return False

if __name__ == "__main__":
    success = download_all_csv_files()
    exit(0 if success else 1)