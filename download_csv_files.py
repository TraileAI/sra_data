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

# Mapping from database table names to their required CSV files
TABLE_TO_CSV_MAPPING = {
    # FMP equity tables
    'equity_profile': ['equity_profile.csv'],
    'equity_income': ['equity_income.csv'],
    'equity_balance': ['equity_balance.csv'],
    'equity_cashflow': ['equity_cash_flow.csv'],
    'equity_earnings': ['equity_earnings.csv'],
    'equity_peers': ['equity_peers.csv'],
    'equity_financial_ratio': ['equity_ratios.csv'],
    'equity_key_metrics': ['equity_key_metrics.csv'],
    'equity_balance_growth': ['equity_balance_growth.csv'],
    'equity_cashflow_growth': ['equity_cashflow_growth.csv'],
    'equity_financial_growth': ['equity_financial_growth.csv'],
    'equity_financial_scores': ['equity_financial_scores.csv'],
    'equity_income_growth': ['equity_income_growth.csv'],

    # FMP ETF tables
    'etfs_profile': ['etfs_profile.csv'],
    'etfs_peers': ['etfs_peers.csv'],
    'etfs_data': ['etfs_data.csv'],

    # Quote tables (these require special handling for multiple files)
    'equity_quotes': 'equity_quotes_files',  # Special marker for dynamic quote files
    'etfs_quotes': 'etfs_quotes_files',      # Special marker for dynamic quote files

    # Fundata tables
    'fund_general': ['FundGeneralSeed.csv'],
    'benchmark_general': ['BenchmarkGeneralSeed.csv'],
    'fund_daily_nav': ['FundDailyNAVPSSeed.csv'],
    'instrument_identifier': ['InstrumentIdentifierSeed.csv'],
    'fund_performance_summary': ['FundPerformanceSummarySeed.csv'],
    'fund_allocation': ['FundAllocationSeed.csv'],
    'fund_expenses': ['FundExpensesSeed.csv'],
    'fund_yearly_performance': ['FundYearlyPerformanceSeed.csv'],
    'fund_quotes': ['FundDailyNAVPSSeed.csv', 'Pricing2015to2025/Pricing2015to2017.csv',
                   'Pricing2015to2025/Pricing2018to2019.csv', 'Pricing2015to2025/Pricing2020to2021.csv',
                   'Pricing2015to2025/Pricing2022to2023.csv', 'Pricing2015to2025/Pricing2024to2025.csv'],
}

# Get all quote files dynamically from B2 (will be populated at runtime)
ALL_QUOTE_FILES = []

def ensure_directories():
    """Create necessary directories."""
    logger.info(f"Creating directories in CWD: {os.getcwd()}")
    os.makedirs('fmp_data', exist_ok=True)
    os.makedirs('fmp_data/equity_quotes', exist_ok=True)
    os.makedirs('fmp_data/etfs_quotes', exist_ok=True)
    os.makedirs('fundata', exist_ok=True)
    os.makedirs('fundata/data', exist_ok=True)
    os.makedirs('fundata/quotes', exist_ok=True)
    os.makedirs('fundata/quotes/Pricing2015to2025', exist_ok=True)

    # Verify the directories actually exist
    equity_quotes_path = 'fmp_data/equity_quotes'
    if os.path.exists(equity_quotes_path):
        logger.info(f"✓ Created equity_quotes at: {equity_quotes_path}")
    else:
        logger.error(f"✗ Failed to create equity_quotes at: {equity_quotes_path}")

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
        # Debug: Show where we're downloading to
        logger.info(f"Download target: {local_path}, CWD: {os.getcwd()}")

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
            # Verify the file actually exists after download
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                logger.info(f"✓ Downloaded: {filename} - Verified at {local_path} ({file_size} bytes)")
            else:
                logger.error(f"✗ CRITICAL: B2 download reported success but file NOT FOUND at: {local_path}")
                logger.error(f"  CWD: {os.getcwd()}")
                logger.error(f"  Expected path: {local_path}")
                # List directory contents to debug
                parent_dir = os.path.dirname(local_path)
                if os.path.exists(parent_dir):
                    contents = os.listdir(parent_dir)
                    logger.error(f"  Directory {parent_dir} contains: {contents[:5]}...")
                else:
                    logger.error(f"  Parent directory {parent_dir} does not exist!")
                return False
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

def download_csv_files_for_tables(under_seeded_tables: List[str]) -> bool:
    """Download only the CSV files needed for specific under-seeded tables."""
    logger.info(f"=== Starting Selective CSV Download for {len(under_seeded_tables)} tables ===")
    logger.info(f"Under-seeded tables: {under_seeded_tables}")

    # Ensure directories exist
    ensure_directories()

    # Check B2 authentication
    if not check_b2_auth():
        logger.error("B2 authentication required. Please run: b2 account authorize")
        return False

    # Determine which files need to be downloaded
    fmp_files_needed = set()
    fundata_files_needed = set()
    need_equity_quotes = False
    need_etfs_quotes = False

    for table in under_seeded_tables:
        if table in TABLE_TO_CSV_MAPPING:
            csv_files = TABLE_TO_CSV_MAPPING[table]

            if csv_files == 'equity_quotes_files':
                need_equity_quotes = True
            elif csv_files == 'etfs_quotes_files':
                need_etfs_quotes = True
            elif isinstance(csv_files, list):
                for csv_file in csv_files:
                    # Determine if it's FMP or fundata based on file name
                    if csv_file.startswith(('Fund', 'Benchmark', 'Instrument', 'Pricing')):
                        fundata_files_needed.add(csv_file)
                    else:
                        fmp_files_needed.add(csv_file)

    logger.info(f"Files to download - FMP: {len(fmp_files_needed)}, Fundata: {len(fundata_files_needed)}, "
                f"Equity quotes: {need_equity_quotes}, ETF quotes: {need_etfs_quotes}")

    total_success = 0
    total_files = 0

    # Download specific FMP files
    if fmp_files_needed:
        fmp_success, fmp_total = download_specific_fmp_files(list(fmp_files_needed))
        total_success += fmp_success
        total_files += fmp_total

    # Download quote files if needed
    if need_equity_quotes or need_etfs_quotes:
        quote_success, quote_total = download_specific_quotes(need_equity_quotes, need_etfs_quotes)
        total_success += quote_success
        total_files += quote_total

    # Download fundata files if needed
    if fundata_files_needed:
        fundata_success, fundata_total = download_specific_fundata_files(list(fundata_files_needed))
        total_success += fundata_success
        total_files += fundata_total

    logger.info(f"=== Selective Download Summary: {total_success}/{total_files} files downloaded ===")

    # Consider successful if we got at least 80% of what we tried to download
    success_rate = total_success / total_files if total_files > 0 else 1.0
    return success_rate >= 0.8

def download_specific_fmp_files(files_needed: List[str]) -> Tuple[int, int]:
    """Download specific FMP files from the core files list."""
    logger.info(f"Downloading {len(files_needed)} specific FMP files: {files_needed}")

    success_count = 0
    total_count = len(files_needed)

    for filename in files_needed:
        try:
            result = subprocess.run(['b2', 'file', 'download', f'b2://sra-data-csv/{filename}', f'fmp_data/{filename}'],
                                   capture_output=True, text=True, check=True)
            logger.info(f"Downloaded: {filename}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to download {filename}: {e.stderr}")
        except Exception as e:
            logger.warning(f"Error downloading {filename}: {e}")

    return success_count, total_count

def download_specific_quotes(need_equity: bool, need_etfs: bool) -> Tuple[int, int]:
    """Download specific quote files based on what's needed."""
    logger.info(f"Downloading quotes - Equity: {need_equity}, ETFs: {need_etfs}")

    success_count = 0
    total_count = 0

    try:
        # Get list of available quote files from B2
        result = subprocess.run(['b2', 'ls', '--recursive', 'b2://sra-data-csv'],
                               capture_output=True, text=True, check=True)

        equity_files = []
        etf_files = []

        for line in result.stdout.splitlines():
            if line and 'quote' in line.lower():
                parts = line.strip().split()
                if parts:
                    filename = parts[-1]
                    if filename.startswith('equity_quote_'):
                        equity_files.append(filename)
                    elif filename.startswith('etfs_quote_'):
                        etf_files.append(filename)

        files_to_download = []
        if need_equity:
            files_to_download.extend(equity_files)
        if need_etfs:
            files_to_download.extend(etf_files)

        total_count = len(files_to_download)

        for filename in files_to_download:
            try:
                # Determine target directory (files are named equity_quote_YYYY.csv not equity_quotes_)
                target_dir = 'fmp_data/equity_quotes' if filename.startswith('equity_quote_') else 'fmp_data/etfs_quotes'
                target_path = f'{target_dir}/{filename}'

                result = subprocess.run(['b2', 'file', 'download', f'b2://sra-data-csv/{filename}', target_path],
                                       capture_output=True, text=True, check=True)
                logger.info(f"Downloaded: {filename}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to download {filename}: {e.stderr}")
            except Exception as e:
                logger.warning(f"Error downloading {filename}: {e}")

    except Exception as e:
        logger.error(f"Error getting quote file list: {e}")

    return success_count, total_count

def download_specific_fundata_files(files_needed: List[str]) -> Tuple[int, int]:
    """Download specific fundata files."""
    logger.info(f"Downloading {len(files_needed)} specific fundata files: {files_needed}")

    success_count = 0
    total_count = len(files_needed)

    for filename in files_needed:
        try:
            # Determine target directory based on file type
            if filename.startswith('Pricing'):
                target_path = f'fundata/quotes/{filename}'
            else:
                target_path = f'fundata/data/{filename}'

            result = subprocess.run(['b2', 'file', 'download', f'b2://sra-data-csv/{filename}', target_path],
                                   capture_output=True, text=True, check=True)
            logger.info(f"Downloaded: {filename}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to download {filename}: {e.stderr}")
        except Exception as e:
            logger.warning(f"Error downloading {filename}: {e}")

    return success_count, total_count

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