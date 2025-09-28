#!/usr/bin/env python3
"""Test loading only FMP data to avoid disk space issues"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from FMP.load_from_csv import load_all_fmp_csvs
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Load only FMP data, skipping Fundata to avoid disk space issues"""
    logger.info("=== Testing FMP Data Loading Only ===")

    try:
        # Call the FMP loading function directly
        success = load_all_fmp_csvs()

        if success:
            logger.info("✅ FMP data loading completed successfully")
        else:
            logger.error("❌ FMP data loading failed")

        return success

    except Exception as e:
        logger.error(f"Error during FMP loading: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)