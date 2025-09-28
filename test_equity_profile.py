#!/usr/bin/env python3
"""Test loading just equity_profile.csv to check CSV parsing"""

import os
import sys
import psycopg2
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables
    load_dotenv()

    conn_params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', 5432),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD', '')
    }

    logger.info(f"Connecting to database: {conn_params['host']}:{conn_params['port']}/{conn_params['database']} as {conn_params['user']}")

    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # Clear the table first
                cur.execute("DELETE FROM equity_profile")
                logger.info("Cleared equity_profile table")

                # Test loading with proper CSV format
                csv_path = "fmp_data/equity_profile.csv"
                copy_sql = """
                    COPY equity_profile FROM STDIN
                    WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                """
                logger.info("Starting COPY operation...")

                with open(csv_path, 'r', encoding='utf-8') as f:
                    cur.copy_expert(copy_sql, f)

                conn.commit()

                # Get row count
                cur.execute("SELECT COUNT(*) FROM equity_profile")
                count = cur.fetchone()[0]
                logger.info(f"Successfully loaded {count} rows into equity_profile")

                # Show sample data
                cur.execute("SELECT symbol, company_name, sector FROM equity_profile LIMIT 5")
                rows = cur.fetchall()
                logger.info("Sample data:")
                for row in rows:
                    logger.info(f"  {row[0]}: {row[1]} ({row[2]})")

    except Exception as e:
        logger.error(f"Error: {e}")
        return False

    logger.info("Test completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)