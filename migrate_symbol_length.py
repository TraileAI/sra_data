#!/usr/bin/env python3
"""
Migration script to fix VARCHAR(10) symbol fields that are too small for Canadian symbols.
Canadian symbols like 'ENB-PFA.TO' are 10 characters, and some are up to 13 characters.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def run_migration():
    """Run database migration to expand symbol field sizes."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    try:
        with conn.cursor() as cur:
            print("🔧 Running symbol field migration...")

            # Check if tables exist and alter them
            migrations = [
                ("equity_profile", "symbol", "VARCHAR(15)"),
                ("equity_profile", "currency", "VARCHAR(15)"),
                ("equity_income", "symbol", "VARCHAR(15)"),
                ("equity_income", "cik", "VARCHAR(15)"),
                ("equity_peers", "symbol", "VARCHAR(15)"),
                ("equity_peers", "peer_symbol", "VARCHAR(15)"),
                ("etfs_peers", "symbol", "VARCHAR(15)"),
                ("etfs_peers", "peer_symbol", "VARCHAR(15)"),
                ("equity_quotes", "symbol", "VARCHAR(15)"),  # Just in case
            ]

            for table_name, column_name, new_type in migrations:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    )
                """, (table_name,))

                if cur.fetchone()[0]:
                    # Check current column type
                    cur.execute("""
                        SELECT data_type, character_maximum_length
                        FROM information_schema.columns
                        WHERE table_name = %s AND column_name = %s
                    """, (table_name, column_name))

                    result = cur.fetchone()
                    if result:
                        current_type, max_length = result
                        if current_type == 'character varying' and max_length == 10:
                            print(f"📝 Altering {table_name}.{column_name} from VARCHAR(10) to {new_type}")
                            cur.execute(f"ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE {new_type}")
                        else:
                            print(f"✅ {table_name}.{column_name} already correct: {current_type}({max_length})")
                    else:
                        print(f"⚠️ Column {table_name}.{column_name} not found")
                else:
                    print(f"⚠️ Table {table_name} not found (will be created with correct schema)")

            conn.commit()
            print("✅ Migration completed successfully")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()