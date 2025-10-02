#!/usr/bin/env python3
"""
Export all database tables to CSV files
"""
import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Output directory
OUTPUT_DIR = "/Users/adam/Library/CloudStorage/OneDrive-SharedLibraries-BucklerSolutionsInc/Buckler - Documents/Development/seeded_data_snapshot"

def get_all_tables(conn):
    """Get list of all tables in the database."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        return [row[0] for row in cur.fetchall()]

def export_table_to_csv(conn, table_name, output_dir):
    """Export a single table to CSV."""
    output_file = os.path.join(output_dir, f"{table_name}.csv")

    # Get row count first
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cur.fetchone()[0]

    print(f"📊 Exporting {table_name} ({row_count:,} rows)...", end=" ", flush=True)

    # Export using COPY command (much faster than pandas for large tables)
    with conn.cursor() as cur:
        with open(output_file, 'w') as f:
            cur.copy_expert(f"COPY {table_name} TO STDOUT WITH CSV HEADER", f)

    # Get file size
    file_size = os.path.getsize(output_file)
    size_mb = file_size / (1024 * 1024)

    print(f"✅ ({size_mb:.1f} MB)")
    return row_count, file_size

def main():
    print(f"🚀 Starting database export to: {OUTPUT_DIR}")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Connect to database
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    try:
        # Get all tables
        tables = get_all_tables(conn)
        print(f"📋 Found {len(tables)} tables to export")
        print()

        total_rows = 0
        total_size = 0

        # Export each table
        for i, table in enumerate(tables, 1):
            print(f"[{i}/{len(tables)}] ", end="")
            rows, size = export_table_to_csv(conn, table, OUTPUT_DIR)
            total_rows += rows
            total_size += size

        print()
        print("=" * 60)
        print(f"✅ Export completed successfully!")
        print(f"📊 Total tables: {len(tables)}")
        print(f"📊 Total rows: {total_rows:,}")
        print(f"📊 Total size: {total_size / (1024 * 1024):.1f} MB")
        print(f"📂 Location: {OUTPUT_DIR}")
        print("=" * 60)

    finally:
        conn.close()

if __name__ == "__main__":
    main()
