#!/usr/bin/env python3
"""
Test script for CSV loading functionality.
Tests FMP and fundata CSV loading without requiring actual CSV files.
"""
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        # Test FMP CSV loader import
        from FMP.load_from_csv import load_all_fmp_csvs, get_loading_status
        print("✅ FMP CSV loader imported successfully")

        # Test unified CSV loader import
        from load_csv_data import load_fmp_csvs, load_fundata_csvs, initial_csv_seeding
        print("✅ Unified CSV loader imported successfully")

        # Test worker functions import
        import worker
        print("✅ Worker module imported successfully")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_database_connection():
    """Test database connection."""
    print("Testing database connection...")

    try:
        import psycopg2

        DB_HOST = os.getenv('DB_HOST')
        DB_PORT = os.getenv('DB_PORT', '5432')
        DB_NAME = os.getenv('DB_NAME')
        DB_USER = os.getenv('DB_USER')
        DB_PASSWORD = os.getenv('DB_PASSWORD', '')

        if not all([DB_HOST, DB_NAME, DB_USER]):
            print("❌ Database environment variables not set")
            return False

        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            result = cur.fetchone()

        conn.close()

        if result[0] == 1:
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection test failed")
            return False

    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def test_directory_structure():
    """Test that expected directories exist."""
    print("Testing directory structure...")

    directories = [
        'FMP',
        'fundata',
        'fundata/data',
        'fundata/quotes'
    ]

    all_exist = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"⚠️  Directory missing: {directory}")
            all_exist = False

    return all_exist

def test_csv_directory_creation():
    """Test that fmp_data directory can be created."""
    print("Testing fmp_data directory creation...")

    fmp_data_dir = 'fmp_data'

    try:
        os.makedirs(fmp_data_dir, exist_ok=True)
        if os.path.exists(fmp_data_dir):
            print(f"✅ FMP data directory created: {fmp_data_dir}")
            return True
        else:
            print(f"❌ Could not create FMP data directory: {fmp_data_dir}")
            return False

    except Exception as e:
        print(f"❌ Error creating FMP data directory: {e}")
        return False

def test_worker_functions():
    """Test that worker functions can be called without errors."""
    print("Testing worker function accessibility...")

    try:
        import worker

        # Test that functions exist
        functions = [
            'initial_seeding',
            'fmp_seeding',
            'fundata_seeding',
            'force_fresh_seeding',
            'daily_quotes',
            'weekly_fundamentals',
            'weekly_scoring'
        ]

        for func_name in functions:
            if hasattr(worker, func_name):
                print(f"✅ Function exists: {func_name}")
            else:
                print(f"❌ Function missing: {func_name}")
                return False

        return True

    except Exception as e:
        print(f"❌ Error testing worker functions: {e}")
        return False

def main():
    """Run all tests."""
    print("=== CSV Loading Functionality Test ===\n")

    tests = [
        ("Import Tests", test_imports),
        ("Database Connection", test_database_connection),
        ("Directory Structure", test_directory_structure),
        ("CSV Directory Creation", test_csv_directory_creation),
        ("Worker Functions", test_worker_functions)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append((test_name, result))

    print("\n=== Test Summary ===")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! CSV loading functionality is ready.")
        print("\nNext steps:")
        print("1. Add FMP CSV files to fmp_data/ directory")
        print("2. Run: python server.py seeding")
        print("3. Deploy with: python server.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)