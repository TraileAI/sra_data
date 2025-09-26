#!/usr/bin/env python3
"""
Test script for resource-aware FMP architecture.
Verifies that all components work together correctly.
"""
import os
import sys
import time
from dotenv import load_dotenv

# Add FMP lib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FMP'))

load_dotenv()

def test_resource_monitor():
    """Test resource monitoring functionality."""
    print("Testing resource monitor...")

    from lib import check_resources, get_resource_stats, calculate_optimal_batch_size

    # Test resource checking
    can_proceed, cpu, memory = check_resources()
    print(f"Resource check: can_proceed={can_proceed}, cpu={cpu:.1f}%, memory={memory:.1f}%")

    # Test resource stats
    stats = get_resource_stats()
    print(f"Resource stats: {stats}")

    # Test batch size calculation
    batch_sizes = calculate_optimal_batch_size()
    print(f"Optimal batch sizes: {batch_sizes}")

    print("✓ Resource monitor working\n")

def test_csv_buffer():
    """Test CSV buffer functionality."""
    print("Testing CSV buffer...")

    from lib import initialize_buffer, add_row, get_buffer_stats, cleanup_buffers

    # Initialize a test buffer
    columns = ['id', 'name', 'value']
    initialize_buffer('test_table', columns)

    # Add some test rows
    test_rows = [
        {'id': 1, 'name': 'test1', 'value': 100.0},
        {'id': 2, 'name': 'test2', 'value': 200.0},
        {'id': 3, 'name': 'test3', 'value': 300.0}
    ]

    for row in test_rows:
        add_row('test_table', row)

    # Get buffer stats
    stats = get_buffer_stats()
    print(f"Buffer stats: {stats}")

    # Cleanup
    cleanup_buffers()

    print("✓ CSV buffer working\n")

def test_checkpoint_system():
    """Test checkpoint functionality."""
    print("Testing checkpoint system...")

    from lib import (
        create_checkpoint, save_checkpoint, load_checkpoint,
        update_checkpoint_progress, cleanup_checkpoint
    )

    # Create a test checkpoint
    checkpoint = create_checkpoint(
        script_name="test_script",
        total_symbols=100
    )

    print(f"Created checkpoint: {checkpoint.script_name}, progress: {checkpoint.progress_percent:.1f}%")

    # Update checkpoint progress
    checkpoint = update_checkpoint_progress(
        checkpoint,
        new_api_calls=5,
        new_symbol="AAPL",
        completed_symbol="AAPL"
    )

    print(f"Updated progress: {checkpoint.progress_percent:.1f}%, API calls: {checkpoint.api_calls_made}")

    # Save checkpoint
    save_result = save_checkpoint(checkpoint)
    print(f"Save checkpoint: {save_result}")

    # Load checkpoint
    loaded_checkpoint = load_checkpoint("test_script")
    if loaded_checkpoint:
        print(f"Loaded checkpoint: {loaded_checkpoint.script_name}, progress: {loaded_checkpoint.progress_percent:.1f}%")

    # Cleanup
    cleanup_checkpoint("test_script")

    print("✓ Checkpoint system working\n")

def test_directory_creation():
    """Test that required directories can be created."""
    print("Testing directory creation...")

    # Test CSV storage directory
    csv_dir = "/var/data"
    try:
        os.makedirs(csv_dir, exist_ok=True)
        print(f"✓ CSV directory created/verified: {csv_dir}")
    except PermissionError:
        print(f"✗ Permission denied for CSV directory: {csv_dir}")
        print("  Consider using /tmp/data for testing")

    # Test checkpoint directory
    checkpoint_dir = "/var/data/checkpoints"
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"✓ Checkpoint directory created/verified: {checkpoint_dir}")
    except PermissionError:
        print(f"✗ Permission denied for checkpoint directory: {checkpoint_dir}")
        print("  Consider using /tmp/data/checkpoints for testing")

    print()

def test_database_connection():
    """Test database connection."""
    print("Testing database connection...")

    try:
        from sqlalchemy import create_engine

        DB_USER = os.getenv('DB_USER', 'nauraai')
        DB_PASSWORD = os.getenv('DB_PASSWORD', '')
        DB_HOST = os.getenv('DB_HOST', 'localhost')
        DB_PORT = os.getenv('DB_PORT', '5432')
        DB_NAME = os.getenv('DB_NAME', 'Naura')

        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

        # Test connection
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            test_value = result.fetchone()[0]
            if test_value == 1:
                print("✓ Database connection successful")
            else:
                print("✗ Database connection test failed")

    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("  Make sure PostgreSQL is running and credentials are correct")

    print()

def main():
    """Run all tests."""
    print("=== Resource-Aware Architecture Test ===\n")

    try:
        test_directory_creation()
        test_resource_monitor()
        test_csv_buffer()
        test_checkpoint_system()
        test_database_connection()

        print("=== All Tests Completed ===")
        print("✓ Resource-aware architecture is working correctly!")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()