"""
CSV buffer manager with PostgreSQL COPY for efficient batch loading.
Uses module-level state for memory efficiency.
"""
import os
import csv
import io
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
import tempfile

from .resource_monitor import calculate_optimal_batch_size, check_resources, wait_for_resources

# Module-level buffer state
_csv_buffers: Dict[str, io.StringIO] = {}
_row_counts: Dict[str, int] = {}
_batch_limits: Dict[str, int] = {}
_csv_writers: Dict[str, csv.writer] = {}
_temp_files: Dict[str, str] = {}

# CSV storage directory
CSV_STORAGE_DIR = "/var/data"

def ensure_csv_directory():
    """Ensure CSV storage directory exists."""
    os.makedirs(CSV_STORAGE_DIR, exist_ok=True)

def initialize_buffer(table_name: str, columns: List[str]) -> None:
    """Initialize CSV buffer for a table."""
    global _csv_buffers, _row_counts, _batch_limits, _csv_writers, _temp_files

    # Calculate optimal batch size
    batch_sizes = calculate_optimal_batch_size()
    _batch_limits[table_name] = batch_sizes.get(table_name, 5000)

    # Create string buffer
    _csv_buffers[table_name] = io.StringIO()
    _row_counts[table_name] = 0

    # Create CSV writer
    _csv_writers[table_name] = csv.writer(_csv_buffers[table_name])

    # Write header
    _csv_writers[table_name].writerow(columns)

    # Create temp file path for disk overflow
    ensure_csv_directory()
    temp_fd, temp_path = tempfile.mkstemp(
        suffix=f'_{table_name}.csv',
        dir=CSV_STORAGE_DIR,
        text=True
    )
    os.close(temp_fd)  # Close file descriptor, keep path
    _temp_files[table_name] = temp_path

def add_row(table_name: str, row_data: Dict[str, Any]) -> bool:
    """
    Add a row to the CSV buffer.
    Returns True if buffer was flushed, False otherwise.
    """
    global _csv_buffers, _row_counts, _csv_writers

    if table_name not in _csv_buffers:
        raise ValueError(f"Buffer not initialized for table {table_name}")

    # Wait for resources if needed
    if not wait_for_resources(max_wait=60.0):
        return False

    # Convert row data to list (maintaining column order)
    writer = _csv_writers[table_name]
    writer.writerow(row_data.values())
    _row_counts[table_name] += 1

    # Check if we need to flush
    if _row_counts[table_name] >= _batch_limits[table_name]:
        return flush_buffer(table_name)

    return False

def flush_buffer(table_name: str, engine=None) -> bool:
    """
    Flush CSV buffer to database using COPY.
    Returns True on success, False on failure.
    """
    global _csv_buffers, _row_counts, _temp_files

    if table_name not in _csv_buffers or _row_counts[table_name] == 0:
        return True

    try:
        # Write buffer to temp file
        temp_file = _temp_files[table_name]
        with open(temp_file, 'w', newline='') as f:
            f.write(_csv_buffers[table_name].getvalue())

        # Use COPY to load data
        if engine:
            success = _copy_from_file(temp_file, table_name, engine)
        else:
            success = False

        if success:
            # Clear buffer after successful flush
            _csv_buffers[table_name] = io.StringIO()
            _csv_writers[table_name] = csv.writer(_csv_buffers[table_name])
            _row_counts[table_name] = 0

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        return success

    except Exception as e:
        print(f"Error flushing buffer for {table_name}: {e}")
        return False

def _copy_from_file(file_path: str, table_name: str, engine) -> bool:
    """Execute COPY FROM file using raw psycopg2 connection."""
    try:
        # Get raw connection from SQLAlchemy engine
        with engine.raw_connection() as conn:
            with conn.cursor() as cur:
                with open(file_path, 'r') as f:
                    # Skip header line
                    next(f)
                    # Use COPY FROM
                    cur.copy_from(
                        f,
                        table_name,
                        sep=',',
                        null='',
                        columns=None  # Use all columns in order
                    )
                conn.commit()
        return True
    except Exception as e:
        print(f"COPY FROM failed for {table_name}: {e}")
        return False

def flush_all_buffers(engine) -> Dict[str, bool]:
    """Flush all active buffers to database."""
    results = {}
    for table_name in list(_csv_buffers.keys()):
        results[table_name] = flush_buffer(table_name, engine)
    return results

def get_buffer_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all buffers."""
    stats = {}
    for table_name in _csv_buffers:
        buffer_size_mb = len(_csv_buffers[table_name].getvalue().encode('utf-8')) / (1024 * 1024)
        stats[table_name] = {
            'row_count': _row_counts.get(table_name, 0),
            'batch_limit': _batch_limits.get(table_name, 0),
            'buffer_size_mb': buffer_size_mb,
            'fill_percentage': (_row_counts.get(table_name, 0) / _batch_limits.get(table_name, 1)) * 100
        }
    return stats

def cleanup_buffers():
    """Clean up all buffers and temp files."""
    global _csv_buffers, _row_counts, _batch_limits, _csv_writers, _temp_files

    # Clean up temp files
    for temp_file in _temp_files.values():
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

    # Clear all module state
    _csv_buffers.clear()
    _row_counts.clear()
    _batch_limits.clear()
    _csv_writers.clear()
    _temp_files.clear()

@contextmanager
def csv_buffer_context(table_name: str, columns: List[str], engine):
    """Context manager for CSV buffer operations."""
    try:
        initialize_buffer(table_name, columns)
        yield
    finally:
        # Flush remaining data
        flush_buffer(table_name, engine)
        # Clean up this buffer
        if table_name in _temp_files:
            temp_file = _temp_files[table_name]
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            del _temp_files[table_name]
        if table_name in _csv_buffers:
            del _csv_buffers[table_name]
        if table_name in _row_counts:
            del _row_counts[table_name]
        if table_name in _batch_limits:
            del _batch_limits[table_name]
        if table_name in _csv_writers:
            del _csv_writers[table_name]