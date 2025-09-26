"""
Module-level singleton for resource monitoring.
Uses contiguous arrays and immutable data for cache-friendly performance.
"""
import psutil
import time
import array
from typing import Tuple, Optional
from dataclasses import dataclass
from threading import Lock

# Module-level state using arrays for contiguous memory
_cpu_history = array.array('f', [0.0] * 60)  # 60 samples of CPU usage
_memory_history = array.array('f', [0.0] * 60)  # 60 samples of memory usage
_history_index = 0
_last_check = 0.0
_lock = Lock()

# Configuration (immutable)
@dataclass(frozen=True)
class ResourceLimits:
    max_cpu_percent: float = 70.0
    max_memory_percent: float = 75.0
    check_interval: float = 1.0
    history_size: int = 60
    settle_cpu_percent: float = 30.0
    settle_duration: float = 30.0

LIMITS = ResourceLimits()

def check_resources() -> Tuple[bool, float, float]:
    """
    Check if resources are within acceptable limits.
    Returns (can_proceed, cpu_percent, memory_percent)
    """
    global _history_index, _last_check

    current_time = time.time()
    if current_time - _last_check < LIMITS.check_interval:
        # Use cached values
        cpu_avg = sum(_cpu_history) / len(_cpu_history)
        mem_avg = sum(_memory_history) / len(_memory_history)
        return (
            cpu_avg < LIMITS.max_cpu_percent and mem_avg < LIMITS.max_memory_percent,
            cpu_avg,
            mem_avg
        )

    with _lock:
        # Update measurements
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        _cpu_history[_history_index] = cpu_percent
        _memory_history[_history_index] = memory_percent
        _history_index = (_history_index + 1) % LIMITS.history_size
        _last_check = current_time

        # Calculate rolling averages
        cpu_avg = sum(_cpu_history) / len(_cpu_history)
        mem_avg = sum(_memory_history) / len(_memory_history)

        return (
            cpu_avg < LIMITS.max_cpu_percent and mem_avg < LIMITS.max_memory_percent,
            cpu_avg,
            mem_avg
        )

def wait_for_resources(max_wait: float = 300.0) -> bool:
    """
    Wait until resources are available or timeout.
    Returns True if resources became available, False if timeout.
    """
    start_time = time.time()

    while time.time() - start_time < max_wait:
        can_proceed, cpu, mem = check_resources()
        if can_proceed:
            return True

        # Exponential backoff
        wait_time = min(30.0, 2.0 ** ((time.time() - start_time) / 30.0))
        time.sleep(wait_time)

    return False

def wait_for_system_settled() -> bool:
    """
    Wait for system to settle after resource contention.
    Returns True when CPU is below settle threshold for settle duration.
    """
    settle_start = None

    while True:
        _, cpu_percent, _ = check_resources()

        if cpu_percent < LIMITS.settle_cpu_percent:
            if settle_start is None:
                settle_start = time.time()
            elif time.time() - settle_start >= LIMITS.settle_duration:
                return True
        else:
            settle_start = None

        time.sleep(5.0)  # Check every 5 seconds

def get_resource_stats() -> dict:
    """Get current resource statistics."""
    _, cpu_avg, mem_avg = check_resources()
    available_memory = psutil.virtual_memory().available

    return {
        'cpu_percent': cpu_avg,
        'memory_percent': mem_avg,
        'available_memory_gb': available_memory / (1024**3),
        'cpu_limit': LIMITS.max_cpu_percent,
        'memory_limit': LIMITS.max_memory_percent,
        'is_settled': cpu_avg < LIMITS.settle_cpu_percent
    }

def calculate_optimal_batch_size() -> dict:
    """
    Calculate optimal batch sizes based on available memory.
    Returns dictionary with batch sizes for different table types.
    """
    available_memory = psutil.virtual_memory().available
    # Use 25% of available memory for CSV buffer (conservative)
    buffer_size = available_memory * 0.25

    # Row size estimates (bytes):
    # - equity_profile: ~2KB per row
    # - equity_quotes: ~200 bytes per row
    # - equity_income: ~1KB per row
    # - etfs_profile: ~2KB per row
    # - etfs_peers: ~50 bytes per row

    return {
        'equity_profile': max(1000, int(buffer_size / 2048)),      # Min 1K rows
        'equity_quotes': max(5000, int(buffer_size / 200)),        # Min 5K rows
        'equity_income': max(2000, int(buffer_size / 1024)),       # Min 2K rows
        'etfs_profile': max(1000, int(buffer_size / 2048)),        # Min 1K rows
        'etfs_peers': max(10000, int(buffer_size / 50)),           # Min 10K rows
        'buffer_size_mb': buffer_size / (1024*1024),
        'available_memory_gb': available_memory / (1024**3)
    }