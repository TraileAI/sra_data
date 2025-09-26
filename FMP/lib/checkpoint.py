"""
Checkpoint and resume system for FMP data processing.
Uses module-level state and file-based persistence.
"""
import os
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from threading import Lock

from .resource_monitor import wait_for_system_settled, get_resource_stats

# Checkpoint storage directory
CHECKPOINT_DIR = "/var/data/checkpoints"

# Module-level state
_checkpoint_lock = Lock()
_last_checkpoint_time = 0.0
_checkpoint_interval = 300.0  # 5 minutes

@dataclass
class ProcessingCheckpoint:
    """Immutable checkpoint data structure."""
    script_name: str
    start_time: float
    last_update: float
    api_calls_made: int
    last_symbol: Optional[str]
    last_date: Optional[str]
    completed_symbols: List[str]
    failed_symbols: List[str]
    total_symbols: int
    progress_percent: float
    resource_stats: Dict[str, Any]

def ensure_checkpoint_directory():
    """Ensure checkpoint directory exists."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_checkpoint_path(script_name: str) -> str:
    """Get checkpoint file path for script."""
    ensure_checkpoint_directory()
    safe_name = script_name.replace('/', '_').replace('.py', '')
    return os.path.join(CHECKPOINT_DIR, f"{safe_name}_checkpoint.json")

def create_checkpoint(
    script_name: str,
    api_calls_made: int = 0,
    last_symbol: Optional[str] = None,
    last_date: Optional[str] = None,
    completed_symbols: Optional[List[str]] = None,
    failed_symbols: Optional[List[str]] = None,
    total_symbols: int = 0
) -> ProcessingCheckpoint:
    """Create a new checkpoint."""

    completed = completed_symbols or []
    failed = failed_symbols or []
    progress = (len(completed) / total_symbols * 100) if total_symbols > 0 else 0.0

    return ProcessingCheckpoint(
        script_name=script_name,
        start_time=time.time(),
        last_update=time.time(),
        api_calls_made=api_calls_made,
        last_symbol=last_symbol,
        last_date=last_date,
        completed_symbols=completed,
        failed_symbols=failed,
        total_symbols=total_symbols,
        progress_percent=progress,
        resource_stats=get_resource_stats()
    )

def save_checkpoint(checkpoint: ProcessingCheckpoint) -> bool:
    """Save checkpoint to disk."""
    global _last_checkpoint_time

    current_time = time.time()

    # Rate limit checkpoint saves (every 5 minutes or on significant progress)
    if (current_time - _last_checkpoint_time < _checkpoint_interval and
        checkpoint.progress_percent % 10 != 0):  # Save on 10% increments regardless
        return True

    with _checkpoint_lock:
        try:
            checkpoint_path = get_checkpoint_path(checkpoint.script_name)

            # Update timestamp
            updated_checkpoint = ProcessingCheckpoint(
                script_name=checkpoint.script_name,
                start_time=checkpoint.start_time,
                last_update=current_time,
                api_calls_made=checkpoint.api_calls_made,
                last_symbol=checkpoint.last_symbol,
                last_date=checkpoint.last_date,
                completed_symbols=checkpoint.completed_symbols,
                failed_symbols=checkpoint.failed_symbols,
                total_symbols=checkpoint.total_symbols,
                progress_percent=checkpoint.progress_percent,
                resource_stats=get_resource_stats()
            )

            with open(checkpoint_path, 'w') as f:
                json.dump(asdict(updated_checkpoint), f, indent=2)

            _last_checkpoint_time = current_time
            return True

        except Exception as e:
            print(f"Error saving checkpoint for {checkpoint.script_name}: {e}")
            return False

def load_checkpoint(script_name: str) -> Optional[ProcessingCheckpoint]:
    """Load checkpoint from disk."""
    try:
        checkpoint_path = get_checkpoint_path(script_name)

        if not os.path.exists(checkpoint_path):
            return None

        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        return ProcessingCheckpoint(**data)

    except Exception as e:
        print(f"Error loading checkpoint for {script_name}: {e}")
        return None

def should_resume_processing(script_name: str, max_age_hours: float = 24.0) -> bool:
    """Check if processing should resume from checkpoint."""
    checkpoint = load_checkpoint(script_name)

    if not checkpoint:
        return False

    # Check if checkpoint is too old
    age_seconds = time.time() - checkpoint.last_update
    if age_seconds > (max_age_hours * 3600):
        print(f"Checkpoint for {script_name} is too old ({age_seconds/3600:.1f} hours)")
        return False

    # Check if processing was completed
    if checkpoint.progress_percent >= 100.0:
        print(f"Processing for {script_name} already completed")
        return False

    return True

def wait_for_recovery(script_name: str) -> bool:
    """
    Wait for system to settle before resuming from checkpoint.
    Returns True when safe to proceed.
    """
    checkpoint = load_checkpoint(script_name)

    if not checkpoint:
        return True  # No checkpoint, can start immediately

    print(f"Checkpoint found for {script_name}")
    print(f"Progress: {checkpoint.progress_percent:.1f}% "
          f"({len(checkpoint.completed_symbols)}/{checkpoint.total_symbols} symbols)")
    print(f"Last API calls: {checkpoint.api_calls_made}")
    print(f"Failed symbols: {len(checkpoint.failed_symbols)}")

    # If checkpoint is recent, wait for system to settle
    time_since_checkpoint = time.time() - checkpoint.last_update
    if time_since_checkpoint < 300:  # Less than 5 minutes ago
        print("Recent checkpoint detected. Waiting for system to settle...")
        return wait_for_system_settled()

    # Old checkpoint, safe to proceed
    return True

def cleanup_checkpoint(script_name: str) -> bool:
    """Remove checkpoint file after successful completion."""
    try:
        checkpoint_path = get_checkpoint_path(script_name)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"Cleaned up checkpoint for {script_name}")
        return True
    except Exception as e:
        print(f"Error cleaning up checkpoint for {script_name}: {e}")
        return False

def get_all_checkpoints() -> List[ProcessingCheckpoint]:
    """Get all existing checkpoints."""
    checkpoints = []

    if not os.path.exists(CHECKPOINT_DIR):
        return checkpoints

    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.endswith('_checkpoint.json'):
            script_name = filename.replace('_checkpoint.json', '').replace('_', '/')
            checkpoint = load_checkpoint(script_name)
            if checkpoint:
                checkpoints.append(checkpoint)

    return checkpoints

def update_checkpoint_progress(
    checkpoint: ProcessingCheckpoint,
    new_api_calls: int = 0,
    new_symbol: Optional[str] = None,
    new_date: Optional[str] = None,
    completed_symbol: Optional[str] = None,
    failed_symbol: Optional[str] = None
) -> ProcessingCheckpoint:
    """Update checkpoint with new progress."""

    completed = checkpoint.completed_symbols.copy()
    failed = checkpoint.failed_symbols.copy()

    if completed_symbol and completed_symbol not in completed:
        completed.append(completed_symbol)

    if failed_symbol and failed_symbol not in failed:
        failed.append(failed_symbol)

    progress = (len(completed) / checkpoint.total_symbols * 100) if checkpoint.total_symbols > 0 else 0.0

    return ProcessingCheckpoint(
        script_name=checkpoint.script_name,
        start_time=checkpoint.start_time,
        last_update=time.time(),
        api_calls_made=checkpoint.api_calls_made + new_api_calls,
        last_symbol=new_symbol or checkpoint.last_symbol,
        last_date=new_date or checkpoint.last_date,
        completed_symbols=completed,
        failed_symbols=failed,
        total_symbols=checkpoint.total_symbols,
        progress_percent=progress,
        resource_stats=get_resource_stats()
    )