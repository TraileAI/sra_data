"""
FMP Library - Resource-aware data processing components.
"""

from .resource_monitor import (
    check_resources,
    wait_for_resources,
    wait_for_system_settled,
    get_resource_stats,
    calculate_optimal_batch_size
)

from .csv_buffer import (
    initialize_buffer,
    add_row,
    flush_buffer,
    flush_all_buffers,
    get_buffer_stats,
    cleanup_buffers,
    csv_buffer_context
)

from .checkpoint import (
    ProcessingCheckpoint,
    create_checkpoint,
    save_checkpoint,
    load_checkpoint,
    should_resume_processing,
    wait_for_recovery,
    cleanup_checkpoint,
    get_all_checkpoints,
    update_checkpoint_progress
)

__all__ = [
    # Resource monitoring
    'check_resources',
    'wait_for_resources',
    'wait_for_system_settled',
    'get_resource_stats',
    'calculate_optimal_batch_size',

    # CSV buffering
    'initialize_buffer',
    'add_row',
    'flush_buffer',
    'flush_all_buffers',
    'get_buffer_stats',
    'cleanup_buffers',
    'csv_buffer_context',

    # Checkpointing
    'ProcessingCheckpoint',
    'create_checkpoint',
    'save_checkpoint',
    'load_checkpoint',
    'should_resume_processing',
    'wait_for_recovery',
    'cleanup_checkpoint',
    'get_all_checkpoints',
    'update_checkpoint_progress'
]