"""Retry service with exponential backoff.

This module provides specialized retry logic for data processing operations
with configurable backoff strategies and error handling.
"""

from .data_processing import RetryService

# Re-export the retry service
__all__ = ['RetryService']