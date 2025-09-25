"""CSV processing service for fundata files.

This module provides specialized CSV processing capabilities for fundata
data files and quotes files with validation and error handling.
"""

from .data_processing import CSVProcessingService

# Re-export the main CSV processing service
__all__ = ['CSVProcessingService']