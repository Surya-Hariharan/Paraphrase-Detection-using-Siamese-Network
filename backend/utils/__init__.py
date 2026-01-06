"""
Utility modules for document processing and configuration.

This module contains helper functions and classes for document loading,
processing, and quick comparisons.
"""

from .document_loader import DocumentLoader
from .document_processor import DocumentProcessor
from .quick_compare import quick_compare
from .config import load_config

__all__ = [
    "DocumentLoader",
    "DocumentProcessor",
    "quick_compare",
    "load_config",
]
