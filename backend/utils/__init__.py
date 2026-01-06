"""
Utility modules for document processing and configuration.

This module contains helper functions and classes for document loading,
processing, and quick comparisons.
"""

from .document_loader import (
    load_document,
    load_text_file,
    load_pdf_file,
    load_two_documents_from_folder,
    clean_text,
    get_document_info,
)
from .document_processor import DocumentProcessor, DocumentPairProcessor

__all__ = [
    "load_document",
    "load_text_file",
    "load_pdf_file",
    "load_two_documents_from_folder",
    "clean_text",
    "get_document_info",
    "DocumentProcessor",
    "DocumentPairProcessor",
]
