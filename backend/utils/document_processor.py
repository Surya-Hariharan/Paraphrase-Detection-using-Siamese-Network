"""
Document Processor - Extract text from various file formats
============================================================

Supports: .txt, .pdf, .docx

Usage:
    from backend.utils.document_processor import extract_text_from_file
    
    text = extract_text_from_file("document.pdf")
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional
import tempfile
import os


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as string
    """
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from TXT file.
    
    Args:
        file_path: Path to the TXT file
        
    Returns:
        File content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"Failed to read TXT file: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from DOCX file using python-docx.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text as string
    """
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except ImportError:
        raise ImportError("python-docx is not installed. Install with: pip install python-docx")
    except Exception as e:
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from file based on extension.
    
    Supported formats: .txt, .pdf, .docx
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text as string
        
    Raises:
        ValueError: If file format is not supported or extraction fails
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = path.suffix.lower()
    
    if extension == '.txt':
        return extract_text_from_txt(file_path)
    elif extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {extension}. Supported: .txt, .pdf, .docx")


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from file bytes (for file uploads).
    
    Args:
        file_bytes: File content as bytes
        filename: Original filename (used to determine file type)
        
    Returns:
        Extracted text as string
    """
    # Create temporary file
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    
    try:
        # Extract text from temporary file
        text = extract_text_from_file(tmp_path)
        return text
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def validate_text(text: str, min_length: int = 10, max_length: int = 100000) -> tuple[bool, Optional[str]]:
    """
    Validate extracted text.
    
    Args:
        text: Extracted text to validate
        min_length: Minimum acceptable text length
        max_length: Maximum acceptable text length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Extracted text is empty"
    
    text_length = len(text.strip())
    
    if text_length < min_length:
        return False, f"Text too short (minimum {min_length} characters)"
    
    if text_length > max_length:
        return False, f"Text too long (maximum {max_length} characters)"
    
    return True, None
