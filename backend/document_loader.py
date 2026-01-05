"""
Document Loader - Load and Extract Text from Multiple File Formats

Supports:
- Text files (.txt)
- PDF files (.pdf)
- Automatic text extraction and preprocessing
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import re


def load_text_file(file_path: str) -> str:
    """Load text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()


def load_pdf_file(file_path: str) -> str:
    """Load text from a PDF file."""
    try:
        import PyPDF2
        
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return text.strip()
    
    except ImportError:
        raise ImportError(
            "PyPDF2 is required to load PDF files. "
            "Install with: pip install PyPDF2"
        )


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.,!?;:\-\'\"()]', '', text)
    return text.strip()


def load_document(file_path: str, clean: bool = True) -> str:
    """
    Load document from file (auto-detect format).
    
    Args:
        file_path: Path to the document file
        clean: Whether to clean the text
        
    Returns:
        Extracted text content
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Determine file type
    extension = file_path.suffix.lower()
    
    if extension == '.txt':
        text = load_text_file(str(file_path))
    elif extension == '.pdf':
        text = load_pdf_file(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {extension}")
    
    if clean:
        text = clean_text(text)
    
    return text


def load_two_documents_from_folder(
    folder_path: str,
    extensions: List[str] = ['.txt', '.pdf']
) -> Tuple[str, str, str, str]:
    """
    Load exactly two documents from a folder.
    
    Args:
        folder_path: Path to folder containing documents
        extensions: Allowed file extensions
        
    Returns:
        (doc_a_text, doc_b_text, doc_a_path, doc_b_path)
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    
    # Find all supported files
    files = []
    for ext in extensions:
        files.extend(list(folder.glob(f"*{ext}")))
    
    # Sort for consistency
    files = sorted(files)
    
    if len(files) < 2:
        raise ValueError(
            f"Need at least 2 documents in {folder_path}. "
            f"Found {len(files)} files with extensions {extensions}"
        )
    
    if len(files) > 2:
        print(f"Warning: Found {len(files)} files. Using first 2: {files[0].name}, {files[1].name}")
    
    # Load first two documents
    doc_a_path = str(files[0])
    doc_b_path = str(files[1])
    
    doc_a_text = load_document(doc_a_path)
    doc_b_text = load_document(doc_b_path)
    
    return doc_a_text, doc_b_text, doc_a_path, doc_b_path


def get_document_info(file_path: str) -> dict:
    """Get information about a document."""
    file_path = Path(file_path)
    
    text = load_document(str(file_path))
    
    return {
        'file_name': file_path.name,
        'file_path': str(file_path),
        'file_type': file_path.suffix,
        'file_size': file_path.stat().st_size,
        'text_length': len(text),
        'word_count': len(text.split()),
        'preview': text[:200] + '...' if len(text) > 200 else text
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = "datasets"
    
    print(f"\nLoading documents from: {folder}")
    print("-" * 50)
    
    try:
        doc_a, doc_b, path_a, path_b = load_two_documents_from_folder(folder)
        
        print(f"\nDocument A: {Path(path_a).name}")
        print(f"  Length: {len(doc_a)} characters")
        print(f"  Preview: {doc_a[:100]}...")
        
        print(f"\nDocument B: {Path(path_b).name}")
        print(f"  Length: {len(doc_b)} characters")
        print(f"  Preview: {doc_b[:100]}...")
        
    except Exception as e:
        print(f"\nError: {e}")
