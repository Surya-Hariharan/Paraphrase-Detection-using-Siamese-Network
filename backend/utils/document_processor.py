"""
Document Processor - Handle Large Documents with Chunking

Handles documents of any size by loading from various formats (PDF, DOCX, TXT),
chunking large documents, and aggregating embeddings for comparison.
"""

import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import torch

# Document loading
try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Text splitting
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    SPLITTER_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        SPLITTER_AVAILABLE = True
    except ImportError:
        SPLITTER_AVAILABLE = False


class DocumentProcessor:
    """
    Processes documents of any size for paraphrase detection.
    
    Features:
    - Load PDF, DOCX, TXT files
    - Chunk large documents (handles multi-page documents)
    - Preserve document structure
    - Aggregate embeddings for comparison
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        aggregation_method: str = "mean"
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            aggregation_method: How to combine chunk embeddings ("mean", "max", "weighted")
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.aggregation_method = aggregation_method
        
        # Initialize text splitter if available
        if SPLITTER_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        else:
            self.text_splitter = None
            print("langchain-text-splitters not available. Using simple chunking.")
    
    def load_document(self, file_path: Union[str, Path]) -> str:
        """
        Load document from file.
        
        Supports: .txt, .pdf, .docx
        
        Args:
            file_path: Path to the document
            
        Returns:
            Document text as string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        # Text file
        if extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # PDF file
        elif extension == '.pdf':
            if not PDF_AVAILABLE:
                raise ImportError("PyPDF2 not installed. Install with: pip install pypdf2")
            
            reader = PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())
            return "\n\n".join(text_parts)
        
        # DOCX file
        elif extension == '.docx':
            if not DOCX_AVAILABLE:
                raise ImportError("python-docx not installed. Install with: pip install python-docx")
            
            doc = DocxDocument(str(file_path))
            text_parts = [paragraph.text for paragraph in doc.paragraphs]
            return "\n\n".join(text_parts)
        
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        # Use advanced splitter if available
        if self.text_splitter:
            chunks = self.text_splitter.split_text(text)
            return chunks
        
        # Fallback: simple chunking
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def aggregate_embeddings(
        self,
        embeddings: List[np.ndarray],
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Aggregate multiple chunk embeddings into a single representation.
        
        Args:
            embeddings: List of embedding vectors
            method: Aggregation method (overrides default)
            
        Returns:
            Aggregated embedding
        """
        method = method or self.aggregation_method
        
        if len(embeddings) == 0:
            raise ValueError("No embeddings to aggregate")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Stack embeddings
        stacked = np.stack(embeddings, axis=0)
        
        # Aggregation methods
        if method == "mean":
            return np.mean(stacked, axis=0)
        
        elif method == "max":
            return np.max(stacked, axis=0)
        
        elif method == "weighted":
            # Weight by position (earlier chunks get higher weight)
            weights = np.linspace(1.0, 0.5, len(embeddings))
            weights = weights / weights.sum()
            return np.average(stacked, axis=0, weights=weights)
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_document_info(self, text: str) -> Dict[str, Any]:
        """Get information about a document."""
        chunks = self.chunk_text(text)
        
        return {
            "total_length": len(text),
            "num_chunks": len(chunks),
            "avg_chunk_length": np.mean([len(c) for c in chunks]) if chunks else 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "estimated_pages": len(text) / 3000  # Rough estimate
        }


class DocumentPairProcessor:
    """
    Process pairs of documents for comparison.
    
    Handles:
    - Loading document pairs
    - Chunking both documents
    - Preparing for embedding and comparison
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize with document processor."""
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
    
    def load_document_pair(
        self,
        doc_a_path: Union[str, Path],
        doc_b_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Load and process a pair of documents.
        
        Args:
            doc_a_path: Path to first document
            doc_b_path: Path to second document
            
        Returns:
            Dictionary with document texts, chunks, and metadata
        """
        # Load documents
        text_a = self.processor.load_document(doc_a_path)
        text_b = self.processor.load_document(doc_b_path)
        
        # Chunk documents
        chunks_a = self.processor.chunk_text(text_a)
        chunks_b = self.processor.chunk_text(text_b)
        
        # Get metadata
        info_a = self.processor.get_document_info(text_a)
        info_b = self.processor.get_document_info(text_b)
        
        return {
            "doc_a": {
                "path": str(doc_a_path),
                "text": text_a,
                "chunks": chunks_a,
                "info": info_a
            },
            "doc_b": {
                "path": str(doc_b_path),
                "text": text_b,
                "chunks": chunks_b,
                "info": info_b
            }
        }
    
    def prepare_for_training(
        self,
        text_a: str,
        text_b: str,
        label: int
    ) -> Dict[str, Any]:
        """
        Prepare a document pair for training.
        
        Args:
            text_a: First document text
            text_b: Second document text
            label: 1 for paraphrase/duplicate, 0 for unique
            
        Returns:
            Training example dictionary
        """
        # Chunk both documents
        chunks_a = self.processor.chunk_text(text_a)
        chunks_b = self.processor.chunk_text(text_b)
        
        return {
            "text_a": text_a,
            "text_b": text_b,
            "chunks_a": chunks_a,
            "chunks_b": chunks_b,
            "label": label,
            "num_chunks_a": len(chunks_a),
            "num_chunks_b": len(chunks_b)
        }


# Utility functions
def create_dataset_from_files(
    file_pairs: List[tuple],
    labels: List[int],
    chunk_size: int = 1000
) -> List[Dict[str, Any]]:
    """
    Create training dataset from file pairs.
    
    Args:
        file_pairs: List of (doc_a_path, doc_b_path) tuples
        labels: List of labels (1=paraphrase, 0=unique)
        chunk_size: Chunk size for processing
        
    Returns:
        List of training examples
    """
    processor = DocumentPairProcessor(chunk_size=chunk_size)
    dataset = []
    
    for (doc_a_path, doc_b_path), label in zip(file_pairs, labels):
        try:
            pair_data = processor.load_document_pair(doc_a_path, doc_b_path)
            
            example = {
                "text_a": pair_data["doc_a"]["text"],
                "text_b": pair_data["doc_b"]["text"],
                "chunks_a": pair_data["doc_a"]["chunks"],
                "chunks_b": pair_data["doc_b"]["chunks"],
                "label": label,
                "metadata": {
                    "file_a": str(doc_a_path),
                    "file_b": str(doc_b_path),
                    "info_a": pair_data["doc_a"]["info"],
                    "info_b": pair_data["doc_b"]["info"]
                }
            }
            dataset.append(example)
            
        except Exception as e:
            print(f"Error processing {doc_a_path}, {doc_b_path}: {e}")
            continue
    
    return dataset


# Testing
if __name__ == "__main__":
    print("\nTesting Document Processor")
    print("-" * 40)
    
    # Test text chunking
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
    
    # Sample long document
    sample_text = """
    This is a long document that spans multiple pages. It contains various paragraphs
    and sections that discuss different topics. The document processor should be able
    to handle this by chunking it into smaller pieces.
    
    In a real-world scenario, this could be a research paper, a legal document, or
    any other multi-page content that needs to be processed for paraphrase detection.
    
    The chunking strategy ensures that we don't lose context at chunk boundaries by
    using overlapping windows. This helps maintain semantic coherence across chunks.
    """ * 10  # Repeat to make it longer
    
    chunks = processor.chunk_text(sample_text)
    info = processor.get_document_info(sample_text)
    
    print(f"Document Info:")
    print(f"  Total length: {info['total_length']} characters")
    print(f"  Number of chunks: {info['num_chunks']}")
    print(f"  Average chunk length: {info['avg_chunk_length']:.0f} characters")
    print(f"  Estimated pages: {info['estimated_pages']:.1f}")
    
    print(f"\nFirst chunk preview:")
    print(f"  {chunks[0][:200]}...")
    
    print("\nDocument processor test complete.")
