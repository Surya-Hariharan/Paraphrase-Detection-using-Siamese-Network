"""
Document-Level Siamese Pipeline - Complete Implementation

ARCHITECTURE (MATCHES DIAGRAM EXACTLY):
    
    Doc A ──▶ Chunk ──▶ SBERT ──▶ Aggregate ──▶ NN ──▶ FV_A ──┐
                                                              ├──▶ Cosine Similarity ──▶ Threshold ──▶ Decision
    Doc B ──▶ Chunk ──▶ SBERT ──▶ Aggregate ──▶ NN ──▶ FV_B ──┘

STEP-BY-STEP EXPLANATION:

Step 1: CHUNKING
- Split long documents into smaller chunks (paragraphs or token windows)
- Preserve semantic coherence at chunk boundaries
- Handle multi-page documents

Step 2: SBERT ENCODING (FROZEN)
- Encode each chunk independently using pretrained SBERT
- SBERT weights are FROZEN (no training updates)
- Each chunk → 384-dim embedding vector

Step 3: AGGREGATION (Mean Pooling)
- Combine chunk embeddings into single document embedding
- Method: Mean pooling across all chunks
- Output: Single 384-dim vector per document

Step 4: NN PROJECTION (SIAMESE, TRAINABLE)
- Project document embedding through shared neural network
- Architecture: Dense → Tanh → Dense
- ONE network for both Doc A and Doc B (Siamese)
- Output: Task-specific feature vectors FV_A and FV_B (256-dim)

Step 5: COSINE SIMILARITY (L)
- Compute cosine similarity between FV_A and FV_B
- Range: [-1.0, 1.0]
- Semantic similarity measure

Step 6: THRESHOLD DECISION
- Apply configurable threshold
- if similarity ≥ threshold → Paraphrase
- else → Not Paraphrase
- NO classifier layer, NO softmax, NO cross-entropy

This is a SEMANTIC SIMILARITY system, not generative AI.
No RL, no online learning, no black-box decisions.
Explainable threshold-based approach suitable for academic evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from pathlib import Path

from neural_engine import TrainableSiameseModel, SiameseProjectionModel
from document_processor import DocumentProcessor


class DocumentLevelSiameseModel:
    """
    Complete document-level paraphrase detection pipeline.
    
    This class implements the full architecture from the diagram:
    Document → Chunk → SBERT (frozen) → Aggregate → NN (Siamese) → FV → Cosine Similarity → Decision
    
    Key Features:
    - Handles documents of any length
    - Preserves semantic coherence through chunking
    - SBERT weights frozen (no updates)
    - Siamese projection head (single shared network)
    - Threshold-based decisions (explainable)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        aggregation_method: str = "mean",
        threshold: float = 0.75,
        device: Optional[str] = None
    ):
        """
        Initialize document-level pipeline.
        
        Args:
            model_path: Path to trained projection head weights (optional)
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks
            aggregation_method: How to aggregate chunk embeddings ("mean", "max", "weighted")
            threshold: Similarity threshold for paraphrase decision
            device: torch device (auto-detected if None)
        """
        print("\n" + "=" * 70)
        print("INITIALIZING DOCUMENT-LEVEL PARAPHRASE DETECTION PIPELINE")
        print("=" * 70)
        
        # Initialize document processor
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            aggregation_method=aggregation_method
        )
        
        # Initialize Siamese model
        if model_path and Path(model_path).exists():
            print(f"\nLoading trained model from: {model_path}")
            self.model = TrainableSiameseModel(projection_dim=256, freeze_sbert=True)
            self.model.load_model(model_path)
            self.model.eval()
        else:
            print("\nUsing untrained projection head (SBERT embeddings only)")
            self.model = SiameseProjectionModel(projection_dim=256)
        
        # Set device
        self.device = device or self.model.device
        
        # Configuration
        self.threshold = threshold
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.aggregation_method = aggregation_method
        
        print(f"\n✓ Pipeline initialized")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Chunk overlap: {chunk_overlap}")
        print(f"  Aggregation: {aggregation_method}")
        print(f"  Threshold: {threshold}")
        print(f"  Device: {self.device}")
        print("=" * 70 + "\n")
    
    def process_document(self, document_text: str) -> torch.Tensor:
        """
        Process a full document following the pipeline:
        Document → Chunk → SBERT → Aggregate → NN → FV
        
        Args:
            document_text: Full document text
            
        Returns:
            Feature vector (FV) as torch.Tensor [256]
        """
        # STEP 1: CHUNKING
        # Split document into semantically coherent chunks
        chunks = self.document_processor.chunk_text(document_text)
        
        if len(chunks) == 0:
            raise ValueError("Document produced no chunks")
        
        # STEP 2: SBERT ENCODING (FROZEN)
        # Encode each chunk independently
        # SBERT weights are frozen - no gradient updates
        chunk_embeddings = []
        with torch.no_grad():
            for chunk in chunks:
                # Get SBERT embedding for this chunk
                emb = self.model.encode_with_sbert(chunk)  # [384]
                chunk_embeddings.append(emb.cpu().numpy())
        
        # STEP 3: AGGREGATION (Mean Pooling)
        # Combine chunk embeddings into single document representation
        doc_embedding = self.document_processor.aggregate_embeddings(
            chunk_embeddings,
            method=self.aggregation_method
        )
        
        # Convert back to tensor
        doc_embedding = torch.from_numpy(doc_embedding).float().to(self.device)
        
        # STEP 4: NN PROJECTION (SIAMESE)
        # Project through shared neural network to get feature vector
        with torch.no_grad():
            feature_vector = self.model.projection_head(doc_embedding)  # [256]
        
        return feature_vector
    
    def compare_documents(
        self,
        doc_a: str,
        doc_b: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline for document pair comparison.
        
        Implements the full diagram:
        Doc A → SBERT → NN → FV_A ┐
                                  ├→ Cosine Similarity → Threshold → Decision
        Doc B → SBERT → NN → FV_B ┘
        
        Args:
            doc_a: First document text
            doc_b: Second document text
            verbose: Print detailed processing info
            
        Returns:
            Dictionary with similarity score, decision, and metadata
        """
        if verbose:
            print("\n" + "=" * 70)
            print("DOCUMENT-LEVEL PARAPHRASE DETECTION")
            print("=" * 70)
        
        # Get document info
        info_a = self.document_processor.get_document_info(doc_a)
        info_b = self.document_processor.get_document_info(doc_b)
        
        if verbose:
            print(f"\nDoc A: {info_a['total_length']} chars, {info_a['num_chunks']} chunks")
            print(f"Doc B: {info_b['total_length']} chars, {info_b['num_chunks']} chunks")
        
        # Process both documents through the pipeline
        if verbose:
            print("\nProcessing Doc A...")
        fv_a = self.process_document(doc_a)
        
        if verbose:
            print("Processing Doc B...")
        fv_b = self.process_document(doc_b)
        
        # STEP 5: COSINE SIMILARITY (L)
        # Compute semantic similarity between feature vectors
        cosine_sim = F.cosine_similarity(
            fv_a.unsqueeze(0),
            fv_b.unsqueeze(0)
        ).item()
        
        # STEP 6: THRESHOLD DECISION
        # Apply threshold to make paraphrase/not-paraphrase decision
        # This is EXPLAINABLE - no black-box classifier
        is_paraphrase = cosine_sim >= self.threshold
        
        # Calculate confidence based on distance from threshold
        confidence_margin = abs(cosine_sim - self.threshold)
        if confidence_margin > 0.15:
            confidence = "HIGH"
        elif confidence_margin > 0.05:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        result = {
            # Core outputs
            "cosine_similarity": float(cosine_sim),
            "is_paraphrase": is_paraphrase,
            "decision": "PARAPHRASE" if is_paraphrase else "NOT PARAPHRASE",
            "confidence": confidence,
            
            # Thresholding
            "threshold": self.threshold,
            "margin": float(confidence_margin),
            
            # Document metadata
            "doc_a_length": info_a['total_length'],
            "doc_b_length": info_b['total_length'],
            "doc_a_chunks": info_a['num_chunks'],
            "doc_b_chunks": info_b['num_chunks'],
            
            # Pipeline info
            "architecture": "Doc → SBERT (frozen) → NN (Siamese) → FV → Cosine Similarity",
            "sbert_model": "all-MiniLM-L6-v2",
            "projection_dim": 256,
            "aggregation_method": self.aggregation_method,
            "device": str(self.device)
        }
        
        if verbose:
            print(f"\n{'=' * 70}")
            print("RESULTS")
            print("=" * 70)
            print(f"Cosine Similarity: {cosine_sim:.4f}")
            print(f"Threshold: {self.threshold}")
            print(f"Decision: {result['decision']}")
            print(f"Confidence: {confidence}")
            print("=" * 70 + "\n")
        
        return result
    
    def compare_from_files(
        self,
        file_a: str,
        file_b: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Load and compare two documents from files.
        
        Supports: .txt, .pdf, .docx
        
        Args:
            file_a: Path to first document
            file_b: Path to second document
            verbose: Print processing info
            
        Returns:
            Comparison results dictionary
        """
        # Load documents
        doc_a = self.document_processor.load_document(file_a)
        doc_b = self.document_processor.load_document(file_b)
        
        # Compare
        result = self.compare_documents(doc_a, doc_b, verbose=verbose)
        
        # Add file paths to result
        result["file_a"] = str(file_a)
        result["file_b"] = str(file_b)
        
        return result
    
    def batch_compare(
        self,
        document_pairs: List[Tuple[str, str]],
        labels: Optional[List[int]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compare multiple document pairs and compute metrics.
        
        Args:
            document_pairs: List of (doc_a, doc_b) text pairs
            labels: Optional ground truth labels (1=paraphrase, 0=not paraphrase)
            verbose: Print per-pair results
            
        Returns:
            Batch results with metrics if labels provided
        """
        results = []
        
        for i, (doc_a, doc_b) in enumerate(document_pairs):
            if verbose:
                print(f"\nPair {i+1}/{len(document_pairs)}")
            
            result = self.compare_documents(doc_a, doc_b, verbose=verbose)
            
            if labels is not None:
                result["true_label"] = labels[i]
                result["correct"] = (result["is_paraphrase"] == bool(labels[i]))
            
            results.append(result)
        
        # Compute metrics if labels provided
        batch_result = {"results": results}
        
        if labels is not None:
            correct = sum(r["correct"] for r in results)
            total = len(results)
            accuracy = correct / total if total > 0 else 0
            
            # True positives, false positives, etc.
            tp = sum(1 for r in results if r["is_paraphrase"] and r["true_label"] == 1)
            fp = sum(1 for r in results if r["is_paraphrase"] and r["true_label"] == 0)
            tn = sum(1 for r in results if not r["is_paraphrase"] and r["true_label"] == 0)
            fn = sum(1 for r in results if not r["is_paraphrase"] and r["true_label"] == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            batch_result["metrics"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "correct": correct,
                "total": total,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn
            }
        
        return batch_result


# ============================================================
# Command-Line Interface
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Document-Level Paraphrase Detection with Siamese Network"
    )
    parser.add_argument("--doc-a", type=str, help="First document (text or file path)")
    parser.add_argument("--doc-b", type=str, help="Second document (text or file path)")
    parser.add_argument("--model", type=str, help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=0.75, help="Similarity threshold")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size")
    parser.add_argument("--test", action="store_true", help="Run test examples")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DocumentLevelSiameseModel(
        model_path=args.model,
        chunk_size=args.chunk_size,
        threshold=args.threshold
    )
    
    if args.test:
        # Test with sample documents
        doc_a = """
        Machine learning is a subset of artificial intelligence that enables computers
        to learn from data without being explicitly programmed. It uses statistical
        techniques to give computer systems the ability to progressively improve their
        performance on a specific task. Deep learning, a subset of machine learning,
        uses neural networks with multiple layers to process complex patterns.
        """
        
        doc_b = """
        Artificial intelligence includes machine learning as one of its branches.
        Machine learning allows computer systems to learn automatically from experience
        without manual programming. By applying statistical methods, these systems
        can enhance their task performance over time. Neural networks with many layers,
        known as deep learning, represent an advanced machine learning approach.
        """
        
        doc_c = """
        Natural language processing is a field concerned with the interactions between
        computers and human language. It involves programming computers to process
        and analyze large amounts of natural language data. Applications include
        translation, sentiment analysis, and question answering systems.
        """
        
        print("\n" + "=" * 70)
        print("TEST 1: Similar Documents (Should be PARAPHRASE)")
        print("=" * 70)
        result1 = pipeline.compare_documents(doc_a, doc_b)
        
        print("\n" + "=" * 70)
        print("TEST 2: Different Documents (Should be NOT PARAPHRASE)")
        print("=" * 70)
        result2 = pipeline.compare_documents(doc_a, doc_c)
    
    elif args.doc_a and args.doc_b:
        # Check if inputs are file paths or raw text
        from pathlib import Path
        
        if Path(args.doc_a).exists() and Path(args.doc_b).exists():
            result = pipeline.compare_from_files(args.doc_a, args.doc_b)
        else:
            result = pipeline.compare_documents(args.doc_a, args.doc_b)
    
    else:
        print("\nUsage:")
        print("  python document_siamese_pipeline.py --test")
        print("  python document_siamese_pipeline.py --doc-a 'text1' --doc-b 'text2'")
        print("  python document_siamese_pipeline.py --doc-a file1.txt --doc-b file2.txt --model models/trained.pt")
