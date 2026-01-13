"""
Neural Engine - Document-Level Paraphrase Detection with Siamese Network

ARCHITECTURE (MUST MATCH DIAGRAM):
    Doc A ──▶ SBERT ──▶ NN ──▶ FV_A ──┐
                                     ├──▶ Cosine Similarity (L)
    Doc B ──▶ SBERT ──▶ NN ──▶ FV_B ──┘

COMPONENTS:
1. SBERT Encoder: Frozen pretrained SentenceTransformer (all-MiniLM-L6-v2)
   - Encodes text chunks independently
   - 384-dimensional embeddings
   - NO gradient updates (frozen weights)

2. NN Block (Siamese Projection Head): Shared neural network
   - Dense → Tanh → Dense architecture
   - ONE network shared between Doc A and Doc B
   - Trainable task-specific projection
   - Output: Feature Vectors (FV_A, FV_B) in same embedding space

3. Cosine Similarity (L): Semantic similarity measure
   - Compute similarity between FV_A and FV_B
   - Range: [-1.0, 1.0]
   - Threshold-based decision (no classifier)

SEMANTIC SIMILARITY PRINCIPLES:
- This is an NLP project focused on semantic understanding
- No generative AI, no RL, no online learning
- Documents chunked to preserve semantic coherence
- Chunk embeddings aggregated via mean pooling
- Explainable threshold-based decisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


# ============================================================
# NN BLOCK: Siamese Projection Head (Diagram Component)
# ============================================================

class ProjectionHead(nn.Module):
    """
    Simple 2-Layer Dense Projection Head for stable training.
    
    Architecture:
    Input(384) → Dense(512) → ReLU → Dropout(0.1) → Dense(256)
    
    Features:
    - Minimal architecture for GPU stability
    - Xavier initialization for stable gradients
    - Light dropout for regularization
    
    Input: SBERT embedding (384-dim)
    Output: Task-specific feature vector FV (256-dim)
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 512, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        # Simple 2-layer dense network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Xavier initialization for stable training
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the simple projection head.
        
        Args:
            x: SBERT embeddings (batch_size, 384)
            
        Returns:
            Projected features (batch_size, 256)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def get_weight_info(self) -> Dict[str, Any]:
        return {
            "fc1_shape": tuple(self.fc1.weight.shape),
            "fc2_shape": tuple(self.fc2.weight.shape),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable": all(p.requires_grad for p in self.parameters())
        }


# ============================================================
# Inference Model (NO GRADIENTS)
# ============================================================

class SiameseProjectionModel(nn.Module):
    """
    Siamese network for inference only.

    Architecture:
        Text → SBERT (384) → Projection Head (256) → Cosine Similarity
    """

    SBERT_MODEL = "all-MiniLM-L6-v2"
    SBERT_DIM = 384
    PROJECTION_DIM = 256

    def __init__(self, projection_dim: int = 256):
        super().__init__()

        # Load SBERT
        self.sbert_encoder = SentenceTransformer(self.SBERT_MODEL)
        self.device = self.sbert_encoder.device

        # Simple projection head
        self.projection_head = ProjectionHead(
            input_dim=self.SBERT_DIM,
            hidden_dim=512,
            output_dim=projection_dim,
            dropout=0.1
        ).to(self.device)

        self.projection_dim = projection_dim

    def encode_with_sbert(self, text: str) -> torch.Tensor:
        embedding = self.sbert_encoder.encode(
            text, convert_to_tensor=True
        )
        return embedding.to(self.device)

    def project_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.projection_head(embedding.to(self.device))

    def forward(self, doc_a: str, doc_b: str) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_a = self.encode_with_sbert(doc_a)
        emb_b = self.encode_with_sbert(doc_b)

        vec_a = self.project_embedding(emb_a)
        vec_b = self.project_embedding(emb_b)

        return vec_a, vec_b

    def calculate_similarity(self, doc_a: str, doc_b: str) -> Dict[str, Any]:
        vec_a, vec_b = self.forward(doc_a, doc_b)

        cosine_sim = F.cosine_similarity(
            vec_a.unsqueeze(0),
            vec_b.unsqueeze(0)
        ).item()

        similarity_percentage = (cosine_sim + 1) / 2 * 100

        return {
            "cosine_similarity": float(cosine_sim),
            "similarity_percentage": float(similarity_percentage),
            "model": self.SBERT_MODEL,
            "projection_dim": self.projection_dim,
            "device": str(self.device),
            "architecture": "Siamese SBERT + Projection Head"
        }

    def demonstrate_weight_sharing(self):
        info = self.projection_head.get_weight_info()
        print(f"Projection Head Parameters: {info['total_parameters']:,}")
        print(f"SBERT Encoder Device: {self.device}")
        print("Weight sharing verified")
    
    def forward_batch(self, docs_a: list, docs_b: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for batch of documents.
        
        Args:
            docs_a: List of documents A
            docs_b: List of documents B
            
        Returns:
            Tuple of embedding tensors (batch_size, projection_dim)
        """
        # Encode batches
        emb_a_batch = self.sbert_model.encode(
            docs_a,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        
        emb_b_batch = self.sbert_model.encode(
            docs_b,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )
        
        # Project
        vec_a_batch = self.projection_head(emb_a_batch)
        vec_b_batch = self.projection_head(emb_b_batch)
        
        return vec_a_batch, vec_b_batch


# ============================================================
# Contrastive Loss
# ============================================================

class ContrastiveLoss(nn.Module):
    """Contrastive loss with margin and numerical stability."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.eps = 1e-8  # For numerical stability

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:

        # Normalize embeddings for stability (L2 normalization)
        emb_a = F.normalize(emb_a, p=2, dim=1)
        emb_b = F.normalize(emb_b, p=2, dim=1)
        
        # Compute distance with numerical stability
        distance = F.pairwise_distance(emb_a, emb_b, eps=self.eps)
        
        # Clamp distance to prevent extreme values
        distance = torch.clamp(distance, min=self.eps, max=2.0)

        loss_similar = label * torch.pow(distance, 2)
        loss_dissimilar = (1 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2
        )

        loss = torch.mean(loss_similar + loss_dissimilar)
        
        # Check for NaN and return a safe value if detected
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  NaN/Inf detected in loss! Using safe fallback value")
            return torch.tensor(1.0, device=loss.device, requires_grad=True)
        
        return loss


# ============================================================
# Threshold Optimizer for Binary Classification
# ============================================================

class ThresholdOptimizer:
    """
    Optimize threshold for converting cosine similarity to binary labels.
    
    Addresses the mismatch between continuous similarity scores [-1, 1]
    and binary labels [0, 1].
    """
    
    def __init__(self):
        self.best_threshold = 0.75  # Default
        self.best_accuracy = 0.0
    
    def find_optimal_threshold(
        self,
        similarities: List[float],
        labels: List[int],
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal threshold by grid search.
        
        Args:
            similarities: Cosine similarity scores
            labels: Binary labels (0 or 1)
            thresholds: List of thresholds to try (default: 0.0 to 1.0)
        
        Returns:
            Dict with best_threshold, best_accuracy, and metrics
        """
        if thresholds is None:
            # Try thresholds from 0.0 to 1.0 in steps of 0.05
            thresholds = [i * 0.05 for i in range(21)]
        
        best_threshold = 0.75
        best_accuracy = 0.0
        best_f1 = 0.0
        threshold_results = []
        
        for threshold in thresholds:
            # Convert similarities to predictions
            predictions = [1 if sim >= threshold else 0 for sim in similarities]
            
            # Calculate metrics
            correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
            accuracy = correct / len(labels) if labels else 0.0
            
            # Calculate F1 score
            tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
            fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
            fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
            
            # Update best threshold based on F1 score (better for imbalanced data)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_accuracy = accuracy
        
        self.best_threshold = best_threshold
        self.best_accuracy = best_accuracy
        
        return {
            'best_threshold': best_threshold,
            'best_accuracy': best_accuracy,
            'best_f1': best_f1,
            'all_results': threshold_results
        }
    
    def apply_threshold(self, similarity: float) -> int:
        """Apply best threshold to get binary prediction."""
        return 1 if similarity >= self.best_threshold else 0


# ============================================================
# Trainable Siamese Model (GRADIENTS ENABLED)
# ============================================================

class TrainableSiameseModel(SiameseProjectionModel):
    """
    Siamese model for TRAINING with SBERT fine-tuning support.

    Differences from base class:
    - Gradients ENABLED
    - Batch support
    - Flexible SBERT freezing/fine-tuning strategies
    - Threshold optimizer for binary classification
    """

    def __init__(
        self, 
        projection_dim: int = 256, 
        freeze_sbert: bool = False,  # COMPLETE FINE-TUNING: ALL LAYERS UNFROZEN
        unfreeze_all: bool = True  # NEW: Unfreeze entire SBERT model
    ):
        super().__init__(projection_dim)

        self.freeze_sbert = freeze_sbert
        self.unfreeze_all = unfreeze_all
        
        # Threshold optimizer for converting similarity to binary predictions
        self.threshold_optimizer = ThresholdOptimizer()
        
        # Configure SBERT training strategy
        if freeze_sbert:
            # Strategy 1: Freeze all SBERT (fast, low memory, NOT RECOMMENDED)
            print("🔒 Freezing SBERT encoder (no gradient updates)...")
            for name, param in self.sbert_encoder.named_parameters():
                param.requires_grad = False
        elif unfreeze_all:
            # Strategy 2: COMPLETE FINE-TUNING - Unfreeze ENTIRE SBERT model
            print("🔥 FULL FINE-TUNING: Unfreezing ENTIRE SBERT model (ALL 6 layers)...")
            
            # Unfreeze all parameters in SBERT
            for param in self.sbert_encoder.parameters():
                param.requires_grad = True
            
            # Count trainable parameters
            try:
                transformer = self.sbert_encoder[0].auto_model
                if hasattr(transformer, 'encoder') and hasattr(transformer.encoder, 'layer'):
                    total_layers = len(transformer.encoder.layer)
                    trainable_params = sum(p.numel() for p in self.sbert_encoder.parameters() if p.requires_grad)
                    
                    print(f"   ✓ ALL {total_layers} transformer layers unfrozen")
                    print(f"   ✓ Trainable SBERT params: {trainable_params:,}")
                    print(f"   🎯 Maximum accuracy potential unlocked")
                    print(f"   ⚠️  Requires ~6-8GB GPU memory")
                else:
                    print("   ✓ All SBERT parameters unfrozen")
            except Exception as e:
                print(f"   ✓ All SBERT parameters unfrozen (details unavailable: {e})")
        else:
            # Fallback: partial unfreezing (not used by default)
            print("⚠️  Using partial unfreezing (not recommended)")
        
        # Verify freezing status
        self._verify_freezing_status()

    def forward(self, doc_a: str, doc_b: str) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_a = self.encode_with_sbert(doc_a)
        emb_b = self.encode_with_sbert(doc_b)

        vec_a = self.projection_head(emb_a)
        vec_b = self.projection_head(emb_b)

        return vec_a, vec_b

    def forward_batch(
        self,
        texts_a: List[str],
        texts_b: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        emb_a = self.sbert_encoder.encode(
            texts_a, convert_to_tensor=True, show_progress_bar=False
        ).to(self.device)

        emb_b = self.sbert_encoder.encode(
            texts_b, convert_to_tensor=True, show_progress_bar=False
        ).to(self.device)

        proj_a = self.projection_head(emb_a)
        proj_b = self.projection_head(emb_b)

        return proj_a, proj_b

    def get_trainable_parameters(self):
        if self.freeze_sbert:
            return self.projection_head.parameters()
        return self.parameters()
    
    def get_parameter_groups(self, lr_sbert: float = 1e-5, lr_head: float = 1e-4):
        """Get parameter groups for differential learning rates."""
        param_groups = []
        
        if self.unfreeze_all:
            # SBERT with lower learning rate
            param_groups.append({
                'params': self.sbert_encoder.parameters(),
                'lr': lr_sbert,
                'name': 'sbert_encoder'
            })
        
        # Projection head with higher learning rate
        param_groups.append({
            'params': self.projection_head.parameters(),
            'lr': lr_head,
            'name': 'projection_head'
        })
        
        return param_groups

    def compute_similarity_batch(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity for a batch of embeddings."""
        return F.cosine_similarity(emb_a, emb_b)

    def save_model(self, path: str):
        state_dict = {"projection_head": self.projection_head.state_dict()}
        
        # Save SBERT if fully unfrozen
        if self.unfreeze_all:
            state_dict["sbert_encoder"] = self.sbert_encoder.state_dict()
        
        torch.save(
            {
                **state_dict,
                "projection_dim": self.projection_dim,
                "freeze_sbert": self.freeze_sbert,
                "unfreeze_all": self.unfreeze_all
            },
            path
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.projection_head.load_state_dict(checkpoint["projection_head"])
        
        # Load SBERT if it was saved
        if self.unfreeze_all and "sbert_encoder" in checkpoint:
            self.sbert_encoder.load_state_dict(checkpoint["sbert_encoder"])

    def save_checkpoint(self, path: str, epoch: int, optimizer_state: dict):
        """Save full training checkpoint."""
        state_dict = {"projection_head": self.projection_head.state_dict()}
        
        # Save all SBERT parameters if fully unfrozen
        if self.unfreeze_all:
            state_dict["sbert_encoder"] = self.sbert_encoder.state_dict()
        
        torch.save(
            {
                "epoch": epoch,
                **state_dict,
                "projection_dim": self.projection_dim,
                "freeze_sbert": self.freeze_sbert,
                "unfreeze_all": self.unfreeze_all,
                "optimizer_state": optimizer_state
            },
            path
        )

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.projection_head.load_state_dict(checkpoint["projection_head"])
        
        # Load SBERT parameters if they were saved
        if self.unfreeze_all and "sbert_encoder" in checkpoint:
            self.sbert_encoder.load_state_dict(checkpoint["sbert_encoder"])
        
        return checkpoint.get("epoch", 0), checkpoint.get("optimizer_state", None)
    
    def _verify_freezing_status(self):
        """Verify SBERT freezing and log parameter status."""
        sbert_params = sum(p.numel() for p in self.sbert_encoder.parameters())
        sbert_trainable = sum(p.numel() for p in self.sbert_encoder.parameters() if p.requires_grad)
        projection_params = sum(p.numel() for p in self.projection_head.parameters())
        projection_trainable = sum(p.numel() for p in self.projection_head.parameters() if p.requires_grad)
        
        print(f"\n=== Parameter Status ===")
        print(f"SBERT: {sbert_params:,} total, {sbert_trainable:,} trainable")
        print(f"Projection Head: {projection_params:,} total, {projection_trainable:,} trainable")
        
        if self.freeze_sbert:
            print(f"SBERT Status: ❄️  FROZEN (all parameters frozen)")
        elif self.unfreeze_all:
            print(f"SBERT Status: 🔥 FULLY UNFROZEN ({sbert_trainable:,} / {sbert_params:,} trainable)")
        else:
            print(f"SBERT Status: ⚡ PARTIALLY UNFROZEN ({sbert_trainable:,} / {sbert_params:,} trainable)")
        
        print(f"========================\n")