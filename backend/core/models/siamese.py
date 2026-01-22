"""Siamese Network Model"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from typing import Tuple, Optional
import os


class SiameseModel(nn.Module):
    """
    Siamese Network for paraphrase detection
    
    Architecture:
        Input → SBERT Encoder → Projection Head → L2 Norm → Cosine Similarity
    """
    
    def __init__(
        self,
        encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 768,
        projection_dim: int = 256,
        freeze_encoder: bool = False
    ):
        super().__init__()
        
        self.encoder_name = encoder_name
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        # SBERT encoder
        self.encoder = SentenceTransformer(encoder_name)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def encode(self, texts: list) -> torch.Tensor:
        """Encode texts to embeddings"""
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        return embeddings
    
    def forward(self, text_a: list, text_b: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            text_a: List of first texts
            text_b: List of second texts
        
        Returns:
            embeddings_a, embeddings_b: Projected and normalized embeddings
        """
        # Encode
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        
        # Project
        proj_a = self.projection(emb_a)
        proj_b = self.projection(emb_b)
        
        # L2 normalize
        norm_a = nn.functional.normalize(proj_a, p=2, dim=1)
        norm_b = nn.functional.normalize(proj_b, p=2, dim=1)
        
        return norm_a, norm_b
    
    def predict(self, text_a: str, text_b: str) -> float:
        """
        Predict similarity between two texts
        
        Returns:
            Cosine similarity score [0, 1]
        """
        self.eval()
        with torch.no_grad():
            emb_a, emb_b = self([text_a], [text_b])
            similarity = nn.functional.cosine_similarity(emb_a, emb_b, dim=1)
            return similarity.item()
    
    def save(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'encoder_name': self.encoder_name,
            'embedding_dim': self.embedding_dim,
            'projection_dim': self.projection_dim,
            'projection_state_dict': self.projection.state_dict()
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Verify architecture matches
        assert checkpoint['encoder_name'] == self.encoder_name
        assert checkpoint['embedding_dim'] == self.embedding_dim
        assert checkpoint['projection_dim'] == self.projection_dim
        
        # Load projection weights
        self.projection.load_state_dict(checkpoint['projection_state_dict'])


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            emb_a: First embeddings [batch_size, dim]
            emb_b: Second embeddings [batch_size, dim]
            labels: Binary labels [batch_size] (1 = similar, 0 = dissimilar)
        
        Returns:
            Loss scalar
        """
        # Cosine similarity
        similarity = nn.functional.cosine_similarity(emb_a, emb_b, dim=1)
        
        # Convert to distance (1 - similarity)
        distance = 1 - similarity
        
        # Contrastive loss
        loss = labels * distance.pow(2) + \
               (1 - labels) * torch.clamp(self.margin - distance, min=0).pow(2)
        
        return loss.mean()
