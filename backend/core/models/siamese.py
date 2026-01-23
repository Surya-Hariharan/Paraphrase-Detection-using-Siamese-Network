"""Siamese Network Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import Tuple
import os


class ProjectionHead(nn.Module):
    """Projection head with normalization"""
    
    def __init__(self, input_dim=384, hidden_dim=512, output_dim=256, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.normalize(x, p=2, dim=1)


class SiameseModel(nn.Module):
    """Siamese Network for paraphrase detection"""
    
    def __init__(self, encoder_name="all-MiniLM-L6-v2", embedding_dim=384, 
                 projection_dim=256, freeze_encoder=False):
        super().__init__()
        self.encoder_name = encoder_name
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        # Load SBERT encoder
        self.sbert = SentenceTransformer(encoder_name)
        
        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=embedding_dim,
            hidden_dim=512,
            output_dim=projection_dim,
            dropout=0.1
        )
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.sbert.parameters():
                param.requires_grad = False
        else:
            for param in self.sbert.parameters():
                param.requires_grad = True
    
    def encode(self, texts):
        """Encode texts to embeddings"""
        embeddings = self.sbert.encode(
            texts, 
            convert_to_tensor=True, 
            show_progress_bar=False
        )
        return embeddings
    
    def forward(self, text_a, text_b):
        """Forward pass"""
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        
        fv_a = self.projection_head(emb_a)
        fv_b = self.projection_head(emb_b)
        
        similarity = F.cosine_similarity(fv_a, fv_b)
        
        return similarity, fv_a, fv_b
    
    def predict(self, text_a, text_b):
        """Predict similarity between two texts"""
        self.eval()
        with torch.no_grad():
            similarity, _, _ = self.forward([text_a], [text_b])
            return similarity.item()
    
    def save(self, path):
        """Save model state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'encoder_name': self.encoder_name,
            'embedding_dim': self.embedding_dim,
            'projection_dim': self.projection_dim
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from {path}")
    
    @staticmethod
    def load_from_checkpoint(path, device='cpu'):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        model = SiameseModel(
            encoder_name=checkpoint.get('encoder_name', 'all-MiniLM-L6-v2'),
            embedding_dim=checkpoint.get('embedding_dim', 384),
            projection_dim=checkpoint.get('projection_dim', 256)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese network"""
    
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, similarities, labels):
        labels_float = labels.float()
        loss_positive = labels_float * (1 - similarities)
        loss_negative = (1 - labels_float) * torch.clamp(similarities - self.margin, min=0.0)
        loss = loss_positive + loss_negative
        return loss.mean()
