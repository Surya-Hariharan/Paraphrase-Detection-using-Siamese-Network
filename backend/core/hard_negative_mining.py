"""
Hard Negative Mining Module
============================

Implements online hard negative mining during training to:
1. Focus on difficult examples
2. Improve model discrimination
3. Handle edge cases better
4. Boost accuracy to 90%+
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class HardNegativeMiner:
    """
    Online hard negative mining during training
    
    Identifies and focuses on:
    - Hard negatives: Non-paraphrases with high similarity
    - Hard positives: Paraphrases with low similarity
    """
    
    def __init__(
        self,
        negative_margin: float = 0.3,
        positive_margin: float = 0.7,
        mining_strategy: str = 'semi-hard',  # 'hard', 'semi-hard', 'all'
        use_weighted_loss: bool = True
    ):
        """
        Args:
            negative_margin: Similarity threshold for hard negatives
            positive_margin: Similarity threshold for hard positives
            mining_strategy: 'hard' (hardest), 'semi-hard' (medium), 'all' (no mining)
            use_weighted_loss: Weight loss by difficulty
        """
        self.negative_margin = negative_margin
        self.positive_margin = positive_margin
        self.mining_strategy = mining_strategy
        self.use_weighted_loss = use_weighted_loss
        
        # Statistics
        self.hard_negative_count = 0
        self.hard_positive_count = 0
        self.total_mined = 0
        
    def mine_hard_pairs(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor,
        return_weights: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Identify hard examples and optionally compute sample weights
        
        Args:
            similarities: Cosine similarities [batch_size]
            labels: Ground truth labels [batch_size]
            return_weights: Return sample weights for weighted loss
            
        Returns:
            mask: Boolean mask for hard examples
            weights: Sample weights (if return_weights=True)
        """
        batch_size = similarities.size(0)
        device = similarities.device
        
        # Initialize mask (all True if 'all' strategy)
        if self.mining_strategy == 'all':
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            weights = torch.ones(batch_size, dtype=torch.float32, device=device)
            return mask, weights if return_weights else mask
        
        mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        weights = torch.ones(batch_size, dtype=torch.float32, device=device)
        
        # Separate positive and negative pairs
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        # HARD NEGATIVES: Non-paraphrases with HIGH similarity (above margin)
        # These are the most confusing for the model
        hard_negatives = neg_mask & (similarities > self.negative_margin)
        
        # HARD POSITIVES: Paraphrases with LOW similarity (below margin)
        # These are paraphrases the model struggles to recognize
        hard_positives = pos_mask & (similarities < self.positive_margin)
        
        if self.mining_strategy == 'hard':
            # Only hardest examples
            mask = hard_negatives | hard_positives
            
        elif self.mining_strategy == 'semi-hard':
            # Semi-hard: Include moderately difficult examples
            # Semi-hard negatives: similarity in [0.2, 0.5]
            semi_hard_negatives = neg_mask & (similarities > 0.2) & (similarities < 0.5)
            # Semi-hard positives: similarity in [0.5, 0.8]
            semi_hard_positives = pos_mask & (similarities > 0.5) & (similarities < 0.8)
            
            mask = hard_negatives | hard_positives | semi_hard_negatives | semi_hard_positives
        
        # Update statistics
        self.hard_negative_count += hard_negatives.sum().item()
        self.hard_positive_count += hard_positives.sum().item()
        self.total_mined += mask.sum().item()
        
        # Compute sample weights based on difficulty
        if return_weights and self.use_weighted_loss:
            # Weight by distance from margin
            for i in range(batch_size):
                if labels[i] == 0:  # Negative pair
                    # Higher weight if similarity is close to or above margin
                    if similarities[i] > self.negative_margin:
                        weights[i] = 2.0 + (similarities[i] - self.negative_margin)
                else:  # Positive pair
                    # Higher weight if similarity is far below margin
                    if similarities[i] < self.positive_margin:
                        weights[i] = 2.0 + (self.positive_margin - similarities[i])
        
        return mask, weights if return_weights else mask
    
    def get_statistics(self) -> Dict[str, float]:
        """Get mining statistics"""
        return {
            'hard_negatives': self.hard_negative_count,
            'hard_positives': self.hard_positive_count,
            'total_mined': self.total_mined,
            'mining_rate': self.total_mined / max(self.total_mined + 1, 1)
        }
    
    def reset_statistics(self):
        """Reset statistics for new epoch"""
        self.hard_negative_count = 0
        self.hard_positive_count = 0
        self.total_mined = 0


class AdaptiveThresholdOptimizer:
    """
    Advanced threshold optimization with edge case focus
    
    Optimizes threshold based on:
    - F1 score maximization
    - Edge case performance
    - Confidence calibration
    """
    
    def __init__(
        self,
        edge_case_weight: float = 2.0,
        confidence_bins: int = 10
    ):
        """
        Args:
            edge_case_weight: Weight for edge case examples
            confidence_bins: Number of bins for calibration
        """
        self.edge_case_weight = edge_case_weight
        self.confidence_bins = confidence_bins
        
    def find_optimal_threshold(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        edge_case_mask: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold maximizing weighted F1 score
        
        Args:
            similarities: Predicted similarities
            labels: Ground truth labels
            edge_case_mask: Boolean mask for edge cases (higher weight)
            
        Returns:
            optimal_threshold: Best threshold
            metrics: Dict with precision, recall, F1
        """
        # Sample thresholds to test
        thresholds = np.linspace(0.3, 0.9, 61)
        
        best_f1 = 0.0
        best_threshold = 0.5
        best_metrics = {}
        
        # Apply edge case weights
        if edge_case_mask is not None:
            sample_weights = np.ones_like(labels, dtype=float)
            sample_weights[edge_case_mask] = self.edge_case_weight
        else:
            sample_weights = np.ones_like(labels, dtype=float)
        
        for threshold in thresholds:
            # Predictions
            predictions = (similarities >= threshold).astype(int)
            
            # Weighted metrics
            tp = (((predictions == 1) & (labels == 1)) * sample_weights).sum()
            fp = (((predictions == 1) & (labels == 0)) * sample_weights).sum()
            fn = (((predictions == 0) & (labels == 1)) * sample_weights).sum()
            
            # Avoid division by zero
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'threshold': threshold
                }
        
        return best_threshold, best_metrics
    
    def calibrate_confidence(
        self,
        similarities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Analyze confidence calibration
        
        Returns calibration curve data
        """
        bins = np.linspace(0, 1, self.confidence_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(self.confidence_bins):
            bin_mask = (similarities >= bins[i]) & (similarities < bins[i + 1])
            
            if bin_mask.sum() > 0:
                bin_similarities = similarities[bin_mask]
                bin_labels = labels[bin_mask]
                
                # Accuracy in this bin
                predictions = (bin_similarities >= 0.5).astype(int)
                accuracy = (predictions == bin_labels).mean()
                
                # Average confidence
                confidence = bin_similarities.mean()
                
                bin_accuracies.append(accuracy)
                bin_confidences.append(confidence)
                bin_counts.append(bin_mask.sum())
        
        return {
            'accuracies': bin_accuracies,
            'confidences': bin_confidences,
            'counts': bin_counts
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples
    
    FL(p_t) = -α(1-p_t)^γ log(p_t)
    
    Focuses on hard examples by down-weighting easy ones
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(
        self,
        similarities: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            similarities: Predicted similarities [batch_size]
            labels: Ground truth labels [batch_size]
            
        Returns:
            loss: Focal loss value
        """
        # Convert similarities to probabilities (for positive class)
        probs = similarities
        
        # Compute focal weight: (1-p_t)^gamma
        labels_float = labels.float()
        p_t = torch.where(labels == 1, probs, 1 - probs)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(probs, labels_float, reduction='none')
        
        # Apply focal weight and alpha
        loss = self.alpha * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_edge_case_mask(texts_a: List[str], texts_b: List[str]) -> np.ndarray:
    """
    Identify edge cases based on text patterns
    
    Edge cases include:
    - Negations (not, un-, in-, dis-)
    - Numbers
    - Very short or very long text
    - Similar structure but different words
    
    Returns boolean mask for edge cases
    """
    edge_case_mask = np.zeros(len(texts_a), dtype=bool)
    
    negation_patterns = [
        r'\bnot\b', r'\bno\b', r'\bnever\b', r'\bneither\b',
        r'\bun', r'\bin', r'\bdis', r'\bim',
        r'n\'t\b'
    ]
    
    for i, (text_a, text_b) in enumerate(zip(texts_a, texts_b)):
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()
        
        # Check for negations
        import re
        for pattern in negation_patterns:
            if re.search(pattern, text_a_lower) or re.search(pattern, text_b_lower):
                edge_case_mask[i] = True
                break
        
        # Check for numbers
        if re.search(r'\d+', text_a) or re.search(r'\d+', text_b):
            edge_case_mask[i] = True
        
        # Check for length mismatch
        len_ratio = len(text_a) / (len(text_b) + 1)
        if len_ratio > 2.0 or len_ratio < 0.5:
            edge_case_mask[i] = True
    
    return edge_case_mask
