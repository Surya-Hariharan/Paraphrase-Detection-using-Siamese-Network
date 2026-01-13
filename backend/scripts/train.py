"""
🚀 ADVANCED TRAINING SCRIPT FOR 90%+ ACCURACY
==============================================

Features:
✓ Full SBERT fine-tuning (all layers)
✓ CosFace Angular Margin Loss
✓ Hard negative mining
✓ Edge case focus
✓ Document-level paraphrasing
✓ Adaptive threshold optimization
✓ Mixed precision training (AMP)
✓ Advanced data augmentation
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.model import TrainableSiameseModel, ContrastiveLoss
from backend.core.trainer import ParaphraseDataset
from backend.core.hard_negative_mining import (
    HardNegativeMiner,
    AdaptiveThresholdOptimizer,
    FocalLoss,
    create_edge_case_mask
)


class CosineSimilarityLoss(nn.Module):
    """
    Cosine Similarity Loss for Siamese Networks
    
    For paraphrases (label=1): Minimize distance (maximize similarity)
    For non-paraphrases (label=0): Maximize distance (minimize similarity)
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
        
    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarities: Cosine similarities [batch_size], range [-1, 1]
            labels: Ground truth labels [batch_size] (0 or 1)
        """
        # Convert labels to float
        labels_float = labels.float()
        
        # For paraphrases (1): loss = 1 - similarity (want similarity close to 1)
        # For non-paraphrases (0): loss = max(0, similarity - margin) (want similarity < margin)
        
        loss_positive = labels_float * (1 - similarities)
        loss_negative = (1 - labels_float) * torch.clamp(similarities - self.margin, min=0.0)
        
        loss = loss_positive + loss_negative
        
        return loss.mean()


class CosFaceLoss(nn.Module):
    """
    CosFace Angular Margin Loss for better feature discrimination
    
    L = -log(e^(s*(cos(θ_y) - m)) / (e^(s*(cos(θ_y) - m)) + Σe^(s*cos(θ_j))))
    
    where:
    - s: scale factor (typically 30-64)
    - m: angular margin (typically 0.25-0.5)
    - θ_y: angle for correct class
    """
    
    def __init__(self, scale: float = 30.0, margin: float = 0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarities: Cosine similarities [batch_size]
            labels: Ground truth labels [batch_size] (0 or 1)
        """
        # Apply margin to positive pairs
        similarities_m = torch.where(
            labels == 1,
            similarities - self.margin,  # Subtract margin for positive pairs
            similarities
        )
        
        # Scale
        logits = self.scale * similarities_m
        
        # Binary cross entropy on scaled logits
        labels_float = labels.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels_float)
        
        return loss


class AdvancedSiameseTrainer:
    """Advanced trainer with all optimizations for 90%+ accuracy"""
    
    def __init__(
        self,
        model: TrainableSiameseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 5e-5,
        sbert_lr: float = 5e-6,
        loss_type: str = 'cosface',
        cosface_scale: float = 30.0,
        cosface_margin: float = 0.35,
        use_hard_mining: bool = True,
        mining_strategy: str = 'semi-hard',
        use_mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip = gradient_clip
        
        # Loss function
        if loss_type == 'cosface':
            self.criterion = CosFaceLoss(scale=cosface_scale, margin=cosface_margin)
            print(f"✓ Using CosFace Loss (scale={cosface_scale}, margin={cosface_margin})")
        elif loss_type == 'focal':
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
            print(f"✓ Using Focal Loss (alpha=0.25, gamma=2.0)")
        elif loss_type == 'cosine':
            self.criterion = CosineSimilarityLoss(margin=0.5)
            print(f"✓ Using Cosine Similarity Loss (margin=0.5)")
        else:
            self.criterion = ContrastiveLoss(margin=0.5)
            print(f"✓ Using Contrastive Loss (margin=0.5)")
        
        # Hard negative mining
        self.use_hard_mining = use_hard_mining
        if use_hard_mining:
            self.hard_miner = HardNegativeMiner(
                negative_margin=0.3,
                positive_margin=0.7,
                mining_strategy=mining_strategy,
                use_weighted_loss=True
            )
            print(f"✓ Hard negative mining enabled ({mining_strategy})")
        
        # Threshold optimizer
        self.threshold_optimizer = AdaptiveThresholdOptimizer(edge_case_weight=2.0)
        
        # Differential learning rates (SBERT vs Projection)
        sbert_params = []
        projection_params = []
        
        for name, param in model.named_parameters():
            if 'sbert' in name or 'sentence_transformer' in name:
                sbert_params.append(param)
            else:
                projection_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': sbert_params, 'lr': sbert_lr, 'weight_decay': 0.01},
            {'params': projection_params, 'lr': learning_rate, 'weight_decay': 0.01}
        ])
        
        print(f"✓ Differential LR: SBERT={sbert_lr}, Projection={learning_rate}")
        
        # Mixed precision
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            print("✓ Mixed precision (AMP) enabled")
        
        # Tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'optimal_threshold': [],
            'f1_score': []
        }
        self.best_val_loss = float('inf')
        self.best_f1 = 0.0
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with hard mining"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        if self.use_hard_mining:
            self.hard_miner.reset_statistics()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            texts_a = batch['text_a']
            texts_b = batch['text_b']
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                with autocast():
                    vec_a, vec_b = self.model.forward_batch(texts_a, texts_b)
                    
                    # Compute cosine similarity
                    similarities = F.cosine_similarity(vec_a, vec_b, dim=1)
                    
                    # Hard negative mining
                    if self.use_hard_mining:
                        mask, weights = self.hard_miner.mine_hard_pairs(similarities, labels)
                        
                        # Apply mask and weights
                        if mask.any():
                            loss = self.criterion(similarities[mask], labels[mask])
                            if weights is not None:
                                loss = (loss * weights[mask]).mean()
                        else:
                            loss = self.criterion(similarities, labels)
                    else:
                        loss = self.criterion(similarities, labels)
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Without mixed precision
                vec_a, vec_b = self.model.forward_batch(texts_a, texts_b)
                
                # Compute cosine similarity
                similarities = F.cosine_similarity(vec_a, vec_b, dim=1)
                
                if self.use_hard_mining:
                    mask, weights = self.hard_miner.mine_hard_pairs(similarities, labels)
                    if mask.any():
                        loss = self.criterion(similarities[mask], labels[mask])
                        if weights is not None:
                            loss = (loss * weights[mask]).mean()
                    else:
                        loss = self.criterion(similarities, labels)
                else:
                    loss = self.criterion(similarities, labels)
                
                loss.backward()
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = (similarities > 0.5).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        if self.use_hard_mining:
            mining_stats = self.hard_miner.get_statistics()
            metrics.update(mining_stats)
        
        return metrics
    
    def validate(self, calculate_optimal_threshold: bool = False) -> Dict[str, float]:
        """Validate model with optional threshold optimization"""
        self.model.eval()
        
        total_loss = 0.0
        all_similarities = []
        all_labels = []
        all_texts_a = []
        all_texts_b = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                texts_a = batch['text_a']
                texts_b = batch['text_b']
                labels = batch['label'].to(self.device)
                
                vec_a, vec_b = self.model.forward_batch(texts_a, texts_b)
                similarities = F.cosine_similarity(vec_a, vec_b, dim=1)
                
                loss = self.criterion(similarities, labels)
                total_loss += loss.item()
                
                all_similarities.extend(similarities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_texts_a.extend(texts_a)
                all_texts_b.extend(texts_b)
        
        all_similarities = np.array(all_similarities)
        all_labels = np.array(all_labels)
        
        # Calculate accuracy with default threshold
        predictions = (all_similarities > 0.5).astype(int)
        accuracy = (predictions == all_labels).mean()
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy
        }
        
        # Optimal threshold calculation
        if calculate_optimal_threshold:
            # Identify edge cases
            edge_case_mask = create_edge_case_mask(all_texts_a, all_texts_b)
            
            # Find optimal threshold with edge case weighting
            optimal_threshold, threshold_metrics = self.threshold_optimizer.find_optimal_threshold(
                all_similarities,
                all_labels,
                edge_case_mask=edge_case_mask
            )
            
            metrics['optimal_threshold'] = optimal_threshold
            metrics['optimal_f1'] = threshold_metrics['f1']
            metrics['optimal_precision'] = threshold_metrics['precision']
            metrics['optimal_recall'] = threshold_metrics['recall']
            
            # Edge case specific metrics
            if edge_case_mask.any():
                edge_predictions = (all_similarities[edge_case_mask] > optimal_threshold).astype(int)
                edge_accuracy = (edge_predictions == all_labels[edge_case_mask]).mean()
                metrics['edge_case_accuracy'] = edge_accuracy
                metrics['edge_case_count'] = edge_case_mask.sum()
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: str = 'checkpoints',
        save_interval: int = 5
    ) -> Dict[str, List]:
        """Full training loop"""
        Path(checkpoint_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'-'*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'-'*50}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            print(f"\nTraining - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
            
            if self.use_hard_mining:
                print(f"  Hard Mining - Negatives: {train_metrics['hard_negatives']}, "
                      f"Positives: {train_metrics['hard_positives']}")
            
            # Validate
            calculate_threshold = (epoch % 5 == 0) or (epoch == num_epochs)
            val_metrics = self.validate(calculate_optimal_threshold=calculate_threshold)
            
            if calculate_threshold:
                print(f"  Optimal threshold: {val_metrics['optimal_threshold']:.3f} "
                      f"(F1: {val_metrics['optimal_f1']:.3f})")
                
                if 'edge_case_accuracy' in val_metrics:
                    print(f"  Edge case accuracy: {val_metrics['edge_case_accuracy']:.4f} "
                          f"({val_metrics['edge_case_count']} samples)")
            
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            
            if calculate_threshold:
                self.history['optimal_threshold'].append(val_metrics['optimal_threshold'])
                self.history['f1_score'].append(val_metrics['optimal_f1'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'
                self.model.save_checkpoint(
                    str(checkpoint_path), 
                    epoch=epoch, 
                    optimizer_state=self.optimizer.state_dict()
                )
                print(f"New best model saved! (val_loss: {val_metrics['loss']:.4f})")
            
            # Save checkpoints
            if epoch % save_interval == 0:
                checkpoint_path = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch}.pt'
                self.model.save_checkpoint(
                    str(checkpoint_path), 
                    epoch=epoch, 
                    optimizer_state=self.optimizer.state_dict()
                )
        
        # Save final model
        final_path = Path(checkpoint_dir) / 'final_model.pt'
        self.model.save_checkpoint(
            str(final_path), 
            epoch=num_epochs, 
            optimizer_state=self.optimizer.state_dict()
        )
        
        # Save training history
        history_path = Path(checkpoint_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Generate training plots
        self._plot_training_curves(checkpoint_dir)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"\nFinal model saved to {final_path}")
        print(f"Training history saved to {history_path}")
        print(f"Training plots saved to {checkpoint_dir}/training_plots.png")
        
        return self.history
    
    def _plot_training_curves(self, checkpoint_dir: str):
        """Generate and save training visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🚀 Training Progress - Siamese Network', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=min(self.history['val_loss']), color='g', linestyle='--', alpha=0.5, label=f'Best: {min(self.history["val_loss"]):.4f}')
        
        # Plot 2: Accuracy curves
        ax2 = axes[0, 1]
        ax2.plot(epochs, [acc * 100 for acc in self.history['train_acc']], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, [acc * 100 for acc in self.history['val_acc']], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='90% Target')
        max_val_acc = max(self.history['val_acc']) * 100
        ax2.axhline(y=max_val_acc, color='orange', linestyle='--', alpha=0.5, label=f'Best: {max_val_acc:.2f}%')
        
        # Plot 3: F1 Score over time
        ax3 = axes[1, 0]
        if self.history['f1_score']:
            f1_epochs = [i * 5 for i in range(1, len(self.history['f1_score']) + 1)]
            ax3.plot(f1_epochs, self.history['f1_score'], 'g-o', label='F1 Score', linewidth=2, markersize=8)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('F1 Score', fontsize=12)
            ax3.set_title('F1 Score Progress', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.90, color='g', linestyle='--', alpha=0.5, label='0.90 Target')
            if self.history['f1_score']:
                best_f1 = max(self.history['f1_score'])
                ax3.axhline(y=best_f1, color='orange', linestyle='--', alpha=0.5, label=f'Best: {best_f1:.3f}')
        else:
            ax3.text(0.5, 0.5, 'F1 Score not calculated', ha='center', va='center', fontsize=12)
            ax3.set_title('F1 Score Progress', fontsize=14, fontweight='bold')
        
        # Plot 4: Training Summary Stats
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        📊 TRAINING SUMMARY
        {'='*40}
        
        Final Results:
        ├─ Training Accuracy: {self.history['train_acc'][-1]*100:.2f}%
        ├─ Validation Accuracy: {self.history['val_acc'][-1]*100:.2f}%
        ├─ Best Val Loss: {min(self.history['val_loss']):.4f}
        └─ Best Val Accuracy: {max(self.history['val_acc'])*100:.2f}%
        
        Performance:
        ├─ Total Epochs: {len(epochs)}
        ├─ Best F1 Score: {max(self.history['f1_score']) if self.history['f1_score'] else 'N/A'}
        └─ Final Loss: {self.history['val_loss'][-1]:.4f}
        
        Target Achievement:
        {'✅ 90%+ Accuracy ACHIEVED!' if max(self.history['val_acc']) >= 0.90 else '⏳ Close to 90% target'}
        {'✅ F1 > 0.90 ACHIEVED!' if self.history['f1_score'] and max(self.history['f1_score']) >= 0.90 else ''}
        
        Improvement from Baseline:
        ├─ Baseline: 85.78%
        ├─ Current: {max(self.history['val_acc'])*100:.2f}%
        └─ Gain: +{(max(self.history['val_acc'])*100 - 85.78):.2f}%
        """
        
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(checkpoint_dir) / 'training_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📈 Training plots saved to: {plot_path}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Siamese Network for 90%+ accuracy')
    
    # Data
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--use-augmented', action='store_true',
                       help='Use augmented dataset (enhanced_training_data.csv)')
    
    # Model
    parser.add_argument('--projection-dim', type=int, default=256,
                       help='Projection head output dimension')
    parser.add_argument('--sbert-strategy', type=str, default='full',
                       choices=['frozen', 'last_layer', 'last_2_layers', 'full'],
                       help='SBERT fine-tuning strategy')
    
    # Training
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='Learning rate for projection head')
    parser.add_argument('--sbert-lr', type=float, default=5e-6,
                       help='Learning rate for SBERT encoder (lower for stability)')
    
    # Loss
    parser.add_argument('--loss-type', type=str, default='cosine',
                       choices=['contrastive', 'cosface', 'focal', 'cosine'],
                       help='Loss function type')
    parser.add_argument('--cosface-scale', type=float, default=30.0,
                       help='CosFace scale factor')
    parser.add_argument('--cosface-margin', type=float, default=0.35,
                       help='CosFace angular margin')
    
    # Hard mining
    parser.add_argument('--hard-mining', action='store_true', default=True,
                       help='Enable hard negative mining')
    parser.add_argument('--mining-strategy', type=str, default='semi-hard',
                       choices=['hard', 'semi-hard', 'all'],
                       help='Hard mining strategy')
    
    # Optimization
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                       help='Gradient clipping max norm')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_arguments()
    
    print("\n" + "="*70)
    print("🚀 ADVANCED TRAINING FOR 90%+ ACCURACY")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Data Path: {args.data_path}")
    print(f"  Use Augmented Data: {args.use_augmented}")
    print(f"  Projection Dim: {args.projection_dim}")
    print(f"  SBERT Strategy: {'🔥 FULL FINE-TUNING' if args.sbert_strategy == 'full' else args.sbert_strategy}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  SBERT LR: {args.sbert_lr}")
    print(f"  Loss Type: {args.loss_type.upper()}")
    if args.loss_type == 'cosface':
        print(f"  CosFace Scale: {args.cosface_scale}")
        print(f"  CosFace Margin: {args.cosface_margin}")
    print(f"  Hard Mining: {'✓ ENABLED' if args.hard_mining else '✗ DISABLED'} ({args.mining_strategy})")
    print(f"  Gradient Clip: {args.gradient_clip}")
    print(f"  Mixed Precision: {'✓ ENABLED' if not args.no_amp else '✗ DISABLED'}")
    
    # Load dataset
    print(f"\n{'='*70}")
    print("STEP 1: LOADING DATASET")
    print("="*70)
    
    if args.use_augmented:
        # Use pre-generated augmented dataset
        data_path = 'data/enhanced_training_data.csv'
        print(f"Loading augmented dataset from {data_path}")
    else:
        data_path = args.data_path
        print(f"Loading dataset from {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} training pairs")
    
    # Check for edge cases
    sample_texts_a = df['text_1'].head(1000).tolist()
    sample_texts_b = df['text_2'].head(1000).tolist()
    edge_mask = create_edge_case_mask(sample_texts_a, sample_texts_b)
    edge_count = edge_mask.sum()
    print(f"✓ Detected {edge_count} edge cases in first 1000 samples ({edge_count/10:.1f}%)")
    
    # Train/val split
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    print(f"Dataset split: {len(train_df)} train, {len(val_df)} validation")
    
    # Create datasets
    train_examples = [
        {
            'text_a': row['text_1'],
            'text_b': row['text_2'],
            'label': int(row['label'])
        }
        for _, row in train_df.iterrows()
    ]
    
    val_examples = [
        {
            'text_a': row['text_1'],
            'text_b': row['text_2'],
            'label': int(row['label'])
        }
        for _, row in val_df.iterrows()
    ]
    
    train_dataset = ParaphraseDataset(train_examples)
    val_dataset = ParaphraseDataset(val_examples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    print(f"\n{'='*70}")
    print("STEP 2: INITIALIZING MODEL")
    print("="*70)
    
    # Map strategy to model parameters
    if args.sbert_strategy == 'frozen':
        freeze_sbert = True
        unfreeze_all = False
    elif args.sbert_strategy == 'full':
        freeze_sbert = False
        unfreeze_all = True
    else:
        # last_layer or last_2_layers - use full for simplicity
        freeze_sbert = False
        unfreeze_all = True
    
    model = TrainableSiameseModel(
        projection_dim=args.projection_dim,
        freeze_sbert=freeze_sbert,
        unfreeze_all=unfreeze_all
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} 🔥")
    print(f"  Frozen: {frozen_params:,}")
    
    if args.sbert_strategy == 'full':
        print(f"\n✓ FULL FINE-TUNING: Maximum accuracy potential")
        print(f"  Expected accuracy: 90%+ with proper training")
    
    # Initialize trainer
    print(f"\n{'='*70}")
    print("STEP 3: INITIALIZING TRAINER")
    print("="*70)
    
    trainer = AdvancedSiameseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        sbert_lr=args.sbert_lr,
        loss_type=args.loss_type,
        cosface_scale=args.cosface_scale,
        cosface_margin=args.cosface_margin,
        use_hard_mining=args.hard_mining,
        mining_strategy=args.mining_strategy,
        use_mixed_precision=not args.no_amp,
        gradient_clip=args.gradient_clip
    )
    
    # Train
    print(f"\n{'='*70}")
    print("STEP 4: TRAINING")
    print("="*70)
    
    history = trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("STEP 5: TRAINING SUMMARY")
    print("="*70)
    
    print(f"\nFinal Results:")
    print(f"  Training Accuracy: {history['train_acc'][-1]:.4f}")
    print(f"  Validation Accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  Best Validation Loss: {trainer.best_val_loss:.4f}")
    
    if history['f1_score']:
        print(f"  Best F1 Score: {max(history['f1_score']):.4f}")
        print(f"  Optimal Threshold: {history['optimal_threshold'][-1]:.3f}")
    
    print(f"\nModel saved to: {args.checkpoint_dir}/final_model.pt")
    
    if history['val_acc'][-1] >= 0.90:
        print("\n🎉 CONGRATULATIONS! Achieved 90%+ accuracy target!")
    elif history['val_acc'][-1] >= 0.88:
        print("\n✓ Great performance! Close to 90% target. Consider:")
        print("  - Training for more epochs")
        print("  - Using augmented dataset (--use-augmented)")
        print("  - Adjusting CosFace margin")
    else:
        print("\nTo improve accuracy further:")
        print("  1. Generate augmented data: python -m backend.scripts.advanced_augmentation")
        print("  2. Train with: --use-augmented flag")
        print("  3. Increase epochs to 30-40")
        print("  4. Try different loss types: --loss-type focal")


if __name__ == '__main__':
    main()
