"""
Training Pipeline - Train Siamese Network with AI Agent Supervision

This module implements a complete training pipeline for the Siamese Network with:
1. **AI Agents for Training Supervision**:
   - Data Validator Agent: Validates training data quality
   - Training Monitor Agent: Monitors training progress and suggests adjustments
   
2. **Training Features**:
   - Batch training with contrastive loss
   - Learning rate scheduling
   - Early stopping
   - Checkpoint saving
   - Metrics tracking (loss, accuracy, F1)

3. **Document Processing**:
   - Handles large documents via chunking
   - Aggregates embeddings across chunks
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from transformers import get_cosine_schedule_with_warmup

# Our modules
from .neural_engine import TrainableSiameseModel, ContrastiveLoss
from dotenv import load_dotenv

# Agent system
try:
    from backend.agents import (
        DataValidationAgent,
        TrainingMonitorAgent,
        InferenceValidatorAgent,
        AgentConfig
    )
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False

load_dotenv()


# ============================================================
# Dataset Class
# ============================================================

class ParaphraseDataset(Dataset):
    """
    Dataset for paraphrase detection training.
    
    Each example contains:
    - text_a: First document
    - text_b: Second document
    - label: 1 if paraphrase/duplicate, 0 if unique
    """
    
    def __init__(
        self,
        examples: List[Dict[str, Any]],
        use_chunks: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            examples: List of {text_a, text_b, label} dicts
            use_chunks: If True, use pre-chunked text
        """
        self.examples = examples
        self.use_chunks = use_chunks
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        if self.use_chunks and 'chunks_a' in example:
            # Use first chunk (or aggregate later)
            text_a = example['chunks_a'][0] if example['chunks_a'] else example['text_a']
            text_b = example['chunks_b'][0] if example['chunks_b'] else example['text_b']
        else:
            text_a = example['text_a']
            text_b = example['text_b']
        
        return {
            'text_a': text_a,
            'text_b': text_b,
            'label': example['label']
        }


# ============================================================
# Trainer Class
# ============================================================

class SiameseTrainer:
    """
    Complete training pipeline for Siamese Network.
    
    Features:
    - Contrastive loss training
    - Agent-supervised training
    - Checkpointing
    - Metrics tracking
    """
    
    def __init__(
        self,
        model: TrainableSiameseModel,
        train_dataset: ParaphraseDataset,
        val_dataset: Optional[ParaphraseDataset] = None,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        margin: float = 1.0,
        use_agents: bool = False,
        checkpoint_dir: str = "./checkpoints",
        num_epochs: int = 10
    ):
        """
        Initialize trainer.
        
        Args:
            model: Trainable Siamese model
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            margin: Contrastive loss margin
            use_agents: Deprecated - agents are disabled
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_agents = use_agents
        
        # Initialize AI Agents if available and enabled
        self.agents_enabled = use_agents and AGENTS_AVAILABLE
        if self.agents_enabled:
            try:
                agent_config = AgentConfig(provider="gemini", enable_logging=True)
                self.data_validator = DataValidationAgent(agent_config)
                self.training_monitor = TrainingMonitorAgent(agent_config)
                print("✓ AI Agents initialized (Gemini-powered)")
            except Exception as e:
                print(f"⚠️  AI Agents disabled: {str(e)}")
                self.agents_enabled = False
        else:
            self.data_validator = None
            self.training_monitor = None
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training components
        self.criterion = ContrastiveLoss(margin=margin)
        
        # Use differential learning rates if fully unfreezing SBERT
        if hasattr(model, 'unfreeze_all') and model.unfreeze_all:
            # Lower LR for SBERT, higher for projection head
            param_groups = [
                {'params': model.sbert_encoder.parameters(), 'lr': learning_rate * 0.1, 'name': 'sbert'},
                {'params': model.projection_head.parameters(), 'lr': learning_rate, 'name': 'projection_head'}
            ]
            self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        else:
            self.optimizer = optim.AdamW(
                model.get_trainable_parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
        
        # Linear warmup + cosine annealing: warmup stabilizes fine-tuning, cosine decay prevents overfitting
        self.num_epochs = num_epochs  # Store for scheduler calculation
        steps_per_epoch = len(train_dataset) // batch_size
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # Linear warmup + cosine annealing: warmup stabilizes fine-tuning, cosine decay prevents overfitting
        steps_per_epoch = len(train_dataset) // batch_size
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        else:
            self.val_loader = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Mixed precision training for 2x speedup
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        if self.use_amp:
            print("✓ Mixed precision training (AMP) enabled for 2x speedup")
        
        # Agents permanently disabled for maximum performance
        self.agents = None
    
    def compute_dataset_stats(self):
        """Compute statistics about the dataset."""
        labels = []
        lengths_a = []
        lengths_b = []
        
        # Sample from dataset to compute stats
        for i, item in enumerate(self.train_dataset):
            if i >= 1000:  # Sample first 1000 for efficiency
                break
            labels.append(int(item['label']))
            lengths_a.append(len(item['text_a']))
            lengths_b.append(len(item['text_b']))
        
        positive_count = sum(labels)
        total_count = len(labels)
        
        return {
            'total_examples': len(self.train_dataset),
            'sampled_examples': total_count,
            'positive_examples': positive_count,
            'negative_examples': total_count - positive_count,
            'positive_ratio': positive_count / total_count if total_count > 0 else 0,
            'avg_length_a': np.mean(lengths_a) if lengths_a else 0,
            'avg_length_b': np.mean(lengths_b) if lengths_b else 0,
            'std_length_a': np.std(lengths_a) if lengths_a else 0,
            'std_length_b': np.std(lengths_b) if lengths_b else 0
        }
    
    def validate_data_with_agent(self):
        """Disabled - agents removed for performance."""
        pass
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch with PROPER accuracy calculation using optimal threshold.
        
        Returns:
            (average_loss, accuracy_with_optimal_threshold)
        """
        self.model.train()
        total_loss = 0.0
        all_similarities = []
        all_labels = []
        gradient_verified = False
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            texts_a = batch['text_a']
            texts_b = batch['text_b']
            labels = batch['label'].float().to(self.model.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    embeddings_a, embeddings_b = self.model.forward_batch(texts_a, texts_b)
                    loss = self.criterion(embeddings_a, embeddings_b, labels)
                
                # Check for NaN loss before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Skipping batch due to NaN/Inf loss")
                    continue
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings_a, embeddings_b = self.model.forward_batch(texts_a, texts_b)
                loss = self.criterion(embeddings_a, embeddings_b, labels)
                
                # Check for NaN loss before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  Skipping batch due to NaN/Inf loss")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Gradient verification (first batch of first epoch only)
            if epoch == 1 and not gradient_verified:
                grad_norms = [
                    param.grad.norm().item() 
                    for param in self.model.projection_head.parameters() 
                    if param.grad is not None
                ]
                if all(g > 0 for g in grad_norms):
                    gradient_verified = True
                elif len(grad_norms) == 0:
                    print("Warning: No gradients found in projection head")
            
            # Step scheduler after each batch
            self.scheduler.step()
            
            # Collect similarities for threshold optimization
            similarities = self.model.compute_similarity_batch(embeddings_a, embeddings_b)
            all_similarities.extend(similarities.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
            total_loss += loss.item()
            
            # Update progress bar with running metrics
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })
        
        # Calculate accuracy with OPTIMAL threshold (not hardcoded 0.5)
        threshold_results = self.model.threshold_optimizer.find_optimal_threshold(
            all_similarities,
            all_labels
        )
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = threshold_results['best_accuracy']
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate on validation set with threshold optimization.
        
        Returns:
            (average_loss, accuracy_with_optimal_threshold)
        """
        if not self.val_dataset:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        all_similarities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                texts_a = batch['text_a']
                texts_b = batch['text_b']
                labels = batch['label'].float().to(self.model.device)
                
                # Forward pass
                embeddings_a, embeddings_b = self.model.forward_batch(texts_a, texts_b)
                
                # Compute loss
                loss = self.criterion(embeddings_a, embeddings_b, labels)
                
                # Collect similarities and labels for threshold optimization
                similarities = self.model.compute_similarity_batch(embeddings_a, embeddings_b)
                all_similarities.extend(similarities.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                total_loss += loss.item()
        
        # Optimize threshold for binary classification
        threshold_results = self.model.threshold_optimizer.find_optimal_threshold(
            all_similarities,
            all_labels
        )
        
        avg_loss = total_loss / len(self.val_loader)
        # Use accuracy with optimal threshold (better than fixed 0.5)
        accuracy = threshold_results['best_accuracy']
        
        # Print threshold info occasionally
        if hasattr(self, '_validation_count'):
            self._validation_count += 1
        else:
            self._validation_count = 1
        
        if self._validation_count % 5 == 1:  # Print every 5 validations
            print(f"  Optimal threshold: {threshold_results['best_threshold']:.3f} "
                  f"(F1: {threshold_results['best_f1']:.3f})")
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 5,
        agent_check_every: Optional[int] = None
    ):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Override checkpoint directory (optional)
            checkpoint_every: Save checkpoint every N epochs
            agent_check_every: Run agent monitoring every N epochs (None to disable)
        """
        print("\n" + "-"*50)
        print("STARTING TRAINING")
        print("-"*50)
        
        # Override checkpoint dir if provided
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Pre-training validation with agent
        if self.agents:
            self.validate_data_with_agent()
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'-'*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'-'*50}")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"\nTraining - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            
            # Validate
            if self.val_dataset:
                val_loss, val_acc = self.validate()
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                
                # Agent monitoring (after each epoch if enabled)
                if self.agents_enabled and self.training_monitor:
                    monitor_result = self.training_monitor.monitor_epoch(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_acc=train_acc,
                        val_acc=val_acc
                    )
                    
                    if monitor_result["warnings"]:
                        print(f"\n🤖 AI Agent Warnings:")
                        for warning in monitor_result["warnings"]:
                            print(f"   ⚠️  {warning}")
                    
                    if monitor_result["suggestions"]:
                        print(f"\n💡 AI Agent Suggestions:")
                        for suggestion in monitor_result["suggestions"][:3]:  # Top 3
                            print(f"   • {suggestion}")
                    
                    if monitor_result["should_stop"]:
                        print("\n🛑 AI Agent recommends stopping training!")
                        break
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = self.checkpoint_dir / "best_model.pt"
                    self.model.save_checkpoint(
                        str(checkpoint_path),
                        epoch,
                        self.optimizer.state_dict()
                    )
                    print(f"New best model saved! (val_loss: {val_loss:.4f})")
            
            # Save checkpoint
            if epoch % checkpoint_every == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                self.model.save_checkpoint(
                    str(checkpoint_path),
                    epoch,
                    self.optimizer.state_dict()
                )
        
        print("\n" + "-"*50)
        print("TRAINING COMPLETE!")
        print("-"*50)
        
        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        self.model.save_checkpoint(str(final_path), num_epochs, self.optimizer.state_dict())
        
        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nFinal model saved to {final_path}")
        print(f"Training history saved to {history_path}")


# ============================================================
# Convenience Functions
# ============================================================

def create_simple_dataset(
    text_pairs: List[Tuple[str, str]],
    labels: List[int]
) -> ParaphraseDataset:
    """
    Create a simple dataset from text pairs and labels.
    
    Args:
        text_pairs: List of (text_a, text_b) tuples
        labels: List of labels (1=paraphrase, 0=unique)
        
    Returns:
        ParaphraseDataset
    """
    examples = [
        {
            'text_a': pair[0],
            'text_b': pair[1],
            'label': label
        }
        for pair, label in zip(text_pairs, labels)
    ]
    
    return ParaphraseDataset(examples)


if __name__ == "__main__":
    print("\nTesting Training Pipeline")
    print("-" * 40)
    
    # Create a small dummy dataset
    text_pairs = [
        ("The cat sat on the mat", "A cat was sitting on the rug"),
        ("Machine learning is AI", "ML is artificial intelligence"),
        ("The weather is sunny", "I love eating pizza"),
        ("Python is a programming language", "Python is used for coding"),
        ("The car is red", "The book is interesting"),
    ]
    
    labels = [1, 1, 0, 1, 0]  # 1=paraphrase, 0=unique
    
    dataset = create_simple_dataset(text_pairs, labels)
    
    print(f"Created dataset with {len(dataset)} examples")
    print(f"  Paraphrases: {sum(labels)}")
    print(f"  Unique: {len(labels) - sum(labels)}")
    
    print("\nTraining pipeline ready!")
    print("Use train.py to start actual training")
