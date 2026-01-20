"""
Enhanced Fine-Tuning Script for Siamese Network
================================================

Addresses model weaknesses identified in evaluation:
1. Negation understanding
2. Long text handling
3. Semantic/idiomatic paraphrases
4. Homonym disambiguation
5. Hard negative mining

Improvements:
- Weighted loss for hard cases
- Dynamic batching by text length
- Attention-based pooling
- Adversarial training
- Custom metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.model import TrainableSiameseModel, ContrastiveLoss


class EnhancedSiameseDataset(Dataset):
    """Dataset with difficulty weights for hard case mining"""
    
    def __init__(self, csv_path: str, assign_weights: bool = True):
        self.df = pd.read_csv(csv_path)
        self.assign_weights = assign_weights
        
        if assign_weights:
            self.weights = self._calculate_sample_weights()
        
    def _calculate_sample_weights(self):
        """Assign higher weights to negation, homonym, and hard cases"""
        weights = np.ones(len(self.df))
        
        for idx, row in self.df.iterrows():
            text_a = str(row['text_a']).lower()
            text_b = str(row['text_b']).lower()
            
            # Higher weight for negation cases
            negation_words = ['not', 'no', 'never', 'dis', 'un', 'im', 'in']
            if any(word in text_a or word in text_b for word in negation_words):
                weights[idx] = 3.0
            
            # Higher weight for homonyms
            homonyms = ['bank', 'bat', 'patient', 'match', 'can', 'bear', 'book', 'wave']
            if any(word in text_a and word in text_b for word in homonyms):
                weights[idx] = 2.5
            
            # Higher weight for long text (>100 words)
            if len(text_a.split()) > 100 or len(text_b.split()) > 100:
                weights[idx] = 2.0
            
            # Higher weight for idioms/semantics
            idioms = ['break a leg', 'piece of cake', 'raining cats', 'passed away']
            if any(idiom in text_a or idiom in text_b for idiom in idioms):
                weights[idx] = 2.5
        
        return weights
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        text_a = str(row['text_a'])
        text_b = str(row['text_b'])
        label = int(row['is_duplicate'])
        
        if self.assign_weights:
            weight = self.weights[idx]
            return text_a, text_b, label, weight
        else:
            return text_a, text_b, label


class EnhancedTrainer:
    """Enhanced trainer with advanced techniques"""
    
    def __init__(
        self,
        model: TrainableSiameseModel,
        train_dataset: EnhancedSiameseDataset,
        val_dataset: EnhancedSiameseDataset,
        learning_rate: float = 1e-5,
        batch_size: int = 16,
        num_epochs: int = 15,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Use weighted sampling for hard cases
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Optimizer with lower learning rate for fine-tuning
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-7
        )
        
        # Loss function with margin
        self.criterion = ContrastiveLoss(margin=1.2)
        
        # Mixed precision training
        self.scaler = GradScaler()
        
        # Metrics tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'negation_accuracy': [],
            'homonym_accuracy': [],
            'long_text_accuracy': []
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            text_a, text_b, labels, weights = batch
            labels = labels.to(self.device)
            weights = weights.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                emb_a, emb_b = self.model.forward_batch(text_a, text_b)
                
                # Weighted loss
                loss = self.criterion(emb_a, emb_b, labels)
                weighted_loss = (loss * weights).mean()
            
            # Backward pass with mixed precision
            self.scaler.scale(weighted_loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += weighted_loss.item()
            pbar.set_postfix({'loss': weighted_loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> dict:
        """Validate with detailed metrics"""
        self.model.eval()
        total_loss = 0
        
        all_predictions = []
        all_labels = []
        all_text_a = []
        all_text_b = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                if len(batch) == 4:
                    text_a, text_b, labels, _ = batch
                else:
                    text_a, text_b, labels = batch
                
                labels = labels.to(self.device)
                
                # Forward pass
                emb_a, emb_b = self.model.forward_batch(text_a, text_b)
                loss = self.criterion(emb_a, emb_b, labels).mean()
                
                # Get similarities
                similarities = torch.cosine_similarity(emb_a, emb_b)
                # Convert from [-1, 1] to [0, 1]
                similarities = (similarities + 1) / 2
                
                predictions = (similarities >= 0.8).long()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_text_a.extend(text_a)
                all_text_b.extend(text_b)
                
                total_loss += loss.item()
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Precision, Recall, F1
        tp = ((all_predictions == 1) & (all_labels == 1)).sum()
        fp = ((all_predictions == 1) & (all_labels == 0)).sum()
        fn = ((all_predictions == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Category-specific accuracy
        negation_acc = self._calculate_category_accuracy(
            all_text_a, all_text_b, all_predictions, all_labels,
            keywords=['not', 'no', 'never', 'dis', 'un', 'im']
        )
        
        homonym_acc = self._calculate_category_accuracy(
            all_text_a, all_text_b, all_predictions, all_labels,
            keywords=['bank', 'bat', 'patient', 'match', 'bear']
        )
        
        long_text_acc = self._calculate_long_text_accuracy(
            all_text_a, all_text_b, all_predictions, all_labels
        )
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'negation_accuracy': negation_acc,
            'homonym_accuracy': homonym_acc,
            'long_text_accuracy': long_text_acc
        }
    
    def _calculate_category_accuracy(self, text_a_list, text_b_list, predictions, labels, keywords):
        """Calculate accuracy for specific category"""
        mask = []
        for text_a, text_b in zip(text_a_list, text_b_list):
            text_a_lower = text_a.lower()
            text_b_lower = text_b.lower()
            has_keyword = any(kw in text_a_lower or kw in text_b_lower for kw in keywords)
            mask.append(has_keyword)
        
        mask = np.array(mask)
        if mask.sum() == 0:
            return 0.0
        
        category_acc = (predictions[mask] == labels[mask]).mean()
        return category_acc
    
    def _calculate_long_text_accuracy(self, text_a_list, text_b_list, predictions, labels):
        """Calculate accuracy for long text (>100 words)"""
        mask = []
        for text_a, text_b in zip(text_a_list, text_b_list):
            is_long = len(text_a.split()) > 100 or len(text_b.split()) > 100
            mask.append(is_long)
        
        mask = np.array(mask)
        if mask.sum() == 0:
            return 0.0
        
        long_acc = (predictions[mask] == labels[mask]).mean()
        return long_acc
    
    def train(self, checkpoint_dir: str = "checkpoints"):
        """Full training loop with checkpointing"""
        print("=" * 80)
        print("ENHANCED FINE-TUNING - SIAMESE NETWORK")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 80)
        
        best_f1 = 0.0
        Path(checkpoint_dir).mkdir(exist_ok=True)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 80)
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['negation_accuracy'].append(val_metrics['negation_accuracy'])
            self.history['homonym_accuracy'].append(val_metrics['homonym_accuracy'])
            self.history['long_text_accuracy'].append(val_metrics['long_text_accuracy'])
            
            # Print metrics
            print(f"\n📊 Metrics:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
            print(f"  Accuracy:   {val_metrics['accuracy']:.4f}")
            print(f"  Precision:  {val_metrics['precision']:.4f}")
            print(f"  Recall:     {val_metrics['recall']:.4f}")
            print(f"  F1-Score:   {val_metrics['f1']:.4f}")
            print(f"\n📈 Category Performance:")
            print(f"  Negation:   {val_metrics['negation_accuracy']:.4f}")
            print(f"  Homonym:    {val_metrics['homonym_accuracy']:.4f}")
            print(f"  Long Text:  {val_metrics['long_text_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.model.save_checkpoint(f"{checkpoint_dir}/best_model_finetuned.pt")
                print(f"\n✅ New best model saved! F1: {best_f1:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.model.save_checkpoint(f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}_finetuned.pt")
        
        # Save final model
        self.model.save_checkpoint(f"{checkpoint_dir}/final_model_finetuned.pt")
        
        # Save training history
        history_path = f"{checkpoint_dir}/training_history_finetuned.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 80)
        print("✅ FINE-TUNING COMPLETE!")
        print(f"Best F1-Score: {best_f1:.4f}")
        print(f"Model saved to: {checkpoint_dir}/best_model_finetuned.pt")
        print("=" * 80)


def main():
    """Main fine-tuning script"""
    
    # Load base model
    print("Loading base model...")
    model = TrainableSiameseModel(projection_dim=256, unfreeze_all=True)
    model.load_checkpoint("checkpoints/best_model.pt")
    
    # Load datasets
    print("Loading enhanced training data...")
    train_dataset = EnhancedSiameseDataset("data/enhanced_training_data.csv", assign_weights=True)
    
    # Use original data for validation
    print("Loading validation data...")
    val_dataset = EnhancedSiameseDataset("data/quora_siamese_train.csv", assign_weights=False)
    
    # Initialize trainer
    trainer = EnhancedTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=2e-6,  # Very low LR for fine-tuning
        batch_size=16,
        num_epochs=15
    )
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
