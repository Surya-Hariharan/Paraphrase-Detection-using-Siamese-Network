"""Training pipeline"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict, List
import pandas as pd
from pathlib import Path
import json

from backend.core.models.siamese import SiameseModel, ContrastiveLoss


class ParaphraseDataset(Dataset):
    """Dataset for paraphrase detection"""
    
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def load_dataset(data_path: str, train_ratio=0.8, val_ratio=0.1):
    """Load and split dataset"""
    print(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    train_size = int(train_ratio * len(df))
    val_size = int(val_ratio * len(df))
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    print(f"Dataset split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    train_examples = [
        {'text_a': row['text_1'], 'text_b': row['text_2'], 'label': int(row['label'])}
        for _, row in train_df.iterrows()
    ]
    
    val_examples = [
        {'text_a': row['text_1'], 'text_b': row['text_2'], 'label': int(row['label'])}
        for _, row in val_df.iterrows()
    ]
    
    test_examples = [
        {'text_a': row['text_1'], 'text_b': row['text_2'], 'label': int(row['label'])}
        for _, row in test_df.iterrows()
    ]
    
    return train_examples, val_examples, test_examples


class Trainer:
    """Training pipeline for Siamese Network"""
    
    def __init__(self, model, train_dataset, val_dataset=None, test_dataset=None,
                 batch_size=32, learning_rate=5e-5, num_epochs=20, device='cuda'):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None
        
        # Loss and optimizer
        self.criterion = ContrastiveLoss(margin=0.5)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        self.scaler = GradScaler()
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_acc': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            text_a = batch['text_a']
            text_b = batch['text_b']
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                similarities, _, _ = self.model(text_a, text_b)
                loss = self.criterion(similarities, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += loss.item()
            predictions = (similarities >= 0.5).long()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = train_loss / len(self.train_loader)
        accuracy = train_correct / train_total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        if not self.val_loader:
            return None, None
        
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                text_a = batch['text_a']
                text_b = batch['text_b']
                labels = batch['label'].to(self.device)
                
                similarities, _, _ = self.model(text_a, text_b)
                loss = self.criterion(similarities, labels)
                
                val_loss += loss.item()
                predictions = (similarities >= 0.5).long()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        avg_loss = val_loss / len(self.val_loader)
        accuracy = val_correct / val_total
        
        return avg_loss, accuracy
    
    def test(self):
        """Test the model"""
        if not self.test_loader:
            return None
        
        self.model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                text_a = batch['text_a']
                text_b = batch['text_b']
                labels = batch['label'].to(self.device)
                
                similarities, _, _ = self.model(text_a, text_b)
                predictions = (similarities >= 0.5).long()
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)
        
        accuracy = test_correct / test_total
        return accuracy
    
    def train(self, checkpoint_dir='checkpoints', save_every=5, early_stopping_patience=None, min_epochs=10):
        """Full training loop"""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        print()
        
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Test
            test_acc = self.test()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            if test_acc is not None:
                self.history['test_acc'].append(test_acc)
            
            # Print metrics
            print(f"Training   - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            if val_loss is not None:
                print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            if test_acc is not None:
                print(f"Test       - Accuracy: {test_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss is not None and val_loss < self.best_val_loss:
                print(f"✓ New best model saved! (val_loss: {val_loss:.4f})")
                self.best_val_loss = val_loss
                self.model.save(str(checkpoint_path / 'best_model.pt'))
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.model.save(str(checkpoint_path / f'checkpoint_epoch_{epoch+1}.pt'))
            
            # Early stopping
            if early_stopping_patience and epoch >= min_epochs:
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        # Save final model
        self.model.save(str(checkpoint_path / 'final_model.pt'))
        
        # Save history
        with open(checkpoint_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nFinal model saved to {checkpoint_path}/final_model.pt")
        print(f"Training history saved to {checkpoint_path}/training_history.json")
        
        return self.history
