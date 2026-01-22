"""Training pipeline"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import Optional, Dict, List
import pandas as pd
from pathlib import Path
import json

from backend.core.models.siamese import SiameseModel, ContrastiveLoss


class ParaphraseDataset(Dataset):
    """Dataset for paraphrase detection"""
    
    def __init__(self, examples: List[Dict]):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class Trainer:
    """Training pipeline for Siamese Network"""
    
    def __init__(
        self,
        model: SiameseModel,
        train_dataset: ParaphraseDataset,
        val_dataset: Optional[ParaphraseDataset] = None,
        batch_size: int = 32,
        learning_rate: float = 2e-4,
        num_epochs: int = 50,
        checkpoint_dir: str = "checkpoints",
        device: str = "cuda"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = ContrastiveLoss(margin=1.0)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler
        total_steps = (len(train_dataset) // batch_size) * num_epochs
        warmup_steps = int(0.1 * total_steps)
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
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self) -> tuple:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            text_a = batch['text_a']
            text_b = batch['text_b']
            labels = torch.tensor(batch['label'], dtype=torch.float).to(self.device)
            
            # Forward
            emb_a, emb_b = self.model(text_a, text_b)
            loss = self.criterion(emb_a, emb_b, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Metrics
            with torch.no_grad():
                similarity = nn.functional.cosine_similarity(emb_a, emb_b, dim=1)
                predictions = (similarity > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> tuple:
        """Validate model"""
        if not self.val_dataset:
            return 0, 0
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                text_a = batch['text_a']
                text_b = batch['text_b']
                labels = torch.tensor(batch['label'], dtype=torch.float).to(self.device)
                
                emb_a, emb_b = self.model(text_a, text_b)
                loss = self.criterion(emb_a, emb_b, labels)
                
                similarity = nn.functional.cosine_similarity(emb_a, emb_b, dim=1)
                predictions = (similarity > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, save_every: int = 5, early_stopping_patience: int = 8, min_epochs: int = 30):
        """Full training loop"""
        print(f"\nTraining on {len(self.train_dataset)} examples")
        print(f"Device: {self.device}\n")
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            if self.val_dataset:
                val_loss, val_acc = self.validate()
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Early stopping on validation loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self.model.save(str(self.checkpoint_dir / "best_model.pt"))
                    print("✓ New best model saved")
                else:
                    patience_counter += 1
            else:
                # Early stopping on training loss
                if train_loss < best_loss:
                    best_loss = train_loss
                    patience_counter = 0
                    self.model.save(str(self.checkpoint_dir / "best_model.pt"))
                    print("✓ New best model saved")
                else:
                    patience_counter += 1
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.model.save(str(self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"))
            
            # Early stopping
            if epoch >= min_epochs and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {patience_counter} epochs)")
                break
        
        # Save final model
        self.model.save(str(self.checkpoint_dir / "final_model.pt"))
        
        # Save history
        with open(self.checkpoint_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\nTraining complete!")
        return self.history


def load_dataset(csv_path: str) -> List[Dict]:
    """Load dataset from CSV"""
    df = pd.read_csv(csv_path)
    
    examples = []
    for _, row in df.iterrows():
        examples.append({
            'text_a': str(row['text_1']),
            'text_b': str(row['text_2']),
            'label': int(row['label'])
        })
    
    return examples
