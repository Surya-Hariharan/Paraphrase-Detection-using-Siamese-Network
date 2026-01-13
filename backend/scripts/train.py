"""
Training Script for Siamese Network - From Scratch
===================================================

Complete training pipeline for paraphrase detection using Siamese architecture.

Usage:
    python backend/scripts/train.py
    
    # With custom parameters:
    python backend/scripts/train.py --epochs 20 --batch-size 32 --lr 5e-5
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.model import TrainableSiameseModel, ContrastiveLoss
from backend.core.trainer import ParaphraseDataset, SiameseTrainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Siamese Network for Paraphrase Detection")
    
    # Data arguments
    parser.add_argument('--data-path', type=str, 
                       default='data/quora_siamese_train.csv',
                       help='Path to training CSV file')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, 
                       default='all-MiniLM-L6-v2',
                       help='Sentence-BERT model name')
    parser.add_argument('--projection-dim', type=int, default=256,
                       help='Projection head output dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay for optimizer')
    parser.add_argument('--warmup-steps', type=int, default=500,
                       help='Warmup steps for learning rate scheduler')
    parser.add_argument('--margin', type=float, default=0.5,
                       help='Contrastive loss margin')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    return parser.parse_args()


def setup_directories(checkpoint_dir: str):
    """Create necessary directories"""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)


def main():
    """Main training function"""
    args = parse_arguments()
    
    print("=" * 80)
    print("Siamese Network Training - From Scratch")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data_path}")
    print(f"  Model: {args.model_name}")
    print(f"  Projection Dim: {args.projection_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Mixed Precision: {args.mixed_precision}")
    print()
    
    # Setup directories
    setup_directories(args.checkpoint_dir)
    
    # Initialize model
    print("Initializing model...")
    model = TrainableSiameseModel(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        device=args.device
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = SiameseTrainer(
        model=model,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    print(f"Total samples: {len(df)}")
    
    # Split data
    val_size = int(len(df) * args.val_split)
    train_size = len(df) - val_size
    
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = ParaphraseDataset(train_df)
    val_dataset = ParaphraseDataset(val_df)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Setup training components
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = optim.AdamW(
        model.projection_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-7
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start = datetime.now()
        
        print(f"Epoch [{epoch+1}/{args.epochs}]")
        print("-" * 80)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            text_a = batch['text_a']
            text_b = batch['text_b']
            labels = batch['label'].to(args.device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if args.mixed_precision:
                with autocast():
                    emb_a, emb_b = model.forward_batch(text_a, text_b)
                    loss = criterion(emb_a, emb_b, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.projection_head.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                emb_a, emb_b = model.forward_batch(text_a, text_b)
                loss = criterion(emb_a, emb_b, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.projection_head.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            # Calculate accuracy
            with torch.no_grad():
                distances = torch.nn.functional.cosine_similarity(emb_a, emb_b)
                predictions = (distances > 0.5).float()
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
            
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                text_a = batch['text_a']
                text_b = batch['text_b']
                labels = batch['label'].to(args.device)
                
                emb_a, emb_b = model.forward_batch(text_a, text_b)
                loss = criterion(emb_a, emb_b, labels)
                
                distances = torch.nn.functional.cosine_similarity(emb_a, emb_b)
                predictions = (distances > 0.5).float()
                
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        # Print epoch summary
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_accuracy:.2f}%")
        print(f"  Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint_path = Path(args.checkpoint_dir) / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
            }, checkpoint_path)
            print(f"  ✓ Best model saved (Val Acc: {val_accuracy:.2f}%)")
        
        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved")
        
        print()
    
    # Save final model
    final_checkpoint_path = Path(args.checkpoint_dir) / "final_model.pt"
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, final_checkpoint_path)
    
    # Save training history
    history_path = Path(args.checkpoint_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {args.checkpoint_dir}/")
    print(f"Training history saved to: {history_path}")
    print()


if __name__ == "__main__":
    main()
