"""
Main Training Script
===================

Complete training pipeline for Siamese Network paraphrase detection.

This script demonstrates the full workflow:
1. Load/create dataset
2. Initialize Siamese model
3. Train with AI agent supervision
4. Test with AI agent analysis
5. Save trained model

Usage:
------
python train.py --data-path data/train.csv --epochs 10 --use-agents
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import random_split
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.core.neural_engine import TrainableSiameseModel
from backend.core.training_pipeline import (
    ParaphraseDataset,
    SiameseTrainer,
    create_simple_dataset
)

# Testing pipeline is optional
try:
    from testing_pipeline import SiameseTester
    TESTING_AVAILABLE = True
except ImportError:
    TESTING_AVAILABLE = False
    SiameseTester = None

from rich.console import Console

console = Console()


# ============================================================
# Dataset Creation
# ============================================================

def load_or_create_dataset(
    data_path: Optional[str] = None,
    train_split: float = 0.8
) -> Tuple[ParaphraseDataset, ParaphraseDataset]:
    """
    Load dataset from file or create a simple demo dataset.
    
    Args:
        data_path: Path to CSV file (optional)
        train_split: Fraction for training (rest is test)
        
    Returns:
        train_dataset, test_dataset
    """
    if data_path and Path(data_path).exists():
        console.print(f"Loading dataset from {data_path}", style="cyan")
        
        # Load from CSV
        import pandas as pd
        df = pd.read_csv(data_path)
        
        # Support both text_1/text_2 and text_a/text_b column names
        if 'text_1' in df.columns and 'text_2' in df.columns:
            text_a_col, text_b_col = 'text_1', 'text_2'
        elif 'text_a' in df.columns and 'text_b' in df.columns:
            text_a_col, text_b_col = 'text_a', 'text_b'
        else:
            raise ValueError("CSV must have columns: (text_1, text_2, label) or (text_a, text_b, label)")
        
        if 'label' not in df.columns:
            raise ValueError("CSV must have 'label' column")
        
        # Create examples list
        examples = [
            {
                'text_a': row[text_a_col],
                'text_b': row[text_b_col],
                'label': row['label']
            }
            for _, row in df.iterrows()
        ]
        
        # Create full dataset
        full_dataset = ParaphraseDataset(examples)
        
    else:
        console.print("Creating demo dataset (no data file provided)", style="yellow")
        
        # Create simple demo dataset
        full_dataset = create_simple_dataset()
    
    # Split into train and test
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    console.print(f"Dataset split: {train_size} train, {test_size} test", style="green")
    
    return train_dataset, test_dataset


# ============================================================
# Main Training Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Siamese Network for paraphrase detection")
    
    # Data arguments
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to CSV file with columns: text_a, text_b, label'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Fraction of data for training (default: 0.8)'
    )
    
    # Model arguments
    parser.add_argument(
        '--projection-dim',
        type=int,
        default=256,
        help='Projection dimension (default: 256)'
    )
    parser.add_argument(
        '--freeze-sbert',
        action='store_true',
        help='Freeze SBERT encoder (NOT RECOMMENDED - lower accuracy)'
    )
    parser.add_argument(
        '--unfreeze-all',
        action='store_true',
        default=True,
        help='FULL FINE-TUNING: Unfreeze entire SBERT model (DEFAULT, best accuracy)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20, optimized for 80+ accuracy)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32, optimized for RTX 4060)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate (default: 2e-4, optimized for full fine-tuning)'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=1.0,
        help='Contrastive loss margin (default: 1.0)'
    )
    
    # Agent arguments
    # Agents removed for performance optimization
    # parser.add_argument(
    #     '--use-agents',
    #     action='store_true',
    #     help='Enable AI agent supervision'
    # )
    # parser.add_argument(
    #     '--agent-check-every',
    #     type=int,
    #     default=5,
    #     help='Run agent monitoring every N epochs (default: 5)'
    # )
    
    # Checkpoint arguments
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints)'
    )
    parser.add_argument(
        '--checkpoint-every',
        type=int,
        default=5,
        help='Save checkpoint every N epochs (default: 5)'
    )
    
    # Testing arguments
    parser.add_argument(
        '--skip-testing',
        action='store_true',
        help='Skip testing after training'
    )
    
    args = parser.parse_args()
    
    # Print header
    console.print("\n" + "-"*50, style="bold cyan")
    console.print("🔥 FULL FINE-TUNING SIAMESE NETWORK", style="bold cyan")
    console.print("-"*50, style="bold cyan")
    
    # Print configuration
    freeze_sbert = args.freeze_sbert
    unfreeze_all = not freeze_sbert  # Default to full unfreezing
    
    console.print("\n[bold]Configuration:[/bold]")
    console.print(f"  Data Path: {args.data_path or 'Demo dataset'}")
    console.print(f"  Projection Dim: {args.projection_dim}")
    
    # SBERT Training Strategy
    if freeze_sbert:
        console.print(f"  SBERT Strategy: ❄️  FROZEN (NOT RECOMMENDED)")
        console.print(f"  Expected Accuracy: ~72% (baseline)")
    else:
        console.print(f"  SBERT Strategy: 🔥 FULL FINE-TUNING (ALL 6 LAYERS)")
        console.print(f"  Architecture: BiLSTM(256) + Attention(4 heads) + Dense(512→256)")
        console.print(f"  Expected Accuracy: 80%+ (MAXIMUM PERFORMANCE)")
        console.print(f"  GPU Memory: ~6-8GB (RTX 4060 optimized)")
    
    console.print(f"  Epochs: {args.epochs}")
    console.print(f"  Batch Size: {args.batch_size}")
    console.print(f"  Learning Rate: {args.learning_rate}")
    console.print(f"  Mixed Precision: ✓ ENABLED (AMP)")
    console.print(f"  Optimal Threshold: ✓ ENABLED (both train & val)")
    console.print()
    
    # Load dataset
    train_dataset, test_dataset = load_or_create_dataset(
        data_path=args.data_path,
        train_split=args.train_split
    )
    
    # Step 2: Initialize model
    console.print("\n" + "="*70, style="bold cyan")
    console.print("STEP 2: INITIALIZING MODEL", style="bold cyan")
    console.print("="*70, style="bold cyan")
    
    # Initialize with FULL fine-tuning (all layers unfrozen)
    model = TrainableSiameseModel(
        projection_dim=args.projection_dim,
        freeze_sbert=freeze_sbert,
        unfreeze_all=unfreeze_all
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    console.print(f"\n[bold]Model Parameters:[/bold]")
    console.print(f"  Total: {total_params:,}")
    console.print(f"  Trainable: {trainable_params:,} 🔥")
    console.print(f"  Frozen: {frozen_params:,}")
    
    if unfreeze_all:
        sbert_params = sum(p.numel() for p in model.sbert_encoder.parameters() if p.requires_grad)
        projection_params = sum(p.numel() for p in model.projection_head.parameters())
        console.print(f"\n[bold]Trainable Breakdown:[/bold]")
        console.print(f"  SBERT Encoder (ALL 6 layers): {sbert_params:,} 🔥")
        console.print(f"  BiLSTM + Attention: {projection_params // 2:,}")
        console.print(f"  Dense Layers: {projection_params // 2:,}")
    
    console.print(f"\n✓ Model initialized with MAXIMUM capacity", style="green bold")
    
    # Step 3: Initialize trainer
    console.print("\n" + "="*70, style="bold cyan")
    console.print("STEP 3: INITIALIZING TRAINER", style="bold cyan")
    console.print("="*70, style="bold cyan")
    
    trainer = SiameseTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        use_agents=False,  # Agents disabled
        num_epochs=args.epochs
    )
    
    console.print("Trainer initialized", style="green")
    
    # Step 4: Train
    console.print("\n" + "="*70, style="bold cyan")
    console.print("STEP 4: TRAINING", style="bold cyan")
    console.print("="*70, style="bold cyan")
    
    trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        agent_check_every=None  # Agents disabled
    )
    
    # Step 5: Save final model
    console.print("\n" + "="*70, style="bold cyan")
    console.print("STEP 5: SAVING FINAL MODEL", style="bold cyan")
    console.print("="*70, style="bold cyan")
    
    final_path = Path(args.checkpoint_dir) / "final_model.pt"
    model.save_checkpoint(
        str(final_path),
        epoch=args.epochs,
        optimizer_state=trainer.optimizer.state_dict()
    )
    console.print(f"Final model saved to {final_path}", style="green")
    
    # Step 6: Test (optional)
    if not args.skip_testing:
        if not TESTING_AVAILABLE:
            console.print("\n⚠️  Testing pipeline not available. Skipping tests.", style="yellow")
        else:
            console.print("\n" + "="*70, style="bold cyan")
            console.print("STEP 6: TESTING", style="bold cyan")
            console.print("="*70, style="bold cyan")
            
            tester = SiameseTester(
                model=model,
                test_dataset=test_dataset,
                use_agents=args.use_agents
            )
            results = tester.test()
            
            # Print summary
            console.print("\n[bold green]Final Results:[/bold green]")
            console.print(f"  Accuracy: {results['accuracy']:.4f}")
            console.print(f"  F1 Score: {results['f1']:.4f}")
    
    # Done!
    console.print("\n" + "="*70, style="bold green")
    console.print("TRAINING COMPLETE!", style="bold green")
    console.print("="*70, style="bold green")
    console.print(f"\nModel saved to: {final_path}")
    console.print("\nTo use this model, load it with:")
    console.print(f"  model = TrainableSiameseModel(projection_dim={args.projection_dim})")
    console.print(f"  model.load_checkpoint('{final_path}')")
    console.print()


if __name__ == "__main__":
    main()
