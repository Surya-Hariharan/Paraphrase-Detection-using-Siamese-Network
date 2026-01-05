"""
Train Neural Network on Two Documents
======================================

This script implements the EXACT workflow you described:
1. Take Document A and Document B
2. Generate embeddings using SBERT
3. Train neural network on those embeddings
4. Compute cosine similarity

Usage:
    python backend/train_on_documents.py
    python backend/train_on_documents.py --doc-a datasets/c1.txt --doc-b datasets/c2.txt
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from document_loader import load_document
from neural_engine import TrainableSiameseModel, ContrastiveLoss

console = Console()


def train_on_two_documents(
    doc_a_path: str,
    doc_b_path: str,
    epochs: int = 50,
    learning_rate: float = 0.001,
    label: int = 1  # 1=paraphrase, 0=different
):
    """
    Train neural network on two specific documents.
    
    Args:
        doc_a_path: Path to document A
        doc_b_path: Path to document B
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        label: 1 if paraphrase/similar, 0 if different
    """
    console.print("\n" + "="*70)
    console.print("[bold cyan]TRAINING NEURAL NETWORK ON TWO DOCUMENTS[/bold cyan]")
    console.print("="*70)
    
    # Step 1: Load documents
    console.print("\n[bold]Step 1: Loading Documents[/bold]")
    doc_a = load_document(doc_a_path)
    doc_b = load_document(doc_b_path)
    
    console.print(f"  Document A: {doc_a_path} ({len(doc_a)} chars)")
    console.print(f"  Document B: {doc_b_path} ({len(doc_b)} chars)")
    console.print(f"  Label: {'Paraphrase/Similar' if label == 1 else 'Different'}")
    
    # Step 2: Initialize model
    console.print("\n[bold]Step 2: Initializing Siamese Neural Network[/bold]")
    model = TrainableSiameseModel(projection_dim=256, freeze_sbert=True)
    
    console.print(f"  SBERT Encoder: {model.SBERT_MODEL}")
    console.print(f"  SBERT Dimension: {model.SBERT_DIM}")
    console.print(f"  Projection Dimension: {model.projection_dim}")
    console.print(f"  Trainable Parameters: {sum(p.numel() for p in model.get_trainable_parameters()):,}")
    
    # Step 3: Generate initial embeddings (before training)
    console.print("\n[bold]Step 3: Generating Embeddings with SBERT[/bold]")
    
    with torch.no_grad():
        # SBERT embeddings
        emb_a_sbert = model.encode_with_sbert(doc_a)
        emb_b_sbert = model.encode_with_sbert(doc_b)
        
        # Initial projection (untrained)
        emb_a_proj_init = model.project_embedding(emb_a_sbert)
        emb_b_proj_init = model.project_embedding(emb_b_sbert)
        
        # Initial similarity
        import torch.nn.functional as F
        similarity_init = F.cosine_similarity(
            emb_a_proj_init.unsqueeze(0),
            emb_b_proj_init.unsqueeze(0)
        ).item()
    
    console.print(f"  Embedding A (SBERT): {emb_a_sbert.shape}")
    console.print(f"  Embedding B (SBERT): {emb_b_sbert.shape}")
    console.print(f"  Initial Similarity (untrained): {similarity_init:.4f}")
    
    # Step 4: Train neural network
    console.print("\n[bold]Step 4: Training Neural Network[/bold]")
    
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(margin=1.0)
    
    label_tensor = torch.tensor([label], dtype=torch.float32)
    
    console.print(f"  Optimizer: Adam (lr={learning_rate})")
    console.print(f"  Loss Function: Contrastive Loss")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Training on: 1 document pair\n")
    
    # Training loop
    model.train()
    
    training_table = Table(show_header=True, header_style="bold magenta")
    training_table.add_column("Epoch", style="cyan", width=10)
    training_table.add_column("Loss", style="yellow", width=15)
    training_table.add_column("Similarity", style="green", width=15)
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # Forward pass through Siamese network
        proj_a, proj_b = model.forward(doc_a, doc_b)
        
        # Compute loss
        loss = criterion(
            proj_a.unsqueeze(0),
            proj_b.unsqueeze(0),
            label_tensor
        )
        
        # Backward pass
        loss.backward()
        
        # GRADIENT VERIFICATION: Ensure projection head receives gradients
        if epoch == 1:
            grad_norms = {
                name: param.grad.norm().item() 
                for name, param in model.projection_head.named_parameters() 
                if param.grad is not None
            }
            assert all(g > 0 for g in grad_norms.values()), "Zero gradients detected!"
            console.print(f"  ✓ Gradient flow verified: {grad_norms}", style="dim green")
        
        optimizer.step()
        
        # Calculate current similarity
        with torch.no_grad():
            similarity = F.cosine_similarity(
                proj_a.unsqueeze(0),
                proj_b.unsqueeze(0)
            ).item()
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            training_table.add_row(
                f"{epoch}/{epochs}",
                f"{loss.item():.6f}",
                f"{similarity:.4f}"
            )
    
    console.print(training_table)
    
    # Step 5: Final evaluation
    console.print("\n[bold]Step 5: Final Evaluation[/bold]")
    
    model.eval()
    with torch.no_grad():
        result = model.calculate_similarity(doc_a, doc_b)
    
    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Before Training", style="yellow")
    results_table.add_column("After Training", style="green")
    
    results_table.add_row(
        "Cosine Similarity",
        f"{similarity_init:.4f}",
        f"{result['cosine_similarity']:.4f}"
    )
    results_table.add_row(
        "Similarity %",
        f"{(similarity_init + 1) / 2 * 100:.2f}%",
        f"{result['similarity_percentage']:.2f}%"
    )
    
    console.print(results_table)
    
    # Step 6: Save trained model
    console.print("\n[bold]Step 6: Saving Trained Model[/bold]")
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "siamese_trained.pt"
    
    model.save_model(str(model_path))
    console.print(f"  ✓ Model saved to: {model_path}")
    
    # Summary
    console.print("\n" + "="*70)
    console.print("[bold green]TRAINING COMPLETE![/bold green]")
    console.print("="*70)
    
    console.print("\n[bold]What was trained:[/bold]")
    console.print(f"  • SBERT Encoder: [yellow]Frozen[/yellow] (pre-trained, not changed)")
    console.print(f"  • Projection Head: [green]Trained[/green] (384→256 dimensions)")
    console.print(f"  • Training Data: 1 document pair")
    console.print(f"  • Epochs: {epochs}")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Compare documents: python backend/quick_compare.py")
    console.print("  2. Or use in API: python backend/app.py")
    
    return model, result


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network on two specific documents"
    )
    parser.add_argument(
        '--doc-a',
        type=str,
        default='datasets/c1.txt',
        help='Path to document A (default: datasets/c1.txt)'
    )
    parser.add_argument(
        '--doc-b',
        type=str,
        default='datasets/c2.txt',
        help='Path to document B (default: datasets/c2.txt)'
    )
    parser.add_argument(
        '--label',
        type=int,
        default=1,
        choices=[0, 1],
        help='Label: 1=paraphrase/similar, 0=different (default: 1)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not Path(args.doc_a).exists():
        console.print(f"[red]Error:[/red] Document A not found: {args.doc_a}")
        return
    
    if not Path(args.doc_b).exists():
        console.print(f"[red]Error:[/red] Document B not found: {args.doc_b}")
        return
    
    # Train
    train_on_two_documents(
        doc_a_path=args.doc_a,
        doc_b_path=args.doc_b,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        label=args.label
    )


if __name__ == "__main__":
    main()
