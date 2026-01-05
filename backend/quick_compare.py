"""
Quick Compare Script - Compare Two Documents from datasets/ Folder

This script:
1. Loads two documents (txt or pdf) from datasets/ folder
2. Uses the TRAINED model (if available) or untrained model
3. Calculates similarity score
4. Optionally uses AI agents for verdict

NO RETRAINING - Just inference using existing trained model.

Usage:
    python backend/quick_compare.py
    python backend/quick_compare.py --use-agents
    python backend/quick_compare.py --folder datasets/
"""

import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from document_loader import load_two_documents_from_folder
from neural_engine import TrainableSiameseModel, SiameseProjectionModel

console = Console()

# Model path
MODEL_PATH = Path("models/siamese_trained.pt")


def load_model(use_trained: bool = True):
    """Load trained model if available, otherwise use untrained."""
    if use_trained and MODEL_PATH.exists():
        console.print(f"\n[green]✓[/green] Loading trained model from {MODEL_PATH}")
        model = TrainableSiameseModel()
        model.load_model(str(MODEL_PATH))
        model.eval()
        return model, True
    else:
        if use_trained:
            console.print(f"\n[yellow]⚠[/yellow] No trained model found at {MODEL_PATH}")
            console.print("[yellow]⚠[/yellow] Using untrained projection head")
            console.print("[yellow]⚠[/yellow] Train first: python backend/train.py --data-path datasets/train.csv")
        model = SiameseProjectionModel()
        return model, False


def compare_documents(
    doc_a: str,
    doc_b: str,
    model,
    use_agents: bool = False,
    provider: str = "groq"
):
    """Compare two documents."""
    console.print("\n" + "=" * 70)
    console.print("[bold cyan]PARAPHRASE DETECTION - SIMILARITY ANALYSIS[/bold cyan]")
    console.print("=" * 70)
    
    # Phase 1: Neural Similarity
    console.print("\n[bold]Phase 1: Neural Network Analysis[/bold]")
    
    similarity_data = model.calculate_similarity(doc_a, doc_b)
    similarity_score = similarity_data['cosine_similarity']
    
    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Cosine Similarity", f"{similarity_score:.4f}")
    table.add_row("Similarity %", f"{similarity_data['similarity_percentage']:.2f}%")
    table.add_row("Doc A Length", f"{len(doc_a)} chars")
    table.add_row("Doc B Length", f"{len(doc_b)} chars")
    table.add_row("SBERT Model", similarity_data['model'])
    table.add_row("Projection Dim", str(similarity_data['projection_dim']))
    
    console.print(table)
    
    # Interpretation
    if similarity_score > 0.85:
        interpretation = "[bold red]LIKELY DUPLICATE[/bold red]"
    elif similarity_score > 0.70:
        interpretation = "[bold yellow]POSSIBLE PARAPHRASE[/bold yellow]"
    else:
        interpretation = "[bold green]LIKELY UNIQUE[/bold green]"
    
    console.print(f"\n[bold]Initial Verdict:[/bold] {interpretation}")
    
    # Phase 2: AI Agents (optional)
    if use_agents:
        console.print("\n[bold]Phase 2: Multi-Agent Analysis[/bold]")
        try:
            from agent_crew import create_crew
            
            crew = create_crew(provider)
            result = crew.analyze(doc_a, doc_b, similarity_score)
            
            console.print(f"\n[bold]Final Verdict:[/bold] [bold cyan]{result['verdict']}[/bold cyan]")
            console.print(f"[bold]Confidence:[/bold] {result['confidence']}")
            console.print(f"[bold]Model:[/bold] {result['model']}")
            
            console.print("\n[bold]Full Analysis:[/bold]")
            console.print(Panel(result['full_analysis'], title="AI Agent Analysis", border_style="cyan"))
            
        except Exception as e:
            console.print(f"\n[red]✗[/red] Agent analysis failed: {e}")
            console.print("[yellow]Continuing with Phase 1 results only[/yellow]")
    
    console.print("\n" + "=" * 70)
    return similarity_score


def main():
    parser = argparse.ArgumentParser(
        description="Compare two documents from datasets/ folder"
    )
    parser.add_argument(
        '--folder',
        type=str,
        default='datasets',
        help='Folder containing the two documents (default: datasets/)'
    )
    parser.add_argument(
        '--use-agents',
        action='store_true',
        help='Enable AI agent analysis (requires GROQ_API_KEY)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='groq',
        choices=['groq', 'ollama'],
        help='LLM provider for agents (default: groq)'
    )
    parser.add_argument(
        '--no-trained-model',
        action='store_true',
        help='Use untrained model instead of trained one'
    )
    
    args = parser.parse_args()
    
    # Load documents
    console.print(f"\n[bold]Loading documents from:[/bold] {args.folder}")
    
    try:
        doc_a, doc_b, path_a, path_b = load_two_documents_from_folder(args.folder)
        
        console.print(f"\n[green]✓[/green] Document A: {Path(path_a).name}")
        console.print(f"   Length: {len(doc_a)} characters, {len(doc_a.split())} words")
        console.print(f"   Preview: {doc_a[:100]}...")
        
        console.print(f"\n[green]✓[/green] Document B: {Path(path_b).name}")
        console.print(f"   Length: {len(doc_b)} characters, {len(doc_b.split())} words")
        console.print(f"   Preview: {doc_b[:100]}...")
        
    except Exception as e:
        console.print(f"\n[red]✗[/red] Error loading documents: {e}")
        return
    
    # Load model
    model, is_trained = load_model(use_trained=not args.no_trained_model)
    
    if is_trained:
        console.print("[green]✓[/green] Using trained model")
    else:
        console.print("[yellow]⚠[/yellow] Using untrained model - results may be unreliable")
    
    # Compare
    compare_documents(
        doc_a,
        doc_b,
        model,
        use_agents=args.use_agents,
        provider=args.provider
    )


if __name__ == "__main__":
    main()
