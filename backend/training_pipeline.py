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
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Our modules
from neural_engine import TrainableSiameseModel, ContrastiveLoss
from document_processor import DocumentProcessor
from dotenv import load_dotenv

# CrewAI for agents (optional)
try:
    from crewai import Agent, Task, Crew, Process, LLM
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# LLM (optional)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

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
# AI Agents for Training (Optional - Only if CrewAI available)
# ============================================================

if CREWAI_AVAILABLE:
    class TrainingAgents:
        """
        AI Agents that supervise the training process.
        
        Agents:
        1. Data Validator: Checks data quality before training
        2. Training Monitor: Analyzes training progress and suggests improvements
        """
        
        def __init__(self, provider: str = "groq"):
            """Initialize training agents."""
            self.provider = provider
            self.llm = self._initialize_llm()
            
            # Create agents
            self.data_validator = self._create_data_validator()
            self.training_monitor = self._create_training_monitor()
        
        def _initialize_llm(self):
            """Initialize LLM for agents."""
            if self.provider == "groq":
                if not GROQ_AVAILABLE:
                    raise ImportError("langchain-groq not installed")
                
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in environment")
                
                return LLM(
                    model="groq/llama3-8b-8192",
                    api_key=api_key,
                    temperature=0.2,
                    max_tokens=2048
                )
            else:
                raise ValueError(f"Provider {self.provider} not supported yet")
        
        def _create_data_validator(self) -> Agent:
            """Create the Data Validator agent."""
            return Agent(
                role="Training Data Validator",
                goal=(
                    "Analyze training data for quality issues, imbalances, and potential "
                    "problems that could affect model training. Provide actionable recommendations."
                ),
                backstory=(
                    "You are a machine learning data scientist with expertise in NLP and "
                    "dataset curation. You have trained hundreds of text similarity models "
                    "and know all the common pitfalls: class imbalance, data leakage, "
                    "insufficient diversity, and annotation errors. You provide clear, "
                    "actionable feedback to improve training data quality."
                ),
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            )
        
        def _create_training_monitor(self) -> Agent:
            """Create the Training Monitor agent."""
            return Agent(
                role="Training Progress Monitor",
                goal=(
                    "Monitor training metrics (loss, accuracy) and identify issues like "
                    "overfitting, underfitting, or training instability. Suggest hyperparameter "
                    "adjustments and training strategies."
                ),
                backstory=(
                    "You are a senior ML engineer specializing in deep learning optimization. "
                    "You can diagnose training problems by looking at loss curves and metrics. "
                    "You know when to adjust learning rates, when to stop early, and how to "
                    "prevent overfitting. Your advice has helped many teams achieve state-of-the-art "
                    "results."
                ),
                verbose=True,
                allow_delegation=False,
                llm=self.llm
            )
        
        def validate_data(self, dataset_stats: Dict[str, Any]) -> str:
            """
            Run data validation agent.
            
            Args:
                dataset_stats: Statistics about the training dataset
                
            Returns:
                Validation report from agent
            """
            task = Task(
                description=f"""
Analyze this training dataset for a paraphrase detection model:

## DATASET STATISTICS:
- Total examples: {dataset_stats['total_examples']}
- Paraphrase/Duplicate examples (label=1): {dataset_stats['positive_examples']}
- Unique examples (label=0): {dataset_stats['negative_examples']}
- Class balance ratio: {dataset_stats['positive_ratio']:.2%}
- Average text length (A): {dataset_stats['avg_length_a']:.0f} chars
- Average text length (B): {dataset_stats['avg_length_b']:.0f} chars
- Text length std dev (A): {dataset_stats['std_length_a']:.0f} chars
- Text length std dev (B): {dataset_stats['std_length_b']:.0f} chars

## YOUR TASK:
1. **Assess Class Balance**: Is the dataset balanced? Should we use class weights?
2. **Check Data Diversity**: Are text lengths varied enough?
3. **Identify Risks**: What could go wrong with this dataset?
4. **Provide Recommendations**: Concrete steps to improve data quality

## OUTPUT FORMAT:
**OVERALL ASSESSMENT:** [GOOD / ACCEPTABLE / POOR]

**CLASS BALANCE:**
[Your analysis]

**DATA DIVERSITY:**
[Your analysis]

**IDENTIFIED RISKS:**
- [Risk 1]
- [Risk 2]
...

**RECOMMENDATIONS:**
1. [Recommendation 1]
2. [Recommendation 2]
...

**SUGGESTED HYPERPARAMETERS:**
- Batch size: [value]
- Learning rate: [value]
- Class weights: [if needed]
""",
                expected_output=(
                    "Detailed data validation report with assessment, risks, and recommendations"
                ),
                agent=self.data_validator
            )
            
            crew = Crew(
                agents=[self.data_validator],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            return str(result)
        
        def monitor_training(self, training_stats: Dict[str, Any]) -> str:
            """
            Run training monitor agent.
            
            Args:
                training_stats: Training metrics over epochs
                
            Returns:
                Monitoring report from agent
            """
            task = Task(
                description=f"""
Analyze the training progress of a Siamese Network for paraphrase detection:

## TRAINING METRICS:
- Current Epoch: {training_stats['current_epoch']}/{training_stats['total_epochs']}
- Training Loss History: {training_stats['train_loss_history']}
- Validation Loss History: {training_stats.get('val_loss_history', 'N/A')}
- Training Accuracy: {training_stats.get('train_accuracy', 'N/A')}
- Validation Accuracy: {training_stats.get('val_accuracy', 'N/A')}
- Learning Rate: {training_stats['learning_rate']}
- Batch Size: {training_stats['batch_size']}

## YOUR TASK:
1. **Diagnose Training State**: Is the model learning well? Any issues?
2. **Check for Overfitting/Underfitting**: Compare train vs val metrics
3. **Assess Convergence**: Is training loss decreasing? Plateauing?
4. **Provide Recommendations**: Should we continue? Adjust hyperparameters?

## OUTPUT FORMAT:
**TRAINING STATUS:** [HEALTHY / WARNING / CRITICAL]

**DIAGNOSIS:**
[Your analysis of the training state]

**OVERFITTING CHECK:**
[Analysis of train vs validation gap]

**CONVERGENCE ANALYSIS:**
[Is the model converging? How much longer to train?]

**RECOMMENDATIONS:**
1. [Action 1]
2. [Action 2]
...

**SUGGESTED ADJUSTMENTS:**
- Learning rate: [keep / increase / decrease / schedule]
- Continue training: [yes / no / X more epochs]
- Early stopping: [recommended / not needed]
""",
                expected_output=(
                    "Training progress analysis with diagnosis and recommendations"
                ),
                agent=self.training_monitor
            )
            
            crew = Crew(
                agents=[self.training_monitor],
                tasks=[task],
                process=Process.sequential,
                verbose=False
            )
            
            result = crew.kickoff()
            return str(result)

else:
    # Dummy class when CrewAI not available
    class TrainingAgents:
        """Dummy class - CrewAI not available."""
        def __init__(self, provider: str = "groq"):
            raise ImportError("CrewAI not available. Install with: pip install crewai")


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
        use_agents: bool = True,
        checkpoint_dir: str = "./checkpoints"
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
            use_agents: Whether to use AI agents for supervision
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_agents = use_agents
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training components
        self.criterion = ContrastiveLoss(margin=margin)
        
        if model.unfreeze_last_sbert_layer:
            param_groups = model.get_parameter_groups(lr_sbert=1e-5, lr_head=learning_rate)
            self.optimizer = optim.AdamW(param_groups, weight_decay=0.01)
        else:
            self.optimizer = optim.AdamW(
                model.get_trainable_parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=2,
            factor=0.5
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
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # AI Agents
        if use_agents:
            if not CREWAI_AVAILABLE:
                print("⚠️  CrewAI not available. Training without AI agents.")
                self.agents = None
            else:
                try:
                    self.agents = TrainingAgents()
                    print("✓ AI Training Agents initialized")
                except Exception as e:
                    print(f"⚠️  Could not initialize agents: {e}")
                    print("Continuing training without AI agents.")
                    self.agents = None
        else:
            self.agents = None
    
    def compute_dataset_stats(self) -> Dict[str, Any]:
        """Compute statistics about the training dataset."""
        # Handle both Dataset and Subset
        if hasattr(self.train_dataset, 'examples'):
            examples = self.train_dataset.examples
        elif hasattr(self.train_dataset, 'dataset'):
            examples = self.train_dataset.dataset.examples
        else:
            raise ValueError("Cannot access examples from dataset")
        
        labels = [ex['label'] for ex in examples]
        lengths_a = [len(str(ex['text_a'])) for ex in examples]
        lengths_b = [len(str(ex['text_b'])) for ex in examples]
        
        positive_count = sum(labels)
        total_count = len(labels)
        
        return {
            'total_examples': total_count,
            'positive_examples': positive_count,
            'negative_examples': total_count - positive_count,
            'positive_ratio': positive_count / total_count if total_count > 0 else 0,
            'avg_length_a': np.mean(lengths_a),
            'avg_length_b': np.mean(lengths_b),
            'std_length_a': np.std(lengths_a),
            'std_length_b': np.std(lengths_b)
        }
    
    def validate_data_with_agent(self):
        """Run data validation agent before training."""
        if not self.agents:
            return
        
        print("\n" + "-"*50)
        print("AGENT: DATA VALIDATION")
        print("-"*50)
        
        stats = self.compute_dataset_stats()
        report = self.agents.validate_data(stats)
        
        print("\nData Validation Report:")
        print(report)
        print("-"*50)
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        gradient_verified = False
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            texts_a = batch['text_a']
            texts_b = batch['text_b']
            labels = batch['label'].float().to(self.model.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings_a, embeddings_b = self.model.forward_batch(texts_a, texts_b)
            
            # Compute loss
            loss = self.criterion(embeddings_a, embeddings_b, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Gradient verification (first batch of first epoch only)
            if epoch == 1 and not gradient_verified:
                grad_norms = [
                    param.grad.norm().item() 
                    for param in self.model.projection_head.parameters() 
                    if param.grad is not None
                ]
                assert all(g > 0 for g in grad_norms), "Zero gradients in projection head!"
                gradient_verified = True
            
            self.optimizer.step()
            
            # Compute accuracy
            similarities = self.model.compute_similarity_batch(embeddings_a, embeddings_b)
            predictions = (similarities > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Returns:
            (average_loss, accuracy)
        """
        if not self.val_dataset:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                texts_a = batch['text_a']
                texts_b = batch['text_b']
                labels = batch['label'].float().to(self.model.device)
                
                # Forward pass
                embeddings_a, embeddings_b = self.model.forward_batch(texts_a, texts_b)
                
                # Compute loss
                loss = self.criterion(embeddings_a, embeddings_b, labels)
                
                # Compute accuracy
                similarities = self.model.compute_similarity_batch(embeddings_a, embeddings_b)
                predictions = (similarities > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def monitor_with_agent(self, epoch: int):
        """Run training monitor agent."""
        if not self.agents or epoch < 2:  # Need at least 2 epochs of data
            return
        
        print("\n" + "-"*50)
        print(f"AGENT: TRAINING MONITORING (Epoch {epoch})")
        print("-"*50)
        
        training_stats = {
            'current_epoch': epoch,
            'total_epochs': epoch,  # Unknown total
            'train_loss_history': self.history['train_loss'][-5:],  # Last 5 epochs
            'val_loss_history': self.history['val_loss'][-5:] if self.history['val_loss'] else None,
            'train_accuracy': self.history['train_acc'][-1] if self.history['train_acc'] else None,
            'val_accuracy': self.history['val_acc'][-1] if self.history['val_acc'] else None,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'batch_size': self.batch_size
        }
        
        report = self.agents.monitor_training(training_stats)
        
        print("\nTraining Monitor Report:")
        print(report)
        print("-"*50)
    
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
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
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
            
            # Agent monitoring
            if self.agents and agent_check_every and epoch % agent_check_every == 0:
                self.monitor_with_agent(epoch)
        
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
