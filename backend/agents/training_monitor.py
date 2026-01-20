"""
Training Monitor Agent
======================

AI agent for monitoring and analyzing training progress.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from .agent_config import AgentConfig


class TrainingMonitorAgent:
    """Agent for monitoring training progress and detecting issues"""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize training monitor agent
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'timestamps': []
        }
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.warnings = []
        
    def record_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        learning_rate: float
    ):
        """
        Record metrics for an epoch
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            learning_rate: Current learning rate
        """
        self.training_history['epochs'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_acc'].append(train_acc)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_acc'].append(val_acc)
        self.training_history['learning_rates'].append(learning_rate)
        self.training_history['timestamps'].append(datetime.now().isoformat())
        
        # Check for improvement
        if val_acc > self.best_val_acc + self.config.min_improvement:
            self.best_val_acc = val_acc
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
    
    def should_stop_early(self) -> Tuple[bool, str]:
        """
        Check if training should stop early
        
        Returns:
            Tuple of (should_stop, reason)
        """
        if self.epochs_without_improvement >= self.config.early_stopping_patience:
            return True, (
                f"No improvement for {self.epochs_without_improvement} epochs "
                f"(patience: {self.config.early_stopping_patience})"
            )
        
        # Check for divergence
        if len(self.training_history['train_loss']) >= 3:
            recent_losses = self.training_history['train_loss'][-3:]
            if all(recent_losses[i] > recent_losses[i-1] for i in range(1, len(recent_losses))):
                return True, "Training loss diverging (increasing for 3 consecutive epochs)"
        
        # Check for overfitting
        if len(self.training_history['train_acc']) >= 2 and len(self.training_history['val_acc']) >= 2:
            train_acc = self.training_history['train_acc'][-1]
            val_acc = self.training_history['val_acc'][-1]
            if train_acc - val_acc > 0.2:  # 20% gap
                self.warnings.append(
                    f"Severe overfitting detected: train_acc ({train_acc:.2%}) - "
                    f"val_acc ({val_acc:.2%}) > 20%"
                )
        
        return False, ""
    
    def analyze_training(self) -> Dict[str, Any]:
        """
        Analyze training progress
        
        Returns:
            Analysis dictionary with metrics and insights
        """
        if not self.training_history['epochs']:
            return {'error': 'No training history available'}
        
        analysis = {
            'total_epochs': len(self.training_history['epochs']),
            'best_val_acc': self.best_val_acc,
            'current_val_acc': self.training_history['val_acc'][-1],
            'best_epoch': self.training_history['val_acc'].index(max(self.training_history['val_acc'])) + 1,
            'epochs_without_improvement': self.epochs_without_improvement,
            'current_learning_rate': self.training_history['learning_rates'][-1],
            'warnings': self.warnings.copy()
        }
        
        # Calculate trends
        if len(self.training_history['val_loss']) >= 5:
            recent_val_losses = self.training_history['val_loss'][-5:]
            analysis['val_loss_trend'] = 'improving' if recent_val_losses[-1] < recent_val_losses[0] else 'degrading'
        
        # Check for plateau
        if len(self.training_history['val_acc']) >= 10:
            recent_val_accs = self.training_history['val_acc'][-10:]
            variance = np.var(recent_val_accs)
            if variance < 0.0001:  # Very small variance
                analysis['warnings'].append(
                    "Validation accuracy has plateaued (low variance in last 10 epochs)"
                )
        
        # Calculate improvement rate
        if len(self.training_history['val_acc']) >= 2:
            first_val_acc = self.training_history['val_acc'][0]
            current_val_acc = self.training_history['val_acc'][-1]
            improvement = (current_val_acc - first_val_acc) / first_val_acc
            analysis['total_improvement'] = improvement
            
            if improvement < 0:
                analysis['warnings'].append(
                    f"Negative improvement: validation accuracy decreased by {abs(improvement):.2%}"
                )
        
        # Check overfitting
        if self.training_history['train_acc'] and self.training_history['val_acc']:
            train_val_gap = (
                self.training_history['train_acc'][-1] - 
                self.training_history['val_acc'][-1]
            )
            analysis['train_val_gap'] = train_val_gap
            
            if train_val_gap > 0.15:
                analysis['warnings'].append(
                    f"Overfitting detected: train-val gap = {train_val_gap:.2%}"
                )
        
        return analysis
    
    def generate_report(self) -> str:
        """
        Generate training progress report
        
        Returns:
            Formatted report string
        """
        analysis = self.analyze_training()
        
        if 'error' in analysis:
            return f"Error: {analysis['error']}"
        
        report = []
        report.append("=" * 80)
        report.append("TRAINING MONITOR REPORT")
        report.append("=" * 80)
        
        report.append(f"\nTraining Progress:")
        report.append(f"  Total Epochs: {analysis['total_epochs']}")
        report.append(f"  Best Validation Accuracy: {analysis['best_val_acc']:.4f} (Epoch {analysis['best_epoch']})")
        report.append(f"  Current Validation Accuracy: {analysis['current_val_acc']:.4f}")
        report.append(f"  Epochs Without Improvement: {analysis['epochs_without_improvement']}")
        
        if 'total_improvement' in analysis:
            report.append(f"  Total Improvement: {analysis['total_improvement']:.2%}")
        
        if 'train_val_gap' in analysis:
            report.append(f"  Train-Val Gap: {analysis['train_val_gap']:.2%}")
        
        report.append(f"\nCurrent Learning Rate: {analysis['current_learning_rate']:.2e}")
        
        if 'val_loss_trend' in analysis:
            report.append(f"Validation Loss Trend: {analysis['val_loss_trend']}")
        
        if analysis['warnings']:
            report.append("\n⚠️  Warnings:")
            for warning in analysis['warnings']:
                report.append(f"  • {warning}")
        
        # Check if early stopping should trigger
        should_stop, reason = self.should_stop_early()
        if should_stop:
            report.append(f"\n🛑 Early Stopping Recommended: {reason}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics
        
        Returns:
            Dictionary with metric summaries
        """
        if not self.training_history['epochs']:
            return {}
        
        return {
            'epochs': self.training_history['epochs'][-1],
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_train_acc': self.training_history['train_acc'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'final_val_acc': self.training_history['val_acc'][-1],
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.training_history['val_acc'].index(max(self.training_history['val_acc'])) + 1
        }
    
    def clear_history(self):
        """Clear training history"""
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'timestamps': []
        }
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.warnings = []
