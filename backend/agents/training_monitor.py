"""
Training Monitor Agent
======================

Monitors training progress and suggests improvements.

Triggers:
- After each epoch
- When loss/accuracy stagnates
- When NaN/Inf is detected
- When overfitting is detected
"""

from typing import Dict, Any, List, Optional
from .agent_config import AgentConfig


class TrainingMonitorAgent:
    """Agent for monitoring training progress."""
    
    def __init__(self, config: AgentConfig):
        """
        Initialize training monitor agent.
        
        Args:
            config: Agent configuration with LLM client
        """
        self.config = config
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.patience_counter = 0
        self.best_val_loss = float('inf')
    
    def monitor_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float
    ) -> Dict[str, Any]:
        """
        Monitor training after an epoch and provide suggestions.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
        
        Returns:
            Monitoring results with suggestions
        """
        # Update history
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_acc"].append(val_acc)
        
        results = {
            "status": "normal",
            "warnings": [],
            "suggestions": [],
            "should_stop": False
        }
        
        # Check for NaN/Inf
        if self._check_nan_inf(train_loss, val_loss):
            results["status"] = "critical"
            results["warnings"].append("NaN/Inf detected in loss")
            results["suggestions"].extend([
                "Reduce learning rate by 10x",
                "Check for gradient explosion",
                "Verify data preprocessing"
            ])
            results["should_stop"] = True
            
            self.config.log_intervention(
                "TrainingMonitor",
                f"CRITICAL: NaN/Inf at epoch {epoch}"
            )
            return results
        
        # Check for stagnation
        if self._check_stagnation():
            results["warnings"].append("Training stagnating")
            results["suggestions"].extend([
                "Reduce learning rate by 2x",
                "Try data augmentation",
                "Check if model has converged"
            ])
            
            self.config.log_intervention(
                "TrainingMonitor",
                f"WARNING: Training stagnating at epoch {epoch}"
            )
        
        # Check for overfitting
        if self._check_overfitting(train_loss, val_loss, train_acc, val_acc):
            results["warnings"].append("Overfitting detected")
            results["suggestions"].extend([
                "Increase dropout rate",
                "Add L2 regularization",
                "Reduce model complexity",
                "Use early stopping"
            ])
            
            self.config.log_intervention(
                "TrainingMonitor",
                f"WARNING: Overfitting at epoch {epoch} (train_acc={train_acc:.3f}, val_acc={val_acc:.3f})"
            )
        
        # Check for underfitting
        if self._check_underfitting(train_acc):
            results["warnings"].append("Underfitting detected")
            results["suggestions"].extend([
                "Increase model capacity",
                "Train for more epochs",
                "Increase learning rate",
                "Check data quality"
            ])
            
            self.config.log_intervention(
                "TrainingMonitor",
                f"WARNING: Underfitting at epoch {epoch} (train_acc={train_acc:.3f})"
            )
        
        # Update patience for early stopping
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
            if self.patience_counter >= 5:
                results["suggestions"].append("Consider early stopping (no improvement for 5 epochs)")
        
        # Ask LLM for advanced suggestions if issues detected
        if results["warnings"] and epoch > 3:
            llm_suggestion = self._get_llm_suggestions(epoch, results["warnings"])
            if llm_suggestion:
                results["suggestions"].append(f"LLM Insight: {llm_suggestion}")
        
        return results
    
    def _check_nan_inf(self, train_loss: float, val_loss: float) -> bool:
        """Check if loss is NaN or Inf."""
        import math
        return (
            math.isnan(train_loss) or math.isinf(train_loss) or
            math.isnan(val_loss) or math.isinf(val_loss)
        )
    
    def _check_stagnation(self, window: int = 3) -> bool:
        """Check if validation loss is stagnating."""
        if len(self.history["val_loss"]) < window + 1:
            return False
        
        recent_losses = self.history["val_loss"][-window:]
        
        # Check if change is less than 0.1% over the window
        if max(recent_losses) - min(recent_losses) < 0.001:
            return True
        
        return False
    
    def _check_overfitting(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float
    ) -> bool:
        """Check if model is overfitting."""
        # Overfitting indicators:
        # 1. Train accuracy much higher than val accuracy
        # 2. Val loss increasing while train loss decreasing
        
        if train_acc - val_acc > 0.1:  # 10% gap
            return True
        
        if len(self.history["val_loss"]) > 2:
            if (val_loss > self.history["val_loss"][-2] and
                train_loss < self.history["train_loss"][-2]):
                return True
        
        return False
    
    def _check_underfitting(self, train_acc: float) -> bool:
        """Check if model is underfitting."""
        # Underfitting: poor performance on training set
        if train_acc < 0.6:  # Less than 60% train accuracy
            return True
        
        return False
    
    def _get_llm_suggestions(self, epoch: int, warnings: List[str]) -> Optional[str]:
        """Get advanced suggestions from LLM."""
        prompt = f"""You are a deep learning training expert. Analyze this training situation:

Current Epoch: {epoch}
Recent Training History (last 5 epochs):
- Train Loss: {self.history['train_loss'][-5:]}
- Val Loss: {self.history['val_loss'][-5:]}
- Train Acc: {self.history['train_acc'][-5:]}
- Val Acc: {self.history['val_acc'][-5:]}

Current Issues:
{chr(10).join(f'- {w}' for w in warnings)}

Provide ONE specific, actionable suggestion to improve training.
Keep it brief (max 50 words).
"""
        
        try:
            response = self.config.generate_response(prompt, temperature=0.7)
            # Extract first sentence or up to 200 chars
            suggestion = response.split('.')[0][:200].strip()
            return suggestion
        
        except Exception as e:
            self.config.log_intervention("TrainingMonitor", f"LLM suggestion failed: {str(e)}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training progress."""
        if not self.history["val_loss"]:
            return {"status": "no_data"}
        
        return {
            "total_epochs": len(self.history["train_loss"]),
            "best_val_loss": min(self.history["val_loss"]),
            "best_val_acc": max(self.history["val_acc"]),
            "final_train_loss": self.history["train_loss"][-1],
            "final_val_loss": self.history["val_loss"][-1],
            "final_train_acc": self.history["train_acc"][-1],
            "final_val_acc": self.history["val_acc"][-1],
            "patience_counter": self.patience_counter
        }
