"""
Agent Configuration for CrewAI Integration
===========================================

Centralized configuration for all AI agents used in the paraphrase detection system.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for CrewAI agents"""
    
    # Model configuration
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_tokens: int = 2000
    
    # Agent behavior
    verbose: bool = True
    allow_delegation: bool = False
    max_iterations: int = 5
    
    # Performance thresholds
    min_confidence: float = 0.7
    similarity_threshold: float = 0.5
    
    # Data validation
    min_text_length: int = 10
    max_text_length: int = 10000
    min_training_samples: int = 100
    
    # Training monitoring
    checkpoint_frequency: int = 5
    early_stopping_patience: int = 3
    min_improvement: float = 0.001
    
    # Inference validation
    response_time_threshold_ms: float = 1000.0
    batch_size_limit: int = 100
    
    # Additional settings
    extra_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "verbose": self.verbose,
            "allow_delegation": self.allow_delegation,
            "max_iterations": self.max_iterations,
            "min_confidence": self.min_confidence,
            "similarity_threshold": self.similarity_threshold,
            "min_text_length": self.min_text_length,
            "max_text_length": self.max_text_length,
            "min_training_samples": self.min_training_samples,
            "checkpoint_frequency": self.checkpoint_frequency,
            "early_stopping_patience": self.early_stopping_patience,
            "min_improvement": self.min_improvement,
            "response_time_threshold_ms": self.response_time_threshold_ms,
            "batch_size_limit": self.batch_size_limit,
            **self.extra_settings
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AgentConfig":
        """Create configuration from dictionary"""
        known_fields = {
            "model_name", "temperature", "max_tokens", "verbose",
            "allow_delegation", "max_iterations", "min_confidence",
            "similarity_threshold", "min_text_length", "max_text_length",
            "min_training_samples", "checkpoint_frequency",
            "early_stopping_patience", "min_improvement",
            "response_time_threshold_ms", "batch_size_limit"
        }
        
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        extra_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        
        return cls(**known_params, extra_settings=extra_params)


# Default configuration instance
default_config = AgentConfig()
