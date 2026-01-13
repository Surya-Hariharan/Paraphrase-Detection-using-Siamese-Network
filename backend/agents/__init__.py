"""
Multi-Agent System for Paraphrase Detection
============================================

This module provides AI agents for validation, monitoring, and analysis.
"""

from .agent_config import AgentConfig, default_config
from .data_validator import DataValidationAgent
from .inference_validator import InferenceValidatorAgent
from .training_monitor import TrainingMonitorAgent

__all__ = [
    'AgentConfig',
    'default_config',
    'DataValidationAgent',
    'InferenceValidatorAgent',
    'TrainingMonitorAgent'
]
