"""
Agentic AI Module for Paraphrase Detection Workflow
====================================================

This module provides AI agents that act as supervisors and validators
throughout the data processing, training, and inference pipeline.

Agents:
- DataValidationAgent: Validates and cleans training data
- TrainingMonitorAgent: Monitors training progress and suggests improvements
- InferenceValidatorAgent: Validates predictions and handles edge cases
"""

from .data_validator import DataValidationAgent
from .training_monitor import TrainingMonitorAgent
from .inference_validator import InferenceValidatorAgent
from .agent_config import AgentConfig

__all__ = [
    'DataValidationAgent',
    'TrainingMonitorAgent',
    'InferenceValidatorAgent',
    'AgentConfig'
]
