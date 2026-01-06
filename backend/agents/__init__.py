"""
Agentic evaluation and crew-based analysis.

This module provides multi-agent systems for semantic analysis
and paraphrase evaluation using CrewAI.
"""

from .agent_crew import MultiAgentPipeline
from .agentic_evaluator import MultiAgentTestCaseGenerator

__all__ = [
    "MultiAgentPipeline",
    "MultiAgentTestCaseGenerator",
]
