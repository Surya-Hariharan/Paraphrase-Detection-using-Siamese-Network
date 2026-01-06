"""
Paraphrase Detection Backend Module

This package provides a complete Siamese neural network implementation
for paraphrase and plagiarism detection.

Modules:
    core: Neural network models and training pipeline
    agents: Multi-agent evaluation using CrewAI
    utils: Document processing and utility functions
    api: FastAPI REST endpoints
"""

from .core import (
    TrainableSiameseModel,
    ThresholdOptimizer,
    DocumentLevelSiameseModel,
)

# Optional: Agent modules (requires crewai)
try:
    from .agents import (
        MultiAgentPipeline,
        MultiAgentTestCaseGenerator,
    )
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    MultiAgentPipeline = None
    MultiAgentTestCaseGenerator = None

from .utils import (
    load_document,
    DocumentProcessor,
)

__version__ = "1.0.0"

__all__ = [
    "TrainableSiameseModel",
    "ThresholdOptimizer",
    "DocumentLevelSiameseModel",
    "MultiAgentPipeline",
    "MultiAgentTestCaseGenerator",
    "DocumentLoader",
    "DocumentProcessor",
]
