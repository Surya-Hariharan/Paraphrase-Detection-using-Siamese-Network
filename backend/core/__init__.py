"""
Core neural network components and training pipeline.

This module contains the fundamental building blocks of the Siamese network
for paraphrase detection.
"""

from .neural_engine import (
    TrainableSiameseModel,
    ThresholdOptimizer,
    SBERTEncoder,
    ProjectionHead,
)

from .document_siamese_pipeline import DocumentLevelSiameseModel

__all__ = [
    "TrainableSiameseModel",
    "ThresholdOptimizer",
    "SBERTEncoder",
    "ProjectionHead",
    "DocumentLevelSiameseModel",
]
