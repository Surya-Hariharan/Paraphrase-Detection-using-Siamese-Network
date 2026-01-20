"""
Core neural network components and training pipeline.

This module contains the fundamental building blocks of the Siamese network
for paraphrase detection.
"""

from .model import (
    TrainableSiameseModel,
    ThresholdOptimizer,
    SiameseProjectionModel,
    ProjectionHead,
    ContrastiveLoss,
)

__all__ = [
    "TrainableSiameseModel",
    "ThresholdOptimizer",
    "SiameseProjectionModel",
    "ProjectionHead",
    "ContrastiveLoss",
]
