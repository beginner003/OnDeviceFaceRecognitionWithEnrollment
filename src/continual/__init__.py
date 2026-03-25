"""Continual-learning classifier heads and update rules."""

from src.continual.classifier import CosineLinear
from src.continual.naive_ft import NaiveFTConfig, incremental_train_naive

__all__ = [
    "CosineLinear",
    "NaiveFTConfig",
    "incremental_train_naive",
]
