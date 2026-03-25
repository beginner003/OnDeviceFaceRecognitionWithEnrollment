"""Continual-learning classifier heads and update rules."""

from src.continual.classifier import CosineLinear
from src.continual.exemplar_replay import ExemplarReplayStrategy
from src.continual.naive_ft import NaiveFTConfig, NaiveFTStrategy, incremental_train_naive
from src.continual.replay_lwf import ReplayLwFStrategy

__all__ = [
    "CosineLinear",
    "ExemplarReplayStrategy",
    "NaiveFTConfig",
    "NaiveFTStrategy",
    "ReplayLwFStrategy",
    "incremental_train_naive",
]
