"""Continual-learning classifier heads and update rules."""

from src.continual.classifier import CosineLinear
from src.continual.exemplar_replay import ExemplarReplayStrategy
from src.continual.lwf import LwFConfig, LwFStrategy, incremental_train_lwf
from src.continual.naive_ft import NaiveFTConfig, NaiveFTStrategy, incremental_train_naive
from src.continual.replay_lwf import ReplayLwFConfig, ReplayLwFStrategy, incremental_train_replay_lwf

__all__ = [
    "CosineLinear",
    "ExemplarReplayStrategy",
    "LwFConfig",
    "LwFStrategy",
    "NaiveFTConfig",
    "NaiveFTStrategy",
    "ReplayLwFConfig",
    "ReplayLwFStrategy",
    "incremental_train_lwf",
    "incremental_train_naive",
    "incremental_train_replay_lwf",
]
