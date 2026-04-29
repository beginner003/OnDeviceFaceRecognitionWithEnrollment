"""Compatibility aliases for the implemented no-replay LwF strategy."""

from __future__ import annotations

from src.continual.lwf import LwFConfig, LwFStrategy, incremental_train_lwf

ReplayLwFConfig = LwFConfig
ReplayLwFStrategy = LwFStrategy
incremental_train_replay_lwf = incremental_train_lwf

__all__ = [
    "ReplayLwFConfig",
    "ReplayLwFStrategy",
    "incremental_train_replay_lwf",
]
