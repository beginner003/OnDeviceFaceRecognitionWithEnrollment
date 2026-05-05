"""Helpers for lightweight process memory metrics in experiment scripts."""

from __future__ import annotations

import platform
import resource
from dataclasses import dataclass


def _maxrss_bytes() -> int:
    """Return process peak RSS in bytes on Raspberry Pi OS / Linux."""
    if platform.system() != "Linux":
        raise RuntimeError("Experiment memory metrics are supported only on Raspberry Pi OS / Linux.")
    maxrss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maxrss * 1024


@dataclass(frozen=True)
class MemorySnapshot:
    peak_rss_bytes: int

    @property
    def peak_rss_mb(self) -> float:
        return float(self.peak_rss_bytes) / (1024.0 * 1024.0)


def snapshot_peak_memory() -> MemorySnapshot:
    """Capture current process peak RSS snapshot."""
    return MemorySnapshot(peak_rss_bytes=_maxrss_bytes())


def peak_delta_mb(before: MemorySnapshot, after: MemorySnapshot) -> float:
    """Compute increase in process peak RSS between snapshots in MB."""
    delta = max(0, int(after.peak_rss_bytes) - int(before.peak_rss_bytes))
    return float(delta) / (1024.0 * 1024.0)
