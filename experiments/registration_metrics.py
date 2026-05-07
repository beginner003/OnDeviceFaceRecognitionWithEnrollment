"""Per-face registration timing and storage logging helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.system import RegistrationResult


@dataclass(frozen=True)
class StorageSnapshot:
    exemplar_disk_bytes: int
    gaussian_disk_bytes: int

    @property
    def total_disk_bytes(self) -> int:
        return int(self.exemplar_disk_bytes + self.gaussian_disk_bytes)


@dataclass(frozen=True)
class RegistrationEvent:
    identity: str
    elapsed_s: float
    disk_delta_total_bytes: int
    disk_delta_exemplar_bytes: int
    disk_delta_gaussian_bytes: int


def _sum_npz_size_bytes(root: Path) -> int:
    if not root.is_dir():
        return 0
    return int(sum(p.stat().st_size for p in root.rglob("*.npz") if p.is_file()))


def take_storage_snapshot(workspace: Path) -> StorageSnapshot:
    workspace = Path(workspace)
    exemplar_root = workspace / "exemplars"
    gaussian_root = workspace / "gaussians"
    return StorageSnapshot(
        exemplar_disk_bytes=_sum_npz_size_bytes(exemplar_root),
        gaussian_disk_bytes=_sum_npz_size_bytes(gaussian_root),
    )


def log_registration_event(
    *,
    metrics,
    task_name: str,
    result: RegistrationResult,
    before: StorageSnapshot,
    after: StorageSnapshot,
) -> RegistrationEvent:
    d_ex = int(after.exemplar_disk_bytes - before.exemplar_disk_bytes)
    d_ga = int(after.gaussian_disk_bytes - before.gaussian_disk_bytes)
    d_total = int(after.total_disk_bytes - before.total_disk_bytes)
    metrics.info(
        "Per-face registration | %s/%s | elapsed %.3f s | mem(ex=%d B, gauss=%d B) | "
        "disk(ex=%d B, gauss=%d B, total=%d B, delta=%+d B)",
        task_name,
        result.identity,
        float(result.elapsed_s),
        int(result.exemplar_bytes),
        int(result.gaussian_bytes),
        int(after.exemplar_disk_bytes),
        int(after.gaussian_disk_bytes),
        int(after.total_disk_bytes),
        int(d_total),
    )
    return RegistrationEvent(
        identity=result.identity,
        elapsed_s=float(result.elapsed_s),
        disk_delta_total_bytes=d_total,
        disk_delta_exemplar_bytes=d_ex,
        disk_delta_gaussian_bytes=d_ga,
    )


def log_registration_summary(*, metrics, events: List[RegistrationEvent]) -> None:
    if not events:
        metrics.info("Registration summary: no registration events recorded.")
        return

    total_faces = len(events)
    total_elapsed = float(sum(e.elapsed_s for e in events))
    avg_elapsed = total_elapsed / float(total_faces)

    total_delta_ex = int(sum(e.disk_delta_exemplar_bytes for e in events))
    total_delta_ga = int(sum(e.disk_delta_gaussian_bytes for e in events))
    total_delta = int(sum(e.disk_delta_total_bytes for e in events))

    avg_delta_ex = float(total_delta_ex) / float(total_faces)
    avg_delta_ga = float(total_delta_ga) / float(total_faces)
    avg_delta_total = float(total_delta) / float(total_faces)

    metrics.info(
        "Registration summary | faces=%d | total_elapsed=%.3f s | avg_elapsed=%.3f s/face",
        total_faces,
        total_elapsed,
        avg_elapsed,
    )
    metrics.info(
        "Storage summary | exemplar_total_delta=%+d B (avg %+0.1f B/face) | "
        "gaussian_total_delta=%+d B (avg %+0.1f B/face) | "
        "overall_total_delta=%+d B (avg %+0.1f B/face)",
        total_delta_ex,
        avg_delta_ex,
        total_delta_ga,
        avg_delta_ga,
        total_delta,
        avg_delta_total,
    )
