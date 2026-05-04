"""Persist UI-facing `SystemConfig` fields so the desktop matches the saved workspace."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.system import SystemConfig

_PREFS_NAME = "ui_preferences.json"

_VALID_REGISTRATION = frozenset({"naive", "replay", "replay_lwf", "synthetic_replay"})
_VALID_EXEMPLAR_SEL = frozenset({"herding", "random"})
_VALID_RECOGNITION = frozenset({"ncm", "classifier"})


@dataclass
class UiPreferences:
    registration: str = "replay_lwf"
    exemplar_selection: str = "herding"
    recognition: str = "classifier"
    exemplar_k: int = 50
    confidence_threshold: float = 0.5

    def to_system_config(self) -> SystemConfig:
        return SystemConfig(
            registration=self.registration,
            exemplar_selection=self.exemplar_selection,
            recognition=self.recognition,
            exemplar_k=int(self.exemplar_k),
            confidence_threshold=float(self.confidence_threshold),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def preferences_path(workspace: Path) -> Path:
    return Path(workspace) / _PREFS_NAME


def load_preferences(workspace: Path) -> UiPreferences:
    path = preferences_path(workspace)
    if not path.is_file():
        return UiPreferences()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return UiPreferences(
        registration=str(raw.get("registration", "replay_lwf")),
        exemplar_selection=str(raw.get("exemplar_selection", "herding")),
        recognition=str(raw.get("recognition", "classifier")),
        exemplar_k=int(raw.get("exemplar_k", 50)),
        confidence_threshold=float(raw.get("confidence_threshold", 0.5)),
    )


def save_preferences(workspace: Path, prefs: UiPreferences) -> None:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    preferences_path(workspace).write_text(
        json.dumps(prefs.to_json_dict(), indent=2),
        encoding="utf-8",
    )


def validate_preferences_dict(data: dict[str, Any]) -> UiPreferences:
    reg = str(data.get("registration", "replay_lwf"))
    exs = str(data.get("exemplar_selection", "herding"))
    rec = str(data.get("recognition", "classifier"))
    if reg not in _VALID_REGISTRATION:
        raise ValueError(f"registration must be one of {sorted(_VALID_REGISTRATION)}")
    if exs not in _VALID_EXEMPLAR_SEL:
        raise ValueError(f"exemplar_selection must be one of {sorted(_VALID_EXEMPLAR_SEL)}")
    if rec not in _VALID_RECOGNITION:
        raise ValueError(f"recognition must be one of {sorted(_VALID_RECOGNITION)}")
    k = int(data.get("exemplar_k", 50))
    thr = float(data.get("confidence_threshold", 0.5))
    if k <= 0:
        raise ValueError("exemplar_k must be > 0")
    if not (0.0 <= thr <= 1.0):
        raise ValueError("confidence_threshold must be in [0, 1]")
    return UiPreferences(
        registration=reg,
        exemplar_selection=exs,
        recognition=rec,
        exemplar_k=k,
        confidence_threshold=thr,
    )
