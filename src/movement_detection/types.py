from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TrackState:
    displacement: float
    class_id: int
