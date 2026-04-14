from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import load_yaml


@dataclass(slots=True)
class RenderConfig:
    resize_width: int | None = None
    resize_height: int | None = None


@dataclass(slots=True)
class AppConfig:
    model_path: str = "yolov8x.pt"
    device: str = "auto"
    classes: list[int] = field(default_factory=lambda: [0])
    source_video: str = "assets/videos/test1.mp4"
    output_dir: str = "outputs"
    frame_step: int = 10
    absolute_speed_threshold: float = 10.0
    relative_speed_factor: float = 0.00045
    trace_length: int = 70
    show: bool = False
    save_video: bool = True
    fps: float | None = None
    codec: str = "mp4v"
    render: RenderConfig = field(default_factory=RenderConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        raw: dict[str, Any] = load_yaml(path)
        render = RenderConfig(**raw.pop("render", {}))
        return cls(render=render, **raw)
