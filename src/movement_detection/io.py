from __future__ import annotations

from pathlib import Path

import cv2
import supervision as sv

from .config import AppConfig
from .utils import ensure_dir


def load_video_info(source_video: str) -> sv.VideoInfo:
    return sv.VideoInfo.from_video_path(video_path=source_video)


def create_writer(config: AppConfig, output_name: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    output_dir = ensure_dir(config.output_dir)
    output_path = Path(output_dir) / output_name
    codec = cv2.VideoWriter_fourcc(*config.codec.upper())
    return cv2.VideoWriter(str(output_path), codec, fps, (width, height))
