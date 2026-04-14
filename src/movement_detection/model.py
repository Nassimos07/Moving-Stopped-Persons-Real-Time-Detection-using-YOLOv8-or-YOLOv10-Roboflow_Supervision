from __future__ import annotations

import torch
from ultralytics import YOLO


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(model_path: str, device: str) -> YOLO:
    resolved_device = resolve_device(device)
    return YOLO(model_path).to(resolved_device)
