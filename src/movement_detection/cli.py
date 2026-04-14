from __future__ import annotations

from .config import AppConfig
from .pipeline import MovementPipeline


class MovementCLI:
    """CLI for moving vs stopped person detection."""

    def run(
        self,
        config: str = "configs/default.yaml",
        mode: str = "relative",
        filter_mode: str = "all",
    ) -> dict[str, str | int]:
        app_config = AppConfig.from_yaml(config)
        pipeline = MovementPipeline(app_config)
        artifacts = pipeline.run(mode=mode, filter_mode=filter_mode)
        return {
            "output_path": str(artifacts.output_path),
            "processed_frames": artifacts.processed_frames,
        }
