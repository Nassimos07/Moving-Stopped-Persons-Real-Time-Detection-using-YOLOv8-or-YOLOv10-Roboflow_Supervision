from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from .config import AppConfig
from .io import create_writer, load_video_info
from .model import load_model
from .motion import MOVING, PROCESSING, STOPPED, absolute_state, get_bbox_center, relative_state
from .types import TrackState

CLASS_NAMES = ["Processing..", "Stopped", "Moving"]
COLOR_PALETTE = sv.ColorPalette.from_hex(["#FFDBAC", "#E87A5C", "#B6E696"])
BLACK = sv.Color.BLACK
ROBOFLOW = sv.Color.ROBOFLOW


@dataclass(slots=True)
class PipelineArtifacts:
    output_path: Path
    processed_frames: int


class MovementPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = load_model(config.model_path, config.device)
        self.video_info = load_video_info(config.source_video)
        self.tracker = sv.ByteTrack(frame_rate=self.video_info.fps)
        self.smoother = sv.DetectionsSmoother()

    def run(self, mode: str = "relative", filter_mode: str = "all") -> PipelineArtifacts:
        capture = cv2.VideoCapture(self.config.source_video)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {self.config.source_video}")

        width = int(self.video_info.width)
        height = int(self.video_info.height)
        fps = float(self.config.fps or self.video_info.fps)
        output_name = f"{mode}_{filter_mode}.mp4"
        writer = create_writer(self.config, output_name, width, height, fps)

        trace_annotator = sv.TraceAnnotator(color=ROBOFLOW, trace_length=self.config.trace_length, thickness=3)
        label_annotator = sv.LabelAnnotator(
            color=COLOR_PALETTE,
            text_color=BLACK,
            text_position=sv.Position.TOP_LEFT,
            text_scale=0.7,
        )
        color_annotator = sv.ColorAnnotator(color=COLOR_PALETTE, opacity=0.3)
        ellipse_annotator = sv.EllipseAnnotator(color=COLOR_PALETTE, thickness=4)

        frame_gap = 0
        first_attempt = True
        prior_detections = None
        current_points: dict[int, np.ndarray | list[object]] = {}
        track_states: dict[int, TrackState] = {}
        processed_frames = 0

        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            result = self.model(frame, classes=self.config.classes)[0]
            frame_gap += 1

            if len(result[0]) > 0:
                detections = sv.Detections.from_ultralytics(result)
                detections = self.tracker.update_with_detections(detections)
                detections = self.smoother.update_with_detections(detections)

                annotated_frame = trace_annotator.annotate(scene=frame.copy(), detections=detections)

                if frame_gap == self.config.frame_step:
                    if first_attempt:
                        prior_detections = detections
                        first_attempt = False
                    else:
                        previous_points = {
                            int(prior_detections.tracker_id[i]): get_bbox_center(prior_detections.xyxy[i])
                            for i in range(len(prior_detections.tracker_id))
                        }
                        track_states = {}
                        current_points = {}

                        for i in range(len(detections.tracker_id)):
                            tracker_id = int(detections.tracker_id[i])
                            box = detections.xyxy[i]
                            center = get_bbox_center(box)
                            current_points[tracker_id] = center

                            if tracker_id not in previous_points:
                                track_states[tracker_id] = TrackState(displacement=99999, class_id=PROCESSING)
                                continue

                            if mode == "absolute":
                                displacement = float(np.linalg.norm(center - previous_points[tracker_id]))
                                track_states[tracker_id] = absolute_state(
                                    displacement=displacement,
                                    threshold=self.config.absolute_speed_threshold,
                                )
                            else:
                                width_box = float(box[2] - box[0])
                                height_box = float(box[3] - box[1])
                                displacement = float(
                                    np.linalg.norm(center - previous_points[tracker_id]) / max(width_box, height_box)
                                )
                                track_states[tracker_id] = relative_state(
                                    relative_displacement=displacement,
                                    box_width=width_box,
                                    box_height=height_box,
                                    factor=self.config.relative_speed_factor,
                                )

                        prior_detections = detections
                    frame_gap = 0

                for tracker_id in detections.tracker_id.tolist():
                    tracker_id = int(tracker_id)
                    track_states.setdefault(tracker_id, TrackState(displacement=99999, class_id=PROCESSING))

                detections.class_id = np.array([track_states[int(tid)].class_id for tid in detections.tracker_id])
                detections = self._filter_detections(detections, filter_mode)

                labels = [
                    self._make_label(int(tracker_id), int(class_id), track_states[int(tracker_id)].displacement, mode)
                    for _, _, _, class_id, tracker_id, _ in detections
                ]

                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame.copy(),
                    labels=labels,
                    detections=detections,
                )
                annotated_frame = color_annotator.annotate(scene=annotated_frame.copy(), detections=detections)
                frame = ellipse_annotator.annotate(scene=annotated_frame.copy(), detections=detections)

            if self.config.save_video:
                writer.write(frame)
            if self.config.show:
                self._show_frame(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            processed_frames += 1

        capture.release()
        writer.release()
        cv2.destroyAllWindows()
        return PipelineArtifacts(output_path=Path(self.config.output_dir) / output_name, processed_frames=processed_frames)

    def _filter_detections(self, detections: sv.Detections, filter_mode: str) -> sv.Detections:
        if filter_mode == "moving":
            return detections[np.isin(detections.class_id, [PROCESSING, MOVING])]
        if filter_mode == "stopped":
            return detections[np.isin(detections.class_id, [PROCESSING, STOPPED])]
        return detections

    def _make_label(self, tracker_id: int, class_id: int, displacement: float, mode: str) -> str:
        metric_name = "Absolute speed" if mode == "absolute" else "Relative speed"
        return f"ID: {tracker_id}, {CLASS_NAMES[class_id]}, {metric_name}: {np.round(displacement, 3)}"

    def _show_frame(self, frame: np.ndarray) -> None:
        width = self.config.render.resize_width or 920
        height = self.config.render.resize_height or 560
        cv2.imshow("movement-detection", cv2.resize(frame, (width, height)))
