from __future__ import annotations

import numpy as np

from .types import TrackState

PROCESSING = 0
STOPPED = 1
MOVING = 2


def get_bbox_center(box: np.ndarray) -> np.ndarray:
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return np.array([center_x, center_y])


def absolute_state(displacement: float, threshold: float) -> TrackState:
    if displacement == 99999:
        return TrackState(displacement=displacement, class_id=PROCESSING)
    if displacement > threshold:
        return TrackState(displacement=displacement, class_id=MOVING)
    return TrackState(displacement=displacement, class_id=STOPPED)


def relative_state(relative_displacement: float, box_width: float, box_height: float, factor: float) -> TrackState:
    threshold = factor * max(box_width, box_height)
    if relative_displacement == 99999:
        return TrackState(displacement=relative_displacement, class_id=PROCESSING)
    if relative_displacement > threshold:
        return TrackState(displacement=relative_displacement, class_id=MOVING)
    return TrackState(displacement=relative_displacement, class_id=STOPPED)
