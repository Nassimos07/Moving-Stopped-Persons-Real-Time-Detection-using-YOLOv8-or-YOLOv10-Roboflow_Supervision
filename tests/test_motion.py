import numpy as np

from movement_detection.motion import MOVING, PROCESSING, STOPPED, absolute_state, get_bbox_center, relative_state


def test_get_bbox_center():
    center = get_bbox_center(np.array([0, 0, 10, 20]))
    assert center.tolist() == [5.0, 10.0]


def test_absolute_state_processing():
    state = absolute_state(99999, threshold=10)
    assert state.class_id == PROCESSING


def test_absolute_state_moving():
    state = absolute_state(15, threshold=10)
    assert state.class_id == MOVING


def test_relative_state_stopped():
    state = relative_state(0.001, box_width=100, box_height=120, factor=0.00045)
    assert state.class_id == STOPPED
