import cv2
import numpy as np

from dtf_eval.trackers import FarnebackChainTracker


def test_farneback_tracker_has_consistent_shapes() -> None:
    first = np.zeros((64, 96, 3), np.uint8)
    second = first.copy()
    cv2.rectangle(first, (20, 20), (40, 40), (255, 255, 255), -1)
    cv2.rectangle(second, (23, 20), (43, 40), (255, 255, 255), -1)

    result = FarnebackChainTracker().track((first, second), 0, (96, 64))

    assert result.coordinates.shape == (2, 64, 96, 2)
    assert result.visibility.shape == (2, 64, 96)
    moved = result.coordinates[1, 30, 30, 0] - result.reference_grid[30, 30, 0]
    assert moved > 1.0

