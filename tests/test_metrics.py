import numpy as np

from dtf_eval.dataset import Frame, Instance
from dtf_eval.field import TrajectoryField
from dtf_eval.metrics import cycle_consistency, region_consistency


def _grid(height: int, width: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack((x, y), axis=-1).astype(np.float32)


def test_translation_remains_in_same_object() -> None:
    grid = _grid(8, 8)
    coordinates = np.stack((grid, grid + np.array([1, 0], dtype=np.float32)))
    field = TrajectoryField(coordinates, np.ones((2, 8, 8)), grid, 0, 0.1)
    mask0 = np.zeros((8, 8), bool)
    mask1 = np.zeros((8, 8), bool)
    mask0[2:5, 2:5] = True
    mask1[2:5, 3:6] = True
    image = np.zeros((8, 8, 3), np.uint8)
    frames = (
        Frame(0, image, (Instance(7, "car", mask0),)),
        Frame(1, image, (Instance(7, "car", mask1),)),
    )

    result = region_consistency(field, frames)["overall"]

    assert result["same_object_recall"] == 1.0
    assert result["identity_leak_rate"] == 0.0


def test_identity_cycle_has_zero_error() -> None:
    grid = _grid(8, 8)
    coordinates = np.stack((grid, grid))
    forward = TrajectoryField(coordinates, np.ones((2, 8, 8)), grid, 0, 0.1)
    reverse = TrajectoryField(coordinates, np.ones((2, 8, 8)), grid, 1, 0.1)

    result = cycle_consistency(forward, reverse)

    assert result["samples"] == 64
    assert result["median_px"] == 0.0

