import numpy as np

from dtf_eval.field import TrajectoryField
from dtf_eval.runtime import benchmark_tracker
from dtf_eval.trackers import DenseTracker


class _Tracker(DenseTracker):
    name = "test"

    def __init__(self) -> None:
        self.calls = 0

    def track(self, frames, reference_index, inference_size):
        self.calls += 1
        width, height = inference_size
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        grid = np.stack((x, y), axis=-1).astype(np.float32)
        coordinates = np.repeat(grid[None], len(frames), axis=0)
        return TrajectoryField(
            coordinates,
            np.ones(coordinates.shape[:-1], dtype=np.float32),
            grid,
            reference_index,
            0.01,
        )


def test_runtime_benchmark_separates_warmup_from_measurement() -> None:
    tracker = _Tracker()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    result = benchmark_tracker(
        tracker,
        (frame, frame),
        0,
        (8, 8),
        warmup=1,
        repeats=3,
    )

    assert tracker.calls == 4
    assert result["measured_runs"] == 3
    assert result["core_compute"]["median_seconds"] == 0.01
    assert result["end_to_end"]["median_seconds"] > 0
