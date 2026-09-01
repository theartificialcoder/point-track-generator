import numpy as np

from dtf_eval.dataset import Frame, Instance
from dtf_eval.qualification import score_annotated_recording
from dtf_eval.rolling import RecordedTrajectories


def _frame(index: int) -> Frame:
    mask = np.zeros((8, 12), dtype=bool)
    mask[2:6, 2 + index : 6 + index] = True
    return Frame(
        index,
        np.zeros((8, 12, 3), dtype=np.uint8),
        (Instance(7, "car", mask),),
        index / 10.0,
    )


def test_annotated_recording_scores_birth_assignment_and_coverage() -> None:
    recording = RecordedTrajectories(
        coordinates=np.array(
            [
                [[3, 3], [10, 0]],
                [[4, 3], [10, 0]],
                [[5, 3], [10, 0]],
            ],
            dtype=np.float32,
        ),
        confidence=np.ones((3, 2), dtype=np.float32),
        birth_frames=np.zeros(2, dtype=np.int32),
        source_frame_indices=np.arange(3),
        fps=10.0,
        frame_size=(12, 8),
        provider="test",
        future_context_frames=0,
        metadata={"runtime_seconds": 0.1, "peak_gpu_memory_bytes": 0},
    )

    report = score_annotated_recording(
        recording,
        tuple(_frame(index) for index in range(3)),
        visibility_threshold=0.6,
    )

    assert report["birth_assignment"]["fraction"] == 0.5
    assert report["object_coverage"]["object_frame_coverage"] == 1.0
    retention = report["region_retention_by_confidence"]["0.60"]
    assert retention["same_object_recall"] == 1.0
