import numpy as np
import pytest

from dtf_eval.batch_archive import BatchArchive


def test_batch_archive_resumes_after_completed_batch(tmp_path) -> None:
    path = tmp_path / "run.partial"
    archive = BatchArchive.open(
        path, frame_count=3, point_count=4, identity={"cohort": "fixed"}
    )
    coordinates = np.ones((3, 2, 2), dtype=np.float32)
    visibility = np.full((3, 2), 0.75, dtype=np.float32)
    archive.write(
        0,
        coordinates,
        visibility,
        runtime_seconds=2.0,
        peak_gpu_memory_bytes=100,
    )

    resumed = BatchArchive.open(
        path, frame_count=3, point_count=4, identity={"cohort": "fixed"}
    )

    assert resumed.next_query == 2
    assert np.array_equal(resumed.coordinates[:, :2], coordinates)
    assert resumed.metadata["runtime_seconds"] == 2.0


def test_batch_archive_rejects_a_different_run(tmp_path) -> None:
    path = tmp_path / "run.partial"
    BatchArchive.open(path, frame_count=3, point_count=4, identity={"cohort": "first"})

    with pytest.raises(ValueError, match="identity"):
        BatchArchive.open(
            path, frame_count=3, point_count=4, identity={"cohort": "second"}
        )
