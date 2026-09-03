import numpy as np

from dtf_eval.dataset import Frame, Instance
from dtf_eval.sparse import QuerySet, SparseTracks
from dtf_eval.sparse_viewer import write_neutral_track_viewer, write_sparse_viewer


def test_sparse_viewer_is_self_contained(tmp_path) -> None:
    mask = np.zeros((8, 12), dtype=bool)
    mask[2:6, 3:7] = True
    frame = Frame(10, np.zeros((8, 12, 3), dtype=np.uint8), (Instance(4, "car", mask),))
    queries = QuerySet(np.array([[4, 3]], dtype=np.float32), np.array([4]), np.array(["car"]))
    tracks = SparseTracks(np.array([[[4, 3]]], dtype=np.float32), np.ones((1, 1)), queries, 0.1)
    output = tmp_path / "viewer.html"
    write_sparse_viewer(output, (frame,), {"test": tracks}, (12, 8))
    html = output.read_text(encoding="utf-8")
    assert "__SPARSE_TRACK_DATA__" not in html
    assert "data:image/jpeg;base64" in html
    assert '"category":"car"' in html


def test_neutral_viewer_does_not_require_annotations(tmp_path) -> None:
    queries = QuerySet(np.array([[4, 3]], dtype=np.float32), np.array([4]), np.array(["car"]))
    tracks = SparseTracks(np.array([[[4.25, 3.5]]], dtype=np.float32), np.ones((1, 1)), queries, 0.1)
    output = tmp_path / "viewer.html"
    write_neutral_track_viewer(
        output,
        tracks,
        video_file="clip.mp4",
        frame_numbers=np.array([10]),
        timestamps_seconds=np.array([0.0]),
        size=(12, 8),
    )
    html = output.read_text(encoding="utf-8")
    assert "__TRACK_DATA__" not in html
    assert '"videoFile":"clip.mp4"' in html
    assert "Reference mask" not in html
    assert "Selected trail" in html
    assert "Inactive positions" in html


def test_neutral_viewer_accepts_different_query_sets(tmp_path) -> None:
    first = SparseTracks(
        np.array([[[4, 3]]], dtype=np.float32),
        np.ones((1, 1)),
        QuerySet(np.array([[4, 3]], dtype=np.float32), np.array([1]), np.array(["car"])),
        0.1,
    )
    second = SparseTracks(
        np.array([[[4, 3], [8, 3]]], dtype=np.float32),
        np.ones((1, 2)),
        QuerySet(
            np.array([[4, 3], [8, 3]], dtype=np.float32),
            np.array([1, 1]),
            np.array(["car", "car"]),
        ),
        0.2,
    )
    output = tmp_path / "comparison.html"

    write_neutral_track_viewer(
        output,
        {"Stride 8": first, "Stride 4": second},
        video_file="clip.mp4",
        frame_numbers=np.array([10]),
        timestamps_seconds=np.array([0.0]),
        size=(12, 8),
    )

    html = output.read_text(encoding="utf-8")
    assert '"Stride 8"' in html
    assert '"Stride 4"' in html
    assert "Result <select" in html
    assert "video.currentTime>=DATA.timestamps[DATA.frameCount-1]" in html
