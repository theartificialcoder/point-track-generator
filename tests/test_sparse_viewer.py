import numpy as np

from dtf_eval.dataset import Frame, Instance
from dtf_eval.sparse import QuerySet, SparseTracks
from dtf_eval.sparse_viewer import write_sparse_viewer


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
