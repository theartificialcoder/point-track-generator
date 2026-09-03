import numpy as np

from dtf_eval.dataset import Frame, Instance
from dtf_eval.sparse import (
    QuerySet,
    SparseTracks,
    continuous_strided_queries,
    reference_queries,
    score_object_frame_coverage,
    score_region_retention,
    strided_reference_queries,
)


def _frame(offset: int = 0) -> Frame:
    mask = np.zeros((8, 12), dtype=bool)
    mask[2:6, 2 + offset : 6 + offset] = True
    return Frame(0, np.zeros((8, 12, 3), dtype=np.uint8), (Instance(7, "car", mask),))


def test_reference_queries_are_balanced_and_inside() -> None:
    queries = reference_queries(_frame(), (12, 8), max_points_per_object=5)
    assert 0 < len(queries.points) <= 5
    assert np.all(queries.track_ids == 7)
    assert np.all(_frame().instances[0].mask[queries.points[:, 1].astype(int), queries.points[:, 0].astype(int)])


def test_strided_queries_use_regular_unambiguous_locations() -> None:
    queries = strided_reference_queries(_frame(), (12, 8), stride=2)
    np.testing.assert_array_equal(queries.points % 2, 1)
    assert np.all(queries.track_ids == 7)


def test_continuous_queries_add_objects_at_first_appearance() -> None:
    first = _frame()
    second_mask = np.zeros((8, 12), dtype=bool)
    second_mask[1:4, 7:11] = True
    second = Frame(
        1,
        first.image,
        first.instances + (Instance(9, "truck", second_mask),),
    )
    queries = continuous_strided_queries((first, second), (12, 8), stride=2)
    np.testing.assert_array_equal(
        np.unique(queries.frame_indices[queries.track_ids == 7]), [0]
    )
    np.testing.assert_array_equal(
        np.unique(queries.frame_indices[queries.track_ids == 9]), [1]
    )


def test_continuous_queries_cover_small_object_before_lattice_intersection() -> None:
    tiny = np.zeros((8, 12), dtype=bool)
    tiny[0, 0] = True
    grown = np.zeros((8, 12), dtype=bool)
    grown[0:5, 0:5] = True
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    frames = (
        Frame(0, image, (Instance(7, "car", tiny),)),
        Frame(1, image, (Instance(7, "car", grown),)),
    )

    queries = continuous_strided_queries(frames, (12, 8), stride=8)

    np.testing.assert_array_equal(queries.points, [[0.0, 0.0]])
    np.testing.assert_array_equal(queries.frame_indices, [0])


def test_continuous_queries_add_coverage_as_object_grows() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    small = np.zeros((16, 16), dtype=bool)
    grown = np.zeros((16, 16), dtype=bool)
    small[1:4, 1:4] = True
    grown[1:15, 1:15] = True
    frames = (
        Frame(0, image, (Instance(7, "car", small),)),
        Frame(1, image, (Instance(7, "car", grown),)),
    )

    queries = continuous_strided_queries(frames, (16, 16), stride=4)

    assert np.count_nonzero(queries.frame_indices == 0) == 1
    assert np.count_nonzero(queries.frame_indices == 1) > 1
    assert len(queries.points) == 16


def test_region_retention_separates_background_leak() -> None:
    queries = QuerySet(np.array([[3, 3]], dtype=np.float32), np.array([7]), np.array(["car"]))
    tracks = SparseTracks(
        coordinates=np.array([[[3, 3]], [[9, 3]]], dtype=np.float32),
        visibility=np.ones((2, 1), dtype=np.float32),
        queries=queries,
        runtime_seconds=0.1,
    )
    score = score_region_retention(tracks, (_frame(), _frame(1)), (12, 8))
    assert score["same_object_recall"] == 0.5
    assert score["background_leak_rate"] == 0.5
    assert score["by_source_scale"]["small"]["same_object_recall"] == 0.5


def test_track_age_is_relative_to_each_query_birth() -> None:
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    mask_a = np.zeros((8, 12), dtype=bool)
    mask_b = np.zeros((8, 12), dtype=bool)
    mask_a[1:4, 1:4] = True
    mask_b[4:7, 8:11] = True
    frames = (
        Frame(0, image, (Instance(1, "car", mask_a),)),
        Frame(1, image, (Instance(1, "car", mask_a), Instance(2, "car", mask_b))),
        Frame(2, image, (Instance(1, "car", mask_a), Instance(2, "car", mask_b))),
    )
    queries = QuerySet(
        np.array([[2, 2], [9, 5]], dtype=np.float32),
        np.array([1, 2]),
        np.array(["car", "car"]),
        birth_frames=np.array([0, 1]),
    )
    tracks = SparseTracks(
        np.array(
            [
                [[2, 2], [0, 0]],
                [[2, 2], [9, 5]],
                [[2, 2], [9, 5]],
            ],
            dtype=np.float32,
        ),
        np.ones((3, 2), dtype=np.float32),
        queries,
        0.1,
    )

    score = score_region_retention(tracks, frames, (12, 8))

    assert score["track_age_frames"]["1"]["eligible"] == 2
    assert score["track_age_frames"]["1"]["same_object_recall"] == 1.0


def test_source_scale_is_measured_at_query_birth() -> None:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    small = np.zeros((128, 128), dtype=bool)
    large = np.zeros((128, 128), dtype=bool)
    small[1:5, 1:5] = True
    large[10:110, 10:110] = True
    frames = (
        Frame(0, image, (Instance(7, "car", small),)),
        Frame(1, image, (Instance(7, "car", large),)),
    )
    queries = QuerySet(
        np.array([[50, 50]], dtype=np.float32),
        np.array([7]),
        np.array(["car"]),
        birth_frames=np.array([1]),
    )
    tracks = SparseTracks(
        np.array([[[0, 0]], [[50, 50]]], dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
        queries,
        0.1,
    )

    score = score_region_retention(tracks, frames, (128, 128))

    assert set(score["by_source_scale"]) == {"large"}


def test_object_balanced_score_does_not_favor_object_with_more_queries() -> None:
    mask_a = np.zeros((8, 12), dtype=bool)
    mask_b = np.zeros((8, 12), dtype=bool)
    mask_a[1:4, 1:5] = True
    mask_b[4:7, 7:11] = True
    frame = Frame(
        0,
        np.zeros((8, 12, 3), dtype=np.uint8),
        (Instance(1, "car", mask_a), Instance(2, "car", mask_b)),
    )
    queries = QuerySet(
        np.array([[2, 2], [3, 2], [4, 2], [8, 5]], dtype=np.float32),
        np.array([1, 1, 1, 2]),
        np.array(["car"] * 4),
    )
    tracks = SparseTracks(
        np.array([[[2, 2], [3, 2], [4, 2], [0, 0]]], dtype=np.float32),
        np.ones((1, 4), dtype=np.float32),
        queries,
        0.1,
    )
    score = score_region_retention(tracks, (frame,), (12, 8))
    assert score["same_object_recall"] == 0.75
    assert score["object_balanced"]["same_object_recall"] == 0.5


def test_sparse_tracks_round_trip(tmp_path) -> None:
    queries = QuerySet(np.array([[1, 2]], dtype=np.float32), np.array([4]), np.array(["car"]))
    tracks = SparseTracks(np.zeros((2, 1, 2)), np.ones((2, 1)), queries, 0.3, 42)
    path = tmp_path / "tracks.npz"
    tracks.save(path)
    loaded = SparseTracks.load(path)
    assert loaded.runtime_seconds == 0.3
    assert loaded.peak_gpu_memory_bytes == 42
    np.testing.assert_array_equal(loaded.queries.track_ids, queries.track_ids)


def test_query_set_round_trip(tmp_path) -> None:
    queries = QuerySet(
        np.array([[1, 2]], dtype=np.float32),
        np.array([4]),
        np.array(["car"]),
        birth_frames=np.array([3]),
    )
    path = tmp_path / "queries.npz"
    queries.save(path)
    loaded = QuerySet.load(path)
    np.testing.assert_array_equal(loaded.points, queries.points)
    np.testing.assert_array_equal(loaded.track_ids, queries.track_ids)
    np.testing.assert_array_equal(loaded.frame_indices, queries.frame_indices)


def test_object_frame_coverage_requires_one_correct_active_point() -> None:
    queries = QuerySet(
        np.array([[3, 3]], dtype=np.float32),
        np.array([7]),
        np.array(["car"]),
    )
    tracks = SparseTracks(
        np.array([[[3, 3]], [[9, 3]]], dtype=np.float32),
        np.ones((2, 1), dtype=np.float32),
        queries,
        0.1,
    )

    report = score_object_frame_coverage(tracks, (_frame(), _frame(1)), (12, 8))

    assert report["annotated_object_frames"] == 2
    assert report["covered_object_frames"] == 1
    assert report["coverage_rate"] == 0.5
