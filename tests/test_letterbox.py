import numpy as np

from dtf_eval.letterbox import SquareLetterbox


def test_letterbox_point_round_trip_preserves_native_coordinates() -> None:
    mapping = SquareLetterbox(1424, 802, 512)
    points = np.array([[0.0, 0.0], [712.0, 401.0], [1423.0, 801.0]])
    np.testing.assert_allclose(
        mapping.to_native(mapping.to_model_image(points)), points, atol=1e-4
    )


def test_letterbox_preserves_aspect_ratio() -> None:
    mapping = SquareLetterbox(1424, 802, 512)
    width, height = mapping.resized_size
    assert width == 512
    assert abs(width / height - 1424 / 802) < 0.01
