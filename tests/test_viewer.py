import base64

import cv2
import numpy as np

from dtf_eval.field import GroupField, TrajectoryField
from dtf_eval.viewer import _dense_field_uri, _forward_splat, _group_visuals


def test_dense_identity_field_covers_the_complete_grid() -> None:
    x, y = np.meshgrid(np.arange(8), np.arange(6))
    grid = np.stack((x, y), axis=-1).astype(np.float32)
    field = TrajectoryField(
        coordinates=grid[None],
        visibility=np.ones((1, 6, 8), dtype=np.float32),
        reference_grid=grid,
        reference_index=0,
        runtime_seconds=0.0,
    )

    uri = _dense_field_uri(field, 0)
    encoded = np.frombuffer(base64.b64decode(uri.partition(",")[2]), dtype=np.uint8)
    overlay = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    assert overlay.shape == (6, 8, 4)
    assert np.all(overlay[..., 3] > 0)


def test_identity_splat_preserves_reference_values() -> None:
    x, y = np.meshgrid(np.arange(5), np.arange(4))
    grid = np.stack((x, y), axis=-1).astype(np.float32)
    field = TrajectoryField(
        coordinates=grid[None],
        visibility=np.ones((1, 4, 5), dtype=np.float32),
        reference_grid=grid,
        reference_index=0,
        runtime_seconds=0.0,
    )
    values = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)

    output = _forward_splat(field, 0, values)

    np.testing.assert_array_equal(output[..., :3], values)
    assert np.all(output[..., 3] == 255)


def test_group_visuals_preserve_distinct_memberships() -> None:
    x, y = np.meshgrid(np.arange(4), np.arange(2))
    grid = np.stack((x, y), axis=-1).astype(np.float32)
    field = TrajectoryField(
        coordinates=grid[None],
        visibility=np.ones((1, 2, 4), dtype=np.float32),
        reference_grid=grid,
        reference_index=0,
        runtime_seconds=0.0,
    )
    groups = GroupField(
        labels=np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint16),
        confidence=np.full((2, 4), 0.75, dtype=np.float32),
        group_count=2,
        layer_index=7,
    )

    visuals = _group_visuals(field, groups)
    encoded = np.frombuffer(
        base64.b64decode(visuals["membership"][0].partition(",")[2]),
        dtype=np.uint8,
    )
    overlay = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)

    assert visuals["groupCount"] == 2
    assert len(np.unique(overlay[..., :3].reshape(-1, 3), axis=0)) == 2
