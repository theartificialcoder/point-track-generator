"""Tracking metrics that state exactly what the traffic annotations can support."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from .dataset import Frame
from .field import TrajectoryField


def _values_at(mask: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    finite = np.isfinite(coordinates).all(axis=-1)
    safe = np.where(finite[..., None], coordinates, 0)
    x = np.rint(safe[..., 0]).astype(np.int32)
    y = np.rint(safe[..., 1]).astype(np.int32)
    inside = finite & (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    values = np.zeros(x.shape, dtype=bool)
    values[inside] = mask[y[inside], x[inside]]
    return values


def _scale_name(area: int) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def region_consistency(
    field: TrajectoryField,
    frames: tuple[Frame, ...],
    visibility_level: float = 0.5,
) -> dict[str, Any]:
    """Measure whether reference-object points remain in the same annotated track.

    This is a region-membership proxy, not exact point-trajectory ground truth.
    """

    reference_frame = frames[field.reference_index]
    tracks_by_frame = [
        {instance.track_id: instance for instance in frame.instances if instance.track_id is not None}
        for frame in frames
    ]
    union_by_frame = []
    for frame in frames:
        union = np.zeros(frame.image.shape[:2], dtype=bool)
        for instance in frame.instances:
            union |= instance.mask
        union_by_frame.append(union)

    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    grid = field.reference_grid
    for reference_instance in reference_frame.instances:
        track_id = reference_instance.track_id
        if track_id is None:
            continue
        seeds = _values_at(reference_instance.mask, grid)
        if not np.any(seeds):
            continue
        scale = _scale_name(int(reference_instance.mask.sum()))
        for target_index, target_tracks in enumerate(tracks_by_frame):
            if target_index == field.reference_index or track_id not in target_tracks:
                continue
            coordinates = field.coordinates[target_index]
            visible = field.visibility[target_index] >= visibility_level
            active = seeds & visible
            total = int(seeds.sum())
            visible_count = int(active.sum())
            same = int((_values_at(target_tracks[track_id].mask, coordinates) & active).sum())
            any_object = _values_at(union_by_frame[target_index], coordinates) & active
            other = int(any_object.sum()) - same
            background = visible_count - same - other
            for bucket in ("overall", scale):
                totals[bucket]["seed_observations"] += total
                totals[bucket]["visible"] += visible_count
                totals[bucket]["same_object"] += same
                totals[bucket]["other_object"] += max(0, other)
                totals[bucket]["background"] += max(0, background)

    output: dict[str, Any] = {}
    for bucket, counts in totals.items():
        seeds = max(1, counts["seed_observations"])
        visible = max(1, counts["visible"])
        output[bucket] = {
            **counts,
            "visibility_rate": counts["visible"] / seeds,
            "same_object_recall": counts["same_object"] / seeds,
            "same_object_precision": counts["same_object"] / visible,
            "identity_leak_rate": counts["other_object"] / visible,
            "background_leak_rate": counts["background"] / visible,
        }
    return output


def cycle_consistency(
    forward: TrajectoryField,
    reverse: TrajectoryField,
    visibility_level: float = 0.5,
) -> dict[str, float | int]:
    """Return forward/backward closure error in original-image pixels."""

    target_index = reverse.reference_index
    if forward.reference_index == target_index:
        raise ValueError("cycle endpoints must differ")
    if forward.grid_shape != reverse.grid_shape:
        raise ValueError("cycle fields must use the same grid")

    target = forward.coordinates[target_index]
    grid = reverse.reference_grid
    dx = float(np.median(np.diff(grid[0, :, 0]))) if grid.shape[1] > 1 else 1.0
    dy = float(np.median(np.diff(grid[:, 0, 1]))) if grid.shape[0] > 1 else 1.0
    map_x = ((target[..., 0] - grid[0, 0, 0]) / dx).astype(np.float32)
    map_y = ((target[..., 1] - grid[0, 0, 1]) / dy).astype(np.float32)
    target_in_bounds = (map_x >= 0) & (map_x <= grid.shape[1] - 1)
    target_in_bounds &= (map_y >= 0) & (map_y <= grid.shape[0] - 1)

    backward_to_origin = reverse.coordinates[forward.reference_index]
    returned = np.stack(
        [
            cv2.remap(
                backward_to_origin[..., channel],
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            for channel in range(2)
        ],
        axis=-1,
    )
    reverse_visibility = cv2.remap(
        reverse.visibility[forward.reference_index].astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    valid = target_in_bounds & (forward.visibility[target_index] >= visibility_level)
    valid &= reverse_visibility >= visibility_level
    valid &= np.isfinite(returned).all(axis=-1)
    error = np.linalg.norm(returned - forward.reference_grid, axis=-1)[valid]
    if not error.size:
        return {"samples": 0, "median_px": float("nan"), "p90_px": float("nan")}
    return {
        "samples": int(error.size),
        "median_px": float(np.median(error)),
        "p90_px": float(np.percentile(error, 90)),
    }
