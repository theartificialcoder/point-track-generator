"""Tracker-neutral trajectory field representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrajectoryField:
    """Coordinates of every reference-grid point throughout one clip.

    Coordinates and the reference grid use original-image pixel coordinates.
    Arrays have shapes ``(time, height, width, 2)``, ``(time, height, width)``,
    and ``(height, width, 2)`` respectively.
    """

    coordinates: np.ndarray
    visibility: np.ndarray
    reference_grid: np.ndarray
    reference_index: int
    runtime_seconds: float

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 4 or self.coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (T, H, W, 2)")
        if self.visibility.shape != self.coordinates.shape[:-1]:
            raise ValueError("visibility must have shape (T, H, W)")
        if self.reference_grid.shape != self.coordinates.shape[1:]:
            raise ValueError("reference_grid must have shape (H, W, 2)")
        if not 0 <= self.reference_index < self.coordinates.shape[0]:
            raise ValueError("reference_index lies outside the clip")

    @property
    def length(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.coordinates.shape[1]), int(self.coordinates.shape[2])


@dataclass(frozen=True, slots=True)
class GroupField:
    """Final-layer DTF centroid membership on the reference frame."""

    labels: np.ndarray
    confidence: np.ndarray
    group_count: int
    layer_index: int

    def __post_init__(self) -> None:
        if self.labels.ndim != 2:
            raise ValueError("group labels must have shape (H, W)")
        if self.confidence.shape != self.labels.shape:
            raise ValueError("group confidence must match group labels")
        if self.group_count < 1:
            raise ValueError("group count must be positive")
