"""Bounded rolling admission and recording for point-trajectory providers."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .admission import AdmissionSupport


@dataclass(frozen=True, slots=True)
class WindowTracks:
    coordinates: np.ndarray
    confidence: np.ndarray
    runtime_seconds: float
    peak_gpu_memory_bytes: int

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 2:
            raise ValueError("window coordinates must have shape (T,N,2)")
        if self.confidence.shape != self.coordinates.shape[:2]:
            raise ValueError("window confidence shape mismatch")


class WindowTracker(Protocol):
    name: str
    future_context_frames: int

    def track(
        self,
        frames_rgb: np.ndarray,
        query_points: np.ndarray,
        *,
        native_size: tuple[int, int],
    ) -> WindowTracks: ...


@dataclass(frozen=True, slots=True)
class RecordedTrajectories:
    coordinates: np.ndarray
    confidence: np.ndarray
    birth_frames: np.ndarray
    source_frame_indices: np.ndarray
    fps: float
    frame_size: tuple[int, int]
    provider: str
    future_context_frames: int
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        frames, tracks = self.coordinates.shape[:2]
        if self.coordinates.shape != (frames, tracks, 2):
            raise ValueError("trajectory coordinates must have shape (T,N,2)")
        if self.confidence.shape != (frames, tracks):
            raise ValueError("trajectory confidence must have shape (T,N)")
        if self.birth_frames.shape != (tracks,):
            raise ValueError("trajectory births must contain one value per track")
        if self.source_frame_indices.shape != (frames,):
            raise ValueError("source frame indices must contain one value per frame")
        if frames < 2 or tracks < 1:
            raise ValueError("trajectory recording requires at least two frames and one track")
        if not np.isfinite(self.coordinates).all() or not np.isfinite(self.confidence).all():
            raise ValueError("trajectory coordinates and confidence must be finite")
        if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
            raise ValueError("trajectory confidence must lie in [0,1]")
        if np.any((self.birth_frames < 0) | (self.birth_frames >= frames)):
            raise ValueError("trajectory births must lie inside the recording")
        if not np.all(np.diff(self.source_frame_indices) == 1):
            raise ValueError("source frames must be chronological and consecutive")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("trajectory fps must be positive")
        if min(self.frame_size) < 1 or not self.provider:
            raise ValueError("trajectory geometry and provider are required")
        if self.future_context_frames < 0:
            raise ValueError("future context must be non-negative")

    def save(self, path: str | Path, *, visibility_threshold: float) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "fps": self.fps,
            "frame_size": list(self.frame_size),
            "provider": self.provider,
            "causal": False,
            "future_context_frames": self.future_context_frames,
            "metadata": self.metadata,
        }
        width, height = self.frame_size
        inside = (
            (self.coordinates[..., 0] >= 0.0)
            & (self.coordinates[..., 0] < width)
            & (self.coordinates[..., 1] >= 0.0)
            & (self.coordinates[..., 1] < height)
        )
        np.savez_compressed(
            destination,
            coordinates=self.coordinates.astype(np.float32),
            visibility=(self.confidence >= visibility_threshold) & inside,
            confidence=self.confidence.astype(np.float32),
            birth_frames=self.birth_frames.astype(np.int32),
            track_labels=np.arange(len(self.birth_frames), dtype=np.int64),
            source_frame_indices=self.source_frame_indices.astype(np.int64),
            document=np.asarray(json.dumps(document, sort_keys=True)),
        )


def record_rolling_trajectories(
    frames_rgb: np.ndarray,
    support: AdmissionSupport,
    tracker: WindowTracker,
    *,
    stride: int,
    coverage_radius: float,
    advance_frames: int,
    window_frames: int,
    max_active_tracks: int,
    visibility_threshold: float,
) -> RecordedTrajectories:
    """Track corrected-motion points with bounded active state and stable IDs."""

    frame_count = len(support.source_frame_indices)
    if frames_rgb.shape[0] != frame_count or frames_rgb.ndim != 4:
        raise ValueError("tracker frames must align with admission support")
    if min(stride, advance_frames, window_frames, max_active_tracks) < 1:
        raise ValueError("rolling controls must be positive")
    if window_frames <= advance_frames or coverage_radius <= 0.0:
        raise ValueError("rolling window must exceed its advance and radius must be positive")
    if not 0.0 < visibility_threshold < 1.0:
        raise ValueError("visibility threshold must lie in (0,1)")

    width, height = support.frame_size
    xs = np.arange(stride // 2, width, stride, dtype=np.int32)
    ys = np.arange(stride // 2, height, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coordinate_columns: list[np.ndarray] = []
    confidence_columns: list[np.ndarray] = []
    birth_frames: list[int] = []
    active_ids = np.empty(0, dtype=np.int64)
    active_points = np.empty((0, 2), dtype=np.float32)
    active_confidence = np.empty(0, dtype=np.float32)
    admitted = retired = saturated_windows = 0
    runtime_seconds = 0.0
    peak_memory = 0
    started = time.perf_counter()

    for boundary in range(0, frame_count, advance_frames):
        active_valid = _valid_points(
            active_points,
            active_confidence,
            width=width,
            height=height,
            threshold=visibility_threshold,
        )
        retired += int(len(active_ids) - np.count_nonzero(active_valid))
        active_ids = active_ids[active_valid]
        active_points = active_points[active_valid]
        active_confidence = active_confidence[active_valid]

        selected = support.frame(boundary)[grid_y, grid_x]
        candidates = np.column_stack((grid_x[selected], grid_y[selected])).astype(np.float32)
        uncovered = candidates[
            ~_covered_sites(
                candidates,
                active_points,
                shape=(height, width),
                radius=coverage_radius,
            )
        ]
        available = max_active_tracks - len(active_ids)
        if len(uncovered) > available:
            saturated_windows += 1
            uncovered = _farthest_sample(uncovered, available)
        new_ids = np.arange(
            len(coordinate_columns),
            len(coordinate_columns) + len(uncovered),
            dtype=np.int64,
        )
        for point in uncovered:
            coordinate_columns.append(np.repeat(point[None], frame_count, axis=0))
            confidence_columns.append(np.zeros(frame_count, dtype=np.float32))
            birth_frames.append(boundary)
        admitted += len(uncovered)

        query_ids = np.concatenate((active_ids, new_ids))
        query_points = np.concatenate((active_points, uncovered))
        if not len(query_ids):
            continue
        window = frames_rgb[boundary : boundary + window_frames]
        if len(window) < window_frames:
            window = np.concatenate(
                (window, np.repeat(window[-1:], window_frames - len(window), axis=0))
            )
        result = tracker.track(window, query_points, native_size=support.frame_size)
        if result.coordinates.shape != (window_frames, len(query_ids), 2):
            raise ValueError("tracker returned the wrong rolling-window shape")
        runtime_seconds += result.runtime_seconds
        peak_memory = max(peak_memory, result.peak_gpu_memory_bytes)

        commit = min(advance_frames, frame_count - boundary)
        for local_index, track_id in enumerate(query_ids):
            coordinate_columns[int(track_id)][boundary : boundary + commit] = (
                result.coordinates[:commit, local_index]
            )
            confidence_columns[int(track_id)][boundary : boundary + commit] = (
                result.confidence[:commit, local_index]
            )
        next_index = min(advance_frames, window_frames - 1)
        active_ids = query_ids
        active_points = result.coordinates[next_index].astype(np.float32, copy=False)
        active_confidence = result.confidence[next_index].astype(np.float32, copy=False)

    if not coordinate_columns:
        raise ValueError("admission support produced no trajectory queries")
    coordinates = np.stack(coordinate_columns, axis=1).astype(np.float32, copy=False)
    confidence = np.stack(confidence_columns, axis=1).astype(np.float32, copy=False)
    births = np.asarray(birth_frames, dtype=np.int32)
    before_birth = np.arange(frame_count)[:, None] < births[None]
    confidence[before_birth] = 0.0
    return RecordedTrajectories(
        coordinates=coordinates,
        confidence=np.clip(confidence, 0.0, 1.0),
        birth_frames=births,
        source_frame_indices=support.source_frame_indices,
        fps=support.fps,
        frame_size=support.frame_size,
        provider=tracker.name,
        future_context_frames=tracker.future_context_frames,
        metadata={
            "runtime_seconds": runtime_seconds,
            "wall_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_memory,
            "query_source": "blob-sim calibration-only motion support",
            "query_stride_pixels": stride,
            "coverage_radius_pixels": coverage_radius,
            "rolling_advance_frames": advance_frames,
            "rolling_window_frames": window_frames,
            "maximum_active_tracks": max_active_tracks,
            "tracks_admitted": admitted,
            "tracks_retired_at_window_boundaries": retired,
            "capacity_limited_windows": saturated_windows,
            "visibility_threshold": visibility_threshold,
        },
    )


def _valid_points(
    points: np.ndarray,
    confidence: np.ndarray,
    *,
    width: int,
    height: int,
    threshold: float,
) -> np.ndarray:
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0.0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < height)
        & (confidence >= threshold)
    )


def _covered_sites(
    candidates: np.ndarray,
    points: np.ndarray,
    *,
    shape: tuple[int, int],
    radius: float,
) -> np.ndarray:
    if not len(points):
        return np.zeros(len(candidates), dtype=bool)
    height, width = shape
    occupancy = np.zeros(shape, dtype=np.uint8)
    rounded = np.rint(points).astype(np.int32)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    occupancy[rounded[valid, 1], rounded[valid, 0]] = 1
    distance = cv2.distanceTransform(1 - occupancy, cv2.DIST_L2, 5)
    x = candidates[:, 0].astype(np.int32)
    y = candidates[:, 1].astype(np.int32)
    return distance[y, x] <= radius


def _farthest_sample(points: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or not len(points):
        return np.empty((0, 2), dtype=np.float32)
    if len(points) <= limit:
        return points
    centre = np.mean(points, axis=0)
    selected = [int(np.argmin(np.sum((points - centre) ** 2, axis=1)))]
    minimum_distance = np.full(len(points), np.inf, dtype=np.float64)
    while len(selected) < limit:
        latest = points[selected[-1]]
        minimum_distance = np.minimum(
            minimum_distance,
            np.sum((points - latest) ** 2, axis=1),
        )
        minimum_distance[selected] = -1.0
        selected.append(int(np.argmax(minimum_distance)))
    return points[np.asarray(selected)]


__all__ = [
    "RecordedTrajectories",
    "WindowTracker",
    "WindowTracks",
    "record_rolling_trajectories",
]
