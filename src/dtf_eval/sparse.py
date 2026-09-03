"""Common query and scoring types for sparse correspondence trackers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .dataset import Frame


@dataclass(frozen=True, slots=True)
class QuerySet:
    points: np.ndarray
    track_ids: np.ndarray
    categories: np.ndarray
    frame_index: int = 0
    birth_frames: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = len(self.points)
        if self.points.shape != (count, 2):
            raise ValueError("query points must have shape (N, 2)")
        if self.track_ids.shape != (count,) or self.categories.shape != (count,):
            raise ValueError("query metadata length mismatch")
        if self.birth_frames is not None and self.birth_frames.shape != (count,):
            raise ValueError("query birth-frame length mismatch")

    @property
    def frame_indices(self) -> np.ndarray:
        if self.birth_frames is not None:
            return self.birth_frames.astype(np.int64, copy=False)
        return np.full(len(self.points), self.frame_index, dtype=np.int64)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            query_points=self.points.astype(np.float32),
            query_track_ids=self.track_ids.astype(np.int64),
            query_categories=self.categories.astype(str),
            query_frame_index=np.int64(self.frame_index),
            query_birth_frames=self.frame_indices,
        )

    @classmethod
    def load(cls, path: str | Path) -> QuerySet:
        with np.load(path) as data:
            birth_frames = (
                data["query_birth_frames"]
                if "query_birth_frames" in data.files
                else None
            )
            return cls(
                data["query_points"],
                data["query_track_ids"],
                data["query_categories"],
                int(data["query_frame_index"]),
                birth_frames,
            )


@dataclass(frozen=True, slots=True)
class SparseTracks:
    coordinates: np.ndarray
    visibility: np.ndarray
    queries: QuerySet
    runtime_seconds: float
    peak_gpu_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 3 or self.coordinates.shape[-1] != 2:
            raise ValueError("coordinates must have shape (T, N, 2)")
        if self.visibility.shape != self.coordinates.shape[:2]:
            raise ValueError("visibility shape mismatch")
        if self.coordinates.shape[1] != len(self.queries.points):
            raise ValueError("track/query count mismatch")

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            coordinates=self.coordinates.astype(np.float32),
            visibility=self.visibility.astype(np.float32),
            query_points=self.queries.points.astype(np.float32),
            query_track_ids=self.queries.track_ids.astype(np.int64),
            query_categories=self.queries.categories.astype(str),
            query_frame_index=np.int64(self.queries.frame_index),
            query_birth_frames=self.queries.frame_indices,
            runtime_seconds=np.float64(self.runtime_seconds),
            peak_gpu_memory_bytes=np.int64(self.peak_gpu_memory_bytes or -1),
        )

    @classmethod
    def load(cls, path: str | Path) -> SparseTracks:
        with np.load(path) as data:
            birth_frames = (
                data["query_birth_frames"]
                if "query_birth_frames" in data.files
                else None
            )
            queries = QuerySet(
                data["query_points"],
                data["query_track_ids"],
                data["query_categories"],
                int(data["query_frame_index"]),
                birth_frames,
            )
            memory = int(data["peak_gpu_memory_bytes"])
            return cls(
                data["coordinates"],
                data["visibility"],
                queries,
                float(data["runtime_seconds"]),
                None if memory < 0 else memory,
            )


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)


def _spatial_sample(mask: np.ndarray, limit: int) -> np.ndarray:
    """Select repeatable, spatially distributed interior points."""

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    candidates = np.argwhere(distance >= 1.0)
    if not len(candidates):
        candidates = np.argwhere(mask)
    if not len(candidates):
        return np.empty((0, 2), dtype=np.float32)

    first = int(np.argmax(distance[candidates[:, 0], candidates[:, 1]]))
    selected = [first]
    minimum_distance = np.full(len(candidates), np.inf, dtype=np.float32)
    while len(selected) < min(limit, len(candidates)):
        latest = candidates[selected[-1]]
        squared = np.sum((candidates - latest) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, squared)
        minimum_distance[selected] = -1
        selected.append(int(np.argmax(minimum_distance)))
    points_yx = candidates[selected]
    return points_yx[:, ::-1].astype(np.float32)


def reference_queries(
    frame: Frame,
    size: tuple[int, int],
    *,
    max_points_per_object: int = 16,
) -> QuerySet:
    """Seed a balanced set of points from annotated object interiors."""

    points: list[np.ndarray] = []
    track_ids: list[int] = []
    categories: list[str] = []
    for instance in frame.instances:
        if instance.track_id is None:
            continue
        selected = _spatial_sample(_resize_mask(instance.mask, size), max_points_per_object)
        if not len(selected):
            continue
        points.append(selected)
        track_ids.extend([instance.track_id] * len(selected))
        categories.extend([instance.category] * len(selected))
    if not points:
        raise ValueError("reference frame has no trackable annotated points")
    return QuerySet(
        np.concatenate(points),
        np.asarray(track_ids, dtype=np.int64),
        np.asarray(categories),
    )


def strided_reference_queries(
    frame: Frame,
    size: tuple[int, int],
    *,
    stride: int,
) -> QuerySet:
    """Place a regular query lattice inside unambiguous annotated regions."""

    if stride < 1:
        raise ValueError("stride must be positive")
    instances = [instance for instance in frame.instances if instance.track_id is not None]
    masks = [_resize_mask(instance.mask, size) for instance in instances]
    ownership_count = np.sum(masks, axis=0) if masks else np.zeros(size[::-1], dtype=int)
    xs = np.arange(stride // 2, size[0], stride, dtype=np.int32)
    ys = np.arange(stride // 2, size[1], stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    points: list[np.ndarray] = []
    track_ids: list[int] = []
    categories: list[str] = []
    for instance, mask in zip(instances, masks, strict=True):
        selected = mask[grid_y, grid_x] & (ownership_count[grid_y, grid_x] == 1)
        object_points = np.column_stack([grid_x[selected], grid_y[selected]]).astype(np.float32)
        if not len(object_points):
            continue
        points.append(object_points)
        track_ids.extend([int(instance.track_id)] * len(object_points))
        categories.extend([instance.category] * len(object_points))
    if not points:
        raise ValueError("reference frame has no unambiguous trackable points")
    return QuerySet(
        np.concatenate(points),
        np.asarray(track_ids, dtype=np.int64),
        np.asarray(categories),
    )


def continuous_strided_queries(
    frames: tuple[Frame, ...],
    size: tuple[int, int],
    *,
    stride: int,
) -> QuerySet:
    """Maintain approximately stride-spaced query coverage as objects grow."""

    if stride < 1:
        raise ValueError("stride must be positive")
    xs = np.arange(stride // 2, size[0], stride, dtype=np.int32)
    ys = np.arange(stride // 2, size[1], stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    admitted_counts: dict[int, int] = {}
    points: list[np.ndarray] = []
    track_ids: list[int] = []
    categories: list[str] = []
    birth_frames: list[int] = []

    for frame_index, frame in enumerate(frames):
        instances = [instance for instance in frame.instances if instance.track_id is not None]
        all_masks = [
            _resize_mask(instance.mask, size)
            for instance in frame.instances
            if instance.track_id is not None
        ]
        ownership_count = (
            np.sum(all_masks, axis=0)
            if all_masks
            else np.zeros(size[::-1], dtype=np.int32)
        )
        for instance in instances:
            object_id = int(instance.track_id)
            mask = _resize_mask(instance.mask, size)
            unambiguous = mask & (ownership_count == 1)
            if not np.any(unambiguous):
                continue

            selected = unambiguous[grid_y, grid_x]
            lattice_points = np.column_stack([grid_x[selected], grid_y[selected]]).astype(
                np.float32
            )
            target_count = max(1, len(lattice_points))
            admitted = admitted_counts.get(object_id, 0)
            add_count = target_count - admitted
            if add_count <= 0:
                continue

            if len(lattice_points):
                candidate_mask = np.zeros_like(unambiguous)
                candidate_mask[
                    lattice_points[:, 1].astype(np.int32),
                    lattice_points[:, 0].astype(np.int32),
                ] = True
                object_points = _spatial_sample(candidate_mask, add_count)
            else:
                object_points = _spatial_sample(unambiguous, 1)
            if not len(object_points):
                continue

            admitted_counts[object_id] = admitted + len(object_points)
            points.append(object_points)
            count = len(object_points)
            track_ids.extend([object_id] * count)
            categories.extend([instance.category] * count)
            birth_frames.extend([frame_index] * count)

    if not points:
        raise ValueError("clip has no unambiguous trackable points")
    return QuerySet(
        np.concatenate(points),
        np.asarray(track_ids, dtype=np.int64),
        np.asarray(categories),
        frame_index=0,
        birth_frames=np.asarray(birth_frames, dtype=np.int64),
    )


def score_region_retention(
    tracks: SparseTracks,
    frames: tuple[Frame, ...],
    size: tuple[int, int],
    *,
    visibility_level: float = 0.5,
) -> dict[str, Any]:
    """Score same-object retention without claiming exact point ground truth."""

    if len(frames) != tracks.coordinates.shape[0]:
        raise ValueError("frame/track length mismatch")
    count_keys = ("eligible", "visible", "same", "other", "background")
    totals = {key: 0 for key in count_keys}
    birth_frames = tracks.queries.frame_indices
    source_scale = _query_source_scales(tracks.queries, frames, size)
    totals_by_scale: dict[str, dict[str, int]] = {}
    totals_by_object: dict[int, dict[str, int]] = {}
    requested_ages = {age for age in (0, 1, 8, 16, 30, 60, 120, len(frames) - 1) if age >= 0}
    totals_by_age = {
        age: {key: 0 for key in count_keys}
        for age in sorted(requested_ages)
    }

    for frame_index, frame in enumerate(frames):
        by_id = {
            instance.track_id: _resize_mask(instance.mask, size)
            for instance in frame.instances
            if instance.track_id is not None
        }
        union = np.zeros((size[1], size[0]), dtype=bool)
        for mask in by_id.values():
            union |= mask
        coordinates = tracks.coordinates[frame_index]
        inside = np.isfinite(coordinates).all(axis=1)
        safe_coordinates = np.where(np.isfinite(coordinates), coordinates, 0)
        rounded = np.rint(safe_coordinates).astype(np.int32)
        inside &= (rounded[:, 0] >= 0) & (rounded[:, 0] < size[0])
        inside &= (rounded[:, 1] >= 0) & (rounded[:, 1] < size[1])
        visible = (tracks.visibility[frame_index] >= visibility_level) & inside

        for query_index, track_id in enumerate(tracks.queries.track_ids):
            object_id = int(track_id)
            age = frame_index - int(birth_frames[query_index])
            if age < 0:
                continue
            target = by_id.get(object_id)
            if target is None:
                continue
            age_totals = totals_by_age.get(age)
            totals["eligible"] += 1
            if age_totals is not None:
                age_totals["eligible"] += 1
            object_totals = totals_by_object.setdefault(
                object_id, {key: 0 for key in count_keys}
            )
            object_totals["eligible"] += 1
            scale_totals = totals_by_scale.setdefault(
                source_scale[query_index], {key: 0 for key in count_keys}
            )
            scale_totals["eligible"] += 1
            if not visible[query_index]:
                continue
            totals["visible"] += 1
            if age_totals is not None:
                age_totals["visible"] += 1
            object_totals["visible"] += 1
            scale_totals["visible"] += 1
            x, y = rounded[query_index]
            if target[y, x]:
                outcome = "same"
                object_totals["same"] += 1
                scale_totals["same"] += 1
            elif union[y, x]:
                outcome = "other"
                object_totals["other"] += 1
                scale_totals["other"] += 1
            else:
                outcome = "background"
                object_totals["background"] += 1
                scale_totals["background"] += 1
            totals[outcome] += 1
            if age_totals is not None:
                age_totals[outcome] += 1

    return {
        **_rates(totals),
        "object_balanced": _mean_rates(totals_by_object.values()),
        "by_source_scale": {
            scale: _rates(counts) for scale, counts in sorted(totals_by_scale.items())
        },
        "track_age_frames": {
            str(age): _rates(counts)
            for age, counts in totals_by_age.items()
            if counts["eligible"]
        },
    }


def _query_source_scales(
    queries: QuerySet,
    frames: tuple[Frame, ...],
    size: tuple[int, int],
) -> list[str]:
    """Return each query's object scale at that query's birth frame."""

    scales: list[str] = []
    cache: dict[tuple[int, int], str] = {}
    for track_id, birth_frame in zip(
        queries.track_ids, queries.frame_indices, strict=True
    ):
        if not 0 <= birth_frame < len(frames):
            raise ValueError("query birth frame lies outside the scored clip")
        key = (int(birth_frame), int(track_id))
        if key in cache:
            scales.append(cache[key])
            continue
        instance = next(
            (
                item
                for item in frames[int(birth_frame)].instances
                if item.track_id == int(track_id)
            ),
            None,
        )
        if instance is None:
            raise ValueError("query source object is absent at its birth frame")
        scale = _scale_name(int(_resize_mask(instance.mask, size).sum()))
        cache[key] = scale
        scales.append(scale)
    return scales


def score_object_frame_coverage(
    tracks: SparseTracks,
    frames: tuple[Frame, ...],
    size: tuple[int, int],
    *,
    visibility_level: float = 0.5,
) -> dict[str, float | int]:
    """Measure whether each annotated object-frame has one retained point."""

    if len(frames) != tracks.coordinates.shape[0]:
        raise ValueError("frame/track length mismatch")
    covered = total = 0
    births = tracks.queries.frame_indices
    for frame_index, frame in enumerate(frames):
        coordinates = tracks.coordinates[frame_index]
        rounded = np.rint(np.where(np.isfinite(coordinates), coordinates, 0)).astype(np.int32)
        valid = (tracks.visibility[frame_index] >= visibility_level) & (births <= frame_index)
        valid &= np.isfinite(coordinates).all(axis=1)
        valid &= (rounded[:, 0] >= 0) & (rounded[:, 0] < size[0])
        valid &= (rounded[:, 1] >= 0) & (rounded[:, 1] < size[1])
        for instance in frame.instances:
            if instance.track_id is None:
                continue
            total += 1
            members = valid & (tracks.queries.track_ids == int(instance.track_id))
            if not np.any(members):
                continue
            mask = _resize_mask(instance.mask, size)
            points = rounded[members]
            if np.any(mask[points[:, 1], points[:, 0]]):
                covered += 1
    return {
        "annotated_object_frames": total,
        "covered_object_frames": covered,
        "coverage_rate": covered / max(1, total),
    }


def _scale_name(area: int) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def _rates(counts: dict[str, int]) -> dict[str, float | int]:
    eligible = max(1, counts["eligible"])
    visible = max(1, counts["visible"])
    return {
        **counts,
        "visibility_rate": counts["visible"] / eligible,
        "same_object_recall": counts["same"] / eligible,
        "same_object_precision": counts["same"] / visible,
        "identity_leak_rate": counts["other"] / visible,
        "background_leak_rate": counts["background"] / visible,
    }


def _mean_rates(objects: Any) -> dict[str, float | int]:
    rates = [_rates(counts) for counts in objects]
    rate_keys = (
        "visibility_rate",
        "same_object_recall",
        "same_object_precision",
        "identity_leak_rate",
        "background_leak_rate",
    )
    return {
        "object_count": len(rates),
        **{
            key: float(np.mean([float(item[key]) for item in rates])) if rates else 0.0
            for key in rate_keys
        },
    }
