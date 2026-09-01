"""Continuous rolling-tracker qualification on reviewed traffic masks."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .admission import AdmissionSupport
from .cotracker_rolling import RollingCoTracker
from .dataset import CocoTrafficArchive, Frame
from .rolling import RecordedTrajectories, record_rolling_trajectories
from .sparse import QuerySet, SparseTracks, score_region_retention


def qualify_rolling_cotracker(
    archive_path: str | Path,
    output_directory: str | Path,
    *,
    checkpoint: str | Path,
    tracker_root: str | Path,
    start: int,
    length: int,
    device: str,
    tracker_size: tuple[int, int],
    stride: int,
    coverage_radius: float,
    advance_frames: int,
    max_active_tracks: int,
    visibility_threshold: float,
) -> dict[str, object]:
    """Run rolling tracking with annotation-only admission and hidden scoring."""

    frames = CocoTrafficArchive(archive_path).clip(start, length)
    support = _oracle_support(frames)
    tracker_frames = np.stack(
        [
            cv2.resize(frame.image, tracker_size, interpolation=cv2.INTER_AREA)[..., ::-1]
            for frame in frames
        ]
    ).copy()
    tracker = RollingCoTracker(checkpoint, tracker_root, device=device)
    recording = record_rolling_trajectories(
        tracker_frames,
        support,
        tracker,
        stride=stride,
        coverage_radius=coverage_radius,
        advance_frames=advance_frames,
        window_frames=2 * tracker.future_context_frames,
        max_active_tracks=max_active_tracks,
        visibility_threshold=visibility_threshold,
    )

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    recording_path = destination / "trajectories.npz"
    recording.save(recording_path, visibility_threshold=visibility_threshold)
    report = score_annotated_recording(
        recording,
        frames,
        visibility_threshold=visibility_threshold,
    )
    report.update(
        {
            "methodology": {
                "task": "Continuous rolling point tracking with new objects admitted throughout the clip.",
                "admission": "Reviewed masks provide an oracle support upper bound for this qualification only.",
                "scoring": "Annotations assign birth regions and later same-object membership; they never correct tracks.",
                "ground_truth_limit": "Masks do not provide exact physical point trajectories or valid occlusion labels.",
            },
            "clip": {
                "start": start,
                "length": length,
                "source_frames": [frames[0].frame_number, frames[-1].frame_number],
                "native_size": list(support.frame_size),
                "tracker_input_size": list(tracker_size),
            },
            "recording": recording.metadata,
            "trajectory_archive": str(recording_path.resolve()),
        }
    )
    report_path = destination / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def score_annotated_recording(
    recording: RecordedTrajectories,
    frames: tuple[Frame, ...],
    *,
    visibility_threshold: float,
) -> dict[str, object]:
    """Score a neutral recording without modifying its coordinates or confidence."""

    if len(frames) != recording.coordinates.shape[0]:
        raise ValueError("annotation and trajectory frame counts differ")
    labels, categories = _birth_labels(recording, frames)
    labelled = labels >= 0
    if not np.any(labelled):
        raise ValueError("no recorded trajectory was born inside one labelled object")
    queries = QuerySet(
        points=recording.coordinates[recording.birth_frames, np.arange(len(labels))][labelled],
        track_ids=labels[labelled],
        categories=categories[labelled],
        birth_frames=recording.birth_frames[labelled],
    )
    tracks = SparseTracks(
        coordinates=recording.coordinates[:, labelled],
        visibility=recording.confidence[:, labelled],
        queries=queries,
        runtime_seconds=float(recording.metadata.get("runtime_seconds", 0.0)),
        peak_gpu_memory_bytes=int(recording.metadata.get("peak_gpu_memory_bytes", 0)),
    )
    thresholds = sorted({0.3, 0.5, visibility_threshold, 0.7, 0.9})
    return {
        "birth_assignment": {
            "tracks": int(len(labels)),
            "uniquely_labelled": int(np.count_nonzero(labelled)),
            "fraction": float(np.mean(labelled)),
        },
        "object_coverage": _object_coverage(
            recording,
            frames,
            labels,
            visibility_threshold=visibility_threshold,
        ),
        "region_retention_by_confidence": {
            f"{threshold:.2f}": score_region_retention(
                tracks,
                frames,
                recording.frame_size,
                visibility_level=threshold,
            )
            for threshold in thresholds
        },
    }


def _oracle_support(frames: tuple[Frame, ...]) -> AdmissionSupport:
    height, width = frames[0].image.shape[:2]
    probability = np.zeros((len(frames), height, width), dtype=np.uint8)
    for frame_index, frame in enumerate(frames):
        for instance in frame.instances:
            if instance.track_id is not None:
                probability[frame_index, instance.mask] = 255
    timestamps = [frame.timestamp_seconds for frame in frames]
    span = None if None in timestamps else float(timestamps[-1] - timestamps[0])
    fps = (len(frames) - 1) / span if span and span > 0.0 else 30.0
    return AdmissionSupport(
        probability,
        np.asarray([frame.frame_number for frame in frames], dtype=np.int64),
        fps,
        (width, height),
        {
            "point_trajectory_authority": False,
            "learned_mask_authority": False,
            "body_dynamics_authority": False,
            "qualification_only": True,
        },
    )


def _birth_labels(
    recording: RecordedTrajectories,
    frames: tuple[Frame, ...],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(recording.birth_frames)
    labels = np.full(count, -1, dtype=np.int64)
    categories = np.full(count, "unassigned", dtype="<U32")
    for index, birth in enumerate(recording.birth_frames):
        x, y = np.rint(recording.coordinates[int(birth), index]).astype(np.int64)
        if not (0 <= x < recording.frame_size[0] and 0 <= y < recording.frame_size[1]):
            continue
        matches = [
            instance
            for instance in frames[int(birth)].instances
            if instance.track_id is not None and instance.mask[y, x]
        ]
        if len(matches) == 1:
            labels[index] = int(matches[0].track_id)
            categories[index] = matches[0].category
    return labels, categories


def _object_coverage(
    recording: RecordedTrajectories,
    frames: tuple[Frame, ...],
    labels: np.ndarray,
    *,
    visibility_threshold: float,
) -> dict[str, float | int]:
    first_present: dict[int, int] = {}
    first_admitted: dict[int, int] = {}
    covered = eligible = 0
    for frame_index, frame in enumerate(frames):
        for instance in frame.instances:
            if instance.track_id is None:
                continue
            object_id = int(instance.track_id)
            first_present.setdefault(object_id, frame_index)
            members = np.flatnonzero(labels == object_id)
            born = members[recording.birth_frames[members] <= frame_index]
            if len(born):
                first_admitted.setdefault(object_id, int(recording.birth_frames[born].min()))
            eligible += 1
            if not len(born):
                continue
            points = np.rint(recording.coordinates[frame_index, born]).astype(np.int64)
            valid = recording.confidence[frame_index, born] >= visibility_threshold
            valid &= (points[:, 0] >= 0) & (points[:, 0] < recording.frame_size[0])
            valid &= (points[:, 1] >= 0) & (points[:, 1] < recording.frame_size[1])
            sampled = instance.mask[
                points[:, 1].clip(0, recording.frame_size[1] - 1),
                points[:, 0].clip(0, recording.frame_size[0] - 1),
            ]
            if np.any(valid & sampled):
                covered += 1
    delays = [first_admitted[key] - value for key, value in first_present.items() if key in first_admitted]
    return {
        "annotated_objects": len(first_present),
        "objects_admitted": len(first_admitted),
        "object_frame_coverage": covered / max(1, eligible),
        "admission_delay_median_frames": float(np.median(delays)) if delays else 0.0,
        "admission_delay_p90_frames": float(np.percentile(delays, 90)) if delays else 0.0,
    }


__all__ = ["qualify_rolling_cotracker", "score_annotated_recording"]
