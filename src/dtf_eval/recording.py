"""I/O orchestration for rolling point-trajectory recording."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .admission import AdmissionSupport
from .cotracker_rolling import RollingCoTracker
from .rolling import RecordedTrajectories, record_rolling_trajectories


def record_cotracker(
    video: str | Path,
    support_path: str | Path,
    output: str | Path,
    *,
    checkpoint: str | Path,
    tracker_root: str | Path,
    device: str,
    tracker_size: tuple[int, int],
    stride: int,
    coverage_radius: float,
    advance_frames: int,
    max_active_tracks: int,
    visibility_threshold: float,
) -> RecordedTrajectories:
    support = AdmissionSupport.load(support_path)
    frames = _decode_tracker_frames(video, support, tracker_size)
    tracker = RollingCoTracker(checkpoint, tracker_root, device=device)
    result = record_rolling_trajectories(
        frames,
        support,
        tracker,
        stride=stride,
        coverage_radius=coverage_radius,
        advance_frames=advance_frames,
        window_frames=2 * tracker.future_context_frames,
        max_active_tracks=max_active_tracks,
        visibility_threshold=visibility_threshold,
    )
    destination = Path(output)
    result.save(destination, visibility_threshold=visibility_threshold)
    report = {
        "scope": "rolling-reset continuity audit; not qualified as simulator input",
        "video": str(Path(video).resolve()),
        "support": str(Path(support_path).resolve()),
        "output": str(destination.resolve()),
        "frames": len(result.source_frame_indices),
        "tracks": len(result.birth_frames),
        "provider": result.provider,
        "future_context_frames": result.future_context_frames,
        "metadata": result.metadata,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return result


def _decode_tracker_frames(
    video: str | Path,
    support: AdmissionSupport,
    tracker_size: tuple[int, int],
) -> np.ndarray:
    source = cv2.VideoCapture(str(video))
    if not source.isOpened():
        raise ValueError(f"cannot open trajectory source video: {video}")
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != support.frame_size:
        source.release()
        raise ValueError("video and admission support geometry differ")
    source.set(cv2.CAP_PROP_POS_FRAMES, float(support.source_frame_indices[0]))
    frames: list[np.ndarray] = []
    for _ in support.source_frame_indices:
        ok, frame = source.read()
        if not ok:
            source.release()
            raise ValueError("video ended before the admission-support archive")
        frames.append(
            cv2.resize(frame, tracker_size, interpolation=cv2.INTER_AREA)[..., ::-1].copy()
        )
    source.release()
    return np.stack(frames)


__all__ = ["record_cotracker"]
