"""Build an annotation-free browser viewer for prerecorded point tracks."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import SparseTracks
from dtf_eval.sparse_viewer import write_neutral_track_viewer


def _video_timeline(path: Path, length: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("viewer media must report a positive frame rate")
    frame_numbers = []
    size = None
    try:
        for frame_number in range(length):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"video ends before track frame {frame_number}")
            if size is None:
                size = (image.shape[1], image.shape[0])
            frame_numbers.append(frame_number)
    finally:
        capture.release()
    assert size is not None
    timeline = np.arange(length, dtype=np.float64) / fps
    return np.asarray(frame_numbers), timeline, size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-file", required=True, help="Media URL relative to the viewer")
    parser.add_argument("--tracks", required=True, nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.tracks):
        parser.error("--labels must contain one value per track file")
    labels = args.labels or [path.stem for path in args.tracks]
    tracks = {
        label: SparseTracks.load(path)
        for label, path in zip(labels, args.tracks, strict=True)
    }
    frame_counts = {result.coordinates.shape[0] for result in tracks.values()}
    if len(frame_counts) != 1:
        parser.error("all track files must contain the same number of frames")
    frame_numbers, timestamps, size = _video_timeline(
        args.video, frame_counts.pop()
    )
    if args.archive is not None:
        frame_numbers, timestamps = CocoTrafficArchive(args.archive).timeline(
            args.start, len(frame_numbers)
        )
        timestamps -= timestamps[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_neutral_track_viewer(
        args.output,
        tracks,
        video_file=args.video_file,
        frame_numbers=frame_numbers,
        timestamps_seconds=timestamps,
        size=size,
    )
    print(f"Viewer: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
