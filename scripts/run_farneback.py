"""Run chained Farneback as a common-format sparse correspondence baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import QuerySet, SparseTracks


def _sample(flow: np.ndarray, points: np.ndarray) -> np.ndarray:
    samples = []
    for start in range(0, len(points), 32_000):
        batch = points[start : start + 32_000]
        samples.append(
            cv2.remap(
                flow,
                batch[:, 0].astype(np.float32),
                batch[:, 1].astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=np.nan,
            ).reshape(-1, 2)
        )
    return np.concatenate(samples) if samples else np.empty((0, 2), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--length", type=int, default=126)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=216)
    args = parser.parse_args()

    size = (args.width, args.height)
    archive = CocoTrafficArchive(args.archive)
    queries = QuerySet.load(args.queries)
    gray = [
        cv2.cvtColor(
            cv2.resize(archive.image(index), size, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        for index in range(args.start, args.start + args.length)
    ]
    point_count = len(queries.points)
    births = queries.frame_indices
    coordinates = np.full((args.length, point_count, 2), np.nan, dtype=np.float32)
    visibility = np.zeros((args.length, point_count), dtype=np.float32)
    born = births == 0
    coordinates[0, born] = queries.points[born]
    visibility[0, born] = 1.0

    started = time.perf_counter()
    for frame_index, (previous, image) in enumerate(
        zip(gray[:-1], gray[1:], strict=True), start=1
    ):
        flow = cv2.calcOpticalFlowFarneback(previous, image, None, 0.5, 5, 15, 3, 5, 1.2, 0)
        carried = births < frame_index
        previous_points = coordinates[frame_index - 1, carried]
        coordinates[frame_index, carried] = previous_points + _sample(flow, previous_points)
        born = births == frame_index
        coordinates[frame_index, born] = queries.points[born]
        current = coordinates[frame_index]
        valid = births <= frame_index
        valid &= np.isfinite(current).all(axis=1)
        valid &= (current[:, 0] >= 0) & (current[:, 0] < args.width)
        valid &= (current[:, 1] >= 0) & (current[:, 1] < args.height)
        visibility[frame_index] = valid.astype(np.float32)

    result = SparseTracks(
        coordinates,
        visibility,
        queries,
        time.perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(
        {
            "frames": args.length,
            "queries": len(queries.points),
            "seconds": result.runtime_seconds,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
