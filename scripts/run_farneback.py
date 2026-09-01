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
    return cv2.remap(
        flow,
        points[:, 0].astype(np.float32),
        points[:, 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).reshape(-1, 2)


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
    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    queries = QuerySet.load(args.queries)
    gray = [
        cv2.cvtColor(cv2.resize(frame.image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        for frame in frames
    ]
    coordinates = [queries.points.copy()]
    visibility = [np.ones(len(queries.points), dtype=np.float32)]
    current = queries.points.copy()

    started = time.perf_counter()
    for previous, image in zip(gray[:-1], gray[1:], strict=True):
        flow = cv2.calcOpticalFlowFarneback(previous, image, None, 0.5, 5, 15, 3, 5, 1.2, 0)
        current = current + _sample(flow, current)
        valid = np.isfinite(current).all(axis=1)
        valid &= (current[:, 0] >= 0) & (current[:, 0] < args.width)
        valid &= (current[:, 1] >= 0) & (current[:, 1] < args.height)
        coordinates.append(current.copy())
        visibility.append(valid.astype(np.float32))

    result = SparseTracks(
        np.stack(coordinates),
        np.stack(visibility),
        queries,
        time.perf_counter() - started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print({"frames": len(frames), "queries": len(queries.points), "seconds": result.runtime_seconds})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
