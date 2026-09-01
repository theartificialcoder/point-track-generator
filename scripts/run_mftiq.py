"""Run official MFTIQ+RAFT and write the common sparse-track format."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import QuerySet, SparseTracks


def _numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--mftiq-root", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--length", type=int, default=126)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=216)
    args = parser.parse_args()

    os.chdir(args.mftiq_root)
    from MFTIQ.config import load_config
    from MFTIQ.point_tracking import convert_to_point_tracking

    size = (args.width, args.height)
    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    queries = QuerySet.load(args.queries)
    images = [cv2.resize(frame.image, size, interpolation=cv2.INTER_AREA) for frame in frames]

    config = load_config("configs/MFTIQ4_RAFT_200k_cfg.py")
    config.timers_enabled = False
    model = config.tracker_class(config)
    query_tensor = torch.from_numpy(queries.points).float().cuda()
    coordinates: list[np.ndarray] = [queries.points.copy()]
    visibility: list[np.ndarray] = [np.ones(len(queries.points), dtype=np.float32)]

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.init(images[0])
    for image in images[1:]:
        metadata = model.track(image)
        points, occlusion = convert_to_point_tracking(metadata.result, query_tensor)
        coordinates.append(_numpy(points))
        visibility.append(1.0 - _numpy(occlusion).astype(np.float32))
    torch.cuda.synchronize()

    result = SparseTracks(
        np.stack(coordinates),
        np.stack(visibility),
        queries,
        time.perf_counter() - started,
        torch.cuda.max_memory_allocated(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(
        {
            "frames": len(frames),
            "queries": len(queries.points),
            "seconds": result.runtime_seconds,
            "peak_mib": result.peak_gpu_memory_bytes / 1024**2,
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
