"""Run official AllTracker at native resolution in the common track format."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from nets.alltracker import Net

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import QuerySet, SparseTracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--length", type=int, default=126)
    parser.add_argument("--window-length", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda")
    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    queries = QuerySet.load(args.queries)
    height, width = frames[0].image.shape[:2]
    if any(frame.image.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all frames must have the same native resolution")
    if np.any(queries.points[:, 0] >= width) or np.any(queries.points[:, 1] >= height):
        raise ValueError("queries are outside the native frame")

    images = np.stack([frame.image[..., ::-1] for frame in frames]).copy()
    video = torch.from_numpy(images).permute(0, 3, 1, 2).float()[None]

    model = Net(args.window_length)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        flows, visibilities, _, _ = model.forward_sliding(
            video,
            iters=args.iterations,
            window_len=args.window_length,
            is_training=False,
        )
    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started

    x = torch.from_numpy(queries.points[:, 0].round().astype(np.int64))
    y = torch.from_numpy(queries.points[:, 1].round().astype(np.int64))
    sampled_flow = flows[0, :, :, y, x].permute(0, 2, 1)
    origin = torch.from_numpy(queries.points).unsqueeze(0)
    coordinates = (sampled_flow + origin).numpy()
    confidence = (
        visibilities[0, :, 0, y, x] * visibilities[0, :, 1, y, x]
    ).numpy()

    result = SparseTracks(
        coordinates,
        confidence,
        queries,
        runtime_seconds,
        torch.cuda.max_memory_allocated(device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(
        {
            "frames": len(frames),
            "queries": len(queries.points),
            "native_size": [width, height],
            "seconds": runtime_seconds,
            "effective_input_fps": len(frames) / runtime_seconds,
            "peak_mib": result.peak_gpu_memory_bytes / 1024**2,
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
