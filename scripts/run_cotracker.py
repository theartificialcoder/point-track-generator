"""Run official CoTracker3-online and write the common sparse-track format."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from cotracker.predictor import CoTrackerOnlinePredictor

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
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=216)
    parser.add_argument("--model-width", type=int)
    parser.add_argument("--model-height", type=int)
    args = parser.parse_args()

    model_width = args.model_width or args.width
    model_height = args.model_height or args.height
    model_size = (model_width, model_height)
    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    queries = QuerySet.load(args.queries)
    images = np.stack(
        [
            cv2.resize(frame.image, model_size, interpolation=cv2.INTER_AREA)[..., ::-1]
            for frame in frames
        ]
    ).copy()

    device = torch.device("cuda")
    model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint).to(device)
    query_values = np.column_stack(
        [
            queries.frame_indices.astype(np.float32),
            queries.points
            * np.asarray(
                [model_width / args.width, model_height / args.height], dtype=np.float32
            ),
        ]
    )
    query_tensor = torch.from_numpy(query_values[None]).to(device)
    model(
        torch.empty((1, 1, 3, model_height, model_width), device=device),
        is_first_step=True,
        queries=query_tensor,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    tracks = visibility = None
    padded_length = int(np.ceil(len(images) / model.step) * model.step)
    if padded_length > len(images):
        padding = np.repeat(images[-1:], padded_length - len(images), axis=0)
        images = np.concatenate([images, padding])
    for begin in range(0, len(images) - model.step, model.step):
        chunk = images[begin : begin + model.step * 2]
        video = torch.from_numpy(chunk).permute(0, 3, 1, 2).float()[None].to(device)
        tracks, visibility = model(video)
    torch.cuda.synchronize(device)
    if tracks is None or visibility is None:
        raise RuntimeError("CoTracker produced no tracks")

    coordinates = tracks[0, : len(frames), : len(queries.points)].cpu().numpy()
    coordinates *= np.asarray([args.width / model_width, args.height / model_height])
    result = SparseTracks(
        coordinates,
        visibility[0, : len(frames), : len(queries.points)].float().cpu().numpy(),
        queries,
        time.perf_counter() - started,
        torch.cuda.max_memory_allocated(device),
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
