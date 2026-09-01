"""Run official LocoTrack-Base on native-resolution traffic frames."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from models.locotrack_model import LocoTrack

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import QuerySet, SparseTracks


def _multiple_of_eight(value: int) -> int:
    return (value + 7) // 8 * 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--length", type=int, default=126)
    parser.add_argument("--query-chunk-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    queries = QuerySet.load(args.queries)
    height, width = frames[0].image.shape[:2]
    if any(frame.image.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all frames must have the same native resolution")
    if np.any(queries.points[:, 0] >= width) or np.any(queries.points[:, 1] >= height):
        raise ValueError("queries are outside the native frame")

    padded_height = _multiple_of_eight(height)
    padded_width = _multiple_of_eight(width)
    images = np.stack([frame.image[..., ::-1] for frame in frames]).copy()
    images = np.pad(
        images,
        ((0, 0), (0, padded_height - height), (0, padded_width - width), (0, 0)),
        mode="edge",
    )
    video = torch.from_numpy(images)[None].to(device=device, dtype=torch.bfloat16)
    video = video.div_(127.5).sub_(1.0)

    query_points = np.column_stack(
        [
            np.full(len(queries.points), queries.frame_index, dtype=np.float32),
            queries.points[:, 1],
            queries.points[:, 0],
        ]
    )
    query_tensor = torch.from_numpy(query_points)[None].to(device)

    model = LocoTrack(model_size="base")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("model."): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(
            video,
            query_tensor,
            query_chunk_size=args.query_chunk_size,
        )
    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started

    coordinates = result["tracks"][0].permute(1, 0, 2).float().cpu().numpy()
    occlusion = torch.sigmoid(result["occlusion"])
    expected_error = torch.sigmoid(result["expected_dist"])
    visibility = ((1.0 - occlusion) * (1.0 - expected_error))[0]
    visibility = visibility.permute(1, 0).float().cpu().numpy()
    if not np.isfinite(coordinates).all() or not np.isfinite(visibility).all():
        raise RuntimeError("LocoTrack produced non-finite native-resolution output")

    tracks = SparseTracks(
        coordinates,
        visibility,
        queries,
        runtime_seconds,
        torch.cuda.max_memory_allocated(device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tracks.save(args.output)
    print(
        {
            "frames": len(frames),
            "queries": len(queries.points),
            "native_size": [width, height],
            "model_size": [padded_width, padded_height],
            "seconds": runtime_seconds,
            "effective_input_fps": len(frames) / runtime_seconds,
            "peak_mib": tracks.peak_gpu_memory_bytes / 1024**2,
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
