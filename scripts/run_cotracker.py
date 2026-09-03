"""Run official CoTracker3 and write the common sparse-track format."""

from __future__ import annotations

import argparse
import gc
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor

from dtf_eval.batch_archive import BatchArchive
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
    parser.add_argument("--query-batch-size", type=int)
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    args = parser.parse_args()

    if args.query_batch_size is not None and args.query_batch_size < 1:
        raise ValueError("query batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CoTracker benchmark requires CUDA")

    model_width = args.model_width or args.width
    model_height = args.model_height or args.height
    model_size = (model_width, model_height)
    archive = CocoTrafficArchive(args.archive)
    queries = QuerySet.load(args.queries)
    if np.any(queries.frame_indices < 0) or np.any(queries.frame_indices >= args.length):
        raise ValueError("query birth frame lies outside the clip")
    point_count = len(queries.points)
    batch_size = args.query_batch_size or point_count
    identity = {
        "archive": str(args.archive.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "queries_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        "start": args.start,
        "length": args.length,
        "size": [args.width, args.height],
        "model_size": [model_width, model_height],
        "mode": args.mode,
    }
    partial = BatchArchive.open(
        args.output.with_suffix(args.output.suffix + ".partial"),
        frame_count=args.length,
        point_count=point_count,
        identity=identity,
    )
    scale_to_model = np.asarray(
        [model_width / args.width, model_height / args.height], dtype=np.float32
    )
    scale_to_native = np.asarray(
        [args.width / model_width, args.height / model_height], dtype=np.float32
    )

    if partial.next_query < point_count:
        images = np.stack(
            [
                cv2.resize(
                    archive.image(index), model_size, interpolation=cv2.INTER_AREA
                )[..., ::-1]
                for index in range(args.start, args.start + args.length)
            ]
        ).copy()
        device = torch.device("cuda")
        model = (
            CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
            if args.mode == "online"
            else CoTrackerPredictor(checkpoint=args.checkpoint, offline=True)
        ).to(device)
        if args.mode == "online":
            padded_length = int(np.ceil(len(images) / model.step) * model.step)
            if padded_length > len(images):
                padding = np.repeat(images[-1:], padded_length - len(images), axis=0)
                images = np.concatenate([images, padding])

    for batch_start in range(partial.next_query, point_count, batch_size):
        batch_end = min(point_count, batch_start + batch_size)
        member = slice(batch_start, batch_end)
        query_values = np.column_stack(
            [
                queries.frame_indices[member].astype(np.float32),
                queries.points[member] * scale_to_model,
            ]
        )
        query_tensor = torch.from_numpy(query_values[None]).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        if args.mode == "online":
            model(
                torch.empty((1, 1, 3, model_height, model_width), device=device),
                is_first_step=True,
                queries=query_tensor,
            )
            tracks = visibility = None
            for begin in range(0, len(images) - model.step, model.step):
                chunk = images[begin : begin + model.step * 2]
                video = torch.from_numpy(chunk).permute(0, 3, 1, 2).float()[None].to(device)
                tracks, visibility = model(video)
            if tracks is None or visibility is None:
                raise RuntimeError("CoTracker produced no tracks")
        else:
            video = torch.from_numpy(images).permute(0, 3, 1, 2).float()[None].to(device)
            tracks, visibility = model(video, queries=query_tensor)
        batch_coordinates = (
            tracks[0, : args.length, : batch_end - batch_start].cpu().numpy()
            * scale_to_native
        )
        batch_visibility = visibility[
            0, : args.length, : batch_end - batch_start
        ].float().cpu().numpy()
        torch.cuda.synchronize(device)
        partial.write(
            batch_start,
            batch_coordinates,
            batch_visibility,
            runtime_seconds=time.perf_counter() - started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),
        )
        print(f"tracked {batch_end}/{point_count} points", flush=True)
        del batch_coordinates, batch_visibility, query_tensor, tracks, visibility
        del video
        gc.collect()
        torch.cuda.empty_cache()

    result = SparseTracks(
        np.asarray(partial.coordinates),
        np.asarray(partial.visibility),
        queries,
        float(partial.metadata["runtime_seconds"]),
        int(partial.metadata["peak_gpu_memory_bytes"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    print(
        {
            "frames": args.length,
            "queries": len(queries.points),
            "query_batch_size": batch_size,
            "mode": args.mode,
            "seconds": result.runtime_seconds,
            "peak_mib": result.peak_gpu_memory_bytes / 1024**2,
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
