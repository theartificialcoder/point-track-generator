"""Run official Online BootsTAPIR in the common sparse-track format."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tree
from tapnet.torch import tapir_model

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.sparse import QuerySet, SparseTracks


def _visible(occlusion: torch.Tensor, expected_distance: torch.Tensor) -> torch.Tensor:
    probability = (1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected_distance))
    return probability > 0.5


def _features(
    model: tapir_model.TAPIR,
    frames: torch.Tensor,
    query_points: torch.Tensor,
):
    frames = frames.float().div(127.5).sub(1)
    grids = model.get_feature_grids(frames, is_training=False)
    return model.get_query_features(
        frames,
        is_training=False,
        query_points=query_points,
        feature_grids=grids,
    )


def _predict(
    model: tapir_model.TAPIR,
    frame: torch.Tensor,
    query_features,
    causal_state,
):
    frame = frame.float().div(127.5).sub(1)
    grids = model.get_feature_grids(frame, is_training=False)
    result = model.estimate_trajectories(
        frame.shape[-3:-1],
        is_training=False,
        feature_grids=grids,
        query_features=query_features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_state,
        get_causal_context=True,
    )
    causal_state = result.pop("causal_context")
    tracks = result["tracks"][-1]
    visibility = _visible(result["occlusion"][-1], result["expected_dist"][-1])
    return tracks, visibility, causal_state


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
    parser.add_argument("--model-width", type=int, default=256)
    parser.add_argument("--model-height", type=int, default=256)
    args = parser.parse_args()

    if args.model_width % 8 or args.model_height % 8:
        raise ValueError("BootsTAPIR model dimensions must be divisible by eight")

    device = torch.device("cuda")
    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    queries = QuerySet.load(args.queries)
    images = np.stack(
        [
            cv2.resize(frame.image, (args.width, args.height), interpolation=cv2.INTER_AREA)[
                ..., ::-1
            ]
            for frame in frames
        ]
    ).copy()
    model_images = np.stack(
        [
            cv2.resize(image, (args.model_width, args.model_height), interpolation=cv2.INTER_AREA)
            for image in images
        ]
    )

    model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    query_values = np.column_stack(
        [
            np.zeros(len(queries.points), dtype=np.float32),
            queries.points[:, 1] * args.model_height / args.height,
            queries.points[:, 0] * args.model_width / args.width,
        ]
    )
    query_tensor = torch.from_numpy(query_values[None]).to(device)
    image_tensor = torch.from_numpy(model_images).to(device)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    init_started = time.perf_counter()
    with torch.inference_mode():
        query_features = _features(model, image_tensor[None, 0:1], query_tensor)
        causal_state = model.construct_initial_causal_state(
            len(queries.points), len(query_features.resolutions) - 1
        )
        causal_state = tree.map_structure(lambda value: value.to(device), causal_state)
    torch.cuda.synchronize(device)
    initialization_seconds = time.perf_counter() - init_started

    predictions: list[np.ndarray] = []
    visibilities: list[np.ndarray] = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(len(image_tensor)):
            tracks, visible, causal_state = _predict(
                model,
                image_tensor[None, index : index + 1],
                query_features,
                causal_state,
            )
            coordinates = tracks[0, :, -1].detach().cpu().numpy()
            coordinates[:, 0] *= args.width / args.model_width
            coordinates[:, 1] *= args.height / args.model_height
            predictions.append(coordinates)
            visibilities.append(visible[0, :, -1].float().cpu().numpy())
    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started

    result = SparseTracks(
        np.stack(predictions),
        np.stack(visibilities),
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
            "initialization_seconds": initialization_seconds,
            "tracking_seconds": runtime_seconds,
            "effective_input_fps": len(frames) / runtime_seconds,
            "peak_mib": result.peak_gpu_memory_bytes / 1024**2,
            "model_canvas": [args.model_width, args.model_height],
            "output": str(args.output),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
