"""Run official causal TAPNext++ on continuous native-coordinate traffic queries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dtf_eval.dataset import CocoTrafficArchive
from dtf_eval.letterbox import SquareLetterbox
from dtf_eval.sparse import QuerySet, SparseTracks


def _load_tapnext(checkpoint: Path, vendor: Path, device: torch.device):
    sys.path.insert(0, str(vendor))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    wrapper = TAPNextPP.from_checkpoint(
        checkpoint,
        device=device,
        half_precision=True,
        input_resolution=512,
    )
    return wrapper._model  # Official wrapper does not expose timestamped queries.


def _frame_tensor(image_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = image_bgr[..., ::-1].copy()
    tensor = torch.from_numpy(rgb).to(device, non_blocking=True).half()
    return tensor.div_(127.5).sub_(1.0).unsqueeze(0).unsqueeze(0)


def _query_tensor(
    points_native: np.ndarray,
    birth_frames: np.ndarray,
    letterbox: SquareLetterbox,
    device: torch.device,
) -> torch.Tensor:
    square_xy = letterbox.to_model_image(points_native)
    base_xy = square_xy * (256.0 / letterbox.model_size)
    queries = np.column_stack([birth_frames, base_xy[:, 1], base_xy[:, 0]])
    return torch.from_numpy(queries.astype(np.float32))[None].to(device)


def _native_output(tracks_yx: torch.Tensor, letterbox: SquareLetterbox) -> np.ndarray:
    base_xy = tracks_yx[0, 0].float().cpu().numpy()[:, ::-1].copy()
    square_xy = base_xy * (letterbox.model_size / 256.0)
    return letterbox.to_native(square_xy).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--vendor", type=Path, default=Path("vendor/tapnet"))
    parser.add_argument("--start", type=int, default=200)
    parser.add_argument("--length", type=int, default=137)
    parser.add_argument("--query-batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.query_batch_size < 1:
        raise ValueError("query batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("TAPNext++ benchmark requires CUDA")

    frames = CocoTrafficArchive(args.archive).clip(args.start, args.length)
    queries = QuerySet.load(args.queries)
    height, width = frames[0].image.shape[:2]
    if any(frame.image.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all frames must have the same native resolution")
    if np.any(queries.frame_indices < 0) or np.any(queries.frame_indices >= len(frames)):
        raise ValueError("query birth frame lies outside the clip")

    device = torch.device("cuda")
    letterbox = SquareLetterbox(width, height, 512)
    images = [letterbox.image(frame.image) for frame in frames]
    model = _load_tapnext(args.checkpoint, args.vendor.resolve(), device)

    point_count = len(queries.points)
    coordinates = np.full((len(frames), point_count, 2), np.nan, dtype=np.float32)
    visibility = np.zeros((len(frames), point_count), dtype=np.float32)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for batch_start in range(0, point_count, args.query_batch_size):
            batch_end = min(point_count, batch_start + args.query_batch_size)
            member = slice(batch_start, batch_end)
            query_tensor = _query_tensor(
                queries.points[member],
                queries.frame_indices[member],
                letterbox,
                device,
            )
            state = None
            for frame_index, image in enumerate(images):
                frame_tensor = _frame_tensor(image, device)
                if state is None:
                    tracks, _, visible_logits, state = model(
                        video=frame_tensor,
                        query_points=query_tensor,
                    )
                else:
                    tracks, _, visible_logits, state = model(
                        video=frame_tensor,
                        state=state,
                    )
                born = queries.frame_indices[member] <= frame_index
                batch_coordinates = _native_output(tracks, letterbox)
                batch_visibility = torch.sigmoid(
                    visible_logits[0, 0, :, 0]
                ).float().cpu().numpy()
                coordinates[frame_index, member][born] = batch_coordinates[born]
                visibility[frame_index, member][born] = batch_visibility[born]
            print(f"tracked {batch_end}/{point_count} points", flush=True)

    torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device)
    tracks = SparseTracks(
        coordinates,
        visibility,
        queries,
        runtime_seconds,
        peak_memory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tracks.save(args.output)
    metadata = {
        "tracker": "TAPNext++ 512",
        "frames": len(frames),
        "queries": point_count,
        "objects": int(np.unique(queries.track_ids).size),
        "native_size": [width, height],
        "query_lattice": "stride 8 in native coordinates",
        "model_input": [512, 512],
        "preprocess": "aspect-ratio-preserving letterbox",
        "query_batch_size": args.query_batch_size,
        "runtime_seconds": runtime_seconds,
        "effective_sequence_fps": len(frames) / runtime_seconds,
        "peak_gpu_memory_bytes": peak_memory,
        "caution": "Mask annotations score region retention, not exact point correspondence.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
