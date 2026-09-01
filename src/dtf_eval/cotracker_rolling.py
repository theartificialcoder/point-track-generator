"""Official CoTracker3-online adapter for bounded rolling windows."""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .rolling import WindowTracks


class RollingCoTracker:
    name = "cotracker3-online-rolling"

    def __init__(
        self,
        checkpoint: str | Path,
        tracker_root: str | Path,
        *,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CoTracker requested CUDA, but CUDA is unavailable")
        root = Path(tracker_root).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if not (root / "cotracker" / "predictor.py").is_file():
            raise ValueError(f"invalid CoTracker source root: {root}")
        if not checkpoint_path.is_file():
            raise ValueError(f"missing CoTracker checkpoint: {checkpoint_path}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        predictor_type = importlib.import_module(
            "cotracker.predictor"
        ).CoTrackerOnlinePredictor
        self.predictor = predictor_type(checkpoint=str(checkpoint_path)).to(self.device)
        self.predictor.eval()
        self.future_context_frames = int(self.predictor.step)

    @torch.inference_mode()
    def track(
        self,
        frames_rgb: np.ndarray,
        query_points: np.ndarray,
        *,
        native_size: tuple[int, int],
    ) -> WindowTracks:
        if len(frames_rgb) != 2 * self.predictor.step:
            raise ValueError("CoTracker rolling windows must contain exactly two model steps")
        input_height, input_width = frames_rgb.shape[1:3]
        native_width, native_height = native_size
        interpolation_height, interpolation_width = self.predictor.interp_shape
        query_values = np.column_stack(
            (
                np.zeros(len(query_points), dtype=np.float32),
                query_points
                * np.asarray(
                    [input_width / native_width, input_height / native_height],
                    dtype=np.float32,
                ),
            )
        )
        queries = torch.from_numpy(query_values[None]).to(self.device)
        queries[:, :, 1:] *= queries.new_tensor(
            [
                (interpolation_width - 1) / (input_width - 1),
                (interpolation_height - 1) / (input_height - 1),
            ]
        )
        video = (
            torch.from_numpy(frames_rgb)
            .permute(0, 3, 1, 2)
            .float()
            .to(self.device)
        )
        video = F.interpolate(
            video,
            tuple(self.predictor.interp_shape),
            mode="bilinear",
            align_corners=True,
        )[None]
        self.predictor.model.init_video_online_processing()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        tracks, visibility, confidence = self.predictor.model(
            video=video,
            queries=queries,
            iters=6,
            is_online=True,
        )[:3]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        runtime = time.perf_counter() - started
        coordinates = tracks[0].float().cpu().numpy()
        coordinates *= np.asarray(
            [
                native_width / (interpolation_width - 1),
                native_height / (interpolation_height - 1),
            ],
            dtype=np.float32,
        )
        score = (
            visibility[0] * confidence[0]
        ).float().clamp(0.0, 1.0).cpu().numpy()
        peak_memory = (
            int(torch.cuda.max_memory_allocated(self.device))
            if self.device.type == "cuda"
            else 0
        )
        return WindowTracks(coordinates, score, runtime, peak_memory)


__all__ = ["RollingCoTracker"]
