"""Repeatable component-level runtime measurements for dense trackers."""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .trackers import DenseTracker


def _distribution(values: Sequence[float]) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    return {
        "median_seconds": float(np.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
        "minimum_seconds": float(np.min(samples)),
    }


def benchmark_tracker(
    tracker: DenseTracker,
    frames: tuple[np.ndarray, ...],
    reference_index: int,
    inference_size: tuple[int, int],
    *,
    warmup: int,
    repeats: int,
    measure_cuda_memory: bool = False,
) -> dict[str, Any]:
    """Measure an already-loaded tracker without video decoding or annotations."""

    if warmup < 1 or repeats < 2:
        raise ValueError("runtime benchmark requires at least one warmup and two repeats")
    for _ in range(warmup):
        tracker.track(frames, reference_index, inference_size)

    torch = None
    if measure_cuda_memory:
        import torch as torch_module

        torch = torch_module
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    core_times = []
    end_to_end_times = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = tracker.track(frames, reference_index, inference_size)
        end_to_end_times.append(time.perf_counter() - started)
        core_times.append(result.runtime_seconds)
        del result

    peak_memory = None
    if torch is not None:
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())
    gc.collect()

    end_to_end = _distribution(end_to_end_times)
    median = end_to_end["median_seconds"]
    return {
        "warmup_runs": warmup,
        "measured_runs": repeats,
        "core_compute": _distribution(core_times),
        "end_to_end": end_to_end,
        "window_updates_per_second": 1.0 / median,
        "effective_input_frames_per_second": len(frames) / median,
        "peak_gpu_memory_bytes": peak_memory,
    }
