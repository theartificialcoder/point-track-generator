"""Media-timeline utilities for qualitative tracker reports."""

from __future__ import annotations

import numpy as np


def constant_rate_source_indices(timestamps: np.ndarray, fps: float) -> np.ndarray:
    """Select the latest source frame available at each constant output time."""

    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("timestamps must be a non-empty vector")
    if np.any(np.diff(values) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("output frame rate must be positive")
    relative = values - values[0]
    output_times = np.arange(int(np.floor(relative[-1] * fps)) + 1) / fps
    return np.searchsorted(relative, output_times + 1e-9, side="right") - 1
