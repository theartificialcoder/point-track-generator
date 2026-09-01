"""Read calibration-only query support exported by blob-sim."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class AdmissionSupport:
    probability_u8: np.ndarray
    source_frame_indices: np.ndarray
    fps: float
    frame_size: tuple[int, int]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        frames, height, width = self.probability_u8.shape
        if self.probability_u8.dtype != np.uint8:
            raise TypeError("admission probability must use uint8 storage")
        if self.source_frame_indices.shape != (frames,):
            raise ValueError("support frame metadata length mismatch")
        if frames < 2 or not np.all(np.diff(self.source_frame_indices) == 1):
            raise ValueError("support frames must be consecutive")
        if self.frame_size != (width, height):
            raise ValueError("support geometry does not match frame_size")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("support fps must be positive")

    def frame(self, index: int) -> np.ndarray:
        """Return the equal-cost support decision for one relative frame."""

        return self.probability_u8[index] >= 128

    @classmethod
    def load(cls, path: str | Path) -> AdmissionSupport:
        with np.load(path, allow_pickle=False) as source:
            document = json.loads(str(source["document"]))
            if int(document.get("schema_version", -1)) != 1:
                raise ValueError("unsupported admission-support schema")
            metadata = dict(document.get("metadata", {}))
            forbidden = (
                "point_trajectory_authority",
                "learned_mask_authority",
                "body_dynamics_authority",
            )
            if any(metadata.get(name) is not False for name in forbidden):
                raise ValueError("admission support contains forbidden runtime authority")
            return cls(
                probability_u8=source["probability_u8"],
                source_frame_indices=source["source_frame_indices"],
                fps=float(document["fps"]),
                frame_size=tuple(map(int, document["frame_size"])),
                metadata=metadata,
            )


__all__ = ["AdmissionSupport"]
