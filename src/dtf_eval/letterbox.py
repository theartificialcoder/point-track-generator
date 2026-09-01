"""Aspect-ratio-preserving mapping between native frames and square models."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SquareLetterbox:
    source_width: int
    source_height: int
    model_size: int

    @property
    def scale(self) -> float:
        return min(
            self.model_size / self.source_width,
            self.model_size / self.source_height,
        )

    @property
    def resized_size(self) -> tuple[int, int]:
        return (
            max(1, round(self.source_width * self.scale)),
            max(1, round(self.source_height * self.scale)),
        )

    @property
    def offset(self) -> tuple[int, int]:
        width, height = self.resized_size
        return ((self.model_size - width) // 2, (self.model_size - height) // 2)

    def image(self, source: np.ndarray) -> np.ndarray:
        if source.shape[:2] != (self.source_height, self.source_width):
            raise ValueError("source image does not match letterbox geometry")
        width, height = self.resized_size
        offset_x, offset_y = self.offset
        resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.model_size, self.model_size, 3), dtype=source.dtype)
        canvas[offset_y : offset_y + height, offset_x : offset_x + width] = resized
        return canvas

    def to_model_image(self, points: np.ndarray) -> np.ndarray:
        offset_x, offset_y = self.offset
        return points * self.scale + np.array([offset_x, offset_y], dtype=np.float32)

    def to_native(self, points: np.ndarray) -> np.ndarray:
        offset_x, offset_y = self.offset
        return (
            points - np.array([offset_x, offset_y], dtype=np.float32)
        ) / self.scale
