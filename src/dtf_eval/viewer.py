"""Generate one self-contained browser viewer for a tracker comparison."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import Frame
from .field import GroupField, TrajectoryField


def _jpeg_uri(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise ValueError("cannot encode viewer frame")
    payload = base64.b64encode(encoded).decode("ascii")
    return "data:image/jpeg;base64," + payload


def _sample_field(field: TrajectoryField, step: int) -> dict[str, object]:
    coordinates = field.coordinates[:, ::step, ::step]
    visibility = field.visibility[:, ::step, ::step]
    reference = field.reference_grid[::step, ::step]
    return {
        "coordinates": np.round(coordinates, 2).tolist(),
        "visibility": np.round(visibility, 3).tolist(),
        "reference": np.round(reference, 2).tolist(),
        "referenceIndex": field.reference_index,
    }


def _reference_texture(height: int, width: int) -> np.ndarray:
    """Return a stable UV-style colour texture for trajectory inspection."""

    x = np.arange(width, dtype=np.float32)[None, :]
    y = np.arange(height, dtype=np.float32)[:, None]
    hsv = np.empty((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = np.broadcast_to(179 * x / max(1, width - 1), (height, width))
    hsv[..., 1] = 190
    checker = ((x.astype(int) // 8 + y.astype(int) // 8) % 2).astype(np.uint8)
    hsv[..., 2] = 175 + 55 * checker
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    alpha = np.full((height, width, 1), 255, dtype=np.uint8)
    return np.concatenate((bgr, alpha), axis=-1)


def _forward_splat(
    field: TrajectoryField,
    time_index: int,
    values: np.ndarray,
) -> np.ndarray:
    """Move reference-grid values to their tracked sub-pixel coordinates."""

    height, width = field.grid_shape
    if values.shape != (height, width, 3):
        raise ValueError("splat values must match the trajectory grid")
    flat_values = values.reshape(-1, 3).astype(np.float32)
    coordinates = field.coordinates[time_index]
    reference = field.reference_grid
    dx = float(np.median(np.diff(reference[0, :, 0]))) if width > 1 else 1.0
    dy = float(np.median(np.diff(reference[:, 0, 1]))) if height > 1 else 1.0
    target_x = (coordinates[..., 0] - reference[0, 0, 0]) / dx
    target_y = (coordinates[..., 1] - reference[0, 0, 1]) / dy
    visible = field.visibility[time_index] >= 0.5
    visible &= np.isfinite(target_x) & np.isfinite(target_y)

    x0 = np.floor(np.where(visible, target_x, 0)).astype(np.int32)
    y0 = np.floor(np.where(visible, target_y, 0)).astype(np.int32)
    fractions_x = np.where(visible, target_x - x0, 0).astype(np.float32)
    fractions_y = np.where(visible, target_y - y0, 0).astype(np.float32)
    accumulator = np.zeros((height, width, 4), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    flat_visible = visible.ravel()

    for offset_x, offset_y, weight in (
        (0, 0, (1 - fractions_x) * (1 - fractions_y)),
        (1, 0, fractions_x * (1 - fractions_y)),
        (0, 1, (1 - fractions_x) * fractions_y),
        (1, 1, fractions_x * fractions_y),
    ):
        target_ix = (x0 + offset_x).ravel()
        target_iy = (y0 + offset_y).ravel()
        flat_weight = weight.ravel()
        valid = flat_visible & (target_ix >= 0) & (target_ix < width)
        valid &= (target_iy >= 0) & (target_iy < height) & (flat_weight > 0)
        indices = (target_iy[valid], target_ix[valid])
        np.add.at(weight_sum, indices, flat_weight[valid])
        for channel in range(3):
            np.add.at(
                accumulator[..., channel],
                indices,
                flat_values[valid, channel] * flat_weight[valid],
            )

    occupied = weight_sum > 1e-5
    accumulator[occupied, :3] /= weight_sum[occupied, None]
    accumulator[..., 3] = np.where(occupied, 255, 0)
    return np.clip(accumulator, 0, 255).astype(np.uint8)


def _png_uri(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cannot encode viewer overlay")
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def _dense_field_uri(field: TrajectoryField, time_index: int) -> str:
    """Render stable reference-coordinate colour through the trajectory field."""

    height, width = field.grid_shape
    texture = _reference_texture(height, width)[..., :3]
    return _png_uri(_forward_splat(field, time_index, texture))


def _field_visuals(
    field: TrajectoryField,
    frames: tuple[Frame, ...],
) -> dict[str, list[str]]:
    """Build dense field, warped-reference, and residual views once."""

    height, width = field.grid_shape
    reference_image = cv2.resize(
        frames[field.reference_index].image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    texture = _reference_texture(height, width)[..., :3]
    dense_fields: list[str] = []
    warped_references: list[str] = []
    differences: list[str] = []

    for time_index, frame in enumerate(frames):
        dense = _forward_splat(field, time_index, texture)
        warped = _forward_splat(field, time_index, reference_image)
        current = cv2.resize(frame.image, (width, height), interpolation=cv2.INTER_AREA)
        residual = np.mean(
            np.abs(current.astype(np.float32) - warped[..., :3].astype(np.float32)),
            axis=-1,
        ).astype(np.uint8)
        difference = cv2.applyColorMap(residual, cv2.COLORMAP_INFERNO)
        difference = np.concatenate((difference, warped[..., 3:4]), axis=-1)
        dense_fields.append(_png_uri(dense))
        warped_references.append(_png_uri(warped))
        differences.append(_png_uri(difference))

    return {
        "denseField": dense_fields,
        "warpedReference": warped_references,
        "difference": differences,
    }


def _group_visuals(field: TrajectoryField, groups: GroupField) -> dict[str, object]:
    """Transport final-layer centroid labels through the dense field."""

    height, width = field.grid_shape
    labels = cv2.resize(
        groups.labels,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    confidence = cv2.resize(
        groups.confidence,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    ids = np.arange(groups.group_count, dtype=np.uint16)
    hsv = np.zeros((1, groups.group_count, 3), dtype=np.uint8)
    hsv[0, :, 0] = (ids * 67 % 180).astype(np.uint8)
    hsv[0, :, 1] = 205
    hsv[0, :, 2] = 235
    palette = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]
    group_colours = palette[labels]
    confidence_colours = cv2.applyColorMap(
        np.round(255 * confidence).astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    return {
        "groupCount": groups.group_count,
        "layerIndex": groups.layer_index,
        "membership": [
            _png_uri(_forward_splat(field, time_index, group_colours))
            for time_index in range(field.length)
        ],
        "groupConfidence": [
            _png_uri(_forward_splat(field, time_index, confidence_colours))
            for time_index in range(field.length)
        ],
    }


def write_viewer(
    output: str | Path,
    frames: tuple[Frame, ...],
    fields: dict[str, TrajectoryField],
    report: dict[str, object],
    sample_step: int = 4,
    group_fields: dict[str, GroupField] | None = None,
) -> None:
    height, width = frames[0].image.shape[:2]
    annotations = []
    for frame in frames:
        contours = []
        for instance in frame.instances:
            found, _ = cv2.findContours(
                instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in found:
                contours.append(contour[:, 0, :].astype(int).tolist())
        annotations.append(contours)

    group_fields = group_fields or {}
    viewer_fields = {}
    for name, field in fields.items():
        viewer_fields[name] = {
            **_sample_field(field, sample_step),
            **_field_visuals(field, frames),
        }
        if name in group_fields:
            viewer_fields[name]["internalGroups"] = _group_visuals(
                field, group_fields[name]
            )

    payload = {
        "width": width,
        "height": height,
        "frames": [_jpeg_uri(frame.image) for frame in frames],
        "frameNumbers": [frame.frame_number for frame in frames],
        "annotations": annotations,
        "fields": viewer_fields,
        "report": report,
    }
    template = (Path(__file__).with_name("viewer.html")).read_text(encoding="utf-8")
    html = template.replace("__DTF_EVAL_DATA__", json.dumps(payload, separators=(",", ":")))
    Path(output).write_text(html, encoding="utf-8")
