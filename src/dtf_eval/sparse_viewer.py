"""Self-contained visual inspection for common sparse-track results."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import Frame
from .sparse import SparseTracks


def _image_uri(image: np.ndarray, size: tuple[int, int]) -> str:
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise ValueError("cannot encode viewer frame")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def _resized_masks(frame: Frame, size: tuple[int, int]) -> dict[int, np.ndarray]:
    return {
        int(instance.track_id): cv2.resize(
            instance.mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        for instance in frame.instances
        if instance.track_id is not None
    }


def _track_status(
    tracks: SparseTracks,
    frames: tuple[Frame, ...],
    size: tuple[int, int],
) -> tuple[np.ndarray, list[dict[str, list[list[list[int]]]]]]:
    """Classify display points; annotations remain visual diagnostics only."""

    status = np.zeros(tracks.visibility.shape, dtype=np.uint8)
    outlines: list[dict[str, list[list[list[int]]]]] = []
    for time_index, frame in enumerate(frames):
        masks = _resized_masks(frame, size)
        union = np.zeros((size[1], size[0]), dtype=bool)
        frame_outlines: dict[str, list[list[list[int]]]] = {}
        for track_id, mask in masks.items():
            union |= mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            frame_outlines[str(track_id)] = [
                contour[:, 0].astype(int).tolist() for contour in contours
            ]
        outlines.append(frame_outlines)

        coordinates = tracks.coordinates[time_index]
        safe = np.where(np.isfinite(coordinates), coordinates, 0)
        rounded = np.rint(safe).astype(np.int32)
        inside = np.isfinite(coordinates).all(axis=1)
        inside &= (rounded[:, 0] >= 0) & (rounded[:, 0] < size[0])
        inside &= (rounded[:, 1] >= 0) & (rounded[:, 1] < size[1])
        visible = (tracks.visibility[time_index] >= 0.5) & inside
        for query_index, track_id in enumerate(tracks.queries.track_ids):
            if time_index < tracks.queries.frame_indices[query_index]:
                status[time_index, query_index] = 4
                continue
            target = masks.get(int(track_id))
            if target is None:
                status[time_index, query_index] = 4
            elif not visible[query_index]:
                status[time_index, query_index] = 0
            else:
                x, y = rounded[query_index]
                status[time_index, query_index] = 1 if target[y, x] else 2 if union[y, x] else 3
    return status, outlines


def write_sparse_viewer(
    output: str | Path,
    frames: tuple[Frame, ...],
    results: dict[str, SparseTracks],
    size: tuple[int, int],
) -> None:
    if not results:
        raise ValueError("viewer requires at least one tracker result")
    query_sets = [result.queries for result in results.values()]
    reference = query_sets[0]
    for queries in query_sets[1:]:
        if not np.array_equal(queries.points, reference.points):
            raise ValueError("viewer results must use the same queries")
        if not np.array_equal(queries.track_ids, reference.track_ids):
            raise ValueError("viewer results must use the same query identities")

    algorithms = {}
    outlines = None
    for name, result in results.items():
        status, result_outlines = _track_status(result, frames, size)
        outlines = result_outlines
        algorithms[name] = {
            "coordinates": np.round(result.coordinates, 2).tolist(),
            "visibility": np.round(result.visibility, 3).tolist(),
            "status": status.tolist(),
            "runtimeSeconds": round(result.runtime_seconds, 3),
        }
    objects = []
    for track_id in np.unique(reference.track_ids):
        index = int(np.flatnonzero(reference.track_ids == track_id)[0])
        objects.append(
            {
                "id": int(track_id),
                "category": str(reference.categories[index]),
                "birth": int(reference.frame_indices[index]),
            }
        )
    candidates = []
    for item in objects:
        member = reference.track_ids == item["id"]
        if int(member.sum()) >= 8 and item["birth"] == 0:
            mean_y = float(reference.points[member, 1].mean())
            candidates.append((abs(mean_y - 0.55 * size[1]), item["id"]))
    default_object = min(candidates)[1] if candidates else objects[0]["id"]

    payload = {
        "width": size[0],
        "height": size[1],
        "frames": [_image_uri(frame.image, size) for frame in frames],
        "frameNumbers": [frame.frame_number for frame in frames],
        "queryTrackIds": reference.track_ids.astype(int).tolist(),
        "queryBirthFrames": reference.frame_indices.astype(int).tolist(),
        "objects": objects,
        "defaultObject": default_object,
        "outlines": outlines,
        "algorithms": algorithms,
    }
    template = Path(__file__).with_name("sparse_viewer.html").read_text(encoding="utf-8")
    Path(output).write_text(
        template.replace("__SPARSE_TRACK_DATA__", json.dumps(payload, separators=(",", ":"))),
        encoding="utf-8",
    )
