"""Minimal read-only access to the reviewed COCO-RLE traffic sequence."""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Instance:
    track_id: int | None
    category: str
    mask: np.ndarray


@dataclass(frozen=True, slots=True)
class Frame:
    frame_number: int
    image: np.ndarray
    instances: tuple[Instance, ...]
    timestamp_seconds: float | None = None


def _decode_counts(value: str | Sequence[int]) -> list[int]:
    if not isinstance(value, str):
        return [int(run) for run in value]
    runs: list[int] = []
    position = 0
    while position < len(value):
        run = 0
        shift = 0
        while True:
            code = ord(value[position]) - 48
            position += 1
            run |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                run |= -1 << (5 * (shift + 1))
            shift += 1
            if not more:
                break
        if len(runs) > 2:
            run += runs[-2]
        if run < 0:
            raise ValueError("negative COCO RLE run")
        runs.append(run)
    return runs


def decode_rle(segmentation: dict[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in segmentation["size"])
    runs = _decode_counts(segmentation["counts"])
    if sum(runs) != height * width:
        raise ValueError("COCO RLE does not cover the declared image")
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    for index, run in enumerate(runs):
        if index % 2:
            flat[offset : offset + run] = 1
        offset += run
    return flat.reshape((height, width), order="F").astype(bool)


class CocoTrafficArchive:
    """Load frames and masks without extracting or modifying the archive."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not zipfile.is_zipfile(self.path):
            raise ValueError("traffic benchmark must be a ZIP archive")
        with zipfile.ZipFile(self.path) as source:
            members = source.namelist()
            documents = [name for name in members if name.endswith("annotations.coco.json")]
            if len(documents) != 1:
                raise ValueError("archive must contain exactly one annotations.coco.json")
            self._document_name = documents[0]
            document = json.loads(source.read(self._document_name))

        self._members = set(members)
        self._categories = {
            int(item["id"]): str(item["name"]) for item in document.get("categories", [])
        }
        self._images = sorted(
            document["images"],
            key=lambda item: int(item.get("source_frame_index", item.get("frame_index", 0))),
        )
        self._annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in document.get("annotations", []):
            self._annotations[int(annotation["image_id"])].append(annotation)

    def __len__(self) -> int:
        return len(self._images)

    def _image_member(self, file_name: str) -> str:
        parent = Path(self._document_name).parent
        candidates = (str(parent / file_name), str(parent / "frames" / file_name))
        for candidate in candidates:
            if candidate in self._members:
                return candidate
        matches = [name for name in self._members if name.endswith("/" + file_name)]
        if len(matches) != 1:
            raise ValueError(f"missing or ambiguous frame: {file_name}")
        return matches[0]

    def frame(self, index: int) -> Frame:
        image_info = self._images[index]
        member = self._image_member(str(image_info["file_name"]))
        with zipfile.ZipFile(self.path) as source:
            encoded = np.frombuffer(source.read(member), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode frame {member}")

        instances = []
        for annotation in self._annotations[int(image_info["id"])]:
            attributes = annotation.get("attributes", {})
            raw_track_id = annotation.get("track_id", attributes.get("cvat_track_id"))
            instances.append(
                Instance(
                    track_id=None if raw_track_id is None else int(raw_track_id),
                    category=self._categories[int(annotation["category_id"])],
                    mask=decode_rle(annotation["segmentation"]),
                )
            )
        frame_number = int(
            image_info.get("source_frame_index", image_info.get("frame_index", index))
        )
        raw_timestamp = image_info.get("timestamp_s")
        timestamp = None if raw_timestamp is None else float(raw_timestamp)
        return Frame(frame_number, image, tuple(instances), timestamp)

    def clip(self, start: int, length: int) -> tuple[Frame, ...]:
        if start < 0 or length < 2 or start + length > len(self):
            raise ValueError("requested clip lies outside the archive")
        return tuple(self.frame(index) for index in range(start, start + length))
