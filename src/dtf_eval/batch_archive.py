"""Resumable storage for large point-tracker batches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class BatchArchive:
    """Persist contiguous query batches without retaining them in process memory."""

    path: Path
    coordinates: np.memmap
    visibility: np.memmap
    metadata: dict[str, Any]

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        frame_count: int,
        point_count: int,
        identity: dict[str, Any],
    ) -> BatchArchive:
        root = Path(path)
        manifest_path = root / "manifest.json"
        expected = {
            "version": 1,
            "frame_count": frame_count,
            "point_count": point_count,
            "identity": identity,
        }
        if manifest_path.exists():
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(f"partial archive {key} does not match this run")
            mode = "r+"
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError("partial archive exists without a manifest")
            root.mkdir(parents=True, exist_ok=True)
            metadata = {
                **expected,
                "next_query": 0,
                "runtime_seconds": 0.0,
                "peak_gpu_memory_bytes": 0,
            }
            cls._write_manifest(root, metadata)
            mode = "w+"
        coordinates = np.lib.format.open_memmap(
            root / "coordinates.npy",
            mode=mode,
            dtype=np.float32,
            shape=(frame_count, point_count, 2),
        )
        visibility = np.lib.format.open_memmap(
            root / "visibility.npy",
            mode=mode,
            dtype=np.float32,
            shape=(frame_count, point_count),
        )
        return cls(root, coordinates, visibility, metadata)

    @property
    def next_query(self) -> int:
        return int(self.metadata["next_query"])

    def write(
        self,
        start: int,
        coordinates: np.ndarray,
        visibility: np.ndarray,
        *,
        runtime_seconds: float,
        peak_gpu_memory_bytes: int,
    ) -> None:
        if start != self.next_query:
            raise ValueError("query batches must be written contiguously")
        end = start + coordinates.shape[1]
        if coordinates.shape != (self.coordinates.shape[0], end - start, 2):
            raise ValueError("coordinate batch shape mismatch")
        if visibility.shape != coordinates.shape[:2]:
            raise ValueError("visibility batch shape mismatch")
        self.coordinates[:, start:end] = coordinates
        self.visibility[:, start:end] = visibility
        self.coordinates.flush()
        self.visibility.flush()
        self.metadata["next_query"] = end
        self.metadata["runtime_seconds"] += float(runtime_seconds)
        self.metadata["peak_gpu_memory_bytes"] = max(
            int(self.metadata["peak_gpu_memory_bytes"]), int(peak_gpu_memory_bytes)
        )
        self._write_manifest(self.path, self.metadata)

    @staticmethod
    def _write_manifest(path: Path, metadata: dict[str, Any]) -> None:
        temporary = path / "manifest.json.tmp"
        temporary.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        temporary.replace(path / "manifest.json")
