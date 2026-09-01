"""Common adapters for DTF-Net and a chained Farneback baseline."""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from .field import GroupField, TrajectoryField


def _grid(height: int, width: int) -> np.ndarray:
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return np.stack((x, y), axis=-1)


def _to_original(
    coordinates: np.ndarray, original_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    infer_h, infer_w = coordinates.shape[1:3]
    original_h, original_w = original_shape
    scale = np.array([original_w / infer_w, original_h / infer_h], dtype=np.float32)
    reference = (_grid(infer_h, infer_w) + 0.5) * scale - 0.5
    converted = (coordinates + 0.5) * scale[None, None, None, :] - 0.5
    return converted.astype(np.float32), reference.astype(np.float32)


def _resize_frames(frames: tuple[np.ndarray, ...], size: tuple[int, int]) -> list[np.ndarray]:
    width, height = size
    return [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]


class DenseTracker(ABC):
    name: str

    @abstractmethod
    def track(
        self,
        frames: tuple[np.ndarray, ...],
        reference_index: int,
        inference_size: tuple[int, int],
    ) -> TrajectoryField:
        """Track the dense reference grid throughout the supplied clip."""


class DtfNetTracker(DenseTracker):
    name = "DTF-Net"

    def __init__(
        self,
        checkpoint: str | Path,
        vendor_root: str | Path,
        device: str = "cuda",
        capture_groups: bool = False,
    ) -> None:
        import torch

        root = Path(vendor_root).resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from dtf_core.networks.dtfnet import DtfNet

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
        self._device = torch.device(device)
        self._capture_groups = capture_groups
        self.last_groups: GroupField | None = None
        self._model = DtfNet()
        state = torch.load(checkpoint, map_location=self._device, weights_only=True)
        state = state.get("state_dict", state)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self._model.load_state_dict(state)
        self._model.to(self._device).requires_grad_(False).eval()

    @staticmethod
    def _group_field(
        module: object,
        inputs: tuple[object, ...],
        kwargs: dict[str, object],
        layer_index: int,
    ) -> GroupField:
        import torch

        tokens, sequence, _, _, _, _, position_embedding = inputs
        batch, time, channels, height, width = sequence.shape
        features = sequence.reshape(batch * time, channels, height, width)
        positions = position_embedding.reshape(
            batch * time, position_embedding.shape[2], height, width
        )
        strategy = module.pos_emb_strat.name
        if strategy in {"NORM_ADD", "NORM_CONCAT"}:
            features = module.compress_norm_kv(features)
        if strategy in {"NORM_ADD", "ADD_NORM"}:
            features = features + positions
        elif strategy in {"CONCAT_NORM", "NORM_CONCAT"}:
            features = torch.cat((features, positions), dim=1)
        if strategy in {"ADD_NORM", "CONCAT_NORM"}:
            features = module.compress_norm_kv(features)

        heads = module.num_heads
        projected_queries = module.proj_q(tokens)
        queries = projected_queries.reshape(
            batch, -1, heads, projected_queries.shape[-1] // heads
        )
        queries = queries.permute(0, 2, 1, 3)
        keys = module.proj_k(features).reshape(batch, time, heads, -1, height, width)
        logits = torch.einsum("bnzc,btnchw->btnzhw", queries, keys)
        membership = torch.softmax(logits * module.dot_scale * module.softmax_temp, dim=3)
        reference_index = kwargs.get("ref_idx", 0)
        if torch.is_tensor(reference_index):
            reference_index = int(reference_index.flatten()[0])
        membership = membership[0, int(reference_index)].mean(dim=0)
        labels = membership.argmax(dim=0)
        entropy = -(membership * membership.clamp_min(1e-12).log()).sum(dim=0)
        confidence = 1 - entropy / np.log(membership.shape[0])
        return GroupField(
            labels=labels.cpu().numpy().astype(np.uint16),
            confidence=confidence.clamp(0, 1).cpu().numpy().astype(np.float32),
            group_count=int(membership.shape[0]),
            layer_index=layer_index,
        )

    def track(
        self,
        frames: tuple[np.ndarray, ...],
        reference_index: int,
        inference_size: tuple[int, int],
    ) -> TrajectoryField:
        import torch

        resized = _resize_frames(frames, inference_size)
        rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in resized])
        sequence = torch.from_numpy(rgb).permute(0, 3, 1, 2).float().unsqueeze(0)
        sequence = sequence.to(self._device)
        captured: dict[str, object] = {}
        hook = None
        if self._capture_groups:

            def capture(module: object, inputs: tuple[object, ...], kwargs: dict[str, object]) -> None:
                captured.update(module=module, inputs=inputs, kwargs=kwargs)

            hook = self._model.process[-1].register_forward_pre_hook(capture, with_kwargs=True)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                trajectories, visibility = self._model(sequence, reference_index)
        finally:
            if hook is not None:
                hook.remove()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        runtime = time.perf_counter() - started
        self.last_groups = None
        if captured:
            with torch.inference_mode():
                self.last_groups = self._group_field(
                    captured["module"],
                    captured["inputs"],
                    captured["kwargs"],
                    self._model.nb_layers - 1,
                )
        coordinates = trajectories[0, -1].permute(0, 2, 3, 1).cpu().numpy()
        visible = visibility[0, -1, :, 0].cpu().numpy()
        converted, reference = _to_original(coordinates, frames[0].shape[:2])
        return TrajectoryField(converted, visible, reference, reference_index, runtime)


def _sample(field: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    map_x = coordinates[..., 0].astype(np.float32)
    map_y = coordinates[..., 1].astype(np.float32)
    return cv2.remap(
        field,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


class FarnebackChainTracker(DenseTracker):
    name = "Farneback chain"

    def track(
        self,
        frames: tuple[np.ndarray, ...],
        reference_index: int,
        inference_size: tuple[int, int],
    ) -> TrajectoryField:
        resized = _resize_frames(frames, inference_size)
        gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in resized]
        started = time.perf_counter()
        forward = [
            cv2.calcOpticalFlowFarneback(
                gray[index],
                gray[index + 1],
                None,
                0.5,
                5,
                15,
                3,
                5,
                1.2,
                0,
            )
            for index in range(reference_index, len(gray) - 1)
        ]
        backward = [
            cv2.calcOpticalFlowFarneback(
                gray[index + 1],
                gray[index],
                None,
                0.5,
                5,
                15,
                3,
                5,
                1.2,
                0,
            )
            for index in range(reference_index - 1, -1, -1)
        ]
        height, width = gray[0].shape
        origin = _grid(height, width)
        coordinates = np.empty((len(gray), height, width, 2), dtype=np.float32)
        coordinates[reference_index] = origin

        current = origin.copy()
        for offset, index in enumerate(range(reference_index, len(gray) - 1)):
            current = current + _sample(forward[offset], current)
            coordinates[index + 1] = current
        current = origin.copy()
        for offset, index in enumerate(range(reference_index - 1, -1, -1)):
            current = current + _sample(backward[offset], current)
            coordinates[index] = current

        visible = np.isfinite(coordinates).all(axis=-1)
        visible &= coordinates[..., 0] >= 0
        visible &= coordinates[..., 0] < width
        visible &= coordinates[..., 1] >= 0
        visible &= coordinates[..., 1] < height
        runtime = time.perf_counter() - started
        converted, reference = _to_original(coordinates, frames[0].shape[:2])
        return TrajectoryField(converted, visible.astype(np.float32), reference, reference_index, runtime)
