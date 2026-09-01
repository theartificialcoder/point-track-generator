import numpy as np
import pytest

from dtf_eval.admission import AdmissionSupport
from dtf_eval.rolling import RecordedTrajectories, WindowTracks, record_rolling_trajectories


class _StationaryTracker:
    name = "stationary-test"
    future_context_frames = 2

    def track(self, frames_rgb, query_points, *, native_size):
        del native_size
        coordinates = np.repeat(query_points[None], len(frames_rgb), axis=0)
        confidence = np.ones(coordinates.shape[:2], dtype=np.float32)
        return WindowTracks(coordinates, confidence, 0.01, 0)


def test_rolling_recorder_adds_only_uncovered_new_support() -> None:
    probability = np.zeros((6, 16, 24), dtype=np.uint8)
    probability[:, 4:12, 4:12] = 255
    probability[2:, 4:12, 16:24] = 255
    support = AdmissionSupport(
        probability,
        np.arange(20, 26),
        12.5,
        (24, 16),
        {
            "point_trajectory_authority": False,
            "learned_mask_authority": False,
            "body_dynamics_authority": False,
        },
    )
    frames = np.zeros((6, 8, 12, 3), dtype=np.uint8)

    result = record_rolling_trajectories(
        frames,
        support,
        _StationaryTracker(),
        stride=4,
        coverage_radius=1.0,
        advance_frames=2,
        window_frames=4,
        max_active_tracks=20,
        visibility_threshold=0.6,
    )

    np.testing.assert_array_equal(np.unique(result.birth_frames), [0, 2])
    assert len(result.birth_frames) == 8
    assert result.metadata["tracks_retired_at_window_boundaries"] == 0


def test_rolling_recorder_enforces_active_capacity() -> None:
    probability = np.full((4, 16, 24), 255, dtype=np.uint8)
    support = AdmissionSupport(
        probability,
        np.arange(4),
        10.0,
        (24, 16),
        {
            "point_trajectory_authority": False,
            "learned_mask_authority": False,
            "body_dynamics_authority": False,
        },
    )

    result = record_rolling_trajectories(
        np.zeros((4, 8, 12, 3), dtype=np.uint8),
        support,
        _StationaryTracker(),
        stride=2,
        coverage_radius=1.0,
        advance_frames=2,
        window_frames=4,
        max_active_tracks=5,
        visibility_threshold=0.6,
    )

    assert len(result.birth_frames) == 5
    assert result.metadata["capacity_limited_windows"] >= 1


def test_saved_visibility_excludes_out_of_frame_tracks(tmp_path) -> None:
    recording = RecordedTrajectories(
        coordinates=np.array([[[3.0, 2.0]], [[12.0, 2.0]]], dtype=np.float32),
        confidence=np.ones((2, 1), dtype=np.float32),
        birth_frames=np.array([0], dtype=np.int32),
        source_frame_indices=np.array([4, 5], dtype=np.int64),
        fps=10.0,
        frame_size=(8, 6),
        provider="test",
        future_context_frames=1,
        metadata={},
    )

    path = tmp_path / "tracks.npz"
    recording.save(path, visibility_threshold=0.6)

    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["visibility"][:, 0], [True, False])


def test_recording_rejects_nonfinite_coordinates() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        RecordedTrajectories(
            coordinates=np.array([[[0.0, 0.0]], [[np.nan, 0.0]]], dtype=np.float32),
            confidence=np.ones((2, 1), dtype=np.float32),
            birth_frames=np.array([0]),
            source_frame_indices=np.arange(2),
            fps=10.0,
            frame_size=(8, 6),
            provider="test",
            future_context_frames=0,
            metadata={},
        )
