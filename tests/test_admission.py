import json

import numpy as np
import pytest

from dtf_eval.admission import AdmissionSupport


def _write_support(path, *, body_authority=False) -> None:
    document = {
        "schema_version": 1,
        "fps": 12.5,
        "frame_size": [6, 4],
        "metadata": {
            "point_trajectory_authority": False,
            "learned_mask_authority": False,
            "body_dynamics_authority": body_authority,
        },
    }
    np.savez_compressed(
        path,
        probability_u8=np.asarray(
            [np.zeros((4, 6)), np.full((4, 6), 255)], dtype=np.uint8
        ),
        source_frame_indices=np.asarray([7, 8]),
        document=np.asarray(json.dumps(document)),
    )


def test_admission_support_loads_equal_cost_decision(tmp_path) -> None:
    path = tmp_path / "support.npz"
    _write_support(path)

    support = AdmissionSupport.load(path)

    assert not support.frame(0).any()
    assert support.frame(1).all()


def test_admission_support_rejects_simulator_authority(tmp_path) -> None:
    path = tmp_path / "support.npz"
    _write_support(path, body_authority=True)

    with pytest.raises(ValueError, match="forbidden"):
        AdmissionSupport.load(path)
