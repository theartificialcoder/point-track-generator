import numpy as np
import pytest

from dtf_eval.media import constant_rate_source_indices


def test_constant_rate_indices_preserve_physical_timing() -> None:
    indices = constant_rate_source_indices(np.array([4.0, 4.1, 4.4]), fps=10.0)

    assert indices.tolist() == [0, 1, 1, 1, 2]


def test_constant_rate_indices_reject_non_monotonic_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        constant_rate_source_indices(np.array([0.0, 0.1, 0.1]), fps=30.0)
