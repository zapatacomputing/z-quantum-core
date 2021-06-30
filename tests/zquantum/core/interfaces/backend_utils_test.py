import numpy as np
import pytest
from pyquil.wavefunction import Wavefunction
from zquantum.core.interfaces.backend import flip_wavefunction


@pytest.mark.parametrize(
    "input_amplitudes, expected_output_amplitudes",
    [
        (
            np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0]),
            np.array([1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0]),
        ),
        (
            np.array([0.5, 8 ** -0.5, 0, 0, 0, 8 ** -0.5, 0.5, 0.5]),
            np.array([0.5, 0, 0, 0.5, 8 ** -0.5, 8 ** -0.5, 0, 0.5]),
        ),
    ],
)
def test_flipped_wavefunction_comprises_expected_amplitudes(
    input_amplitudes, expected_output_amplitudes
):
    np.testing.assert_array_equal(
        flip_wavefunction(Wavefunction(input_amplitudes)).amplitudes,
        expected_output_amplitudes,
    )
