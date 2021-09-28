import numpy as np
import pytest
from zquantum.core import circuits
from zquantum.core.interfaces.backend import flip_wavefunction, split_circuit
from zquantum.core.wavefunction import Wavefunction


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


def test_splitting_circuits_partitions_it_into_expected_chunks():
    def _predicate(operation):
        return isinstance(
            operation, circuits.GateOperation
        ) and operation.gate.name in ("RX", "RY", "RZ")

    circuit = circuits.Circuit(
        [
            circuits.RX(np.pi)(0),
            circuits.RZ(np.pi / 2)(1),
            circuits.CNOT(2, 3),
            circuits.RY(np.pi / 4)(2),
            circuits.X(0),
            circuits.Y(1),
        ]
    )

    expected_partition = [
        (
            True,
            circuits.Circuit(
                [circuits.RX(np.pi)(0), circuits.RZ(np.pi / 2)(1)], n_qubits=4
            ),
        ),
        (False, circuits.Circuit([circuits.CNOT(2, 3)], n_qubits=4)),
        (True, circuits.Circuit([circuits.RY(np.pi / 4)(2)], n_qubits=4)),
        (False, circuits.Circuit([circuits.X(0), circuits.Y(1)], n_qubits=4)),
    ]

    assert list(split_circuit(circuit, _predicate)) == expected_partition
