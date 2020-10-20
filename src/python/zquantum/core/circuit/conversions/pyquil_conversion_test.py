"""Test cases for pyquil conversion."""
from ...circuit.conversions.pyquil_conversions import convert_to_pyquil
from ...circuit.gates import X, Y, Z, RX, RY, RZ, PHASE, T, I, H, CZ, CNOT, CPHASE, SWAP
import numpy as np
import pytest


@pytest.mark.parametrize("qubit", [0, 1, 5, 13])
@pytest.mark.parametrize("gate_cls", [X, Y, Z, T, I, H])
def test_converting_single_qubit_nonparametric_gate_to_pyquil_preserves_qubit_index(
    qubit, gate_cls
):
    pyquil_gate = convert_to_pyquil(gate_cls(qubit))

    assert len(pyquil_gate.qubits) == 1
    assert pyquil_gate.qubits[0].index == qubit


@pytest.mark.parametrize("qubit", [0, 4, 10, 11])
@pytest.mark.parametrize("angle", [np.pi, np.pi/2, 0.4])
@pytest.mark.parametrize("gate_cls", [RX, RY, RZ, PHASE])
def test_converting_rotation_gate_to_pyquil_preserves_qubit_index_and_angle(
    qubit, angle, gate_cls
):
    pyquil_gate = convert_to_pyquil(gate_cls(qubit, angle))

    assert len(pyquil_gate.qubits) == 1
    assert pyquil_gate.qubits[0].index == qubit

    assert len(pyquil_gate.params) == 1
    assert pyquil_gate.params[0] == angle


@pytest.mark.parametrize("qubits", [[0, 1], [2, 10], [4, 7]])
@pytest.mark.parametrize("angle", [0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_pyquil_gate_created_from_zquantum_cphase_gate_has_the_same_qubits_and_angle_as_the_original_one(
    qubits, angle
):
    pyquil_gate = convert_to_pyquil(CPHASE(*qubits, angle))

    assert len(pyquil_gate.qubits) == 2
    assert pyquil_gate.qubits[0].index == qubits[0]
    assert pyquil_gate.qubits[1].index == qubits[1]

    assert len(pyquil_gate.params) == 1
    assert pyquil_gate.params[0] == angle


@pytest.mark.parametrize("qubits", [[0, 1], [2, 10], [4, 7]])
def test_converting_swap_gate_to_pyquil_preserves_qubits(qubits):
    pyquil_gate = convert_to_pyquil(SWAP(qubits))

    assert pyquil_gate.qubits[0].index == qubits[0]
    assert pyquil_gate.qubits[1].index == qubits[1]


@pytest.mark.parametrize("control, target", [(0, 1), (2, 3), (0, 10)])
@pytest.mark.parametrize("gate_cls", [CZ, CNOT])
def test_converting_two_qubit_controlled_gate_to_pyquil_preserves_qubit_indices(
    control, target, gate_cls
):
    pyquil_gate = convert_to_pyquil(gate_cls(control, target))

    assert len(pyquil_gate.qubits) == 2
    assert pyquil_gate.qubits[0].index == control
    assert pyquil_gate.qubits[1].index == target


@pytest.mark.parametrize(
    "gate, expected_pyquil_name",
    [
        (X(2), "X"),
        (Y(0), "Y"),
        (Z(1), "Z"),
        (H(0), "H"),
        (PHASE(0, np.pi), "PHASE"),
        (T(2), "T"),
        (I(10), "I"),
        (RX(0, np.pi), "RX"),
        (RY(0, np.pi / 2), "RY"),
        (RZ(0, 0.0), "RZ"),
        (CNOT(0, 1), "CNOT"),
        (CZ(2, 12), "CZ"),
        (SWAP((2, 4)), "SWAP"),
        (CPHASE(2, 4, np.pi / 4), "CPHASE")
    ]
)
def test_converting_gate_to_pyquil_preserves_its_type(gate, expected_pyquil_name):
    pyquil_gate = convert_to_pyquil(gate)

    assert pyquil_gate.name == expected_pyquil_name
