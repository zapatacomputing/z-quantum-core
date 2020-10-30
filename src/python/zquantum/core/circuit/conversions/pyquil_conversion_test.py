"""Test cases for pyquil conversion."""
import pyquil
import pyquil.gates
import sympy
from pyquil.simulation.matrices import QUANTUM_GATES
from pyquil.simulation.tools import lifted_gate
from ...circuit.gates import ControlledGate, CustomGate
from ...circuit.conversions.pyquil_conversions import convert_to_pyquil
from ...circuit.gates import X, Y, Z, RX, RY, RZ, PHASE, T, I, H, CZ, CNOT, CPHASE, SWAP, Dagger
import numpy as np
import pytest


ORQUESTRA_GATE_TYPE_TO_PYQUIL_NAME = {
    X: "X",
    Y: "Y",
    Z: "Z",
    H: "H",
    PHASE: "PHASE",
    T: "T",
    I: "I",
    RX: "RX",
    RY: "RY",
    RZ: "RZ",
    CNOT: "CNOT",
    CZ: "CZ",
    SWAP: "SWAP",
    CPHASE: "CPHASE"
}


def pyquil_gate_matrix(gate: pyquil.gates.Gate) -> np.ndarray:
    """Get numpy matrix corresponding to pyquil Gate.

    This is based on PyQuil's source code in pyquil.simulation.tools.
    """
    if len(gate.params) > 0:
        return QUANTUM_GATES[gate.name](*gate.params)
    else:
        return QUANTUM_GATES[gate.name]


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
    "gate",
    [
        X(2),
        Y(0),
        Z(1),
        H(0),
        PHASE(0, np.pi),
        T(2),
        I(10),
        RX(0, np.pi),
        RY(0, np.pi / 2),
        RZ(0, 0.0),
        CNOT(0, 1),
        CZ(2, 12),
        SWAP((2, 4)),
        CPHASE(2, 4, np.pi / 4),
    ]
)
def test_converting_gate_to_pyquil_preserves_its_type_and_matrix(gate):
    pyquil_gate = convert_to_pyquil(gate)

    assert pyquil_gate.name == ORQUESTRA_GATE_TYPE_TO_PYQUIL_NAME[type(gate)]
    assert np.allclose(pyquil_gate_matrix(pyquil_gate), np.array(gate.matrix.tolist(), dtype=complex))


# Below we use multiple control qubits. What we mean is that we construct
# controlled gate from controlled gate iteratively, e, g.
# X(2), (0, 1) -> ControlledGate(ControlledGate(X(2), 0), 1)
# This is to test whether pyquil CONTROLLED modifier gets applied correct
# of times.
@pytest.mark.parametrize(
    "target_gate, control_qubits",
    [
        (X(2), (1,)),
        (Y(1), (0,)),
        (PHASE(4, np.pi), (1, 2, 3)),
        (CZ(2, 12), (0, 3))
    ]
)
class TestControlledGateConversion:

    def make_controlled_gate(self, target_gate, control_qubits):
        if control_qubits:
            return self.make_controlled_gate(
                ControlledGate(target_gate, control_qubits[0]), control_qubits[1:]
            )
        return target_gate

    def test_converting_controlled_gate_to_pyquil_gives_gate_with_appropriate_name(
        self, target_gate, control_qubits
    ):
        controlled_gate = self.make_controlled_gate(target_gate, control_qubits)

        pyquil_gate = convert_to_pyquil(controlled_gate)

        assert pyquil_gate.name == ORQUESTRA_GATE_TYPE_TO_PYQUIL_NAME[type(target_gate)]

    def test_converting_controlled_gate_to_pyquil_gives_gate_with_correct_qubits(
        self, target_gate, control_qubits
    ):
        controlled_gate = self.make_controlled_gate(target_gate, control_qubits)

        pyquil_gate = convert_to_pyquil(controlled_gate)

        assert all(
            pyquil_qubit.index == qubit
            for pyquil_qubit, qubit in zip(pyquil_gate.qubits, controlled_gate.qubits)
        )

    def test_converting_controlled_gate_to_pyquil_gives_gate_with_controlled_modifier(
        self, target_gate, control_qubits
    ):
        controlled_gate = self.make_controlled_gate(target_gate, control_qubits)

        pyquil_gate = convert_to_pyquil(controlled_gate)

        assert pyquil_gate.modifiers == len(control_qubits) * ["CONTROLLED"]


def test_converting_dagger_object_to_pyquil_gives_gate_with_dagger_modifier():
    gate = Dagger(X(1))
    assert convert_to_pyquil(gate).modifiers == ["DAGGER"]


@pytest.mark.parametrize(
    "custom_gate",
    [
        CustomGate(0.5 * sympy.Matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]), (0,), name="my_gate"),
        CustomGate(sympy.Matrix(
            [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, -1j, 0]
            ]
        ), (0, 1))
    ]
)
def test_converting_custom_gate_to_pyquil_adds_its_definition_to_program(
    custom_gate
):
    program = pyquil.Program()
    convert_to_pyquil(custom_gate, program)
    assert program.defined_gates == [
        pyquil.quil.DefGate(custom_gate.name, np.array(custom_gate.matrix, dtype=complex))
    ]


@pytest.mark.parametrize(
    "times_to_convert", [2, 3, 5, 6]
)
def test_converting_gate_with_the_same_name_multiple_times_adds_only_a_single_definition_to_pyquil_program(
    times_to_convert
):
    program = pyquil.Program()
    gate = CustomGate(0.5 * sympy.Matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]), (0,), name="my_gate")
    for _ in range(times_to_convert):
        convert_to_pyquil(gate, program)

    assert program.defined_gates == [
        pyquil.quil.DefGate(gate.name, np.array(gate.matrix, dtype=complex))
    ]
