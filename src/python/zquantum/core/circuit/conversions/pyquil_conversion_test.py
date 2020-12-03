"""Test cases for pyquil conversion."""
import pyquil
import pyquil.gates
from pyquil import quilatom
import sympy
from pyquil.simulation.matrices import QUANTUM_GATES
from ..circuit import Circuit
from ...circuit.gates import ControlledGate, CustomGate
from ...circuit.conversions.pyquil_conversions import convert_to_pyquil, convert_from_pyquil
from ...circuit.gates import (
    X,
    Y,
    Z,
    RX,
    RY,
    RZ,
    PHASE,
    T,
    I,
    H,
    CZ,
    CNOT,
    CPHASE,
    SWAP,
    Dagger,
)
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
    CPHASE: "CPHASE",
}

EXAMPLE_PARAMETRIZED_ANGLES = [
    (sympy.Symbol("theta"), pyquil.quil.Parameter("theta")),
    (
        sympy.Symbol("x") + sympy.Symbol("y"),
        pyquil.quil.Parameter("x") + pyquil.quil.Parameter("y"),
    ),
    (2 * sympy.Symbol("phi"), 2 * pyquil.quil.Parameter("phi")),
]


def pyquil_gate_matrix(gate: pyquil.gates.Gate) -> np.ndarray:
    """Get numpy matrix corresponding to pyquil Gate.

    This is based on PyQuil's source code in pyquil.simulation.tools.
    """
    if len(gate.params) > 0:
        return QUANTUM_GATES[gate.name](*gate.params)
    else:
        return QUANTUM_GATES[gate.name]


@pytest.mark.parametrize("qubit_index", [0, 1, 5, 13])
class TestSingleQubitNonParametricGatesConversion:

    @pytest.mark.parametrize("orquestra_gate_cls", [X, Y, Z, T, I, H])
    def test_conversion_from_orquestra_to_pyquil_preserves_qubit_index(
        self, qubit_index, orquestra_gate_cls
    ):
        pyquil_gate = convert_to_pyquil(orquestra_gate_cls(qubit_index))

        assert pyquil_gate.qubits == [pyquil.quil.Qubit(qubit_index)]

    @pytest.mark.parametrize(
        "pyquil_gate_func",
        [
            pyquil.gates.X,
            pyquil.gates.Y,
            pyquil.gates.Z,
            pyquil.gates.T,
            pyquil.gates.I,
            pyquil.gates.H
        ]
    )
    def test_converting_single_qubit_nonparametric_gate_from_pyquil_preserves_qubit_index(
        self, qubit_index, pyquil_gate_func
    ):
        orquestra_gate = convert_from_pyquil(pyquil_gate_func(qubit_index))

        assert orquestra_gate.qubits == (qubit_index,)


@pytest.mark.parametrize("qubit", [0, 4, 10, 11])
@pytest.mark.parametrize("angle", [np.pi, np.pi / 2, 0.4])
@pytest.mark.parametrize("gate_cls", [RX, RY, RZ, PHASE])
def test_converting_rotation_gate_to_pyquil_preserves_qubit_index_and_angle(
    qubit, angle, gate_cls
):
    pyquil_gate = convert_to_pyquil(gate_cls(qubit, angle))

    assert len(pyquil_gate.qubits) == 1
    assert pyquil_gate.qubits[0].index == qubit

    assert len(pyquil_gate.params) == 1
    assert pyquil_gate.params[0] == angle


@pytest.mark.parametrize("qubit", [0, 4, 10, 11])
@pytest.mark.parametrize("zquantum_angle, pyquil_angle", EXAMPLE_PARAMETRIZED_ANGLES)
@pytest.mark.parametrize("gate_cls", [RX, RY, RZ, PHASE])
def test_converting_parametrized_rotation_gate_to_pyquil_translates_angle_expression(
    qubit, zquantum_angle, pyquil_angle, gate_cls
):
    program = pyquil.Program()
    pyquil_gate = convert_to_pyquil(gate_cls(qubit, zquantum_angle), program)
    assert pyquil_gate.params[0] == pyquil_angle


@pytest.mark.parametrize("qubit", [0, 4, 10, 11])
@pytest.mark.parametrize("zquantum_angle, pyquil_angle", EXAMPLE_PARAMETRIZED_ANGLES)
@pytest.mark.parametrize("gate_cls", [RX, RY, RZ, PHASE])
def test_converting_parametrized_rotation_gate_to_pyquil_adds_its_symbolic_params_to_program(
    qubit, zquantum_angle, pyquil_angle, gate_cls
):
    program = pyquil.Program()
    zquantum_gate = gate_cls(qubit, zquantum_angle)
    convert_to_pyquil(zquantum_gate, program)
    assert program.instructions == [
        pyquil.quil.Declare(str(param), "REAL")
        for param in zquantum_gate.symbolic_params
    ]


@pytest.mark.parametrize("qubits", [[0, 1], [2, 10], [4, 7]])
@pytest.mark.parametrize("angle", [0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
def test_pyquil_gate_created_from_zquantum_cphase_gate_has_the_same_qubits_and_angle_as_the_original_one(
    qubits, angle
):
    program = pyquil.Program()

    pyquil_gate = convert_to_pyquil(CPHASE(*qubits, angle), program)

    assert len(pyquil_gate.qubits) == 2
    assert pyquil_gate.qubits[0].index == qubits[0]
    assert pyquil_gate.qubits[1].index == qubits[1]

    assert len(pyquil_gate.params) == 1
    assert pyquil_gate.params[0] == angle


@pytest.mark.parametrize("qubits", [[0, 1], [2, 10], [4, 7]])
@pytest.mark.parametrize("zquantum_angle, pyquil_angle", EXAMPLE_PARAMETRIZED_ANGLES)
def test_angle_of_parametrized_cphase_gate_is_translated_when_converting_to_pyquil(
    qubits, zquantum_angle, pyquil_angle
):
    program = pyquil.Program()
    pyquil_gate = convert_to_pyquil(CPHASE(*qubits, zquantum_angle), program)
    assert pyquil_gate.params[0] == pyquil_angle


@pytest.mark.parametrize("qubits", [[0, 1], [2, 10], [4, 7]])
@pytest.mark.parametrize("zquantum_angle, pyquil_angle", EXAMPLE_PARAMETRIZED_ANGLES)
def test_converting_parametrized_cphase_gate_to_pyquil_adds_its_symbolic_params_to_program(
    qubits, zquantum_angle, pyquil_angle
):
    zquantum_gate = CPHASE(*qubits, zquantum_angle)
    program = pyquil.Program()
    convert_to_pyquil(zquantum_gate, program)

    assert program.instructions == [
        pyquil.quil.Declare(str(param), "REAL")
        for param in zquantum_gate.symbolic_params
    ]


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
    ],
)
def test_converting_gate_to_pyquil_preserves_its_type_and_matrix(gate):
    pyquil_gate = convert_to_pyquil(gate)

    assert pyquil_gate.name == ORQUESTRA_GATE_TYPE_TO_PYQUIL_NAME[type(gate)]
    assert np.allclose(
        pyquil_gate_matrix(pyquil_gate), np.array(gate.matrix.tolist(), dtype=complex)
    )


# Below we use multiple control qubits. What we mean is that we construct
# controlled gate from controlled gate iteratively, e, g.
# X(2), (0, 1) -> ControlledGate(ControlledGate(X(2), 0), 1)
# This is to test whether pyquil CONTROLLED modifier gets applied correct
# of times.
@pytest.mark.parametrize(
    "target_gate, control_qubits",
    [(X(2), (1,)), (Y(1), (0,)), (PHASE(4, np.pi), (1, 2, 3)), (CZ(2, 12), (0, 3))],
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
        CustomGate(
            0.5 * sympy.Matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
            (0,),
            name="my_gate",
        ),
        CustomGate(
            sympy.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]]),
            (0, 1),
        ),
    ],
)
def test_converting_custom_gate_to_pyquil_adds_its_definition_to_program(custom_gate):
    program = pyquil.Program()
    convert_to_pyquil(custom_gate, program)
    assert program.defined_gates == [
        pyquil.quil.DefGate(
            custom_gate.name, np.array(custom_gate.matrix, dtype=complex)
        )
    ]


def test_converting_parametrized_custom_gate_to_pyquil_adds_its_definition_to_program():
    x, y, theta = sympy.symbols("x, y, theta")
    # Clearly, the below gate is not unitary. For the purpose of this test
    # it does not matter though.
    custom_gate = CustomGate(
        sympy.Matrix(
            [
                [sympy.cos(x + y), sympy.sin(theta), 0, 0],
                [-sympy.sin(theta), sympy.cos(x + y), 0, 0],
                [0, 0, sympy.sqrt(x), sympy.exp(y)],
                [0, 0, 1.0, sympy.I],
            ]
        ),
        (0, 2),
        "my_gate",
    )

    program = pyquil.Program()
    convert_to_pyquil(custom_gate, program)

    quil_x = pyquil.quil.Parameter("x")
    quil_y = pyquil.quil.Parameter("y")
    quil_theta = pyquil.quil.Parameter("theta")

    expected_pyquil_matrix = np.array(
        [
            [quilatom.quil_cos(quil_x + quil_y), quilatom.quil_sin(quil_theta), 0, 0],
            [-quilatom.quil_sin(quil_theta), quilatom.quil_cos(quil_x + quil_y), 0, 0],
            [0, 0, quilatom.quil_sqrt(quil_x), quilatom.quil_exp(quil_y)],
            [0, 0, 1.0, 1j],
        ]
    )

    # Note: we cannot replace this with a single assertion. This is because
    # the order of custom_gate.symbolic_params is not known.
    gate_definition = program.defined_gates[0]
    assert len(program.defined_gates) == 1
    assert len(gate_definition.parameters) == 3
    assert set(gate_definition.parameters) == {quil_x, quil_y, quil_theta}
    assert np.array_equal(gate_definition.matrix, expected_pyquil_matrix)


@pytest.mark.parametrize("times_to_convert", [2, 3, 5, 6])
def test_converting_gate_with_the_same_name_multiple_times_adds_only_a_single_definition_to_pyquil_program(
    times_to_convert,
):
    program = pyquil.Program()
    gate = CustomGate(
        0.5 * sympy.Matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]), (0,), name="my_gate"
    )
    for _ in range(times_to_convert):
        convert_to_pyquil(gate, program)

    assert program.defined_gates == [
        pyquil.quil.DefGate(gate.name, np.array(gate.matrix, dtype=complex))
    ]


def test_converting_circuit_to_pyquil_gives_program_with_the_same_gates():
    # The goal of the program constructed below is to include a diverse range
    # of gates.
    theta = sympy.Symbol("theta")
    quil_theta = pyquil.quil.Parameter("theta")

    circuit = Circuit(
        [
            CustomGate(
                sympy.Matrix(
                    [
                        [sympy.cos(theta), sympy.sin(theta)],
                        [-sympy.sin(theta), sympy.cos(theta)],
                    ]
                ),
                (0,),
                name="V",
            ),
            X(0),
            Y(1).dagger,
            Z(3),
            ControlledGate(SWAP((0, 2)), 1),
            CustomGate(
                sympy.Matrix(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]]
                ),
                (1, 3),
                name="U",
            ).dagger,
            RX(2, np.pi),
            CNOT(1, 3),
        ]
    )

    converted_program = convert_to_pyquil(circuit)

    u_gate_definition = pyquil.quil.DefGate(
        "U", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]]
    )

    v_gate_definition = pyquil.quil.DefGate(
        "V",
        [
            [quilatom.quil_cos(quil_theta), quilatom.quil_sin(quil_theta)],
            [-quilatom.quil_sin(quil_theta), quilatom.quil_cos(quil_theta)],
        ],
        [quil_theta],
    )

    # The below methods for getting constructors differ, because behaviours of
    # get_constructors differs depending on whether custom gate has
    # parameters or not.
    u_gate_constructor = u_gate_definition.get_constructor()
    v_gate_constructor = v_gate_definition.get_constructor()(quil_theta)

    expected_program = pyquil.Program(
        pyquil.quil.Declare("theta", "REAL"),
        v_gate_definition,
        u_gate_definition,
        v_gate_constructor(0),
        pyquil.gates.X(0),
        pyquil.gates.Y(1),
        pyquil.gates.Z(3),
        pyquil.gates.SWAP(0, 2).controlled(1),
        u_gate_constructor(1, 3).dagger(),
        pyquil.gates.RX(np.pi, 2),
        pyquil.gates.CNOT(1, 3),
    )

    assert expected_program == converted_program


def test_pyquil_circuit_obtained_from_empty_circuit_is_also_empty():
    circuit = Circuit()

    pyquil_circuit = convert_to_pyquil(circuit)

    assert not pyquil_circuit.instructions
