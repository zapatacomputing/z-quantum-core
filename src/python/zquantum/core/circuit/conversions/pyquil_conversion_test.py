"""Test cases for pyquil conversion."""
from copy import deepcopy
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

ORQUESTRA_SINGLE_QUBIT_ROTATION_GATES = [RX, RY, RZ, PHASE]

PYQUIL_SINGLE_QUBIT_ROTATION_GATES = [
    pyquil.gates.RX,
    pyquil.gates.RY,
    pyquil.gates.RZ,
    pyquil.gates.PHASE
]

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


@pytest.mark.parametrize("qubit_index", [0, 4, 10, 11])
@pytest.mark.parametrize("angle", [np.pi, np.pi / 2, 0.4])
class TestSingleQubitRotationGatesConversion:

    @pytest.mark.parametrize("orquestra_gate_cls", [RX, RY, RZ, PHASE])
    def test_conversion_from_orquestra_to_pyquil_preserves_qubit_index(
        self, qubit_index, angle, orquestra_gate_cls
    ):
        pyquil_gate = convert_to_pyquil(orquestra_gate_cls(qubit_index, angle))

        assert pyquil_gate.qubits == [pyquil.quil.Qubit(qubit_index)]

    @pytest.mark.parametrize(
        "orquestra_gate_cls", ORQUESTRA_SINGLE_QUBIT_ROTATION_GATES
    )
    def test_conversion_from_orquestra_to_pyquil_preserves_angle(
        self, qubit_index, angle, orquestra_gate_cls
    ):
        pyquil_gate = convert_to_pyquil(orquestra_gate_cls(qubit_index, angle))

        assert pyquil_gate.params == [angle]

    @pytest.mark.parametrize(
        "pyquil_gate_func", PYQUIL_SINGLE_QUBIT_ROTATION_GATES
    )
    def test_conversion_from_pyquil_to_orquestra_preserves_qubit_index(
        self, qubit_index, angle, pyquil_gate_func
    ):
        orquestra_gate = convert_from_pyquil(pyquil_gate_func(angle, qubit_index))

        assert orquestra_gate.qubits == (qubit_index,)

    @pytest.mark.parametrize(
        "pyquil_gate_func", PYQUIL_SINGLE_QUBIT_ROTATION_GATES
    )
    def test_conversion_from_pyquil_to_orquestra_preserves_angle(
        self, qubit_index, angle, pyquil_gate_func
    ):
        orquestra_gate = convert_from_pyquil(pyquil_gate_func(angle, qubit_index))

        assert orquestra_gate.angle == angle


@pytest.mark.parametrize("qubit_index", [0, 4, 10, 11])
@pytest.mark.parametrize("orquestra_angle, pyquil_angle", EXAMPLE_PARAMETRIZED_ANGLES)
class TestSingleQubitRotationGatesWithSymbolicParamsConversion:

    @pytest.mark.parametrize(
        "orquestra_gate_cls",
        ORQUESTRA_SINGLE_QUBIT_ROTATION_GATES
    )
    def test_angle_expression_is_translated_when_converting_from_orquestra_to_pyquil(
        self, qubit_index, orquestra_angle, pyquil_angle, orquestra_gate_cls
    ):
        program = pyquil.Program()
        pyquil_gate = convert_to_pyquil(orquestra_gate_cls(qubit_index, orquestra_angle), program)
        assert pyquil_gate.params == [pyquil_angle]

    @pytest.mark.parametrize(
        "orquestra_gate_cls",
        ORQUESTRA_SINGLE_QUBIT_ROTATION_GATES
    )
    def test_translated_angle_expression_is_added_to_program(
        self, qubit_index, orquestra_angle, pyquil_angle, orquestra_gate_cls
    ):
        program = pyquil.Program()
        zquantum_gate = orquestra_gate_cls(qubit_index, orquestra_angle)
        convert_to_pyquil(zquantum_gate, program)

        assert program.instructions == [
            pyquil.quil.Declare(str(param), "REAL")
            for param in zquantum_gate.symbolic_params
        ]

    @pytest.mark.parametrize("pyquil_gate_func", PYQUIL_SINGLE_QUBIT_ROTATION_GATES)
    def test_angle_expression_is_translated_when_converting_from_pyquil_to_orquestra(
        self, qubit_index, orquestra_angle, pyquil_angle, pyquil_gate_func
    ):
        orquestra_gate = convert_from_pyquil(
            pyquil_gate_func(pyquil_angle, qubit_index)
        )

        assert orquestra_gate.angle == orquestra_angle


@pytest.mark.parametrize("qubit_indices", [[0, 1], [2, 10], [4, 7]])
@pytest.mark.parametrize("angle", [0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi])
class TestCPHASEGateConversion:

    def test_conversion_from_orquestra_to_pyquil_preserves_qubit_indices(
        self, qubit_indices, angle
    ):
        pyquil_gate = convert_to_pyquil(CPHASE(*qubit_indices, angle))

        assert pyquil_gate.qubits == [pyquil.quil.Qubit(i) for i in qubit_indices]

    def test_conversion_from_orquestra_to_pyquil_preserves_angle(
        self, qubit_indices, angle
    ):
        pyquil_gate = convert_to_pyquil(CPHASE(*qubit_indices, angle))

        assert pyquil_gate.params == [angle]

    def test_conversion_from_pyquil_to_orquestra_preserves_qubit_indices(
        self, qubit_indices, angle
    ):
        orquestra_gate = convert_from_pyquil(pyquil.gates.CPHASE(angle, *qubit_indices))

        assert orquestra_gate.qubits == tuple(qubit_indices)

    def test_conversion_from_pyquil_to_orquestra_preserves_angle(
        self, qubit_indices, angle
    ):
        orquestra_gate = convert_from_pyquil(pyquil.gates.CPHASE(angle, *qubit_indices))

        assert orquestra_gate.angle == angle


@pytest.mark.parametrize("qubit_indices", [[0, 1], [2, 10], [4, 7]])
@pytest.mark.parametrize("orquestra_angle, pyquil_angle", EXAMPLE_PARAMETRIZED_ANGLES)
class TestCPHASEGateWithSymbolicParamsConversion:

    def test_angle_is_translated_when_converting_from_orquestra_to_pyquil(
        self, qubit_indices, orquestra_angle, pyquil_angle
    ):
        program = pyquil.Program()
        pyquil_gate = convert_to_pyquil(CPHASE(*qubit_indices, orquestra_angle), program)

        assert pyquil_gate.params == [pyquil_angle]

    def test_translated_angle_is_added_to_program(
        self, qubit_indices, orquestra_angle, pyquil_angle
    ):
        program = pyquil.Program()
        orquestra_gate = CPHASE(*qubit_indices, orquestra_angle)
        convert_to_pyquil(orquestra_gate, program)

        assert program.instructions == [
            pyquil.quil.Declare(str(param), "REAL")
            for param in orquestra_gate.symbolic_params
        ]

    def test_angle_is_translated_when_converting_from_pyquil_to_orquestra(
        self, qubit_indices, orquestra_angle, pyquil_angle
    ):
        orquestra_gate = convert_from_pyquil(
            pyquil.gates.CPHASE(pyquil_angle, *qubit_indices)
        )

        assert orquestra_gate.angle == orquestra_angle


@pytest.mark.parametrize("qubit_indices", [[0, 1], [2, 10], [4, 7]])
class TestSWAPGateConversion:

    def test_conversion_from_pyquil_to_orquestra_preserves_qubit_indices(
        self, qubit_indices
    ):
        pyquil_gate = convert_to_pyquil(SWAP(qubit_indices))

        assert pyquil_gate.qubits == [
            pyquil.quil.Qubit(i) for i in qubit_indices
        ]

    def test_conversion_from_orquestra_to_pyquil_preserves_qubit_indices(
        self, qubit_indices
    ):
        orquestra_gate = convert_from_pyquil(pyquil.gates.SWAP(*qubit_indices))

        assert orquestra_gate.qubits == tuple(qubit_indices)


@pytest.mark.parametrize("control, target", [(0, 1), (2, 3), (0, 10)])
class TestTwoQubitPredefinedControlledGatesConversion:

    @pytest.mark.parametrize("orquestra_gate_cls", [CZ, CNOT])
    def test_conversion_from_orquestra_to_qubit_preserves_qubit_indices(
        self, control, target, orquestra_gate_cls
    ):
        pyquil_gate = convert_to_pyquil(orquestra_gate_cls(control, target))

        assert pyquil_gate.qubits == [
            pyquil.quil.Qubit(control), pyquil.quil.Qubit(target)
        ]

    @pytest.mark.parametrize("pyquil_gate_func", [pyquil.gates.CZ, pyquil.gates.CNOT])
    def test_conversion_from_pyquil_to_orquestra_preserves_qubit_indices(
        self, control, target, pyquil_gate_func
    ):
        orquestra_gate = convert_from_pyquil(pyquil_gate_func(control, target))

        assert orquestra_gate.qubits == (control, target)


class TestCorrectnessOfGateTypeAndMatrix:

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
    def test_conversion_from_orquestra_to_pyquil_preserves_gate_type_and_matrix(
        self, gate
    ):
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
    "orquestra_gate, pyquil_gate, control_qubits",
    [
        (X(2), pyquil.gates.X(2), (1,)),
        (Y(1), pyquil.gates.Y(1), (0,)),
        (PHASE(4, np.pi), pyquil.gates.PHASE(np.pi, 4), (1, 2, 3)),
        (CZ(2, 12), pyquil.gates.CZ(2, 12), (0, 3))
    ],
    scope="function"
)
class TestControlledGateConversion:
    def make_orquestra_controlled_gate(self, gate, control_qubits):
        if control_qubits:
            return self.make_orquestra_controlled_gate(
                ControlledGate(gate, control_qubits[0]), control_qubits[1:]
            )
        return gate

    def make_pyquil_controlled_gate(self, gate, control_qubits):
        # Copy below is extremely important, because pyquil applies modifiers
        # in place, meaning that only a first function would get intended
        # params.
        gate = deepcopy(gate)
        if control_qubits:
            return self.make_pyquil_controlled_gate(
                gate.controlled(control_qubits[0]), control_qubits[1:]
            )
        return gate

    def test_converting_nested_controlled_gate_gives_pyquil_gate_with_applied_controlled_modifiers(
        self, orquestra_gate, pyquil_gate, control_qubits
    ):
        assert (
            self.make_pyquil_controlled_gate(pyquil_gate, control_qubits) ==
            convert_to_pyquil(
                self.make_orquestra_controlled_gate(orquestra_gate, control_qubits)
            )
        )

    def test_converting_pyquil_gate_with_controlled_modifiers_gives_nested_controlled_gate(
        self, orquestra_gate, pyquil_gate, control_qubits
    ):
        assert (
            self.make_orquestra_controlled_gate(orquestra_gate, control_qubits) ==
            convert_from_pyquil(
                self.make_pyquil_controlled_gate(pyquil_gate, control_qubits)
            )
        )


class TestGatesWithDaggerConversion:

    def test_dagger_object_gets_converted_to_gate_with_dagger_modifier(self):
        assert convert_to_pyquil(Dagger(X(1))) == pyquil.gates.X(1).dagger()

    @pytest.mark.parametrize(
        "pyquil_gate",
        [
            pyquil.gates.RX(np.pi / 4, 0),
            pyquil.gates.PHASE(0.5, 2),
            pyquil.gates.RX(0.5, 1).controlled(3),
        ]
    )
    def test_dagger_of_orquestra_gate_is_taken_when_converting_gate_with_dagger_modifier(
        self, pyquil_gate
    ):
        pyquil_dagger = deepcopy(pyquil_gate).dagger()
        assert (
            convert_from_pyquil(pyquil_dagger) ==
            convert_from_pyquil(pyquil_gate).dagger
        )


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
