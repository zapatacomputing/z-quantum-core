import numpy as np
import pytest
import json
import os
import copy
import sympy
import random


from ...utils import SCHEMA_VERSION
from .gates import X, Y, Z, H, I, RX, RY, RZ, PHASE, T, CustomGate, CNOT
from ._circuit import Circuit

CustomParameterizedGate = CustomGate(
    matrix=sympy.Matrix(
        [
            [1, 0, sympy.Symbol("theta_0"), 0],
            [0, 1, 0, sympy.Symbol("theta_0")],
            [sympy.Symbol("theta_0"), 0, 0, 1],
            [0, sympy.Symbol("theta_0"), 1, 0],
        ]
    ),
    qubits=(0, 2),
)


RandomGateList = [
    X(0),
    X(1),
    X(2),
    X(3),
    Y(0),
    Y(1),
    Y(2),
    Y(3),
    Z(0),
    Z(1),
    Z(2),
    Z(3),
    H(0),
    H(1),
    H(2),
    H(3),
    I(0),
    I(1),
    I(2),
    I(3),
    PHASE(0),
    PHASE(1),
    PHASE(2),
    PHASE(3),
    T(0),
    T(1),
    T(2),
    T(3),
    RX(0),
    RX(1),
    RX(2),
    RX(3),
    RX(0, sympy.Symbol("gamma")),
    RX(1, sympy.Symbol("gamma")),
    RX(2, sympy.Symbol("gamma")),
    RX(3, sympy.Symbol("gamma")),
    RX(0, 0),
    RX(1, 0),
    RX(2, 0),
    RX(3, 0),
    RX(0, 0.5),
    RX(1, 0.5),
    RX(2, 0.5),
    RX(3, 0.5),
    RX(0, np.pi / 2),
    RX(1, np.pi / 2),
    RX(2, np.pi / 2),
    RX(3, np.pi / 2),
    RX(0, np.pi),
    RX(1, np.pi),
    RX(2, np.pi),
    RX(3, np.pi),
    RX(0, 2 * np.pi),
    RX(1, 2 * np.pi),
    RX(2, 2 * np.pi),
    RX(3, 2 * np.pi),
    RY(0),
    RY(1),
    RY(2),
    RY(3),
    RY(0, sympy.Symbol("gamma")),
    RY(1, sympy.Symbol("gamma")),
    RY(2, sympy.Symbol("gamma")),
    RY(3, sympy.Symbol("gamma")),
    RY(0, 0),
    RY(1, 0),
    RY(2, 0),
    RY(3, 0),
    RY(0, 0.5),
    RY(1, 0.5),
    RY(2, 0.5),
    RY(3, 0.5),
    RY(0, np.pi / 2),
    RY(1, np.pi / 2),
    RY(2, np.pi / 2),
    RY(3, np.pi / 2),
    RY(0, np.pi),
    RY(1, np.pi),
    RY(2, np.pi),
    RY(3, np.pi),
    RY(0, 2 * np.pi),
    RY(1, 2 * np.pi),
    RY(2, 2 * np.pi),
    RY(3, 2 * np.pi),
    RZ(0),
    RZ(1),
    RZ(2),
    RZ(3),
    RZ(0, sympy.Symbol("gamma")),
    RZ(1, sympy.Symbol("gamma")),
    RZ(2, sympy.Symbol("gamma")),
    RZ(3, sympy.Symbol("gamma")),
    RZ(0, 0),
    RZ(1, 0),
    RZ(2, 0),
    RZ(3, 0),
    RZ(0, 0.5),
    RZ(1, 0.5),
    RZ(2, 0.5),
    RZ(3, 0.5),
    RZ(0, np.pi / 2),
    RZ(1, np.pi / 2),
    RZ(2, np.pi / 2),
    RZ(3, np.pi / 2),
    RZ(0, np.pi),
    RZ(1, np.pi),
    RZ(2, np.pi),
    RZ(3, np.pi),
    RZ(0, 2 * np.pi),
    RZ(1, 2 * np.pi),
    RZ(2, 2 * np.pi),
    RZ(3, 2 * np.pi),
    CNOT(0, 1),
    CNOT(0, 9),
    CustomParameterizedGate,
]


RNG = np.random.default_rng(42)


CIRCUITS = [
    Circuit(),
    Circuit(gates=[X(0)]),
    Circuit(gates=[X(1)]),
    Circuit(gates=[X(0), X(1)]),
    Circuit(
        gates=[
            H(0),
            CNOT(0, 1),
            RX(0),
            CNOT(0, 1),
            H(0),
        ]
    ),
    Circuit(gates=[CustomParameterizedGate]),
    Circuit(
        gates=[
            RX(0),
            RY(0),
            RZ(0),
            CustomParameterizedGate,
        ]
    ),
    Circuit(gates=[I(0) for _ in range(100)]),
    Circuit(gates=RNG.choice(RandomGateList, size=100, replace=True)),
    Circuit(gates=RNG.choice(RandomGateList, size=1000, replace=True)),
]


#### __init__ ####


@pytest.mark.parametrize(
    "gates",
    [
        [],
        [X(0)],
        [H(0)],
        [X(0), X(0)],
        [CNOT(0, 1)],
        [X(0), X(0), CNOT(0, 1), X(0)],
        [random.choice(RandomGateList) for _ in range(100)],
    ],
)
def test_creating_circuit_has_correct_gates(gates):
    """The Circuit class should have the correct gates that are passed in"""
    # When
    circuit = Circuit(gates=gates)
    # Then
    assert circuit.gates == gates


def test_appending_to_circuit_works():
    """The Circuit class should have the correct gates that are passed in"""
    # Given
    expected_circuit = Circuit(gates=[H(0), CNOT(0, 1)])
    # When
    circuit = Circuit()
    circuit += H(0)
    circuit += CNOT(0, 1)
    # Then
    assert circuit.gates == expected_circuit.gates
    assert circuit.n_qubits == expected_circuit.n_qubits


#### qubits ####


@pytest.mark.parametrize(
    "gates, n_qubits",
    [
        ([], 0),
        ([X(0)], 1),
        ([X(1)], 2),
        ([X(0), X(1)], 2),
        ([CNOT(0, 1)], 2),
        ([X(0), X(0), CNOT(0, 1), X(0)], 2),
    ],
)
def test_creating_circuit_has_correct_number_of_qubits(gates, n_qubits):
    """The Circuit class should have the correct qubits based on the gates that are passed in"""
    # When
    circuit = Circuit(gates=gates)
    # Then
    assert circuit.n_qubits == n_qubits


def test_creating_circuit_has_correct_qubits_with_gaps():
    """The Circuit class should have the correct qubits even if there is a gap in the qubit indices"""
    # Given/When
    circuit = Circuit(gates=[X(0), CNOT(0, 1), CNOT(0, 9)])

    # Then
    assert circuit.n_qubits == 10


#### symbolic_params ####
def test_symbolic_params_are_empty_with_no_parameterized_gates():
    # Given
    circuit = Circuit(
        gates=[
            X(0),
            CNOT(0, 1),
            X(0),
            H(0),
            CNOT(0, 1),
        ]
    )

    # When/Then
    assert len(circuit.symbolic_params) == 0


def test_symbolic_params_are_correct_for_one_gate_one_parameter():
    # Given
    matrix = sympy.Matrix(
        [
            [1, 0, sympy.Symbol("theta_0"), 0],
            [0, 1, 0, sympy.Symbol("theta_0")],
            [sympy.Symbol("theta_0"), 0, 0, 1],
            [0, sympy.Symbol("theta_0"), 1, 0],
        ]
    )
    gate = CustomGate(matrix=matrix, qubits=(0, 2))
    circuit = Circuit(gates=[gate])

    # When/Then
    assert circuit.symbolic_params == {sympy.Symbol("theta_0")}


def test_symbolic_params_are_correct_for_one_gate_two_parameters():
    # Given
    gate = CustomGate(
        matrix=sympy.Matrix(
            [
                [1, 0, sympy.Symbol("theta_0"), 0],
                [0, 1, 0, sympy.Symbol("theta_1")],
                [sympy.Symbol("theta_1"), 0, 0, 1],
                [0, sympy.Symbol("theta_0"), 1, 0],
            ]
        ),
        qubits=(0, 2),
    )
    circuit = Circuit(gates=[gate])

    # When/Then
    assert circuit.symbolic_params == {sympy.Symbol("theta_0"), sympy.Symbol("theta_1")}


def test_symbolic_params_are_correct_for_multiple_gates_with_overlapping_parameters():
    # Given
    gate1 = CustomGate(
        matrix=sympy.Matrix(
            [
                [1, 0, sympy.Symbol("theta_0"), 0],
                [0, 1, 0, sympy.Symbol("theta_1")],
                [sympy.Symbol("theta_1"), 0, 0, 1],
                [0, sympy.Symbol("theta_0"), 1, 0],
            ]
        ),
        qubits=(0, 2),
    )
    gate2 = CustomGate(
        matrix=sympy.Matrix(
            [
                [sympy.Symbol("theta_0"), 0],
                [0, sympy.Symbol("theta_1")],
            ]
        ),
        qubits=(1,),
    )
    gate3 = CustomGate(
        matrix=sympy.Matrix(
            [
                [sympy.Symbol("theta_0"), 0],
                [0, sympy.Symbol("theta_0")],
            ]
        ),
        qubits=(1,),
    )
    gate4 = CustomGate(
        matrix=sympy.Matrix(
            [
                [sympy.Symbol("gamma_0"), 0],
                [0, sympy.Symbol("gamma_1")],
            ]
        ),
        qubits=(1,),
    )

    # When
    circuit = Circuit(gates=[gate1, gate2, gate3, gate4])

    # Then
    assert circuit.symbolic_params == {
        sympy.Symbol("theta_0"),
        sympy.Symbol("theta_1"),
        sympy.Symbol("gamma_0"),
        sympy.Symbol("gamma_1"),
    }


#### __eq__ ####
@pytest.mark.parametrize(
    "circuit1, circuit2",
    [
        [
            Circuit(),
            Circuit(),
        ],
        [
            Circuit(gates=[X(0), H(0), CNOT(0, 1)]),
            Circuit(gates=[X(0), H(0), CNOT(0, 1)]),
        ],
        [
            Circuit(gates=[X(0), H(0), CNOT(0, 1)]),
            Circuit(gates=[copy.deepcopy(X(0)), H(0), CNOT(0, 1)]),
        ],
        [
            Circuit(
                gates=[
                    X(0),
                    H(0),
                    CNOT(0, 1),
                    CustomParameterizedGate,
                ]
            ),
            Circuit(
                gates=[
                    X(0),
                    H(0),
                    CNOT(0, 1),
                    CustomParameterizedGate,
                ]
            ),
        ],
    ],
)
def test_circuit_eq_same_gates(circuit1, circuit2):
    assert circuit1 == circuit2


@pytest.mark.parametrize(
    "circuit1, circuit2",
    [
        [
            Circuit(),
            Circuit(gates=[H(0)]),
        ],
        [
            Circuit(gates=[H(0)]),
            Circuit(),
        ],
        [
            Circuit(
                gates=[
                    X(0),
                    H(0),
                    CNOT(0, 1),
                    CustomParameterizedGate,
                ]
            ),
            Circuit(gates=[X(0), H(0), CNOT(0, 1)]),
        ],
        [
            Circuit(gates=[X(0), H(0), CNOT(0, 1)]),
            Circuit(
                gates=[
                    X(0),
                    H(0),
                    CNOT(0, 1),
                    CustomParameterizedGate,
                ]
            ),
        ],
        [
            Circuit(gates=[H(0), X(1), CNOT(0, 1)]),
            Circuit(gates=[X(1), H(0), CNOT(0, 1)]),
        ],
        [
            Circuit(
                gates=[
                    CustomGate(
                        matrix=sympy.Matrix(
                            [
                                [sympy.Symbol("theta_0"), 0],
                                [0, sympy.Symbol("theta_1")],
                            ]
                        ),
                        qubits=(0,),
                    )
                ]
            ),
            Circuit(
                gates=[
                    CustomGate(
                        matrix=sympy.Matrix(
                            [
                                [sympy.Symbol("theta_1"), 0],
                                [0, sympy.Symbol("theta_0")],
                            ]
                        ),
                        qubits=(0,),
                    )
                ]
            ),
        ],
        [
            Circuit(
                gates=[
                    CustomGate(
                        matrix=sympy.Matrix(
                            [
                                [sympy.Symbol("theta_0"), 0],
                                [0, sympy.Symbol("theta_1")],
                            ]
                        ),
                        qubits=(0,),
                    )
                ]
            ),
            Circuit(
                gates=[
                    CustomGate(
                        matrix=sympy.Matrix(
                            [
                                [sympy.Symbol("gamma_0"), 0],
                                [0, sympy.Symbol("gamma_1")],
                            ]
                        ),
                        qubits=(0,),
                    )
                ]
            ),
        ],
    ],
)
def test_gate_eq_not_same_gates(circuit1, circuit2):
    """The Circuit class should be able to be able to compare two unequal circuits"""
    assert circuit1 != circuit2


#### __add__ ####
@pytest.mark.parametrize(
    "circuit1, circuit2, expected_circuit",
    [
        [
            Circuit(),
            Circuit(gates=[H(0)]),
            Circuit(gates=[H(0)]),
        ],
        [
            Circuit(),
            Circuit(),
            Circuit(),
        ],
        [
            Circuit(gates=[H(0)]),
            Circuit(),
            Circuit(gates=[H(0)]),
        ],
        [
            Circuit(gates=[H(0), CNOT(0, 1)]),
            Circuit(gates=[CNOT(0, 1), H(0)]),
            Circuit(gates=[H(0), CNOT(0, 1), CNOT(0, 1), H(0)]),
        ],
        [
            Circuit(gates=[H(0), CNOT(0, 1), CustomParameterizedGate]),
            Circuit(gates=[CNOT(0, 1), H(0)]),
            Circuit(
                gates=[
                    H(0),
                    CNOT(0, 1),
                    CustomParameterizedGate,
                    CNOT(0, 1),
                    H(0),
                ]
            ),
        ],
    ],
)
def test_add_circuits(circuit1, circuit2, expected_circuit):
    """The Circuit class should be able to handling adding circuits together"""
    assert circuit1 + circuit2 == expected_circuit


# Evaluation


def test_circuit_evaluated_with_all_params_comprises_evaluated_gates():
    theta1, theta2, theta3 = sympy.symbols("theta1:4")
    circuit = Circuit(
        gates=[
            RX(0, theta1),
            RY(1, theta2),
            RZ(0, theta3),
            RX(0, theta3),
        ]
    )

    symbols_map = {theta1: 0.5, theta2: 3.14, theta3: 0}

    expected_circuit = Circuit(
        gates=[
            RX(0, theta1).evaluate(symbols_map),
            RY(1, theta2).evaluate(symbols_map),
            RZ(0, theta3).evaluate(symbols_map),
            RX(0, theta3).evaluate(symbols_map),
        ]
    )

    evaluated_circuit = circuit.evaluate(symbols_map)

    assert evaluated_circuit == expected_circuit


def test_circuit_evaluated_with_all_params_has_no_free_params():
    alpha, beta, gamma = sympy.symbols("alpha,beta,gamma")
    circuit = Circuit(
        gates=[
            RX(0, alpha),
            RY(1, beta),
            RZ(0, gamma),
            RX(0, gamma),
        ]
    )

    symbols_map = {alpha: 0.5, beta: 3.14, gamma: 0}

    evaluated_circuit = circuit.evaluate(symbols_map)

    assert not evaluated_circuit.symbolic_params


@pytest.mark.xfail
def test_circuit_evaluate_with_too_many_params_specified():
    # Given
    symbols_map = {
        sympy.Symbol("theta_0"): 0.5,
        sympy.Symbol("theta_1"): 0.6,
        sympy.Symbol("theta_2"): 0.7
    }
    RYGateQubit0 = RY(0).evaluate(symbols_map)
    RZGateQubit0 = RZ(0).evaluate(symbols_map)
    RZGateQubit0DifferentAngle = RZ(0).evaluate({sympy.Symbol("theta_1"): 0.4})
    circuit = Circuit(
        gates=[
            RX(0),
            RY(0),
            RZ(0),
            RZGateQubit0DifferentAngle,
        ]
    )
    target_circuit = Circuit(
        gates=[
            RX(0),
            RYGateQubit0,
            RZGateQubit0,
            RZGateQubit0DifferentAngle,
        ]
    )

    # When/Then
    with pytest.warns(Warning):
        evaluated_circuit = circuit.evaluate(symbols_map)
    assert evaluated_circuit == target_circuit


def test_circuit_evaluate_with_some_params_specified():
    # Given
    symbols_map = {sympy.Symbol("theta_0"): 0.5}
    RYGateQubit0 = RY(0).evaluate(symbols_map)
    RZGateQubit0 = RZ(0).evaluate(symbols_map)
    RZGateQubit0DifferentAngle = RZ(0).evaluate({sympy.Symbol("theta_1"): 0.4})
    circuit = Circuit(
        gates=[
            RX(0),
            RY(0),
            RZ(0),
            RZGateQubit0DifferentAngle,
        ]
    )
    target_circuit = Circuit(
        gates=[
            RX(0),
            RYGateQubit0,
            RZGateQubit0,
            RZGateQubit0DifferentAngle,
        ]
    )

    # When
    evaluated_circuit = circuit.evaluate(symbols_map)

    # Then
    assert evaluated_circuit == target_circuit


def test_circuit_evaluate_with_wrong_params():
    # Given
    symbols_map = {sympy.Symbol("theta_2"): 0.7}
    RZGateQubit0DifferentAngle = RZ(0).evaluate({sympy.Symbol("theta_1"): 0.4})
    circuit = Circuit(
        gates=[
            RX(0),
            RY(0),
            RZ(0),
            RZGateQubit0DifferentAngle,
        ]
    )
    target_circuit = Circuit(
        gates=[
            RX(0),
            RY(0),
            RZ(0),
            RZGateQubit0DifferentAngle,
        ]
    )

    # When
    evaluated_circuit = circuit.evaluate(symbols_map)

    # Then
    assert evaluated_circuit == target_circuit


#### to_dict ####


@pytest.mark.parametrize("circuit", CIRCUITS)
def test_gate_is_successfully_converted_to_dict_form(circuit):
    """The Circuit class should be able to be converted to a serializable dict with the underlying gates
    also converted to serializable dictionaries"""
    # When
    circuit_dict = circuit.to_dict()

    # Then
    assert circuit_dict["schema"] == SCHEMA_VERSION + "-circuit"
    assert circuit_dict["n_qubits"] == circuit.n_qubits
    assert circuit_dict["symbolic_params"] == [
        str(param) for param in circuit.symbolic_params
    ]
    assert isinstance(circuit_dict["gates"], list)
    for gate_dict, gate in zip(circuit_dict["gates"], circuit.gates):
        assert gate_dict == gate.to_dict()


#### save ####
@pytest.mark.parametrize("circuit", CIRCUITS)
def test_circuit_is_successfully_saved_to_a_file(circuit):
    # When
    circuit.save("circuit.json")
    with open("circuit.json", "r") as f:
        saved_data = json.loads(f.read())

    # Then
    assert saved_data["schema"] == SCHEMA_VERSION + "-circuit"
    assert saved_data["n_qubits"] == circuit.n_qubits
    assert saved_data["gates"] == [
        gate.to_dict() for gate in circuit.gates
    ]
    assert saved_data["symbolic_params"] == [
        str(param) for param in circuit.symbolic_params
    ]

    os.remove("circuit.json")


#### load ####
@pytest.mark.parametrize("circuit", CIRCUITS)
def test_circuit_is_successfully_loaded_from_a_file(circuit):
    # Given
    circuit.save("circuit.json")

    # When
    new_circuit = Circuit.load("circuit.json")

    # Then
    assert circuit == new_circuit

    os.remove("circuit.json")


@pytest.mark.parametrize("circuit", CIRCUITS)
def test_circuit_is_successfully_loaded_from_a_dict(circuit):
    # Given
    circuit_dict = circuit.to_dict()

    # When
    new_circuit = Circuit.load(circuit_dict)

    # Then
    assert circuit == new_circuit
