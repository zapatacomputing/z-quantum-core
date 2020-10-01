import numpy as np
import pytest
import json
import os
import copy
import sympy
import random
from ...utils import SCHEMA_VERSION
from ..gate import Gate
from ._circuit import Circuit

XGateQubit0 = Gate(
    matrix=sympy.Matrix(
        [[complex(0, 0), complex(1, 0)], [complex(1, 0), complex(0, 0)],]
    ),
    qubits=(0,),
)
HGateQubit0 = Gate(
    matrix=sympy.Matrix(
        [
            [(1 / np.sqrt(2)) * complex(1, 0), (1 / np.sqrt(2)) * complex(1, 0)],
            [(1 / np.sqrt(2)) * complex(1, 0), (1 / np.sqrt(2)) * complex(-1, 0)],
        ]
    ),
    qubits=(0,),
)
XGateQubit1 = Gate(
    matrix=sympy.Matrix(
        [[complex(0, 0), complex(1, 0)], [complex(1, 0), complex(0, 0)],]
    ),
    qubits=(1,),
)
CNOTGateQubits01 = Gate(
    matrix=sympy.Matrix(
        [
            [complex(1, 0), complex(0, 0), complex(0, 0), complex(0, 0)],
            [complex(0, 0), complex(1, 0), complex(0, 0), complex(0, 0)],
            [complex(0, 0), complex(0, 0), complex(0, 0), complex(1, 0)],
            [complex(0, 0), complex(0, 0), complex(1, 0), complex(0, 0)],
        ]
    ),
    qubits=(0, 1,),
)
IGateQubit0 = Gate(matrix=sympy.Matrix(np.eye(2)), qubits=(0,))
IGateQubit1 = Gate(matrix=sympy.Matrix(np.eye(2)), qubits=(1,))
ParameterizedGateQubits02 = Gate(
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
ParameterizedRXGateQubit0 = Gate(
    matrix=sympy.Matrix(
        [
            [
                sympy.cos(sympy.Symbol("theta_0") / 2),
                complex(0, -1) * sympy.sin(sympy.Symbol("theta_0") / 2),
            ],
            [
                complex(0, -1) * sympy.sin(sympy.Symbol("theta_0") / 2),
                sympy.cos(sympy.Symbol("theta_0") / 2),
            ],
        ]
    ),
    qubits=(0,),
)
ParameterizedRYGateQubit0 = Gate(
    matrix=sympy.Matrix(
        [
            [
                sympy.cos(sympy.Symbol("theta_0") / 2),
                -1 * sympy.sin(sympy.Symbol("theta_0") / 2),
            ],
            [
                sympy.sin(sympy.Symbol("theta_0") / 2),
                sympy.cos(sympy.Symbol("theta_0") / 2),
            ],
        ]
    ),
    qubits=(0,),
)
ParameterizedRZGateQubit0 = Gate(
    matrix=sympy.Matrix(
        [
            [sympy.exp(complex(0, -1) * sympy.Symbol("theta_1") / 2), complex(0, 0),],
            [complex(0, 0), sympy.exp(complex(0, 1) * sympy.Symbol("theta_1") / 2)],
        ]
    ),
    qubits=(0,),
)

RandomGateList = [
    XGateQubit0,
    XGateQubit1,
    HGateQubit0,
    CNOTGateQubits01,
    IGateQubit0,
    IGateQubit1,
    ParameterizedGateQubits02,
    ParameterizedRXGateQubit0,
    ParameterizedRYGateQubit0,
    ParameterizedRZGateQubit0,
]

#### __init__ ####
@pytest.mark.parametrize(
    "gates",
    [
        [],
        [XGateQubit0],
        [HGateQubit0],
        [XGateQubit0, XGateQubit0],
        [CNOTGateQubits01],
        [XGateQubit0, XGateQubit0, CNOTGateQubits01, XGateQubit0],
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
    expected_circuit = Circuit(gates=[HGateQubit0, CNOTGateQubits01])
    # When
    circuit = Circuit(gates=[])
    circuit.gates.append(HGateQubit0)
    circuit.gates.append(CNOTGateQubits01)
    # Then
    assert circuit.gates == expected_circuit.gates
    assert circuit.qubits == expected_circuit.qubits


#### qubits ####
@pytest.mark.parametrize(
    "gates, qubits",
    [
        [[], tuple()],
        [[XGateQubit0], (0,)],
        [[XGateQubit1], (1,)],
        [[XGateQubit0, XGateQubit1], (0, 1,)],
        [[CNOTGateQubits01], (0, 1)],
        [[XGateQubit0, XGateQubit0, CNOTGateQubits01, XGateQubit0], (0, 1)],
    ],
)
def test_creating_circuit_has_correct_qubits(gates, qubits):
    """The Circuit class should have the correct qubits based on the gates that are passed in"""
    # When
    circuit = Circuit(gates=gates)
    # Then
    assert circuit.qubits == qubits


def test_creating_circuit_has_correct_qubits_with_gaps():
    """The Circuit class should have the correct qubits even if there is a gap in the qubit indices"""
    # Given/When
    CNOTGateQubits01_0_9 = copy.deepcopy(CNOTGateQubits01)
    CNOTGateQubits01_0_9.qubits = (0, 9)
    circuit = Circuit(gates=[XGateQubit0, CNOTGateQubits01, CNOTGateQubits01_0_9])

    # Then
    assert circuit.qubits == (0, 1, 9)


#### symbolic_params ####
def test_symbolic_params_are_empty_with_no_parameterized_gates():
    # Given
    circuit = Circuit(
        gates=[
            XGateQubit0,
            CNOTGateQubits01,
            XGateQubit0,
            HGateQubit0,
            CNOTGateQubits01,
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
    gate = Gate(matrix=matrix, qubits=(0, 2))
    circuit = Circuit(gates=[gate])

    # When/Then
    assert circuit.symbolic_params == set([sympy.Symbol("theta_0")])


def test_symbolic_params_are_correct_for_one_gate_two_parameters():
    # Given
    gate = Gate(
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
    assert circuit.symbolic_params == set(
        [sympy.Symbol("theta_0"), sympy.Symbol("theta_1")]
    )


def test_symbolic_params_are_correct_for_multiple_gates_with_overlapping_parameters():
    # Given
    gate1 = Gate(
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
    gate2 = Gate(
        matrix=sympy.Matrix(
            [[sympy.Symbol("theta_0"), 0], [0, sympy.Symbol("theta_1")],]
        ),
        qubits=(1,),
    )
    gate3 = Gate(
        matrix=sympy.Matrix(
            [[sympy.Symbol("theta_0"), 0], [0, sympy.Symbol("theta_0")],]
        ),
        qubits=(1,),
    )
    gate4 = Gate(
        matrix=sympy.Matrix(
            [[sympy.Symbol("gamma_0"), 0], [0, sympy.Symbol("gamma_1")],]
        ),
        qubits=(1,),
    )

    # When
    circuit = Circuit(gates=[gate1, gate2, gate3, gate4])

    # Then
    assert circuit.symbolic_params == set(
        [
            sympy.Symbol("theta_0"),
            sympy.Symbol("theta_1"),
            sympy.Symbol("gamma_0"),
            sympy.Symbol("gamma_1"),
        ]
    )


#### __eq__ ####
@pytest.mark.parametrize(
    "circuit1, circuit2",
    [
        [Circuit(gates=[]), Circuit(gates=[]),],
        [
            Circuit(gates=[XGateQubit0, HGateQubit0, CNOTGateQubits01]),
            Circuit(gates=[XGateQubit0, HGateQubit0, CNOTGateQubits01]),
        ],
        [
            Circuit(gates=[XGateQubit0, HGateQubit0, CNOTGateQubits01]),
            Circuit(gates=[copy.deepcopy(XGateQubit0), HGateQubit0, CNOTGateQubits01]),
        ],
        [
            Circuit(
                gates=[
                    XGateQubit0,
                    HGateQubit0,
                    CNOTGateQubits01,
                    ParameterizedGateQubits02,
                ]
            ),
            Circuit(
                gates=[
                    XGateQubit0,
                    HGateQubit0,
                    CNOTGateQubits01,
                    ParameterizedGateQubits02,
                ]
            ),
        ],
    ],
)
def test_circuit_eq_same_gates(circuit1, circuit2):
    """The Circuit class should be able to be able to compare two equal circuit"""
    # When
    are_equal = circuit1 == circuit2

    # Then
    assert are_equal


@pytest.mark.parametrize(
    "circuit1, circuit2",
    [
        [Circuit(gates=[]), Circuit(gates=[HGateQubit0]),],
        [Circuit(gates=[HGateQubit0]), Circuit(gates=[]),],
        [
            Circuit(
                gates=[
                    XGateQubit0,
                    HGateQubit0,
                    CNOTGateQubits01,
                    ParameterizedGateQubits02,
                ]
            ),
            Circuit(gates=[XGateQubit0, HGateQubit0, CNOTGateQubits01]),
        ],
        [
            Circuit(gates=[XGateQubit0, HGateQubit0, CNOTGateQubits01]),
            Circuit(
                gates=[
                    XGateQubit0,
                    HGateQubit0,
                    CNOTGateQubits01,
                    ParameterizedGateQubits02,
                ]
            ),
        ],
        [
            Circuit(gates=[HGateQubit0, XGateQubit1, CNOTGateQubits01]),
            Circuit(gates=[XGateQubit1, HGateQubit0, CNOTGateQubits01]),
        ],
        [
            Circuit(
                gates=[
                    Gate(
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
                    Gate(
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
                    Gate(
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
                    Gate(
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
    # When
    are_equal = circuit1 == circuit2

    # Then
    assert not are_equal


#### __add__ ####
@pytest.mark.parametrize(
    "circuit1, circuit2, expected_circuit",
    [
        [
            Circuit(gates=[]),
            Circuit(gates=[HGateQubit0]),
            Circuit(gates=[HGateQubit0]),
        ],
        [Circuit(gates=[]), Circuit(gates=[]), Circuit(gates=[]),],
        [
            Circuit(gates=[HGateQubit0]),
            Circuit(gates=[]),
            Circuit(gates=[HGateQubit0]),
        ],
        [
            Circuit(gates=[HGateQubit0, CNOTGateQubits01]),
            Circuit(gates=[CNOTGateQubits01, HGateQubit0]),
            Circuit(
                gates=[HGateQubit0, CNOTGateQubits01, CNOTGateQubits01, HGateQubit0]
            ),
        ],
        [
            Circuit(gates=[HGateQubit0, CNOTGateQubits01, ParameterizedGateQubits02]),
            Circuit(gates=[CNOTGateQubits01, HGateQubit0]),
            Circuit(
                gates=[
                    HGateQubit0,
                    CNOTGateQubits01,
                    ParameterizedGateQubits02,
                    CNOTGateQubits01,
                    HGateQubit0,
                ]
            ),
        ],
    ],
)
def test_add_circuits(circuit1, circuit2, expected_circuit):
    """The Circuit class should be able to handling adding circuits together"""
    # When
    new_circuit = circuit1 + circuit2

    # Then
    assert new_circuit == expected_circuit


#### evaluate ####
def test_circuit_evaluate_with_all_params_specified():
    # Given
    symbols_map = {"theta_0": 0.5, "theta_1": 0.6}
    RXGateQubit0 = ParameterizedRXGateQubit0.evaluate(symbols_map)
    RYGateQubit0 = ParameterizedRYGateQubit0.evaluate(symbols_map)
    RZGateQubit0 = ParameterizedRZGateQubit0.evaluate(symbols_map)
    RZGateQubit0DifferentAngle = ParameterizedRZGateQubit0.evaluate({"theta_1": 0.4})
    circuit = Circuit(
        gates=[
            ParameterizedRXGateQubit0,
            ParameterizedRYGateQubit0,
            ParameterizedRZGateQubit0,
            RZGateQubit0DifferentAngle,
        ]
    )

    target_circuit = Circuit(
        gates=[RXGateQubit0, RYGateQubit0, RZGateQubit0, RZGateQubit0DifferentAngle]
    )

    # When
    evaluated_circuit = circuit.evaluate(symbols_map)

    # Then
    assert evaluated_circuit == target_circuit


def test_circuit_evaluate_with_too_many_params_specified():
    # Given
    symbols_map = {"theta_0": 0.5, "theta_1": 0.6, "theta_2": 0.7}
    RXGateQubit0 = ParameterizedRXGateQubit0.evaluate(symbols_map)
    RYGateQubit0 = ParameterizedRYGateQubit0.evaluate(symbols_map)
    RZGateQubit0 = ParameterizedRZGateQubit0.evaluate(symbols_map)
    RZGateQubit0DifferentAngle = ParameterizedRZGateQubit0.evaluate({"theta_1": 0.4})
    circuit = Circuit(
        gates=[
            ParameterizedRXGateQubit0,
            ParameterizedRYGateQubit0,
            ParameterizedRZGateQubit0,
            RZGateQubit0DifferentAngle,
        ]
    )
    target_circuit = Circuit(
        gates=[RXGateQubit0, RYGateQubit0, RZGateQubit0, RZGateQubit0DifferentAngle,]
    )

    # When/Then
    with pytest.warns(Warning):
        evaluated_circuit = circuit.evaluate(symbols_map)
    assert evaluated_circuit == target_circuit


def test_circuit_evaluate_with_some_params_specified():
    # Given
    symbols_map = {"theta_0": 0.5}
    RXGateQubit0 = ParameterizedRXGateQubit0.evaluate(symbols_map)
    RYGateQubit0 = ParameterizedRYGateQubit0.evaluate(symbols_map)
    RZGateQubit0 = ParameterizedRZGateQubit0.evaluate(symbols_map)
    RZGateQubit0DifferentAngle = ParameterizedRZGateQubit0.evaluate({"theta_1": 0.4})
    circuit = Circuit(
        gates=[
            ParameterizedRXGateQubit0,
            ParameterizedRYGateQubit0,
            ParameterizedRZGateQubit0,
            RZGateQubit0DifferentAngle,
        ]
    )
    target_circuit = Circuit(
        gates=[RXGateQubit0, RYGateQubit0, RZGateQubit0, RZGateQubit0DifferentAngle,]
    )

    # When
    evaluated_circuit = circuit.evaluate(symbols_map)

    # Then
    assert evaluated_circuit == target_circuit


def test_circuit_evaluate_with_wrong_params():
    # Given
    symbols_map = {"theta_2": 0.7}
    RXGateQubit0 = ParameterizedRXGateQubit0.evaluate(symbols_map)
    RYGateQubit0 = ParameterizedRYGateQubit0.evaluate(symbols_map)
    RZGateQubit0 = ParameterizedRZGateQubit0.evaluate(symbols_map)
    RZGateQubit0DifferentAngle = ParameterizedRZGateQubit0.evaluate({"theta_1": 0.4})
    circuit = Circuit(
        gates=[
            ParameterizedRXGateQubit0,
            ParameterizedRYGateQubit0,
            ParameterizedRZGateQubit0,
            RZGateQubit0DifferentAngle,
        ]
    )
    target_circuit = Circuit(
        gates=[
            ParameterizedRXGateQubit0,
            ParameterizedRYGateQubit0,
            ParameterizedRZGateQubit0,
            RZGateQubit0DifferentAngle,
        ]
    )

    # When
    evaluated_circuit = circuit.evaluate(symbols_map)

    # Then
    assert evaluated_circuit == target_circuit


#### to_dict ####
@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(gates=[]),
        Circuit(gates=[XGateQubit0]),
        Circuit(gates=[XGateQubit1]),
        Circuit(gates=[XGateQubit0, XGateQubit1]),
        Circuit(
            gates=[
                HGateQubit0,
                CNOTGateQubits01,
                ParameterizedRXGateQubit0,
                CNOTGateQubits01,
                HGateQubit0,
            ]
        ),
        Circuit(gates=[ParameterizedGateQubits02]),
        Circuit(
            gates=[
                ParameterizedRXGateQubit0,
                ParameterizedRYGateQubit0,
                ParameterizedRZGateQubit0,
                ParameterizedGateQubits02,
            ]
        ),
        Circuit(gates=[IGateQubit0 for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(1000)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(10000)]),
    ],
)
def test_circuit_is_successfully_converted_to_dict_form(circuit):
    """The Circuit class should be able to be converted to a dict with the underlying gates
    also converted to dictionaries"""
    # When
    circuit_dict = circuit.to_dict(serializable=False)

    # Then
    assert circuit_dict["schema"] == SCHEMA_VERSION + "-circuit"
    assert circuit_dict["qubits"] == circuit.qubits
    assert circuit_dict["symbolic_params"] == circuit.symbolic_params
    assert isinstance(circuit_dict["gates"], list)
    for gate_dict, gate in zip(circuit_dict["gates"], circuit.gates):
        assert gate_dict == gate.to_dict(serializable=False)


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(gates=[]),
        Circuit(gates=[XGateQubit0]),
        Circuit(gates=[XGateQubit1]),
        Circuit(gates=[XGateQubit0, XGateQubit1]),
        Circuit(
            gates=[
                HGateQubit0,
                CNOTGateQubits01,
                ParameterizedRXGateQubit0,
                CNOTGateQubits01,
                HGateQubit0,
            ]
        ),
        Circuit(gates=[ParameterizedGateQubits02]),
        Circuit(
            gates=[
                ParameterizedRXGateQubit0,
                ParameterizedRYGateQubit0,
                ParameterizedRZGateQubit0,
                ParameterizedGateQubits02,
            ]
        ),
        Circuit(gates=[IGateQubit0 for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(1000)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(10000)]),
    ],
)
def test_gate_is_successfully_converted_to_serializable_dict_form(circuit):
    """The Circuit class should be able to be converted to a serializble dict with the underlying gates
    also converted to serializable dictionaries"""
    # When
    circuit_dict = circuit.to_dict(serializable=True)

    # Then
    assert circuit_dict["schema"] == SCHEMA_VERSION + "-circuit"
    assert circuit_dict["qubits"] == list(circuit.qubits)
    assert circuit_dict["symbolic_params"] == [
        str(param) for param in circuit.symbolic_params
    ]
    assert isinstance(circuit_dict["gates"], list)
    for gate_dict, gate in zip(circuit_dict["gates"], circuit.gates):
        assert gate_dict == gate.to_dict(serializable=True)


#### save ####
@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(gates=[]),
        Circuit(gates=[XGateQubit0]),
        Circuit(gates=[XGateQubit1]),
        Circuit(gates=[XGateQubit0, XGateQubit1]),
        Circuit(
            gates=[
                HGateQubit0,
                CNOTGateQubits01,
                ParameterizedRXGateQubit0,
                CNOTGateQubits01,
                HGateQubit0,
            ]
        ),
        Circuit(gates=[ParameterizedGateQubits02]),
        Circuit(
            gates=[
                ParameterizedRXGateQubit0,
                ParameterizedRYGateQubit0,
                ParameterizedRZGateQubit0,
                ParameterizedGateQubits02,
            ]
        ),
        Circuit(gates=[IGateQubit0 for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(1000)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(10000)]),
    ],
)
def test_circuit_is_successfully_saved_to_a_file(circuit):
    # When
    circuit.save("circuit.json")
    with open("circuit.json", "r") as f:
        saved_data = json.loads(f.read())

    # Then
    assert saved_data["schema"] == SCHEMA_VERSION + "-circuit"
    assert saved_data["qubits"] == list(circuit.qubits)
    assert saved_data["gates"] == [
        gate.to_dict(serializable=True) for gate in circuit.gates
    ]
    assert saved_data["symbolic_params"] == [
        str(param) for param in circuit.symbolic_params
    ]

    os.remove("circuit.json")


#### load ####
@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(gates=[]),
        Circuit(gates=[XGateQubit0]),
        Circuit(gates=[XGateQubit1]),
        Circuit(gates=[XGateQubit0, XGateQubit1]),
        Circuit(
            gates=[
                HGateQubit0,
                CNOTGateQubits01,
                ParameterizedRXGateQubit0,
                CNOTGateQubits01,
                HGateQubit0,
            ]
        ),
        Circuit(gates=[ParameterizedGateQubits02]),
        Circuit(
            gates=[
                ParameterizedRXGateQubit0,
                ParameterizedRYGateQubit0,
                ParameterizedRZGateQubit0,
                ParameterizedGateQubits02,
            ]
        ),
        Circuit(gates=[IGateQubit0 for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(1000)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(10000)]),
    ],
)
def test_circuit_is_successfully_loaded_from_a_file(circuit):
    # Given
    circuit.save("circuit.json")

    # When
    new_circuit = Circuit.load("circuit.json")

    # Then
    assert circuit == new_circuit

    os.remove("circuit.json")


@pytest.mark.parametrize(
    "circuit",
    [
        Circuit(gates=[]),
        Circuit(gates=[XGateQubit0]),
        Circuit(gates=[XGateQubit1]),
        Circuit(gates=[XGateQubit0, XGateQubit1]),
        Circuit(
            gates=[
                HGateQubit0,
                CNOTGateQubits01,
                ParameterizedRXGateQubit0,
                CNOTGateQubits01,
                HGateQubit0,
            ]
        ),
        Circuit(gates=[ParameterizedGateQubits02]),
        Circuit(
            gates=[
                ParameterizedRXGateQubit0,
                ParameterizedRYGateQubit0,
                ParameterizedRZGateQubit0,
                ParameterizedGateQubits02,
            ]
        ),
        Circuit(gates=[IGateQubit0 for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(100)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(1000)]),
        Circuit(gates=[random.choice(RandomGateList) for _ in range(10000)]),
    ],
)
def test_circuit_is_successfully_loaded_from_a_dict(circuit):
    for serializable in [True, False]:
        # Given
        circuit_dict = circuit.to_dict()

        # When
        new_circuit = Circuit.load(circuit_dict)

        # Then
        assert circuit == new_circuit
