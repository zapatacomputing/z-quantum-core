import numpy as np
import pytest
import json
import os
import sympy
from ...utils import SCHEMA_VERSION
from ._gate import Gate, CustomGate


THETA = sympy.Symbol("theta")


@pytest.mark.parametrize(
    "first, second",
    [
        (0.25, 0.25),
        (THETA, THETA),
        (sympy.I, sympy.I),
        (sympy.cos(sympy.sin(THETA + 1)), sympy.cos(sympy.sin(THETA+1))),
        (2.0, 2),
        (sympy.Number(2), 2.0),
        (1j, sympy.I),
        (-3 + 0.5j, -3 + sympy.I / 2),
        ((THETA + 1) ** 2, THETA ** 2 + 2 * THETA + 1),
        (sympy.exp(1j * sympy.pi / 4), sympy.exp(1j * np.pi / 4)),
        (sympy.exp(sympy.I * sympy.pi / 4), sympy.exp(1j * np.pi / 4)),
        (sympy.exp(1j * sympy.pi / 4), sympy.exp(sympy.I * np.pi / 4)),
        (sympy.exp(1j * sympy.pi / 3), sympy.cos(sympy.pi / 3) + 1j * sympy.sin(sympy.pi / 3)),
        (sympy.exp(1j * sympy.pi / 5), np.cos(np.pi / 5) + 1j * np.sin(np.pi / 5))
    ]
)
def test_equal_or_close_elements_are_considered_to_be_equal_in_gate_matrix(first, second):
    assert Gate.are_elements_equal(first, second)
    assert Gate.are_elements_equal(second, first)


def test_different_symbolic_expressions_are_not_considered_to_be_equal_in_gate_matrix():
    assert not Gate.are_elements_equal(sympy.Symbol("theta"), sympy.Symbol("sigma"))


@pytest.mark.parametrize(
    "matrix",
    [
        sympy.Matrix([[0, 0], [0, 0], [0, 0]]),
        sympy.Matrix([[0], [0]]),
        sympy.Matrix([[0, 0, 0], [0, 0, 0]]),
        sympy.Matrix([[0]]),
        sympy.Matrix([[sympy.Symbol("theta")]]),
    ],
)
def test_creating_gate_with_rectangular_matrix_fails(matrix):
    """The Gate class should raise an assertion error if the matrix is not square"""
    qubits = (0,)
    with pytest.raises(ValueError):
        CustomGate(matrix, qubits)


@pytest.mark.parametrize(
    "matrix",
    [
        sympy.Matrix([[], []]),
        sympy.Matrix([[0], [0]]),
        sympy.Matrix([[0, 0, 0], [0, 0, 0]]),
        sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        sympy.Matrix([[0, 0, 0, 0], [0, 0, 0, 0]]),
        sympy.Matrix([[sympy.Symbol("theta")], [sympy.Symbol("theta")]]),
    ],
)
def test_creating_gate_with_invalid_matrix_size_for_one_qubit_fails(matrix):
    """The CustomGate class should raise an assertion error if the matrix is not 2x2."""
    qubits = (0,)
    with pytest.raises(ValueError):
        CustomGate(matrix, qubits)


@pytest.mark.parametrize(
    "matrix",
    [
        sympy.Matrix([[], []]),
        sympy.Matrix([[0], [0]]),
        sympy.Matrix([[0, 0], [0, 0]]),
        sympy.Matrix([[0, 0, 0], [0, 0, 0]]),
        sympy.Matrix([[sympy.Symbol("theta")], [sympy.Symbol("theta")]]),
    ],
)
def test_creating_gate_with_invalid_matrix_size_for_two_qubit_fails(matrix):
    qubits = (0, 1)
    with pytest.raises(ValueError):
        CustomGate(matrix, qubits)


@pytest.mark.parametrize(
    "qubits", [(0, 0), (0, 1, 1), (0, 1, 0), (0, 1, 2, 3, 4, 5, 6, 7, 3)]
)
def test_creating_gate_with_repeated_qubits_fails(qubits):
    matrix = sympy.Matrix(np.eye(2 ** len(qubits)))

    with pytest.raises(ValueError):
        CustomGate(matrix, qubits)


@pytest.mark.parametrize(
    "matrix",
    [
        sympy.Matrix(
            [
                [(1 / np.sqrt(2)) * complex(1, 0), (1 / np.sqrt(2)) * complex(1, 0)],
                [(1 / np.sqrt(2)) * complex(1, 0), (1 / np.sqrt(2)) * complex(-1, 0)],
            ]
        ),
        sympy.Matrix([[complex(0), complex(0 + 1j)], [complex(0 - 1j), complex(0)]]),
        sympy.Matrix(np.zeros(shape=(2, 2))),
    ],
)
def test_matrix_of_custom_gate_is_a_copy_of_matrix_passed_to_initializer(matrix):
    qubits = (0,)

    gate = CustomGate(matrix, qubits)

    assert gate.matrix.equals(matrix)


@pytest.mark.parametrize(
    "matrix,expected_symbolic_params",
    [
        (
            sympy.Matrix(
                [[sympy.Symbol("theta_0"), 0], [0, 3 * sympy.Symbol("theta_0")]]
            ),
            {sympy.Symbol("theta_0")},
        ),
        (
            sympy.Matrix(
                [[sympy.Symbol("theta_0"), 0], [0, 3 * sympy.Symbol("theta_1")]]
            ),
            {sympy.Symbol("theta_0"), sympy.Symbol("theta_1")},
        ),
    ],
)
def test_symbolic_params_set_of_a_gate_contains_all_symbols_from_provided_matrix(
    matrix, expected_symbolic_params
):
    qubits = (0,)
    gate = CustomGate(matrix, qubits)

    assert gate.symbolic_params == expected_symbolic_params


@pytest.mark.parametrize(
    "matrix, qubits",
    [
        (sympy.Matrix(np.eye(2)), (0,)),
        (sympy.Matrix(np.eye(2)), (1,)),
        (
            sympy.Matrix(
                [
                    [0, 0 + 1j],
                    [0 - 1j, 0],
                ]
            ),
            (0,),
        ),
        (
            sympy.Matrix(
                [
                    [0, 0 + 1j],
                    [0 - 1j, 0],
                ]
            ),
            (1,),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
            (0, 2),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, sympy.Symbol("theta_0"), 0],
                    [0, 1, 0, sympy.Symbol("theta_0")],
                    [sympy.Symbol("theta_0"), 0, 0, 1],
                    [0, sympy.Symbol("theta_0"), 1, 0],
                ]
            ),
            (0, 2),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, sympy.Symbol("theta_0") + 1 + 2j, 0],
                    [0, 1, 0, sympy.Symbol("theta_0") + 1 + 2j],
                    [sympy.Symbol("theta_0") + 1 + 2j, 0, 0, 1],
                    [0, sympy.Symbol("theta_0") + 1 + 2j, 1, 0],
                ]
            ),
            (0, 2),
        ),
    ],
)
def test_gates_created_from_the_same_matrix_and_with_the_same_qubits_are_equivalent(
    matrix, qubits
):
    assert CustomGate(matrix, qubits) == CustomGate(matrix, qubits)


@pytest.mark.parametrize(
    "matrix, qubits1, qubits2",
    [
        (sympy.Matrix(np.eye(2)), (0,), (1,)),
        (sympy.Matrix(np.eye(2)), (1,), (0,)),
        (
            sympy.Matrix(
                [
                    [0, 0 + 1j],
                    [0 - 1j, 0],
                ]
            ),
            (0,),
            (1,),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
            (1, 2),
            (0, 4),
        ),
    ],
)
def test_gates_created_with_the_same_matrix_but_acting_on_different_qubits_are_not_equivalent(
    matrix, qubits1, qubits2
):
    assert not CustomGate(matrix, qubits1) == CustomGate(matrix, qubits2)


@pytest.mark.parametrize(
    "matrix1, matrix2, qubits",
    [
        (
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ),
            (0, 2),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, sympy.Symbol("theta_0"), 0],
                    [0, 1, 0, sympy.Symbol("theta_0")],
                    [sympy.Symbol("theta_0"), 0, 0, 1],
                    [0, sympy.Symbol("theta_0"), 1, 0],
                ]
            ),
            sympy.Matrix(
                [
                    [1, 0, sympy.Symbol("theta_1"), 0],
                    [0, 1, 0, sympy.Symbol("theta_1")],
                    [sympy.Symbol("theta_1"), 0, 0, 1],
                    [0, sympy.Symbol("theta_1"), 1, 0],
                ]
            ),
            (1, 4),
        ),
    ],
)
def test_gates_created_with_different_matrices_and_acting_on_the_same_qubits_are_not_equivalent(
    matrix1, matrix2, qubits
):
    assert not CustomGate(matrix1, qubits) == CustomGate(matrix2, qubits)


@pytest.mark.parametrize(
    "matrix, qubits",
    [
        (sympy.Matrix(np.eye(2)), (0,)),
        (sympy.Matrix(np.eye(2)), (1,)),
        (
            sympy.Matrix(
                [
                    [0, 0 + 1j],
                    [0 - 1j, 0],
                ]
            ),
            (0,),
        ),
        (
            sympy.Matrix(
                [
                    [0, 0 + 1j],
                    [0 - 1j, 0],
                ]
            ),
            (1,),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
            (0, 2),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, sympy.Symbol("theta_0"), 0],
                    [0, 1, 0, sympy.Symbol("theta_0")],
                    [sympy.Symbol("theta_0"), 0, 0, 1],
                    [0, sympy.Symbol("theta_0"), 1, 0],
                ]
            ),
            (0, 2),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, sympy.Symbol("theta_0") + 1 + 2j, 0],
                    [
                        0,
                        1,
                        0,
                        sympy.Symbol("theta_0") + 1 + 2j,
                    ],
                    [
                        sympy.Symbol("theta_0") + 1 + 2j,
                        0,
                        0,
                        1,
                    ],
                    [
                        0,
                        sympy.Symbol("theta_0") + 1 + 2j,
                        1,
                        0,
                    ],
                ]
            ),
            (0, 2),
        ),
    ],
)
def test_representation_of_custom_gate_contains_class_path_matrix_and_qubits(
    matrix, qubits
):
    gate = CustomGate(matrix, qubits)

    representation = gate.__repr__()

    assert "zquantum.core.circuit.gate.CustomGate" in representation
    assert "matrix" in representation
    assert "qubits" in representation


@pytest.mark.parametrize(
    "matrix, qubits",
    [
        (sympy.Matrix(np.eye(2)), (0,)),
        (sympy.Matrix(np.eye(2)), (1,)),
        (
            sympy.Matrix(
                [
                    [0, 0 + 1j],
                    [0 - 1j, 0],
                ]
            ),
            (0,),
        ),
        (
            sympy.Matrix(
                [
                    [0, 0 + 1j],
                    [0 - 1j, 0],
                ]
            ),
            (1,),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
            (0, 2),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, sympy.Symbol("theta_0"), 0],
                    [0, 1, 0, sympy.Symbol("theta_0")],
                    [sympy.Symbol("theta_0"), 0, 0, 1],
                    [0, sympy.Symbol("theta_0"), 1, 0],
                ]
            ),
            (0, 2),
        ),
        (
            sympy.Matrix(
                [
                    [1, 0, sympy.Symbol("theta_0") + (1 + 2j), 0],
                    [0, 1, 0, sympy.Symbol("theta_0") + (1 + 2j)],
                    [sympy.Symbol("theta_0") + (1 + 2j), 0, 0, 1],
                    [0, sympy.Symbol("theta_0") + (1 + 2j), 1, 0],
                ]
            ),
            (0, 2),
        ),
        (
            sympy.Matrix(
                [
                    [
                        sympy.cos(sympy.Symbol("gamma") / 2),
                        -sympy.I * sympy.sin(sympy.Symbol("gamma") / 2),
                    ],
                    [
                        -sympy.I * sympy.sin(sympy.Symbol("gamma") / 2),
                        sympy.cos(sympy.Symbol("gamma") / 2),
                    ],
                ]
            ),
            (0,),
        ),
    ],
)
class TestGateSerialization:
    def test_dictionary_created_from_gate_contains_all_necessary_items(
        self, matrix, qubits
    ):
        gate = CustomGate(matrix, qubits)

        gate_dict = gate.to_dict(serializable=False)

        assert gate_dict["schema"] == SCHEMA_VERSION + "-gate"
        assert gate_dict["qubits"] == qubits
        assert gate_dict["matrix"] == matrix
        assert gate_dict["symbolic_params"] == gate.symbolic_params

    def test_gates_matrix_is_expanded_if_serializable_is_set_to_true(
        self, matrix, qubits
    ):
        gate = CustomGate(matrix, qubits)

        gate_dict = gate.to_dict(serializable=True)

        assert gate_dict["schema"] == SCHEMA_VERSION + "-gate"
        assert gate_dict["qubits"] == list(qubits)

        symbols = {symbol: sympy.Symbol(symbol) for symbol in gate_dict["symbolic_params"]}

        for row_index, row in enumerate(gate_dict["matrix"]):
            for col_index, element in enumerate(row["elements"]):
                assert sympy.sympify(element, locals=symbols) == matrix[row_index, col_index]
        assert gate_dict["symbolic_params"] == [
            str(param) for param in gate.symbolic_params
        ]

    def test_saving_gate_to_a_file_outputs_the_same_dictionary_as_to_dict_with_serializable_set_to_true(
        self, matrix, qubits
    ):
        gate = CustomGate(matrix, qubits)

        gate.save("gate.json")

        with open("gate.json", "r") as f:
            saved_data = json.load(f)

        assert saved_data == gate.to_dict(serializable=True)

        os.remove("gate.json")

    def test_loading_saved_gate_gives_the_same_gate(self, matrix, qubits):
        gate = CustomGate(matrix, qubits)

        gate.save("gate.json")
        new_gate = Gate.load("gate.json")

        assert gate == new_gate

        os.remove("gate.json")

    def test_loading_gate_from_dict_gives_the_same_gate(self, matrix, qubits):
        for serializable in [True, False]:
            gate = CustomGate(matrix, qubits)

            gate_dict = gate.to_dict(serializable=serializable)

            assert gate == Gate.load(gate_dict)


@pytest.mark.parametrize(
    "matrix, evaluated_matrix, symbols_map",
    [
        (
            sympy.Matrix(
                [
                    [sympy.Symbol("theta"), 0, 0, 0],
                    [0, sympy.Symbol("theta"), 0, 0],
                    [0, 0, 0, sympy.Symbol("theta")],
                    [0, 0, sympy.Symbol("theta"), 0],
                ]
            ),
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
            {"theta": 1},
        ),
        (
            sympy.Matrix(
                [
                    [sympy.Symbol("theta"), 0, 0, 0],
                    [0, sympy.Symbol("theta"), 0, 0],
                    [0, 0, 0, -1j * sympy.Symbol("gamma")],
                    [0, 0, -1j * sympy.Symbol("gamma"), 0],
                ]
            ),
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ]
            ),
            {"theta": 1, "gamma": 1j},
        ),
        (
            sympy.Matrix(
                [
                    [sympy.Symbol("theta"), 0, 0, 0],
                    [0, sympy.Symbol("theta"), 0, 0],
                    [0, 0, 0, -1j * sympy.Symbol("gamma")],
                    [0, 0, -1j * sympy.Symbol("gamma"), 0],
                ]
            ),
            sympy.Matrix(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, -1j * sympy.Symbol("gamma")],
                    [0, 0, -1j * sympy.Symbol("gamma"), 0],
                ]
            ),
            {"theta": 1},
        ),
    ],
)
def test_evaluating_gate_creates_gate_with_symbols_substituted_into_gates_matrix(
    matrix, evaluated_matrix, symbols_map
):
    qubits = (0, 2)
    gate = CustomGate(matrix, qubits)
    expected_new_gate = CustomGate(evaluated_matrix, qubits)

    new_gate = gate.evaluate(symbols_map)

    assert new_gate == expected_new_gate


def test_evaluating_gate_raises_warnings_when_extra_params_are_present_in_symbols_map():
    qubits = (0, 2)
    matrix = sympy.Matrix(
        [
            [sympy.Symbol("theta"), 0, 0, 0],
            [0, sympy.Symbol("theta"), 0, 0],
            [0, 0, 0, -1j * sympy.Symbol("gamma")],
            [0, 0, -1j * sympy.Symbol("gamma"), 0],
        ]
    )
    gate = CustomGate(matrix, qubits)
    symbols_map = {"theta": 1, "lambda": 0.2}

    with pytest.warns(Warning):
        gate.evaluate(symbols_map)
