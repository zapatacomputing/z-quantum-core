"""Test cases for _gates module."""
from unittest.mock import Mock

import pytest
import sympy

from ._gates import MatrixFactoryGate


def example_one_qubit_matrix_factory(a, b):
    return sympy.Matrix([[a, b], [b, a]])


def example_two_qubit_matrix_factory(a, b, c):
    return sympy.Matrix([
        [a, 0, 0, 0],
        [0, b, 0, 0],
        [0, 0, c, 0],
        [0, 0, 0, 1]
    ])


class TestMatrixFactoryGate:

    @pytest.mark.parametrize(
        "params, factory, num_qubits",
        [
            ((0.5, sympy.Symbol("theta")), example_one_qubit_matrix_factory, 1),
            (
                (sympy.Symbol("alpha"), sympy.Symbol("beta"), 1),
                example_two_qubit_matrix_factory,
                2
            )
        ]
    )
    def test_constructs_its_matrix_by_calling_factory_with_bound_parameter(
        self, params, factory, num_qubits
    ):
        wrapped_factory = Mock(wraps=factory)
        gate = MatrixFactoryGate("U", wrapped_factory, params, num_qubits)
        assert gate.matrix == factory(*params)
        wrapped_factory.assert_called_once_with(*params)

    def test_binding_parameters_creates_new_instance_with_substituted_free_params(self):
        gamma, theta, x, y = sympy.symbols("gamma, theta, x, y")
        params = (theta, x + y)
        gate = MatrixFactoryGate("U", example_one_qubit_matrix_factory, params, 1)

        new_gate = gate.bind({theta: 0.5, x: gamma, y: 3})

        assert new_gate.name == gate.name
        assert new_gate.matrix_factory == gate.matrix_factory
        assert new_gate.num_qubits == gate.num_qubits
        assert new_gate.params == (0.5, gamma + 3)

    def test_binding_parameters_with_symbol_outside_of_gates_free_symbols_set_does_not_raise_error(
        self
    ):
        gamma, theta = sympy.symbols("gamma, theta")
        params = (theta, 2 * theta)
        gate = MatrixFactoryGate("U", example_one_qubit_matrix_factory, params, 1)

        new_gate = gate.bind({gamma: 0.5, theta: 1})

        assert new_gate.params == (1, 2)

    def test_binding_parameters_does_not_change_parameters_without_free_symbols(
        self
    ):
        theta = sympy.Symbol("theta")
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 2), 1)

        new_gate = gate.bind({theta: 5.0})

        assert new_gate.params == (1, 2)
