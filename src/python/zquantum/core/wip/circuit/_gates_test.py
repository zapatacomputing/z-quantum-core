"""Test cases for _gates module."""
from unittest.mock import Mock

import pytest
import sympy

from . import _builtin_gates as bg
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

    def test_daggers_matrix_is_adjoint_of_original_gates_matrix(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 2), 1)
        assert gate.dagger.matrix == gate.matrix.adjoint()

    def test_dagger_has_the_same_params_and_num_qubits_as_wrapped_gate(self):
        gate = MatrixFactoryGate("U", example_two_qubit_matrix_factory, (0.5, 0.1, sympy.Symbol("a")), 2)
        assert gate.dagger.num_qubits == gate.num_qubits
        assert gate.dagger.params == gate.params

    def test_dagger_is_named_dagger(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (0.5, 2.5), 1)
        assert gate.dagger.name == "dagger"

    def test_dagger_of_hermitian_gate_is_the_same_gate(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 0), 1, is_hermitian=True)
        assert gate.dagger is gate

    def test_binding_gates_in_dagger_is_propagated_to_wrapped_gate(self):
        theta = sympy.Symbol("theta")
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (theta, 0), 1)

        assert gate.dagger.bind({theta: 0.5}) == gate.bind({theta: 0.5}).dagger

    def test_dagger_of_dagger_is_the_same_as_original_gate(self):
        gate = MatrixFactoryGate("V", example_one_qubit_matrix_factory, (1, 0), 1)
        assert gate.dagger.dagger is gate


@pytest.mark.parametrize(
    "gate",
    [
        bg.X,
        bg.Y,
        bg.Z,
        bg.T,
        bg.H,
        bg.I,
        bg.RX(sympy.Symbol("theta")),
        bg.RY(0.5),
        bg.RZ(0),
        bg.PHASE(sympy.pi / 5),
        bg.CZ,
        bg.CNOT,
        bg.SWAP,
        bg.ISWAP,
        bg.XX(sympy.cos(sympy.Symbol("phi"))),
        bg.YY(sympy.pi),
        bg.ZZ(sympy.Symbol("x") + sympy.Symbol("y")),
        bg.CPHASE(1.5)
    ]
)
class TestControlledGate:

    def test_is_named_control(self, gate):
        assert gate.controlled(2).name == "control"

    def test_has_number_of_qubits_equal_to_wrapped_gates_num_qubits_plus_num_controlled_qubits(self, gate):
        assert gate.controlled(3).num_qubits == gate.num_qubits + 3

    def test_has_matrix_with_ones_on_the_diagonal_and_wrapped_gates_matrix_as_bottom_left_block(self, gate):
        controlled_gate = gate.controlled(2)
        n = gate.matrix.shape[0]
        assert gate.matrix.shape[1] == n
        assert controlled_gate.matrix[0:-n, 0:-n] == sympy.eye(2 ** controlled_gate.num_qubits - n)
        assert controlled_gate.matrix[-n:, -n:] == gate.matrix

    def test_controlled_of_controlled_gate_is_controlled_gate_with_summed_number_of_control_qubits(self, gate):
        controlled_gate = gate.controlled(2)
        double_controlled_gate = controlled_gate.controlled(3)

        assert double_controlled_gate.wrapped_gate == gate
        assert double_controlled_gate.num_qubits == gate.num_qubits + 2 + 3
        assert double_controlled_gate.num_control_qubits == 2 + 3
        assert double_controlled_gate.matrix.shape == 2 * (2 ** (gate.num_qubits + 2 + 3),)

    def test_has_the_same_parameters_as_wrapped_gate(self, gate):
        controlled_gate = gate.controlled(4)

        assert controlled_gate.params == gate.params

    def test_dagger_of_controlled_gate_is_controlled_gate_wrapping_dagger(self, gate):
        controlled_gate = gate.controlled(4)

        assert controlled_gate.dagger == gate.dagger.controlled(4)

    def test_binding_parameters_in_control_gate_is_propagated_to_wrapped_gate(self, gate):
        controlled_gate = gate.controlled(2)
        symbols_map = {sympy.Symbol("theta"): 0.5, sympy.Symbol("x"): 3}
        assert controlled_gate.bind(symbols_map) == gate.bind(symbols_map).controlled(2)
