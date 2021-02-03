import unittest
import cirq
from math import pi
import numpy as np

from .evolution import (
    time_evolution,
    time_evolution_derivatives,
    generate_circuit_sequence,
    time_evolution_for_term,
)
from .utils import compare_unitary
from .testing import create_random_circuit
from pyquil.paulis import PauliSum, PauliTerm
import sympy


class TestTimeEvolution(unittest.TestCase):
    def test_time_evolution(self):
        # Given
        hamiltonian = PauliSum(
            [
                PauliTerm("X", 0) * PauliTerm("X", 1),
                PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
                PauliTerm("Z", 0, 0.3) * PauliTerm("Z", 1),
            ]
        )
        time = 0.4
        order = 2

        circuit = cirq.Circuit()
        q1 = cirq.LineQubit(0)
        q2 = cirq.LineQubit(1)
        for _ in range(0, order):
            circuit.append(
                cirq.XX(q1, q2)
                ** (hamiltonian.terms[0].coefficient * 2 * time / order / pi)
            )
            circuit.append(
                cirq.YY(q1, q2)
                ** (hamiltonian.terms[1].coefficient * 2 * time / order / pi)
            )
            circuit.append(
                cirq.ZZ(q1, q2)
                ** (hamiltonian.terms[2].coefficient * 2 * time / order / pi)
            )
        target_unitary = circuit._unitary_()

        # When
        unitary_evolution = time_evolution(hamiltonian, time, trotter_order=order)
        final_unitary = unitary_evolution.to_unitary()

        # Then
        self.assertEqual(
            compare_unitary(final_unitary, target_unitary, tol=1e-10), True
        )

    def test_time_evolution_with_symbolic_parameter(self):
        # Given
        hamiltonian = PauliSum(
            [
                PauliTerm("X", 0) * PauliTerm("X", 1),
                PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
                PauliTerm("Z", 0, 0.3) * PauliTerm("Z", 1),
            ]
        )
        time_symbol = sympy.Symbol("t")
        time_value = 0.4
        symbols_map = [(time_symbol, time_value)]
        order = 2

        circuit = cirq.Circuit()
        q1 = cirq.LineQubit(0)
        q2 = cirq.LineQubit(1)
        for _ in range(0, order):
            circuit.append(
                cirq.XX(q1, q2)
                ** (hamiltonian.terms[0].coefficient * 2 * time_value / order / pi)
            )
            circuit.append(
                cirq.YY(q1, q2)
                ** (hamiltonian.terms[1].coefficient * 2 * time_value / order / pi)
            )
            circuit.append(
                cirq.ZZ(q1, q2)
                ** (hamiltonian.terms[2].coefficient * 2 * time_value / order / pi)
            )
        target_unitary = circuit._unitary_()

        # When
        unitary_evolution_symbolic = time_evolution(
            hamiltonian, time_symbol, trotter_order=order
        )
        unitary_evolution = unitary_evolution_symbolic.evaluate(symbols_map)
        final_unitary = unitary_evolution.to_unitary()
        # Then
        self.assertEqual(
            compare_unitary(final_unitary, target_unitary, tol=1e-10), True
        )

    def test_time_evolution_derivatives(self):
        # Given
        hamiltonian = PauliSum(
            [
                PauliTerm("X", 0) * PauliTerm("X", 1),
                PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
                PauliTerm("Z", 0, 0.3) * PauliTerm("Z", 1),
            ]
        )
        time_symbol = sympy.Symbol("t")
        time_value = 0.4
        symbols_map = [(time_symbol, time_value)]

        order = 3
        reference_factors_1 = [1.0 / order, 0.5 / order, 0.3 / order] * 3
        reference_factors_2 = [-1.0 * x for x in reference_factors_1]

        # When
        derivatives, factors = time_evolution_derivatives(
            hamiltonian, time_symbol, trotter_order=order
        )

        # Then
        self.assertEqual(len(derivatives), order * 2 * len(hamiltonian.terms))
        self.assertEqual(len(factors), order * 2 * len(hamiltonian.terms))
        self.assertListEqual(reference_factors_1, factors[0:18:2])
        self.assertListEqual(reference_factors_2, factors[1:18:2])

    def test_time_evolution_derivatives_with_symbolic_parameter(self):
        # Given
        hamiltonian = PauliSum(
            [
                PauliTerm("X", 0) * PauliTerm("X", 1),
                PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
                PauliTerm("Z", 0, 0.3) * PauliTerm("Z", 1),
            ]
        )
        time = 0.4
        order = 3
        reference_factors_1 = [1.0 / order, 0.5 / order, 0.3 / order] * 3
        reference_factors_2 = [-1.0 * x for x in reference_factors_1]

        # When
        derivatives, factors = time_evolution_derivatives(
            hamiltonian, time, trotter_order=order
        )

        # Then
        self.assertEqual(len(derivatives), order * 2 * len(hamiltonian.terms))
        self.assertEqual(len(factors), order * 2 * len(hamiltonian.terms))
        self.assertListEqual(reference_factors_1, factors[0:18:2])
        self.assertListEqual(reference_factors_2, factors[1:18:2])

    def test_generate_circuit_sequence(self):
        # Given
        repeated_circuit_len = 3
        different_circuit_len = 5
        length = 3
        position_1 = 0
        position_2 = 1
        repeated_circuit = create_random_circuit(2, repeated_circuit_len)
        different_circuit = create_random_circuit(2, different_circuit_len)

        # When
        sequence_1 = generate_circuit_sequence(
            repeated_circuit, different_circuit, length, position_1
        )
        sequence_2 = generate_circuit_sequence(
            repeated_circuit, different_circuit, length, position_2
        )

        # Then
        self.assertEqual(
            len(sequence_1.gates),
            different_circuit_len + repeated_circuit_len * (length - 1),
        )
        different_circuit_start_1 = repeated_circuit_len * position_1
        different_circuit_start_2 = repeated_circuit_len * position_2
        self.assertListEqual(
            sequence_1.gates[
                different_circuit_start_1 : different_circuit_start_1
                + different_circuit_len
            ],
            different_circuit.gates,
        )
        self.assertListEqual(
            sequence_2.gates[
                different_circuit_start_2 : different_circuit_start_2
                + different_circuit_len
            ],
            different_circuit.gates,
        )

        # Given
        length = 3
        position = 10

        # When/Then
        with self.assertRaises(ValueError):
            sequence = generate_circuit_sequence(
                repeated_circuit, different_circuit, length, position
            )

    def test_time_evolution_for_term(self):
        # Given
        term_1 = PauliTerm("X", 0) * PauliTerm("X", 1)
        term_2 = PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1)
        term_3 = PauliTerm("Z", 0) * PauliTerm("Z", 1)
        term_4 = PauliTerm("I", 0) * PauliTerm("I", 1)
        time = pi

        target_unitary_1 = -np.eye(4)
        target_unitary_2 = np.zeros((4, 4), dtype=np.complex)
        target_unitary_2[0][3] = 1j
        target_unitary_2[1][2] = -1j
        target_unitary_2[2][1] = -1j
        target_unitary_2[3][0] = 1j
        target_unitary_3 = -np.eye(4)
        target_unitary_4 = -np.eye(2)

        # When
        unitary_1 = time_evolution_for_term(term_1, time).to_unitary()
        unitary_2 = time_evolution_for_term(term_2, time).to_unitary()
        unitary_3 = time_evolution_for_term(term_3, time).to_unitary()
        unitary_4 = time_evolution_for_term(term_4, time).to_unitary()

        # Then
        np.testing.assert_array_almost_equal(unitary_1, target_unitary_1)
        np.testing.assert_array_almost_equal(unitary_2, target_unitary_2)
        np.testing.assert_array_almost_equal(unitary_3, target_unitary_3)
        np.testing.assert_array_almost_equal(unitary_4, target_unitary_4)

    def test_time_evolution_for_term_with_symbolic_parameter(self):
        # Given
        term_1 = PauliTerm("X", 0) * PauliTerm("X", 1)
        term_2 = PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1)
        term_3 = PauliTerm("Z", 0) * PauliTerm("Z", 1)
        term_4 = PauliTerm("I", 0) * PauliTerm("I", 1)
        time = sympy.Symbol("t")
        time_value = pi
        symbols_map = [(time, time_value)]

        target_unitary_1 = -np.eye(4)
        target_unitary_2 = np.zeros((4, 4), dtype=np.complex)
        target_unitary_2[0][3] = 1j
        target_unitary_2[1][2] = -1j
        target_unitary_2[2][1] = -1j
        target_unitary_2[3][0] = 1j
        target_unitary_3 = -np.eye(4)
        target_unitary_4 = -np.eye(2)

        # When
        unitary_1 = (
            time_evolution_for_term(term_1, time).evaluate(symbols_map).to_unitary()
        )
        unitary_2 = (
            time_evolution_for_term(term_2, time).evaluate(symbols_map).to_unitary()
        )
        unitary_3 = (
            time_evolution_for_term(term_3, time).evaluate(symbols_map).to_unitary()
        )
        unitary_4 = (
            time_evolution_for_term(term_4, time).evaluate(symbols_map).to_unitary()
        )

        # Then
        np.testing.assert_array_almost_equal(unitary_1, target_unitary_1)
        np.testing.assert_array_almost_equal(unitary_2, target_unitary_2)
        np.testing.assert_array_almost_equal(unitary_3, target_unitary_3)
        np.testing.assert_array_almost_equal(unitary_4, target_unitary_4)
