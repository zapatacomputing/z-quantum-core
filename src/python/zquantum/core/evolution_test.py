import unittest
import cirq
from math import pi

from .evolution import (
    time_evolution,
    time_evolution_derivatives,
    generate_circuit_sequence,
)
from .utils import compare_unitary
from .testing import create_random_circuit
from pyquil.paulis import PauliSum, PauliTerm


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
        u = circuit._unitary_()

        # When
        unitary_evolution = time_evolution(hamiltonian, time, trotter_order=order)
        u1 = unitary_evolution.to_unitary()

        # Then
        self.assertEqual(compare_unitary(u1, u, tol=1e-10), True)

    def test_time_evolution_derivatives(self):
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
