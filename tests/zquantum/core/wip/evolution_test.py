import cirq
import numpy as np
import pytest
import sympy
from pyquil.paulis import PauliSum, PauliTerm

from zquantum.core.utils import compare_unitary
from zquantum.core.wip import circuits
from zquantum.core.wip.evolution import (
    generate_circuit_sequence,
    time_evolution,
    time_evolution_derivatives,
    time_evolution_for_term,
)

PAULI_STRING_TO_CIRQ_GATE = {"XX": cirq.XX, "YY": cirq.YY, "ZZ": cirq.ZZ}


def _cirq_exponentiate_term(term, qubits, time, order):
    base_exponent = 2 * time / order / np.pi
    base_gate = PAULI_STRING_TO_CIRQ_GATE[term.pauli_string()](*qubits)
    return base_gate ** (term.coefficient * base_exponent)


def _cirq_exponentiate_hamiltonian(hamiltonian, qubits, time, order):
    return cirq.Circuit(
        [
            _cirq_exponentiate_term(term, qubits, time, order)
            for term in hamiltonian.terms
        ] * order
    )


@pytest.mark.parametrize(
    "term, time, expected_unitary",
    [
        (PauliTerm("X", 0) * PauliTerm("X", 1), np.pi, -np.eye(4)),
        (
            PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
            np.pi,
            np.diag([1j, -1j, -1j, 1j])[::-1]
        ),
        (
            PauliTerm("Z", 0) * PauliTerm("Z", 1),
            np.pi,
            -np.eye(4)
        ),
        (
            PauliTerm("I", 0) * PauliTerm("I", 1),
            np.pi,
            -np.eye(2)
        )
    ]
)
class TestTimeEvolutionOfTerm:

    def test_evolving_pauli_term_with_numerical_time_gives_correct_unitary(
        self, term, time, expected_unitary
    ):
        actual_unitary = time_evolution_for_term(term, time).to_unitary()
        np.testing.assert_array_almost_equal(actual_unitary, expected_unitary)

    def test_evolving_pauli_term_with_symbolic_time_gives_correct_unitary(
        self, term, time, expected_unitary
    ):
        time_symbol = sympy.Symbol("t")
        symbol_map = {time_symbol: time}
        evolution_circuit = time_evolution_for_term(term, time_symbol)

        actual_unitary = evolution_circuit.bind(symbol_map).to_unitary()
        np.testing.assert_array_almost_equal(actual_unitary, expected_unitary)


class TestTimeEvolutionOfPauliSum:

    @pytest.fixture()
    def hamiltonian(self):
        return PauliSum(
            [
                PauliTerm("X", 0) * PauliTerm("X", 1),
                PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
                PauliTerm("Z", 0, 0.3) * PauliTerm("Z", 1),
            ]
        )

    @pytest.mark.parametrize("time", [0.1, 0.4, 1.0])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_evolution_with_numerical_parameter_produces_correct_result(
        self, hamiltonian, time, order
    ):
        cirq_qubits = cirq.LineQubit(0), cirq.LineQubit(1)
        expected_cirq_circuit = _cirq_exponentiate_hamiltonian(
            hamiltonian, cirq_qubits, time, order
        )

        reference_unitary = cirq.unitary(expected_cirq_circuit)
        unitary = time_evolution(hamiltonian, time, trotter_order=order).to_unitary()

        assert compare_unitary(unitary, reference_unitary, tol=1e-10)

    @pytest.mark.parametrize("time_value", [0.1, 0.4, 1.0])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_time_evolution_with_symbolic_parameter_produces_correct_unitary(
        self, hamiltonian, time_value, order
    ):
        time_symbol = sympy.Symbol("t")
        symbols_map = {time_symbol: time_value}

        cirq_qubits = cirq.LineQubit(0), cirq.LineQubit(1)

        expected_cirq_circuit = _cirq_exponentiate_hamiltonian(
            hamiltonian, cirq_qubits, time_value, order
        )

        reference_unitary = cirq.unitary(expected_cirq_circuit)

        unitary = time_evolution(
            hamiltonian, time_symbol, trotter_order=order
        ).bind(symbols_map).to_unitary()

        assert compare_unitary(unitary, reference_unitary, tol=1e-10)


class TestGeneratingCircuitSequence:

    @pytest.mark.parametrize(
        "repeated_circuit, different_circuit, length, position, expected_result",
        [
            (
                circuits.Circuit([circuits.X(0), circuits.Y(1)]),
                circuits.Circuit([circuits.Z(1)]),
                5,
                1,
                circuits.Circuit([
                    *[circuits.X(0), circuits.Y(1)],
                    circuits.Z(1),
                    *([circuits.X(0), circuits.Y(1)] * 3)
                ])
            ),
            (
                circuits.Circuit([circuits.RX(0.5)(1)]),
                circuits.Circuit([circuits.CNOT(0, 2)]),
                3,
                0,
                circuits.Circuit(
                    [
                        circuits.CNOT(0, 2),
                        circuits.RX(0.5)(1),
                        circuits.RX(0.5)(1)
                    ]
                )
            ),
            (
                circuits.Circuit([circuits.RX(0.5)(1)]),
                circuits.Circuit([circuits.CNOT(0, 2)]),
                3,
                2,
                circuits.Circuit(
                    [
                        circuits.RX(0.5)(1),
                        circuits.RX(0.5)(1),
                        circuits.CNOT(0, 2),
                    ]
                )
            )
        ]
    )
    def test_generate_circuit_sequence(
        self, repeated_circuit, different_circuit, position, length, expected_result
    ):
        actual_result = generate_circuit_sequence(
            repeated_circuit,
            different_circuit,
            length,
            position
        )

        assert actual_result == expected_result


class TestTimeEvolutionDerivatives:

    @pytest.mark.parametrize("time", [0.4, sympy.Symbol("t")])
    def test_time_evolution_derivatives_gives_correct_number_of_derivatives_and_factors(
        self, time
    ):
        hamiltonian = PauliSum(
            [
                PauliTerm("X", 0) * PauliTerm("X", 1),
                PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
                PauliTerm("Z", 0, 0.3) * PauliTerm("Z", 1),
            ]
        )

        order = 3
        reference_factors_1 = [1.0 / order, 0.5 / order, 0.3 / order] * 3
        reference_factors_2 = [-1.0 * x for x in reference_factors_1]

        derivatives, factors = time_evolution_derivatives(
            hamiltonian, time, trotter_order=order
        )

        assert len(derivatives) == order * 2 * len(hamiltonian.terms)
        assert len(factors) == order * 2 * len(hamiltonian.terms)
        assert factors[0:18:2] == reference_factors_1
        assert factors[1:18:2] == reference_factors_2
