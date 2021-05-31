import cirq
import numpy as np
import pytest
import sympy
from openfermion import QubitOperator
from pyquil.paulis import PauliSum, PauliTerm
from zquantum.core import circuits
from zquantum.core.evolution import (
    _generate_circuit_sequence,
    time_evolution,
    time_evolution_derivatives,
    time_evolution_for_term,
)
from zquantum.core.utils import compare_unitary

PAULI_STRING_TO_CIRQ_GATE = {"XX": cirq.XX, "YY": cirq.YY, "ZZ": cirq.ZZ}
OPENFERMION_TERM_TO_CIRQ_GATE = {
    ((0, "X"), (1, "X")): cirq.XX,
    ((0, "Y"), (1, "Y")): cirq.YY,
    ((0, "Z"), (1, "Z")): cirq.ZZ,
}


def _cirq_exponentiate_pauli_term(term, qubits, time, trotter_order):
    base_exponent = 2 * time / trotter_order / np.pi
    base_gate = PAULI_STRING_TO_CIRQ_GATE[term.pauli_string()](*qubits)
    return base_gate ** (term.coefficient * base_exponent)


def _cirq_exponentiate_qubit_hamiltonian_term(term, qubits, time, trotter_order):
    base_exponent = 2 * time / trotter_order / np.pi
    base_gate = OPENFERMION_TERM_TO_CIRQ_GATE[list(term.terms.keys())[0]](*qubits)
    coefficient = list(term.terms.values())[0]
    return base_gate ** (coefficient * base_exponent)


def _cirq_exponentiate_hamiltonian(hamiltonian, qubits, time, trotter_order):
    if isinstance(hamiltonian, QubitOperator):
        return cirq.Circuit(
            [
                _cirq_exponentiate_qubit_hamiltonian_term(
                    term, qubits, time, trotter_order
                )
                for term in hamiltonian.get_operators()
            ]
            * trotter_order
        )
    elif isinstance(hamiltonian, PauliSum):
        return cirq.Circuit(
            [
                _cirq_exponentiate_pauli_term(term, qubits, time, trotter_order)
                for term in hamiltonian.terms
            ]
            * trotter_order
        )


@pytest.mark.parametrize(
    "term, time, expected_unitary",
    [
        (PauliTerm("X", 0) * PauliTerm("X", 1), np.pi, -np.eye(4)),
        (
            PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
            np.pi,
            np.diag([1j, -1j, -1j, 1j])[::-1],
        ),
        (PauliTerm("Z", 0) * PauliTerm("Z", 1), np.pi, -np.eye(4)),
        (PauliTerm("I", 0) * PauliTerm("I", 1), np.pi, -np.eye(2)),
        (QubitOperator("[X0 X1]"), np.pi, -np.eye(4)),
        (
            QubitOperator("0.5 [Y0 Y1]"),
            np.pi,
            np.diag([1j, -1j, -1j, 1j])[::-1],
        ),
        (QubitOperator("[Z0 Z1]"), np.pi, -np.eye(4)),
    ],
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
    @pytest.fixture(
        params=[
            PauliSum(
                [
                    PauliTerm("X", 0) * PauliTerm("X", 1),
                    PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
                    PauliTerm("Z", 0, 0.3) * PauliTerm("Z", 1),
                ]
            ),
            QubitOperator("[X0 X1] + 0.5[Y0 Y1] + 0.3[Z0 Z1]"),
        ]
    )
    def hamiltonian(self, request):
        return request.param

    @pytest.mark.parametrize("time", [0.1, 0.4, 1.0])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_evolution_with_numerical_time_produces_correct_result(
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
    def test_time_evolution_with_symbolic_time_produces_correct_unitary(
        self, hamiltonian, time_value, order
    ):
        time_symbol = sympy.Symbol("t")
        symbols_map = {time_symbol: time_value}

        cirq_qubits = cirq.LineQubit(0), cirq.LineQubit(1)

        expected_cirq_circuit = _cirq_exponentiate_hamiltonian(
            hamiltonian, cirq_qubits, time_value, order
        )

        reference_unitary = cirq.unitary(expected_cirq_circuit)

        unitary = (
            time_evolution(hamiltonian, time_symbol, trotter_order=order)
            .bind(symbols_map)
            .to_unitary()
        )

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
                circuits.Circuit(
                    [
                        *[circuits.X(0), circuits.Y(1)],
                        circuits.Z(1),
                        *([circuits.X(0), circuits.Y(1)] * 3),
                    ]
                ),
            ),
            (
                circuits.Circuit([circuits.RX(0.5)(1)]),
                circuits.Circuit([circuits.CNOT(0, 2)]),
                3,
                0,
                circuits.Circuit(
                    [circuits.CNOT(0, 2), circuits.RX(0.5)(1), circuits.RX(0.5)(1)]
                ),
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
                ),
            ),
        ],
    )
    def test_produces_correct_sequence(
        self, repeated_circuit, different_circuit, position, length, expected_result
    ):
        actual_result = _generate_circuit_sequence(
            repeated_circuit, different_circuit, length, position
        )

        assert actual_result == expected_result

    @pytest.mark.parametrize("length, invalid_position", [(5, 5), (4, 6)])
    def test_raises_error_if_position_is_larger_or_equal_to_length(
        self, length, invalid_position
    ):
        repeated_circuit = circuits.Circuit([circuits.X(0)])
        different_circuit = circuits.Circuit([circuits.Y(0)])

        with pytest.raises(ValueError):
            _generate_circuit_sequence(
                repeated_circuit, different_circuit, length, invalid_position
            )


class TestTimeEvolutionDerivatives:
    @pytest.fixture(
        params=[
            PauliSum(
                [
                    PauliTerm("X", 0) * PauliTerm("X", 1),
                    PauliTerm("Y", 0, 0.5) * PauliTerm("Y", 1),
                    PauliTerm("Z", 0, 0.3) * PauliTerm("Z", 1),
                ]
            ),
            QubitOperator("[X0 X1] + 0.5[Y0 Y1] + 0.3[Z0 Z1]"),
        ]
    )
    def hamiltonian(self, request):
        return request.param

    @pytest.mark.parametrize("time", [0.4, sympy.Symbol("t")])
    def test_gives_correct_number_of_derivatives_and_factors(self, time, hamiltonian):

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
