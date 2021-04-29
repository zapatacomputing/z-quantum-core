"""Functions for constructing circuits simulating evolution under given Hamiltonian."""
import operator
from functools import reduce, singledispatch
from itertools import chain
from typing import Union, Tuple, List

import numpy as np
import pyquil.paulis
import sympy
from openfermion import QubitOperator
from .circuit import Circuit, Gate, Qubit


def time_evolution(
    hamiltonian: Union[pyquil.paulis.PauliSum, QubitOperator],
    time: Union[float, sympy.Expr],
    method: str = "Trotter",
    trotter_order: int = 1,
) -> Circuit:
    """Create a circuit simulating evolution under given Hamiltonian.

    Args:
        hamiltonian: The Hamiltonian to be evolved under.
        time: Time duration of the evolution.
        method: Time evolution method. Currently the only option is 'Trotter'.
        trotter_order: order of Trotter evolution (1 by default).

    Returns:
        Circuit approximating evolution under `hamiltonian`.
        Circuit's unitary i approximately equal to exp(-i * time * hamiltonian).
    """
    if method != "Trotter":
        raise ValueError(f"Currently the method {method} is not supported.")
    if isinstance(hamiltonian, QubitOperator):
        terms = hamiltonian.get_operators()
    elif isinstance(hamiltonian, pyquil.paulis.PauliSum):
        terms = hamiltonian.terms

    return reduce(
        operator.add,
        (
            time_evolution_for_term(term, time / trotter_order)
            for _index_order in range(trotter_order)
            for term in terms
        ),
    )


@singledispatch
def time_evolution_for_term(term, time: Union[float, sympy.Expr]):
    raise NotImplementedError


@time_evolution_for_term.register
def time_evolution_for_term_pyquil(
    term: pyquil.paulis.PauliTerm, time: Union[float, sympy.Expr]
) -> Circuit:
    """Evolves a Pauli term for a given time and returns a circuit representing it.
    Args:
        term: Pauli term to be evolved
        time: time of evolution
    Returns:
        Circuit: Circuit representing evolved pyquil term.
    """
    if isinstance(time, sympy.Expr):
        circuit = Circuit(pyquil.paulis.exponentiate(term))
        for gate in circuit.gates:
            if len(gate.params) == 0:
                pass
            elif len(gate.params) > 1:
                raise (
                    NotImplementedError(
                        "Time evolution of multi-parametered gates with symbolic "
                        "parameters is not supported."
                    )
                )
            elif gate.name == "Rz" or gate.name == "PHASE":
                # We only want to modify the parameter of Rz gate or PHASE gate.
                gate.params[0] = gate.params[0] * time
    else:
        exponent = term * time
        assert isinstance(exponent, pyquil.paulis.PauliTerm)
        circuit = Circuit(pyquil.paulis.exponentiate(exponent))
    return circuit


@time_evolution_for_term.register
def time_evolution_for_term_qubit_operator(
    term: QubitOperator, time: Union[float, sympy.Expr]
) -> Circuit:
    """Evolves a Pauli term for a given time and returns a circuit representing it.
    Args:
        term: Pauli term to be evolved
        time: time of evolution
    Returns:
        Circuit: Circuit representing evolved pyquil term.
    """

    if len(term.terms) != 1:
        raise ValueError("This function works only on a single term.")
    term_components = list(term.terms.keys())[0]
    base_changes = []
    base_reversals = []
    cnot_gates = []
    central_gate = None
    term_types = [component[1] for component in term_components]
    qubit_indices = [component[0] for component in term_components]
    coefficient = list(term.terms.values())[0]

    for i, (term_type, qubit_id) in enumerate(zip(term_types, qubit_indices)):
        # TODO: comments
        if term_type == "X":
            base_changes.append(Gate("H", qubits=[Qubit(qubit_id)]))
            base_reversals.append(Gate("H", qubits=[Qubit(qubit_id)]))
        elif term_type == "Y":
            base_changes.append(
                Gate("Rx", qubits=[Qubit(qubit_id)], params=[np.pi / 2])
            )
            base_reversals.append(
                Gate("Rx", qubits=[Qubit(qubit_id)], params=[-np.pi / 2])
            )
        if i == len(term_components) - 1:
            central_gate = Gate(
                "Rz", qubits=[Qubit(qubit_id)], params=[2 * time * coefficient]
            )
        else:
            cnot_gates.append(
                Gate("CNOT", qubits=[Qubit(qubit_id), Qubit(qubit_indices[i + 1])])
            )

    circuit = Circuit()
    for gate in base_changes:
        circuit.gates.append(gate)

    for gate in cnot_gates:
        circuit.gates.append(gate)

    circuit.gates.append(central_gate)

    for gate in cnot_gates[::-1]:
        circuit.gates.append(gate)

    for gate in base_reversals:
        circuit.gates.append(gate)

    return circuit


def time_evolution_derivatives(
    hamiltonian: Union[pyquil.paulis.PauliSum, QubitOperator],
    time: float,
    method: str = "Trotter",
    trotter_order: int = 1,
) -> Tuple[List[Circuit], List[float]]:
    """Generates derivative circuits for the time evolution operator defined in
    function time_evolution

    Args:
        hamiltonian: The Hamiltonian to be evolved under. It should contain numeric
            coefficients, symbolic expressions aren't supported.
        time: time duration of the evolution.
        method: time evolution method. Currently the only option is 'Trotter'.
        trotter_order: order of Trotter evolution

    Returns:
        A Circuit simulating time evolution.
    """
    if method != "Trotter":
        raise ValueError(f"The method {method} is currently not supported.")

    single_trotter_derivatives = []
    factors = [1.0, -1.0]
    output_factors = []
    if isinstance(hamiltonian, QubitOperator):
        terms = hamiltonian.get_operators()
    elif isinstance(hamiltonian, pyquil.paulis.PauliSum):
        terms = hamiltonian.terms

    for i, term_1 in enumerate(terms):
        for factor in factors:
            output = Circuit()

            try:
                if isinstance(term_1, QubitOperator):
                    r = list(term_1.terms.values())[0]
                else:
                    r = complex(term_1.coefficient).real / trotter_order
            except TypeError:
                raise ValueError(
                    "Term coefficients need to be numerical. "
                    f"Offending term: {term_1}"
                )
            output_factors.append(r * factor)
            shift = factor * (np.pi / (4.0 * r))

            for j, term_2 in enumerate(terms):
                output += time_evolution_for_term(
                    term_2,
                    (time + shift) / trotter_order if i == j else time / trotter_order,
                )

            single_trotter_derivatives.append(output)

    if trotter_order > 1:
        output_circuits = []
        final_factors = []

        repeated_circuit = time_evolution(
            hamiltonian, time, method="Trotter", trotter_order=1
        )

        for position in range(trotter_order):
            for factor, different_circuit in zip(
                output_factors, single_trotter_derivatives
            ):
                output_circuits.append(
                    _generate_circuit_sequence(
                        repeated_circuit, different_circuit, trotter_order, position
                    )
                )
                final_factors.append(factor)
        return output_circuits, final_factors
    else:
        return single_trotter_derivatives, output_factors


def _generate_circuit_sequence(
    repeated_circuit: Circuit, different_circuit: Circuit, length: int, position: int
) -> Circuit:
    """
    Auxiliary function to generate a sequence of the "repeated_circuit",
    "length" times, where at position "position" we have "different_circuit"
    instead.
    Args:
        repeated_circuit (core.circuit.Circuit)
        different_circuit (core.circuit.Circuit)
        length (int)
        position (int)
    Returns:
        circuit_sequence (core.circuit.Circuit))
    """
    if position >= length:
        raise ValueError("The position must be less than the total length")

    circuit_sequence = Circuit()
    for index in range(length):
        if index == position:
            circuit_sequence += different_circuit
        else:
            circuit_sequence += repeated_circuit
    return circuit_sequence
