"""Functions for constructing circuits simulating evolution under given Hamiltonian."""
import operator
import warnings
from functools import reduce, singledispatch
from itertools import chain
from typing import List, Tuple, Union

import numpy as np
import pyquil.paulis
import sympy
from openfermion import QubitOperator
from zquantum.core import circuits
from zquantum.core.circuits import CNOT, RX, RZ, H


def time_evolution(
    hamiltonian: Union[pyquil.paulis.PauliSum, QubitOperator],
    time: Union[float, sympy.Expr],
    method: str = "Trotter",
    trotter_order: int = 1,
) -> circuits.Circuit:
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
        terms = list(hamiltonian.get_operators())
    elif isinstance(hamiltonian, pyquil.paulis.PauliSum):
        warnings.warn(
            "PauliSum as an input to time_evolution will be depreciated, please change "
            "to QubitOperator instead.",
            DeprecationWarning,
        )
        terms = hamiltonian.terms

    return reduce(
        operator.add,
        (
            time_evolution_for_term(term, time / trotter_order)
            for _index_order in range(trotter_order)
            for term in terms
        ),
    )


def _adjust_gate_angle(operation: circuits.GateOperation, time):
    """Adjust angle in gate operation to account for evolution in time.

    Since this handles outputs from `pyquil.paulis.exponentiate`, the only
    time-dependent gates are RZ and PHASE (other rotations have fixed
    angles as they correspond to change of basis).

    Therefore, we multiply angles in RZ and PHASE by `time`, and leave other
    gates unchanged.
    """
    # The below should not happen, however, we leave it to reproduce logic from
    # the original code.
    if len(operation.params) > 1:
        raise ValueError(
            "Time evolution of multi-parameter gates with symbolic parameter is "
            "not supported."
        )
    if operation.gate.name in ("RZ", "PHASE"):
        evolved = operation.replace_params((operation.params[0] * time,))
    else:
        evolved = operation

    return evolved


@singledispatch
def time_evolution_for_term(term, time: Union[float, sympy.Expr]):
    raise NotImplementedError


@time_evolution_for_term.register
def _time_evolution_for_term_pyquil(
    term: pyquil.paulis.PauliTerm, time: Union[float, sympy.Expr]
) -> circuits.Circuit:
    """Construct a circuit simulating evolution under a given Pauli hamiltonian.

    Args:
        term: Pauli term describing evolution
        time: time of evolution
    Returns:
        Circuit simulating time evolution.
    """
    if isinstance(time, sympy.Expr):
        circuit = circuits.import_from_pyquil(pyquil.paulis.exponentiate(term))

        new_circuit = circuits.Circuit(
            [_adjust_gate_angle(operation, time) for operation in circuit.operations]
        )
    else:
        exponent = term * time
        new_circuit = circuits.import_from_pyquil(pyquil.paulis.exponentiate(exponent))

    return new_circuit


@time_evolution_for_term.register
def _time_evolution_for_term_qubit_operator(
    term: QubitOperator, time: Union[float, sympy.Expr]
) -> circuits.Circuit:
    """Evolves a Pauli term for a given time and returns a circuit representing it.
    Based on section 4 from https://arxiv.org/abs/1001.3855 .
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
        if term_type == "X":
            base_changes.append(H(qubit_id))
            base_reversals.append(H(qubit_id))
        elif term_type == "Y":
            base_changes.append(RX(np.pi / 2)(qubit_id))
            base_reversals.append(RX(-np.pi / 2)(qubit_id))
        if i == len(term_components) - 1:
            central_gate = RZ(2 * time * coefficient)(qubit_id)
        else:
            cnot_gates.append(CNOT(qubit_id, qubit_indices[i + 1]))

    circuit = circuits.Circuit()
    for gate in base_changes:
        circuit += gate

    for gate in cnot_gates:
        circuit += gate

    circuit += central_gate

    for gate in reversed(cnot_gates):
        circuit += gate

    for gate in base_reversals:
        circuit += gate

    return circuit


def time_evolution_derivatives(
    hamiltonian: pyquil.paulis.PauliSum,
    time: float,
    method: str = "Trotter",
    trotter_order: int = 1,
) -> Tuple[List[circuits.Circuit], List[float]]:
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
        terms = list(hamiltonian.get_operators())
    elif isinstance(hamiltonian, pyquil.paulis.PauliSum):
        warnings.warn(
            "PauliSum as an input to time_evolution_derivatives will be depreciated, "
            "please change to QubitOperator instead.",
            DeprecationWarning,
        )
        terms = hamiltonian.terms

    for i, term_1 in enumerate(terms):
        for factor in factors:
            output = circuits.Circuit()

            try:
                if isinstance(term_1, QubitOperator):
                    r = list(term_1.terms.values())[0] / trotter_order
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
    repeated_circuit: circuits.Circuit,
    different_circuit: circuits.Circuit,
    length: int,
    position: int,
):
    """Join multiple copies of circuit, replacing one copy with a different circuit.

    Args:
        repeated_circuit: circuit which copies should be concatenated
        different_circuit: circuit that will replace one copy of `repeated_circuit
        length: total number of circuits to join
        position: which copy of repeated_circuit should be replaced by
        `different_circuit`.
    Returns:
        Concatenation of circuits C_1, ..., C_length, where C_i = `repeated_circuit`
        if i != position and C_i = `different_circuit` if i == position.
    """
    if position >= length:
        raise ValueError(f"Position {position} should be < {length}")

    return circuits.Circuit(
        list(
            chain.from_iterable(
                [
                    (
                        repeated_circuit if i != position else different_circuit
                    ).operations
                    for i in range(length)
                ]
            )
        )
    )
