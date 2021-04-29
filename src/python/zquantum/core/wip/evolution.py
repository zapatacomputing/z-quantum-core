from functools import reduce
import operator
import numpy as np
import sympy
import pyquil
from typing import Union, Sequence, Tuple, List
from zquantum.core.wip import circuits


def time_evolution(
    hamiltonian: pyquil.paulis.PauliSum,
    time: Union[float, sympy.Expr],
    method: str = "Trotter",
    trotter_order: int = 1,
) -> circuits.Circuit:

    """Generates circuit for performing time evolution under a Hamiltonian H.
    The default setting is first-order Trotterization. The goal is to approximate
    the operation exp(-iHt).

    Args:
        hamiltonian: The Hamiltonian to be evolved under.
        time: Time duration of the evolution.
        method: Time evolution method. Currently the only option is 'Trotter'.
        trotter_order: order of Trotter evolution

    Returns:
        A Circuit (core.circuit) object representing the time evolution.
    """

    if method != "Trotter":
        raise ValueError(f"Currently the method {method} is not supported.")

    return reduce(
        operator.add,
        (
            time_evolution_for_term(term, time / trotter_order)
            for _index_order in range(trotter_order)
            for term in hamiltonian.terms
        )
    )


def _evolve_gate_operation(operation: circuits.GateOperation, time):
    """Evolve gate operation.

    This is meant to be used with outputs from pyuil.paulis.exponentiate.
    The resulting operation has angle multiplied by time if it is RZ or PHASE,
    other gates are left unchanged.
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


def time_evolution_for_term(
    term: pyquil.paulis.PauliTerm, time: Union[float, sympy.Expr]
) -> circuits.Circuit:
    """Evolves a Pauli term for a given time and returns a circuit representing it.
    Args:
        term: Pauli term to be evolved
        time: time of evolution
    Returns:
        Circuit: Circuit representing evolved pyquil term.
    """
    if isinstance(time, sympy.Expr):
        circuit = circuits.import_from_pyquil(pyquil.paulis.exponentiate(term))

        new_circuit = circuits.Circuit([
            _evolve_gate_operation(operation, time)
            for operation in circuit.operations
        ])
    else:
        exponent = term * time
        new_circuit = circuits.import_from_pyquil(pyquil.paulis.exponentiate(exponent))

    return new_circuit


def time_evolution_derivatives(
    hamiltonian: pyquil.paulis.PauliSum,
    time: float,
    method: str = "Trotter",
    trotter_order: int = 1,
) -> Tuple[List[circuits.Circuit], List[float]]:
    if method != "Trotter":
        raise ValueError(f"The method {method} is currently not supported.")

    single_trotter_derivatives = []
    factors = [1.0, -1.0]
    output_factors = []

    for i, term_1 in enumerate(hamiltonian.terms):
        for factor in factors:
            output = circuits.Circuit()

            try:
                r = complex(term_1.coefficient).real / trotter_order
            except TypeError:
                raise ValueError(
                    "Term coefficients need to be numerical. "
                    f"Offending term: {term_1}")
            output_factors.append(r * factor)
            shift = factor * (np.pi / (4.0 * r))

            for j, term_2 in enumerate(hamiltonian.terms):
                output += time_evolution_for_term(
                    term_2,
                    (time + shift) / trotter_order if i == j else time / trotter_order
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
                    generate_circuit_sequence(repeated_circuit, different_circuit, trotter_order, position)
                )
                final_factors.append(factor)
        return output_circuits, final_factors
    else:
        return single_trotter_derivatives, output_factors


def generate_circuit_sequence(repeated_circuit, different_circuit, length, position):
    return circuits.Circuit([
        *[*repeated_circuit.operations] * position,
        *different_circuit.operations,
        *[*repeated_circuit.operations] * (length - position - 1)
        ]
    )
