from functools import singledispatch
from typing import List, Tuple, Union

import numpy as np
from openfermion import QubitOperator
import pyquil
from pyquil.quilatom import Qubit
import sympy
from pyquil.paulis import exponentiate as pyquil_exponentiate
from typing_extensions import final

from .circuit import Circuit, Gate, Qubit
from .wip.circuits import Circuit as NewCircuit
from .wip.circuits import H, RX, RZ, CNOT


def time_evolution(
    hamiltonian: Union[pyquil.paulis.PauliSum, QubitOperator],
    time: Union[float, sympy.Expr],
    method: str = "Trotter",
    trotter_order: int = 1,
) -> Circuit:

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

    if method == "Trotter":
        output = Circuit()
        for index_order in range(0, trotter_order):  # iterate over Trotter orders
            if isinstance(hamiltonian, QubitOperator):
                terms = hamiltonian.get_operators()
            elif isinstance(hamiltonian, pyquil.paulis.PauliSum):
                terms = hamiltonian.terms

            for term in terms:
                output += time_evolution_for_term(term, time / trotter_order)

    else:
        raise ValueError("Currently the method {} is not supported".format(method))

    return output


def time_evolution_derivatives(
    hamiltonian: pyquil.paulis.PauliSum,
    time: float,
    method: str = "Trotter",
    trotter_order: int = 1,
) -> Tuple[List[Circuit], List[float]]:

    """Generates derivative circuits for the time evolution operator defined in
    function time_evolution

    Args:
        hamiltonian: The Hamiltonian to be evolved under. It should contain numeric
            coefficients, symbolic expressions aren't supported.
        time: Time duration of the evolution.
        method: Time evolution method. Currently the only option is 'Trotter'.
        trotter_order: order of Trotter evolution

    Returns:
        A Circuit (core.circuit) object representing the time evolution.
    """
    if method == "Trotter":

        # derivative for a single Trotter step
        single_trotter_derivatives = []
        factors = [1.0, -1.0]
        output_factors = []

        for index_term1 in range(0, len(hamiltonian.terms)):

            for factor in factors:

                output = Circuit()
                # r is the eigenvalue of the generator of the gate. The value is
                # modified to take into account the coefficient and trotter step in
                # front of the Pauli term.
                coefficient = hamiltonian[index_term1].coefficient
                if isinstance(coefficient, complex):
                    real_coefficient = coefficient.real
                elif isinstance(coefficient, (int, float)):
                    real_coefficient = float(coefficient)
                else:
                    raise ValueError(
                        "Evolution only works with numeric coefficients. "
                        f"{coefficient} ({type(coefficient)}) is unsupported"
                    )
                r = real_coefficient / trotter_order
                output_factors.append(factor * r)
                shift = factor * (np.pi / (4.0 * r))

                for index_term2 in range(0, len(hamiltonian.terms)):
                    if index_term1 == index_term2:
                        expitH_circuit = time_evolution_for_term(
                            hamiltonian[index_term2], ((time + shift) / trotter_order)
                        )
                        output += expitH_circuit
                    else:
                        expitH_circuit = time_evolution_for_term(
                            hamiltonian[index_term2], (time / trotter_order)
                        )
                        output += expitH_circuit

                single_trotter_derivatives.append(output)

        if trotter_order > 1:

            output_circuits = []
            final_factors = []

            repeated_circuit = time_evolution(
                hamiltonian, time, method="Trotter", trotter_order=1
            )

            for position in range(0, trotter_order):
                for circuit_factor, different_circuit in zip(
                    output_factors, single_trotter_derivatives
                ):

                    output_circuits.append(
                        generate_circuit_sequence(
                            repeated_circuit, different_circuit, trotter_order, position
                        )
                    )
                    final_factors.append(circuit_factor)

            return output_circuits, final_factors

        else:

            return single_trotter_derivatives, output_factors

    else:

        raise ValueError("Currently the method {} is not supported".format(method))


def generate_circuit_sequence(
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
        circuit = Circuit(pyquil_exponentiate(term))
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
        circuit = Circuit(pyquil_exponentiate(exponent))
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


# @time_evolution_for_term.register
# def time_evolution_for_term_qubit_operator(
#     term: QubitOperator, time: Union[float, sympy.Expr]
# ) -> NewCircuit:
#     """Evolves a Pauli term for a given time and returns a circuit representing it.
#     Args:
#         term: Pauli term to be evolved
#         time: time of evolution
#     Returns:
#         Circuit: Circuit representing evolved pyquil term.
#     """

#     if len(term.terms) != 1:
#         raise ValueError("This function works only on a single term.")
#     term_components = list(term.terms.keys())[0]
#     base_changes = []
#     base_reversals = []
#     cnot_gates = []
#     central_gate = None
#     term_types = [component[1] for component in term_components]
#     qubit_indices = [component[0] for component in term_components]
#     coefficient = term.terms.values()

#     for i, (term_type, qubit_id) in enumerate(zip(term_types, qubit_indices)):
#         # TODO: comments
#         if term_type == "X":
#             base_changes.append(H([qubit_id]))
#             base_reversals.append(H([qubit_id]))
#         elif term_type == "Y":
#             base_changes.append(RX(np.pi / 2)([qubit_id]))
#             base_reversals.append(RX(-np.pi / 2)([qubit_id]))
#         if i == len(term_components) - 1:
#             central_gate = RZ(2 * time * coefficient)([qubit_id])
#         else:
#             cnot_gates.append(CNOT([qubit_id, qubit_indices[i + 1]]))

#     circuit = NewCircuit()
#     for gate in base_changes:
#         circuit += gate

#     for gate in cnot_gates:
#         circuit += gate

#     circuit += central_gate

#     for gate in cnot_gates[::-1]:
#         circuit += gate

#     for gate in base_reversals:
#         circuit += gate

#     return circuit