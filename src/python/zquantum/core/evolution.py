import openfermion, pyquil
from pyquil.paulis import exponentiate as pyquil_exponentiate
from .openfermion import qubitop_to_pyquilpauli
import numpy as np
from .circuit import Circuit
from typing import Tuple, List, Union
import sympy


def time_evolution(
    hamiltonian: pyquil.paulis.PauliSum,
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
            for index_term in range(0, len(hamiltonian.terms)):
                output += time_evolution_for_term(
                    hamiltonian[index_term], time / trotter_order
                )

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
        hamiltonian: The Hamiltonian to be evolved under.
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
                # r is the eigenvalue of the generator of the gate. The value is modified
                # to take into account the coefficient and trotter step in front of the
                # Pauli term
                r = hamiltonian[index_term1].coefficient.real / trotter_order
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


def time_evolution_for_term(
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
                        "Time evolution of multi-parametered gates with symbolic parameters is not supported."
                    )
                )
            elif gate.name == "Rz" or gate.name == "PHASE":
                # We only want to modify the parameter of Rz gate or PHASE gate.
                gate.params[0] = gate.params[0] * time
    else:
        circuit = Circuit(pyquil_exponentiate(term * time))
    return circuit
