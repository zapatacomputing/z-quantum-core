################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
from typing import Collection, Optional, Union, cast
from warnings import warn

import numpy as np

from ._builtin_gates import GatePrototype, I
from ._circuit import Circuit
from ._gates import Gate


def create_layer_of_gates(
    number_of_qubits: int,
    gate_factory: Union[GatePrototype, Gate],
    parameters: Optional[np.ndarray] = None,
) -> Circuit:
    """Creates a circuit consisting of a layer of single-qubit gates acting on all
    qubits.

    Args:
        number_of_qubits: number of qubits in the circuit
        gate_factory: the single qubit gate to be applied to each qubit
        parameters: parameters of the single-qubit gates

    Returns:
        circuit: Created circuit.
    """
    circuit = Circuit()

    return apply_gate_to_qubits(
        circuit, range(number_of_qubits), gate_factory, parameters
    )


def apply_gate_to_qubits(
    circuit: Circuit,
    qubit_indices: Collection[int],
    gate_factory: Union[GatePrototype, Gate],
    parameters: Optional[np.ndarray] = None,
) -> Circuit:
    """Apply the passed gate to the specified qubits

    Args:
        circuit: circuit to add the gate to
        qubit_indices: list of qubits that to apply the gate on,
            duplicates will be ignored
        gate_factory: the single qubit gate to be applied to each qubit
        parameters: parameters of the single-qubit gate, this is a list of size
            dependent on the gate used, e.g. 1 for RY gates, 3 for U3 gates
    Returns:
        circuit with the gate added to the specified qubits
    """
    unique_qubit_idx = set(qubit_indices)

    if len(unique_qubit_idx) != len(qubit_indices):
        warn("Duplicate qubits were passed! Duplicates will be ignored.")

    if parameters is not None:
        assert len(parameters) == len(unique_qubit_idx)
        gate_factory = cast(GatePrototype, gate_factory)
        for qubit, parameter in zip(unique_qubit_idx, parameters):
            circuit += gate_factory(*parameter)(qubit)
    else:
        gate_factory = cast(Gate, gate_factory)
        for qubit in unique_qubit_idx:
            circuit += gate_factory(qubit)

    return circuit


def add_ancilla_register(circuit: Circuit, n_ancilla_qubits: int) -> Circuit:
    """Add a register of ancilla qubits (qubit + identity gate) to an existing circuit.

    Args:
        circuit: circuit to be extended
        n_ancilla_qubits: number of ancilla qubits to add
    Returns:
        extended circuit
    """
    extended_circuit = circuit
    for ancilla_qubit_i in range(n_ancilla_qubits):
        qubit_index = circuit.n_qubits + ancilla_qubit_i
        extended_circuit += I(qubit_index)
    return extended_circuit
