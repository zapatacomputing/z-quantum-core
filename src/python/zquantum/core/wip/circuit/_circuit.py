import json
from typing import Dict, Union, TextIO, Iterable, Optional
from functools import reduce

from .gates import Gate
from ...utils import SCHEMA_VERSION


def _circuit_size_by_gates(gates):
    return (
        0
        if not gates
        else max(qubit_index for gate in gates for qubit_index in gate.qubits) + 1
    )


class Circuit:
    """Orquestra representation of a quantum circuit."""

    def __init__(self, gates: Iterable[Gate], n_qubits: Optional[int] = None):
        self._gates = gates
        self._n_qubits = (
            n_qubits if n_qubits is not None else _circuit_size_by_gates(gates)
        )

    @property
    def gates(self):
        """Sequence of quantum gates to apply to qubits in this circuit."""
        return self._gates

    @property
    def n_qubits(self):
        """Number of qubits in this circuit.
        Not every qubit has to be used by a gate.
        """
        return self._n_qubits

    @property
    def symbolic_params(self):
        """Set of all the sympy symbols used as params of gates in the circuit."""
        return reduce(set.union, (set(gate.symbolic_params) for gate in self._gates))

    def __eq__(self, other: "Circuit"):
        if not isinstance(other, type(self)):
            return False

        if self.n_qubits != other.n_qubits:
            return False

        if list(self.gates) != list(other.gates):
            return False

        return True

    def __add__(self, other_circuit):
        """Add two circuits."""
        new_circuit = type(self)()
        new_circuit.gates = self.gates + other_circuit.gates
        return new_circuit

    def evaluate(self, symbols_map: Dict):
        """Create a copy of the current Circuit with the parameters of each gate evaluated to the values
        provided in the input symbols map

        Args:
            symbols_map (Dict): A map of the symbols/gate parameters to new values
        """
        circuit_class = type(self)
        evaluated_gate_list = [gate.evaluate(symbols_map) for gate in self.gates]
        evaluated_circuit = circuit_class(gates=evaluated_gate_list)
        return evaluated_circuit

    def to_dict(self, serializable: bool = True):
        """Creates a dictionary representing a circuit.

        Args:
            serializable (bool): If true, the returned dictionary is serializable so that it can be stored
                in JSON format
        Returns:
            Dict: keys are schema, qubits, gates, and symbolic_params
        """
        circuit_dict = {"schema": SCHEMA_VERSION + "-circuit"}
        if serializable:
            circuit_dict["qubits"] = list(self.qubits)
            circuit_dict["gates"] = [
                gate.to_dict(serializable=True) for gate in self.gates
            ]
            circuit_dict["symbolic_params"] = [
                str(param) for param in self.symbolic_params
            ]
        else:
            circuit_dict["qubits"] = self.qubits
            circuit_dict["gates"] = [
                gate.to_dict(serializable=False) for gate in self.gates
            ]
            circuit_dict["symbolic_params"] = self.symbolic_params
        return circuit_dict

    def save(self, filename: str):
        """Save the Circuit object to file in JSON format

        Args:
            filename (str): The path to the file to store the Circuit
        """
        with open(filename, "w") as f:
            f.write(json.dumps(self.to_dict(serializable=True), indent=2))

    @classmethod
    def load(cls, data: Union[Dict, TextIO]):
        """Load a Circuit object from either a file/file-like object or a dictionary

        Args:
            data (Union[Dict, TextIO]): The data to load into the Circuit object

        Returns:
            Circuit
        """
        if isinstance(data, str):
            with open(data, "r") as f:
                data = json.load(f)
        elif not isinstance(data, dict):
            data = json.load(data)

        gates = [Gate.load(gate_data) for gate_data in data["gates"]]
        return cls(gates=gates)

    def __repr__(self):
        return f"{type(self).__name__}(gates={self.gates}, n_qubits={self.n_qubits})"
