import json
from .gates import Gate
from ...utils import SCHEMA_VERSION
from typing import List, Dict, Union, TextIO


class Circuit(object):
    """Base class for quantum circuits.
    
    Attributes:
        gates (List[Gate]): The sequence of gates to be applied
        qubits (tuple(int)): The set of qubits that the circuit acts on
        symbolic_params (set(sympy.Symbol)): A set of the parameter names used in the circuit. If the circuit
            does not contain parameterized gates, this value is the empty set.
    """

    def __init__(self, gates: List[Gate] = None):
        """Initialize a quantum circuit 

        Args:
            gates (List[Gate]): See class definition
            qubits (tuple(int)): See class definition
        """
        if gates is None:
            self.gates = []
        else:
            self.gates = gates

    @property
    def qubits(self):
        """ The qubits that are used by the gates in the circuit

        Returns:
            tuple(int)
        """
        qubits = {
            qubit
            for gate in self.gates
            for qubit in gate.qubits
        }
        return tuple(sorted(qubits))

    @property
    def symbolic_params(self):
        """ The set of symbolic parameters used in the circuit

        Returns:
            set: set of all the sympy symbols used as params of gates in the circuit.
        """
        symbolic_params = []
        for gate in self.gates:
            symbolic_params_per_gate = gate.symbolic_params
            symbolic_params += symbolic_params_per_gate

        return set(symbolic_params)

    def __eq__(self, anotherCircuit):
        """Comparison between two Circuit objects.
        """
        if self.qubits != anotherCircuit.qubits:
        # TODO: figure out if this check is redundant when reworking Circuit
            return False

        if len(self.gates) != len(anotherCircuit.gates):
            return False

        for i in range(len(self.gates)):
            if self.gates[i] != anotherCircuit.gates[i]:
                return False

        if len(self.symbolic_params) != len(anotherCircuit.symbolic_params):
        # TODO: figure out if this check is redundant when reworking Circuit
            return False

        return True

    def __add__(self, other_circuit):
        """Add two circuits.
        """
        new_circuit = type(self)()
        new_circuit.gates = self.gates + other_circuit.gates
        return new_circuit

    def evaluate(self, symbols_map: Dict):
        """ Create a copy of the current Circuit with the parameters of each gate evaluated to the values 
        provided in the input symbols map

        Args:
            symbols_map (Dict): A map of the symbols/gate parameters to new values
        """
        circuit_class = type(self)
        evaluated_gate_list = [gate.evaluate(symbols_map) for gate in self.gates]
        evaluated_circuit = circuit_class(gates=evaluated_gate_list)
        return evaluated_circuit

    def to_dict(self, serializable: bool = True):
        """ Creates a dictionary representing a circuit.

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
        """ Save the Circuit object to file in JSON format

        Args:
            filename (str): The path to the file to store the Circuit
        """
        with open(filename, "w") as f:
            f.write(json.dumps(self.to_dict(serializable=True), indent=2))

    @classmethod
    def load(cls, data: Union[Dict, TextIO]):
        """ Load a Circuit object from either a file/file-like object or a dictionary

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
