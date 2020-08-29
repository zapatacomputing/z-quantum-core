import json
import numpy as np

# from ..utils import convert_array_to_dict, convert_dict_to_array
from ..gate import Gate
from ...utils import SCHEMA_VERSION
from typing import List, Dict


class Circuit(object):
    """Base class for quantum circuits.
    
    Attributes:
        gates (List[Gate]): The sequence of gates to be applied
        qubits (tuple(int)): The set of qubits that the circuit acts on
        symbolic_params (set(sympy.Symbol)): A set of the parameter names used in the circuit. If the circuit
            does not contain parameterized gates, this value is the empty set.
    """

    def __init__(self, gates=None):
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
        qubits = []
        for gate in self.gates:
            for qubit in gate.qubits:
                if qubit not in qubits:
                    qubits.append(qubit)
        return tuple(qubits)

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
            return False

        if len(self.gates) != len(anotherCircuit.gates):
            return False

        for i in range(len(self.gates)):
            if self.gates[i] != anotherCircuit.gates[i]:
                return False

        if len(self.symbolic_params) != len(anotherCircuit.symbolic_params):
            return False

        return True

    def __add__(self, other_circuit):
        """Add two circuits.
        """
        new_circuit = type(self)()
        new_circuit.gates = self.gates + other_circuit.gates
        return new_circuit

    def evaluate(self, symbols_map):
        """ Create a copy of the current Circuit with the parameters of each gate evaluated to the values 
        provided in the input symbols map

        Args:
            symbols_map (Dict): A map of the symbols/gate parameters to new values
        """
        circuit_class = type(self)
        evaluated_gate_list = [gate.evaluate(symbols_map) for gate in self.gates]
        evaluated_circuit = circuit_class(gates=evaluated_gate_list)
        return evaluated_circuit

    def to_dict(self, serialize_gate_params=True):
        """Creates a dictionary representing a circuit.

        Args:
            serialize_gate_params(bool): if true, it will change gate params from sympy to strings (if applicable)

        Returns:
            dictionary (dict): the dictionary
        """

        # if self.gates != None:
        #     gates_entry = [
        #         gate.to_dict(serialize_params=serialize_gate_params)
        #         for gate in self.gates
        #     ]
        # else:
        #     gates_entry = None

        # if self.qubits != None:
        #     qubits_entry = [qubit.to_dict() for qubit in self.qubits]
        # else:
        #     qubits_entry = None

        # dictionary = {
        #     "schema": SCHEMA_VERSION + "-circuit",
        #     "name": self.name,
        #     "gates": gates_entry,
        #     "qubits": qubits_entry,
        #     "info": self.info,
        # }

        # return dictionary
        pass

    def save(self, filename: str):
        """ Save the Gate object to file in JSON format

        Args:
            filename (str): The path to the file to store the Gate
        """
        pass

    @classmethod
    def load(cls, dictionary):
        """Loads information of the circuit from a dictionary. This corresponds to the
        serialization routines to_dict for Circuit, Gate and Qubit.

        Args:
            dictionary (dict): the dictionary

        Returns:
            A core.circuit.Circuit object
        """

        # output = cls(name=dictionary["name"])
        # if dictionary["gates"] != None:
        #     output.gates = [Gate.from_dict(gate) for gate in dictionary["gates"]]
        # else:
        #     output.gates = None

        # if dictionary["qubits"] != None:
        #     output.qubits = [Qubit.from_dict(qubit) for qubit in dictionary["qubits"]]
        # else:
        #     output.qubits = None
        # output.info = dictionary["info"]
        # return output
        pass
