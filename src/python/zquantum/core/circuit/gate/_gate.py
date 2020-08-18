import numpy as np
import json
from typing import Tuple, Union, Dict, TextIO
from collections import Counter
from ...utils import SCHEMA_VERSION, convert_array_to_dict, convert_dict_to_array


class Gate(object):
    """Class for storing information associated with a quantum gate.
 
    Attributes:
        matrix (np.ndarray): two-dimensional array defining the matrix representing the quantum operator
        qubits (tuple[int]): A list of qubit indices that the operator acts on 
    """

    def __init__(self, matrix: np.ndarray, qubits: Tuple[int]):
        assert self._is_valid_operator(matrix, qubits)
        assert self._qubits_are_valid(qubits)
        self.matrix = matrix
        self.qubits = qubits

    def _is_valid_operator(self, matrix: np.ndarray, qubits: Tuple[int]):
        # Make sure matrix is square
        is_square = True
        num_rows = len(matrix)
        for row in matrix:
            if len(row) != num_rows:
                is_square = False

        # Make sure matrix is associated with correct number of qubits
        correct_num_qubits = 2 ** len(qubits) == num_rows

        return is_square and correct_num_qubits

    def _qubits_are_valid(self, qubits: Tuple[int]):
        qubits = Counter(list(qubits))
        for qubit_appearences in qubits.values():
            if qubit_appearences > 1:
                return False
        return True

    def __eq__(self, another_gate):
        if len(self.qubits) != len(another_gate.qubits):
            return False
        for qubit, another_qubit in zip(self.qubits, another_gate.qubits):
            if qubit != another_qubit:
                return False

        if len(self.matrix) != len(another_gate.matrix):
            return False
        for row, another_row in zip(self.matrix, another_gate.matrix):
            if any(row != another_row):
                return False

        return True

    def __repr__(self):
        return f"zquantum.core.circuit.gate.Gate(matrix={self.matrix}, qubits={self.qubits})"

    def to_dict(self, serializable: bool = True):
        gate_dict = {"schema": SCHEMA_VERSION + "-gate"}
        if serializable:
            gate_dict["qubits"] = list(self.qubits)
            gate_dict["matrix"] = [convert_array_to_dict(row) for row in self.matrix]
        else:
            gate_dict["qubits"] = self.qubits
            gate_dict["matrix"] = self.matrix

        return gate_dict

    def save(self, filename: str):
        with open(filename, "w") as f:
            f.write(json.dumps(self.to_dict(serializable=True), indent=2))

    @classmethod
    def load(cls, data: Union[Dict, TextIO]):
        if isinstance(data, str):
            with open(data, "r") as f:
                data = json.load(f)
        elif not isinstance(data, dict):
            data = json.load(data)
        
        qubits = tuple(data["qubits"])
        matrix = np.asarray([convert_dict_to_array(row) for row in data["matrix"]])
        return cls(matrix, qubits)