import numpy as np
import sympy
import json
from typing import Tuple, Union, Dict, TextIO
from collections import Counter
from ...utils import SCHEMA_VERSION, convert_array_to_dict, convert_dict_to_array


class Gate(object):
    """Class for storing information associated with a quantum gate.
 
    Attributes:
        matrix (sympy.Matrix): two-dimensional array defining the matrix representing the quantum operator
        qubits (tuple[int]): A list of qubit indices that the operator acts on 
    """

    def __init__(self, matrix: sympy.Matrix, qubits: Tuple[int]):
        self._assert_is_valid_operator(matrix, qubits)
        self._assert_qubits_are_unique(qubits)
        self.matrix = matrix
        self.qubits = qubits
        self.symbolic_params = self._get_symbolic_params()

    def _assert_is_valid_operator(self, matrix: sympy.Matrix, qubits: Tuple[int]):
        # Make sure matrix is square
        is_square = True
        shape = matrix.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]

        # Make sure matrix is associated with correct number of qubits
        assert 2 ** len(qubits) == shape[0]

    def _assert_qubits_are_unique(self, qubits: Tuple[int]):
        assert len(set(qubits)) == len(qubits)

    def _get_symbolic_params(self):
        """
        Returns a list of symbolic parameters used in the gate

        Returns:
            set: set containing all the sympy symbols used in gate params
        """
        all_symbols = []
        for element in self.matrix:
            if isinstance(element, sympy.Expr):
                for symbol in element.free_symbols:
                    all_symbols.append(symbol)
        return set(all_symbols)

    def __eq__(self, another_gate):
        if len(self.qubits) != len(another_gate.qubits):
            return False
        for qubit, another_qubit in zip(self.qubits, another_gate.qubits):
            if qubit != another_qubit:
                return False

        if len(self.matrix) != len(another_gate.matrix):
            return False
        for element, another_element in zip(self.matrix, another_gate.matrix):
            if element != another_element:
                return False

        if self.symbolic_params != another_gate.symbolic_params:
            return False

        return True

    def __repr__(self):
        return f"zquantum.core.circuit.gate.Gate(matrix={self.matrix}, qubits={self.qubits})"

    def to_dict(self, serializable: bool = True):
        gate_dict = {"schema": SCHEMA_VERSION + "-gate"}
        if serializable:
            gate_dict["qubits"] = list(self.qubits)
            gate_dict["matrix"] = []
            for i in range(self.matrix.shape[0]):
                gate_dict["matrix"].append({"real": [], "imag": []})
                for element in self.matrix.row(i):
                    gate_dict["matrix"][-1]["real"].append(str(sympy.re(element)))
                    gate_dict["matrix"][-1]["imag"].append(str(sympy.im(element)))
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

        if not isinstance(data["matrix"], sympy.Matrix):
            matrix = []
            for row_index, row in enumerate(data["matrix"]):
                new_row = []
                for element_index in range(len(row["real"])):
                    new_row.append(
                        complex(
                            float(row["real"][element_index]),
                            float(row["imag"][element_index]),
                        )
                    )
                matrix.append(new_row)
            matrix = sympy.Matrix(matrix)
        else:
            matrix = data["matrix"]

        return cls(matrix, qubits)
